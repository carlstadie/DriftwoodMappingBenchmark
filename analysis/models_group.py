# models_group.py
from __future__ import annotations

from typing import Sequence, Tuple

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import matplotlib.pyplot as plt

from pathlib import Path

from bayes_utils import pick_hdi_columns, summarize_draws
from config import (
    CHAINS,
    DRAWS,
    HDI_PROB,
    RANDOM_SEED,
    TARGET_ACCEPT,
    TUNE,
)
from reporting import add_table_with_column_guide, safe_slug, HtmlReport
from colguides import COLGUIDE_CONSECUTIVE_ROPE
from transforms import higher_is_better, transform_y
from plots import save_fig


# =============================================================================
# RANKING HELPERS
# =============================================================================
def rank_probabilities_from_draws(
    draws: np.ndarray,
    group_names: Sequence[str],
    higher_better: bool,
) -> pd.DataFrame:
    """Compute Pr(rank=k), E[rank], and Pr(best) from posterior draws.

    Args:
        draws: (samples, groups) draws of group means on ORIGINAL scale
        group_names: names matching columns
        higher_better: direction

    Returns:
        DataFrame sorted by E[rank] (1=best).
    """
    s_count, g_count = draws.shape
    score = draws if higher_better else -draws

    order = np.argsort(-score, axis=1)
    ranks = np.empty_like(order)
    for s in range(s_count):
        ranks[s, order[s]] = np.arange(1, g_count + 1)

    out = {"group": list(group_names)}
    for k in range(1, g_count + 1):
        out[f"Pr(rank={k})"] = [(ranks[:, j] == k).mean() for j in range(g_count)]
    out["E[rank]"] = [ranks[:, j].mean() for j in range(g_count)]
    out["Pr(best)"] = out["Pr(rank=1)"]

    return pd.DataFrame(out).sort_values("E[rank]").reset_index(drop=True)


# =============================================================================
# GROUP MODEL WITH HIERARCHICAL VARIANCE
# =============================================================================
def fit_group_model_with_ranking(
    df: pd.DataFrame,
    group_col: str,
    metric_col: str,
    draws: int = DRAWS,
    tune: int = TUNE,
    chains: int = CHAINS,
    target_accept: float = TARGET_ACCEPT,
    seed: int = RANDOM_SEED,
) -> Tuple[
    az.InferenceData,
    pd.DataFrame,
    pd.DataFrame,
    np.ndarray,
    str,
    Sequence[str],
    object,
    np.ndarray,
]:
    """Hierarchical group model WITH hierarchical variance structure.

    Returns:
        idata,
        perf_table,
        post_rank_table,
        post_mu_draws,          # (samples, groups) on original scale
        tname,                  # 'logit' or 'log'
        groups,                 # group names (coords)
        post_pred,              # posterior predictive object
        prior_mu_draws,         # (samples, groups) on original scale (for PPC / intuition)
    """
    data = df[[group_col, metric_col]].dropna().copy()

    if isinstance(data[group_col].dtype, pd.CategoricalDtype):
        data[group_col] = data[group_col].cat.remove_unused_categories()

    gcat = pd.Categorical(data[group_col])
    g_idx = gcat.codes
    groups = list(gcat.categories)

    y_raw = data[metric_col].to_numpy()
    y_t, inv, tname = transform_y(y_raw, metric_col)

    coords = {"group": groups}

    # Data-informed prior center for intercept
    mu0_loc = float(np.median(y_t))

    with pm.Model(coords=coords) as model:
        g = pm.Data("g", g_idx)
        y_obs = pm.Data("y_obs", y_t)

        # Hierarchical mean structure
        mu0 = pm.Normal("mu0", mu0_loc, 1.0)
        tau = pm.HalfNormal("tau", 1.0)
        mu = pm.Normal("mu", mu0, tau, dims="group")

        # Hierarchical variance structure: each group has its own sigma
        sigma0 = pm.HalfNormal("sigma0", 1.0)
        sigma = pm.HalfNormal("sigma", sigma0, dims="group")

        pm.Normal("y", mu=mu[g], sigma=sigma[g], observed=y_obs)

        # Deterministic: group means on original scale
        mu_orig = pm.Deterministic("mu_orig", inv(mu), dims="group")

        idata = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            target_accept=target_accept,
            random_seed=seed,
            return_inferencedata=True,
        )

        # Prior predictive for PPC / prior ranks
        prior = pm.sample_prior_predictive(var_names=["mu_orig"], random_seed=seed)

        # Posterior predictive check
        post_pred = pm.sample_posterior_predictive(idata, random_seed=seed)

    # Posterior draws of mu_orig
    post_mu_draws = (
        idata.posterior["mu_orig"].stack(sample=("chain", "draw")).transpose("sample", "group").values
    )

    hib = higher_is_better(metric_col)
    post_rank_table = rank_probabilities_from_draws(post_mu_draws, groups, higher_better=hib)

    prior_mu_draws = (
        prior.prior["mu_orig"].stack(sample=("chain", "draw")).transpose("sample", "group").values
    )
    prior_rank_table = rank_probabilities_from_draws(prior_mu_draws, groups, higher_better=hib)

    # Summary table for mu_orig
    summ = az.summary(idata, var_names=["mu_orig"], hdi_prob=HDI_PROB).reset_index()
    summ = summ.rename(columns={"index": "param"})
    hdi_lo_col, hdi_hi_col = pick_hdi_columns(summ)
    summ["group"] = summ["param"].str.replace(r"mu_orig\[|\]", "", regex=True)

    perf_table = (
        summ.loc[:, ["group", "mean", "sd", hdi_lo_col, hdi_hi_col]]
        .merge(post_rank_table, on="group", how="left")
        .merge(
            prior_rank_table.loc[:, ["group", "Pr(best)"]].rename(columns={"Pr(best)": "Pr(best)_prior"}),
            on="group",
            how="left",
        )
        .sort_values("E[rank]")
        .reset_index(drop=True)
    )

    # Extract sigma summary and merge into perf_table
    sigma_summ = az.summary(idata, var_names=["sigma"], hdi_prob=HDI_PROB).reset_index()
    sigma_summ = sigma_summ.rename(columns={"index": "param"})
    hdi_lo_col_s, hdi_hi_col_s = pick_hdi_columns(sigma_summ)
    sigma_summ["group"] = sigma_summ["param"].str.replace(r"sigma\[|\]", "", regex=True)

    sigma_view = sigma_summ[["group", "mean", "sd", hdi_lo_col_s, hdi_hi_col_s]].copy()
    sigma_view = sigma_view.rename(
        columns={
            "mean": "sigma_mean",
            "sd": "sigma_sd",
            hdi_lo_col_s: "sigma_hdi_lo",
            hdi_hi_col_s: "sigma_hdi_hi",
        }
    )
    perf_table = perf_table.merge(sigma_view, on="group", how="left")

    return (
        idata,
        perf_table,
        post_rank_table,
        post_mu_draws,
        tname,
        groups,
        post_pred,
        prior_mu_draws,
    )


# =============================================================================
# CONSECUTIVE RANKING: ROPE-BASED PRACTICAL DIFFERENCE
# =============================================================================
def compute_consecutive_rope_comparisons(
    idata: az.InferenceData,
    mu_draws_original: np.ndarray,
    rank_table: pd.DataFrame,
    group_names: Sequence[str],
    higher_better: bool,
    rope_transformed: float,
    hdi_prob: float = HDI_PROB,
) -> pd.DataFrame:
    """Compare each expected-rank-adjacent pair using ROPE on the transformed scale.

    We compute the difference on the model's transformed scale (mu), because that is
    where the ROPE is defined/consistent. For interpretability we also summarize the
    difference on the original scale (mu_orig).

    The delta is oriented so that positive means "this row's group is better".
    """
    # Extract posterior draws of group means on transformed scale
    mu_t_draws = (
        idata.posterior["mu"].stack(sample=("chain", "draw")).transpose("sample", "group").values
    )

    sorted_ranks = rank_table.sort_values("E[rank]").reset_index(drop=True)

    rows = []
    for i in range(len(sorted_ranks) - 1):
        group_i = sorted_ranks.loc[i, "group"]
        group_j = sorted_ranks.loc[i + 1, "group"]
        rank_i = i + 1

        idx_i = list(group_names).index(group_i)
        idx_j = list(group_names).index(group_j)

        if higher_better:
            delta_t = mu_t_draws[:, idx_i] - mu_t_draws[:, idx_j]
            delta_o = mu_draws_original[:, idx_i] - mu_draws_original[:, idx_j]
        else:
            # for lower-is-better metrics, reverse so positive means i is better
            delta_t = mu_t_draws[:, idx_j] - mu_t_draws[:, idx_i]
            delta_o = mu_draws_original[:, idx_j] - mu_draws_original[:, idx_i]

        s_o = summarize_draws(delta_o, hdi_prob=hdi_prob)
        dt = np.asarray(delta_t).ravel()

        rows.append(
            {
                "rank": rank_i,
                "group": group_i,
                "next_rank_group": group_j,
                "Δ_mean": s_o["mean"],
                "Δ_lo": s_o["hdi_lo"],
                "Δ_hi": s_o["hdi_hi"],
                "Pr(better)": float((dt > 0).mean()),
                "Pr(Δ>ROPE)": float((dt > rope_transformed).mean()),
                "Pr(|Δ|<ROPE)": float((np.abs(dt) < rope_transformed).mean()),
                "Pr(Δ<-ROPE)": float((dt < -rope_transformed).mean()),
            }
        )

    return pd.DataFrame(rows)


def add_consecutive_rope_analysis(
    report: HtmlReport,
    idata: az.InferenceData,
    mu_draws_original: np.ndarray,
    perf_table: pd.DataFrame,
    group_names: Sequence[str],
    metric: str,
    higher_better: bool,
    rope_transformed: float,
    assets_dir: Path,
) -> None:
    """Add consecutive ranking practical-difference analysis + plot to HTML report."""

    comp_df = compute_consecutive_rope_comparisons(
        idata=idata,
        mu_draws_original=mu_draws_original,
        rank_table=perf_table,
        group_names=group_names,
        higher_better=higher_better,
        rope_transformed=rope_transformed,
        hdi_prob=HDI_PROB,
    )

    report.add_h3("Consecutive ranking comparisons")
    report.add_paragraph(
        "For each adjacent pair in the expected-rank order, we report the probability that the higher-ranked group is "
        "(a) better, (b) meaningfully better (exceeds ROPE), or (c) practically tied (inside ROPE). "
        "ROPE is applied on the model's transformed scale for consistency."
    )
    add_table_with_column_guide(report, comp_df, COLGUIDE_CONSECUTIVE_ROPE)

    # Visualize Pr(Δ>ROPE) (meaningful separation)
    fig, ax = plt.subplots(figsize=(10, max(4, 0.45 * len(comp_df))))
    y = np.arange(len(comp_df))
    x = comp_df["Pr(Δ>ROPE)"].to_numpy()

    colors = ["green" if p > 0.95 else "orange" if p > 0.80 else "red" for p in x]

    ax.barh(y, x, color=colors, alpha=0.6)
    ax.axvline(0.5, linestyle="--", color="gray", linewidth=1, alpha=0.5)
    ax.axvline(0.95, linestyle="--", color="green", linewidth=1, alpha=0.3, label="95% threshold")

    labels = [f"{r.group} vs {r.next_rank_group}" for r in comp_df.itertuples()]
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("Pr(meaningfully better than next-ranked)")
    ax.set_title(f"Consecutive ranking comparisons (ROPE): {metric}")
    ax.legend()
    ax.grid(True, axis="x", alpha=0.2)

    plot_name = f"consecutive_rope_{safe_slug(metric)}.png"
    save_fig(fig, assets_dir / plot_name)

    report.add_image(f"{assets_dir.name}/{plot_name}", alt=f"Consecutive ROPE plot for {metric}")
    report.add_footnote(
        "Interpreting consecutive comparisons",
        """
<ul>
  <li><b>Pr(Δ&gt;ROPE)</b> close to 1.0 → strong confidence the ranking separation is practically meaningful</li>
  <li><b>Pr(|Δ|&lt;ROPE)</b> close to 1.0 → strong evidence the two groups are practically tied</li>
  <li><b>Pr(better)</b> can be high even when the effect is tiny; use ROPE columns for a significance-like decision</li>
  <li><b>Δ_mean / Δ HDI</b> are on the original metric scale and help judge magnitude</li>
</ul>
""",
    )


def compute_pairwise_rope_vs_best(
    idata: az.InferenceData,
    mu_draws_original: np.ndarray,
    perf_table: pd.DataFrame,
    group_names: Sequence[str],
    higher_better: bool,
    rope_transformed: float,
    hdi_prob: float = HDI_PROB,
) -> pd.DataFrame:
    """Compute ROPE probabilities for each group vs the best (highest ranking) group.
    
    Args:
        idata: InferenceData with posterior["mu"] on transformed scale
        mu_draws_original: (samples, groups) draws on original scale
        perf_table: ranking table sorted by E[rank]
        group_names: names of groups
        higher_better: direction of metric
        rope_transformed: ROPE threshold on transformed scale
        hdi_prob: HDI probability level
        
    Returns:
        DataFrame with ROPE columns for each group
    """
    # Extract posterior draws of group means on transformed scale
    mu_t_draws = (
        idata.posterior["mu"].stack(sample=("chain", "draw")).transpose("sample", "group").values
    )
    
    # Best group is the first in perf_table (E[rank]=1)
    best_group = perf_table.iloc[0]["group"]
    best_idx = list(group_names).index(best_group)
    
    rows = []
    for i, row in perf_table.iterrows():
        group = row["group"]
        group_idx = list(group_names).index(group)
        
        if group == best_group:
            # vs itself
            rows.append({
                "group": group,
                "vs_group": "N/A",
                "Δ_mean": 0.0,
                "Δ_lo": 0.0,
                "Δ_hi": 0.0,
                "Pr(better)": 0.5,
                "Pr(Δ>ROPE)": np.nan,
                "Pr(|Δ|<ROPE)": np.nan,
                "Pr(Δ<-ROPE)": np.nan,
            })
        else:
            if higher_better:
                delta_t = mu_t_draws[:, group_idx] - mu_t_draws[:, best_idx]
                delta_o = mu_draws_original[:, group_idx] - mu_draws_original[:, best_idx]
            else:
                delta_t = mu_t_draws[:, best_idx] - mu_t_draws[:, group_idx]
                delta_o = mu_draws_original[:, best_idx] - mu_draws_original[:, group_idx]
            
            s_o = summarize_draws(delta_o, hdi_prob=hdi_prob)
            dt = np.asarray(delta_t).ravel()
            
            rows.append({
                "group": group,
                "vs_group": best_group,
                "Δ_mean": s_o["mean"],
                "Δ_lo": s_o["hdi_lo"],
                "Δ_hi": s_o["hdi_hi"],
                "Pr(better)": float((dt > 0).mean()),
                "Pr(Δ>ROPE)": float((dt > rope_transformed).mean()),
                "Pr(|Δ|<ROPE)": float((np.abs(dt) < rope_transformed).mean()),
                "Pr(Δ<-ROPE)": float((dt < -rope_transformed).mean()),
            })
    
    return pd.DataFrame(rows)

