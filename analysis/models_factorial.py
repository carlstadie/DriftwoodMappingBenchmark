# models_factorial.py
from __future__ import annotations

from typing import Dict, List, Tuple

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm

from bayes_utils import extract_draws_1d, pick_hdi_columns, summarize_draws
from config import (
    ARCH_ORDER,
    CHAINS,
    DATASET_ORDER,
    DRAWS,
    HDI_PROB,
    PRIOR_SD_LOG,
    PRIOR_SD_LOGIT,
    RANDOM_SEED,
    TARGET_ACCEPT,
    TUNE,
)
from transforms import transform_y


# =============================================================================
# IMPLICIT EFFECTS COMPLETION (SUM-TO-ZERO / EFFECTS CODING)
# =============================================================================
def add_implicit_effects_to_report(
    report: pd.DataFrame,
    idata: az.InferenceData,
    datasets: List[str],
    archs: List[str],
    tname: str,
    rope_transformed: float,
    hdi_prob: float,
    hdi_lo_col: str,
    hdi_hi_col: str,
) -> pd.DataFrame:
    """Add implicit (effects-coded) levels so tables/plots show ALL levels.

    With effects coding:
      - beta_dataset has K-1 parameters, LAST is -sum(others)
      - beta_arch has J-1 parameters, LAST is -sum(others)
      - beta_interaction has (K-1)(J-1) parameters; remaining cells implied by row/col sum-to-zero.

    We compute posterior draws for these implicit levels and add them to the report.
    """

    def _summarize_1d(draws_1d: np.ndarray) -> Dict[str, float]:
        x = np.asarray(draws_1d).ravel()
        hdi = az.hdi(x, hdi_prob=hdi_prob)
        return {
            "mean": float(x.mean()),
            "sd": float(x.std(ddof=1)),
            "hdi_lo": float(hdi[0]),
            "hdi_hi": float(hdi[1]),
        }

    def _rope_probs(draws_1d: np.ndarray) -> Dict[str, float]:
        x = np.asarray(draws_1d).ravel()
        return {
            "Pr(>0)": float((x > 0).mean()),
            "Pr(Δ>ROPE)": float((x > rope_transformed).mean()),
            "Pr(Δ<-ROPE)": float((x < -rope_transformed).mean()),
            "Pr(|Δ|<ROPE)": float((np.abs(x) < rope_transformed).mean()),
        }

    def _make_row(param: str, post_draws: np.ndarray) -> Dict[str, object]:
        s = _summarize_1d(post_draws)
        rp = _rope_probs(post_draws)
        return {
            "param": param,
            "mean_transformed": s["mean"],
            "sd_transformed": s["sd"],
            f"{hdi_lo_col}_transformed": s["hdi_lo"],
            f"{hdi_hi_col}_transformed": s["hdi_hi"],
            **rp,
            "scale": tname,
            "prior": "Implied (sum-to-zero) from effects coding",
            # Filled later from predicted-means contrasts:
            "OR": np.nan,
            "Δp_mean": np.nan,
            "Δp_lo": np.nan,
            "Δp_hi": np.nan,
            "Abs_Δ_mean": np.nan,
            "Abs_Δ_lo": np.nan,
            "Abs_Δ_hi": np.nan,
            "%_change": np.nan,
            "baseline": np.nan,
        }

    existing = set(report["param"].astype(str).tolist())
    derived_rows: List[Dict[str, object]] = []

    # ---- DATASET: add LAST level
    if len(datasets) >= 2:
        d_last = datasets[-1]
        post_d = (
            idata.posterior["beta_dataset"]
            .stack(sample=("chain", "draw"))
            .transpose("sample", "dataset_effect")
            .to_numpy()
        )
        post_last = -post_d.sum(axis=1)

        p = f"beta_dataset[{d_last}]"
        if p not in existing:
            derived_rows.append(_make_row(p, post_last))
            existing.add(p)

    # ---- ARCH: add LAST level
    if len(archs) >= 2:
        a_last = archs[-1]
        post_a = (
            idata.posterior["beta_arch"]
            .stack(sample=("chain", "draw"))
            .transpose("sample", "arch_effect")
            .to_numpy()
        )
        post_last = -post_a.sum(axis=1)

        p = f"beta_arch[{a_last}]"
        if p not in existing:
            derived_rows.append(_make_row(p, post_last))
            existing.add(p)

    # ---- INTERACTIONS: add ALL cells (including implicit)
    if len(datasets) >= 2 and len(archs) >= 2:
        K = len(datasets)
        J = len(archs)
        K1 = K - 1
        J1 = J - 1

        post_i = (
            idata.posterior["beta_interaction"]
            .stack(sample=("chain", "draw"))
            .transpose("sample", "interaction_effect")
            .to_numpy()
        )
        post_grid = post_i.reshape(post_i.shape[0], K1, J1)

        post_full = np.zeros((post_i.shape[0], K, J))
        post_full[:, :K1, :J1] = post_grid

        # last dataset row: -sum over datasets
        post_full[:, K1, :J1] = -post_full[:, :K1, :J1].sum(axis=1)

        # last arch col: -sum over archs
        post_full[:, :K1, J1] = -post_full[:, :K1, :J1].sum(axis=2)

        # last×last: implied (consistent)
        post_full[:, K1, J1] = post_full[:, :K1, :J1].sum(axis=(1, 2))

        for di, d in enumerate(datasets):
            for ai, a in enumerate(archs):
                p = f"beta_interaction[{d}×{a}]"
                if p in existing:
                    continue
                derived_rows.append(_make_row(p, post_full[:, di, ai]))
                existing.add(p)

    if derived_rows:
        report = pd.concat([report, pd.DataFrame(derived_rows)], ignore_index=True)

    return report


# =============================================================================
# FACTORIAL MODEL (EFFECTS CODING) + ROPE-BASED PRACTICAL SIGNIFICANCE
# =============================================================================
def fit_factorial_model(
    df: pd.DataFrame,
    dataset_col: str,
    arch_col: str,
    metric_col: str,
    rope_transformed: float,
    draws: int = DRAWS,
    tune: int = TUNE,
    chains: int = CHAINS,
    target_accept: float = TARGET_ACCEPT,
    seed: int = RANDOM_SEED,
    hdi_prob: float = HDI_PROB,
) -> Tuple[
    az.InferenceData,
    pd.DataFrame,
    pd.DataFrame,
    np.ndarray,
    np.ndarray,
    Tuple[List[str], List[str]],
    object,
]:
    """Fit a dataset×architecture factorial model with effects coding.

    Key idea: instead of Bayes factors for "non-zero", we report ROPE-based probabilities
    (on the transformed scale where the model lives):
      - Pr(Δ>ROPE): probability the effect is meaningfully positive
      - Pr(|Δ|<ROPE): probability the effect is practically negligible
      - Pr(Δ<-ROPE): probability the effect is meaningfully negative

    Interpretable effect sizes (OR/Δp for logit, or absolute/% change for log) are computed
    via contrasts on predicted means (correct under effects coding).
    """

    data = df[[dataset_col, arch_col, metric_col]].dropna().copy()

    data[dataset_col] = pd.Categorical(data[dataset_col], categories=DATASET_ORDER, ordered=True)
    data[arch_col] = pd.Categorical(data[arch_col], categories=ARCH_ORDER, ordered=True)
    data = data.dropna(subset=[dataset_col, arch_col, metric_col]).copy()
    data[dataset_col] = data[dataset_col].cat.remove_unused_categories()
    data[arch_col] = data[arch_col].cat.remove_unused_categories()

    dcat = pd.Categorical(data[dataset_col])
    acat = pd.Categorical(data[arch_col])

    len_datasets = len(dcat.categories)
    len_archs = len(acat.categories)

    print(f"Fitting a {len_datasets}×{len_archs} factorial model for '{metric_col}'")
    print(f"Datasets: {list(dcat.categories)}")
    print(f"Architectures: {list(acat.categories)}")

    d_idx = dcat.codes
    a_idx = acat.codes
    n_obs = len(d_idx)

    # Effects coding matrices
    d_mat = np.zeros((n_obs, len_datasets - 1))
    a_mat = np.zeros((n_obs, len_archs - 1))

    for i in range(n_obs):
        if d_idx[i] < len_datasets - 1:
            d_mat[i, d_idx[i]] = 1
        else:
            d_mat[i, :] = -1

        if a_idx[i] < len_archs - 1:
            a_mat[i, a_idx[i]] = 1
        else:
            a_mat[i, :] = -1

    da_mat = np.zeros((n_obs, (len_datasets - 1) * (len_archs - 1)))
    col_idx = 0
    for d in range(len_datasets - 1):
        for a in range(len_archs - 1):
            da_mat[:, col_idx] = d_mat[:, d] * a_mat[:, a]
            col_idx += 1

    y_raw = data[metric_col].to_numpy()
    y_t, inv, tname = transform_y(y_raw, metric_col)
    effect_sd = PRIOR_SD_LOGIT if tname == "logit" else PRIOR_SD_LOG

    # Data-informed prior center for intercept
    intercept_loc = float(np.median(y_t))

    coords = {
        "dataset_effect": list(dcat.categories)[:-1],
        "arch_effect": list(acat.categories)[:-1],
        "interaction_effect": [
            f"{dcat.categories[d]}×{acat.categories[a]}"
            for d in range(len_datasets - 1)
            for a in range(len_archs - 1)
        ],
    }

    with pm.Model(coords=coords) as model:
        d = pm.Data("D", d_mat)
        a = pm.Data("A", a_mat)
        da = pm.Data("DA", da_mat)

        intercept = pm.Normal("intercept", intercept_loc, 1.0)
        beta_dataset = pm.Normal("beta_dataset", 0.0, effect_sd, dims="dataset_effect")
        beta_arch = pm.Normal("beta_arch", 0.0, effect_sd, dims="arch_effect")
        beta_interaction = pm.Normal("beta_interaction", 0.0, effect_sd, dims="interaction_effect")

        mu = intercept + pm.math.dot(d, beta_dataset) + pm.math.dot(a, beta_arch) + pm.math.dot(da, beta_interaction)

        sigma = pm.HalfNormal("sigma", sigma=1.0)
        pm.Normal("y", mu=mu, sigma=sigma, observed=y_t)

        idata = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            target_accept=target_accept,
            random_seed=seed,
            return_inferencedata=True,
        )

        post_pred = pm.sample_posterior_predictive(idata, random_seed=seed)

    summary = (
        az.summary(idata, var_names=["beta_dataset", "beta_arch", "beta_interaction"], hdi_prob=hdi_prob)
        .reset_index()
        .rename(columns={"index": "param"})
    )
    hdi_lo_col, hdi_hi_col = pick_hdi_columns(summary)

    # --- coefficient report rows (transformed scale) ---
    rows: List[Dict[str, object]] = []

    def _rope_probs(draws_1d: np.ndarray) -> Dict[str, float]:
        x = np.asarray(draws_1d).ravel()
        return {
            "Pr(>0)": float((x > 0).mean()),
            "Pr(Δ>ROPE)": float((x > rope_transformed).mean()),
            "Pr(Δ<-ROPE)": float((x < -rope_transformed).mean()),
            "Pr(|Δ|<ROPE)": float((np.abs(x) < rope_transformed).mean()),
        }

    def _add_param_rows(var_name: str, dim_name: str) -> None:
        coord_vals = list(idata.posterior[var_name].coords[dim_name].values)
        for dim_val in coord_vals:
            sel = {dim_name: dim_val}
            post_draws = extract_draws_1d(idata, "posterior", var_name, sel=sel)
            rp = _rope_probs(post_draws)

            param_str = f"{var_name}[{dim_val}]"
            base_row = summary.loc[summary["param"] == param_str].iloc[0].to_dict()

            rows.append(
                {
                    "param": param_str,
                    "mean_transformed": base_row["mean"],
                    "sd_transformed": base_row["sd"],
                    f"{hdi_lo_col}_transformed": base_row[hdi_lo_col],
                    f"{hdi_hi_col}_transformed": base_row[hdi_hi_col],
                    **rp,
                    "scale": tname,
                    "prior": f"Normal(0, {effect_sd}) on {tname} scale",
                    # Filled later from predicted-means contrasts:
                    "OR": np.nan,
                    "Δp_mean": np.nan,
                    "Δp_lo": np.nan,
                    "Δp_hi": np.nan,
                    "Abs_Δ_mean": np.nan,
                    "Abs_Δ_lo": np.nan,
                    "Abs_Δ_hi": np.nan,
                    "%_change": np.nan,
                    "baseline": np.nan,
                }
            )

    _add_param_rows("beta_dataset", "dataset_effect")
    _add_param_rows("beta_arch", "arch_effect")
    _add_param_rows("beta_interaction", "interaction_effect")
    report = pd.DataFrame(rows)

    # Predicted means per dataset×arch
    datasets = list(dcat.categories)
    archs = list(acat.categories)

    report = add_implicit_effects_to_report(
        report=report,
        idata=idata,
        datasets=datasets,
        archs=archs,
        tname=tname,
        rope_transformed=rope_transformed,
        hdi_prob=hdi_prob,
        hdi_lo_col=hdi_lo_col,
        hdi_hi_col=hdi_hi_col,
    )

    # Build effect-code rows
    d_rows = []
    for d_i in range(len_datasets):
        if d_i < len_datasets - 1:
            row = np.zeros(len_datasets - 1)
            row[d_i] = 1.0
        else:
            row = -np.ones(len_datasets - 1)
        d_rows.append(row)

    a_rows = []
    for a_i in range(len_archs):
        if a_i < len_archs - 1:
            row = np.zeros(len_archs - 1)
            row[a_i] = 1.0
        else:
            row = -np.ones(len_archs - 1)
        a_rows.append(row)

    # Stack posterior samples robustly
    intercept_s = idata.posterior["intercept"].stack(sample=("chain", "draw")).to_numpy().ravel()
    beta_d_s = (
        idata.posterior["beta_dataset"].stack(sample=("chain", "draw")).transpose("sample", "dataset_effect").to_numpy()
    )
    beta_a_s = (
        idata.posterior["beta_arch"].stack(sample=("chain", "draw")).transpose("sample", "arch_effect").to_numpy()
    )
    beta_i_s = (
        idata.posterior["beta_interaction"]
        .stack(sample=("chain", "draw"))
        .transpose("sample", "interaction_effect")
        .to_numpy()
    )

    pred_draws_transformed = np.zeros((intercept_s.shape[0], len(datasets), len(archs)))
    pred_draws_original = np.zeros((intercept_s.shape[0], len(datasets), len(archs)))
    pred_rows: List[Dict[str, object]] = []

    for di, d_name in enumerate(datasets):
        for ai, a_name in enumerate(archs):
            d_row = d_rows[di]
            a_row = a_rows[ai]

            da_row = np.zeros((len_datasets - 1) * (len_archs - 1))
            c = 0
            for d_j in range(len_datasets - 1):
                for a_j in range(len_archs - 1):
                    da_row[c] = d_row[d_j] * a_row[a_j]
                    c += 1

            mu_s = intercept_s + (beta_d_s @ d_row) + (beta_a_s @ a_row) + (beta_i_s @ da_row)
            mu_orig = inv(mu_s)

            mu_s = np.asarray(mu_s).ravel()
            mu_orig = np.asarray(mu_orig).ravel()

            pred_draws_transformed[:, di, ai] = mu_s
            pred_draws_original[:, di, ai] = mu_orig

            mu_stats_orig = summarize_draws(mu_orig, hdi_prob=hdi_prob)
            pred_rows.append(
                {
                    "dataset": d_name,
                    "arch": a_name,
                    "mean_original": mu_stats_orig["mean"],
                    "hdi_lo_original": mu_stats_orig["hdi_lo"],
                    "hdi_hi_original": mu_stats_orig["hdi_hi"],
                }
            )

    pred_table = pd.DataFrame(pred_rows)

    # Fill interpretable effect sizes via contrasts on predicted means
    dataset_marg_t = pred_draws_transformed.mean(axis=2)
    dataset_marg_o = pred_draws_original.mean(axis=2)
    arch_marg_t = pred_draws_transformed.mean(axis=1)
    arch_marg_o = pred_draws_original.mean(axis=1)

    grand_mean_dataset_t = dataset_marg_t.mean(axis=1)
    grand_mean_dataset_o = dataset_marg_o.mean(axis=1)
    grand_mean_arch_t = arch_marg_t.mean(axis=1)
    grand_mean_arch_o = arch_marg_o.mean(axis=1)
    grand_mean_cell_t = pred_draws_transformed.mean(axis=(1, 2))
    grand_mean_cell_o = pred_draws_original.mean(axis=(1, 2))

    def _write_effect(param: str, delta_t: np.ndarray, delta_o: np.ndarray, baseline_o: np.ndarray) -> None:
        # Practical-significance probabilities (on transformed scale)
        dt = np.asarray(delta_t).ravel()
        report.loc[report["param"] == param, "Pr(>0)"] = float((dt > 0).mean())
        report.loc[report["param"] == param, "Pr(Δ>ROPE)"] = float((dt > rope_transformed).mean())
        report.loc[report["param"] == param, "Pr(Δ<-ROPE)"] = float((dt < -rope_transformed).mean())
        report.loc[report["param"] == param, "Pr(|Δ|<ROPE)"] = float((np.abs(dt) < rope_transformed).mean())

        # Natural-scale effect sizes
        s_o = summarize_draws(delta_o, hdi_prob=hdi_prob)
        s_base = summarize_draws(baseline_o, hdi_prob=hdi_prob)

        if tname == "log":
            ratio = np.exp(delta_t)
            pct = (ratio - 1.0) * 100.0
            s_pct = summarize_draws(pct, hdi_prob=hdi_prob)

            report.loc[report["param"] == param, "Abs_Δ_mean"] = s_o["mean"]
            report.loc[report["param"] == param, "Abs_Δ_lo"] = s_o["hdi_lo"]
            report.loc[report["param"] == param, "Abs_Δ_hi"] = s_o["hdi_hi"]
            report.loc[report["param"] == param, "%_change"] = s_pct["mean"]
            report.loc[report["param"] == param, "baseline"] = s_base["mean"]
        else:
            OR = np.exp(delta_t)
            s_or = summarize_draws(OR, hdi_prob=hdi_prob)

            report.loc[report["param"] == param, "OR"] = s_or["mean"]
            report.loc[report["param"] == param, "Δp_mean"] = s_o["mean"]
            report.loc[report["param"] == param, "Δp_lo"] = s_o["hdi_lo"]
            report.loc[report["param"] == param, "Δp_hi"] = s_o["hdi_hi"]
            report.loc[report["param"] == param, "baseline"] = s_base["mean"]

    # Dataset effects: ALL datasets vs grand mean (marginal over arch)
    for d_name in datasets:
        di = datasets.index(d_name)
        delta_t = dataset_marg_t[:, di] - grand_mean_dataset_t
        delta_o = dataset_marg_o[:, di] - grand_mean_dataset_o
        _write_effect(f"beta_dataset[{d_name}]", delta_t, delta_o, grand_mean_dataset_o)

    # Arch effects: ALL archs vs grand mean (marginal over dataset)
    for a_name in archs:
        ai = archs.index(a_name)
        delta_t = arch_marg_t[:, ai] - grand_mean_arch_t
        delta_o = arch_marg_o[:, ai] - grand_mean_arch_o
        _write_effect(f"beta_arch[{a_name}]", delta_t, delta_o, grand_mean_arch_o)

    # Interaction effects: ALL dataset×arch cells
    for d_name in datasets:
        for a_name in archs:
            di = datasets.index(d_name)
            ai = archs.index(a_name)

            delta_t = (
                pred_draws_transformed[:, di, ai]
                - dataset_marg_t[:, di]
                - arch_marg_t[:, ai]
                + grand_mean_cell_t
            )
            delta_o = (
                pred_draws_original[:, di, ai]
                - dataset_marg_o[:, di]
                - arch_marg_o[:, ai]
                + grand_mean_cell_o
            )
            _write_effect(f"beta_interaction[{d_name}×{a_name}]", delta_t, delta_o, grand_mean_cell_o)

    return (
        idata,
        report,
        pred_table,
        pred_draws_transformed,
        pred_draws_original,
        (datasets, archs),
        post_pred,
    )


# Backwards-compatible alias (older entrypoint name)
fit_factorial_model_with_bf = fit_factorial_model