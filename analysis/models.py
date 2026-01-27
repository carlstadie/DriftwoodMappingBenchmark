"""
Bayesian hierarchical models for benchmark analysis.
"""

from typing import Dict, List, Optional, Sequence, Tuple

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm

from config import (
    PRIOR_SD_LOGIT,
    PRIOR_SD_LOG,
    DATASET_ORDER,
    ARCH_ORDER,
    HDI_PROB,
    DRAWS,
    TUNE,
    CHAINS,
    TARGET_ACCEPT,
    RANDOM_SEED,
)
from data_utils import transform_y, higher_is_better
from statistical_utils import (
    _pick_hdi_columns,
    _odds,
    _extract_draws_1d,
    rank_probabilities_from_draws,
    _summarize_draws,
    _bf10_from_prior_posterior,
    interpret_bf10,
)


def fit_group_model_with_ranking(
    df: pd.DataFrame,
    group_col: str,
    metric_col: str,
    bounded_metrics: set,
    maximize_metrics: set,
    draws=DRAWS,
    tune=TUNE,
    chains=CHAINS,
    target_accept=TARGET_ACCEPT,
    seed=RANDOM_SEED,
):
    """
    Fit hierarchical group model with hierarchical variance structure.
    
    Parameters
    ----------
    df : pd.DataFrame
        Data with group and metric columns
    group_col : str
        Column name for groups
    metric_col : str
        Column name for metric
    bounded_metrics : set
        Set of bounded (0,1) metric names
    maximize_metrics : set
        Set of metrics to maximize
    draws : int
        Number of posterior draws
    tune : int
        Number of tuning steps
    chains : int
        Number of chains
    target_accept : float
        Target acceptance rate
    seed : int
        Random seed
    
    Returns
    -------
    Tuple
        (idata, perf_table, rank_table, mu_draws, scale_name, groups, post_pred)
    """
    data = df[[group_col, metric_col]].dropna().copy()

    if pd.api.types.is_categorical_dtype(data[group_col]):
        data[group_col] = data[group_col].cat.remove_unused_categories()

    gcat = pd.Categorical(data[group_col])
    g_idx = gcat.codes
    groups = list(gcat.categories)
    y_raw = data[metric_col].to_numpy()
    y_t, inv, tname = transform_y(y_raw, metric_col, bounded_metrics)
    coords = {"group": groups}

    with pm.Model(coords=coords) as model:
        g = pm.Data("g", g_idx)
        y_obs = pm.Data("y_obs", y_t)

        # Hierarchical mean structure
        mu0 = pm.Normal("mu0", 0.0, 1.5)
        tau = pm.HalfNormal("tau", 1.0)
        mu = pm.Normal("mu", mu0, tau, dims="group")

        # Hierarchical variance structure
        sigma_mu = pm.HalfNormal("sigma_mu", sigma=1.0)
        sigma_tau = pm.HalfNormal("sigma_tau", sigma=0.5)
        sigma_offset = pm.HalfNormal("sigma_offset", sigma=1.0, dims="group")
        sigma = pm.Deterministic("sigma", sigma_mu + sigma_tau * sigma_offset, dims="group")

        # Robust likelihood
        nu = pm.Exponential("nu", 1 / 30) + 1
        pm.StudentT("y", nu=nu, mu=mu[g], sigma=sigma[g], observed=y_obs)

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

        # Prior predictive for BF
        prior = pm.sample_prior_predictive(var_names=["mu_orig"], random_seed=seed)

        # Posterior predictive check
        post_pred = pm.sample_posterior_predictive(idata, random_seed=seed)

    # Extract posterior draws
    post_mu_draws = (
        idata.posterior["mu_orig"]
        .stack(sample=("chain", "draw"))
        .transpose("sample", "group")
        .values
    )
    hib = higher_is_better(metric_col, maximize_metrics)
    post_rank_table = rank_probabilities_from_draws(
        post_mu_draws, groups, higher_better=hib
    )

    prior_mu_draws = (
        prior.prior["mu_orig"]
        .stack(sample=("chain", "draw"))
        .transpose("sample", "group")
        .values
    )
    prior_rank_table = rank_probabilities_from_draws(
        prior_mu_draws, groups, higher_better=hib
    )

    summ = az.summary(idata, var_names=["mu_orig"], hdi_prob=HDI_PROB).reset_index()
    summ = summ.rename(columns={"index": "param"})
    hdi_lo_col, hdi_hi_col = _pick_hdi_columns(summ)

    summ["group"] = summ["param"].str.replace(r"mu_orig\[|\]", "", regex=True)

    perf_table = (
        summ.loc[:, ["group", "mean", "sd", hdi_lo_col, hdi_hi_col]]
        .merge(post_rank_table, on="group", how="left")
        .merge(
            prior_rank_table.loc[:, ["group", "Pr(best)"]].rename(
                columns={"Pr(best)": "Pr(best)_prior"}
            ),
            on="group",
            how="left",
        )
        .sort_values("E[rank]")
        .reset_index(drop=True)
    )

    post_p = perf_table["Pr(best)"].to_numpy()
    prior_p = perf_table["Pr(best)_prior"].to_numpy()
    bf_best = _odds(post_p) / _odds(prior_p)
    perf_table["BF_best"] = bf_best

    return idata, perf_table, post_rank_table, post_mu_draws, tname, groups, post_pred


def add_implicit_effects_to_report(
    report: pd.DataFrame,
    idata: az.InferenceData,
    datasets: List[str],
    archs: List[str],
    tname: str,
    effect_sd: float,
    hdi_prob: float,
    hdi_lo_col: str,
    hdi_hi_col: str,
    sd_kde_bandwidths: List,
) -> pd.DataFrame:
    """
    Add implicit (sum-to-zero) effects to the factorial effects report.
    
    For effects coding, the last level of each factor is implicitly defined
    as the negative sum of the explicit levels.
    
    Parameters
    ----------
    report : pd.DataFrame
        Existing effects report
    idata : az.InferenceData
        Inference data with posterior and prior
    datasets : List[str]
        Dataset names
    archs : List[str]
        Architecture names
    tname : str
        Transform name ('logit' or 'log')
    effect_sd : float
        Prior standard deviation for effects
    hdi_prob : float
        HDI probability
    hdi_lo_col : str
        HDI lower column name
    hdi_hi_col : str
        HDI upper column name
    sd_kde_bandwidths : List
        KDE bandwidths for BF calculation
    
    Returns
    -------
    pd.DataFrame
        Updated report with implicit effects
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

    def _make_row(param: str, post_draws: np.ndarray, prior_draws: np.ndarray) -> Dict[str, object]:
        s = _summarize_1d(post_draws)
        pr_gt0 = float((np.asarray(post_draws).ravel() > 0).mean())

        bf_results = {}
        for bw in sd_kde_bandwidths:
            bf10 = _bf10_from_prior_posterior(prior_draws, post_draws, bw=bw)
            bw_str = bw if isinstance(bw, str) else f"{bw:.1f}"
            bf_results[f"BF10_bw_{bw_str}"] = bf10

        bf10_default = bf_results.get("BF10_bw_scott", bf_results[list(bf_results.keys())[0]])

        return {
            "param": param,
            "mean_transformed": s["mean"],
            "sd_transformed": s["sd"],
            f"{hdi_lo_col}_transformed": s["hdi_lo"],
            f"{hdi_hi_col}_transformed": s["hdi_hi"],
            "Pr(>0)": pr_gt0,
            "BF10_interpretation": interpret_bf10(bf10_default),
            **bf_results,
            "scale": tname,
            "prior": "Implied (sum-to-zero) from effects coding",
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
    derived_rows = []

    # Dataset: add last level
    if len(datasets) >= 2:
        d_last = datasets[-1]
        post_d = (
            idata.posterior["beta_dataset"]
            .stack(sample=("chain", "draw"))
            .transpose("sample", "dataset_effect")
            .to_numpy()
        )
        prior_d = (
            idata.prior["beta_dataset"]
            .stack(sample=("chain", "draw"))
            .transpose("sample", "dataset_effect")
            .to_numpy()
        )
        post_last = -post_d.sum(axis=1)
        prior_last = -prior_d.sum(axis=1)
        p = f"beta_dataset[{d_last}]"
        if p not in existing:
            derived_rows.append(_make_row(p, post_last, prior_last))
            existing.add(p)

    # Arch: add last level
    if len(archs) >= 2:
        a_last = archs[-1]
        post_a = (
            idata.posterior["beta_arch"]
            .stack(sample=("chain", "draw"))
            .transpose("sample", "arch_effect")
            .to_numpy()
        )
        prior_a = (
            idata.prior["beta_arch"]
            .stack(sample=("chain", "draw"))
            .transpose("sample", "arch_effect")
            .to_numpy()
        )
        post_last = -post_a.sum(axis=1)
        prior_last = -prior_a.sum(axis=1)
        p = f"beta_arch[{a_last}]"
        if p not in existing:
            derived_rows.append(_make_row(p, post_last, prior_last))
            existing.add(p)

    # Interactions: add all implicit cells
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
        prior_i = (
            idata.prior["beta_interaction"]
            .stack(sample=("chain", "draw"))
            .transpose("sample", "interaction_effect")
            .to_numpy()
        )

        post_grid = post_i.reshape(post_i.shape[0], K1, J1)
        prior_grid = prior_i.reshape(prior_i.shape[0], K1, J1)

        post_full = np.zeros((post_i.shape[0], K, J))
        prior_full = np.zeros((prior_i.shape[0], K, J))

        post_full[:, :K1, :J1] = post_grid
        prior_full[:, :K1, :J1] = prior_grid

        post_full[:, K1, :J1] = -post_full[:, :K1, :J1].sum(axis=1)
        prior_full[:, K1, :J1] = -prior_full[:, :K1, :J1].sum(axis=1)

        post_full[:, :K1, J1] = -post_full[:, :K1, :J1].sum(axis=2)
        prior_full[:, :K1, J1] = -prior_full[:, :K1, :J1].sum(axis=2)

        post_full[:, K1, J1] = post_full[:, :K1, :J1].sum(axis=(1, 2))
        prior_full[:, K1, J1] = prior_full[:, :K1, :J1].sum(axis=(1, 2))

        for di, d in enumerate(datasets):
            for ai, a in enumerate(archs):
                p = f"beta_interaction[{d}×{a}]"
                if p in existing:
                    continue
                derived_rows.append(_make_row(p, post_full[:, di, ai], prior_full[:, di, ai]))
                existing.add(p)

    if derived_rows:
        report = pd.concat([report, pd.DataFrame(derived_rows)], ignore_index=True)

    return report


def fit_factorial_model_with_bf(
    df: pd.DataFrame,
    dataset_col: str,
    arch_col: str,
    metric_col: str,
    bounded_metrics: set,
    sd_kde_bandwidths: List,
    draws=DRAWS,
    tune=TUNE,
    chains=CHAINS,
    target_accept=TARGET_ACCEPT,
    seed=RANDOM_SEED,
    hdi_prob=HDI_PROB,
):
    """
    Fit factorial model with effects coding and compute contrasts.
    
    Parameters
    ----------
    df : pd.DataFrame
        Data with dataset, arch, and metric columns
    dataset_col : str
        Dataset column name
    arch_col : str
        Architecture column name
    metric_col : str
        Metric column name
    bounded_metrics : set
        Set of bounded (0,1) metrics
    sd_kde_bandwidths : List
        KDE bandwidths for BF sensitivity
    draws : int
        Number of posterior draws
    tune : int
        Number of tuning steps
    chains : int
        Number of chains
    target_accept : float
        Target acceptance rate
    seed : int
        Random seed
    hdi_prob : float
        HDI probability
    
    Returns
    -------
    Tuple
        (idata, effects_report, pred_table, pred_draws_transformed, 
         pred_draws_original, (datasets, archs), post_pred)
    """
    data = df[[dataset_col, arch_col, metric_col]].dropna().copy()

    data[dataset_col] = pd.Categorical(
        data[dataset_col], categories=DATASET_ORDER, ordered=True
    )
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
    y_t, inv, tname = transform_y(y_raw, metric_col, bounded_metrics)
    effect_sd = PRIOR_SD_LOGIT if tname == "logit" else PRIOR_SD_LOG

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

        # Priors
        intercept = pm.Normal("intercept", 0.0, 1.5)
        beta_dataset = pm.Normal("beta_dataset", 0.0, effect_sd, dims="dataset_effect")
        beta_arch = pm.Normal("beta_arch", 0.0, effect_sd, dims="arch_effect")
        beta_interaction = pm.Normal(
            "beta_interaction", 0.0, effect_sd, dims="interaction_effect"
        )

        mu = intercept + pm.math.dot(d, beta_dataset) + pm.math.dot(
            a, beta_arch
        ) + pm.math.dot(da, beta_interaction)

        sigma = pm.HalfNormal("sigma", sigma=1.0)

        nu = pm.Exponential("nu", 1 / 30) + 1
        pm.StudentT("y", nu=nu, mu=mu, sigma=sigma, observed=y_t)

        idata = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            target_accept=target_accept,
            random_seed=seed,
            return_inferencedata=True,
        )

        prior = pm.sample_prior_predictive(
            var_names=["beta_dataset", "beta_arch", "beta_interaction"],
            random_seed=seed,
        )
        idata.extend(prior)

        post_pred = pm.sample_posterior_predictive(idata, random_seed=seed)

    summary = (
        az.summary(
            idata,
            var_names=["beta_dataset", "beta_arch", "beta_interaction"],
            hdi_prob=hdi_prob,
        )
        .reset_index()
        .rename(columns={"index": "param"})
    )
    hdi_lo_col, hdi_hi_col = _pick_hdi_columns(summary)

    # Calculate effects + BF sensitivity
    rows = []

    def _add_param_rows(var_name: str, dim_name: str) -> None:
        extracted = az.extract(idata, group="posterior", var_names=[var_name])
        from statistical_utils import _as_dataarray
        da_ex = _as_dataarray(extracted, var_name)

        coord_vals = list(da_ex.coords[dim_name].values)
        for dim_val in coord_vals:
            sel = {dim_name: dim_val}
            post_draws = _extract_draws_1d(idata, "posterior", var_name, sel=sel)
            prior_draws = _extract_draws_1d(idata, "prior", var_name, sel=sel)

            pr_gt0 = float((post_draws > 0).mean())

            bf_results = {}
            for bw in sd_kde_bandwidths:
                bf10 = _bf10_from_prior_posterior(prior_draws, post_draws, bw=bw)
                bw_str = bw if isinstance(bw, str) else f"{bw:.1f}"
                bf_results[f"BF10_bw_{bw_str}"] = bf10

            bf10_default = bf_results.get("BF10_bw_scott", bf_results[list(bf_results.keys())[0]])

            param_str = f"{var_name}[{dim_val}]"
            base_row = summary.loc[summary["param"] == param_str].iloc[0].to_dict()

            row = {
                "param": param_str,
                "mean_transformed": base_row["mean"],
                "sd_transformed": base_row["sd"],
                f"{hdi_lo_col}_transformed": base_row[hdi_lo_col],
                f"{hdi_hi_col}_transformed": base_row[hdi_hi_col],
                "Pr(>0)": pr_gt0,
                "BF10_interpretation": interpret_bf10(bf10_default),
                **bf_results,
                "scale": tname,
                "prior": f"Normal(0, {effect_sd}) on {tname} scale",
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
            rows.append(row)

    _add_param_rows("beta_dataset", "dataset_effect")
    _add_param_rows("beta_arch", "arch_effect")
    _add_param_rows("beta_interaction", "interaction_effect")

    report = pd.DataFrame(rows)

    datasets = list(dcat.categories)
    archs = list(acat.categories)

    # Add implicit effects
    report = add_implicit_effects_to_report(
        report=report,
        idata=idata,
        datasets=datasets,
        archs=archs,
        tname=tname,
        effect_sd=effect_sd,
        hdi_prob=hdi_prob,
        hdi_lo_col=hdi_lo_col,
        hdi_hi_col=hdi_hi_col,
        sd_kde_bandwidths=sd_kde_bandwidths,
    )

    # Compute predicted means (code continues in next message due to length)
    # ... [prediction code from original implementation]
    
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

    # Stack chain/draw properly
    intercept_s = idata.posterior["intercept"].stack(sample=("chain", "draw")).to_numpy().ravel()
    beta_d_s = (
        idata.posterior["beta_dataset"]
        .stack(sample=("chain", "draw"))
        .transpose("sample", "dataset_effect")
        .to_numpy()
    )
    beta_a_s = (
        idata.posterior["beta_arch"]
        .stack(sample=("chain", "draw"))
        .transpose("sample", "arch_effect")
        .to_numpy()
    )
    beta_i_s = (
        idata.posterior["beta_interaction"]
        .stack(sample=("chain", "draw"))
        .transpose("sample", "interaction_effect")
        .to_numpy()
    )

    pred_draws_transformed = np.zeros((intercept_s.shape[0], len(datasets), len(archs)))
    pred_draws_original = np.zeros((intercept_s.shape[0], len(datasets), len(archs)))
    pred_rows = []

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

            mu_stats_orig = _summarize_draws(mu_orig, hdi_prob=hdi_prob)
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

    # Fill interpretable effect sizes via contrasts
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
        s_o = _summarize_draws(delta_o, hdi_prob=hdi_prob)
        s_base = _summarize_draws(baseline_o, hdi_prob=hdi_prob)

        if tname == "log":
            ratio = np.exp(delta_t)
            pct = (ratio - 1.0) * 100.0
            s_pct = _summarize_draws(pct, hdi_prob=hdi_prob)

            report.loc[report["param"] == param, "Abs_Δ_mean"] = s_o["mean"]
            report.loc[report["param"] == param, "Abs_Δ_lo"] = s_o["hdi_lo"]
            report.loc[report["param"] == param, "Abs_Δ_hi"] = s_o["hdi_hi"]
            report.loc[report["param"] == param, "%_change"] = s_pct["mean"]
            report.loc[report["param"] == param, "baseline"] = s_base["mean"]
        else:
            OR = np.exp(delta_t)
            s_or = _summarize_draws(OR, hdi_prob=hdi_prob)

            report.loc[report["param"] == param, "OR"] = s_or["mean"]
            report.loc[report["param"] == param, "Δp_mean"] = s_o["mean"]
            report.loc[report["param"] == param, "Δp_lo"] = s_o["hdi_lo"]
            report.loc[report["param"] == param, "Δp_hi"] = s_o["hdi_hi"]
            report.loc[report["param"] == param, "baseline"] = s_base["mean"]

    # Dataset coefficients
    for d_name in datasets:
        di = datasets.index(d_name)
        delta_t = dataset_marg_t[:, di] - grand_mean_dataset_t
        delta_o = dataset_marg_o[:, di] - grand_mean_dataset_o
        base_o = grand_mean_dataset_o
        _write_effect(f"beta_dataset[{d_name}]", delta_t, delta_o, base_o)

    # Arch coefficients
    for a_name in archs:
        ai = archs.index(a_name)
        delta_t = arch_marg_t[:, ai] - grand_mean_arch_t
        delta_o = arch_marg_o[:, ai] - grand_mean_arch_o
        base_o = grand_mean_arch_o
        _write_effect(f"beta_arch[{a_name}]", delta_t, delta_o, base_o)

    # Interaction coefficients
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

            base_o = grand_mean_cell_o
            _write_effect(f"beta_interaction[{d_name}×{a_name}]", delta_t, delta_o, base_o)

    return (
        idata,
        report,
        pred_table,
        pred_draws_transformed,
        pred_draws_original,
        (datasets, archs),
        post_pred,
    )


def overall_ranking_across_metrics(
    mu_draws_by_metric: Dict[str, np.ndarray],
    higher_is_better_by_metric: Dict[str, bool],
    group_names: Sequence[str],
    weights: Optional[Dict[str, float]] = None,
):
    """
    Compute overall ranking across multiple metrics using z-score standardization.
    
    Parameters
    ----------
    mu_draws_by_metric : Dict[str, np.ndarray]
        Posterior draws for each metric, shape (samples, groups)
    higher_is_better_by_metric : Dict[str, bool]
        Whether higher is better for each metric
    group_names : Sequence[str]
        Group names
    weights : Optional[Dict[str, float]]
        Metric weights (None = equal weights)
    
    Returns
    -------
    Tuple
        (composite_draws, ranking_table, weights_used)
    """
    metric_list = list(mu_draws_by_metric.keys())

    if weights is None:
        weights = {m: 1.0 for m in metric_list}
    wsum = sum(weights.values())
    weights = {m: weights[m] / wsum for m in metric_list}

    s_count = min(mu_draws_by_metric[m].shape[0] for m in metric_list)
    g_count = len(group_names)

    composite = np.zeros((s_count, g_count))
    for m in metric_list:
        mu = mu_draws_by_metric[m][:s_count, :]
        score = mu if higher_is_better_by_metric[m] else -mu

        sd = score.std() + 1e-12
        z = score / sd

        composite += weights[m] * z

    overall_rank_table = rank_probabilities_from_draws(
        composite, group_names, higher_better=True
    )

    return composite, overall_rank_table, weights