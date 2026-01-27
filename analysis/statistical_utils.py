"""
Core statistical utilities for Bayesian analysis.
"""

from typing import Callable, Dict, Optional, Sequence, Tuple

import arviz as az
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde


def _summarize_draws(draws: np.ndarray, hdi_prob: float) -> Dict[str, float]:
    """
    Summarize posterior draws with mean and HDI.
    
    Parameters
    ----------
    draws : np.ndarray
        Posterior draws
    hdi_prob : float
        HDI probability (e.g., 0.95 for 95% HDI)
    
    Returns
    -------
    Dict[str, float]
        Dictionary with 'mean', 'hdi_lo', 'hdi_hi'
    """
    draws = np.asarray(draws).ravel()
    lo_q = (1.0 - hdi_prob) / 2.0
    hi_q = 1.0 - lo_q
    return {
        "mean": float(np.mean(draws)),
        "hdi_lo": float(np.quantile(draws, lo_q)),
        "hdi_hi": float(np.quantile(draws, hi_q)),
    }


def _density_at_zero(draws_1d: np.ndarray, bw) -> float:
    """
    Compute KDE density at zero.
    
    Parameters
    ----------
    draws_1d : np.ndarray
        1D array of draws
    bw : str or float
        KDE bandwidth
    
    Returns
    -------
    float
        Density value at zero
    """
    draws_1d = np.asarray(draws_1d).ravel()
    kde = gaussian_kde(draws_1d, bw_method=bw)
    return float(kde.evaluate(0.0)[0])


def _bf10_from_prior_posterior(
    prior_draws: np.ndarray,
    post_draws: np.ndarray,
    bw,
) -> float:
    """
    Compute Savage-Dickey Bayes Factor BF10.
    
    BF10 = (prior density at 0) / (posterior density at 0)
    
    Parameters
    ----------
    prior_draws : np.ndarray
        Prior samples
    post_draws : np.ndarray
        Posterior samples
    bw : str or float
        KDE bandwidth
    
    Returns
    -------
    float
        Bayes factor for H1 vs H0
    """
    prior0 = _density_at_zero(prior_draws, bw=bw)
    post0 = _density_at_zero(post_draws, bw=bw)
    if post0 == 0 or post0 < 1e-10:
        return np.inf
    bf10 = float(prior0 / post0)
    # Cap overly large BFs to inf for display
    if bf10 > 1e6:
        return np.inf
    return bf10


def interpret_bf10(bf10: float) -> str:
    """
    Qualitative interpretation of Bayes Factor.
    
    Parameters
    ----------
    bf10 : float
        Bayes factor value
    
    Returns
    -------
    str
        Interpretation string
    """
    if bf10 < 1 / 10:
        return "strong evidence for H0 (no effect)"
    if bf10 < 1 / 3:
        return "moderate evidence for H0"
    if bf10 < 1:
        return "anecdotal evidence for H0"
    if bf10 < 3:
        return "anecdotal evidence for H1 (effect)"
    if bf10 < 10:
        return "moderate evidence for H1"
    if bf10 < 30:
        return "strong evidence for H1"
    if bf10 < 100:
        return "very strong evidence for H1"
    return "extreme evidence for H1"


def _odds(p: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Convert probability to odds.
    
    Parameters
    ----------
    p : np.ndarray
        Probability values
    eps : float
        Small value to prevent division by zero
    
    Returns
    -------
    np.ndarray
        Odds values
    """
    p = np.clip(p, eps, 1.0 - eps)
    return p / (1.0 - p)


def _pick_hdi_columns(summary_df: pd.DataFrame) -> Tuple[str, str]:
    """
    Identify HDI column names from ArviZ summary.
    
    Parameters
    ----------
    summary_df : pd.DataFrame
        ArviZ summary DataFrame
    
    Returns
    -------
    Tuple[str, str]
        (lower_hdi_col, upper_hdi_col)
    """
    hdi_cols = [
        c for c in summary_df.columns if c.startswith("hdi_") and c.endswith("%")
    ]
    if len(hdi_cols) < 2:
        raise ValueError(f"Could not find HDI columns in: {list(summary_df.columns)}")

    def _as_float(col: str) -> float:
        return float(col.replace("hdi_", "").replace("%", ""))

    hdi_cols_sorted = sorted(hdi_cols, key=_as_float)
    return hdi_cols_sorted[0], hdi_cols_sorted[-1]


def _as_dataarray(extracted, var_name: str):
    """Extract DataArray from ArviZ extraction."""
    if hasattr(extracted, "data_vars") and var_name in extracted.data_vars:
        return extracted[var_name]
    return extracted


def _extract_draws_1d(
    idata: az.InferenceData,
    group: str,
    var_name: str,
    sel: Optional[Dict[str, str]] = None,
) -> np.ndarray:
    """
    Robust extraction of 1D draws from InferenceData.
    
    Stacks (chain, draw) dimensions into a single sample dimension.
    
    Parameters
    ----------
    idata : az.InferenceData
        Inference data object
    group : str
        Group name (e.g., 'posterior', 'prior')
    var_name : str
        Variable name
    sel : Optional[Dict[str, str]]
        Selection dictionary for coordinates
    
    Returns
    -------
    np.ndarray
        1D array of draws
    """
    grp = getattr(idata, group)
    da = grp[var_name]
    if sel:
        da = da.sel(sel)
    if "chain" in da.dims and "draw" in da.dims:
        da = da.stack(sample=("chain", "draw"))
        return da.to_numpy().ravel()
    return da.to_numpy().ravel()


def rank_probabilities_from_draws(
    draws: np.ndarray,
    group_names: Sequence[str],
    higher_better: bool,
) -> pd.DataFrame:
    """
    Compute ranking probabilities from posterior draws.
    
    Parameters
    ----------
    draws : np.ndarray
        Posterior draws, shape (samples, groups)
    group_names : Sequence[str]
        Group names
    higher_better : bool
        Whether higher values are better
    
    Returns
    -------
    pd.DataFrame
        Ranking table with Pr(rank=k), E[rank], Pr(best)
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


def summarize_contrast(
    delta_transformed: np.ndarray,
    delta_original: np.ndarray,
    rope_transformed: float,
    hdi_prob: float,
) -> Dict[str, float]:
    """
    Summarize contrasts on both transformed and original scales.
    
    Parameters
    ----------
    delta_transformed : np.ndarray
        Contrast on transformed scale
    delta_original : np.ndarray
        Contrast on original scale
    rope_transformed : float
        ROPE threshold on transformed scale
    hdi_prob : float
        HDI probability
    
    Returns
    -------
    Dict[str, float]
        Summary statistics including mean, HDI, Pr(>0), Pr(|Δ|<ROPE)
    """
    s_t = _summarize_draws(delta_transformed, hdi_prob=hdi_prob)
    s_o = _summarize_draws(delta_original, hdi_prob=hdi_prob)

    return {
        "mean_transformed": s_t["mean"],
        "hdi_lo_transformed": s_t["hdi_lo"],
        "hdi_hi_transformed": s_t["hdi_hi"],
        "mean_original": s_o["mean"],
        "hdi_lo_original": s_o["hdi_lo"],
        "hdi_hi_original": s_o["hdi_hi"],
        "Pr(>0)": float((delta_transformed > 0).mean()),
        "Pr(<0)": float((delta_transformed < 0).mean()),
        "Pr(|Δ|<ROPE)": float((np.abs(delta_transformed) < rope_transformed).mean()),
    }