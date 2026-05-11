# bayes_utils.py
from __future__ import annotations

from typing import Dict, Optional, Tuple
import numpy as np
import arviz as az
from scipy.stats import gaussian_kde


def summarize_draws(draws: np.ndarray, hdi_prob: float) -> Dict[str, float]:
    draws = np.asarray(draws).ravel()
    hdi = az.hdi(draws, hdi_prob=hdi_prob)
    return {
        "mean": float(np.mean(draws)),
        "hdi_lo": float(hdi[0]),
        "hdi_hi": float(hdi[1]),
    }


def density_at_zero(draws_1d: np.ndarray, bw) -> float:
    draws_1d = np.asarray(draws_1d).ravel()
    kde = gaussian_kde(draws_1d, bw_method=bw)
    return float(kde.evaluate(0.0)[0])


def bf10_from_prior_posterior(
    prior_draws: np.ndarray,
    post_draws: np.ndarray,
    bw,
) -> float:
    prior0 = density_at_zero(prior_draws, bw=bw)
    post0 = density_at_zero(post_draws, bw=bw)
    if post0 == 0 or post0 < 1e-10:
        return np.inf
    bf10 = float(prior0 / post0)
    return np.inf if bf10 > 1e6 else bf10


def interpret_bf10(bf10: float) -> str:
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


def odds(p: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    p = np.clip(p, eps, 1.0 - eps)
    return p / (1.0 - p)


def pick_hdi_columns(summary_df) -> Tuple[str, str]:
    hdi_cols = [c for c in summary_df.columns if c.startswith("hdi_") and c.endswith("%")]
    if len(hdi_cols) < 2:
        raise ValueError(f"Could not find HDI columns in: {list(summary_df.columns)}")

    def _as_float(col: str) -> float:
        return float(col.replace("hdi_", "").replace("%", ""))

    hdi_cols_sorted = sorted(hdi_cols, key=_as_float)
    return hdi_cols_sorted[0], hdi_cols_sorted[-1]


def extract_draws_1d(
    idata: az.InferenceData,
    group: str,
    var_name: str,
    sel: Optional[Dict[str, str]] = None,
) -> np.ndarray:
    """
    Robust draw extractor: stacks (chain, draw) -> sample and preserves coord order.
    Works for posterior and prior groups.
    """
    grp = getattr(idata, group)
    da = grp[var_name]
    if sel:
        da = da.sel(sel)
    if "chain" in da.dims and "draw" in da.dims:
        da = da.stack(sample=("chain", "draw"))
        return da.to_numpy().ravel()
    return da.to_numpy().ravel()
