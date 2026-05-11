# transforms.py
from __future__ import annotations

from typing import Callable, Tuple
import warnings
import numpy as np
import pymc as pm

from config import BOUNDED_01_METRICS


def base_metric_name(metric_col: str) -> str:
    return metric_col[4:] if metric_col.startswith("val_") else metric_col


def is_bounded_01(metric_col: str) -> bool:
    return base_metric_name(metric_col) in BOUNDED_01_METRICS


def higher_is_better(metric_col: str) -> bool:
    """Determine if higher values are better for a metric.
    Handles both raw and prefixed metric names.
    """
    m = base_metric_name(metric_col)
    if m in {"IoU", "dice_coef", "normalized_surface_distance"}:
        return True
    if m in {"mean_epistemic_uncertainty", "mean_aleatoric_uncertainty"}:
        return False
    raise ValueError(f"Unknown metric direction for '{m}'. Please add explicit handling.")


def _inv_logit(x):
    if hasattr(x, "type"):
        return pm.math.sigmoid(x)
    return 1.0 / (1.0 + np.exp(-x))


def _inv_exp(x):
    if hasattr(x, "type"):
        return pm.math.exp(x)
    return np.exp(x)


def transform_y(
    y: np.ndarray,
    metric_col: str,
    eps_logit: float = 1e-6,
    eps_log: float = 1e-12,
) -> Tuple[np.ndarray, Callable, str]:
    if is_bounded_01(metric_col):
        y_clip = np.clip(y, eps_logit, 1 - eps_logit)
        y_t = np.log(y_clip / (1 - y_clip))
        return y_t, _inv_logit, "logit"

    # Log transform for unbounded metrics (uncertainties)
    y_arr = np.asarray(y, dtype=float)

    # Check for negative values and handle appropriately
    y_min = np.min(y_arr)
    if y_min < 0:
        if y_min >= -1e-6:
            # Clamp tiny negative numerical noise to zero
            y_arr = np.maximum(y_arr, 0.0)
            warnings.warn(
                f"Metric '{metric_col}' has small negative values (min={y_min:.2e}). "
                "Clamping to 0.0."
            )
        else:
            raise ValueError(
                f"Metric '{metric_col}' has negative values (min={y_min:.2e} < -1e-6), "
                "invalid for log transform"
            )

    # Clip with eps_log and warn if many values are clipped
    y_clip = np.clip(y_arr, eps_log, None)
    frac_clipped = np.mean(y_arr <= eps_log)
    if frac_clipped > 0.5:
        warnings.warn(
            f"Metric '{metric_col}': {frac_clipped*100:.1f}% of values clipped to eps_log={eps_log}"
        )

    # Take log
    y_t = np.log(y_clip)
    return y_t, _inv_exp, "log"
