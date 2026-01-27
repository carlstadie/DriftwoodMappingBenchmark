"""
Data loading and transformation utilities.
"""

import os
from typing import Callable, List, Tuple

import numpy as np
import pandas as pd


def load_checkpoint_metrics(csv_specs: List[Tuple], metrics: List[str]) -> pd.DataFrame:
    """
    Load and combine checkpoint metrics from multiple CSVs.
    
    Parameters
    ----------
    csv_specs : List[Tuple]
        List of (path, group_label, dataset, arch) tuples
    metrics : List[str]
        Metrics to extract
    
    Returns
    -------
    pd.DataFrame
        Combined DataFrame with all metrics
    """
    dfs = []
    for path, group, dataset, arch in csv_specs:
        if not os.path.exists(path):
            print(f"Warning: path {path} does not exist, skipping.")
            continue
        df = pd.read_csv(path)
        needed = [m for m in metrics if m in df.columns]
        subset = df[needed].copy()
        subset["group"] = group
        subset["dataset"] = dataset
        subset["arch"] = arch
        dfs.append(subset)
    return pd.concat(dfs, ignore_index=True)


def base_metric_name(metric_col: str) -> str:
    """
    Extract base metric name (remove 'val_' prefix if present).
    
    Parameters
    ----------
    metric_col : str
        Metric column name
    
    Returns
    -------
    str
        Base metric name
    """
    return metric_col[4:] if metric_col.startswith("val_") else metric_col


def is_bounded_01(metric_col: str, bounded_metrics: set) -> bool:
    """
    Check if metric is bounded in (0, 1).
    
    Parameters
    ----------
    metric_col : str
        Metric column name
    bounded_metrics : set
        Set of metric names that are bounded
    
    Returns
    -------
    bool
        True if metric is bounded in (0, 1)
    """
    return base_metric_name(metric_col) in bounded_metrics


def higher_is_better(metric_col: str, maximize_metrics: set) -> bool:
    """
    Check if higher values are better for this metric.
    
    Parameters
    ----------
    metric_col : str
        Metric column name
    maximize_metrics : set
        Set of metric names to maximize
    
    Returns
    -------
    bool
        True if higher is better
    """
    m = base_metric_name(metric_col)
    return m in maximize_metrics


def _inv_logit(x):
    """Inverse logit transformation (works with PyMC and NumPy)."""
    try:
        import pymc as pm
        if hasattr(x, "type"):
            return pm.math.sigmoid(x)
    except ImportError:
        pass
    return 1.0 / (1.0 + np.exp(-x))


def _inv_exp(x):
    """Inverse exponential transformation (works with PyMC and NumPy)."""
    try:
        import pymc as pm
        if hasattr(x, "type"):
            return pm.math.exp(x)
    except ImportError:
        pass
    return np.exp(x)


def transform_y(
    y: np.ndarray,
    metric_col: str,
    bounded_metrics: set,
    eps: float = 1e-6,
) -> Tuple[np.ndarray, Callable, str]:
    """
    Transform metric to appropriate scale for modeling.
    
    Parameters
    ----------
    y : np.ndarray
        Original metric values
    metric_col : str
        Metric name
    bounded_metrics : set
        Set of bounded (0,1) metric names
    eps : float
        Small value to prevent log(0)
    
    Returns
    -------
    Tuple[np.ndarray, Callable, str]
        (transformed_y, inverse_transform_func, scale_name)
    """
    if is_bounded_01(metric_col, bounded_metrics):
        y_clip = np.clip(y, eps, 1 - eps)
        y_t = np.log(y_clip / (1 - y_clip))
        return y_t, _inv_logit, "logit"

    y_clip = np.clip(y, eps, None)
    y_t = np.log(y_clip)
    return y_t, _inv_exp, "log"


def generate_test_data(
    dataset_order: List[str],
    arch_order: List[str],
    metrics: List[str],
    group_order: List[str],
    n_rows: int = 10,
    seed: int = 1701,
) -> pd.DataFrame:
    """
    Generate synthetic test data with known structure.
    
    Parameters
    ----------
    dataset_order : List[str]
        Dataset names
    arch_order : List[str]
        Architecture names
    metrics : List[str]
        Metric names
    group_order : List[str]
        Group labels
    n_rows : int
        Samples per group
    seed : int
        Random seed
    
    Returns
    -------
    pd.DataFrame
        Synthetic data
    """
    rng = np.random.default_rng(seed)

    def make_group(dataset: str, arch: str) -> str:
        if arch == "U-Net":
            return f"U-Net | {dataset}"
        if arch == "Swin":
            return f"Swin U-Net | {dataset}"
        if arch == "Terramind":
            return f"Terramind | {dataset}"
        return f"{arch} | {dataset}"

    kappa_by_metric = {m: 120.0 for m in metrics}

    base_mean = {
        "IoU": 0.45,
        "dice_coef": 0.55,
        "normalized_surface_distance": 0.35,
        "mean_epistemic_uncertainty": 0.25,
        "mean_aleatoric_uncertainty": 0.25,
    }

    arch_shift = {
        "U-Net": {m: 0.00 for m in metrics},
        "Swin": {
            "IoU": 0.15,
            "dice_coef": 0.12,
            "normalized_surface_distance": 0.05,
            "mean_epistemic_uncertainty": -0.03,
            "mean_aleatoric_uncertainty": -0.02,
        },
        "Terramind": {
            "IoU": -0.10,
            "dice_coef": -0.10,
            "normalized_surface_distance": -0.05,
            "mean_epistemic_uncertainty": 0.08,
            "mean_aleatoric_uncertainty": 0.06,
        },
    }

    dataset_shift = {
        "AE": {
            "IoU": 0.10,
            "dice_coef": 0.08,
            "normalized_surface_distance": 0.03,
            "mean_epistemic_uncertainty": -0.02,
            "mean_aleatoric_uncertainty": -0.01,
        },
        "PS": {m: 0.00 for m in metrics},
        "S2": {
            "IoU": -0.12,
            "dice_coef": -0.10,
            "normalized_surface_distance": -0.05,
            "mean_epistemic_uncertainty": 0.05,
            "mean_aleatoric_uncertainty": 0.04,
        },
    }

    mean_jitter_sd = 0.015

    def beta_sample(mu: float, kappa: float, n: int) -> np.ndarray:
        mu = float(np.clip(mu, 1e-4, 1 - 1e-4))
        a = mu * kappa
        b = (1.0 - mu) * kappa
        return rng.beta(a, b, size=n)

    test_frames = []
    for dataset in dataset_order:
        for arch in arch_order:
            group = make_group(dataset, arch)

            row = {
                "dataset": [dataset] * n_rows,
                "arch": [arch] * n_rows,
                "group": [group] * n_rows,
            }
            for metric in metrics:
                mu = (
                    base_mean.get(metric, 0.5)
                    + arch_shift.get(arch, {}).get(metric, 0.0)
                    + dataset_shift.get(dataset, {}).get(metric, 0.0)
                    + rng.normal(0.0, mean_jitter_sd)
                )
                mu = float(np.clip(mu, 1e-3, 1 - 1e-3))
                row[metric] = beta_sample(mu, kappa_by_metric.get(metric, 100.0), n_rows)

            test_frames.append(pd.DataFrame(row))

    return pd.concat(test_frames, ignore_index=True)