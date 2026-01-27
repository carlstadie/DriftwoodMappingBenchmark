"""
Contrast computation for factorial designs.
"""

from typing import Sequence

import numpy as np
import pandas as pd

from statistical_utils import summarize_contrast


def contrasts_within_each_dataset(
    pred_draws_transformed: np.ndarray,
    pred_draws_original: np.ndarray,
    datasets: Sequence[str],
    archs: Sequence[str],
    baseline_arch: str,
    rope_transformed: float,
    hdi_prob: float,
) -> pd.DataFrame:
    """
    Compute architecture contrasts within each dataset.
    
    Parameters
    ----------
    pred_draws_transformed : np.ndarray
        Predicted values on transformed scale, shape (samples, datasets, archs)
    pred_draws_original : np.ndarray
        Predicted values on original scale, shape (samples, datasets, archs)
    datasets : Sequence[str]
        Dataset names
    archs : Sequence[str]
        Architecture names
    baseline_arch : str
        Baseline architecture for contrasts
    rope_transformed : float
        ROPE threshold on transformed scale
    hdi_prob : float
        HDI probability
    
    Returns
    -------
    pd.DataFrame
        Contrast results
    """
    if baseline_arch not in archs:
        baseline_arch = archs[0]

    a0 = archs.index(baseline_arch)
    rows = []
    for di, d in enumerate(datasets):
        base_t = pred_draws_transformed[:, di, a0]
        base_o = pred_draws_original[:, di, a0]
        for ai, a in enumerate(archs):
            if a == baseline_arch:
                continue
            delta_t = pred_draws_transformed[:, di, ai] - base_t
            delta_o = pred_draws_original[:, di, ai] - base_o
            stats = summarize_contrast(delta_t, delta_o, rope_transformed, hdi_prob)
            rows.append({"dataset": d, "contrast": f"{a} − {baseline_arch}", **stats})
    return pd.DataFrame(rows)


def contrasts_within_each_architecture(
    pred_draws_transformed: np.ndarray,
    pred_draws_original: np.ndarray,
    datasets: Sequence[str],
    archs: Sequence[str],
    baseline_dataset: str,
    rope_transformed: float,
    hdi_prob: float,
) -> pd.DataFrame:
    """
    Compute dataset contrasts within each architecture.
    
    Parameters
    ----------
    pred_draws_transformed : np.ndarray
        Predicted values on transformed scale, shape (samples, datasets, archs)
    pred_draws_original : np.ndarray
        Predicted values on original scale, shape (samples, datasets, archs)
    datasets : Sequence[str]
        Dataset names
    archs : Sequence[str]
        Architecture names
    baseline_dataset : str
        Baseline dataset for contrasts
    rope_transformed : float
        ROPE threshold on transformed scale
    hdi_prob : float
        HDI probability
    
    Returns
    -------
    pd.DataFrame
        Contrast results
    """
    if baseline_dataset not in datasets:
        baseline_dataset = datasets[0]

    d0 = datasets.index(baseline_dataset)
    rows = []
    for ai, a in enumerate(archs):
        base_t = pred_draws_transformed[:, d0, ai]
        base_o = pred_draws_original[:, d0, ai]
        for di, d in enumerate(datasets):
            if d == baseline_dataset:
                continue
            delta_t = pred_draws_transformed[:, di, ai] - base_t
            delta_o = pred_draws_original[:, di, ai] - base_o
            stats = summarize_contrast(delta_t, delta_o, rope_transformed, hdi_prob)
            rows.append({"arch": a, "contrast": f"{d} − {baseline_dataset}", **stats})
    return pd.DataFrame(rows)


def contrasts_arch_overall(
    pred_draws_transformed: np.ndarray,
    pred_draws_original: np.ndarray,
    archs: Sequence[str],
    baseline_arch: str,
    rope_transformed: float,
    hdi_prob: float,
) -> pd.DataFrame:
    """
    Compute overall architecture contrasts (marginalized over datasets).
    
    Parameters
    ----------
    pred_draws_transformed : np.ndarray
        Predicted values on transformed scale, shape (samples, datasets, archs)
    pred_draws_original : np.ndarray
        Predicted values on original scale, shape (samples, datasets, archs)
    archs : Sequence[str]
        Architecture names
    baseline_arch : str
        Baseline architecture
    rope_transformed : float
        ROPE threshold
    hdi_prob : float
        HDI probability
    
    Returns
    -------
    pd.DataFrame
        Overall contrasts
    """
    if baseline_arch not in archs:
        baseline_arch = archs[0]
    a0 = archs.index(baseline_arch)

    marg_t = pred_draws_transformed.mean(axis=1)  # (samples, arch)
    marg_o = pred_draws_original.mean(axis=1)

    base_t = marg_t[:, a0]
    base_o = marg_o[:, a0]

    rows = []
    for ai, a in enumerate(archs):
        if a == baseline_arch:
            continue
        delta_t = marg_t[:, ai] - base_t
        delta_o = marg_o[:, ai] - base_o
        stats = summarize_contrast(delta_t, delta_o, rope_transformed, hdi_prob)
        rows.append({
            "scope": "Overall (avg over datasets)",
            "contrast": f"{a} − {baseline_arch}",
            **stats
        })
    return pd.DataFrame(rows)


def contrasts_dataset_overall(
    pred_draws_transformed: np.ndarray,
    pred_draws_original: np.ndarray,
    datasets: Sequence[str],
    baseline_dataset: str,
    rope_transformed: float,
    hdi_prob: float,
) -> pd.DataFrame:
    """
    Compute overall dataset contrasts (marginalized over architectures).
    
    Parameters
    ----------
    pred_draws_transformed : np.ndarray
        Predicted values on transformed scale, shape (samples, datasets, archs)
    pred_draws_original : np.ndarray
        Predicted values on original scale, shape (samples, datasets, archs)
    datasets : Sequence[str]
        Dataset names
    baseline_dataset : str
        Baseline dataset
    rope_transformed : float
        ROPE threshold
    hdi_prob : float
        HDI probability
    
    Returns
    -------
    pd.DataFrame
        Overall contrasts
    """
    if baseline_dataset not in datasets:
        baseline_dataset = datasets[0]
    d0 = datasets.index(baseline_dataset)

    marg_t = pred_draws_transformed.mean(axis=2)  # (samples, dataset)
    marg_o = pred_draws_original.mean(axis=2)

    base_t = marg_t[:, d0]
    base_o = marg_o[:, d0]

    rows = []
    for di, d in enumerate(datasets):
        if d == baseline_dataset:
            continue
        delta_t = marg_t[:, di] - base_t
        delta_o = marg_o[:, di] - base_o
        stats = summarize_contrast(delta_t, delta_o, rope_transformed, hdi_prob)
        rows.append({
            "scope": "Overall (avg over arch)",
            "contrast": f"{d} − {baseline_dataset}",
            **stats
        })
    return pd.DataFrame(rows)