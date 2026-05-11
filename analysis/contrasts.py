# contrasts.py
from __future__ import annotations

from typing import Dict, Sequence
import numpy as np
import pandas as pd

from bayes_utils import summarize_draws


def summarize_contrast(
    delta_transformed: np.ndarray,
    delta_original: np.ndarray,
    rope_transformed: float,
    hdi_prob: float,
) -> Dict[str, float]:
    """
    Summarize contrasts on both transformed and original scales.
    ROPE is applied on transformed scale for consistency with model.
    """
    s_t = summarize_draws(delta_transformed, hdi_prob=hdi_prob)
    s_o = summarize_draws(delta_original, hdi_prob=hdi_prob)

    dt = np.asarray(delta_transformed).ravel()
    pr_gt0 = float((dt > 0).mean())
    pr_lt0 = float((dt < 0).mean())
    pr_in_rope = float((np.abs(dt) < rope_transformed).mean())
    pr_gt_rope = float((dt > rope_transformed).mean())
    pr_lt_minus_rope = float((dt < -rope_transformed).mean())

    return {
        "mean_transformed": s_t["mean"],
        "hdi_lo_transformed": s_t["hdi_lo"],
        "hdi_hi_transformed": s_t["hdi_hi"],
        "mean_original": s_o["mean"],
        "hdi_lo_original": s_o["hdi_lo"],
        "hdi_hi_original": s_o["hdi_hi"],
        "Pr(>0)": pr_gt0,
        "Pr(<0)": pr_lt0,
        "Pr(Δ>ROPE)": pr_gt_rope,
        "Pr(Δ<-ROPE)": pr_lt_minus_rope,
        "Pr(|Δ|<ROPE)": pr_in_rope,
    }


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
    Architecture contrasts within each dataset (baseline_arch within each dataset).
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
    Dataset contrasts within each architecture (baseline_dataset within each architecture).
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
    Overall architecture contrasts (marginalized over datasets).
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
        rows.append(
            {"scope": "Overall (avg over datasets)", "contrast": f"{a} − {baseline_arch}", **stats}
        )
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
    Overall dataset contrasts (marginalized over architectures).
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
        rows.append(
            {"scope": "Overall (avg over arch)", "contrast": f"{d} − {baseline_dataset}", **stats}
        )
    return pd.DataFrame(rows)