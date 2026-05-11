# overall_ranking.py
from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from models_group import rank_probabilities_from_draws


# =============================================================================
# RANK + Z HELPERS
# =============================================================================
def compute_ranks_per_draw(score_matrix: np.ndarray) -> np.ndarray:
    """Compute ranks for each draw (row) across groups (columns).
    Args:
        score_matrix: (samples, groups) - higher is better
    Returns:
        ranks: (samples, groups) - rank 1 is best
    """
    n_samples, n_groups = score_matrix.shape
    order = np.argsort(-score_matrix, axis=1)
    ranks = np.empty_like(order)
    for s in range(n_samples):
        ranks[s, order[s]] = np.arange(1, n_groups + 1)
    return ranks


def robust_zscore_rows(
    matrix: np.ndarray,
    clip_range: Tuple[float, float] = (-8.0, 8.0),
) -> np.ndarray:
    """Robust z-scores per row using MAD."""
    center = np.median(matrix, axis=1, keepdims=True)
    mad = np.median(np.abs(matrix - center), axis=1, keepdims=True)
    scale = 1.4826 * mad + 1e-12
    z = (matrix - center) / scale
    if clip_range is not None:
        z = np.clip(z, clip_range[0], clip_range[1])
    return z


def standard_zscore_rows(
    matrix: np.ndarray,
    clip_range: Tuple[float, float] = (-8.0, 8.0),
) -> np.ndarray:
    """Standard z-scores per row."""
    mean_per_draw = matrix.mean(axis=1, keepdims=True)
    sd_per_draw = matrix.std(axis=1, keepdims=True)
    z = (matrix - mean_per_draw) / (sd_per_draw + 1e-12)
    if clip_range is not None:
        z = np.clip(z, clip_range[0], clip_range[1])
    return z


# =============================================================================
# OVERALL RANKING ACROSS METRICS
# =============================================================================
def overall_ranking_across_metrics(
    mu_draws_by_metric: Dict[str, np.ndarray],
    higher_is_better_by_metric: Dict[str, bool],
    group_names: Sequence[str],
    weights: Optional[Dict[str, float]] = None,
    mode: str = "rank",
):
    """
    Compute overall ranking across multiple metrics.

    Args:
        mu_draws_by_metric: metric -> (samples, groups) posterior draws on ORIGINAL scale
        higher_is_better_by_metric: metric -> bool direction
        group_names: list of group names matching columns
        weights: metric -> weight (normalized internally)
        mode: "rank", "robust_z", or "z"

    Returns:
        composite_draws: (samples, groups) combined utility scores
        overall_rank_table: DataFrame with ranking probabilities
        used_weights: normalized weights dict
    """
    metric_list = list(mu_draws_by_metric.keys())

    if weights is None:
        weights = {m: 1.0 for m in metric_list}
    wsum = float(sum(weights.values()))
    weights = {m: float(weights[m]) / wsum for m in metric_list}

    # Use common number of samples across metrics
    s_count = min(mu_draws_by_metric[m].shape[0] for m in metric_list)
    g_count = len(group_names)

    composite = np.zeros((s_count, g_count), dtype=float)

    if mode == "rank":
        # Rank-based utility (robust): best gets G, worst gets 1, then weighted sum
        for m in metric_list:
            mu = mu_draws_by_metric[m][:s_count, :]
            score = mu if higher_is_better_by_metric[m] else -mu
            ranks = compute_ranks_per_draw(score)  # 1=best
            utility = (g_count + 1) - ranks
            composite += weights[m] * utility

    elif mode == "robust_z":
        for m in metric_list:
            mu = mu_draws_by_metric[m][:s_count, :]
            score = mu if higher_is_better_by_metric[m] else -mu
            z = robust_zscore_rows(score, clip_range=(-8.0, 8.0))
            composite += weights[m] * z

    elif mode == "z":
        for m in metric_list:
            mu = mu_draws_by_metric[m][:s_count, :]
            score = mu if higher_is_better_by_metric[m] else -mu
            z = standard_zscore_rows(score, clip_range=(-8.0, 8.0))
            composite += weights[m] * z

    else:
        raise ValueError(f"Unknown scoring mode: {mode}. Use 'rank', 'robust_z', or 'z'.")

    overall_rank_table = rank_probabilities_from_draws(composite, group_names, higher_better=True)
    return composite, overall_rank_table, weights


# =============================================================================
# DIAGNOSTICS (TOP2 / CI / ENTROPY / N_eff / TOP-vs-RUNNERUP)
# =============================================================================
def add_overall_draw_diagnostics(
    overall_rank_table: pd.DataFrame,
    composite: np.ndarray,
    group_names: Sequence[str],
    tie_eps: float = 0.1,
    ci: Tuple[float, float] = (0.05, 0.95),
    topk: int = 2,
    rope_composite: float = 0.15,
) -> pd.DataFrame:
    """
    Adds:
      - Pr(top2)
      - rank_ci90 (q05..q95)
      - entropy_rank
      - N_eff_winners (global)
      - runner_up + Pr(win_vs_runnerup) + Pr(tie_vs_runnerup) + draw_index (for top row only)
      - Pr(Δ>ROPE), Pr(|Δ|<ROPE), Pr(Δ<-ROPE) vs best
    """
    df = overall_rank_table.copy()
    n_samples, n_groups = composite.shape

    ranks = compute_ranks_per_draw(composite)  # 1=best
    group_to_idx = {g: i for i, g in enumerate(group_names)}

    # Pr(topK)
    pr_topk = np.zeros(n_groups)
    for j in range(n_groups):
        pr_topk[j] = (ranks[:, j] <= topk).mean()
    df["Pr(top2)"] = [pr_topk[group_to_idx[g]] for g in df["group"]]

    # rank CI
    rank_ci_strs = []
    for g in df["group"]:
        j = group_to_idx[g]
        q_lo = np.quantile(ranks[:, j], ci[0])
        q_hi = np.quantile(ranks[:, j], ci[1])
        rank_ci_strs.append(f"[{q_lo:.1f}, {q_hi:.1f}]")
    df["rank_ci90"] = rank_ci_strs

    # entropy of rank distribution
    entropy = np.zeros(n_groups)
    for j in range(n_groups):
        rank_probs = np.array([(ranks[:, j] == k).mean() for k in range(1, n_groups + 1)])
        rank_probs_nz = rank_probs[rank_probs > 0]
        entropy[j] = -np.sum(rank_probs_nz * np.log(rank_probs_nz))
    df["entropy_rank"] = [entropy[group_to_idx[g]] for g in df["group"]]

    # N_eff_winners
    pr_best_vals = df["Pr(best)"].values
    pr_best_nz = pr_best_vals[pr_best_vals > 1e-12]
    n_eff = 1.0 / np.sum(pr_best_nz ** 2) if len(pr_best_nz) > 0 else np.nan
    df["N_eff_winners"] = n_eff

    # ROPE vs best
    best_group = df.iloc[0]["group"]
    best_idx = group_to_idx[best_group]
    
    pr_delta_ropep = np.zeros(n_groups)  # Pr(Δ > ROPE)
    pr_delta_rope_tie = np.zeros(n_groups)  # Pr(|Δ| < ROPE)
    pr_delta_ropem = np.zeros(n_groups)  # Pr(Δ < -ROPE)
    
    for j in range(n_groups):
        delta = composite[:, j] - composite[:, best_idx]
        pr_delta_ropep[j] = (delta > rope_composite).mean()
        pr_delta_rope_tie[j] = (np.abs(delta) < rope_composite).mean()
        pr_delta_ropem[j] = (delta < -rope_composite).mean()
    
    df["Pr(Δ>ROPE)"] = [pr_delta_ropep[group_to_idx[g]] for g in df["group"]]
    df["Pr(|Δ|<ROPE)"] = [pr_delta_rope_tie[group_to_idx[g]] for g in df["group"]]
    df["Pr(Δ<-ROPE)"] = [pr_delta_ropem[group_to_idx[g]] for g in df["group"]]

    # top vs runner-up diagnostics
    df["runner_up"] = ""
    df["Pr(win_vs_runnerup)"] = np.nan
    df["Pr(tie_vs_runnerup)"] = np.nan
    df["draw_index"] = np.nan

    if len(df) >= 2:
        top_group = df.iloc[0]["group"]
        runner_up_group = df.iloc[1]["group"]

        i1 = group_to_idx[top_group]
        i2 = group_to_idx[runner_up_group]

        delta = composite[:, i1] - composite[:, i2]
        pr_win = (delta > 0).mean()
        pr_tie = (np.abs(delta) < tie_eps).mean()
        draw_idx = 2.0 * np.abs(pr_win - 0.5)

        mask = df["group"].astype(str) == str(top_group)
        df.loc[mask, "runner_up"] = runner_up_group
        df.loc[mask, "Pr(win_vs_runnerup)"] = pr_win
        df.loc[mask, "Pr(tie_vs_runnerup)"] = pr_tie
        df.loc[mask, "draw_index"] = draw_idx

    return df
