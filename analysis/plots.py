# plots.py
from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

from config import ARCH_COLORS, HDI_PROB, METRIC_LABELS, RANDOM_SEED
from reporting import safe_slug
from transforms import higher_is_better


# =============================================================================
# CORE FIG SAVE
# =============================================================================
def save_fig(fig: plt.Figure, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    pdf_path = out_path.with_suffix(".pdf")
    if pdf_path != out_path:
        fig.savefig(pdf_path)
    plt.close(fig)


# =============================================================================
# LABEL HELPERS
# =============================================================================
def pretty_metric(metric: str) -> str:
    return METRIC_LABELS.get(metric, metric.replace("_", " ").title())


def metric_arrow(metric: str) -> str:
    return "↑" if higher_is_better(metric) else "↓"


def arch_from_group(group: str) -> str:
    g = str(group)
    if "Swin" in g:
        return "Swin"
    if "Terramind" in g:
        return "Terramind"
    return "U-Net"


def dataset_from_group(group: str) -> str:
    """Extract dataset name from group string like 'Swin U-Net | AE' -> 'AE'."""
    g = str(group)
    if "|" in g:
        return g.split("|", 1)[1].strip()
    return g


def short_group_label(group: str) -> str:
    g = str(group)
    if "|" in g:
        left, right = [x.strip() for x in g.split("|", 1)]
        arch = "Swin" if "Swin" in left else ("Terramind" if "Terramind" in left else "U-Net")
        return f"{arch}\n{right}"
    return g


def _mathsafe(s: str) -> str:
    s = str(s).replace("-", r"\!-\!")
    s = s.replace(" ", r"\ ")
    return s


def format_effect_label(param: str) -> str:
    p = str(param)
    if p.startswith("beta_dataset["):
        lvl = p[len("beta_dataset["):-1]
        return rf"Dataset main effect$_{{{_mathsafe(lvl)}}}$"
    if p.startswith("beta_arch["):
        lvl = p[len("beta_arch["):-1]
        return rf"Architecture main effect$_{{{_mathsafe(lvl)}}}$"
    if p.startswith("beta_interaction["):
        lvl = p[len("beta_interaction["):-1]
        lvl = lvl.replace("×", r"\times ")
        return rf"Interaction$_{{{lvl}}}$"
    return p


def add_arch_legend(ax) -> None:
    handles = [
        mpatches.Patch(facecolor=ARCH_COLORS[a], edgecolor="black", label=a)
        for a in ["U-Net", "Swin", "Terramind"]
        if a in ARCH_COLORS
    ]
    ax.legend(
        handles=handles,
        title="Architecture",
        frameon=False,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
    )


# =============================================================================
# CONTRAST PLOT
# =============================================================================
def plot_contrasts(
    contrast_df: pd.DataFrame,
    metric: str,
    out_path: Path,
) -> None:
    if contrast_df.empty:
        return

    df = contrast_df.copy()
    if "dataset" in df.columns:
        df = df.sort_values(["dataset", "mean_original"], ascending=[True, False]).reset_index(drop=True)

    labels = []
    if "dataset" in df.columns:
        labels = [f"{r.dataset}: {r.contrast}" for r in df.itertuples(index=False)]
    elif "arch" in df.columns:
        labels = [f"{r.arch}: {r.contrast}" for r in df.itertuples(index=False)]
    else:
        labels = [f"{r.contrast}" for r in df.itertuples(index=False)]

    y = np.arange(len(df))
    x = df["mean_original"].to_numpy()
    lo = df["hdi_lo_original"].to_numpy()
    hi = df["hdi_hi_original"].to_numpy()
    xerr_lo = x - lo
    xerr_hi = hi - x

    fig, ax = plt.subplots(figsize=(10, max(4.0, 0.38 * len(df))))
    ax.errorbar(x, y, xerr=[xerr_lo, xerr_hi], fmt="o", capsize=3)
    ax.axvline(0, linestyle="--", linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel(f"Δ({metric}) on original scale")
    ax.set_title(f"Contrasts: {metric}")
    ax.grid(True, axis="x", alpha=0.2)

    save_fig(fig, out_path)


# =============================================================================
# OBSERVED AND POSTERIOR GROUP PLOTS
# =============================================================================
def plot_observed_boxplot(
    data: pd.DataFrame,
    metric: str,
    out_path: Path,
    group_order: Optional[Sequence[str]] = None,
) -> None:
    df = data.copy()

    if group_order is None:
        group_order = list(df["group"].unique())

    groups = [g for g in group_order if g in set(df["group"].astype(str))]
    values = [df.loc[df["group"] == g, metric].dropna().to_numpy() for g in groups]

    fig, ax = plt.subplots(figsize=(12.5, 4.8))
    bp = ax.boxplot(
        values,
        patch_artist=True,
        widths=0.60,
        showfliers=False,
        medianprops=dict(color="black", linewidth=1.2),
        boxprops=dict(linewidth=1.0, color="black"),
        whiskerprops=dict(linewidth=1.0, color="black"),
        capprops=dict(linewidth=1.0, color="black"),
    )

    for patch, g in zip(bp["boxes"], groups):
        arch = arch_from_group(g)
        patch.set_facecolor(ARCH_COLORS.get(arch, "#CCCCCC"))
        patch.set_alpha(0.65)

    rng = np.random.default_rng(RANDOM_SEED)
    for i, g in enumerate(groups, start=1):
        y = df.loc[df["group"] == g, metric].dropna().to_numpy()
        x = rng.normal(i, 0.07, size=len(y))
        arch = arch_from_group(g)
        ax.scatter(
            x,
            y,
            s=22,
            alpha=0.75,
            linewidths=0.3,
            edgecolors="black",
            facecolors=ARCH_COLORS.get(arch, "#CCCCCC"),
        )

    ax.set_title(f"{pretty_metric(metric)} ({metric_arrow(metric)}) — observed runs")
    ax.set_ylabel(pretty_metric(metric))
    ax.set_xticks(np.arange(1, len(groups) + 1))
    ax.set_xticklabels([short_group_label(g) for g in groups], rotation=0)
    ax.grid(True, axis="y", alpha=0.18)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    add_arch_legend(ax)
    save_fig(fig, out_path)


def plot_posterior_group_boxplot(
    mu_draws: np.ndarray,
    groups: Sequence[str],
    metric: str,
    out_path: Path,
    group_order: Optional[Sequence[str]] = None,
    point_subsample: int = 250,
) -> None:
    if group_order is None:
        group_order = list(groups)
    ordered = [g for g in group_order if g in set(groups)]
    idx = [list(groups).index(g) for g in ordered]

    draws = mu_draws[:, idx]
    values = [draws[:, j] for j in range(draws.shape[1])]

    fig, ax = plt.subplots(figsize=(12.5, 4.8))
    bp = ax.boxplot(
        values,
        patch_artist=True,
        widths=0.60,
        showfliers=False,
        medianprops=dict(color="black", linewidth=1.2),
        boxprops=dict(linewidth=1.0, color="black"),
        whiskerprops=dict(linewidth=1.0, color="black"),
        capprops=dict(linewidth=1.0, color="black"),
    )

    for patch, g in zip(bp["boxes"], ordered):
        arch = arch_from_group(g)
        patch.set_facecolor(ARCH_COLORS.get(arch, "#CCCCCC"))
        patch.set_alpha(0.60)

    rng = np.random.default_rng(RANDOM_SEED)
    for i, g in enumerate(ordered, start=1):
        col = list(groups).index(g)
        y = mu_draws[:, col]
        if len(y) > point_subsample:
            y = rng.choice(y, size=point_subsample, replace=False)
        x = rng.normal(i, 0.07, size=len(y))
        arch = arch_from_group(g)
        ax.scatter(x, y, s=12, alpha=0.35, linewidths=0.0, facecolors=ARCH_COLORS.get(arch, "#CCCCCC"))

    ax.set_title(f"{pretty_metric(metric)} ({metric_arrow(metric)}) — posterior group means")
    ax.set_ylabel(pretty_metric(metric))
    ax.set_xticks(np.arange(1, len(ordered) + 1))
    ax.set_xticklabels([short_group_label(g) for g in ordered], rotation=0)
    ax.grid(True, axis="y", alpha=0.18)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    add_arch_legend(ax)
    save_fig(fig, out_path)


def plot_group_forest(
    perf_table: pd.DataFrame,
    metric: str,
    out_path: Path,
) -> None:
    df = perf_table.copy()

    hdi_cols = [c for c in df.columns if c.startswith("hdi_") and c.endswith("%")]
    if len(hdi_cols) >= 2:
        hdi_lo = sorted(hdi_cols, key=lambda c: float(c.replace("hdi_", "").replace("%", "")))[0]
        hdi_hi = sorted(hdi_cols, key=lambda c: float(c.replace("hdi_", "").replace("%", "")))[-1]
    else:
        hdi_lo = "hdi_3%"
        hdi_hi = "hdi_97%"

    df = df.sort_values("E[rank]").reset_index(drop=True)

    y = np.arange(len(df))
    x = df["mean"].to_numpy()
    xerr_lo = x - df[hdi_lo].to_numpy()
    xerr_hi = df[hdi_hi].to_numpy() - x

    fig, ax = plt.subplots(figsize=(10, max(3.8, 0.45 * len(df))))
    ax.errorbar(x, y, xerr=[xerr_lo, xerr_hi], fmt="o", capsize=3)
    ax.set_yticks(y)
    ax.set_yticklabels(df["group"].tolist())
    ax.invert_yaxis()
    ax.set_xlabel(f"Estimated mean ({metric}) - Original Scale")
    ax.set_title(f"Group means with {HDI_PROB:.0%} interval: {metric}")
    ax.grid(True, axis="x", alpha=0.2)

    save_fig(fig, out_path)


def plot_pr_best(
    perf_table: pd.DataFrame,
    metric: str,
    out_path: Path,
) -> None:
    df = perf_table.copy().sort_values("Pr(best)", ascending=False).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(10.5, max(3.8, 0.45 * len(df))))
    colors = [ARCH_COLORS.get(arch_from_group(g), "#CCCCCC") for g in df["group"].tolist()]
    ax.barh(df["group"], df["Pr(best)"], color=colors, edgecolor="black", linewidth=0.6)

    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("Posterior probability of being best")
    ax.set_title(f"{pretty_metric(metric)} — Pr(best)")
    ax.grid(True, axis="x", alpha=0.18)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    add_arch_legend(ax)
    save_fig(fig, out_path)


# =============================================================================
# FACTORIAL / HEATMAP / EFFECTS PLOTS
# =============================================================================
def plot_factorial_heatmap(
    pred_table: pd.DataFrame,
    metric: str,
    out_path: Path,
) -> None:
    pivot = pred_table.pivot(index="dataset", columns="arch", values="mean_original")

    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    im = ax.imshow(pivot.values, aspect="auto", cmap="viridis")
    ax.set_xticks(np.arange(pivot.shape[1]))
    ax.set_xticklabels(pivot.columns.tolist())
    ax.set_yticks(np.arange(pivot.shape[0]))
    ax.set_yticklabels(pivot.index.tolist())
    ax.set_title(f"Posterior mean by dataset×arch: {metric} (Original Scale)")

    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            ax.text(j, i, f"{pivot.values[i, j]:.3f}", ha="center", va="center", color="white", fontsize=9)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(metric)

    save_fig(fig, out_path)


def plot_factorial_effects(
    effects_report: pd.DataFrame,
    metric: str,
    out_path: Path,
) -> None:
    df = effects_report.copy()

    def _etype(p: str) -> str:
        p = str(p)
        if p.startswith("beta_dataset["):
            return "Dataset"
        if p.startswith("beta_arch["):
            return "Architecture"
        if p.startswith("beta_interaction["):
            return "Interaction"
        return "Other"

    df["etype"] = df["param"].apply(_etype)
    df["label"] = df["param"].apply(format_effect_label)

    etype_order = {"Dataset": 0, "Architecture": 1, "Interaction": 2, "Other": 3}
    df = df.sort_values(["etype"], key=lambda s: s.map(etype_order)).reset_index(drop=True)

    if df["scale"].iloc[0] == "log":
        df = df.dropna(subset=["Abs_Δ_mean"]).reset_index(drop=True)
        x = df["Abs_Δ_mean"].to_numpy()
        lo = df["Abs_Δ_lo"].to_numpy()
        hi = df["Abs_Δ_hi"].to_numpy()
        xlabel = f"Absolute change in {pretty_metric(metric)} (vs grand mean)"
    else:
        df = df.dropna(subset=["Δp_mean"]).reset_index(drop=True)
        x = df["Δp_mean"].to_numpy()
        lo = df["Δp_lo"].to_numpy()
        hi = df["Δp_hi"].to_numpy()
        xlabel = "Δ on original scale (vs grand mean)"

    y = np.arange(len(df))
    xerr_lo = x - lo
    xerr_hi = hi - x

    fig, ax = plt.subplots(figsize=(11.5, max(4.5, 0.34 * len(df))))
    ax.errorbar(x, y, xerr=[xerr_lo, xerr_hi], fmt="o", capsize=3)
    ax.axvline(0, linestyle="--", linewidth=1, alpha=0.55)

    ax.set_yticks(y)
    ax.set_yticklabels(df["label"].tolist())
    ax.invert_yaxis()
    ax.set_xlabel(xlabel)
    ax.set_title(f"{pretty_metric(metric)} — factorial model effect sizes")
    ax.grid(True, axis="x", alpha=0.18)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    save_fig(fig, out_path)


# =============================================================================
# OVERALL PLOTS
# =============================================================================
def plot_overall_pr_best(
    overall_rank_table: pd.DataFrame,
    out_path: Path,
) -> None:
    df = overall_rank_table.copy().sort_values("Pr(best)", ascending=False)
    fig, ax = plt.subplots(figsize=(10, max(3.8, 0.45 * len(df))))
    ax.barh(df["group"], df["Pr(best)"])
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("Pr(best) overall")
    ax.set_title("Overall multi-metric: Pr(best)")
    ax.grid(True, axis="x", alpha=0.2)
    save_fig(fig, out_path)


def plot_overall_draw_top2(
    overall_rank_table: pd.DataFrame,
    out_path: Path,
) -> None:
    """Plot draw/tie diagnostics for top-ranked group vs runner-up."""
    if len(overall_rank_table) < 2:
        return

    top_row = overall_rank_table.iloc[0]
    if pd.isna(top_row.get("Pr(win_vs_runnerup)")):
        return

    top_group = top_row["group"]
    runner_up = top_row["runner_up"]
    pr_win = top_row["Pr(win_vs_runnerup)"]
    pr_tie = top_row["Pr(tie_vs_runnerup)"]
    draw_idx = top_row["draw_index"]

    fig, ax = plt.subplots(figsize=(8, 3.5))

    metrics = ["Pr(win)", "Pr(tie)", "Draw index"]
    values = [pr_win, pr_tie, draw_idx]

    y_pos = np.arange(len(metrics))
    bars = ax.barh(y_pos, values, alpha=0.7, edgecolor="black", linewidth=1)

    for bar, val in zip(bars, values):
        ax.text(val + 0.02, bar.get_y() + bar.get_height() / 2, f"{val:.3f}", va="center", fontsize=10, fontweight="bold")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(metrics)
    ax.set_xlim(0, 1.15)
    ax.set_xlabel("Probability / Index")
    ax.set_title(f"Top-2 Draw Diagnostics\n{top_group} vs {runner_up}", fontsize=11, fontweight="bold")
    ax.grid(True, axis="x", alpha=0.2)
    ax.axvline(0.5, color="red", linestyle="--", linewidth=1, alpha=0.5, label="Threshold (0.5)")

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    ax.legend(loc="lower right", fontsize=9)
    save_fig(fig, out_path)


# =============================================================================
# PREDICTIVE CHECKS
# =============================================================================
def plot_posterior_predictive_check(
    post_pred,
    y_obs_transformed: np.ndarray,
    inv_transform: Callable,
    obs_groups: Sequence[str],
    group_order: Sequence[str],
    metric: str,
    out_path: Path,
) -> None:
    """Publication-style posterior predictive check on original scale."""
    y_pred_t = post_pred.posterior_predictive["y"].values
    y_pred_t = y_pred_t.reshape(-1, y_pred_t.shape[-1])

    y_obs_orig = np.asarray(inv_transform(np.asarray(y_obs_transformed, dtype=float)), dtype=float)
    y_pred_orig = np.asarray(inv_transform(y_pred_t), dtype=float)

    p_lo, p_mid, p_hi = np.percentile(y_pred_orig, [5, 50, 95], axis=0)

    ordered_groups = [g for g in group_order if g in set(obs_groups)]
    group_to_index = {g: i for i, g in enumerate(ordered_groups)}

    fig, (ax1, ax2) = plt.subplots(
        1,
        2,
        figsize=(180 / 25.4, 90 / 25.4),
        gridspec_kw={"width_ratios": [1.2, 1.0]},
    )

    bins = np.histogram_bin_edges(np.concatenate([y_obs_orig, y_pred_orig.flatten()]), bins=35)
    ax1.hist(y_obs_orig, bins=bins, density=True, alpha=0.55, color="#4C78A8", edgecolor="none", label="Observed")
    for i in np.linspace(0, y_pred_orig.shape[0] - 1, min(100, y_pred_orig.shape[0]), dtype=int):
        ax1.hist(y_pred_orig[i, :], bins=bins, density=True, alpha=0.02, color="#E45756", edgecolor="none")
    ax1.hist([], bins=bins, density=True, alpha=0.55, color="#E45756", label="Posterior predictive")
    ax1.set_title(f"(a) Distribution check", fontweight="bold")
    ax1.set_xlabel(pretty_metric(metric))
    ax1.set_ylabel("Density")
    ax1.grid(True, axis="y", alpha=0.2)
    ax1.legend(frameon=False)

    x = np.arange(len(y_obs_orig))
    ax2.fill_between(x, p_lo, p_hi, color="#F29E4C", alpha=0.30, label="Pred 90% interval")
    ax2.plot(x, p_mid, color="#D95F02", linewidth=1.2, label="Pred median")
    ax2.scatter(x, y_obs_orig, s=12, color="#2A9D8F", alpha=0.75, linewidths=0.2, edgecolors="black", label="Observed")

    if ordered_groups:
        group_centers = []
        group_labels = []
        for g in ordered_groups:
            idx = np.where(np.asarray(obs_groups) == g)[0]
            if idx.size == 0:
                continue
            group_centers.append(float(np.mean(idx)))
            group_labels.append(short_group_label(g).replace("\n", " "))
        ax2.set_xticks(group_centers)
        ax2.set_xticklabels(group_labels, rotation=35, ha="right")
    else:
        ax2.set_xticks([])

    ax2.set_title("(b) Observation-level fit", fontweight="bold")
    ax2.set_xlabel("Grouped observations")
    ax2.set_ylabel(pretty_metric(metric))
    ax2.grid(True, axis="y", alpha=0.2)
    ax2.legend(frameon=False, loc="upper right")

    for ax in (ax1, ax2):
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)

    fig.suptitle(f"Posterior Predictive Check: {pretty_metric(metric)}", y=1.02, fontsize=10, fontweight="bold")
    save_fig(fig, out_path)


def plot_prior_predictive_check(
    prior_mu_draws: np.ndarray,
    y_obs_original: np.ndarray,
    obs_groups: Sequence[str],
    groups: Sequence[str],
    out_path: Path,
    metric: str,
) -> None:
    """Publication-style prior predictive check using model-sampled prior group means."""
    ordered = [g for g in groups if g in set(obs_groups)]
    idx = [list(groups).index(g) for g in ordered]

    draws = prior_mu_draws[:, idx] if idx else prior_mu_draws
    x = np.arange(len(ordered))

    fig, (ax1, ax2) = plt.subplots(
        1,
        2,
        figsize=(180 / 25.4, 90 / 25.4),
        gridspec_kw={"width_ratios": [1.25, 1.0]},
    )

    bins = np.histogram_bin_edges(np.concatenate([y_obs_original, draws.flatten()]), bins=35)
    ax1.hist(draws.flatten(), bins=bins, density=True, alpha=0.60, color="#F29E4C", edgecolor="none", label="Prior predictive")
    ax1.hist(y_obs_original, bins=bins, density=True, alpha=0.50, color="#4C78A8", edgecolor="none", label="Observed")
    ax1.set_title("(a) Marginal distribution", fontweight="bold")
    ax1.set_xlabel(pretty_metric(metric))
    ax1.set_ylabel("Density")
    ax1.grid(True, axis="y", alpha=0.2)
    ax1.legend(frameon=False)

    q_lo, q_mid, q_hi = np.percentile(draws, [5, 50, 95], axis=0)
    obs_means = []
    for g in ordered:
        vals = y_obs_original[np.asarray(obs_groups) == g]
        obs_means.append(np.mean(vals) if len(vals) else np.nan)

    ax2.errorbar(
        x,
        q_mid,
        yerr=[q_mid - q_lo, q_hi - q_mid],
        fmt="o",
        color="#D95F02",
        ecolor="#D95F02",
        elinewidth=1.2,
        capsize=2.5,
        markersize=4.5,
        label="Prior median ±90%",
    )
    ax2.scatter(x, obs_means, s=22, color="#2A9D8F", edgecolors="black", linewidths=0.3, label="Observed group mean")
    ax2.set_xticks(x)
    ax2.set_xticklabels([short_group_label(g).replace("\n", " ") for g in ordered], rotation=35, ha="right")
    ax2.set_title("(b) Group-level plausibility", fontweight="bold")
    ax2.set_xlabel("Group")
    ax2.set_ylabel(pretty_metric(metric))
    ax2.grid(True, axis="y", alpha=0.2)
    ax2.legend(frameon=False)

    for ax in (ax1, ax2):
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)

    fig.suptitle(f"Prior Predictive Check: {pretty_metric(metric)}", y=1.02, fontsize=10, fontweight="bold")
    save_fig(fig, out_path)
