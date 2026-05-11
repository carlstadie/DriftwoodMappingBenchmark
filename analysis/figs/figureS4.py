# figureS4_S5_from_data.py
"""
Build Supplementary Figures S4 (Prior Predictive Check) and
S5 (Posterior Predictive Check) from scratch using data exported
by export_ppc_data.py.

Both figures share identical layout:
  - A4 portrait (190 × 257 mm)
  - 5 rows (one per metric) × 2 columns
  - Left column: marginal distribution check  (narrow)
  - Right column: group-level forest plot     (wide)

Run after main.py has populated the ppc_data/ directory:
    python figureS4_S5_from_data.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from scipy.stats import gaussian_kde

# ---------------------------------------------------------------------------
# PATHS  — adjust if your directory structure differs
# ---------------------------------------------------------------------------
THIS_DIR   = Path(__file__).resolve().parent
DATA_DIR   = THIS_DIR.parent / "stats_report_assets" / "ppc_data"
OUT_DIR    = THIS_DIR / "out"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# STYLE  — matches figure3.py / figure4.py
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family":        "sans-serif",
    "font.size":          7,
    "axes.titlesize":     7.5,
    "axes.labelsize":     7,
    "xtick.labelsize":    6,
    "ytick.labelsize":    6,
    "axes.linewidth":     0.7,
    "xtick.major.width":  0.7,
    "ytick.major.width":  0.7,
    "xtick.major.size":   3,
    "ytick.major.size":   3,
    "pdf.fonttype":       42,
    "ps.fonttype":        42,
    "mathtext.default":   "regular",
    "figure.facecolor":   "white",
    "axes.facecolor":     "white",
    "axes.spines.top":    False,
    "axes.spines.right":  False,
})

# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------
# Colour palette — print-safe, consistent with figure4.py
C_OBS         = "#2166AC"   # blue  — observed data
C_PRIOR       = "#E08214"   # amber — prior predictive
C_POST        = "#B2182B"   # red   — posterior predictive
C_PRIOR_FILL  = "#FEE0B6"   # light amber fill
C_POST_FILL   = "#FDDBC7"   # light red fill
C_ZERO        = "#AAAAAA"   # neutral reference line

# Architecture colours (matching config.py)
ARCH_COLORS = {
    "U-Net":     "#B58AE6",
    "Swin":      "#E9797A",
    "Terramind": "#F2D06B",
}

# Metric definitions
METRICS: List[Tuple[str, str, str]] = [
    ("iou",                         "IoU",                        "a"),
    ("dice-coef",                   "Dice coefficient",            "b"),
    ("normalized-surface-distance", "Normalized Surface Distance", "c"),
    ("mean-epistemic-uncertainty",  "Mean epistemic uncertainty",  "d"),
    ("mean-aleatoric-uncertainty",  "Mean aleatoric uncertainty",  "e"),
]

# Preferred group order (same as GROUP_ORDER in config.py)
GROUP_ORDER = [
    "U-Net | AE",      "Swin U-Net | AE",  "Terramind | AE",
    "U-Net | PS",      "Swin U-Net | PS",  "Terramind | PS",
    "U-Net | S2",      "Swin U-Net | S2",  "Terramind | S2",
]

# Short labels for y-axis ticks
SHORT_LABELS = {
    "U-Net | AE":      "U-Net | AE",
    "Swin U-Net | AE": "Swin | AE",
    "Terramind | AE":  "TM | AE",
    "U-Net | PS":      "U-Net | PS",
    "Swin U-Net | PS": "Swin | PS",
    "Terramind | PS":  "TM | PS",
    "U-Net | S2":      "U-Net | S2",
    "Swin U-Net | S2": "Swin | S2",
    "Terramind | S2":  "TM | S2",
}

# Sensor background bands (for grouping in forest plots)
SENSOR_BANDS = [
    (0,  3, "#F7F7F7"),   # AE rows
    (3,  6, "#FFFFFF"),   # PS rows
    (6,  9, "#F7F7F7"),   # S2 rows
]

HDI_PROB = 0.90   # interval width for predictive intervals


# ---------------------------------------------------------------------------
# DATA LOADING
# ---------------------------------------------------------------------------
def load_ppc(slug: str) -> Optional[Dict]:
    path = DATA_DIR / f"ppc_{slug}.npz"
    if not path.exists():
        print(f"  [WARN] missing: {path}")
        return None
    d = np.load(path, allow_pickle=True)
    return {
        "prior_mu_draws": d["prior_mu_draws"].astype(float),   # (S, G)
        "post_pred_y":    d["post_pred_y"].astype(float),       # (S, N)
        "y_obs":          d["y_obs"].astype(float),             # (N,)
        "obs_groups":     d["obs_groups"],                      # (N,) str
        "groups":         list(d["groups"]),                    # [G]
    }


# ---------------------------------------------------------------------------
# STATISTICS HELPERS
# ---------------------------------------------------------------------------
def hdi(x: np.ndarray, prob: float = HDI_PROB) -> Tuple[float, float]:
    """Highest density interval via sorted array method."""
    x = np.sort(np.asarray(x).ravel())
    n = len(x)
    if n == 0:
        return (np.nan, np.nan)
    width = int(np.floor(prob * n))
    if width >= n:
        return float(x[0]), float(x[-1])
    gaps = x[width:] - x[:n - width]
    best = int(np.argmin(gaps))
    return float(x[best]), float(x[best + width])


def group_prior_stats(prior_mu_draws: np.ndarray,
                      groups: List[str]) -> Dict[str, Dict]:
    """Per-group statistics from prior group-mean draws."""
    stats = {}
    for i, g in enumerate(groups):
        draws = prior_mu_draws[:, i]
        lo, hi = hdi(draws, HDI_PROB)
        stats[g] = {
            "median": float(np.median(draws)),
            "lo":     lo,
            "hi":     hi,
        }
    return stats


def group_obs_stats(y_obs: np.ndarray,
                    obs_groups: np.ndarray,
                    groups: List[str]) -> Dict[str, Dict]:
    """Per-group observed statistics."""
    stats = {}
    for g in groups:
        mask = obs_groups == g
        vals = y_obs[mask]
        stats[g] = {
            "mean":   float(np.mean(vals)) if len(vals) > 0 else np.nan,
            "vals":   vals,
        }
    return stats


def group_post_pred_stats(post_pred_y: np.ndarray,
                          obs_groups: np.ndarray,
                          groups: List[str]) -> Dict[str, Dict]:
    """
    Per-group posterior predictive statistics.

    For each group we take the mean across observations within the group
    for each posterior sample, giving a distribution of group-mean
    predictions. This is comparable to the 10 observed run-level values.
    """
    stats = {}
    n_samples = post_pred_y.shape[0]
    for g in groups:
        mask = obs_groups == g
        if not np.any(mask):
            stats[g] = {"median": np.nan, "lo": np.nan, "hi": np.nan}
            continue
        # Shape: (n_samples, n_obs_in_group) -> mean over obs -> (n_samples,)
        group_means = post_pred_y[:, mask].mean(axis=1)
        lo, hi = hdi(group_means, HDI_PROB)
        stats[g] = {
            "median": float(np.median(group_means)),
            "lo":     lo,
            "hi":     hi,
        }
    return stats


def safe_kde(values: np.ndarray, x_grid: np.ndarray) -> np.ndarray:
    """KDE with fallback for degenerate distributions."""
    values = np.asarray(values).ravel()
    values = values[np.isfinite(values)]
    if len(values) < 3 or np.std(values) < 1e-10:
        return np.zeros_like(x_grid, dtype=float)
    try:
        kde = gaussian_kde(values, bw_method="scott")
        return kde(x_grid)
    except Exception:
        return np.zeros_like(x_grid, dtype=float)


# ---------------------------------------------------------------------------
# AXIS RANGE HELPERS
# ---------------------------------------------------------------------------
def metric_xlim(slug: str, y_obs: Optional[np.ndarray] = None) -> Tuple[float, float]:
    """Sensible x-axis limits per metric type."""
    if slug in ("iou", "dice-coef", "normalized-surface-distance"):
        return (0.0, 1.0)
    # Uncertainty metrics: data-driven with padding
    if y_obs is not None:
        finite = y_obs[np.isfinite(y_obs)]
        if len(finite) > 0:
            return (0.0, float(np.percentile(finite, 99)) * 1.4)
    return (0.0, 0.02)


def arch_from_group(g: str) -> str:
    s = str(g)
    if "Swin" in s:
        return "Swin"
    if "Terra" in s:
        return "Terramind"
    return "U-Net"


# ---------------------------------------------------------------------------
# PANEL DRAWING — LEFT (marginal distribution)
# ---------------------------------------------------------------------------
def draw_left_prior(ax: plt.Axes,
                    prior_mu_draws: np.ndarray,
                    y_obs: np.ndarray,
                    xlim: Tuple[float, float],
                    is_top_row: bool,
                    show_legend: bool = False) -> None:
    """
    Overlay KDE of prior predictive group means (pooled) against
    histogram of observed run values.
    """
    ax.set_xlim(*xlim)

    # Observed histogram
    ax.hist(
        y_obs[np.isfinite(y_obs)],
        bins=15, range=xlim,
        density=True,
        color=C_OBS, alpha=0.55,
        linewidth=0,
        zorder=2,
    )

    # Prior predictive KDE (pooled across all groups and samples)
    pooled = prior_mu_draws.ravel()
    pooled = pooled[np.isfinite(pooled)]
    x_grid = np.linspace(*xlim, 300)
    kde_y = safe_kde(pooled, x_grid)
    if np.any(kde_y > 0):
        ax.fill_between(x_grid, kde_y, alpha=0.45,
                        color=C_PRIOR, linewidth=0, zorder=3)
        ax.plot(x_grid, kde_y, color=C_PRIOR, linewidth=0.8, zorder=4)

    ax.set_xlabel("Metric value", labelpad=2)
    ax.set_ylabel("Density", labelpad=2)
    ax.set_ylim(bottom=0)
    ax.tick_params(axis="both", length=2.5)

    if show_legend:
        from matplotlib.patches import Patch
        handles = [
            Patch(facecolor=C_OBS,   alpha=0.55, label="Observed"),
            Patch(facecolor=C_PRIOR, alpha=0.45, label="Prior predictive"),
        ]
        ax.legend(handles=handles, fontsize=5.5, frameon=False,
                  loc="upper right", handlelength=1.2)


def draw_left_posterior(ax: plt.Axes,
                         post_pred_y: np.ndarray,
                         y_obs: np.ndarray,
                         xlim: Tuple[float, float],
                         is_top_row: bool,
                         show_legend: bool = False) -> None:
    """
    Overlay KDE of posterior predictive (pooled) against
    histogram of observed run values.
    """
    ax.set_xlim(*xlim)

    # Observed histogram
    ax.hist(
        y_obs[np.isfinite(y_obs)],
        bins=15, range=xlim,
        density=True,
        color=C_OBS, alpha=0.55,
        linewidth=0,
        zorder=2,
    )

    # Posterior predictive KDE (pooled)
    pooled = post_pred_y.ravel()
    pooled = pooled[np.isfinite(pooled)]
    # Sub-sample if very large for speed
    if len(pooled) > 50_000:
        rng = np.random.default_rng(42)
        pooled = rng.choice(pooled, size=50_000, replace=False)

    x_grid = np.linspace(*xlim, 300)
    kde_y = safe_kde(pooled, x_grid)
    if np.any(kde_y > 0):
        ax.fill_between(x_grid, kde_y, alpha=0.4,
                        color=C_POST, linewidth=0, zorder=3)
        ax.plot(x_grid, kde_y, color=C_POST, linewidth=0.8, zorder=4)

    ax.set_xlabel("Metric value", labelpad=2)
    ax.set_ylabel("Density", labelpad=2)
    ax.set_ylim(bottom=0)
    ax.tick_params(axis="both", length=2.5)

    if show_legend:
        from matplotlib.patches import Patch
        handles = [
            Patch(facecolor=C_OBS,  alpha=0.55, label="Observed"),
            Patch(facecolor=C_POST, alpha=0.40, label="Posterior predictive"),
        ]
        ax.legend(handles=handles, fontsize=5.5, frameon=False,
                  loc="upper right", handlelength=1.2)


# ---------------------------------------------------------------------------
# PANEL DRAWING — RIGHT (group-level forest plot)
# ---------------------------------------------------------------------------
def _setup_forest_ax(ax: plt.Axes,
                     ordered_groups: List[str],
                     xlim: Tuple[float, float],
                     is_top_row: bool,
                     col_header: str) -> None:
    """Shared setup for right-panel forest plots."""
    n = len(ordered_groups)
    ax.set_xlim(*xlim)
    ax.set_ylim(-0.5, n - 0.5)
    ax.set_yticks(range(n))
    ax.set_yticklabels(
        [SHORT_LABELS.get(g, g) for g in ordered_groups],
        fontsize=5.8,
    )
    ax.invert_yaxis()
    ax.set_xlabel("Metric value", labelpad=2)
    ax.tick_params(axis="x", length=2.5)
    ax.tick_params(axis="y", length=0)   # no tick marks on y-axis

    # Sensor band shading
    for lo_i, hi_i, col in SENSOR_BANDS:
        ax.axhspan(lo_i - 0.5, hi_i - 0.5, color=col, zorder=0)

    # Light vertical grid
    ax.xaxis.grid(True, linestyle="--", linewidth=0.3, alpha=0.5, zorder=1)
    ax.set_axisbelow(True)

    if is_top_row:
        ax.set_title(col_header, fontsize=6.5, fontweight="semibold",
                     color="#444444", pad=4, loc="left")


def draw_right_prior(ax: plt.Axes,
                     ordered_groups: List[str],
                     prior_stats: Dict,
                     obs_stats: Dict,
                     xlim: Tuple[float, float],
                     is_top_row: bool,
                     show_legend: bool = False) -> None:
    """Forest plot: prior predictive intervals + observed group means."""
    _setup_forest_ax(ax, ordered_groups, xlim, is_top_row,
                     col_header=f"Group-level plausibility  "
                                f"(prior {int(HDI_PROB*100)}% HDI  ·  observed mean)")

    for i, g in enumerate(ordered_groups):
        ps = prior_stats.get(g, {})
        os_ = obs_stats.get(g, {})

        lo  = ps.get("lo",     np.nan)
        hi  = ps.get("hi",     np.nan)
        med = ps.get("median", np.nan)
        obs_mean = os_.get("mean", np.nan)

        # Prior interval
        if np.isfinite(lo) and np.isfinite(hi):
            ax.plot([lo, hi], [i, i],
                    color=C_PRIOR, linewidth=2.0, solid_capstyle="round",
                    alpha=0.7, zorder=3)
        # Prior median tick
        if np.isfinite(med):
            ax.plot(med, i, marker="|", markersize=5,
                    markeredgewidth=1.2, color=C_PRIOR,
                    zorder=4)
        # Observed group mean
        if np.isfinite(obs_mean):
            col = ARCH_COLORS.get(arch_from_group(g), "#888888")
            ax.scatter(obs_mean, i, s=18, color=col,
                       edgecolors="black", linewidths=0.4,
                       zorder=5)

    if show_legend:
        from matplotlib.lines import Line2D
        from matplotlib.patches import Patch
        handles = [
            Line2D([0], [0], color=C_PRIOR, linewidth=2, alpha=0.7,
                   label=f"Prior {int(HDI_PROB*100)}% HDI"),
            Line2D([0], [0], color=C_PRIOR, marker="|", markersize=5,
                   markeredgewidth=1.2, linestyle="none",
                   label="Prior median"),
        ]
        for arch, col in ARCH_COLORS.items():
            handles.append(
                Line2D([0], [0], marker="o", markersize=4,
                       markerfacecolor=col, markeredgecolor="black",
                       markeredgewidth=0.4, linestyle="none",
                       label=f"Obs. mean ({arch})")
            )
        ax.legend(handles=handles, fontsize=5.2, frameon=False,
                  loc="upper right", handlelength=1.5, ncol=2)


def draw_right_posterior(ax: plt.Axes,
                          ordered_groups: List[str],
                          post_stats: Dict,
                          obs_stats: Dict,
                          xlim: Tuple[float, float],
                          is_top_row: bool,
                          show_legend: bool = False) -> None:
    """
    Forest plot: posterior predictive interval + observed individual runs.

    Each group shows:
      - Shaded band  = posterior predictive HDI for group mean
      - Centre line  = posterior predictive median
      - Dots         = 10 individual observed run values (jittered vertically)
    """
    _setup_forest_ax(ax, ordered_groups, xlim, is_top_row,
                     col_header=f"Observation-level fit  "
                                f"(posterior {int(HDI_PROB*100)}% predictive interval  ·  observed runs)")

    rng = np.random.default_rng(0)

    for i, g in enumerate(ordered_groups):
        ps  = post_stats.get(g, {})
        os_ = obs_stats.get(g, {})

        lo  = ps.get("lo",     np.nan)
        hi  = ps.get("hi",     np.nan)
        med = ps.get("median", np.nan)
        run_vals = os_.get("vals", np.array([]))

        # Posterior predictive interval as a thick bar
        if np.isfinite(lo) and np.isfinite(hi):
            ax.barh(i, hi - lo, left=lo, height=0.32,
                    color=C_POST_FILL, linewidth=0, zorder=2, alpha=0.8)
            ax.plot([lo, hi], [i, i],
                    color=C_POST, linewidth=0.6,
                    solid_capstyle="round", zorder=3)
        # Posterior predictive median
        if np.isfinite(med):
            ax.plot(med, i, marker="|", markersize=6,
                    markeredgewidth=1.4, color=C_POST, zorder=4)

        # Observed run values (jittered vertically within ±0.15)
        if len(run_vals) > 0:
            jitter = rng.uniform(-0.14, 0.14, size=len(run_vals))
            col = ARCH_COLORS.get(arch_from_group(g), "#888888")
            ax.scatter(run_vals, i + jitter,
                       s=9, color=col,
                       edgecolors="black", linewidths=0.3,
                       zorder=5, alpha=0.85)

    if show_legend:
        from matplotlib.lines import Line2D
        from matplotlib.patches import Patch
        handles = [
            Patch(facecolor=C_POST_FILL, edgecolor=C_POST,
                  linewidth=0.6, label=f"Post. pred. {int(HDI_PROB*100)}% HDI"),
            Line2D([0], [0], color=C_POST, marker="|", markersize=6,
                   markeredgewidth=1.4, linestyle="none",
                   label="Post. pred. median"),
        ]
        for arch, col in ARCH_COLORS.items():
            handles.append(
                Line2D([0], [0], marker="o", markersize=3.5,
                       markerfacecolor=col, markeredgecolor="black",
                       markeredgewidth=0.3, linestyle="none",
                       label=f"Obs. runs ({arch})")
            )
        ax.legend(handles=handles, fontsize=5.2, frameon=False,
                  loc="upper right", handlelength=1.5, ncol=2)


# ---------------------------------------------------------------------------
# FIGURE BUILDER
# ---------------------------------------------------------------------------
def _make_figure(check_type: str) -> plt.Figure:
    """
    Build one predictive check figure.

    Args:
        check_type: "prior" or "posterior"
    """
    assert check_type in ("prior", "posterior")
    is_prior = (check_type == "prior")

    # Load all metric data upfront so we can set consistent x-limits
    data_by_slug: Dict[str, Optional[Dict]] = {}
    for slug, _, _ in METRICS:
        data_by_slug[slug] = load_ppc(slug)

    # ------------------------------------------------------------------ #
    # Figure layout                                                         #
    # ------------------------------------------------------------------ #
    fig_w_mm = 190
    fig_h_mm = 257
    fig = plt.figure(figsize=(fig_w_mm / 25.4, fig_h_mm / 25.4), dpi=300)

    n_rows = len(METRICS)
    gs = gridspec.GridSpec(
        n_rows, 2,
        figure=fig,
        width_ratios=[1, 2.6],
        hspace=0.55,       # vertical spacing between metric rows
        wspace=0.30,       # horizontal gap between left and right columns
        left=0.14,         # room for y-axis labels on right panel
        right=0.99,
        top=0.960,
        bottom=0.045,
    )



    # ------------------------------------------------------------------ #
    # Draw rows                                                             #
    # ------------------------------------------------------------------ #
    for row_idx, (slug, label, letter) in enumerate(METRICS):
        d = data_by_slug[slug]
        is_top = (row_idx == 0)
        is_bottom = (row_idx == len(METRICS) - 1)

        ax_left  = fig.add_subplot(gs[row_idx, 0])
        ax_right = fig.add_subplot(gs[row_idx, 1])

        # Panel letter
        ax_left.text(
            -0.30, 1.06, f"({letter})",
            transform=ax_left.transAxes,
            ha="right", va="bottom",
            fontsize=8.5, fontweight="bold", color="#1A1A1A",
        )
        # Metric label — rotated, to the left of the left panel
        ax_left.text(
            -0.22, 0.5, label,
            transform=ax_left.transAxes,
            ha="right", va="center",
            fontsize=6.0, fontweight="semibold", color="#333333",
            rotation=90,
        )

        if d is None:
            for ax in (ax_left, ax_right):
                ax.set_axis_off()
                ax.text(0.5, 0.5, "Data not found",
                        ha="center", va="center", fontsize=7,
                        color="#7A1F1F", transform=ax.transAxes)
            continue

        # Sort groups according to GROUP_ORDER (only those present in data)
        present = set(d["groups"])
        ordered_groups = [g for g in GROUP_ORDER if g in present]
        # Append any extra groups not in GROUP_ORDER
        for g in d["groups"]:
            if g not in ordered_groups:
                ordered_groups.append(g)

        xlim = metric_xlim(slug, d["y_obs"])

        if is_prior:
            prior_stats = group_prior_stats(d["prior_mu_draws"], d["groups"])
            obs_stats   = group_obs_stats(d["y_obs"], d["obs_groups"], ordered_groups)

            draw_left_prior(ax_left, d["prior_mu_draws"], d["y_obs"],
                            xlim, is_top_row=is_top, show_legend=is_bottom)
            draw_right_prior(ax_right, ordered_groups,
                             prior_stats, obs_stats,
                             xlim, is_top_row=is_top, show_legend=is_bottom)
        else:
            post_stats = group_post_pred_stats(d["post_pred_y"],
                                               d["obs_groups"], ordered_groups)
            obs_stats  = group_obs_stats(d["y_obs"], d["obs_groups"], ordered_groups)

            draw_left_posterior(ax_left, d["post_pred_y"], d["y_obs"],
                                xlim, is_top_row=is_top, show_legend=is_bottom)
            draw_right_posterior(ax_right, ordered_groups,
                                 post_stats, obs_stats,
                                 xlim, is_top_row=is_top, show_legend=is_bottom)

    return fig


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main() -> None:

    print("Building Figure S4 (Prior Predictive Check) ...")
    fig_s4 = _make_figure("prior")
    for ext in ("png", "pdf"):
        out = OUT_DIR / f"figureS4_prior_predictive_checks.{ext}"
        fig_s4.savefig(out, dpi=500 if ext == "png" else 300,
                       bbox_inches="tight")
        print(f"  Saved: {out}")
    plt.close(fig_s4)

    print("Building Figure S5 (Posterior Predictive Check) ...")
    fig_s5 = _make_figure("posterior")
    for ext in ("png", "pdf"):
        out = OUT_DIR / f"figureS5_posterior_predictive_checks.{ext}"
        fig_s5.savefig(out, dpi=500 if ext == "png" else 300,
                       bbox_inches="tight")
        print(f"  Saved: {out}")
    plt.close(fig_s5)

    print("Done.")


if __name__ == "__main__":
    main()