

import os
import numpy as np
import pandas as pd
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator, ScalarFormatter
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from html_parser import (
    extract_factorial_effects,
    extract_arch_contrasts,
    extract_overall_arch_contrasts,
    extract_dataset_contrasts,
    extract_overall_dataset_contrasts,
)

try:
    from config import ROPE_BY_METRIC_TRANSFORMED
except Exception:
    ROPE_BY_METRIC_TRANSFORMED = {
        "IoU": 0.05,
        "dice_coef": 0.05,
        "normalized_surface_distance": 0.05,
        "mean_epistemic_uncertainty": float(np.log(1.05)),
        "mean_aleatoric_uncertainty": float(np.log(1.05)),
    }

# ─────────────────────────────────────────────────────────────────────────────
# Plot rcParams — match Figure 4 reference
# ─────────────────────────────────────────────────────────────────────────────
mpl.rcParams.update({
    "font.size": 7,
    "axes.titlesize": 8,
    "axes.labelsize": 7,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6.5,
    "axes.linewidth": 0.7,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "mathtext.default": "regular",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})

# ─────────────────────────────────────────────────────────────────────────────
# Classification / colors — Figure 4 palette
# ─────────────────────────────────────────────────────────────────────────────
LOWER_IS_BETTER_METRICS = {"Epistemic", "Aleatoric"}
PRACTICAL_PROB_THRESHOLD = 0.95

COLOR_IMPROVE      = "#2166AC"   # dark blue
COLOR_DEGRADE      = "#E08214"   # muted orange
COLOR_NEGLIGIBLE   = "#BABABA"   # light grey
COLOR_INCONCLUSIVE = "#878787"   # mid grey
COLOR_ROPE_SHADE   = "#F5F5F5"   # very light grey
COLOR_ZERO_LINE    = "#333333"

PLOT_TO_CONFIG_METRIC = {
    "IoU": "IoU",
    "Dice": "dice_coef",
    "NSD": "normalized_surface_distance",
    "Epistemic": "mean_epistemic_uncertainty",
    "Aleatoric": "mean_aleatoric_uncertainty",
}

# ─────────────────────────────────────────────────────────────────────────────
# Metric ordering / labels
# ─────────────────────────────────────────────────────────────────────────────
METRICS_PREFERRED = ["IoU", "Dice", "NSD", "Epistemic", "Aleatoric"]
CENTER_METRIC_NAME = "NSD"  # put x-axis label under the center column

DATASET_ORDER = ["AE", "PS", "S2"]
ARCH_ORDER = ["U-Net", "Swin", "TM"]
ARCH_CONTRAST_ORDER = ["Swin", "TM"]                 # contrast_clean
DATASET_CONTRAST_ORDER = ["PS - AE", "S2 - AE"]      # contrast_clean

def metric_title(metric: str) -> str:
    if metric == "Epistemic":
        return r"Uncertainty$_\mathrm{epist.}$ (↓)"
    if metric == "Aleatoric":
        return r"Uncertainty$_\mathrm{alea.}$ (↓)"
    arrow = "(↓)" if metric in LOWER_IS_BETTER_METRICS else "(↑)"
    return f"{metric} {arrow}"

# ─────────────────────────────────────────────────────────────────────────────
# Small helpers (copied spirit from Fig 4)
# ─────────────────────────────────────────────────────────────────────────────
def normalize_label(label: str) -> str:
    s = str(label).strip()
    s = s.replace("Terramind", "TM")
    s = s.replace(" x ", "×").replace(" X ", "×").replace(" × ", "×")
    return s

def _abbr(s: str, n: int = 12) -> str:
    s = str(s).strip()
    return s if len(s) <= n else s[:max(1, n - 1)] + "…"

def compact_effect_label(label, max_len=18):
    label = normalize_label(label)
    return _abbr(label, max_len)

def pd_safe_values(series):
    if series is None:
        return np.array([], dtype=float)
    try:
        return np.asarray(series.values, dtype=float)
    except Exception:
        return np.array([], dtype=float)

# ─────────────────────────────────────────────────────────────────────────────
# ROPE helpers (same as Fig 4 logic; uses factorial baselines to back-transform)
# ─────────────────────────────────────────────────────────────────────────────
def _sigmoid(x): return 1.0 / (1.0 + np.exp(-x))

def _logit(p, eps=1e-9):
    p = float(np.clip(p, eps, 1 - eps))
    return np.log(p / (1 - p))

def infer_display_rope_for_metric(metric_plot_name, df_metric):
    cfg_key = PLOT_TO_CONFIG_METRIC.get(metric_plot_name, metric_plot_name)
    rope_t = float(ROPE_BY_METRIC_TRANSFORMED.get(cfg_key, 0.05))

    if df_metric is None or "baseline" not in getattr(df_metric, "columns", []):
        return rope_t

    baseline_vals = pd_safe_values(df_metric.get("baseline"))
    baseline_vals = baseline_vals[np.isfinite(baseline_vals)]
    if len(baseline_vals) == 0:
        return rope_t

    b0 = float(np.nanmedian(baseline_vals))

    if metric_plot_name in {"IoU", "Dice", "NSD"}:
        try:
            z0 = _logit(b0)
            p_plus = _sigmoid(z0 + rope_t)
            p_minus = _sigmoid(z0 - rope_t)
            return float(max(0.5 * (abs(p_plus - b0) + abs(b0 - p_minus)), 1e-6))
        except Exception:
            return rope_t

    if metric_plot_name in {"Epistemic", "Aleatoric"}:
        b0 = max(b0, 1e-12)
        return float(max(b0 * (np.exp(rope_t) - 1.0), 1e-12))

    return rope_t

# ─────────────────────────────────────────────────────────────────────────────
# Axis helpers (same look as Fig 4)
# ─────────────────────────────────────────────────────────────────────────────
def style_xaxis_ticks(ax, x_abs_max, metric_name=None):
    ax.xaxis.set_major_locator(MaxNLocator(nbins=3, symmetric=True))
    if x_abs_max is None or not np.isfinite(x_abs_max):
        return

    if metric_name in {"Epistemic", "Aleatoric"}:
        fmt = ScalarFormatter(useMathText=False)
        fmt.set_scientific(False)
        fmt.set_useOffset(False)
        ax.xaxis.set_major_formatter(fmt)
    elif x_abs_max < 1e-2 or x_abs_max > 1e2:
        fmt = ScalarFormatter(useMathText=True)
        fmt.set_scientific(True)
        fmt.set_powerlimits((-2, 2))
        fmt.set_useOffset(False)
        ax.xaxis.set_major_formatter(fmt)
        ax.xaxis.get_offset_text().set_fontsize(5.5)
    else:
        fmt = ScalarFormatter(useMathText=False)
        fmt.set_useOffset(False)
        ax.xaxis.set_major_formatter(fmt)

def compute_column_xlim(subsets, hdi_lo_col, hdi_hi_col, pad_frac=0.18, min_pad=1e-8, rope_val=None):
    vals = [0.0]
    for sdf in subsets:
        if sdf is None or sdf.empty:
            continue
        lo = pd_safe_values(sdf.get(hdi_lo_col))
        hi = pd_safe_values(sdf.get(hdi_hi_col))
        vals.extend(lo[np.isfinite(lo)].tolist())
        vals.extend(hi[np.isfinite(hi)].tolist())

    if rope_val is not None and np.isfinite(rope_val):
        vals.extend([float(-abs(rope_val)), float(abs(rope_val))])

    if len(vals) == 0:
        return (-1.0, 1.0)

    x_abs_max = np.max(np.abs(vals))
    if not np.isfinite(x_abs_max) or x_abs_max == 0:
        x_abs_max = 1.0

    pad = max(min_pad, x_abs_max * pad_frac)
    return (-x_abs_max - pad, x_abs_max + pad)

# ─────────────────────────────────────────────────────────────────────────────
# Classification (reuse the Fig 4 rules; works with contrasts too)
# ─────────────────────────────────────────────────────────────────────────────
def classify_effect_posterior_rope(row, metric_name):
    lower_is_better = metric_name in LOWER_IS_BETTER_METRICS
    pr_rope    = float(row.get("Pr(|Δ|<ROPE)", np.nan))
    pr_gt_rope = float(row.get("Pr(Δ>ROPE)", np.nan))
    pr_lt_rope = float(row.get("Pr(Δ<-ROPE)", np.nan))

    if not (np.isfinite(pr_rope) and np.isfinite(pr_gt_rope) and np.isfinite(pr_lt_rope)):
        return "inconclusive"

    pr_improve = pr_lt_rope if lower_is_better else pr_gt_rope
    pr_degrade = pr_gt_rope if lower_is_better else pr_lt_rope

    if pr_improve > PRACTICAL_PROB_THRESHOLD:
        return "improve"
    if pr_degrade > PRACTICAL_PROB_THRESHOLD:
        return "degrade"
    if pr_rope > PRACTICAL_PROB_THRESHOLD:
        return "negligible"
    return "inconclusive"

def style_from_class(effect_class):
    if effect_class == "improve":
        return {"line_color": COLOR_IMPROVE, "marker_face": COLOR_IMPROVE,
                "marker_edge": "white", "marker_lw": 0.7}
    if effect_class == "degrade":
        return {"line_color": COLOR_DEGRADE, "marker_face": COLOR_DEGRADE,
                "marker_edge": "white", "marker_lw": 0.7}
    if effect_class == "negligible":
        return {"line_color": COLOR_NEGLIGIBLE, "marker_face": COLOR_NEGLIGIBLE,
                "marker_edge": "white", "marker_lw": 0.7}
    return {"line_color": COLOR_INCONCLUSIVE, "marker_face": "white",
            "marker_edge": COLOR_INCONCLUSIVE, "marker_lw": 1.0}

# ─────────────────────────────────────────────────────────────────────────────
# Legend (same as Fig 4)
# ─────────────────────────────────────────────────────────────────────────────
def add_figure_legend(fig, bottom_y=0.0):
    handles = [
        Line2D([0], [0], color=COLOR_IMPROVE, lw=1.6, marker="o", markersize=4,
               markerfacecolor=COLOR_IMPROVE, markeredgecolor="white", markeredgewidth=0.7,
               label=f"Improvement ($P > {PRACTICAL_PROB_THRESHOLD:.2f}$)"),
        Line2D([0], [0], color=COLOR_DEGRADE, lw=1.6, marker="o", markersize=4,
               markerfacecolor=COLOR_DEGRADE, markeredgecolor="white", markeredgewidth=0.7,
               label=f"Degradation ($P > {PRACTICAL_PROB_THRESHOLD:.2f}$)"),
        Line2D([0], [0], color=COLOR_INCONCLUSIVE, lw=1.6, marker="o", markersize=4,
               markerfacecolor="white", markeredgecolor=COLOR_INCONCLUSIVE, markeredgewidth=1.0,
               label="Inconclusive"),
        Patch(facecolor=COLOR_ROPE_SHADE, edgecolor="#BBBBBB", linewidth=0.5, label="ROPE"),
        Line2D([0], [0], color=COLOR_ZERO_LINE, lw=0.8, ls="--", label="No effect"),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=5,
        frameon=False,
        fontsize=6.5,
        handlelength=2.0,
        columnspacing=1.0,
        handletextpad=0.5,
        bbox_to_anchor=(0.55, bottom_y),
    )

# ─────────────────────────────────────────────────────────────────────────────
# Panel plotting (Figure 4-like)
# ─────────────────────────────────────────────────────────────────────────────
def plot_forest_panel(ax, df, effect_col, hdi_lo_col, hdi_hi_col, rope_val,
                      title="", is_bottom_row=False, is_leftmost_col=False, is_center_col=False,
                      xlim=None, metric_name=None):
    if df is None or df.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center",
                transform=ax.transAxes, fontsize=7, color="#888888")
        ax.set_xlim(-0.5, 0.5)
        ax.set_xticks([]); ax.set_yticks([])
        for side in ["top", "right", "left", "bottom"]:
            ax.spines[side].set_visible(True)
            ax.spines[side].set_linewidth(0.7)
        return

    d = df.copy().reset_index(drop=True)
    n = len(d)
    y = np.arange(n)

    means  = np.asarray(d[effect_col].values, dtype=float)
    hdi_lo = np.asarray(d[hdi_lo_col].values, dtype=float)
    hdi_hi = np.asarray(d[hdi_hi_col].values, dtype=float)

    # Spines
    for side in ["top", "right", "left", "bottom"]:
        ax.spines[side].set_visible(True)
        ax.spines[side].set_linewidth(0.7)
        ax.spines[side].set_color("#333333")

    # ROPE + zero line
    ax.axvspan(-rope_val, rope_val, color=COLOR_ROPE_SHADE, alpha=1.0, zorder=0, lw=0)
    ax.axvline(0, color=COLOR_ZERO_LINE, linestyle="--", linewidth=0.7, alpha=0.6, zorder=1)

    # Subtle horizontal guides
    for yp in y:
        ax.axhline(yp, color="#E8E8E8", linewidth=0.35, zorder=0)

    # Plot each row with classification style
    for i in range(n):
        cls = classify_effect_posterior_rope(d.iloc[i], metric_name)
        st = style_from_class(cls)

        # HDI line (slightly extended for bounded metrics to peek through scatter marker when small)
        hdi_extension = 0.02 if metric_name in {"IoU", "Dice", "NSD"} else 0.0
        ax.plot([hdi_lo[i] - hdi_extension, hdi_hi[i] + hdi_extension], [y[i], y[i]],
                color=st["line_color"], linewidth=1.6,
                solid_capstyle="round", zorder=2)
        ax.scatter(means[i], y[i], s=16,
                   facecolors=st["marker_face"],
                   edgecolors=st["marker_edge"],
                   linewidths=st["marker_lw"],
                   zorder=3)

    # Y labels
    labels = [compact_effect_label(x, max_len=18) for x in d["param_clean"].astype(str).tolist()]
    
    # Adaptive font size based on number of items
    if n <= 4:
        y_fontsize = 7.0
    elif n <= 7:
        y_fontsize = 6.5
    else:
        y_fontsize = 6.0
    
    ax.set_yticks(y)
    if is_leftmost_col:
        ax.set_yticklabels(labels, fontsize=y_fontsize)
    else:
        ax.set_yticklabels([])
    for t in ax.get_yticklabels():
        t.set_ha("right")
        t.set_multialignment("right")
    ax.tick_params(axis="y", length=0, pad=2)

    # X axis
    ax.tick_params(axis="x", labelsize=6, length=3, width=0.7, pad=1)

    if xlim is None:
        all_vals = np.concatenate([hdi_lo, hdi_hi, [0.0]])
        all_vals = all_vals[np.isfinite(all_vals)]
        if len(all_vals) > 0:
            x_abs_max = max(abs(all_vals.min()), abs(all_vals.max()))
            margin = (x_abs_max * 0.18) if x_abs_max > 0 else max(abs(rope_val), 1e-6)
            xlim = (-x_abs_max - margin, x_abs_max + margin)
        else:
            xlim = (-1.0, 1.0)

    ax.set_xlim(*xlim)
    style_xaxis_ticks(ax, max(abs(xlim[0]), abs(xlim[1])), metric_name=metric_name)

    if not is_bottom_row:
        ax.set_xticklabels([])

    ax.set_ylim(-0.65, n - 1 + 0.65)
    ax.grid(axis="x", alpha=0.18, linewidth=0.4, zorder=0)
    ax.set_axisbelow(True)

    if title:
        ax.set_title(title, fontsize=8, fontweight="bold", pad=5, color="#222222")

    # Center-only x label (like Fig 4)
    ax.set_xlabel(
        "Factor effect on Δ against baseline (U-Net/AE)" if (is_bottom_row and is_center_col) else "",
        fontsize=7, labelpad=3
    )

# ─────────────────────────────────────────────────────────────────────────────
# Build the per-metric panel dataframes
# ─────────────────────────────────────────────────────────────────────────────
def build_within_arch_df(df):
    # df columns include: dataset, contrast_clean, Δ_mean, Δ_lo, Δ_hi, probs...
    out = df.copy()
    out["dataset"] = out["dataset"].astype(str).map(normalize_label)
    out["contrast_clean"] = out["contrast_clean"].astype(str).map(normalize_label)
    out["param_clean"] = out.apply(lambda r: f"{r['dataset']}: U-Net→{r['contrast_clean']}", axis=1)

    # Sort by dataset order then contrast order
    ds_key = {k: i for i, k in enumerate(DATASET_ORDER)}
    ct_key = {k: i for i, k in enumerate(ARCH_CONTRAST_ORDER)}
    out["_ds"] = out["dataset"].map(ds_key).fillna(999)
    out["_ct"] = out["contrast_clean"].map(ct_key).fillna(999)
    out = out.sort_values(["_ds", "_ct", "param_clean"]).drop(columns=["_ds", "_ct"])
    return out

def build_within_dataset_df(df):
    # df columns include: arch, contrast_clean, Δ_mean, Δ_lo, Δ_hi, probs...
    out = df.copy()
    out["arch"] = out["arch"].astype(str).map(normalize_label)
    out["contrast_clean"] = out["contrast_clean"].astype(str).map(normalize_label)
    # Convert "X - Y" to "Y → X" for arrow notation
    out["contrast_arrow"] = out["contrast_clean"].str.replace(r"^([A-Z0-9]+) - ([A-Z0-9]+)$", r"\2→\1", regex=True)
    out["param_clean"] = out.apply(lambda r: f"{r['arch']}: {r['contrast_arrow']}", axis=1)

    a_key = {k: i for i, k in enumerate(ARCH_ORDER)}
    c_key = {k: i for i, k in enumerate(DATASET_CONTRAST_ORDER)}
    out["_a"] = out["arch"].map(a_key).fillna(999)
    out["_c"] = out["contrast_clean"].map(c_key).fillna(999)
    out = out.sort_values(["_a", "_c", "param_clean"]).drop(columns=["_a", "_c"])
    return out

def build_marginal_arch_df(df):
    out = df.copy()
    out["contrast_clean"] = out["contrast_clean"].astype(str).map(normalize_label)
    out["param_clean"] = "U-Net→" + out["contrast_clean"]

    ct_key = {k: i for i, k in enumerate(ARCH_CONTRAST_ORDER)}
    out["_ct"] = out["contrast_clean"].map(ct_key).fillna(999)
    out = out.sort_values(["_ct", "param_clean"]).drop(columns=["_ct"])
    return out

def build_marginal_dataset_df(df):
    out = df.copy()
    out["contrast_clean"] = out["contrast_clean"].astype(str).map(normalize_label)
    # Convert "X - Y" to "Y → X" for arrow notation
    out["param_clean"] = out["contrast_clean"].str.replace(r"^([A-Z0-9]+) - ([A-Z0-9]+)$", r"\2→\1", regex=True)

    c_key = {k: i for i, k in enumerate(DATASET_CONTRAST_ORDER)}
    out["_c"] = out["contrast_clean"].map(c_key).fillna(999)
    out = out.sort_values(["_c", "param_clean"]).drop(columns=["_c"])
    return out

# ─────────────────────────────────────────────────────────────────────────────
# Figure creators (GridSpec like Fig 4)
# ─────────────────────────────────────────────────────────────────────────────
def create_contrast_grid_within(arch_w, ds_w, factorial, output_path):
    metrics = [m for m in METRICS_PREFERRED if (m in arch_w and m in ds_w)]
    if not metrics:
        raise ValueError("No expected metrics found in contrast data.")

    # Prepare per-metric dfs
    arch_df = {}
    ds_df = {}
    rope = {}
    for m in metrics:
        arch_df[m] = build_within_arch_df(arch_w[m])
        ds_df[m] = build_within_dataset_df(ds_w[m])
        rope[m] = infer_display_rope_for_metric(m, factorial.get(m))

    # Figure geometry (Fig 4 style)
    N = len(metrics)
    CENTER_COL = metrics.index(CENTER_METRIC_NAME) if CENTER_METRIC_NAME in metrics else (N // 2)

    ITEM_H_CM = 0.34
    MIN_PANEL_H = 2.0
    def panel_height(n_items): return max(MIN_PANEL_H, n_items * ITEM_H_CM)

    h_row = [
        panel_height(max(len(arch_df[m]) for m in metrics)),
        panel_height(max(len(ds_df[m]) for m in metrics)),
    ]

    LABEL_COL_W = 2.1
    FIG_W_CM = 18.0
    PLOT_W_CM = (FIG_W_CM - LABEL_COL_W) / N

    # Keep panels pleasantly square-ish for 6 items
    TARGET_H = PLOT_W_CM * 0.95
    h_row[0] = max(h_row[0], TARGET_H)
    h_row[1] = max(h_row[1], TARGET_H)

    LEGEND_H_CM = 0.9
    FIG_H_CM = sum(h_row) + 0.55 * 2 + 0.15 * 2 + 0.35 + LEGEND_H_CM + 0.4

    fig = plt.figure(figsize=(FIG_W_CM / 2.54, FIG_H_CM / 2.54), dpi=300)
    gs = fig.add_gridspec(
        2, N + 1,
        width_ratios=[LABEL_COL_W / PLOT_W_CM] + [1.0] * N,
        height_ratios=h_row,
        hspace=0.22,
        wspace=0.20,
        left=0.01,
        right=0.99,
        top=0.965,
        bottom=(LEGEND_H_CM + 0.5) / FIG_H_CM,
    )

    # Label column
    label_axes = [fig.add_subplot(gs[i, 0]) for i in range(2)]
    for lax in label_axes:
        lax.axis("off")

    ROW_LABELS = ["Architecture\nContrasts", "Dataset\nContrasts"]
    x_offsets = [-0.1, 0.05]
    for i, lax in enumerate(label_axes):
        lax.text(x_offsets[i], 0.50, ROW_LABELS[i],
                 transform=lax.transAxes, rotation=90,
                 ha="center", va="center",
                 fontsize=7.5, color="#333333", fontweight="semibold")

    # Plot axes
    axes = np.array([[fig.add_subplot(gs[i, j + 1]) for j in range(N)] for i in range(2)])

    for col_idx, m in enumerate(metrics):
        # shared xlim across both rows for this metric
        xlim = compute_column_xlim(
            [arch_df[m], ds_df[m]],
            hdi_lo_col="Δ_lo", hdi_hi_col="Δ_hi",
            rope_val=rope[m],
        )

        # Top row title
        axes[0, col_idx].set_title(metric_title(m), fontsize=8, fontweight="bold", pad=5, color="#222222")

        plot_forest_panel(
            ax=axes[0, col_idx],
            df=arch_df[m],
            effect_col="Δ_mean", hdi_lo_col="Δ_lo", hdi_hi_col="Δ_hi",
            rope_val=rope[m],
            title="",
            is_bottom_row=False,
            is_leftmost_col=(col_idx == 0),
            is_center_col=(col_idx == CENTER_COL),
            xlim=xlim,
            metric_name=m,
        )

        plot_forest_panel(
            ax=axes[1, col_idx],
            df=ds_df[m],
            effect_col="Δ_mean", hdi_lo_col="Δ_lo", hdi_hi_col="Δ_hi",
            rope_val=rope[m],
            title="",
            is_bottom_row=True,
            is_leftmost_col=(col_idx == 0),
            is_center_col=(col_idx == CENTER_COL),
            xlim=xlim,
            metric_name=m,
        )

    add_figure_legend(fig, bottom_y=0.00)
    plt.savefig(output_path, dpi=500, bbox_inches="tight", facecolor="white")
    print(f"Saved → {output_path}")
    plt.close(fig)

def create_contrast_grid_marginal(arch_o, ds_o, factorial, output_path):
    metrics = [m for m in METRICS_PREFERRED if (m in arch_o and m in ds_o)]
    if not metrics:
        raise ValueError("No expected metrics found in marginal contrast data.")

    arch_df = {}
    ds_df = {}
    rope = {}
    for m in metrics:
        arch_df[m] = build_marginal_arch_df(arch_o[m])
        ds_df[m] = build_marginal_dataset_df(ds_o[m])
        rope[m] = infer_display_rope_for_metric(m, factorial.get(m))

    N = len(metrics)
    CENTER_COL = metrics.index(CENTER_METRIC_NAME) if CENTER_METRIC_NAME in metrics else (N // 2)

    ITEM_H_CM = 0.34
    MIN_PANEL_H = 1.6
    def panel_height(n_items): return max(MIN_PANEL_H, n_items * ITEM_H_CM)

    h_row = [
        panel_height(max(len(arch_df[m]) for m in metrics)),
        panel_height(max(len(ds_df[m]) for m in metrics)),
    ]

    LABEL_COL_W = 2.1
    FIG_W_CM = 18.0
    PLOT_W_CM = (FIG_W_CM - LABEL_COL_W) / N

    # Marginal plots: aim for half-square-ish panels
    TARGET_H = PLOT_W_CM / 2.0
    h_row[0] = max(h_row[0], TARGET_H)
    h_row[1] = max(h_row[1], TARGET_H)

    LEGEND_H_CM = 0.9
    FIG_H_CM = sum(h_row) + 0.55 * 2 + 0.15 * 2 + 0.35 + LEGEND_H_CM + 0.4

    fig = plt.figure(figsize=(FIG_W_CM / 2.54, FIG_H_CM / 2.54), dpi=300)
    gs = fig.add_gridspec(
        2, N + 1,
        width_ratios=[LABEL_COL_W / PLOT_W_CM] + [1.0] * N,
        height_ratios=h_row,
        hspace=0.25,
        wspace=0.20,
        left=0.01,
        right=0.99,
        top=0.965,
        bottom=(LEGEND_H_CM + 0.5) / FIG_H_CM,
    )

    label_axes = [fig.add_subplot(gs[i, 0]) for i in range(2)]
    for lax in label_axes:
        lax.axis("off")

    ROW_LABELS = ["Architecture\nContrasts\n(avg.)", "Dataset\nContrasts\n(avg.)"]
    x_offsets = [-0.1, 0.6]
    for i, lax in enumerate(label_axes):
        lax.text(x_offsets[i], 0.50, ROW_LABELS[i],
                 transform=lax.transAxes, rotation=90,
                 ha="center", va="center",
                 fontsize=7.5, color="#333333", fontweight="semibold")

    axes = np.array([[fig.add_subplot(gs[i, j + 1]) for j in range(N)] for i in range(2)])

    for col_idx, m in enumerate(metrics):
        # shared xlim across both rows for this metric
        xlim = compute_column_xlim(
            [arch_df[m], ds_df[m]],
            hdi_lo_col="hdi_lo_original", hdi_hi_col="hdi_hi_original",
            rope_val=rope[m],
        )

        axes[0, col_idx].set_title(metric_title(m), fontsize=8, fontweight="bold", pad=5, color="#222222")

        plot_forest_panel(
            ax=axes[0, col_idx],
            df=arch_df[m],
            effect_col="mean_original", hdi_lo_col="hdi_lo_original", hdi_hi_col="hdi_hi_original",
            rope_val=rope[m],
            is_bottom_row=False,
            is_leftmost_col=(col_idx == 0),
            is_center_col=(col_idx == CENTER_COL),
            xlim=xlim,
            metric_name=m,
        )

        plot_forest_panel(
            ax=axes[1, col_idx],
            df=ds_df[m],
            effect_col="mean_original", hdi_lo_col="hdi_lo_original", hdi_hi_col="hdi_hi_original",
            rope_val=rope[m],
            is_bottom_row=True,
            is_leftmost_col=(col_idx == 0),
            is_center_col=(col_idx == CENTER_COL),
            xlim=xlim,
            metric_name=m,
        )

    add_figure_legend(fig, bottom_y=0.00)
    plt.savefig(output_path, dpi=500, bbox_inches="tight", facecolor="white")
    print(f"Saved → {output_path}")
    plt.close(fig)

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    REPORT_PATH = "/isipd/projects/p_planetdw/git/DriftwoodMappingBenchmark/analysis/stats_report.html"
    OUT_DIR = "./figs/out"
    os.makedirs(OUT_DIR, exist_ok=True)

    factorial = extract_factorial_effects(REPORT_PATH)
    arch_w = extract_arch_contrasts(REPORT_PATH)
    arch_o = extract_overall_arch_contrasts(REPORT_PATH)
    ds_w = extract_dataset_contrasts(REPORT_PATH)
    ds_o = extract_overall_dataset_contrasts(REPORT_PATH)

    for ext in ("pdf", "png"):
        # Combined figure with marginal on top, within on bottom
        create_contrast_grid_combined(
            arch_w, ds_w, arch_o, ds_o, factorial,
            os.path.join(OUT_DIR, f"Figure_5_contrasts.{ext}")
        )

    print("Done!")

def create_contrast_grid_combined(arch_w, ds_w, arch_o, ds_o, factorial, output_path):
    """Combined figure with 4 rows: marginal contrasts on top, within-group contrasts on bottom."""
    metrics = [m for m in METRICS_PREFERRED if (m in arch_w and m in ds_w and m in arch_o and m in ds_o)]
    if not metrics:
        raise ValueError("No expected metrics found in contrast data.")

    # Prepare all per-metric dfs
    arch_marginal_df = {}
    ds_marginal_df = {}
    arch_within_df = {}
    ds_within_df = {}
    rope = {}
    
    for m in metrics:
        arch_marginal_df[m] = build_marginal_arch_df(arch_o[m])
        ds_marginal_df[m] = build_marginal_dataset_df(ds_o[m])
        arch_within_df[m] = build_within_arch_df(arch_w[m])
        ds_within_df[m] = build_within_dataset_df(ds_w[m])
        rope[m] = infer_display_rope_for_metric(m, factorial.get(m))

    plot_export_frames = []
    for m in metrics:
        panel_frames = [
            ("arch_marginal", arch_marginal_df[m]),
            ("dataset_marginal", ds_marginal_df[m]),
            ("arch_within", arch_within_df[m]),
            ("dataset_within", ds_within_df[m]),
        ]
        for panel_name, panel_df in panel_frames:
            if panel_df is None or panel_df.empty:
                continue
            plot_export_frames.append(
                panel_df.assign(metric=m, panel=panel_name)
            )
    plot_export_df = (
        pd.concat(plot_export_frames, ignore_index=True)
        if plot_export_frames else pd.DataFrame(columns=["metric", "panel"])
    )

    N = len(metrics)
    CENTER_COL = metrics.index(CENTER_METRIC_NAME) if CENTER_METRIC_NAME in metrics else (N // 2)

    ITEM_H_CM = 0.34
    MIN_PANEL_H_MARGINAL = 1.6
    MIN_PANEL_H_WITHIN = 2.0
    
    def panel_height_marginal(n_items): return max(MIN_PANEL_H_MARGINAL, n_items * ITEM_H_CM)
    def panel_height_within(n_items): return max(MIN_PANEL_H_WITHIN, n_items * ITEM_H_CM)

    h_row = [
        panel_height_marginal(max(len(arch_marginal_df[m]) for m in metrics)),  # Row 0: arch marginal
        panel_height_marginal(max(len(ds_marginal_df[m]) for m in metrics)),    # Row 1: ds marginal
        panel_height_within(max(len(arch_within_df[m]) for m in metrics)),      # Row 2: arch within
        panel_height_within(max(len(ds_within_df[m]) for m in metrics)),        # Row 3: ds within
    ]

    LABEL_COL_W = 2.1
    FIG_W_CM = 18.0
    PLOT_W_CM = (FIG_W_CM - LABEL_COL_W) / N

    # Adjust heights to be proportional
    TARGET_H_MARGINAL = PLOT_W_CM / 2.0
    TARGET_H_WITHIN = PLOT_W_CM * 0.95
    
    h_row[0] = max(h_row[0], TARGET_H_MARGINAL)
    h_row[1] = max(h_row[1], TARGET_H_MARGINAL)
    h_row[2] = max(h_row[2], TARGET_H_WITHIN)
    h_row[3] = max(h_row[3], TARGET_H_WITHIN)

    LEGEND_H_CM = 0.9
    FIG_H_CM = sum(h_row) + 0.55 * 4 + 0.15 * 4 + 0.35 + LEGEND_H_CM + 0.4

    fig = plt.figure(figsize=(FIG_W_CM / 2.54, FIG_H_CM / 2.54), dpi=300)
    gs = fig.add_gridspec(
        4, N + 1,
        width_ratios=[LABEL_COL_W / PLOT_W_CM] + [1.0] * N,
        height_ratios=h_row,
        hspace=0.25,
        wspace=0.20,
        left=0.01,
        right=0.99,
        top=0.99,
        bottom=(LEGEND_H_CM + 0.5) / FIG_H_CM,
    )

    # Label column
    label_axes = [fig.add_subplot(gs[i, 0]) for i in range(4)]
    for lax in label_axes:
        lax.axis("off")

    ROW_LABELS = [
        "Archictecture \neffect\n(marginal)",
        "Dataset \neffect\n(marginal)",
        "Architecture effect\n(per dataset)",
        "Dataset effect\n(per arch.)",
    ]
    x_offsets = [-0.1, -0.1, -0.1, -0.1]
    for i, lax in enumerate(label_axes):
        lax.text(x_offsets[i], 0.50, ROW_LABELS[i],
                 transform=lax.transAxes, rotation=90,
                 ha="center", va="center",
                 fontsize=7.5, color="#333333", fontweight="semibold")

    # Plot axes - 4 rows
    axes = np.array([[fig.add_subplot(gs[i, j + 1]) for j in range(N)] for i in range(4)])

    for col_idx, m in enumerate(metrics):
        # Shared xlim for marginal plots (rows 0-1)
        xlim_marginal = compute_column_xlim(
            [arch_marginal_df[m], ds_marginal_df[m]],
            hdi_lo_col="hdi_lo_original", hdi_hi_col="hdi_hi_original",
            rope_val=rope[m],
        )
        
        # Shared xlim for within plots (rows 2-3)
        xlim_within = compute_column_xlim(
            [arch_within_df[m], ds_within_df[m]],
            hdi_lo_col="Δ_lo", hdi_hi_col="Δ_hi",
            rope_val=rope[m],
        )

        # Top row title
        axes[0, col_idx].set_title(metric_title(m), fontsize=8, fontweight="bold", pad=5, color="#222222")

        # Row 0: Architecture Contrasts (avg)
        plot_forest_panel(
            ax=axes[0, col_idx],
            df=arch_marginal_df[m],
            effect_col="mean_original", hdi_lo_col="hdi_lo_original", hdi_hi_col="hdi_hi_original",
            rope_val=rope[m],
            is_bottom_row=False,
            is_leftmost_col=(col_idx == 0),
            is_center_col=(col_idx == CENTER_COL),
            xlim=xlim_marginal,
            metric_name=m,
        )

        # Row 1: Dataset Contrasts (avg)
        plot_forest_panel(
            ax=axes[1, col_idx],
            df=ds_marginal_df[m],
            effect_col="mean_original", hdi_lo_col="hdi_lo_original", hdi_hi_col="hdi_hi_original",
            rope_val=rope[m],
            is_bottom_row=False,
            is_leftmost_col=(col_idx == 0),
            is_center_col=(col_idx == CENTER_COL),
            xlim=xlim_marginal,
            metric_name=m,
        )

        # Row 2: Architecture Contrasts (within)
        plot_forest_panel(
            ax=axes[2, col_idx],
            df=arch_within_df[m],
            effect_col="Δ_mean", hdi_lo_col="Δ_lo", hdi_hi_col="Δ_hi",
            rope_val=rope[m],
            is_bottom_row=False,
            is_leftmost_col=(col_idx == 0),
            is_center_col=(col_idx == CENTER_COL),
            xlim=xlim_within,
            metric_name=m,
        )

        # Row 3: Dataset Contrasts (within)
        plot_forest_panel(
            ax=axes[3, col_idx],
            df=ds_within_df[m],
            effect_col="Δ_mean", hdi_lo_col="Δ_lo", hdi_hi_col="Δ_hi",
            rope_val=rope[m],
            is_bottom_row=True,
            is_leftmost_col=(col_idx == 0),
            is_center_col=(col_idx == CENTER_COL),
            xlim=xlim_within,
            metric_name=m,
        )

    add_figure_legend(fig, bottom_y=0.00)

    out_dir = os.path.dirname(output_path)
    out_csv = os.path.join(out_dir, "Figure_5_data.csv")
    plot_export_df.to_csv(out_csv, index=False)
    print(f"Saved → {out_csv}")

    plt.savefig(output_path, dpi=500, bbox_inches="tight", facecolor="white")
    print(f"Saved → {output_path}")
    plt.close(fig)

if __name__ == "__main__":
    main()