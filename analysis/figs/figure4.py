"""
Figure 4: Factorial effects from Bayesian analysis.

"""

import os
import numpy as np
import pandas as pd
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator, ScalarFormatter
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from html_parser import extract_factorial_effects

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

# ----------------------------
# Plot rcParams — publication quality
# ----------------------------
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

# ----------------------------
# Classification / colors — muted, print-safe palette
# ----------------------------
LOWER_IS_BETTER_METRICS = {"Epistemic", "Aleatoric"}
PRACTICAL_PROB_THRESHOLD = 0.95

COLOR_IMPROVE    = "#2166AC"   # dark blue (ColorBrewer RdBu)
COLOR_DEGRADE    = "#E08214"   # muted red
COLOR_NEGLIGIBLE = "#BABABA"   # light grey
COLOR_INCONCLUSIVE = "#878787" # mid grey
COLOR_ROPE_SHADE = "#F5F5F5"   # very light grey fill
COLOR_ZERO_LINE  = "#333333"

PLOT_TO_CONFIG_METRIC = {
    "IoU": "IoU",
    "Dice": "dice_coef",
    "NSD": "normalized_surface_distance",
    "Epistemic": "mean_epistemic_uncertainty",
    "Aleatoric": "mean_aleatoric_uncertainty",
}

# ----------------------------
# Helpers: labels / sorting
# ----------------------------
def normalize_label(label: str) -> str:
    s = str(label).strip()
    s = s.replace("Terramind", "TM")
    s = s.replace(" x ", "×").replace(" X ", "×").replace(" × ", "×")
    return s


def _abbr(s: str, n: int = 10) -> str:
    s = str(s).strip()
    return s if len(s) <= n else s[:max(1, n - 1)] + "…"


def compact_effect_label(label, max_len=16, wrap_interactions=True):
    label = normalize_label(label)
    if "×" in label and wrap_interactions:
        left, right = label.split("×", 1)
        one_line = f"{_abbr(left, 9)}×{_abbr(right, 9)}"
        if len(one_line) <= max_len:
            return one_line
        return f"{_abbr(left, 10)}\n×{_abbr(right, 10)}"
    return _abbr(label, max_len)


def parse_interaction_label(label):
    s = normalize_label(label)
    if "×" not in s:
        return None, None
    a, b = [p.strip() for p in s.split("×", 1)]
    return a, b


def sort_dataset_effects(df):
    order_map = {"AE": 0, "PS": 1, "S2": 2}
    out = df.copy()
    out["param_clean"] = out["param_clean"].astype(str).map(normalize_label)
    out["sort_key"] = out["param_clean"].map(order_map).fillna(999)
    return out.sort_values(["sort_key", "param_clean"]).drop(columns=["sort_key"])


def sort_arch_effects(df):
    order_map = {"U-Net": 0, "Swin": 1, "TM": 2, "Terramind": 2}
    out = df.copy()
    out["param_clean"] = out["param_clean"].astype(str).map(normalize_label)
    out["sort_key"] = out["param_clean"].map(order_map).fillna(999)
    return out.sort_values(["sort_key", "param_clean"]).drop(columns=["sort_key"])


def sort_interaction_effects(df):
    dataset_order = {"AE": 0, "PS": 1, "S2": 2}
    arch_order = {"U-Net": 0, "Swin": 1, "TM": 2, "Terramind": 2}
    out = df.copy()
    out["param_clean"] = out["param_clean"].astype(str).map(normalize_label)
    ds_keys, arch_keys = [], []
    for lbl in out["param_clean"]:
        a, b = parse_interaction_label(lbl)
        if a in dataset_order and b in arch_order:
            ds_keys.append(dataset_order[a]); arch_keys.append(arch_order[b])
        elif b in dataset_order and a in arch_order:
            ds_keys.append(dataset_order[b]); arch_keys.append(arch_order[a])
        else:
            ds_keys.append(999); arch_keys.append(999)
    out["ds_key"] = ds_keys
    out["arch_key"] = arch_keys
    return out.sort_values(["ds_key", "arch_key", "param_clean"]).drop(columns=["ds_key", "arch_key"])


def choose_effect_columns(df):
    if "Δp_mean" in df.columns and "Δp_lo" in df.columns and "Δp_hi" in df.columns:
        return "Δp_mean", "Δp_lo", "Δp_hi"
    if "Abs_Δ_mean" in df.columns and "Abs_Δ_lo" in df.columns and "Abs_Δ_hi" in df.columns:
        return "Abs_Δ_mean", "Abs_Δ_lo", "Abs_Δ_hi"
    raise ValueError(f"Could not infer effect columns from: {list(df.columns)}")


def pd_safe_values(series):
    if series is None:
        return np.array([], dtype=float)
    try:
        return np.asarray(series.values, dtype=float)
    except Exception:
        return np.array([], dtype=float)


# ----------------------------
# ROPE helpers
# ----------------------------
def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _logit(p, eps=1e-9):
    p = float(np.clip(p, eps, 1 - eps))
    return np.log(p / (1 - p))


def infer_display_rope_for_metric(metric_plot_name, df_metric):
    cfg_key = PLOT_TO_CONFIG_METRIC.get(metric_plot_name, metric_plot_name)
    rope_t = float(ROPE_BY_METRIC_TRANSFORMED.get(cfg_key, 0.05))
    if "baseline" not in df_metric.columns:
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


# ----------------------------
# Axis helpers
# ----------------------------
def style_xaxis_ticks(ax, x_abs_max, metric_name=None):
    ax.xaxis.set_major_locator(MaxNLocator(nbins=3, symmetric=True))
    if x_abs_max is None or not np.isfinite(x_abs_max):
        return
    
    # Force numeric (non-exponential) format for uncertainty metrics
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


def compute_column_xlim(subsets, hdi_lo_col, hdi_hi_col, pad_frac=0.18, min_pad=1e-6, rope_val=None):
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


# ----------------------------
# Classification
# ----------------------------
def classify_effect_posterior_rope(row, metric_name):
    lower_is_better = metric_name in LOWER_IS_BETTER_METRICS
    pr_rope   = float(row.get("Pr(|Δ|<ROPE)", np.nan))
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
        return {"line_color": COLOR_IMPROVE,  "marker_face": COLOR_IMPROVE,
                "marker_edge": "white", "marker_lw": 0.7, "marker_symbol": "o"}
    if effect_class == "degrade":
        return {"line_color": COLOR_DEGRADE,  "marker_face": COLOR_DEGRADE,
                "marker_edge": "white", "marker_lw": 0.7, "marker_symbol": "o"}
    if effect_class == "negligible":
        return {"line_color": COLOR_NEGLIGIBLE, "marker_face": COLOR_NEGLIGIBLE,
                "marker_edge": "white", "marker_lw": 0.7, "marker_symbol": "o"}
    return {"line_color": COLOR_INCONCLUSIVE, "marker_face": "white",
            "marker_edge": COLOR_INCONCLUSIVE, "marker_lw": 1.0, "marker_symbol": "o"}


# ----------------------------
# Panel plotting
# ----------------------------
def plot_forest_panel(
    ax, effects_df, effect_col, hdi_lo_col, hdi_hi_col,
    rope_val, title="", is_bottom_row=False, is_leftmost_col=False, is_center_col=False,
    xlim=None, metric_name=None, row_height_in=None,
):
    if effects_df is None or effects_df.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center",
                transform=ax.transAxes, fontsize=7, color="#888888")
        ax.set_xlim(-0.5, 0.5)
        ax.set_xticks([]); ax.set_yticks([])
        for side in ["top", "right", "left", "bottom"]:
            ax.spines[side].set_visible(True)
            ax.spines[side].set_linewidth(0.7)
        return

    df = effects_df.copy().reset_index(drop=True)
    n_effects = len(df)
    y_pos = np.arange(n_effects)

    means  = np.asarray(df[effect_col].values, dtype=float)
    hdi_lo = np.asarray(df[hdi_lo_col].values, dtype=float)
    hdi_hi = np.asarray(df[hdi_hi_col].values, dtype=float)
    labels = df["param_clean"].astype(str).tolist()

    # Spines
    for side in ["top", "right", "left", "bottom"]:
        ax.spines[side].set_visible(True)
        ax.spines[side].set_linewidth(0.7)
        ax.spines[side].set_color("#333333")

    # ROPE band + zero line
    ax.axvspan(-rope_val, rope_val, color=COLOR_ROPE_SHADE, alpha=1.0, zorder=0, lw=0)
    ax.axvline(0, color=COLOR_ZERO_LINE, linestyle="--", linewidth=0.7, alpha=0.6, zorder=1)

    # Horizontal guide lines (very subtle)
    for yp in y_pos:
        ax.axhline(yp, color="#E8E8E8", linewidth=0.35, zorder=0)

    # Forest plot
    for i in range(n_effects):
        cls = classify_effect_posterior_rope(df.iloc[i], metric_name)
        st = style_from_class(cls)
        
        # HDI line (slightly extended for bounded metrics to peek through scatter marker when small)
        hdi_extension = 0.01 if metric_name in {"IoU", "Dice", "NSD"} else 0.0
        ax.plot([hdi_lo[i] - hdi_extension, hdi_hi[i] + hdi_extension], [y_pos[i], y_pos[i]],
                color=st["line_color"], linewidth=1.6,
                solid_capstyle="round", zorder=2)
        
        ax.scatter(means[i], y_pos[i], s=16,
                   marker=st["marker_symbol"],
                   facecolors=st["marker_face"],
                   edgecolors=st["marker_edge"],
                   linewidths=st["marker_lw"],
                   zorder=3)

    # Y-axis labels
    wrap_interactions = any("×" in normalize_label(lbl) for lbl in labels)
    labels_display = [compact_effect_label(lbl, max_len=16,
                      wrap_interactions=wrap_interactions) for lbl in labels]

    # Adaptive font size based on number of items
    if n_effects <= 4:
        y_fontsize = 7.0
    elif n_effects <= 7:
        y_fontsize = 6.5
    else:
        y_fontsize = 6.0

    ax.set_yticks(y_pos)
    if is_leftmost_col:
        ax.set_yticklabels(labels_display, fontsize=y_fontsize)
    else:
        ax.set_yticklabels([])

    for t in ax.get_yticklabels():
        t.set_ha("right")
        t.set_multialignment("right")
    ax.tick_params(axis="y", length=0, pad=2)

    # X-axis
    ax.tick_params(axis="x", labelsize=6, length=3, width=0.7, pad=1)

    # X-limits
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
    
    # Hide x-axis tick labels for upper rows (after formatter is set)
    if not is_bottom_row:
        ax.set_xticklabels([])

    # Y limits — add symmetric padding so top/bottom items don't touch the frame
    ax.set_ylim(-0.65, n_effects - 1 + 0.65)

    # Grid
    ax.grid(axis="x", alpha=0.18, linewidth=0.4, zorder=0)
    ax.set_axisbelow(True)

    # Title (top row only)
    if title:
        ax.set_title(title, fontsize=8, fontweight="bold", pad=5,
                     color="#222222")

    # X-axis label
    ax.set_xlabel("Factor effect on $\Delta$ from grand mean" if (is_bottom_row and is_center_col) else "", fontsize=7, labelpad=3)


# ----------------------------
# Legend
# ----------------------------
def add_figure_legend(fig, bottom_y=0.015):
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
        Patch(facecolor=COLOR_ROPE_SHADE, edgecolor="#BBBBBB", linewidth=0.5,
              label="ROPE"),
        Line2D([0], [0], color=COLOR_ZERO_LINE, lw=0.8, ls="--",
               label="No effect"),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=5,
        frameon=False,  # Remove legend border
        fontsize=6.5,
        handlelength=2.0,
        columnspacing=1.0,
        handletextpad=0.5,
        bbox_to_anchor=(0.55, bottom_y),
    )


# ----------------------------
# Main grid
# ----------------------------
def create_factorial_effects_grid(factorial_data, output_path):
    """
    3×5 grid. Layout goals:
      - Top two rows (dataset + arch effects, ~3 items each) → half-square panels
      - Bottom row (interactions, ~9 items) → square panels
      - Shared x-axis limits per metric column
      - Clean publication styling
    """
    preferred_metrics = ["IoU", "Dice", "NSD", "Epistemic", "Aleatoric"]
    metrics = [m for m in preferred_metrics if m in factorial_data]
    if not metrics:
        raise ValueError("No expected metrics found in factorial_data.")

    subsets_by_metric   = {}
    effect_cols_by_metric = {}
    raw_df_by_metric    = {}
    rope_plot_by_metric = {}
    n_items_by_row      = [0, 0, 0]  # dataset, arch, interaction
    max_interactions    = 0

    for metric in metrics:
        df = factorial_data[metric].copy()
        raw_df_by_metric[metric] = df

        if "param_clean" not in df.columns:
            raise ValueError(f"Metric '{metric}' missing 'param_clean'.")
        if "effect_type" not in df.columns:
            raise ValueError(f"Metric '{metric}' missing 'effect_type'.")

        effect_col, hdi_lo_col, hdi_hi_col = choose_effect_columns(df)
        effect_cols_by_metric[metric] = (effect_col, hdi_lo_col, hdi_hi_col)

        ds_df    = sort_dataset_effects(df[df["effect_type"] == "dataset"].copy())
        arch_df  = sort_arch_effects(df[df["effect_type"] == "arch"].copy())
        inter_df = sort_interaction_effects(df[df["effect_type"] == "interaction"].copy())

        subsets_by_metric[metric] = {"dataset": ds_df, "arch": arch_df, "interaction": inter_df}
        rope_plot_by_metric[metric] = infer_display_rope_for_metric(metric, df)

        n_items_by_row[0] = max(n_items_by_row[0], len(ds_df))
        n_items_by_row[1] = max(n_items_by_row[1], len(arch_df))
        n_items_by_row[2] = max(n_items_by_row[2], len(inter_df))
        max_interactions   = max(max_interactions, len(inter_df))

    plot_export_frames = []
    for metric in metrics:
        for subset_name in ["dataset", "arch", "interaction"]:
            subset_df = subsets_by_metric[metric][subset_name]
            if subset_df is None or subset_df.empty:
                continue
            plot_export_frames.append(
                subset_df.assign(metric=metric, subset=subset_name)
            )
    plot_export_df = (
        pd.concat(plot_export_frames, ignore_index=True)
        if plot_export_frames else pd.DataFrame(columns=["metric", "subset"])
    )

    # ---- Figure geometry ----
    # Each item in a panel needs ~0.32 cm of vertical space (generous for readability)
    # Top/bottom rows have small n → half-square; interaction row → square
    ITEM_H_CM    = 0.34   # vertical space per item
    MIN_PANEL_H  = 1.8    # minimum panel height in cm
    PAD_ABOVE_CM = 0.55   # space above panel for title / row header
    PAD_BELOW_CM = 0.15   # space below panel

    def panel_height(n_items):
        return max(MIN_PANEL_H, n_items * ITEM_H_CM)

    h_row = [panel_height(n) for n in n_items_by_row]

    # For the goal "upper rows = half square, lower row = square":
    # panel width ≈ (FIG_W - label_col) / n_metrics
    # We pick figure width then compute heights that satisfy the constraint.
    N_METRICS     = len(metrics)
    CENTER_COL    = N_METRICS // 2
    LABEL_COL_W   = 1.8   # cm for row-label column
    FIG_W_CM      = 18.0  # total figure width — fits two-column journal page
    PLOT_W_CM     = (FIG_W_CM - LABEL_COL_W) / N_METRICS  # width per metric panel

    # Target: top 2 rows ≈ half square → height = PLOT_W_CM / 2
    # Bottom row ≈ square → height = PLOT_W_CM
    TARGET_H_TOP  = PLOT_W_CM / 2.0
    TARGET_H_BOT  = PLOT_W_CM

    # Use the max of (content-driven, target) for robustness
    h_row[0] = max(h_row[0], TARGET_H_TOP)
    h_row[1] = max(h_row[1], TARGET_H_TOP)
    h_row[2] = max(h_row[2], TARGET_H_BOT)

    LEGEND_H_CM = 0.9
    HSPACE_CM   = 0.35    # vertical space between rows

    FIG_H_CM = (sum(h_row)
                + PAD_ABOVE_CM * 3
                + PAD_BELOW_CM * 3
                + HSPACE_CM * 2
                + LEGEND_H_CM + 0.4)  # 0.4 = bottom margin

    fig = plt.figure(
        figsize=(FIG_W_CM / 2.54, FIG_H_CM / 2.54),
        dpi=300,
    )

    # GridSpec: height_ratios proportional to desired panel heights
    gs = fig.add_gridspec(
        3, N_METRICS + 1,
        width_ratios=[LABEL_COL_W / PLOT_W_CM] + [1.0] * N_METRICS,
        height_ratios=[h_row[0], h_row[1], h_row[2]],
        hspace=0.2,
        wspace=0.2,
        left=0.01,
        right=0.99,
        top=0.97,
        bottom=(LEGEND_H_CM + 0.5) / FIG_H_CM,
    )

    label_axes = [fig.add_subplot(gs[i, 0]) for i in range(3)]
    for lax in label_axes:
        lax.axis("off")

    axes = np.array([
        [fig.add_subplot(gs[i, j + 1]) for j in range(N_METRICS)]
        for i in range(3)
    ])

    metric_directions = {"IoU": "(↑)", "Dice": "(↑)", "NSD": "(↑)",
                         "Epistemic": "(↓)", "Aleatoric": "(↓)"}

    def metric_title(metric):
        if metric == "Epistemic":
            return r"Uncertainty$_\mathrm{epist.}$"
        if metric == "Aleatoric":
            return r"Uncertainty$_\mathrm{alea.}$"
        return metric

    ROW_LABELS = ["Dataset\nEffects", "Architecture\nEffects", "Interaction Effects"]

    # Row labels in the label column
    for row_idx, lax in enumerate(label_axes):
        # Rotated row label
        lax.text(0.38, 0.50, ROW_LABELS[row_idx], transform=lax.transAxes,
                 rotation=90, ha="center", va="center",
                 fontsize=7.5, color="#333333", fontweight="semibold")

    # ---- Plot ----
    class_counts_total = {"improve": 0, "degrade": 0, "negligible": 0, "inconclusive": 0}

    for col_idx, metric in enumerate(metrics):
        df = raw_df_by_metric[metric]
        effect_col, hdi_lo_col, hdi_hi_col = effect_cols_by_metric[metric]
        rope_plot = rope_plot_by_metric[metric]

        ds_df    = subsets_by_metric[metric]["dataset"]
        arch_df  = subsets_by_metric[metric]["arch"]
        inter_df = subsets_by_metric[metric]["interaction"]

        xlim_shared = compute_column_xlim(
            [ds_df, arch_df, inter_df],
            hdi_lo_col=hdi_lo_col,
            hdi_hi_col=hdi_hi_col,
            pad_frac=0.18,
            min_pad=1e-8,
            rope_val=rope_plot,
        )

        for subset_name, subset in [("dataset", ds_df), ("arch", arch_df), ("interaction", inter_df)]:
            if subset is None or subset.empty:
                continue
            for _, row in subset.iterrows():
                cls = classify_effect_posterior_rope(row, metric)
                class_counts_total[cls] += 1

        row_subsets = [ds_df, arch_df, inter_df]
        for row_idx, subset in enumerate(row_subsets):
            ax = axes[row_idx, col_idx]
            title_text = (f"{metric_title(metric)} {metric_directions.get(metric, '')}"
                          if row_idx == 0 else "")

            # Approximate panel height in inches for this row
            panel_h_in = h_row[row_idx] / 2.54

            plot_forest_panel(
                ax=ax,
                effects_df=subset,
                effect_col=effect_col,
                hdi_lo_col=hdi_lo_col,
                hdi_hi_col=hdi_hi_col,
                rope_val=rope_plot,
                title=title_text,
                is_bottom_row=(row_idx == 2),
                is_leftmost_col=(col_idx == 0),
                is_center_col=(col_idx == CENTER_COL),
                xlim=xlim_shared,
                metric_name=metric,
                row_height_in=panel_h_in,
            )

    print("\nOverall classification counts:", class_counts_total)

    # Estimate where the bottom of the lowest axes is in figure coords
    add_figure_legend(fig, bottom_y=0.00)

    out_dir = os.path.dirname(output_path)
    out_csv = os.path.join(out_dir, "Figure_4_data.csv")
    plot_export_df.to_csv(out_csv, index=False)
    print(f"Saved → {out_csv}")

    plt.savefig(output_path, dpi=500, bbox_inches="tight", facecolor="white")
    print(f"Saved → {output_path}")
    plt.close(fig)


# ----------------------------
# Main
# ----------------------------
def main():
    REPORT_PATH = "/isipd/projects/p_planetdw/git/DriftwoodMappingBenchmark/analysis/stats_report.html"
    output_dir  = "/isipd/projects/p_planetdw/git/DriftwoodMappingBenchmark/analysis/figs"
    os.makedirs(output_dir, exist_ok=True)

    factorial_effects = extract_factorial_effects(REPORT_PATH)
    if not factorial_effects:
        print("ERROR: No factorial effects tables found!")
        return

    print(f"Extracted effects for {len(factorial_effects)} metrics:")
    for metric, df in factorial_effects.items():
        print(f"  {metric}: {len(df)} parameters")

    print("\nApprox. displayed ROPE bands:")
    for metric in ["IoU", "Dice", "NSD", "Epistemic", "Aleatoric"]:
        if metric in factorial_effects:
            rope_plot = infer_display_rope_for_metric(metric, factorial_effects[metric])
            print(f"  {metric}: {rope_plot:.6g}")

    print("\nCreating figure …")
    for ext in ("pdf", "png"):
        out = os.path.join(output_dir, f"out/Figure_4_factorial_effects.{ext}")
        print(f"  → {out}")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        create_factorial_effects_grid(factorial_effects, out)

    print("Done!")


if __name__ == "__main__":
    main()