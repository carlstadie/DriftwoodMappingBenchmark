"""
figure_rank_stability_full.py
3×2 rank-stability heatmap for all five metrics + composite ranking.
Row 1: IoU | Dice | NSD
Row 2: Epistemic uncertainty | Aleatoric uncertainty | Composite
"""

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import os
from scipy.stats import kendalltau

# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────

OUT_DIR = "/isipd/projects/p_planetdw/git/DriftwoodMappingBenchmark/analysis/figs/out"
os.makedirs(OUT_DIR, exist_ok=True)

BASE_PATH = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "result_safe", "dataset_level")
)

# ─────────────────────────────────────────────────────────────────────────────
# STYLE
# ─────────────────────────────────────────────────────────────────────────────

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.major.size": 3.5,
    "ytick.major.size": 3.5,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})

COL_DIAGONAL = "#4B4B4B"
SPINE_COL    = "#333333"
GRID_COL     = "#E8E8E8"
CMAP         = cm.get_cmap("Blues", 11)

# ─────────────────────────────────────────────────────────────────────────────
# MODEL CONFIGS
# ─────────────────────────────────────────────────────────────────────────────

BAYESIAN = [
    "UNETxAE", "SWINxPS", "UNETxPS", "SWINxS2", "SWINxAE",
    "TERRAxPS", "TERRAxAE", "TERRAxS2", "UNETxS2",
]
N = len(BAYESIAN)

KEY_MAP = {
    "UNETxAE":  "UNETxAE_dataset_metrics.csv",
    "UNETxPS":  "UNETxPS_dataset_metrics.csv",
    "UNETxS2":  "UNETxS2_dataset_metrics.csv",
    "SWINxAE":  "SWINxAE_dataset_metrics.csv",
    "SWINxPS":  "SWINxPS_dataset_metrics.csv",
    "SWINxS2":  "SWINxS2_dataset_metrics.csv",
    "TERRAxAE": "TERRAMINDxAE_dataset_metrics.csv",
    "TERRAxPS": "TERRAMINDxPS_dataset_metrics.csv",
    "TERRAxS2": "TERRAMINDxS2_dataset_metrics.csv",
}
GROUPS = list(KEY_MAP.keys())

# ─────────────────────────────────────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────────────────────────────────────

METRICS = [
    "IoU",
    "dice_coef",
    "normalized_surface_distance",
    "mean_epistemic_uncertainty",
    "mean_aleatoric_uncertainty",
]

HIGHER_IS_BETTER = {
    "IoU":                          True,
    "dice_coef":                    True,
    "normalized_surface_distance":  True,
    "mean_epistemic_uncertainty":   False,
    "mean_aleatoric_uncertainty":   False,
}

METRIC_IDX = {m: i for i, m in enumerate(METRICS)}
W_VEC      = np.full(len(METRICS), 1 / len(METRICS))

# ─────────────────────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────

raw = {
    g: pd.read_csv(os.path.join(BASE_PATH, f))[METRICS].values.astype(float)
    for g, f in KEY_MAP.items()
}

# ─────────────────────────────────────────────────────────────────────────────
# COMPUTE SINGLE-RUN RANKINGS  (one per metric + composite)
# ─────────────────────────────────────────────────────────────────────────────

def runs_for_metric(metric):
    """Return list of 10 single-run rankings for one metric."""
    idx    = METRIC_IDX[metric]
    higher = HIGHER_IS_BETTER[metric]
    result = []
    for run in range(10):
        vals  = np.array([raw[g][run, idx] for g in GROUPS])
        order = np.argsort(-vals) if higher else np.argsort(vals)
        result.append([GROUPS[i] for i in order])
    return result

def runs_composite():
    """Return list of 10 single-run composite rankings."""
    result = []
    for run in range(10):
        vals   = np.stack([raw[g][run] for g in GROUPS])
        mn, mx = vals.min(axis=0), vals.max(axis=0)
        normed = (vals - mn) / (mx - mn + 1e-9)
        for i, m in enumerate(METRICS):
            if not HIGHER_IS_BETTER[m]:
                normed[:, i] = 1 - normed[:, i]
        scores = normed @ W_VEC
        order  = np.argsort(-scores)
        result.append([GROUPS[i] for i in order])
    return result

# IoU single-run lists from posterior analysis
ALL_RUNS = {
    "IoU": [
        ["UNETxAE","SWINxAE","TERRAxAE","UNETxPS","SWINxPS","TERRAxPS","SWINxS2","TERRAxS2","UNETxS2"],
        ["SWINxAE","UNETxAE","TERRAxAE","UNETxPS","SWINxPS","TERRAxPS","SWINxS2","TERRAxS2","UNETxS2"],
        ["UNETxAE","TERRAxAE","SWINxAE","UNETxPS","SWINxPS","TERRAxPS","SWINxS2","TERRAxS2","UNETxS2"],
        ["UNETxAE","SWINxAE","TERRAxAE","UNETxPS","SWINxPS","TERRAxPS","SWINxS2","TERRAxS2","UNETxS2"],
        ["UNETxAE","TERRAxAE","SWINxAE","SWINxPS","UNETxPS","SWINxS2","TERRAxPS","TERRAxS2","UNETxS2"],
        ["SWINxAE","UNETxAE","TERRAxAE","UNETxPS","SWINxPS","TERRAxPS","SWINxS2","TERRAxS2","UNETxS2"],
        ["UNETxAE","SWINxAE","TERRAxAE","UNETxPS","SWINxPS","TERRAxPS","SWINxS2","TERRAxS2","UNETxS2"],
        ["UNETxAE","TERRAxAE","SWINxAE","UNETxPS","SWINxPS","SWINxS2","TERRAxPS","TERRAxS2","UNETxS2"],
        ["SWINxAE","UNETxAE","TERRAxAE","UNETxPS","SWINxPS","TERRAxPS","SWINxS2","TERRAxS2","UNETxS2"],
        ["UNETxAE","TERRAxAE","SWINxAE","UNETxPS","SWINxPS","TERRAxPS","SWINxS2","TERRAxS2","UNETxS2"],
    ],
    "dice_coef":                    runs_for_metric("dice_coef"),
    "normalized_surface_distance":  runs_for_metric("normalized_surface_distance"),
    "mean_epistemic_uncertainty":   runs_for_metric("mean_epistemic_uncertainty"),
    "mean_aleatoric_uncertainty":   runs_for_metric("mean_aleatoric_uncertainty"),
    "composite":                    runs_composite(),
}

# ─────────────────────────────────────────────────────────────────────────────
# FREQUENCY MATRICES & STATISTICS
# ─────────────────────────────────────────────────────────────────────────────

def freq_matrix(runs):
    mat = np.zeros((N, N), dtype=int)
    for r, g in enumerate(BAYESIAN):
        for run in runs:
            mat[r, run.index(g)] += 1
    return mat

def rank_agreement(freq):
    return 100 * np.trace(freq) / (N * 10)

def kendall_mean(runs):
    ref  = list(range(N))
    taus = [kendalltau(ref, [run.index(g) for g in BAYESIAN])[0] for run in runs]
    return np.mean(taus)

PANELS = [
    # (key,                          label,                      panel_letter)
    ("IoU",                         "IoU",                       "a"),
    ("dice_coef",                   "Dice",                      "b"),
    ("normalized_surface_distance", "NSD",                       "c"),
    ("mean_epistemic_uncertainty",  "Epistemic uncertainty",     "d"),
    ("mean_aleatoric_uncertainty",  "Aleatoric uncertainty",     "e"),
    ("composite",                   "Composite",                 "f"),
]

stats = {}
for key, _, _ in PANELS:
    runs         = ALL_RUNS[key]
    freq         = freq_matrix(runs)
    stats[key]   = dict(freq=freq, ra=rank_agreement(freq), tau=kendall_mean(runs))

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE LAYOUT  –  3 cols × 2 rows, wider than tall
# ─────────────────────────────────────────────────────────────────────────────

NCOLS, NROWS = 3, 2

# Figure: ~250 mm wide × 160 mm tall
FIG_W = 250 / 25.4
FIG_H = 160 / 25.4

fig = plt.figure(figsize=(FIG_W, FIG_H))

# Manual axes grid: leave room for titles above, x-labels below, y-labels left,
# and a shared colorbar strip on the right.
LEFT_MARGIN  = 0.07   # space for y-axis label + ticks
RIGHT_MARGIN = 0.08   # space for colorbar
BOT_MARGIN   = 0.10   # space for x-axis label + ticks
TOP_MARGIN   = 0.14   # space for panel label + stats line

COL_GAP = 0.03
ROW_GAP = 0.14        # enough for two annotation lines between rows

total_w = 1.0 - LEFT_MARGIN - RIGHT_MARGIN
total_h = 1.0 - BOT_MARGIN  - TOP_MARGIN

ax_w = (total_w - (NCOLS - 1) * COL_GAP) / NCOLS
ax_h = (total_h - (NROWS - 1) * ROW_GAP) / NROWS

axes = {}
for row in range(NROWS):
    for col in range(NCOLS):
        idx   = row * NCOLS + col
        key   = PANELS[idx][0]
        x0    = LEFT_MARGIN + col * (ax_w + COL_GAP)
        # row 0 = top row in figure coords → highest y0
        y0    = BOT_MARGIN  + (NROWS - 1 - row) * (ax_h + ROW_GAP)
        ax    = fig.add_axes([x0, y0, ax_w, ax_h])
        axes[key] = ax

# Shared colorbar on far right, spanning full plot height
cax = fig.add_axes([
    LEFT_MARGIN + total_w + COL_GAP * 0.8,
    BOT_MARGIN,
    0.018,
    total_h,
])

# ─────────────────────────────────────────────────────────────────────────────
# DRAW FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def draw_panel(ax, freq, show_ylabel=True, show_xlabel=True):
    ax.imshow(freq, cmap=CMAP, vmin=0, vmax=10, aspect="equal",
              interpolation="nearest")

    # subtle grid
    for i in range(N + 1):
        ax.axhline(i - 0.5, color=GRID_COL, lw=0.35, zorder=1)
        ax.axvline(i - 0.5, color=GRID_COL, lw=0.35, zorder=1)

    # cell annotations
    for r in range(N):
        for c in range(N):
            val = freq[r, c]
            if val == 0:
                continue
            text_col = "white" if val / 10.0 > 0.55 else "#1a1a1a"
            ax.text(c, r, str(val), ha="center", va="center",
                    color=text_col, fontsize=7, fontweight="bold", zorder=3)

    # diagonal agreement boxes
    for i in range(N):
        ax.add_patch(patches.FancyBboxPatch(
            (i - 0.48, i - 0.48), 0.96, 0.96,
            boxstyle="square,pad=0", fill=False,
            linewidth=1.5, edgecolor=COL_DIAGONAL, zorder=4,
        ))

    ax.set_xticks(range(N))
    ax.set_yticks(range(N))

    if show_xlabel:
        ax.set_xticklabels([str(i + 1) for i in range(N)])
        ax.set_xlabel("Single-run rank", labelpad=3)
    else:
        ax.set_xticklabels([])

    if show_ylabel:
        ax.set_yticklabels([str(i + 1) for i in range(N)])
        ax.set_ylabel("Posterior rank", labelpad=3)
    else:
        ax.set_yticklabels([])

    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color(SPINE_COL)

    ax.tick_params(axis="both", direction="out", length=2.5, width=0.8, pad=2)


for idx, (key, label, letter) in enumerate(PANELS):
    row = idx // NCOLS
    col = idx  % NCOLS
    ax  = axes[key]

    show_y = (col == 0)
    show_x = (row == NROWS - 1)

    draw_panel(ax, stats[key]["freq"], show_ylabel=show_y, show_xlabel=show_x)

    # panel header: bold label + stats line
    pos  = ax.get_position()
    xc   = pos.x0 + pos.width / 2
    ytop = pos.y1

    fig.text(xc, ytop + 0.062,
             f"({letter}) {label}",
             ha="center", va="bottom", fontsize=9, fontweight="bold")

    fig.text(xc, ytop + 0.012,
             f"RA: {stats[key]['ra']:.1f}%   τ = {stats[key]['tau']:.2f}",
             ha="center", va="bottom", fontsize=7.5, color="#444444")

# ─────────────────────────────────────────────────────────────────────────────
# SHARED COLORBAR
# ─────────────────────────────────────────────────────────────────────────────

bounds = np.arange(-0.5, 11.5, 1)
norm   = mcolors.BoundaryNorm(bounds, CMAP.N)
sm     = cm.ScalarMappable(cmap=CMAP, norm=norm)
sm.set_array([])

cb = fig.colorbar(sm, cax=cax, orientation="vertical",
                  boundaries=bounds, ticks=[0, 5, 10], spacing="proportional")
cb.set_label("Number of runs", labelpad=6, fontsize=8)
cb.ax.tick_params(labelsize=7.5, length=0, width=0)
cb.outline.set_linewidth(0.7)

# ─────────────────────────────────────────────────────────────────────────────
# SAVE
# ─────────────────────────────────────────────────────────────────────────────

for ext in ("pdf", "png"):
    out = os.path.join(OUT_DIR, f"Figure_S3_rank_stability.{ext}")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Saved → {out}")