#!/usr/bin/env python3
"""
Figure 6: Architecture dependence — all metrics by architecture, faceted by sensor.

2x3 grid: TOP row = performance metrics (IoU, Dice, NSD),
BOT row = uncertainty metrics (Epistemic, Aleatoric).
"""

from __future__ import annotations

import argparse
from io import StringIO
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, ScalarFormatter

# ─────────────────────────────────────────────────────────────────────────────
# rcParams — matches Fig 4 / Fig 5 exactly
# ─────────────────────────────────────────────────────────────────────────────
mpl.rcParams.update({
    "font.size":         7,
    "axes.titlesize":    8,
    "axes.labelsize":    7,
    "xtick.labelsize":   6,
    "ytick.labelsize":   6.5,
    "axes.linewidth":    0.7,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "xtick.major.size":  3,
    "ytick.major.size":  3,
    "pdf.fonttype":      42,
    "ps.fonttype":       42,
    "mathtext.default":  "regular",
    "figure.facecolor":  "white",
    "axes.facecolor":    "white",
})

# ─────────────────────────────────────────────────────────────────────────────
# Palette — Tol bright, CVD-safe, matches Fig 4/5 improve/degrade colors
# ─────────────────────────────────────────────────────────────────────────────
# Okabe-Ito palette — CVD-safe (deuteranopia, protanopia, tritanopia)
# Performance metrics use the first three (visually distinct on white),
# uncertainties use the remaining two (cool/warm contrast).
METRIC_COLORS = {
    "IoU":       "#0072B2",   # Okabe-Ito blue
    "Dice":      "#E69F00",   # Okabe-Ito orange
    "NSD":       "#009E73",   # Okabe-Ito green
    "Epistemic": "#CC79A7",   # Okabe-Ito pink/mauve
    "Aleatoric": "#D55E00",   # Okabe-Ito vermillion
}

METRICS      = ["IoU", "Dice", "NSD", "Epistemic", "Aleatoric"]
PERF_METRICS = ["IoU", "Dice", "NSD"]
UNC_METRICS  = ["Epistemic", "Aleatoric"]

# Fixed y-limits — one break only
SEG_TOP = (0.1, 1.0)    # performance metrics
SEG_BOT = (0.0, 0.003)  # both uncertainty metrics

SENSORS_PREFERRED    = ["AE", "PS", "S2"]
ARCH_ORDER_PREFERRED = ["U-Net", "Swin", "TM"]

TAB_BY_METRIC = {
    "IoU":       "tab-iou",
    "Dice":      "tab-dice-coef",
    "NSD":       "tab-normalized-surface-distance",
    "Epistemic": "tab-mean-epistemic-uncertainty",
    "Aleatoric": "tab-mean-aleatoric-uncertainty",
}
REQUIRED_COLS = {"dataset", "arch", "mean", "hdi_lo", "hdi_hi"}

COLOR_SPINE = "#333333"


def metric_label(metric: str) -> str:
    if metric == "Epistemic":
        return r"Uncertainty$_\mathrm{epist.}$"
    if metric == "Aleatoric":
        return r"Uncertainty$_\mathrm{alea.}$"
    return metric


def normalize_label(label: str) -> str:
    s = str(label).strip()
    s = s.replace("Terramind",  "TM")
    s = s.replace("Planetscope", "PS").replace("PlanetScope", "PS")
    s = s.replace("Sentinel-2", "S2").replace("Sentinel 2", "S2").replace("Sentinel2", "S2")
    s = s.replace("Aerial", "AE")
    return s


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────
def _read_tab_html(html_path: str | Path, tab_id: str) -> str:
    raw = Path(html_path).read_text(encoding="utf-8", errors="replace")
    from lxml import html as lhtml
    root = lhtml.fromstring(raw)
    node = root.get_element_by_id(tab_id)
    return lhtml.tostring(node, encoding="unicode")


def extract_predicted_means_table(html_path: str | Path, metric: str) -> pd.DataFrame:
    tab_id   = TAB_BY_METRIC[metric]
    tab_html = _read_tab_html(html_path, tab_id)
    tables   = pd.read_html(StringIO(tab_html))
    for t in tables:
        cols = {str(c).strip() for c in t.columns}
        if REQUIRED_COLS <= cols:
            df = t.copy()
            df.columns = [str(c).strip() for c in df.columns]
            for c in ["mean", "hdi_lo", "hdi_hi"]:
                df[c] = pd.to_numeric(df[c], errors="coerce")
            df["dataset"] = df["dataset"].astype(str).map(normalize_label)
            df["arch"]    = df["arch"].astype(str).map(normalize_label)
            return df.reset_index(drop=True)
    raise ValueError(f"Could not find predicted-means table in tab {tab_id!r}.")


def load_all_metrics(html_path: str | Path) -> Dict[str, pd.DataFrame]:
    return {m: extract_predicted_means_table(html_path, m) for m in METRICS}


def infer_arch_order(all_tables: Dict[str, pd.DataFrame]) -> List[str]:
    arches = sorted(set().union(*[set(df["arch"].unique()) for df in all_tables.values()]))
    return [a for a in ARCH_ORDER_PREFERRED if a in arches] + \
           [a for a in arches if a not in ARCH_ORDER_PREFERRED]


def infer_sensor_order(all_tables: Dict[str, pd.DataFrame]) -> List[str]:
    sensors = sorted(set().union(*[set(df["dataset"].unique()) for df in all_tables.values()]))
    return [s for s in SENSORS_PREFERRED if s in sensors] + \
           [s for s in sensors if s not in SENSORS_PREFERRED]


def subset_metric_sensor(df: pd.DataFrame, sensor: str, arch_order: List[str]) -> pd.DataFrame:
    sub = df[df["dataset"].astype(str) == sensor].copy()
    sub["arch"] = pd.Categorical(sub["arch"], categories=arch_order, ordered=True)
    sub = sub.sort_values("arch").reset_index(drop=True)
    sub = sub.set_index("arch").reindex(pd.Index(arch_order, name="arch")).reset_index()
    return sub


# ─────────────────────────────────────────────────────────────────────────────
# Draw metrics onto a single axis
# ─────────────────────────────────────────────────────────────────────────────
def plot_metrics_into_axis(
    ax,
    metric_subs: Dict[str, pd.DataFrame],
    arch_order:  List[str],
    metrics_to_plot: List[str],
    ylim: Tuple[float, float],
    show_xticks: bool,
    is_leftmost: bool,
):
    x_base  = np.arange(len(arch_order))
    M       = len(metrics_to_plot)
    width   = 0.11 if M == 2 else 0.07
    offsets = {m: (i - (M - 1) / 2.0) * width for i, m in enumerate(metrics_to_plot)}

    for m in metrics_to_plot:
        sub = metric_subs.get(m)
        if sub is None or sub.empty:
            continue
        x   = x_base + offsets[m]
        y   = np.asarray(sub["mean"].values,   dtype=float)
        lo  = np.asarray(sub["hdi_lo"].values, dtype=float)
        hi  = np.asarray(sub["hdi_hi"].values, dtype=float)
        # clip only performance metrics to [0,1]
        if m in PERF_METRICS:
            y  = np.clip(y,  0.0, 1.0)
            lo = np.clip(lo, 0.0, 1.0)
            hi = np.clip(hi, 0.0, 1.0)
        yerr = np.vstack([y - lo, hi - y])
        # Slightly extend whiskers for visibility
        if m in PERF_METRICS:
            whisker_pad = 0.02 * (ylim[1] - ylim[0])
        else:
            whisker_pad = 0.02 * (ylim[1] - ylim[0])
        yerr = yerr + whisker_pad
        c    = METRIC_COLORS[m]

        # Draw whiskers first (behind marker), then marker on top.
        ax.errorbar(
            x, y, yerr=yerr,
            fmt="none",
            capsize=0,
            elinewidth=0.9,
            color=c,
            zorder=2,
        )
        # Connecting line between points
        ax.plot(x, y, "-", linewidth=1.0, color=c, alpha=0.90, zorder=2)
        # Marker on top (white edge so whisker peeks out around it)
        ax.plot(x, y, "o", markersize=4.5,
                color=c,
                markeredgecolor="white", markeredgewidth=1.2,
                zorder=4)

    ax.set_xlim(-0.5, len(arch_order) - 0.5)
    ax.set_xticks(x_base)
    if show_xticks:
        ax.set_xticklabels(arch_order, fontsize=6)
    else:
        ax.set_xticklabels([])

    ax.set_ylim(ylim)
    
    # Apply styling
    for side in ["left", "right", "top", "bottom"]:
        ax.spines[side].set_linewidth(0.7)
        ax.spines[side].set_color(COLOR_SPINE)

    ax.tick_params(axis="x", labelsize=6, length=3, width=0.7, pad=1)
    ax.tick_params(axis="y", labelsize=6.5, length=3, width=0.7, pad=1,
                   left=True, right=False, labelleft=is_leftmost, direction="out")

    ax.yaxis.set_major_locator(MaxNLocator(nbins=4, prune="both"))
    fmt = ScalarFormatter(useMathText=True)
    fmt.set_scientific(False)
    fmt.set_useOffset(False)
    ax.yaxis.set_major_formatter(fmt)

    ax.grid(True, axis="y", alpha=0.18, linewidth=0.4, linestyle="--")
    ax.set_axisbelow(True)

    ax.set_ylabel("")

    # x-label: bottom row only
    if show_xticks:
        ax.set_xlabel("Architecture", fontsize=7, labelpad=4)
    else:
        ax.set_xlabel("")


# ─────────────────────────────────────────────────────────────────────────────
# Legend — matches Fig 4/5 bottom-center format
# ─────────────────────────────────────────────────────────────────────────────
def add_legend(fig, n_sensors: int):
    handles = [
        Line2D([0], [0],
               color=METRIC_COLORS[m],
               lw=1.8,
               marker="none",        # line only in legend
               label=metric_label(m))
        for m in METRICS
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=len(METRICS),
        frameon=False,
        fontsize=6.5,
        handlelength=2.2,
        columnspacing=1.0,
        handletextpad=0.5,
        bbox_to_anchor=(0.52, -0.035),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main figure
# ─────────────────────────────────────────────────────────────────────────────
def create_figure(all_tables: Dict[str, pd.DataFrame], outpath: str | Path):
    arch_order   = infer_arch_order(all_tables)
    sensor_order = infer_sensor_order(all_tables)
    N_SENSORS    = len(sensor_order)

    # Prepare per-sensor metric subsets
    per_sensor: Dict[str, Dict[str, pd.DataFrame]] = {
        s: {m: subset_metric_sensor(all_tables[m], s, arch_order) for m in METRICS}
        for s in sensor_order
    }

    # ---- Figure geometry ----
    FIG_W_CM  = 16.0
    FIG_H_CM  = 9.5
    LEGEND_FRAC = 0.08    # fraction of figure height reserved for legend

    fig = plt.figure(figsize=(FIG_W_CM / 2.54, FIG_H_CM / 2.54), dpi=300)

    # Create 2x3 grid: 2 rows (performance/uncertainty), 3 columns (sensors)
    gs = GridSpec(
        2, N_SENSORS,
        height_ratios=[1.0, 0.6],  # performance metrics get more space
        wspace=0.1,
        hspace=0.1,
        left=0.12, right=0.99,
        top=0.94,
        bottom=LEGEND_FRAC + 0.08,
        figure=fig,
    )

    # Create all axes
    axes = np.empty((2, N_SENSORS), dtype=object)
    for row in range(2):
        for col in range(N_SENSORS):
            axes[row, col] = fig.add_subplot(gs[row, col])

    # Plot performance metrics (top row)
    for j, sensor in enumerate(sensor_order):
        ax = axes[0, j]
        
        # Sensor title on top row
        ax.set_title(
            sensor, fontsize=8, fontweight="bold", pad=5, color="#222222"
        )
        
        plot_metrics_into_axis(
            ax=ax,
            metric_subs=per_sensor[sensor],
            arch_order=arch_order,
            metrics_to_plot=PERF_METRICS,
            ylim=SEG_TOP,
            show_xticks=False,
            is_leftmost=(j == 0),
        )

    # Plot uncertainty metrics (bottom row)
    for j, sensor in enumerate(sensor_order):
        ax = axes[1, j]
        
        plot_metrics_into_axis(
            ax=ax,
            metric_subs=per_sensor[sensor],
            arch_order=arch_order,
            metrics_to_plot=UNC_METRICS,
            ylim=SEG_BOT,
            show_xticks=True,
            is_leftmost=(j == 0),
        )

    # ---- Row labels ----
    fig.text(
        0.02, 0.72,
        "Performance\nmetrics",
        va="center", ha="center",
        rotation=90,
        fontsize=7.5,
        fontweight="bold",
        color=COLOR_SPINE,
    )
    
    fig.text(
        0.02, 0.32,
        "Uncertainties",
        va="center", ha="center",
        rotation=90,
        fontsize=7.5,
        fontweight="bold",
        color=COLOR_SPINE,
    )

    add_legend(fig, N_SENSORS)

    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)

    plot_export_frames = []
    for sensor in sensor_order:
        for metric in METRICS:
            sub = per_sensor[sensor][metric]
            if sub is None or sub.empty:
                continue
            plot_export_frames.append(sub.assign(sensor=sensor, metric=metric))
    plot_export_df = (
        pd.concat(plot_export_frames, ignore_index=True)
        if plot_export_frames else pd.DataFrame(columns=["sensor", "metric", "arch"])
    )
    out_csv = outpath.parent / "Figure_6_data.csv"
    plot_export_df.to_csv(out_csv, index=False)
    print(f"Saved → {out_csv}")

    for suffix in [".pdf", ".png"]:   # always save both formats
        p = outpath.with_suffix(suffix)
        fig.savefig(p, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"Saved → {p}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "report",
        nargs="?",
        default="/isipd/projects/p_planetdw/git/DriftwoodMappingBenchmark/analysis/stats_report.html",
    )
    ap.add_argument("--out", default="./figs/out/Figure_6_architecture_dependence.pdf")
    args = ap.parse_args()

    tables = load_all_metrics(args.report)
    create_figure(tables, Path(args.out))


if __name__ == "__main__":
    main()