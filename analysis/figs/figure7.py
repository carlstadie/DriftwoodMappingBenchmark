"""
Figure 7a — 3x3 predicted vs observed area scatter (KDE + bias line)
Figure 7b — 9x9 pairwise area ratio heatmaps (median + IQR)

Outputs
-------
Figure_7a_predicted_vs_observed_area.pdf / .png
Figure_7b_pairwise_area_ratio_heatmap.pdf / .png

Both outputs match manuscript style (7 pt font, 0.6 pt spines, pdf.fonttype=42).
Run once: extracts areas, caches to area_raw.csv, writes both figures.
"""
from __future__ import annotations
import os, re, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy import stats
from scipy.stats import gaussian_kde, iqr

warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)

mpl.rcParams.update({
    "font.size": 7, "axes.titlesize": 8, "axes.labelsize": 7,
    "xtick.labelsize": 6, "ytick.labelsize": 6,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5,  "ytick.major.size": 2.5,
    "xtick.minor.width": 0.4, "ytick.minor.width": 0.4,
    "xtick.minor.size": 1.5,  "ytick.minor.size": 1.5,
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "figure.facecolor": "white", "axes.facecolor": "white",
    "font.family": "sans-serif",
})

# ── Paths ─────────────────────────────────────────────────────────────────────
RESULTS_ROOT  = Path(r"N:/isipd/projects/p_planetdw/data/methods_test/results")
TRAINING_GPKG = Path(r"N:/isipd/projects/p_planetdw/data/methods_test/training/training_areas.gpkg")
LABEL_GPKG = {
    "AE": Path(r"N:/isipd/projects/p_planetdw/data/methods_test/training/AE/labels_AE_indv.gpkg"),
    "PS": Path(r"N:/isipd/projects/p_planetdw/data/methods_test/training/PS/labels_PS.gpkg"),
    "S2": Path(r"N:/isipd/projects/p_planetdw/data/methods_test/training/S2/labels_S2_exp.gpkg"),
}

ARCH_MAP     = {"UNET": "U-Net", "SWIN": "Swin U-Net", "TERRAMIND": "TerraMind"}
SENSOR_COLOR = {"AE": "#2166AC", "PS": "#4DAC26", "S2": "#E08214"}
SENSOR_TITLE = {"AE": "AE", "PS": "PS", "S2": "S2"}
SENSORS      = ["AE", "PS", "S2"]
ARCHS        = ["UNET", "SWIN", "TERRAMIND"]
RUN_RE       = re.compile(r"^\d{8}-\d{4}_(\w+)x(\w+)$")
PRED_RE      = re.compile(r"^(\d+)\.tif$")

# Heatmap combo order: arch varies fastest within each sensor block
COMBOS = [
    ("UNET", "AE"), ("SWIN", "AE"), ("TERRAMIND", "AE"),
    ("UNET", "PS"), ("SWIN", "PS"), ("TERRAMIND", "PS"),
    ("UNET", "S2"), ("SWIN", "S2"), ("TERRAMIND", "S2"),
]
COMBO_LABELS = [
    "U-Net|AE", "Swin|AE", "TM|AE",
    "U-Net|PS", "Swin|PS", "TM|PS",
    "U-Net|S2", "Swin|S2", "TM|S2",
]
NC = len(COMBOS)


# ── Data extraction ───────────────────────────────────────────────────────────
def _pred_area_m2(p):
    with rasterio.open(p) as src:
        arr = src.read(1)
        px  = abs(src.transform.a * src.transform.e)
    return float(np.sum(arr > 0) * px)


def _label_area_m2(frame_id, sensor, training_areas, label_gdfs):
    ta = training_areas[training_areas.index == frame_id]
    if ta.empty:
        return np.nan
    ta_geom = ta.geometry.iloc[0]
    labels  = label_gdfs[sensor]
    if labels.crs != training_areas.crs:
        labels = labels.to_crs(training_areas.crs)
    clipped = labels[labels.intersects(ta_geom)].copy()
    if clipped.empty:
        return 0.0
    clipped["geometry"] = clipped.geometry.intersection(ta_geom)
    return float(clipped.geometry.area.sum())


def extract_areas():
    print("Loading training areas ...")
    training_areas = gpd.read_file(TRAINING_GPKG)
    if "fid" in training_areas.columns:
        training_areas = training_areas.set_index("fid")
    else:
        training_areas.index = training_areas.index.astype(int)
    print("Loading label geopackages ...")
    label_gdfs = {s: gpd.read_file(p) for s, p in LABEL_GPKG.items()}
    rows = []
    for arch in ARCHS:
        for sensor in SENSORS:
            combo_dir = RESULTS_ROOT / arch / sensor
            if not combo_dir.exists():
                continue
            run_dirs = sorted([
                d for d in combo_dir.iterdir()
                if d.is_dir() and RUN_RE.match(d.name)
            ])
            print(f"  {arch} x {sensor}: {len(run_dirs)} runs")
            for run_idx, run_dir in enumerate(run_dirs):
                pred_tifs = {
                    int(m.group(1)): run_dir / f.name
                    for f in run_dir.iterdir()
                    if (m := PRED_RE.match(f.name))
                }
                for frame_id, tif_path in pred_tifs.items():
                    try:
                        rows.append({
                            "arch": arch, "sensor": sensor,
                            "frame_id": frame_id, "run": run_idx,
                            "pred_area_m2":  _pred_area_m2(tif_path),
                            "label_area_m2": _label_area_m2(
                                frame_id, sensor, training_areas, label_gdfs),
                        })
                    except Exception as e:
                        print(f"    [warn] {arch}/{sensor}/{frame_id}: {e}")
    return pd.DataFrame(rows)


def aggregate(df):
    def _ci95(x):
        n = len(x.dropna())
        if n < 2:
            return np.nan
        return float(stats.t.ppf(0.975, df=n - 1) * stats.sem(x, nan_policy="omit"))
    return (
        df.groupby(["arch", "sensor", "frame_id"])
          .agg(
              pred_mean  =("pred_area_m2",  "mean"),
              pred_ci95  =("pred_area_m2",  _ci95),
              label_area =("label_area_m2", "mean"),
          )
          .reset_index()
    )


# ── KDE helper ────────────────────────────────────────────────────────────────
def _draw_kde(ax, lx, ly, color, n_grid=200):
    if len(lx) < 15:
        return
    try:
        kde    = gaussian_kde(np.vstack([lx, ly]), bw_method="scott")
        xg     = np.linspace(lx.min() - 0.3, lx.max() + 0.3, n_grid)
        yg     = np.linspace(ly.min() - 0.3, ly.max() + 0.3, n_grid)
        XX, YY = np.meshgrid(xg, yg)
        ZZ     = kde(np.vstack([XX.ravel(), YY.ravel()])).reshape(n_grid, n_grid)
        ZZ    /= ZZ.max()
        cmap   = mpl.colors.LinearSegmentedColormap.from_list(
            "mono", ["#ffffff", color], N=256)
        ax.contourf(10**xg, 10**yg, ZZ,
                    levels=np.linspace(0.05, 1.0, 10),
                    cmap=cmap, alpha=0.50, zorder=1)
        ax.contour(10**xg, 10**yg, ZZ,
                   levels=[0.30, 0.60],
                   colors=[color], linewidths=[0.35, 0.55],
                   alpha=0.75, zorder=2)
    except Exception as e:
        print(f"  [KDE warn] {e}")


# ── Figure 7a ─────────────────────────────────────────────────────────────────
def plot_fig7a(agg, out_dir):
    agg  = agg.copy()
    pos  = pd.concat([
        agg.loc[agg["pred_mean"]  > 0, "pred_mean"],
        agg.loc[agg["label_area"] > 0, "label_area"],
    ]).dropna()
    lo   = 10 ** (np.floor(np.log10(pos.min())) - 0.05)
    hi   = 10 ** (np.ceil( np.log10(pos.max())) + 0.05)
    diag = np.array([lo, hi])

    letters = "abcdefghi"
    FIG_W   = 19.0 / 2.54
    fig     = plt.figure(figsize=(FIG_W, FIG_W * 0.92), dpi=300)
    gs      = GridSpec(3, 3, figure=fig,
                       left=0.14, right=0.97,
                       top=0.93, bottom=0.11,
                       wspace=0.08, hspace=0.10)

    for i, arch in enumerate(ARCHS):
        for j, sensor in enumerate(SENSORS):
            ax    = fig.add_subplot(gs[i, j])
            color = SENSOR_COLOR[sensor]

            sub = agg[(agg["arch"] == arch) & (agg["sensor"] == sensor)].dropna(
                subset=["pred_mean", "label_area"])
            sub = sub[(sub["pred_mean"] > 0) & (sub["label_area"] > 0)]

            ax.set_xscale("log"); ax.set_yscale("log")
            ax.set_xlim(lo, hi);  ax.set_ylim(lo, hi)
            ax.set_aspect("equal")

            ax.plot(diag, diag, color="#BBBBBB", lw=0.9, ls="--", zorder=0)

            if len(sub) >= 15:
                _draw_kde(ax,
                          np.log10(sub["pred_mean"].values),
                          np.log10(sub["label_area"].values),
                          color)

            if not sub.empty:
                ax.scatter(sub["pred_mean"], sub["label_area"],
                           color=color, s=3, alpha=0.30,
                           edgecolors="none", zorder=4)

            ax.grid(which="major", color="#EEEEEE", lw=0.4, zorder=0)
            ax.grid(which="minor", color="#F5F5F5", lw=0.25, zorder=0)
            ax.set_axisbelow(True)
            for sp in ax.spines.values():
                sp.set_linewidth(0.6); sp.set_color("#333333")
            ax.tick_params(which="both", labelsize=6, length=2.5, width=0.6)

            if len(sub) > 2:
                lx = np.log10(sub["pred_mean"].values)
                ly = np.log10(sub["label_area"].values)
                ss_res = np.sum((lx - ly) ** 2)
                ss_tot = np.sum((lx - lx.mean()) ** 2)
                r2     = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
                rmse   = 10 ** np.sqrt(np.mean((lx - ly) ** 2))
                denom  = np.dot(lx, lx)
                slope  = np.dot(lx, ly) / denom if denom > 0 else np.nan

                if np.isfinite(slope):
                    bx = np.array([lo, hi])
                    by = 10 ** (np.log10(lo) + slope *
                                (np.log10(bx) - np.log10(lo)))
                    ax.plot(bx, by, color=color, lw=1.1, ls="-",
                            alpha=0.80, zorder=3)

                ax.text(0.97, 0.04,
                        f"R\u00b2 = {r2:.2f}\nRMSE = {rmse:.1f}\u00d7\n"
                        f"bias = {slope:.2f}\nn = {len(sub)}",
                        transform=ax.transAxes, fontsize=5.5,
                        va="bottom", ha="right", color="#222222",
                        bbox=dict(facecolor="white", edgecolor="none",
                                  alpha=0.85, pad=1.5),
                        zorder=6)

            ax.text(0.03, 0.97, f"({letters[i * 3 + j]})",
                    transform=ax.transAxes, fontsize=10,
                    fontweight="bold", va="top", ha="left",
                    color="#222222", zorder=6)

            if i == 0:
                ax.set_title(SENSOR_TITLE[sensor], fontsize=8,
                             fontweight="bold", pad=5, color="#222222")

            ax.yaxis.set_tick_params(labelleft=(j == 0))
            ax.xaxis.set_tick_params(labelbottom=(i == 2))

    # Left-margin labels anchored to panel geometry
    row_centres = [(gs[i, 0].get_position(fig).y0 +
                    gs[i, 0].get_position(fig).y1) / 2 for i in range(3)]
    panel_left  = gs[0, 0].get_position(fig).x0
    arch_x      = panel_left - 0.012
    ylab_x      = panel_left - 0.055

    for i, arch in enumerate(ARCHS):
        fig.text(arch_x, row_centres[i], ARCH_MAP[arch],
                 ha="center", va="center", fontsize=7,
                 fontweight="bold", color="#222222", rotation=90)

    fig.text(ylab_x, (row_centres[0] + row_centres[2]) / 2,
             "Observed area  (m\u00b2)",
             ha="center", va="center", fontsize=7,
             color="#222222", rotation=90)

    panel_mid_x = (gs[2, 0].get_position(fig).x0 +
                   gs[2, 2].get_position(fig).x1) / 2
    fig.text(panel_mid_x, 0.03, "Predicted area  (m\u00b2)",
             ha="center", va="center", fontsize=7, color="#222222")

    os.makedirs(out_dir, exist_ok=True)
    for ext in ("pdf", "png"):
        p = Path(out_dir) / f"Figure_7a_predicted_vs_observed_area.{ext}"
        plt.savefig(p, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"Saved -> {p}")
    plt.close(fig)


# ── Figure 7b ─────────────────────────────────────────────────────────────────
def _build_matrices(agg):
    agg = agg[(agg["pred_mean"] > 0) & (agg["label_area"] > 0)].copy()
    series = {}
    for arch, sensor in COMBOS:
        sub = agg[(agg["arch"] == arch) & (agg["sensor"] == sensor)]
        series[(arch, sensor)] = sub.set_index("frame_id")["pred_mean"]

    med_mat = np.full((NC, NC), np.nan)
    iqr_mat = np.full((NC, NC), np.nan)

    for i, ci in enumerate(COMBOS):
        for j, cj in enumerate(COMBOS):
            if i == j:
                med_mat[i, j] = 0.0
                iqr_mat[i, j] = 0.0
                continue
            si, sj = series[ci], series[cj]
            shared = si.index.intersection(sj.index)
            if len(shared) < 3:
                continue
            lr = np.log10(si.loc[shared].values / sj.loc[shared].values)
            med_mat[i, j] = np.nanmedian(lr)
            iqr_mat[i, j] = iqr(lr, nan_policy="omit")

    return med_mat, iqr_mat


def plot_fig7b(agg, out_dir):
    med_mat, iqr_mat = _build_matrices(agg)

    FIG_W   = 19.0 / 2.54
    fig, axes = plt.subplots(1, 2,
                             figsize=(FIG_W, FIG_W * 0.56),
                             dpi=300)
    fig.subplots_adjust(left=0.14, right=0.97, top=0.88,
                        bottom=0.24, wspace=0.40)

    def _draw_panel(ax, mat, cmap, vmin, vmax, title, cbar_label, fmt):
        im = ax.imshow(mat, cmap=cmap, vmin=vmin, vmax=vmax,
                       aspect="equal", interpolation="none")

        ax.set_xticks(range(NC))
        ax.set_xticklabels(COMBO_LABELS, rotation=45, ha="right", fontsize=6)
        ax.set_yticks(range(NC))
        ax.set_yticklabels(COMBO_LABELS, fontsize=6)

        for r in range(NC):
            for c in range(NC):
                val = mat[r, c]
                if np.isnan(val):
                    continue
                bg      = (val - vmin) / (vmax - vmin) if vmax != vmin else 0.5
                txt_col = "white" if (bg < 0.22 or bg > 0.78) else "#222222"
                ax.text(c, r, fmt(val),
                        ha="center", va="center",
                        fontsize=5, color=txt_col)

        for edge in [3, 6]:
            ax.axhline(edge - 0.5, color="white", lw=1.2)
            ax.axvline(edge - 0.5, color="white", lw=1.2)

        for sp in ax.spines.values():
            sp.set_linewidth(0.6); sp.set_color("#333333")
        ax.tick_params(which="both", labelsize=6, length=2.5, width=0.6)

        ax.set_title(title, fontsize=8, fontweight="bold",
                     pad=5, color="#222222")

        cb = fig.colorbar(im, ax=ax, orientation="horizontal",
                          pad=0.26, fraction=0.046, shrink=0.85)
        cb.set_label(cbar_label, fontsize=6.5, labelpad=2)
        cb.ax.tick_params(labelsize=6, length=2, width=0.5)
        for sp in cb.ax.spines.values():
            sp.set_linewidth(0.5)

        for lbl, centre in [("AE", 1), ("PS", 4), ("S2", 7)]:
            ax.text(centre, NC + 0.3, lbl,
                    ha="center", va="bottom",
                    fontsize=7, fontweight="bold",
                    color=SENSOR_COLOR[lbl],
                    transform=ax.transData, clip_on=False)

    abs_max = max(np.nanmax(np.abs(med_mat[~np.eye(NC, dtype=bool)])), 0.3)
    _draw_panel(
        axes[0], med_mat,
        cmap="RdBu_r", vmin=-abs_max, vmax=abs_max,
        title="(a)  Median area ratio",
        cbar_label="log\u2081\u2080(row / column)   [red = row predicts more]",
        fmt=lambda v: f"{v:+.2f}",
    )

    iqr_max = np.nanmax(iqr_mat[~np.eye(NC, dtype=bool)])
    _draw_panel(
        axes[1], iqr_mat,
        cmap="YlOrBr", vmin=0, vmax=iqr_max,
        title="(b)  Frame-level disagreement (IQR)",
        cbar_label="IQR of log\u2081\u2080(row / column)   [higher = less consistent]",
        fmt=lambda v: f"{v:.2f}",
    )

    os.makedirs(out_dir, exist_ok=True)
    for ext in ("pdf", "png"):
        p = Path(out_dir) / f"Figure_7b_pairwise_area_ratio_heatmap.{ext}"
        plt.savefig(p, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"Saved -> {p}")
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    OUT_DIR = Path("./figs/out")
    CACHE   = OUT_DIR / "area_raw.csv"

    if CACHE.exists():
        print(f"Loading cached raw data from {CACHE}")
        df = pd.read_csv(CACHE)
    else:
        df = extract_areas()
        os.makedirs(OUT_DIR, exist_ok=True)
        df.to_csv(CACHE, index=False)
        print(f"Saved raw data -> {CACHE}")

    agg = aggregate(df)
    agg.to_csv(OUT_DIR / "area_aggregated.csv", index=False)

    print("Plotting Figure 7a ...")
    plot_fig7a(agg, OUT_DIR)

    print("Plotting Figure 7b ...")
    plot_fig7b(agg, OUT_DIR)

    print("Done.")


if __name__ == "__main__":
    main()