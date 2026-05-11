

from __future__ import annotations
import re as _re
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import tifffile as tiff
import rasterio

# ─────────────────────────────────────────────────────────────────────────────
# rcParams — manuscript style
# ─────────────────────────────────────────────────────────────────────────────
mpl.rcParams.update({
    "font.size":          7,
    "axes.titlesize":     8,
    "axes.labelsize":     7,
    "axes.linewidth":     0.6,
    "xtick.major.width":  0.6,
    "ytick.major.width":  0.6,
    "pdf.fonttype":       42,
    "ps.fonttype":        42,
    "figure.facecolor":   "white",
    "axes.facecolor":     "white",
    "font.family":        "sans-serif",
})

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
RESULTS_DIR = Path(r"/isipd/projects/p_planetdw/data/methods_test/results")

SUMMED_ROOTS = {
    ("UNET",      "AE"): RESULTS_DIR / "UNET"      / "AE",
    ("UNET",      "PS"): RESULTS_DIR / "UNET"      / "PS",
    ("UNET",      "S2"): RESULTS_DIR / "UNET"      / "S2",
    ("SWIN",      "AE"): RESULTS_DIR / "SWIN"      / "AE",
    ("SWIN",      "PS"): RESULTS_DIR / "SWIN"      / "PS",
    ("SWIN",      "S2"): RESULTS_DIR / "SWIN"      / "S2",
    ("TERRAMIND", "AE"): RESULTS_DIR / "TERRAMIND" / "AE",
    ("TERRAMIND", "PS"): RESULTS_DIR / "TERRAMIND" / "PS",
    ("TERRAMIND", "S2"): RESULTS_DIR / "TERRAMIND" / "S2",
}

TRAINING_DIRS = {
    "AE": Path(r"/isipd/projects/p_planetdw/data/methods_test/preprocessed/AE"),
    "PS": Path(r"/isipd/projects/p_planetdw/data/methods_test/preprocessed/PS"),
    "S2": Path(r"/isipd/projects/p_planetdw/data/methods_test/preprocessed/S2"),
}

OUT_DIR  = Path("./figs/out")
FRAME_ID = "83"

# ─────────────────────────────────────────────────────────────────────────────
# Discrete colourmap — 10 bins (1 per run), perceptually uniform "viridis"
# ─────────────────────────────────────────────────────────────────────────────
# Viridis: colorblind-friendly, perceptually uniform, works in grayscale, standard in science
_VIRIDIS_10 = [
    "#440154",   # 1  run  — dark purple
    "#481567",   # 2
    "#3b528b",   # 3
    "#2d708e",   # 4
    "#238a8d",   # 5
    "#1f9f87",   # 6
    "#35b779",   # 7
    "#6ece58",   # 8
    "#b6de2b",   # 9
    "#fde724",   # 10 runs — bright yellow
]

N_BINS   = 10
# BoundaryNorm: boundaries at 0.5, 1.5, ..., 10.5 so each integer lands in its bin
_BOUNDS  = np.arange(0.5, N_BINS + 1.5, 1.0)     # [0.5, 1.5, ..., 10.5]
_CMAP    = mcolors.ListedColormap(_VIRIDIS_10, name="viridis_10")
_NORM    = mcolors.BoundaryNorm(_BOUNDS, _CMAP.N)

LABEL_COLOR = "#fde724"    # bright yellow from max viridis, high contrast against most backgrounds
LABEL_ALPHA = 0.88

ROW_LABELS = ["Aerial\n(10–15 cm)", "PlanetScope\n(3 m)", "Sentinel-2\n(10 m)"]
COL_LABELS = ["U-Net", "Swin U-Net", "TerraMind", "Ground truth"]
MODELS     = ["UNET", "SWIN", "TERRAMIND"]
SENSORS    = ["AE", "PS", "S2"]
RGB_BANDS  = {"AE": (0, 1, 2), "PS": (0, 1, 2), "S2": (1, 2, 3)}
LETTERS    = "abcdefghijkl"


# ─────────────────────────────────────────────────────────────────────────────
# IO helpers
# ─────────────────────────────────────────────────────────────────────────────
def _find(folder: Path, stem: str) -> Optional[Path]:
    for ext in (".tif", ".tiff", ".TIF", ".TIFF"):
        p = folder / f"{stem}{ext}"
        if p.exists():
            return p
    return None

def _bands_first(arr): return arr.ndim == 3 and arr.shape[0] <= 16 and arr.shape[0] < arr.shape[1]

def _norm_ch(ch):
    lo, hi = np.nanpercentile(ch, 2), np.nanpercentile(ch, 98)
    if not np.isfinite(lo) or hi <= lo: return np.zeros_like(ch, dtype=np.float32)
    return np.clip((ch.astype(np.float32) - lo) / (hi - lo), 0, 1)

def load_training(p: Path, sensor: str) -> Tuple[np.ndarray, np.ndarray]:
    arr = tiff.imread(str(p))
    if _bands_first(arr):
        label = arr[-1]; ri, gi, bi = RGB_BANDS[sensor]; R,G,B = arr[ri],arr[gi],arr[bi]
    else:
        label = arr[...,-1]; ri,gi,bi = RGB_BANDS[sensor]; R,G,B = arr[...,ri],arr[...,gi],arr[...,bi]
    label = np.nan_to_num(label.astype(np.float32), nan=0.0)
    return np.stack([_norm_ch(R), _norm_ch(G), _norm_ch(B)], axis=-1), label

def load_overlay(root: Path, stem: str) -> np.ndarray:
    p = _find(root / "summed", stem) or _find(root, stem)
    if p is None: raise FileNotFoundError(f"No overlay for '{stem}' in {root}")
    with rasterio.open(p) as src:
        data = src.read(masked=True).filled(0)
    data = np.nan_to_num(data, nan=0.0)
    counts = data.sum(axis=0) if data.shape[0] > 1 else data[0]
    return np.clip(counts.astype(np.float32), 0, N_BINS)


# ─────────────────────────────────────────────────────────────────────────────
# Panel helpers
# ─────────────────────────────────────────────────────────────────────────────
def _style_ax(ax):
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_linewidth(0.5); sp.set_color("#333333")

def _draw_prediction(ax, rgb, counts):
    ax.imshow(rgb, interpolation="bilinear")
    # Mask zero-count pixels fully transparent
    rgba = _CMAP(_NORM(counts))          # (H, W, 4)
    rgba[counts < 1, 3] = 0.0           # transparent where no prediction
    ax.imshow(rgba, interpolation="nearest")
    # Return a mappable for the colourbar
    return mpl.cm.ScalarMappable(cmap=_CMAP, norm=_NORM)

def _draw_label(ax, rgb, label):
    lum = 0.299*rgb[...,0] + 0.587*rgb[...,1] + 0.114*rgb[...,2]
    ax.imshow(lum, cmap="gray", vmin=0, vmax=1, interpolation="bilinear")
    mask = (label > 0).astype(np.float32)
    rgba = np.zeros((*mask.shape, 4), dtype=np.float32)
    rc, gc, bc = mcolors.to_rgb(LABEL_COLOR)
    rgba[..., 0] = rc; rgba[..., 1] = gc; rgba[..., 2] = bc
    rgba[..., 3] = mask * LABEL_ALPHA
    ax.imshow(rgba, interpolation="nearest")

def _letter(ax, letter):
    ax.text(0.022, 0.966, f"({letter})",
            transform=ax.transAxes, ha="left", va="top",
            fontsize=12, fontweight="bold", color="white", zorder=10)


# ─────────────────────────────────────────────────────────────────────────────
# Figure
# ─────────────────────────────────────────────────────────────────────────────
def make_figure(frame_id: str, out_dir: Path):

    # Load
    rgb, label, ovl = {}, {}, {}
    for sensor in SENSORS:
        p = _find(TRAINING_DIRS[sensor], frame_id)
        if p is None: raise FileNotFoundError(f"{TRAINING_DIRS[sensor]}/{frame_id}.tif")
        rgb[sensor], label[sensor] = load_training(p, sensor)
    for model in MODELS:
        for sensor in SENSORS:
            ovl[(model, sensor)] = load_overlay(SUMMED_ROOTS[(model, sensor)], frame_id)

    # ── Geometry (EO journal: max 19 cm wide, 300 dpi) ──────────────────────
    FIG_W_CM   = 19.0
    LABEL_W    = 1.10    # left row-label strip
    GAP        = 0.12    # inter-panel gap
    CBAR_H     = 0.28    # colourbar bar height
    CBAR_PAD   = 0.60    # distance from bottom panels to colourbar top
    BOT_PAD    = 0.50    # below colourbar
    TOP_PAD    = 0.22

    N_ROWS, N_COLS = 3, 4
    avail_w = FIG_W_CM - LABEL_W - GAP * (N_COLS + 1)
    pw = avail_w / N_COLS        # panel width = panel height (square)
    ph = pw

    FIG_H_CM = TOP_PAD + ph * N_ROWS + GAP * (N_ROWS - 1) + CBAR_PAD + CBAR_H + BOT_PAD

    def fx(x): return x / FIG_W_CM
    def fy(y): return y / FIG_H_CM

    fig = plt.figure(figsize=(FIG_W_CM / 2.54, FIG_H_CM / 2.54), dpi=300)

    row_bot = [FIG_H_CM - TOP_PAD - (r+1)*ph - r*GAP for r in range(N_ROWS)]
    col_left = [LABEL_W + GAP + c*(pw + GAP) for c in range(N_COLS)]

    # ── Panels ───────────────────────────────────────────────────────────────
    letter_idx = 0
    sm = None   # ScalarMappable for colourbar

    for r, sensor in enumerate(SENSORS):
        # Row label
        ax_rl = fig.add_axes([fx(GAP*0.3), fy(row_bot[r]),
                               fx(LABEL_W - GAP*0.3), fy(ph)])
        ax_rl.axis("off")
        ax_rl.text(0.5, 0.5, ROW_LABELS[r], transform=ax_rl.transAxes,
                   ha="center", va="center", fontsize=9,
                   fontweight="semibold", rotation=90, color="#222222")

        for c, model in enumerate(MODELS):
            ax = fig.add_axes([fx(col_left[c]), fy(row_bot[r]), fx(pw), fy(ph)])
            _style_ax(ax)
            _sm = _draw_prediction(ax, rgb[sensor], ovl[(model, sensor)])
            if sm is None: sm = _sm
            if r == 0:
                ax.set_title(COL_LABELS[c], fontsize=9, fontweight="bold",
                             pad=4, color="#222222")
            _letter(ax, LETTERS[letter_idx]); letter_idx += 1

        # Label column
        ax = fig.add_axes([fx(col_left[3]), fy(row_bot[r]), fx(pw), fy(ph)])
        _style_ax(ax)
        _draw_label(ax, rgb[sensor], label[sensor])
        if r == 0:
            ax.set_title(COL_LABELS[3], fontsize=9, fontweight="bold",
                         pad=4, color="#222222")
        _letter(ax, LETTERS[letter_idx]); letter_idx += 1

    # ── Colourbar — full figure width ────────────────────────────────────────
    # Spans from left edge of first image column to right edge of label column
    cbar_l = col_left[0]
    cbar_w = col_left[3] + pw - col_left[0]
    cbar_b = BOT_PAD

    ax_cb = fig.add_axes([fx(cbar_l), fy(cbar_b), fx(cbar_w), fy(CBAR_H)])

    cb = fig.colorbar(sm, cax=ax_cb, orientation="horizontal")
    cb.set_label("Number of positive predictions (out of 10)", fontsize=9, labelpad=3)
 
    # Discrete ticks at integer bin centres
    cb.set_ticks(np.arange(1, N_BINS + 1))
    cb.ax.tick_params(labelsize=6, length=2.0, width=0.6)
    cb.ax.set_xticklabels([str(i) for i in range(1, N_BINS + 1)])
    for sp in cb.ax.spines.values():
        sp.set_linewidth(0.5); sp.set_color("#333333")

    # ── Save ─────────────────────────────────────────────────────────────────
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        p = out_dir / f"Figure_8_prediction_overlay.{ext}"
        fig.savefig(p, dpi=300, facecolor="white", bbox_inches="tight")
        print(f"Saved → {p}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Run-folder summation (self-contained)
# ─────────────────────────────────────────────────────────────────────────────
_RUN_RE = _re.compile(r"^\d{8}-\d{4}_\w+$")

def _run_folders(root, n=10):
    return sorted([d for d in root.iterdir()
                   if d.is_dir() and _RUN_RE.match(d.name)],
                  key=lambda d: d.name[:13])[:n]

def build_summed_if_needed(combo_root, stems, n_runs=10):
    out_dir = combo_root / "summed"
    if out_dir.exists() and all(_find(out_dir, s) is not None for s in stems):
        print(f"  [skip] {combo_root.name}/summed complete"); return
    runs = _run_folders(combo_root, n_runs)
    if not runs: raise FileNotFoundError(f"No run folders in {combo_root}")
    print(f"  Summing {len(runs)} runs in {combo_root.name} ...")
    out_dir.mkdir(exist_ok=True)
    for stem in stems:
        out_path = out_dir / f"{stem}.tif"
        if out_path.exists(): continue
        acc = ref_profile = None
        for run in runs:
            p = _find(run, stem)
            if p is None: continue
            with rasterio.open(p) as src:
                data = src.read(masked=True).filled(0).astype(np.float32)
                if ref_profile is None: ref_profile = src.profile.copy()
            band = data.sum(axis=0) if data.shape[0] > 1 else data[0]
            acc = band if acc is None else acc + band
        if acc is None: print(f"    [warn] no data for {stem}"); continue
        ref_profile.update(driver="GTiff", dtype="float32", count=1,
                           compress="DEFLATE", nodata=None)
        with rasterio.open(out_path, "w", **ref_profile) as dst:
            dst.write(acc[np.newaxis, ...])


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Building summed/ folders if needed ...")
    for key, root in SUMMED_ROOTS.items():
        build_summed_if_needed(root, stems=[FRAME_ID])
    print("Rendering figure ...")
    make_figure(FRAME_ID, OUT_DIR)