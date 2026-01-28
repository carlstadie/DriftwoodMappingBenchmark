"""
End-to-end script:

1) Aggregate (sum) numeric-named GeoTIFFs across subfolders into <root>/summed/<name>.tif
   while preserving georeferencing metadata (CRS/transform/etc.) via rasterio.

2) For VIZ_NAMES, create a 3x4 figure:
   - cols: UNET, SWIN, TERRAMIND, LABEL
   - rows: AE, PS, S2
   Each row uses the *respective* training image as background (RGB from band order).
   Overlays use **magma** (NO normalization; fixed 0..10 scale).
   Overlay opacity is binary: counts==0 => fully transparent, counts>0 => fully opaque.
   The 4th column shows the **last band (label)** from training data, **binarized**,
   displayed with **viridis**, overlayed on RGB.

Band expectations (Label last):
  - AE:  B,G,R,NIR,...,Label
  - PS:  B,G,R,NIR,...,Label
  - S2:  Coastal,B,G,R,...,Label

Dependencies:
  pip install rasterio tifffile numpy matplotlib tqdm
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import tifffile as tiff

try:
    import rasterio
except ImportError as e:
    raise ImportError(
        "rasterio is required to preserve CRS/transform metadata. Install with: pip install rasterio"
    ) from e

try:
    from tqdm.auto import tqdm
except ImportError as e:
    raise ImportError("tqdm is required for progress bars. Install with: pip install tqdm") from e


# -------------------------
# USER CONFIG
# -------------------------
RESULTS_DIR = r"N:\isipd\projects\p_planetdw\data\methods_test\results"

UNET_AE_DIR = os.path.join(RESULTS_DIR, r"UNET\AE")
UNET_PS_DIR = os.path.join(RESULTS_DIR, r"UNET\PS")
UNET_S2_DIR = os.path.join(RESULTS_DIR, r"UNET\S2")

SWIN_AE_DIR = os.path.join(RESULTS_DIR, r"Swin\AE")
SWIN_PS_DIR = os.path.join(RESULTS_DIR, r"Swin\PS")
SWIN_S2_DIR = os.path.join(RESULTS_DIR, r"Swin\S2")

TERRAMIND_AE_DIR = os.path.join(RESULTS_DIR, r"TERRAMIND\AE")
TERRAMIND_PS_DIR = os.path.join(RESULTS_DIR, r"TERRAMIND\PS")
TERRAMIND_S2_DIR = os.path.join(RESULTS_DIR, r"TERRAMIND\S2")

TRAINING_IMAGE_DIR_AE = r"N:\isipd\projects\p_planetdw\data\methods_test\preprocessed\AE"
TRAINING_IMAGE_DIR_PS = r"N:\isipd\projects\p_planetdw\data\methods_test\preprocessed\20260109-1434_UNETxPS"
TRAINING_IMAGE_DIR_S2 = r"N:\isipd\projects\p_planetdw\data\methods_test\preprocessed\20260108-1335_UNETxS2"

VIZ_NAMES = [
    "34","60","6","173","63","74","218","126","347","325","167","290","36","123",
    "324","193","236","182","72","367","220","183","25","340","132","56","238",
    "113","85","245","242","42","213","55","209","386","84","97","64","131",
    "354","272","357","111","164","109","375","346","5","11","196","89","119",
    "259","200","273","199","32","395","298","234","394","227","302","388","289",
    "293","158","237","360","390","291","296","189","161","153","83","70","23","239"
]


# -------------------------
# STYLE SETTINGS
# -------------------------
# Overlay rendering (NO normalization)
OVERLAY_CMAP = "cividis"  # "magma" "plasma" "inferno" "cividis"
OVERLAY_VMIN = 0.0
OVERLAY_VMAX = 10.0
OVERLAY_CLIP = True

# Binary opacity mask threshold (counts > eps => opaque)
OVERLAY_OPAQUE_EPS = 0.0

ADD_CONTOURS = False

# Label col
LABEL_THRESHOLD = 0.0
LABEL_ALPHA = 0.75

# Debug
DEBUG_OVERLAY_STATS = False
DEBUG_LABEL_STATS = False


# -------------------------
# Helpers
# -------------------------
def _find_tiff_by_stem(folder: Path, stem: str) -> Optional[Path]:
    for ext in (".tif", ".tiff", ".TIF", ".TIFF"):
        p = folder / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def _bands_first(arr: np.ndarray) -> bool:
    """Heuristic: (bands, H, W) vs (H, W, bands)."""
    if arr.ndim != 3:
        raise ValueError("Expected 3D array.")
    return arr.shape[0] <= 16 and arr.shape[0] < arr.shape[1] and arr.shape[0] < arr.shape[2]


def _normalize_rgb(rgb: np.ndarray) -> np.ndarray:
    """Percentile stretch to 0..1 per channel (NaN-safe)."""
    rgb = rgb.astype(np.float32, copy=False)
    out = np.empty_like(rgb, dtype=np.float32)
    for c in range(3):
        ch = rgb[..., c]
        lo = np.nanpercentile(ch, 2)
        hi = np.nanpercentile(ch, 98)
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            out[..., c] = 0.0
        else:
            out[..., c] = np.clip((ch - lo) / (hi - lo), 0, 1)
    return out


def _rgb_indices_for_method(method: str) -> Tuple[int, int, int]:
    """
    Returns (B_idx, G_idx, R_idx) in the band axis.
      AE/PS: B,G,R,...,Label => (0,1,2)
      S2:    Coastal,B,G,R,...,Label => (1,2,3)
    """
    method = method.upper()
    if method in ("AE", "PS"):
        return (0, 1, 2)
    if method == "S2":
        return (1, 2, 3)
    raise ValueError(f"Unknown method: {method}")


def _aggregation_done_for_names(root: Path, names: List[str], summed_folder_name: str = "summed") -> bool:
    """Aggregation considered done if root/summed has files for all names."""
    summed_dir = root / summed_folder_name
    if not summed_dir.exists() or not summed_dir.is_dir():
        return False
    for nm in names:
        if not str(nm).isdigit():
            continue
        if _find_tiff_by_stem(summed_dir, str(nm)) is None:
            return False
    return True


# -------------------------
# GeoTIFF Summation (preserve CRS/transform)
# -------------------------
def sum_numeric_named_geotiffs_across_subfolders(
    root_dir: str | Path,
    output_folder_name: str = "summed",
    *,
    require_all_subfolders: bool = True,
    overwrite: bool = True,
    keep_bands: bool = False,
    dtype_out: str = "float32",
    compress: str = "DEFLATE",
    show_progress: bool = True,
    verbose: bool = True,
) -> Path:
    root = Path(root_dir).resolve()
    out_dir = root / output_folder_name
    out_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"\n[SUM-GEO] Root: {root}")

    if not root.exists():
        raise FileNotFoundError(f"Missing root: {root}")

    subfolders = [p for p in root.iterdir() if p.is_dir() and p != out_dir]

    if not subfolders:
        direct_numeric = [
            f for f in root.iterdir()
            if f.is_file() and f.suffix.lower() in (".tif", ".tiff") and f.stem.isdigit()
        ]
        if direct_numeric:
            if verbose:
                print(f"[SUM-GEO][WARN] No subfolders found. Found {len(direct_numeric)} numeric TIFFs in root.")
                print(f"[SUM-GEO][WARN] Skipping summation (treat as already final). Output folder: {out_dir}")
            return out_dir
        raise ValueError(f"No subfolders found under: {root}")

    if verbose:
        print(f"[SUM-GEO] Found {len(subfolders)} subfolders.")

    per_folder: List[Dict[str, Path]] = []
    folder_iter = tqdm(subfolders, desc=f"Scanning ({root.name})", disable=not show_progress)
    for sf in folder_iter:
        m: Dict[str, Path] = {}
        for f in sf.iterdir():
            if not f.is_file():
                continue
            if f.suffix.lower() not in (".tif", ".tiff"):
                continue
            if not f.stem.isdigit():
                continue
            m[f.stem] = f
        per_folder.append(m)
        folder_iter.set_postfix_str(f"{sf.name}: {len(m)} numeric")

    if require_all_subfolders:
        common = set(per_folder[0].keys())
        for m in per_folder[1:]:
            common &= set(m.keys())
        names = sorted(common, key=lambda s: int(s))
        if verbose:
            print(f"[SUM-GEO] Common numeric names across ALL subfolders: {len(names)}")
    else:
        union = set()
        for m in per_folder:
            union |= set(m.keys())
        names = sorted(union, key=lambda s: int(s))
        if verbose:
            print(f"[SUM-GEO] Union numeric names across subfolders: {len(names)}")

    if not names:
        raise ValueError(f"No numeric TIFF names found under {root}.")

    written = 0
    skipped = 0

    name_iter = tqdm(names, desc=f"Summing+Writing ({root.name})", disable=not show_progress)
    for name in name_iter:
        out_path = out_dir / f"{name}.tif"
        if out_path.exists() and not overwrite:
            skipped += 1
            continue

        acc = None
        ref_profile = None
        used_files = 0

        for m in per_folder:
            if name not in m:
                if require_all_subfolders:
                    raise FileNotFoundError(f"Missing {name}.tif in one subfolder under {root}")
                else:
                    continue

            p = m[name]
            with rasterio.open(p) as src:
                data = src.read(masked=True).filled(0)  # (bands, H, W)
                data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

                if ref_profile is None:
                    ref_profile = src.profile.copy()

                if not keep_bands and data.shape[0] > 1:
                    data = data.sum(axis=0, keepdims=True)  # (1, H, W)

                if acc is None:
                    acc = np.zeros_like(data, dtype=np.float64)
                else:
                    if data.shape != acc.shape:
                        raise ValueError(
                            f"Shape mismatch for '{name}' under {root}: got {data.shape}, expected {acc.shape}"
                        )

                acc += data.astype(np.float64, copy=False)
                used_files += 1

        if acc is None or ref_profile is None:
            continue

        out_profile = ref_profile.copy()
        out_profile.update(
            driver="GTiff",
            dtype=dtype_out,
            count=acc.shape[0],
            compress=compress,
            nodata=None,
        )

        out_data = acc.astype(dtype_out, copy=False)
        with rasterio.open(out_path, "w", **out_profile) as dst:
            dst.write(out_data)

        written += 1
        name_iter.set_postfix_str(f"wrote {written} (used {used_files})")

    if verbose:
        print(f"[SUM-GEO] Done. Wrote {written} GeoTIFF(s), skipped {skipped} -> {out_dir}")

    return out_dir


# -------------------------
# Training + overlay loading
# -------------------------
def load_training_rgb_and_label(training_tif: Path, method: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load preprocessed training image -> rgb (H,W,3) + label (H,W, last band)."""
    arr = tiff.imread(str(training_tif))
    if arr.ndim != 3:
        raise ValueError(f"Training image must be multi-band: {training_tif}")

    if _bands_first(arr):
        label = arr[-1]
        b_idx, g_idx, r_idx = _rgb_indices_for_method(method)
        B, G, R = arr[b_idx], arr[g_idx], arr[r_idx]
    else:
        label = arr[..., -1]
        b_idx, g_idx, r_idx = _rgb_indices_for_method(method)
        B, G, R = arr[..., b_idx], arr[..., g_idx], arr[..., r_idx]

    label = np.nan_to_num(label, nan=0.0, posinf=0.0, neginf=0.0)
    rgb = _normalize_rgb(np.stack([R, G, B], axis=-1))
    return rgb, label


def load_overlay_counts_0_10(overlay_path: Path) -> np.ndarray:
    """
    Load overlay as raw counts.
    - Sum bands if multi-band
    - Replace NaN/inf with 0
    - Clip to [0,10]
    """
    with rasterio.open(overlay_path) as src:
        data = src.read(masked=True).filled(0)  # (bands, H, W)

    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

    x = data.sum(axis=0) if data.shape[0] > 1 else data[0]
    x = x.astype(np.float32, copy=False)

    if OVERLAY_CLIP:
        x = np.clip(x, OVERLAY_VMIN, OVERLAY_VMAX)

    return x


def draw_overlay_counts(ax, counts: np.ndarray, *, add_contours: bool = True):
    """
    Draw raw counts using fixed vmin/vmax (0..10).
    Opacity is binary:
      counts <= eps => transparent
      counts >  eps => fully opaque
    """
    # 0 for zero, 1 for nonzero
    alpha_mask = (counts > OVERLAY_OPAQUE_EPS).astype(np.float32)

    im = ax.imshow(
        counts,
        cmap=OVERLAY_CMAP,
        alpha=alpha_mask,
        vmin=OVERLAY_VMIN,
        vmax=OVERLAY_VMAX,
    )

    if add_contours:
        levels = [1, 2, 4, 6, 8, 10]
        levels = [lv for lv in levels if OVERLAY_VMIN < lv < OVERLAY_VMAX]
        if levels:
            ax.contour(counts, levels=levels, colors="white", linewidths=0.7, alpha=0.9)
    
    return im


# -------------------------
# Plotting (3x4: UNET, SWIN, TERRAMIND, LABEL)
# -------------------------
def plot_3x4_overlays_with_label_col(
    name: str,
    training_dirs: Dict[str, Path],
    summed_dirs: Dict[Tuple[str, str], Path],
    out_dir: str | Path,
    *,
    verbose: bool = True,
    add_contours: bool = True,
) -> Path:
    name = str(name)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    models = ["UNET", "SWIN", "TERRAMIND"]
    methods = ["AE", "PS", "S2"]
    col_titles = ["UNET", "SWIN", "TERRAMIND", "LABEL (bin)"]

    # Load row backgrounds and labels
    row_rgb: Dict[str, Optional[np.ndarray]] = {}
    row_label: Dict[str, Optional[np.ndarray]] = {}

    for method in methods:
        tdir = training_dirs[method]
        tpath = _find_tiff_by_stem(tdir, name)
        if tpath is None:
            row_rgb[method] = None
            row_label[method] = None
            if verbose:
                print(f"[PLOT][WARN] Missing training image for method={method}, name={name} in {tdir}")
            continue

        try:
            rgb, lab = load_training_rgb_and_label(tpath, method=method)
            row_rgb[method] = rgb
            row_label[method] = lab

            if DEBUG_LABEL_STATS:
                lb = (lab > LABEL_THRESHOLD).astype(np.uint8)
                print(
                    f"[DEBUG][LABEL] method={method} name={name} "
                    f"lab_min={float(lab.min()):.3g} lab_max={float(lab.max()):.3g} bin_nz={int(lb.sum())}"
                )

        except Exception as ex:
            row_rgb[method] = None
            row_label[method] = None
            if verbose:
                print(f"[PLOT][WARN] Failed to load training rgb/label for method={method}, name={name}: {ex}")

    fig, axes = plt.subplots(3, 4, figsize=(16, 12), constrained_layout=True)

    # Track one imshow object for creating colorbar
    colorbar_im = None

    for r, method in enumerate(methods):
        bg = row_rgb[method]
        lab = row_label[method]

        # cols 0..2 overlays
        for c, model in enumerate(models):
            ax = axes[r, c]
            ax.set_xticks([])
            ax.set_yticks([])

            if bg is None:
                ax.text(
                    0.5, 0.5, "missing\ntraining",
                    transform=ax.transAxes, ha="center", va="center",
                    fontsize=12,
                    bbox=dict(facecolor="white", alpha=0.85, edgecolor="none"),
                )
            else:
                ax.imshow(bg)

            root = summed_dirs[(model, method)]
            summed_folder = root / "summed"

            overlay_path = None
            if summed_folder.exists():
                overlay_path = _find_tiff_by_stem(summed_folder, name)
            if overlay_path is None:
                overlay_path = _find_tiff_by_stem(root, name)

            if overlay_path is None:
                ax.text(
                    0.5, 0.12, "missing\noverlay",
                    transform=ax.transAxes, ha="center", va="center",
                    fontsize=10,
                    bbox=dict(facecolor="white", alpha=0.75, edgecolor="none"),
                )
            else:
                try:
                    counts = load_overlay_counts_0_10(overlay_path)

                    if DEBUG_OVERLAY_STATS:
                        nz_pct = 100.0 * float((counts > 0).mean())
                        print(
                            f"[DEBUG][OVL] {model}/{method} name={name} "
                            f"min={float(counts.min()):.3g} max={float(counts.max()):.3g} nz%={nz_pct:.3f} "
                            f"file={overlay_path}"
                        )

                    im = draw_overlay_counts(ax, counts, add_contours=add_contours)
                    if colorbar_im is None:
                        colorbar_im = im

                except Exception as ex:
                    ax.text(
                        0.5, 0.12, "overlay\nerror",
                        transform=ax.transAxes, ha="center", va="center",
                        fontsize=10,
                        bbox=dict(facecolor="white", alpha=0.75, edgecolor="none"),
                    )
                    if verbose:
                        print(f"[PLOT][WARN] Overlay load failed for {model}/{method}, name={name}: {ex}")

            if r == 0:
                ax.set_title(col_titles[c], fontsize=14)
            if c == 0:
                ax.set_ylabel(method, fontsize=14)

        # col 3: label binarized
        axL = axes[r, 3]
        axL.set_xticks([])
        axL.set_yticks([])

        if bg is None or lab is None:
            axL.text(
                0.5, 0.5, "missing\nlabel",
                transform=axL.transAxes, ha="center", va="center",
                fontsize=12,
                bbox=dict(facecolor="white", alpha=0.85, edgecolor="none"),
            )
        else:
            axL.imshow(bg)
            lab_bin = (lab > LABEL_THRESHOLD).astype(np.uint8)
            axL.imshow(
                lab_bin,
                cmap="viridis",
                vmin=0,
                vmax=1,
                alpha=LABEL_ALPHA,
                interpolation="nearest",
            )

        if r == 0:
            axL.set_title(col_titles[3], fontsize=14)

    fig.suptitle(f"Predictions for Frame {name}", fontsize=16)
    
    # Add colorbar for the overlay counts
    if colorbar_im is not None:
        cbar = fig.colorbar(colorbar_im, ax=axes[:, :3].ravel().tolist(), 
                           orientation='horizontal', pad=0.02, shrink=0.6, aspect=30)
        cbar.set_label('Count', fontsize=12)
    
    out_path = out_dir / f"{name}_overlay_3x4_with_labels.png"
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    return out_path


# -------------------------
# Main runner
# -------------------------
def run_all(
    *,
    show_progress: bool = True,
    verbose: bool = True,
    do_aggregation: bool = True,
    skip_aggregation_if_done: bool = True,
    aggregation_check_names: Optional[List[str]] = None,
    add_contours: bool = True,
):
    if aggregation_check_names is None:
        aggregation_check_names = VIZ_NAMES

    result_roots = [
        Path(UNET_AE_DIR), Path(UNET_PS_DIR), Path(UNET_S2_DIR),
        Path(SWIN_AE_DIR), Path(SWIN_PS_DIR), Path(SWIN_S2_DIR),
        Path(TERRAMIND_AE_DIR), Path(TERRAMIND_PS_DIR), Path(TERRAMIND_S2_DIR),
    ]

    if verbose:
        print("\n=== CONFIG ===")
        print(f"do_aggregation={do_aggregation}")
        print(f"skip_aggregation_if_done={skip_aggregation_if_done}")
        print(f"aggregation_check_names={aggregation_check_names}")
        print(f"OVERLAY_CMAP={OVERLAY_CMAP} vmin/vmax={OVERLAY_VMIN}/{OVERLAY_VMAX} clip={OVERLAY_CLIP}")
        print(f"OVERLAY_OPAQUE_EPS={OVERLAY_OPAQUE_EPS} (counts>eps => opaque)")
        print(f"LABEL_THRESHOLD={LABEL_THRESHOLD} LABEL_ALPHA={LABEL_ALPHA}")
        print(f"DEBUG_OVERLAY_STATS={DEBUG_OVERLAY_STATS} DEBUG_LABEL_STATS={DEBUG_LABEL_STATS}")
        print("==============\n")

    # Step 1: aggregation
    if do_aggregation:
        if verbose:
            print("=== STEP 1: SUM GEO-TIFFS (preserve CRS/transform) ===")

        roots_iter = tqdm(result_roots, desc="Result roots (agg)", disable=not show_progress)
        for root in roots_iter:
            roots_iter.set_postfix_str(root.name)

            if not root.exists():
                print(f"[SUM-GEO][WARN] Missing result dir: {root}")
                continue

            if skip_aggregation_if_done and _aggregation_done_for_names(root, aggregation_check_names):
                if verbose:
                    print(f"[SUM-GEO] Skipping (already done for {aggregation_check_names}): {root / 'summed'}")
                continue

            try:
                sum_numeric_named_geotiffs_across_subfolders(
                    root,
                    output_folder_name="summed",
                    require_all_subfolders=False,
                    overwrite=True,
                    keep_bands=False,
                    dtype_out="float32",
                    compress="DEFLATE",
                    show_progress=show_progress,
                    verbose=verbose,
                )
            except Exception as ex:
                print(f"[SUM-GEO][ERROR] {root} -> {type(ex).__name__}: {ex}")
    else:
        if verbose:
            print("=== STEP 1: SKIPPED (do_aggregation=False) ===")

    # Step 2: plotting
    if verbose:
        print("\n=== STEP 2: PLOT 3x4 (OVERLAYS + LABEL COL) ===")

    summed_dirs = {
        ("UNET", "AE"): Path(UNET_AE_DIR),
        ("UNET", "PS"): Path(UNET_PS_DIR),
        ("UNET", "S2"): Path(UNET_S2_DIR),

        ("SWIN", "AE"): Path(SWIN_AE_DIR),
        ("SWIN", "PS"): Path(SWIN_PS_DIR),
        ("SWIN", "S2"): Path(SWIN_S2_DIR),

        ("TERRAMIND", "AE"): Path(TERRAMIND_AE_DIR),
        ("TERRAMIND", "PS"): Path(TERRAMIND_PS_DIR),
        ("TERRAMIND", "S2"): Path(TERRAMIND_S2_DIR),
    }

    training_dirs = {
        "AE": Path(TRAINING_IMAGE_DIR_AE),
        "PS": Path(TRAINING_IMAGE_DIR_PS),
        "S2": Path(TRAINING_IMAGE_DIR_S2),
    }

    viz_out = Path(RESULTS_DIR) / "viz_overlays_3x4"
    viz_iter = tqdm(VIZ_NAMES, desc="VIZ names (plot)", disable=not show_progress)
    for nm in viz_iter:
        viz_iter.set_postfix_str(f"name={nm}")
        try:
            out_png = plot_3x4_overlays_with_label_col(
                name=nm,
                training_dirs=training_dirs,
                summed_dirs=summed_dirs,
                out_dir=viz_out,
                verbose=verbose,
                add_contours=add_contours,
            )
            print(f"[PLOT] Wrote: {out_png}")
        except Exception as ex:
            print(f"[PLOT][ERROR] name={nm} -> {type(ex).__name__}: {ex}")

    if verbose:
        print("\nDone.")


if __name__ == "__main__":
    run_all(
        show_progress=True,
        verbose=True,
        do_aggregation=True,
        skip_aggregation_if_done=True,
        aggregation_check_names=VIZ_NAMES,
        add_contours=ADD_CONTOURS,
    )
