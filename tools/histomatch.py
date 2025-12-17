#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Histogram matching pipeline (fast + memory-efficient), matching BOTH folders to S2.

Fix included:
- Uses `from rasterio.warp import transform_bounds` instead of `rasterio.warp.transform_bounds`
  to avoid: AttributeError: module 'rasterio' has no attribute 'warp'
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.windows import Window
from rasterio.warp import transform_bounds  # <-- FIX

try:
    from tqdm import tqdm
except Exception:
    tqdm = None


# =============================================================================
# PATHS (hard-coded)
# =============================================================================
REF_IMG_DIR = "/isipd/projects/p_planetdw/data/methods_test/training_images/S2"
MACS_DIR = "/isipd/projects/p_planetdw/data/methods_test/training_images/AE/to_encode"
PS_DIR = "/isipd/projects/p_planetdw/data/methods_test/training_images/PS"
OUTPUT_DIR = (
    "/isipd/projects/p_planetdw/data/methods_test/training_images/histomatch_output"
)

INPUT_SOURCES = [
    ("MACS", MACS_DIR),
    ("PS", PS_DIR),
]

# =============================================================================
# HISTOGRAM MATCH CONFIG
# =============================================================================
NODATA_OUT = 0.0

MAX_SAMPLE_PIXELS = 200_000
N_BINS = 4096
ROBUST_PERCENTILES = (0.5, 99.5)

RANDOM_STATE = 1234
WINDOW_SIZE = 512
MAX_WINDOWS = 256
READ_DOWNSAMPLE = 2

OUTPUT_DTYPE = "float32"
OUTPUT_SCALE = 1.0


# =============================================================================
# FILE DISCOVERY
# =============================================================================
def list_tifs(folder: str) -> List[Path]:
    base = Path(folder)
    if not base.exists():
        return []
    return sorted(list(base.rglob("*.tif")) + list(base.rglob("*.tiff")))


# =============================================================================
# SPATIAL HELPERS (overlap-based ref selection)
# =============================================================================
def _bbox_area(bounds: Tuple[float, float, float, float]) -> float:
    minx, miny, maxx, maxy = bounds
    return max(0.0, maxx - minx) * max(0.0, maxy - miny)


def _bbox_intersection(
    b1: Tuple[float, float, float, float],
    b2: Tuple[float, float, float, float],
) -> Tuple[float, float, float, float]:
    minx1, miny1, maxx1, maxy1 = b1
    minx2, miny2, maxx2, maxy2 = b2

    ix1 = max(minx1, minx2)
    iy1 = max(miny1, miny2)
    ix2 = min(maxx1, maxx2)
    iy2 = min(maxy1, maxy2)
    return ix1, iy1, ix2, iy2


def _intersection_ratio(
    b1: Tuple[float, float, float, float],
    b2: Tuple[float, float, float, float],
) -> float:
    inter = _bbox_area(_bbox_intersection(b1, b2))
    a1 = _bbox_area(b1)
    if a1 <= 0:
        return 0.0
    return inter / a1


def bounds_4326(path: Path) -> Tuple[float, float, float, float]:
    """
    Returns dataset bounds in EPSG:4326.
    """
    with rasterio.open(path) as ds:
        b = ds.bounds
        if ds.crs and ds.crs.to_string() != "EPSG:4326":
            left, bottom, right, top = transform_bounds(
                ds.crs,
                "EPSG:4326",
                b.left,
                b.bottom,
                b.right,
                b.top,
                densify_pts=21,
            )
            return left, bottom, right, top
        return b.left, b.bottom, b.right, b.top


def find_best_s2_ref_by_overlap(
    input_path: Path,
    ref_paths: List[Path],
    min_intersection_ratio: float = 0.2,
) -> Tuple[Optional[Path], Optional[Tuple[float, float, float, float]]]:
    ib = bounds_4326(input_path)

    best_ref: Optional[Path] = None
    best_ratio = 0.0
    best_overlap: Optional[Tuple[float, float, float, float]] = None

    for rp in ref_paths:
        rb = bounds_4326(rp)
        ratio = _intersection_ratio(ib, rb)
        if ratio > best_ratio:
            best_ratio = ratio
            best_ref = rp
            best_overlap = _bbox_intersection(ib, rb)

    if best_ref is None or best_ratio < min_intersection_ratio:
        return None, None

    if best_overlap is None or _bbox_area(best_overlap) <= 0:
        return None, None

    return best_ref, best_overlap


def overlap_window_in_ref(
    ref_ds: rasterio.DatasetReader,
    overlap_bbox_4326: Tuple[float, float, float, float],
) -> Optional[Window]:
    if ref_ds.crs is None:
        return None

    left, bottom, right, top = overlap_bbox_4326
    try:
        l2, b2, r2, t2 = transform_bounds(
            "EPSG:4326",
            ref_ds.crs,
            left,
            bottom,
            right,
            top,
            densify_pts=21,
        )
        win = rasterio.windows.from_bounds(l2, b2, r2, t2, transform=ref_ds.transform)
        win = win.round_offsets().round_lengths()
        win = win.intersection(Window(0, 0, ref_ds.width, ref_ds.height))
        if win.width <= 1 or win.height <= 1:
            return None
        return win
    except Exception:
        return None


# =============================================================================
# S2 BAND RULE: BGRNIR = B2,B3,B4,B8 (fallback to 2,3,4,8)
# =============================================================================
def s2_bgrnir_band_indices(ref_ds: rasterio.DatasetReader) -> List[int]:
    wanted = ["B2", "B3", "B4", "B8"]
    desc = ref_ds.descriptions

    if desc:
        upper = [(d or "").strip().upper() for d in desc]
        idxs: List[int] = []
        ok = True
        for w in wanted:
            if w in upper:
                idxs.append(upper.index(w) + 1)
            else:
                ok = False
                break
        if ok:
            return idxs

    return [2, 3, 4, 8]


# =============================================================================
# FAST HISTOGRAM MATCH: sampling + binned CDF + block apply
# =============================================================================
def _valid_mask(arr: np.ndarray, nodata: Optional[float]) -> np.ndarray:
    mask = np.isfinite(arr)
    if nodata is not None:
        mask &= (arr != nodata)
    return mask


def _sample_band_values(
    ds: rasterio.DatasetReader,
    band_1based: int,
    nodata: Optional[float],
    max_pixels: int,
    window_size: int,
    max_windows: int,
    read_downsample: int,
    seed: int,
    sample_domain: Optional[Window] = None,
) -> np.ndarray:
    rng = np.random.default_rng(seed)

    if sample_domain is None:
        domain = Window(0, 0, ds.width, ds.height)
    else:
        domain = sample_domain

    if domain.width <= 1 or domain.height <= 1:
        domain = Window(0, 0, ds.width, ds.height)

    vals_list: List[np.ndarray] = []
    total = 0

    for _ in range(max_windows):
        if total >= max_pixels:
            break

        max_row = max(0, int(domain.height) - 1)
        max_col = max(0, int(domain.width) - 1)
        if max_row == 0 or max_col == 0:
            break

        dr = int(rng.integers(0, max_row))
        dc = int(rng.integers(0, max_col))

        r0 = int(domain.row_off + dr)
        c0 = int(domain.col_off + dc)

        r1 = min(int(domain.row_off + domain.height), r0 + window_size)
        c1 = min(int(domain.col_off + domain.width), c0 + window_size)

        win = Window(col_off=c0, row_off=r0, width=c1 - c0, height=r1 - r0)
        if win.width <= 1 or win.height <= 1:
            continue

        if read_downsample and read_downsample > 1:
            out_h = max(1, int(win.height // read_downsample))
            out_w = max(1, int(win.width // read_downsample))
            block = ds.read(
                band_1based,
                window=win,
                out_shape=(out_h, out_w),
                resampling=Resampling.nearest,
            ).astype(np.float32, copy=False)
        else:
            block = ds.read(band_1based, window=win).astype(np.float32, copy=False)

        mask = _valid_mask(block, nodata)
        if not mask.any():
            continue

        v = block[mask]
        if v.size == 0:
            continue

        remaining = max_pixels - total
        if v.size > remaining:
            idx = rng.choice(v.size, remaining, replace=False)
            v = v[idx]

        vals_list.append(v)
        total += v.size

    if not vals_list:
        return np.empty((0,), dtype=np.float32)

    vals = np.concatenate(vals_list).astype(np.float32, copy=False)
    return vals[:max_pixels]


def _cdf_from_hist(vals: np.ndarray, vmin: float, vmax: float, n_bins: int):
    hist, edges = np.histogram(vals, bins=n_bins, range=(vmin, vmax))
    if hist.sum() <= 0:
        return None, None
    cdf = np.cumsum(hist, dtype=np.float64)
    cdf /= cdf[-1]
    centers = (edges[:-1] + edges[1:]) * 0.5
    return cdf, centers


def histogram_match_raster(
    input_path: Path,
    ref_path: Path,
    overlap_bbox_4326: Tuple[float, float, float, float],
    output_path: Path,
    input_bands: List[int],
    ref_bands: List[int],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(input_path) as src, rasterio.open(ref_path) as ref:
        src_nodata = src.nodata
        ref_nodata = ref.nodata

        ref_domain = overlap_window_in_ref(ref, overlap_bbox_4326)

        luts: List[np.ndarray] = []
        vmins: List[float] = []
        vmaxs: List[float] = []

        for i, (ib, rb) in enumerate(zip(input_bands, ref_bands)):
            src_vals = _sample_band_values(
                ds=src,
                band_1based=ib,
                nodata=src_nodata,
                max_pixels=MAX_SAMPLE_PIXELS,
                window_size=WINDOW_SIZE,
                max_windows=MAX_WINDOWS,
                read_downsample=READ_DOWNSAMPLE,
                seed=RANDOM_STATE + i * 101,
                sample_domain=None,
            )
            ref_vals = _sample_band_values(
                ds=ref,
                band_1based=rb,
                nodata=ref_nodata,
                max_pixels=MAX_SAMPLE_PIXELS,
                window_size=WINDOW_SIZE,
                max_windows=MAX_WINDOWS,
                read_downsample=READ_DOWNSAMPLE,
                seed=RANDOM_STATE + i * 101 + 1,
                sample_domain=ref_domain,
            )

            if src_vals.size == 0:
                raise ValueError(f"No valid samples in input band {ib}")

            if ref_vals.size == 0:
                # fallback to full ref sampling
                ref_vals = _sample_band_values(
                    ds=ref,
                    band_1based=rb,
                    nodata=ref_nodata,
                    max_pixels=MAX_SAMPLE_PIXELS,
                    window_size=WINDOW_SIZE,
                    max_windows=MAX_WINDOWS,
                    read_downsample=READ_DOWNSAMPLE,
                    seed=RANDOM_STATE + i * 101 + 2,
                    sample_domain=None,
                )
                if ref_vals.size == 0:
                    raise ValueError(f"No valid samples in ref band {rb}")

            lo, hi = ROBUST_PERCENTILES
            src_lo, src_hi = np.percentile(src_vals, [lo, hi])
            ref_lo, ref_hi = np.percentile(ref_vals, [lo, hi])

            vmin = float(min(src_lo, ref_lo))
            vmax = float(max(src_hi, ref_hi))
            if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
                raise ValueError("Bad vmin/vmax from samples.")

            src_cdf, centers = _cdf_from_hist(src_vals, vmin, vmax, N_BINS)
            ref_cdf, _ = _cdf_from_hist(ref_vals, vmin, vmax, N_BINS)
            if src_cdf is None or ref_cdf is None:
                raise ValueError("Failed to build histograms (empty counts).")

            lut = np.interp(src_cdf, ref_cdf, centers).astype(np.float32)
            luts.append(lut)
            vmins.append(vmin)
            vmaxs.append(vmax)

        profile = src.profile.copy()
        profile.update(
            dtype=OUTPUT_DTYPE,
            count=len(input_bands),
            nodata=NODATA_OUT,
            tiled=True,
            compress=profile.get("compress", "deflate"),
            bigtiff="if_safer",
        )

        windows = list(src.block_windows(1))
        iterator: Iterable = windows
        if tqdm is not None:
            iterator = tqdm(windows, desc=f"histmatch {input_path.name}", total=len(windows))

        with rasterio.open(output_path, "w", **profile) as dst:
            for _, win in iterator:
                block = src.read(indexes=input_bands, window=win).astype(np.float32, copy=False)
                out_block = np.full(block.shape, NODATA_OUT, dtype=np.float32)

                for bi in range(block.shape[0]):
                    vals = block[bi]
                    mask = _valid_mask(vals, src_nodata)
                    if not mask.any():
                        continue

                    vmin = vmins[bi]
                    vmax = vmaxs[bi]
                    lut = luts[bi]
                    scale = (N_BINS - 1) / (vmax - vmin)

                    vv = vals[mask]
                    idx = ((vv - vmin) * scale).astype(np.int32)
                    np.clip(idx, 0, N_BINS - 1, out=idx)

                    out_block[bi][mask] = lut[idx]

                if OUTPUT_SCALE != 1.0:
                    out_block *= float(OUTPUT_SCALE)

                dst.write(out_block.astype(OUTPUT_DTYPE, copy=False), window=win)


# =============================================================================
# MAIN: process both folders against S2
# =============================================================================
def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    ref_paths = list_tifs(REF_IMG_DIR)
    if not ref_paths:
        raise FileNotFoundError(f"No S2 .tif found in: {REF_IMG_DIR}")

    print(f"[refs] {len(ref_paths)} S2 file(s) in {REF_IMG_DIR}")
    print(f"[out]  {OUTPUT_DIR}")

    for source_name, source_dir in INPUT_SOURCES:
        input_paths = list_tifs(source_dir)
        if not input_paths:
            print(f"[skip] No inputs found in {source_dir}")
            continue

        out_subdir = Path(OUTPUT_DIR) / source_name
        out_subdir.mkdir(parents=True, exist_ok=True)

        print(f"\n[{source_name}] {len(input_paths)} file(s) in {source_dir}")

        for inp in input_paths:
            ref, overlap = find_best_s2_ref_by_overlap(
                inp, ref_paths, min_intersection_ratio=0.2
            )
            if ref is None or overlap is None:
                print(f"[skip] {source_name}: no overlapping S2 ref for {inp.name}")
                continue

            input_bands = [1, 2, 3, 4]

            with rasterio.open(ref) as ref_ds:
                ref_bands = s2_bgrnir_band_indices(ref_ds)

            out_path = out_subdir / f"{inp.stem}_histmatch.tif"
            print(f"\n[match] {source_name}: {inp.name} <-ref- {ref.name}")
            print(f"        input_bands={input_bands} ref_bands={ref_bands}")

            try:
                histogram_match_raster(
                    input_path=inp,
                    ref_path=ref,
                    overlap_bbox_4326=overlap,
                    output_path=out_path,
                    input_bands=input_bands,
                    ref_bands=ref_bands,
                )
                print(f"[done]  wrote: {out_path}")
            except Exception as exc:
                print(f"[ERROR] {source_name}: {inp.name}: {exc}")


if __name__ == "__main__":
    main()
