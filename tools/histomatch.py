#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import rasterio
import traceback
import numpy as np

# --------------------------
# Paths (edit as needed)
# --------------------------
REF_IMG_DIR = '/isipd/projects/p_planetdw/data/methods_test/training_images/S2'
MACS_DIR   = '/isipd/projects/p_planetdw/data/methods_test/training_images/MACS'
PS_DIR     = '/isipd/projects/p_planetdw/data/methods_test/training_images/PS'
OUTPUT_DIR = '/isipd/projects/p_planetdw/data/methods_test/training_images/histomatch_output'


# --------------------------
# Core: robust histogram matching (no spatial size assumption)
# --------------------------
def histogram_match(input_img, ref_img, nodata_val=0):
    """
    Histogram-match input_img to ref_img per band.
    No-data (== nodata_val) and NaNs are ignored and preserved.

    Parameters
    ----------
    input_img : np.ndarray  (bands, rows, cols)  [n_bands can differ from ref_img, but we require equal band counts]
    ref_img   : np.ndarray  (bands, rows, cols)
    nodata_val : numeric

    Returns
    -------
    np.ndarray (float64) same shape as input_img
    """
    if input_img.ndim != 3 or ref_img.ndim != 3:
        raise ValueError("Expected input and reference arrays with 3 dims: (bands, rows, cols).")

    if input_img.shape[0] != ref_img.shape[0]:
        raise ValueError(
            f"Band mismatch: input has {input_img.shape[0]} bands, ref has {ref_img.shape[0]} bands."
        )

    # Work in float for safety; do not mutate callers' arrays
    input_img = input_img.astype(np.float64, copy=True)
    ref_img   = ref_img.astype(np.float64, copy=True)

    out_img = np.full_like(input_img, nodata_val, dtype=np.float64)

    for band in range(input_img.shape[0]):
        src = input_img[band]
        ref = ref_img[band]

        # Valid where not NaN and not nodata
        src_valid = (~np.isnan(src)) & (src != nodata_val)
        ref_valid = (~np.isnan(ref)) & (ref != nodata_val)

        if not np.any(src_valid):
            # Nothing to do on this band; leave as nodata
            continue

        if not np.any(ref_valid):
            raise ValueError("Reference basemap has only nodata for this scene (band {}).".format(band + 1))

        # Unique values and CDF for input (valid pixels only)
        src_vals, src_counts = np.unique(src[src_valid], return_counts=True)
        src_cdf = np.cumsum(src_counts, dtype=np.float64)
        src_cdf /= src_cdf[-1]

        # Unique values and CDF for reference (valid pixels only)
        ref_vals, ref_counts = np.unique(ref[ref_valid], return_counts=True)
        ref_cdf = np.cumsum(ref_counts, dtype=np.float64)
        ref_cdf /= ref_cdf[-1]

        # Map: input value -> input CDF -> reference value
        ref_vals_for_src = np.interp(src_cdf, ref_cdf, ref_vals)

        # Apply LUT onto all valid pixels
        matched_band_vals = np.interp(src[src_valid], src_vals, ref_vals_for_src)

        # Compose output band (preserve nodata)
        out_band = np.full_like(src, nodata_val, dtype=np.float64)
        out_band[src_valid] = matched_band_vals
        out_img[band] = out_band

    return out_img


# --------------------------
# Sentinel-2 helpers: pick B2,B3,B4,B8
# --------------------------
_BAND_NAME_RE = re.compile(r'\bB(\d{1,2}A?)\b', re.IGNORECASE)

def _find_s2_band_indices_by_description(descriptions, targets=('B2','B3','B4','B8')):
    """
    Try to map band descriptions to indices (0-based) for specified targets.
    Accepts variants like 'B02', 'B2', 'Band 2', 'S2_B2', etc.
    """
    if not descriptions or all(d is None for d in descriptions):
        return None  # no descriptions to parse

    # Build map normalized_name -> index
    name_to_idx = {}
    for idx, desc in enumerate(descriptions):
        if not desc:
            continue
        desc_low = desc.lower()
        # Heuristics: search for B<number> or "band <number>"
        m = _BAND_NAME_RE.search(desc)
        norm = None
        if m:
            norm = 'B' + m.group(1).upper().lstrip('0')  # e.g., 'B02' -> 'B2', 'B8A' -> 'B8A'
        else:
            # fallback: look for 'band ' + number
            m2 = re.search(r'band\s*([0-9]{1,2}a?)', desc_low)
            if m2:
                norm = 'B' + m2.group(1).upper().lstrip('0')
        if norm:
            name_to_idx[norm] = idx

    out = []
    for t in targets:
        # normalized target without leading zero
        tnorm = 'B' + t[1:].upper().lstrip('0')
        if tnorm in name_to_idx:
            out.append(name_to_idx[tnorm])
        else:
            return None  # couldn't resolve all targets

    return tuple(out)


def select_s2_bgrnir(ref_arr, descriptions=None):
    """
    From a stacked Sentinel-2 array (bands, rows, cols), select B2, B3, B4, B8 in that order.
    If descriptions are available, use them; otherwise fall back to common S2 ordering:
    [B1,B2,B3,B4,B5,B6,B7,B8,B8A,B9,B11,B12] -> indices [1,2,3,7]
    """
    n_bands = ref_arr.shape[0]

    # Try description-based mapping first
    idxs = _find_s2_band_indices_by_description(descriptions, ('B2','B3','B4','B8')) if descriptions else None
    if idxs is None:
        # Fallback: typical 12-band S2 stack order (no B10)
        # 1-based: 1..12 => [B1,B2,B3,B4,B5,B6,B7,B8,B8A,B9,B11,B12]
        # 0-based indices for B2,B3,B4,B8 -> 1,2,3,7
        guess = (1, 2, 3, 7)
        if max(guess) >= n_bands:
            raise ValueError(
                f"Cannot select S2 B2,B3,B4,B8 by fallback indices from {n_bands} bands. "
                "Provide proper band descriptions or a known band order."
            )
        idxs = guess

    # Slice and return as new array
    return ref_arr[np.array(idxs), ...]


# --------------------------
# Pairing logic (same as before, with minor safety)
# --------------------------
def list_tifs(folder):
    return [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(('.tif', '.tiff'))]


def find_matching_ref_image(ref_path, img_path, img_type='MACS'):
    """
    Finds matching pairs between reference (Sentinel-2) and input images.

    Reference key: basename.split('_')[0]
    MACS key:      basename.split('_')[1]
    PS key:        basename.split('_')[2]
    """
    ref_paths = list_tifs(ref_path)
    ref_names = [os.path.basename(i).split('_')[0] for i in ref_paths]

    print(f'Found {len(ref_names)} reference images in {ref_path}')

    img_paths = list_tifs(img_path)
    if img_type.upper() == 'MACS':
        img_names = [os.path.basename(i).split('_')[1] if len(os.path.basename(i).split('_')) > 1 else '' for i in img_paths]
    elif img_type.upper() == 'PS':
        img_names = [os.path.basename(i).split('_')[2] if len(os.path.basename(i).split('_')) > 2 else '' for i in img_paths]
    else:
        raise ValueError("img_type must be 'MACS' or 'PS'.")

    print(f'Found {len(img_names)} input images in {img_path}')

    matched_ref_imgs = {}
    for idx, name in enumerate(img_names):
        if name in ref_names:
            matched_ref_imgs[img_paths[idx]] = ref_paths[ref_names.index(name)]
            print(f'\n >> Found pair:\n    {img_paths[idx]}\n    {ref_paths[ref_names.index(name)]}')
        else:
            print(f'\n No matching reference image found for {img_paths[idx]}')

    print('\n' + '=' * 40)
    print(f'Summary of matched images for {img_type}:')
    print('=' * 40)
    for k, v in matched_ref_imgs.items():
        print(f'Input image:     {os.path.basename(k)}')
        print(f'Reference image: {os.path.basename(v)}\n')

    return matched_ref_imgs


# --------------------------
# I/O helpers
# --------------------------
def read_raster_with_meta(path):
    with rasterio.open(path) as ds:
        arr = ds.read()  # (bands, rows, cols)
        profile = ds.profile
        nodata = ds.nodata
        descriptions = tuple(ds.descriptions) if ds.descriptions else None
    return arr, profile, nodata, descriptions


def normalise_ref_nodata(ref_img, ref_nodata, target_nodata):
    """
    Return a copy of ref_img where ref_nodata is replaced by target_nodata.
    If ref_nodata is None or equals target_nodata, returns original array.
    """
    if ref_nodata is None or ref_nodata == target_nodata:
        return ref_img
    ref_img = ref_img.copy()
    ref_img[ref_img == ref_nodata] = target_nodata
    return ref_img


def write_raster(path, arr, profile, nodata):
    """
    Write array to raster with the same datatype and compression settings as input.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    def scale_array(arr, nodata, dtype):
        """
        Scales the array and ensures that the values match the appropriate dtype.
        """
        if dtype == np.uint16:
            # Scaling for uint16
            arr_scaled = np.full_like(arr, 0, dtype=np.uint16)
            nodata_uint16 = 0
            valid_mask = arr != nodata
            if np.any(valid_mask):
                arr_valid = arr[valid_mask]
                arr_min = np.min(arr_valid)
                arr_max = np.max(arr_valid)
                if arr_max != arr_min:
                    scale = 65535.0 / (arr_max - arr_min)
                    arr_scaled[valid_mask] = ((arr_valid - arr_min) * scale).astype(np.uint16)
            return arr_scaled, nodata_uint16
        elif dtype == np.float32:
            # For floating point, no scaling needed, just return
            return arr, nodata
        else:
            raise ValueError(f"Unsupported dtype {dtype} for scaling.")

    # Scale the array based on the required output type (in this case, uint16 or float32)
    arr_scaled, nodata_scaled = scale_array(arr, nodata, profile['dtype'])

    # Copy the original profile and adjust the parameters
    prof = profile.copy()
    prof.update(
        dtype=profile['dtype'],
        nodata=nodata_scaled,
        count=arr_scaled.shape[0],
        compress='lzw',  # LZW compression
        predictor=2,     # Horizontal differencing for better compression with continuous data
        tiled=True,      # Use tiled storage for better compression
        blockxsize=256,
        blockysize=256
    )
    with rasterio.open(path, 'w', **prof) as dst:
        dst.write(arr_scaled)


# --------------------------
# Processing driver
# --------------------------
def process_pairs(pairs_dict, label):
    if not pairs_dict:
        print(f'No {label} pairs to process.')
        return

    print(f'\nProcessing {len(pairs_dict)} {label} pair(s)...\n')

    for input_img_path, ref_img_path in pairs_dict.items():
        try:
            # Read input (MACS/PS = 4-band B,G,R,NIR expected)
            input_img, in_profile, in_nodata, _ = read_raster_with_meta(input_img_path)
            if in_nodata is None:
                in_nodata = 0  # fallback; adjust if your data uses a different no-data

            # Read reference (S2 stacked)
            ref_img_full, _, ref_nodata, ref_desc = read_raster_with_meta(ref_img_path)
            # Select S2 B2,B3,B4,B8 to match B,G,R,NIR
            ref_img_s4 = select_s2_bgrnir(ref_img_full, descriptions=ref_desc)

            # Align nodata values
            ref_img_s4 = normalise_ref_nodata(ref_img_s4, ref_nodata, in_nodata)

            # Safety: ensure both are 4-band
            if input_img.shape[0] != ref_img_s4.shape[0]:
                raise ValueError(
                    f"After S2 selection, band mismatch persists: input {input_img.shape[0]} vs ref {ref_img_s4.shape[0]}"
                )

            # Histogram-match (spatial sizes may differ; that's fine)
            matched = histogram_match(input_img, ref_img_s4, nodata_val=in_nodata)

            # Output path with the same datatype and file size
            out_name = f'histmatch_{label.lower()}_{os.path.basename(input_img_path)}'
            out_path = os.path.join(OUTPUT_DIR, out_name)
            write_raster(out_path, matched, in_profile, in_nodata)

            print(f'YAY!!! Saved: {out_path}')

        except Exception as e:
            print(f'Error processing {input_img_path} and {ref_img_path}: {e}')
            traceback.print_exc()


# --------------------------
# Main
# --------------------------
def main():
    macs_pairs = find_matching_ref_image(REF_IMG_DIR, MACS_DIR, img_type='MACS')
    ps_pairs   = find_matching_ref_image(REF_IMG_DIR, PS_DIR,   img_type='PS')

    process_pairs(macs_pairs, label='MACS')
    process_pairs(ps_pairs,   label='PS')


if __name__ == "__main__":
    main()
