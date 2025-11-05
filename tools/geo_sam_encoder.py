#!/usr/bin/env python3
"""
Encode S2 GeoTIFFs with GeoSAM after reordering bands to 9,3,2 (1-indexed).

- Reads each input .tif
- Extracts bands (9, 3, 2) in that order
- If float, scales to uint16 by *10000 (0..1 -> 0..10000), clamps to [0,65535]
- Writes a temporary 3-band GeoTIFF
- Runs geosam.ImageEncoder on the temporary file
- Normalizes 'res' values in output CSVs to numeric

Multiprocessing: one worker process per GPU in GPUS.
"""

import os
import re
import glob
import time
import shutil
import tempfile
from multiprocessing import Process, current_process

import numpy as np
import pandas as pd
import rasterio
from tqdm import tqdm

import torch
import geosam
from geosam import ImageEncoder


# =========================
# Config
# =========================
checkpoint_path = '/isipd/projects/p_planetdw/data/methods_test/training_images/SAM_checkpoints/sam_vit_l_0b3195.pth'
image_folder    = '/isipd/projects/p_planetdw/data/methods_test/training_images/S2/to_encode'
feature_dir     = '/isipd/projects/p_planetdw/data/methods_test/training_images/S2/encodings'

# Which GPUs to use (IDs as seen by nvidia-smi)
GPUS = [0]  # e.g., [0,1] for two GPUs

# GeoSAM encode params
ENCODE_STRIDE     = 512   # pixels (tile stride)
ENCODE_RESOLUTION = 10    # meters per pixel


# =========================
# Helpers
# =========================
def float32_to_uint16(image: np.ndarray) -> np.ndarray:
    """Convert float image (0..~1) to uint16 by multiplying by 10000 and clipping."""
    img = image.astype(np.float32, copy=False) * 10000.0
    np.clip(img, 0.0, 65535.0, out=img)
    return img.astype(np.uint16, copy=False)


def _parse_res_val(x):
    """Return a float from numbers like 10, '10', '(10,10)', '[10 10]', '10.0 m', else NaN."""
    if x is None:
        return np.nan
    if isinstance(x, (int, float, np.integer, np.floating)):
        return float(x)
    s = str(x)
    m = re.search(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', s)  # first numeric token
    return float(m.group(0)) if m else np.nan


def chunk_list(lst, n):
    """Split list `lst` into `n` roughly equal chunks."""
    if n <= 0:
        return [lst]
    k, m = divmod(len(lst), n)
    return [lst[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)]


def _reorder_to_932(src_path: str, out_dir: str) -> str:
    """
    Read bands 9,3,2 (1-indexed) from src_path, convert to uint16 if float, write temp 3-band GeoTIFF.
    Returns path to the new file.
    """
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(src_path))[0]
    out_path = os.path.join(out_dir, f"{base}_b9_3_2.tif")

    with rasterio.open(src_path) as src:
        if src.count < 9:
            raise RuntimeError(f"{src_path} has only {src.count} bands; need at least 9 for 9,3,2.")

        # Rasterio is 1-indexed for bands
        data = src.read((9, 3, 2))  # (3, H, W)

        # Decide dtype/output
        first_dtype = np.dtype(src.dtypes[0])
        if np.issubdtype(first_dtype, np.floating):
            data = float32_to_uint16(data)
            out_dtype = 'uint16'
        else:
            # Keep original integer type
            out_dtype = src.dtypes[0]

        profile = src.profile.copy()
        profile.update(count=3, dtype=out_dtype)

        # Some profiles may have incompatible photometric; set to RGB safely
        profile.pop("photometric", None)
        profile["photometric"] = "RGB"

        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(data)

    return out_path


# =========================
# Worker
# =========================
def encode_images_on_gpu(gpu_id: int, image_paths, tmp_dir: str):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available but a GPU worker was requested.")

    num_devices = torch.cuda.device_count()
    if gpu_id < 0 or gpu_id >= num_devices:
        raise RuntimeError(f"Requested GPU {gpu_id}, but only {num_devices} CUDA device(s) available.")

    torch.cuda.set_device(gpu_id)
    print(f"[{current_process().name}] Using GPU {gpu_id}: {torch.cuda.get_device_name(gpu_id)}")

    # Tell ImageEncoder which GPU to use
    img_encoder = ImageEncoder(checkpoint_path, gpu_id=gpu_id)

    for image_path in tqdm(image_paths, desc=f"GPU {gpu_id}"):
        try:
            start = time.time()

            # Reorder to bands 9,3,2 (1-indexed) and (if needed) convert to uint16
            tmp_img = _reorder_to_932(image_path, tmp_dir)

            # Encode
            img_encoder.encode_image(
                str(tmp_img),
                feature_dir,
                stride=ENCODE_STRIDE,
                resolution=ENCODE_RESOLUTION,
            )
            took = time.time() - start
            print(f"[GPU {gpu_id}] Encoded {os.path.basename(image_path)} in {took:.2f}s")

            # Uncomment to reclaim disk immediately
            # os.remove(tmp_img)

        except Exception as e:
            print(f"[GPU {gpu_id} ERROR] Failed on {image_path}: {e}")


# =========================
# Main
# =========================
if __name__ == "__main__":
    # Discover images
    image_paths = sorted(glob.glob(os.path.join(image_folder, "*.tif")))
    print(f"[INFO] Found {len(image_paths)} images in {image_folder}")
    print(f"[INFO] Target encodings dir: {feature_dir}")
    print(f"[INFO] GPUs requested: {GPUS}")

    if len(image_paths) == 0:
        raise SystemExit("[INFO] Nothing to do: no .tif images found.")

    if len(GPUS) == 0:
        raise RuntimeError("No CUDA GPUs specified in GPUS.")

    # Create a shared temporary directory for reordered 9-3-2 tiffs
    temp_image_folder = tempfile.mkdtemp(prefix="converted_932_")
    print(f"[INFO] Temporary folder for 9-3-2 tiles: {temp_image_folder}")

    # Divide images among GPUs
    chunks = chunk_list(image_paths, len(GPUS))

    # Launch one process per GPU (map logical index -> actual GPU id in GPUS)
    processes = []
    for i, chunk in enumerate(chunks):
        real_gpu = GPUS[i]
        p = Process(target=encode_images_on_gpu, args=(real_gpu, chunk, temp_image_folder))
        p.start()
        processes.append(p)

    # Wait for all processes to finish
    for p in processes:
        p.join()

    # Normalize 'res' column in all emitted CSVs (feature_dir and subdirs)
    csv_files = glob.glob(os.path.join(feature_dir, "**", "*.csv"), recursive=True)
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
        except Exception as e:
            print(f"[WARN] Skipping {csv_file}: couldn't read CSV ({e})")
            continue

        if 'res' not in df.columns:
            continue

        if not pd.api.types.is_numeric_dtype(df['res']):
            parsed = df['res'].apply(_parse_res_val)

            if parsed.notna().any():
                df['res'] = parsed
                try:
                    df.to_csv(csv_file, index=False)
                except Exception as e:
                    print(f"[WARN] Failed to write updated CSV {csv_file}: {e}")
            else:
                sample = df['res'].head(3).tolist()
                print(f"[WARN] Could not parse any numeric 'res' in {csv_file}. Sample: {sample}")

    # Optional: clean up temp files
    try:
        shutil.rmtree(temp_image_folder)
    except Exception as e:
        print(f"[WARN] Could not remove temp folder {temp_image_folder}: {e}")

    print("[INFO] All encoding complete.")
