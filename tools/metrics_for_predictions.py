"""
Recompute evaluation metrics from existing prediction masks.

This script:
1. Loads existing prediction masks (not re-running inference)
2. Excludes NaN pixels from all metric calculations
3. Computes mean uncertainties (epistemic and aleatoric) per frame and dataset
4. Generates two CSV outputs per model/modality:
   - Dataset-level metrics (one row per checkpoint)
   - Frame-level metrics (one row per frame per checkpoint)
"""

from __future__ import annotations

import csv
import os
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import rasterio
import torch
from tqdm import tqdm
from rasterio.warp import reproject, Resampling

import sys
from pathlib import Path

# add project root (parent of /tools) to PYTHONPATH at runtime
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from core.losses import (
    Hausdorff_distance,
    boundary_intersection_over_union,
    normalized_surface_distance,
)

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def nan_to_zero(a: np.ndarray) -> np.ndarray:
    """Return copy with NaN/Inf replaced by 0.0 (background)."""
    out = a.astype(np.float32, copy=True)
    out[~np.isfinite(out)] = 0.0
    return out


def safe_nanmax(a: np.ndarray) -> float:
    """nanmax that won't warn/crash on all-NaN."""
    if np.isfinite(a).any():
        return float(np.nanmax(a))
    return float("nan")


def maybe_normalize_255_to_1(a: np.ndarray) -> np.ndarray:
    """If values look like 0..255 mask, normalize to 0..1."""
    mx = safe_nanmax(a)
    if np.isfinite(mx) and mx > 1.5:
        return a / 255.0
    return a

def create_unified_valid_mask(
    frame_id: str,
    modalities: List[str],
    training_image_dir: Path,
    target_resolution: str = "highest"  # or "S2", "AE", "PS"
) -> Dict[str, np.ndarray]:
    """
    Create a unified valid mask across all modalities for a given frame.
    
    Returns:
        Dict mapping modality -> valid_mask at that modality's resolution
    """
    import rasterio
    from rasterio.warp import reproject, Resampling
    
    # Step 1: Read all modality images and extract valid masks
    modality_data = {}
    for modality in modalities:
        modality_dir = training_image_dir / modality
        img_path = modality_dir / f"{frame_id}.tif"
        
        if not img_path.exists():
            continue
            
        with rasterio.open(img_path) as src:
            img = src.read().astype(np.float32)  # [C, H, W]
            # Valid where any feature band is non-zero
            valid_mask = np.any(img[:-1, :, :] != 0, axis=0)  # [H, W]
            
            modality_data[modality] = {
                'valid_mask': valid_mask,
                'transform': src.transform,
                'crs': src.crs,
                'shape': valid_mask.shape
            }
    
    if not modality_data:
        return {}
    
    # Step 2: Determine target resolution
    if target_resolution == "highest":
        # Use the modality with highest resolution (smallest pixel size)
        target_mod = min(
            modality_data.keys(),
            key=lambda m: abs(modality_data[m]['transform'].a)  # pixel width
        )
    else:
        target_mod = target_resolution
    
    target_shape = modality_data[target_mod]['shape']
    target_transform = modality_data[target_mod]['transform']
    target_crs = modality_data[target_mod]['crs']
    
    # Step 3: Reproject all masks to target resolution and combine
    unified_mask = np.ones(target_shape, dtype=bool)
    
    for modality, data in modality_data.items():
        if modality == target_mod:
            # No reprojection needed
            unified_mask &= data['valid_mask']
        else:
            # Reproject to target resolution
            reprojected_mask = np.zeros(target_shape, dtype=np.uint8)
            
            reproject(
                source=data['valid_mask'].astype(np.uint8),
                destination=reprojected_mask,
                src_transform=data['transform'],
                src_crs=data['crs'],
                dst_transform=target_transform,
                dst_crs=target_crs,
                resampling=Resampling.nearest
            )
            
            unified_mask &= reprojected_mask.astype(bool)
    
    # Step 4: Reproject unified mask back to each modality's native resolution
    result_masks = {}
    
    for modality, data in modality_data.items():
        if modality == target_mod:
            result_masks[modality] = unified_mask
        else:
            # Reproject from target back to this modality's resolution
            native_mask = np.zeros(data['shape'], dtype=np.uint8)
            
            reproject(
                source=unified_mask.astype(np.uint8),
                destination=native_mask,
                src_transform=target_transform,
                src_crs=target_crs,
                dst_transform=data['transform'],
                dst_crs=data['crs'],
                resampling=Resampling.nearest
            )
            
            result_masks[modality] = native_mask.astype(bool)
    
    return result_masks

# ---------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------


class MetricsCalculator:
    """
    Accumulates metrics at dataset level with NaN-aware processing.

    - MICRO-average for pixel metrics (global confusion matrix)
    - MACRO-average for geometric metrics (per-frame mean)

    NaN pixels are excluded everywhere by:
      - micro: computing confusion on valid pixels only
      - macro: replacing NaNs with 0 (background) before geometric metrics
    """

    def __init__(
        self,
        device: torch.device,
        threshold: float = 0.5,
        fbeta_beta: float = 2.0,
    ) -> None:
        self.device = device
        self.threshold = float(threshold)
        self.fbeta_beta = float(fbeta_beta)

        self.macro_metrics_fns = {
            "normalized_surface_distance": normalized_surface_distance,
            "Hausdorff_distance": Hausdorff_distance,
            "boundary_intersection_over_union": boundary_intersection_over_union,
        }

        self.reset()

    def reset(self) -> None:
        self.tp = 0
        self.tn = 0
        self.fp = 0
        self.fn = 0

        self.n_macro = 0
        self.macro_metrics_sums: Dict[str, float] = {
            "normalized_surface_distance": 0.0,
            "Hausdorff_distance": 0.0,
            "boundary_intersection_over_union": 0.0,
        }

    def _to_tensors(
        self, y_true_arr: np.ndarray, y_pred_arr: np.ndarray
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convert [H,W] numpy arrays to [1,1,H,W] torch tensors."""
        y_true = torch.from_numpy(y_true_arr.astype(np.float32))
        y_pred = torch.from_numpy(y_pred_arr.astype(np.float32))

        if y_true.ndim == 2:
            y_true = y_true.unsqueeze(0).unsqueeze(0)
            y_pred = y_pred.unsqueeze(0).unsqueeze(0)

        y_true = y_true.to(self.device, non_blocking=True)
        y_pred = y_pred.to(self.device, non_blocking=True)
        return y_true, y_pred

    @staticmethod
    def _as_float(value) -> float:
        if isinstance(value, torch.Tensor):
            return float(value.detach().mean().cpu().item())
        return float(value)

    def add(self, y_true_arr: np.ndarray, y_pred_arr: np.ndarray) -> None:
        """
        Update metrics with a single image pair.
        NaN pixels are excluded from all calculations.
        """
        y_true_arr = np.clip(y_true_arr, 0.0, 1.0)
        y_pred_arr = np.clip(y_pred_arr, 0.0, 1.0)

        valid_mask = np.isfinite(y_true_arr) & np.isfinite(y_pred_arr)
        if not np.any(valid_mask):
            return

        # --------------------
        # MICRO: confusion matrix on valid pixels only
        # --------------------
        y_true_valid = y_true_arr[valid_mask]
        y_pred_valid = y_pred_arr[valid_mask]

        gt = (y_true_valid >= 0.5)
        pred = (y_pred_valid >= self.threshold)

        self.tp += int(np.logical_and(gt, pred).sum())
        self.fp += int(np.logical_and(~gt, pred).sum())
        self.tn += int(np.logical_and(~gt, ~pred).sum())
        self.fn += int(np.logical_and(gt, ~pred).sum())

        # --------------------
        # MACRO: geometric metrics per frame
        # NaNs replaced with 0 background so they don't poison geometry ops.
        # Use binarized masks for geometry.
        # --------------------
        y_true_clean = nan_to_zero(y_true_arr)
        y_pred_clean = nan_to_zero(y_pred_arr)

        y_true_t, y_pred_t = self._to_tensors(y_true_clean, y_pred_clean)
        y_true_bin = (y_true_t >= 0.5).float()
        y_pred_bin = (y_pred_t >= self.threshold).float()

        with torch.no_grad():
            for name, fn in self.macro_metrics_fns.items():
                try:
                    val = fn(y_true_bin, y_pred_bin)
                    self.macro_metrics_sums[name] += self._as_float(val)
                except Exception as e:
                    print(f"Warning: could not compute {name} for one sample: {e}")
                    continue

        self.n_macro += 1

    def finalize(self) -> Dict[str, float]:
        eps = 1e-9
        tp = float(self.tp)
        tn = float(self.tn)
        fp = float(self.fp)
        fn = float(self.fn)

        dice = (2.0 * tp) / (2.0 * tp + fp + fn + eps)
        iou = tp / (tp + fp + fn + eps)
        acc = (tp + tn) / (tp + tn + fp + fn + eps)
        sens = tp / (tp + fn + eps)
        spec = tn / (tn + fp + eps)

        prec = tp / (tp + fp + eps)
        rec = sens
        f1 = (2.0 * prec * rec) / (prec + rec + eps)

        b2 = self.fbeta_beta ** 2.0
        fbeta = (1.0 + b2) * (prec * rec) / (b2 * prec + rec + eps)

        out: Dict[str, float] = {
            "dice_coef": float(dice),
            "dice_loss": 1.0 - float(dice),
            "IoU": float(iou),
            "accuracy": float(acc),
            "sensitivity": float(sens),
            "specificity": float(spec),
            "f1_score": float(f1),
            "f_beta": float(fbeta),
        }

        div = float(max(1, self.n_macro))
        out["normalized_surface_distance"] = float(
            self.macro_metrics_sums["normalized_surface_distance"] / div
        )
        out["Hausdorff_distance"] = float(
            self.macro_metrics_sums["Hausdorff_distance"] / div
        )
        out["boundary_intersection_over_union"] = float(
            self.macro_metrics_sums["boundary_intersection_over_union"] / div
        )

        return out


def compute_frame_metrics(
    y_true_arr: np.ndarray,
    y_pred_arr: np.ndarray,
    device: torch.device,
    threshold: float = 0.5,
    fbeta_beta: float = 2.0,
) -> Dict[str, float]:
    """
    Compute metrics for a single frame pair, excluding NaN pixels.
    """
    eps = 1e-9

    y_true_arr = np.clip(y_true_arr, 0.0, 1.0)
    y_pred_arr = np.clip(y_pred_arr, 0.0, 1.0)

    valid_mask = np.isfinite(y_true_arr) & np.isfinite(y_pred_arr)
    if not np.any(valid_mask):
        return {k: float("nan") for k in [
            "dice_coef",
            "dice_loss",
            "IoU",
            "accuracy",
            "sensitivity",
            "specificity",
            "f1_score",
            "f_beta",
            "normalized_surface_distance",
            "Hausdorff_distance",
            "boundary_intersection_over_union",
        ]}

    y_true_valid = y_true_arr[valid_mask]
    y_pred_valid = y_pred_arr[valid_mask]

    gt = (y_true_valid >= 0.5)
    pred = (y_pred_valid >= threshold)

    tp = float(np.logical_and(gt, pred).sum())
    fp = float(np.logical_and(~gt, pred).sum())
    tn = float(np.logical_and(~gt, ~pred).sum())
    fn = float(np.logical_and(gt, ~pred).sum())

    dice = (2.0 * tp) / (2.0 * tp + fp + fn + eps)
    iou = tp / (tp + fp + fn + eps)
    acc = (tp + tn) / (tp + tn + fp + fn + eps)
    sens = tp / (tp + fn + eps)
    spec = tn / (tn + fp + eps)

    prec = tp / (tp + fp + eps)
    rec = sens
    f1 = (2.0 * prec * rec) / (prec + rec + eps)

    b2 = fbeta_beta ** 2.0
    fbeta = (1.0 + b2) * (prec * rec) / (b2 * prec + rec + eps)

    # Geometric metrics: NaN-safe by converting NaNs->0 and binarizing
    y_true_clean = nan_to_zero(y_true_arr)
    y_pred_clean = nan_to_zero(y_pred_arr)

    y_true_t = torch.from_numpy(y_true_clean).unsqueeze(0).unsqueeze(0).to(device)
    y_pred_t = torch.from_numpy(y_pred_clean).unsqueeze(0).unsqueeze(0).to(device)

    y_true_bin = (y_true_t >= 0.5).float()
    y_pred_bin = (y_pred_t >= threshold).float()

    with torch.no_grad():
        try:
            nsd = float(normalized_surface_distance(y_true_bin, y_pred_bin).cpu().item())
        except Exception:
            nsd = float("nan")

        try:
            hd = float(Hausdorff_distance(y_true_bin, y_pred_bin).cpu().item())
        except Exception:
            hd = float("nan")

        try:
            biou = float(boundary_intersection_over_union(y_true_bin, y_pred_bin).cpu().item())
        except Exception:
            biou = float("nan")

    return {
        "dice_coef": float(dice),
        "dice_loss": 1.0 - float(dice),
        "IoU": float(iou),
        "accuracy": float(acc),
        "sensitivity": float(sens),
        "specificity": float(spec),
        "f1_score": float(f1),
        "f_beta": float(fbeta),
        "normalized_surface_distance": nsd,
        "Hausdorff_distance": hd,
        "boundary_intersection_over_union": biou,
    }


# ---------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------


def read_label_img(path: Path) -> Tuple[np.ndarray, np.ndarray, str]:
    """
    Read ground truth label image (LAST band) with valid mask.

    rasterio.read() returns [C, H, W].
    We treat last band as label, others as feature bands for valid mask.

    Returns:
        masked_labels [H,W] (invalid -> NaN),
        valid_mask [H,W],
        crs_info
    """
    with rasterio.open(path) as src:
        img = src.read().astype(np.float32)  # [C,H,W]
        crs_info = f"CRS: {src.crs}, Transform: {src.transform}"

        labels = img[-1, :, :]  # last band
        valid_mask = np.any(img[:-1, :, :] != 0, axis=0)  # [H,W]

        masked_labels = np.where(valid_mask, labels, np.nan)

    return masked_labels, valid_mask, crs_info


def read_prediction_img(path: Path, valid_mask: np.ndarray) -> Tuple[np.ndarray, str]:
    """
    Read prediction image (band 1) and apply valid mask.

    Returns:
        valid_prediction [H,W] (invalid -> NaN),
        crs_info
    """
    with rasterio.open(path) as src:
        img = src.read(1).astype(np.float32)  # [H,W]
        crs_info = f"CRS: {src.crs}, Transform: {src.transform}"

        valid_prediction = np.where(valid_mask, img, np.nan)

    return valid_prediction, crs_info


def get_label_image_paths(base_dir: Path, modality: str, test_frames: List[str]) -> List[Path]:
    modality_dir = base_dir / modality
    if not modality_dir.exists():
        raise FileNotFoundError(f"Modality directory not found: {modality_dir}")

    label_paths = [
        modality_dir / f
        for f in os.listdir(modality_dir)
        if f.endswith(".tif") and Path(f).stem in test_frames
    ]

    return sorted(label_paths)


def get_checkpoint_predictions_for_model(
    results_dir: Path,
    model: str,
    modality: str,
) -> Tuple[Dict[str, List[Path]], Dict[str, List[Path]], Dict[str, List[Path]]]:
    """
    Get prediction, aleatoric, and epistemic paths organized by checkpoint.

    Returns:
        (checkpoint_predictions, checkpoint_aleatoric, checkpoint_epistemic)
    """
    model_modality_dir = results_dir / model / modality
    if not model_modality_dir.exists():
        return {}, {}, {}

    checkpoint_predictions: Dict[str, List[Path]] = {}
    checkpoint_aleatoric: Dict[str, List[Path]] = {}
    checkpoint_epistemic: Dict[str, List[Path]] = {}

    checkpoint_dirs = sorted([d for d in model_modality_dir.iterdir() if d.is_dir()])
    #print("Found", len(checkpoint_dirs), checkpoint_dirs)
    checkpoint_dirs = checkpoint_dirs[:10]
    #print("Found", len(checkpoint_dirs), checkpoint_dirs)

    for checkpoint_dir in checkpoint_dirs:
        if not checkpoint_dir.is_dir():
            continue

        # Prediction masks (no suffix)
        pred_paths = sorted([
            checkpoint_dir / f
            for f in os.listdir(checkpoint_dir)
            if f.endswith(".tif") and not any(suffix in f for suffix in ["_prob", "_epistemic", "_aleatoric"])
        ])
        if pred_paths:
            checkpoint_predictions[checkpoint_dir.name] = pred_paths

        # Aleatoric maps
        aleatoric_paths = sorted([
            checkpoint_dir / f
            for f in os.listdir(checkpoint_dir)
            if f.endswith(".tif") and "_aleatoric" in f
        ])
        if aleatoric_paths:
            checkpoint_aleatoric[checkpoint_dir.name] = aleatoric_paths

        # Epistemic maps
        epistemic_paths = sorted([
            checkpoint_dir / f
            for f in os.listdir(checkpoint_dir)
            if f.endswith(".tif") and "_epistemic" in f
        ])
        if epistemic_paths:
            checkpoint_epistemic[checkpoint_dir.name] = epistemic_paths



    return checkpoint_predictions, checkpoint_aleatoric, checkpoint_epistemic


# ---------------------------------------------------------------------
# Main recompute logic
# ---------------------------------------------------------------------


def recompute_metrics(
    label_paths: List[Path],
    checkpoint_predictions: Dict[str, List[Path]],
    checkpoint_aleatoric: Dict[str, List[Path]],
    checkpoint_epistemic: Dict[str, List[Path]],
    model: str,
    modality: str,
    results_base_dir: Path,
    device: torch.device,
    threshold: float = 0.5,
    fbeta_beta: float = 2.0,
    timestamp: str = None,
    all_modalities: List[str] = None,
    training_image_dir: Path = None,
) -> None:
    """
    Recompute metrics for all checkpoints of a specific model/modality.
    Includes uncertainty metrics from epistemic and aleatoric maps.
    
    Uses unified masking across all modalities to ensure same spatial locations
    are evaluated across different resolutions.
    """
    output_dir = results_base_dir / model / modality / f"metrics_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_csv = output_dir / f"{model}x{modality}_dataset_metrics.csv"
    frame_csv = output_dir / f"{model}x{modality}_frame_metrics.csv"

    dataset_header = [
        "model",
        "modality",
        "checkpoint",
        "dice_coef",
        "dice_loss",
        "IoU",
        "accuracy",
        "sensitivity",
        "specificity",
        "f1_score",
        "f_beta",
        "normalized_surface_distance",
        "Hausdorff_distance",
        "boundary_intersection_over_union",
        "mean_aleatoric_uncertainty",
        "mean_epistemic_uncertainty",
    ]

    frame_header = [
        "model",
        "modality",
        "checkpoint",
        "frame_id",
        "dice_coef",
        "dice_loss",
        "IoU",
        "accuracy",
        "sensitivity",
        "specificity",
        "f1_score",
        "f_beta",
        "normalized_surface_distance",
        "Hausdorff_distance",
        "boundary_intersection_over_union",
        "mean_aleatoric_uncertainty",
        "mean_epistemic_uncertainty",
    ]

    label_lookup = {path.stem: path for path in label_paths}

    print(f"\n{'='*80}")
    print(f"Processing: {model} x {modality}")
    print(f"Found {len(checkpoint_predictions)} checkpoint(s) to process")
    print(f"{'='*80}\n")

    # ========================================================================
    # NEW: PRE-COMPUTE unified masks for all frames
    # ========================================================================
    unified_masks = {}
    if all_modalities is not None and training_image_dir is not None:
        print("Computing unified valid masks across modalities...")
        for frame_id in tqdm(set(p.stem for p in label_paths), desc="Creating unified masks"):
            try:
                masks = create_unified_valid_mask(
                    frame_id=frame_id,
                    modalities=all_modalities,
                    training_image_dir=training_image_dir,
                    target_resolution="highest"  # or specify a modality like "S2"
                )
                if modality in masks:
                    unified_masks[frame_id] = masks[modality]
                else:
                    print(f"  Warning: No mask for modality {modality} in frame {frame_id}")
            except Exception as e:
                print(f"  Warning: Could not create unified mask for {frame_id}: {e}")
        
        print(f"Created {len(unified_masks)} unified masks\n")
    else:
        print("Skipping unified masking (using per-modality masks)\n")
    # ========================================================================

    with open(dataset_csv, "w", newline="") as f:
        csv.writer(f).writerow(dataset_header)

    with open(frame_csv, "w", newline="") as f:
        csv.writer(f).writerow(frame_header)

    for checkpoint_name, pred_paths in checkpoint_predictions.items():
        print(f"\nCheckpoint: {checkpoint_name}")
        print(f"  Found {len(pred_paths)} predictions")

        accumulator = MetricsCalculator(device=device, threshold=threshold, fbeta_beta=fbeta_beta)
        frame_results = []
        n_processed = 0

        # per-frame-mean uncertainties (unweighted)
        all_aleatoric: List[float] = []
        all_epistemic: List[float] = []

        aleatoric_paths = checkpoint_aleatoric.get(checkpoint_name, [])
        epistemic_paths = checkpoint_epistemic.get(checkpoint_name, [])

        aleatoric_lookup = {Path(p).stem.replace("_aleatoric", ""): p for p in aleatoric_paths}
        epistemic_lookup = {Path(p).stem.replace("_epistemic", ""): p for p in epistemic_paths}

        for pred_path in tqdm(pred_paths, desc="  Processing", leave=False):
            frame_id = pred_path.stem

            if frame_id not in label_lookup:
                print(f"  Warning: label for frame {frame_id} not found, skipping")
                continue

            label_path = label_lookup[frame_id]

            # ================================================================
            # FIXED: Properly handle unified mask vs fallback
            # ================================================================
            try:
                if frame_id in unified_masks:
                    # Use the pre-computed unified mask
                    unified_valid_mask = unified_masks[frame_id]
                    
                    # Read label image and apply unified mask
                    label_img, _, _ = read_label_img(label_path)
                    label_img = np.where(unified_valid_mask, label_img, np.nan)
                    
                    # Read prediction with unified mask
                    pred_img, _ = read_prediction_img(pred_path, unified_valid_mask)
                    
                else:
                    # Fallback to original per-modality masking
                    label_img, unified_valid_mask, _ = read_label_img(label_path)
                    pred_img, _ = read_prediction_img(pred_path, unified_valid_mask)
                    
            except Exception as e:
                print(f"  Error reading images for frame {frame_id}: {e}")
                continue
            # ================================================================

            # Normalize to [0,1] if needed
            label_img = maybe_normalize_255_to_1(label_img)
            pred_img = maybe_normalize_255_to_1(pred_img)

            # Frame-level metrics (NaN-safe)
            frame_metrics = compute_frame_metrics(label_img, pred_img, device, threshold, fbeta_beta)

            mean_aleatoric = float("nan")
            mean_epistemic = float("nan")

            # FIXED: Process uncertainty maps with the same mask
            if frame_id in aleatoric_lookup:
                try:
                    aleatoric_img, _ = read_prediction_img(aleatoric_lookup[frame_id], unified_valid_mask)
                    aleatoric_img = maybe_normalize_255_to_1(aleatoric_img)
                    mean_aleatoric = float(np.nanmean(aleatoric_img))
                    all_aleatoric.append(mean_aleatoric)
                except Exception as e:
                    print(f"  Warning: could not read aleatoric for frame {frame_id}: {e}")

            if frame_id in epistemic_lookup:
                try:
                    epistemic_img, _ = read_prediction_img(epistemic_lookup[frame_id], unified_valid_mask)
                    epistemic_img = maybe_normalize_255_to_1(epistemic_img)
                    mean_epistemic = float(np.nanmean(epistemic_img))
                    all_epistemic.append(mean_epistemic)
                except Exception as e:
                    print(f"  Warning: could not read epistemic for frame {frame_id}: {e}")

            frame_results.append({
                "frame_id": frame_id,
                "mean_aleatoric_uncertainty": mean_aleatoric,
                "mean_epistemic_uncertainty": mean_epistemic,
                **frame_metrics,
            })

            accumulator.add(label_img, pred_img)
            n_processed += 1

        if n_processed <= 0:
            print("  No frames processed for this checkpoint.")
            continue

        dataset_metrics = accumulator.finalize()

        dataset_mean_aleatoric = float(np.nanmean(all_aleatoric)) if all_aleatoric else float("nan")
        dataset_mean_epistemic = float(np.nanmean(all_epistemic)) if all_epistemic else float("nan")

        dataset_row = [
            model,
            modality,
            checkpoint_name,
            f"{dataset_metrics['dice_coef']:.6f}",
            f"{dataset_metrics['dice_loss']:.6f}",
            f"{dataset_metrics['IoU']:.6f}",
            f"{dataset_metrics['accuracy']:.6f}",
            f"{dataset_metrics['sensitivity']:.6f}",
            f"{dataset_metrics['specificity']:.6f}",
            f"{dataset_metrics['f1_score']:.6f}",
            f"{dataset_metrics['f_beta']:.6f}",
            f"{dataset_metrics['normalized_surface_distance']:.6f}",
            f"{dataset_metrics['Hausdorff_distance']:.6f}",
            f"{dataset_metrics['boundary_intersection_over_union']:.6f}",
            f"{dataset_mean_aleatoric:.6f}" if np.isfinite(dataset_mean_aleatoric) else "nan",
            f"{dataset_mean_epistemic:.6f}" if np.isfinite(dataset_mean_epistemic) else "nan",
        ]

        with open(dataset_csv, "a", newline="") as f:
            csv.writer(f).writerow(dataset_row)

        with open(frame_csv, "a", newline="") as f:
            writer = csv.writer(f)
            for r in frame_results:
                frame_row = [
                    model,
                    modality,
                    checkpoint_name,
                    r["frame_id"],
                    f"{r['dice_coef']:.6f}" if np.isfinite(r["dice_coef"]) else "nan",
                    f"{r['dice_loss']:.6f}" if np.isfinite(r["dice_loss"]) else "nan",
                    f"{r['IoU']:.6f}" if np.isfinite(r["IoU"]) else "nan",
                    f"{r['accuracy']:.6f}" if np.isfinite(r["accuracy"]) else "nan",
                    f"{r['sensitivity']:.6f}" if np.isfinite(r["sensitivity"]) else "nan",
                    f"{r['specificity']:.6f}" if np.isfinite(r["specificity"]) else "nan",
                    f"{r['f1_score']:.6f}" if np.isfinite(r["f1_score"]) else "nan",
                    f"{r['f_beta']:.6f}" if np.isfinite(r["f_beta"]) else "nan",
                    f"{r['normalized_surface_distance']:.6f}" if np.isfinite(r["normalized_surface_distance"]) else "nan",
                    f"{r['Hausdorff_distance']:.6f}" if np.isfinite(r["Hausdorff_distance"]) else "nan",
                    f"{r['boundary_intersection_over_union']:.6f}" if np.isfinite(r["boundary_intersection_over_union"]) else "nan",
                    f"{r['mean_aleatoric_uncertainty']:.6f}" if np.isfinite(r["mean_aleatoric_uncertainty"]) else "nan",
                    f"{r['mean_epistemic_uncertainty']:.6f}" if np.isfinite(r["mean_epistemic_uncertainty"]) else "nan",
                ]
                writer.writerow(frame_row)

        print(f"  Processed {n_processed} frames")
        print(
            f"  Dataset Dice: {dataset_metrics['dice_coef']:.4f}, "
            f"IoU: {dataset_metrics['IoU']:.4f}, "
            f"F1: {dataset_metrics['f1_score']:.4f}"
        )
        print(
            f"  Mean Aleatoric: {dataset_mean_aleatoric:.6f}, "
            f"Mean Epistemic: {dataset_mean_epistemic:.6f}"
        )
        print(f"  Saved to: {output_dir}")


# ---------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------


def main() -> None:
    RESULTS_BASE_DIR = Path(r"/isipd/projects/p_planetdw/data/methods_test/results")
    TRAINING_IMAGE_DIR = Path(r"/isipd/projects/p_planetdw/data/methods_test/preprocessed")

    MODALITIES = ["AE", "PS", "S2"]
    MODELS = ["UNET", "TERRAMIND", "SWIN"]

    TEST_FRAMES = [
        "34", "60", "6", "173", "63", "74", "218", "126", "347", "325", "167", "290",
        "36", "123", "324", "193", "236", "182", "72", "367", "220", "183", "25", "340",
        "132", "56", "238", "113", "85", "245", "242", "42", "213", "55", "209", "386",
        "84", "97", "64", "131", "354", "272", "357", "111", "164", "109", "375", "346",
        "5", "11", "196", "89", "119", "259", "200", "273", "199", "32", "395", "298",
        "234", "394", "227", "302", "388", "289", "293", "158", "237", "360", "390",
        "291", "296", "189", "161", "153", "83", "70", "23", "239",
    ]

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    threshold = 0.5
    fbeta_beta = 2.0

    starttime = time.strftime("%Y%m%d-%H%M%S")

    print(f"\n{'#'*80}")
    print("# RECOMPUTING METRICS FROM EXISTING PREDICTIONS")

    print(f"{'#'*80}\n")

    for modality in MODALITIES:
        for model in MODELS:
            try:
                label_paths = get_label_image_paths(TRAINING_IMAGE_DIR, modality, TEST_FRAMES)
                print(f"\nFound {len(label_paths)} label images for {modality}")

                checkpoint_predictions, checkpoint_aleatoric, checkpoint_epistemic = (
                    get_checkpoint_predictions_for_model(RESULTS_BASE_DIR, model, modality)
                )

                if not checkpoint_predictions:
                    print(f"No predictions found for {model}/{modality}")
                    continue

                recompute_metrics(
                    label_paths=label_paths,
                    checkpoint_predictions=checkpoint_predictions,
                    checkpoint_aleatoric=checkpoint_aleatoric,
                    checkpoint_epistemic=checkpoint_epistemic,
                    model=model,
                    modality=modality,
                    results_base_dir=RESULTS_BASE_DIR,
                    device=device,
                    threshold=threshold,
                    fbeta_beta=fbeta_beta,
                    timestamp=starttime,
                    all_modalities=MODALITIES,
                    training_image_dir=TRAINING_IMAGE_DIR,
                )

            except FileNotFoundError as e:
                print(f"Error: {e}")
                continue
            except Exception as e:
                print(f"Unexpected error processing {model}/{modality}: {e}")
                import traceback
                traceback.print_exc()
                continue

    print("\n" + "=" * 80)
    print("ALL PROCESSING COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
