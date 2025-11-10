# evaluation.py (PyTorch) — evaluate *all* checkpoints in a folder, save per-frame masks, write one CSV summary
import os
import glob
import time
import json
from typing import List, Tuple, Dict

import numpy as np
from tqdm import tqdm
import rasterio

import torch
import torch.nn as nn

# ===== project imports =====
from core.UNet import UNet
from core.split_frames import split_dataset
from core.frame_info import image_normalize
from core.common.data import get_all_frames
from core.common.console import _col, _C
from core.common.model_utils import (
    _forward_with_autopad,
    _as_probs_from_terratorch,
    _as_probs_from_terratorch_logits_first,
    _ensure_nchw,
    _is_terramind_model,
)

from core.losses import (
    # metrics exposed to training
    dice_coef, dice_loss, accuracy, specificity, sensitivity,
    f_beta, f1_score, IoU, nominal_surface_distance,
    Hausdorff_distance, boundary_intersection_over_union,
    # primitive helpers (to aggregate globally)
    true_positives, false_positives, true_negatives, false_negatives,
)


# -----------------------------
# Small utility
# -----------------------------
def _fmt_seconds(s: float) -> str:
    m, s = divmod(int(s), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


# -----------------------------
# Metrics accumulator (matches core/losses definitions)
# -----------------------------
class MetricAccumulator:
    """
    Aggregates pixel-level totals so final numbers match the definitions in core/losses_pytorch.py.
    Works with variable-sized images (we sum scalars, not per-image means).
    """
    def __init__(self):
        # For "soft" Dice (as in dice_coef: |y_true * y_pred| and sums)
        self._dice_inter = 0.0
        self._dice_true  = 0.0
        self._dice_pred  = 0.0

        # For thresholded metrics (TP/FP/TN/FN as in helpers using round())
        self._tp = 0.0
        self._fp = 0.0
        self._tn = 0.0
        self._fn = 0.0

        # For exact accuracy (mean of equality across whole testset)
        self._acc_correct = 0.0
        self._acc_total   = 0.0

    @staticmethod
    def _ensure_nchw_torch(a: np.ndarray) -> torch.Tensor:
        # a: (H,W) or (H,W,1) or (1,H,W); -> (1,1,H,W) float
        t = torch.from_numpy(a.astype(np.float32))
        if t.ndim == 2:        # H,W
            t = t.unsqueeze(0).unsqueeze(0)
        elif t.ndim == 3:      # H,W,1 or 1,H,W
            if t.shape[2] == 1:  # H,W,1 -> 1,1,H,W
                t = t.permute(2,0,1).unsqueeze(0)
            else:
                t = t.unsqueeze(0)
        elif t.ndim == 4:      # already NCHW
            pass
        else:
            raise ValueError(f"Unsupported tensor shape for metrics: {tuple(t.shape)}")
        return t

    def add(self, y_true_np: np.ndarray, y_pred_prob_np: np.ndarray):
        """
        y_true_np: binary ground truth in [0,1] (H,W) or (H,W,1)
        y_pred_prob_np: prediction probabilities in [0,1] (H,W) or (H,W,1)
        """
        yt = self._ensure_nchw_torch(y_true_np)
        yp = self._ensure_nchw_torch(y_pred_prob_np)

        # ---- soft Dice parts (exactly what dice_coef uses) ----
        self._dice_inter += float(torch.sum(torch.abs(yt * yp)))
        self._dice_true  += float(torch.sum(yt))
        self._dice_pred  += float(torch.sum(yp))

        # ---- thresholded primitives (exactly what losses helpers do) ----
        tp = torch.sum(true_positives(yt, yp))
        fp = torch.sum(false_positives(yt, yp))
        tn = torch.sum(true_negatives(yt, yp))
        fn = torch.sum(false_negatives(yt, yp))
        self._tp += float(tp)
        self._fp += float(fp)
        self._tn += float(tn)
        self._fn += float(fn)

        # ---- accuracy as mean equality across ALL pixels (no per-image mean) ----
        pred_b = torch.round(yp)
        true_b = torch.round(yt)
        self._acc_correct += float((pred_b == true_b).float().sum())
        self._acc_total   += float(true_b.numel())

    def finalize(self) -> Dict[str, float]:
        # Match eps from losses file
        eps_dice = 1.0e-7
        eps      = 1.0e-12

        # dice_coef / dice_loss (soft)
        dice = (2.0 * self._dice_inter + eps_dice) / (self._dice_true + self._dice_pred + eps_dice)
        dloss = 1.0 - dice

        # accuracy (exact global mean)
        acc = self._acc_correct / max(self._acc_total, 1.0)

        # sums for the rest (thresholded)
        tp, fp, tn, fn = self._tp, self._fp, self._tn, self._fn

        sens = tp / (tp + fn + eps)
        spec = tn / (tn + fp + eps)
        iou  = tp / (tp + fp + fn + eps)

        prec = tp / (tp + fp + eps)
        f1   = (2 * prec * sens) / (prec + sens + eps)

        # F-beta (beta=1 same as f1, but keep both for completeness)
        beta  = 1.0
        beta2 = beta * beta
        fbeta = (1 + beta2) * prec * sens / (beta2 * prec + sens + eps)

        # Proxies kept identical to your losses file
        nsd   = fp / (tp + fp + fn + eps)  # nominal_surface_distance proxy
        hd    = fp / (tp + fp + fn + eps)  # Hausdorff proxy
        biou  = tp / (tp + fp + fn + eps)  # boundary IoU proxy

        return {
            "dice_coef": float(dice),
            "dice_loss": float(dloss),
            "accuracy": float(acc),
            "sensitivity": float(sens),
            "specificity": float(spec),
            "IoU": float(iou),
            "f1_score": float(f1),
            "f_beta": float(fbeta),
            "nominal_surface_distance": float(nsd),
            "Hausdorff_distance": float(hd),
            "boundary_intersection_over_union": float(biou),
        }


# -----------------------------
# Data helpers (reuse training ordering)
# -----------------------------
def _list_preprocessed_paths(preprocessed_dir: str) -> List[str]:
    image_paths = sorted(
        glob.glob(os.path.join(preprocessed_dir, "*.tif")),
        key=lambda f: int(os.path.basename(f)[:-4]),
    )
    return image_paths


def _gather_frames_and_test_indices(config) -> Tuple[list, list, list]:
    """Load frames once and read train/val/test split indices."""
    frames = get_all_frames(config)  # uses config + last-band label convention
    image_paths = _list_preprocessed_paths(config.preprocessed_dir)

    print("Reading train-test split from file")
    frames_json = os.path.join(config.preprocessed_dir, "aa_frames_list.json")
    train_idx, val_idx, test_idx = split_dataset(frames, frames_json, config.test_ratio, config.val_ratio)

    print(f"training_frames {len(train_idx)}")
    print(f"validation_frames {len(val_idx)}")
    print(f"testing_frames {len(test_idx)}\n")

    return frames, image_paths, test_idx


# -----------------------------
# Model loading (UNet)
# -----------------------------
def _build_unet(config) -> nn.Module:
    in_ch = len(getattr(config, "channel_list", []))
    model = UNet(
        [config.train_batch_size, *config.patch_size, in_ch],
        [in_ch],
        config.dilation_rate,
    )
    return model


def _load_unet_from_checkpoint(config, ckpt_path: str, device: torch.device) -> nn.Module:
    model = _build_unet(config).to(device=device).eval()
    state = torch.load(ckpt_path, map_location="cpu")

    # Accept both bundle {"model_state": ...} and raw state_dict
    if isinstance(state, dict) and "model_state" in state:
        state_dict = state["model_state"]
    else:
        state_dict = state

    # Sometimes checkpoints were saved as weights-only
    try:
        model.load_state_dict(state_dict, strict=True)
    except Exception:
        # Allow non-strict in case of harmless keys
        model.load_state_dict(state_dict, strict=False)
    return model


# -----------------------------
# Inference (full-frame, not patches)
# -----------------------------
def _select_eval_channel(prob_nchw: torch.Tensor, config) -> torch.Tensor:
    """Select the class/channel for evaluation (binary default: keep channel 1 or single-channel)."""
    prob_nchw = _ensure_nchw(prob_nchw).float()
    if prob_nchw.shape[1] > 1:
        cls_idx = int(getattr(config, "metrics_class", 1))
        cls_idx = max(0, min(cls_idx, prob_nchw.shape[1] - 1))
        return prob_nchw[:, cls_idx:cls_idx + 1]
    return prob_nchw


@torch.no_grad()
def _infer_full_image(model: nn.Module, frame, device: torch.device, config) -> np.ndarray:
    """
    frame.img: H,W,C_in  (already preprocessed feature bands)
    frame.annotations: H,W
    Use only the first K = len(config.channel_list) channels for inputs.
    """
    x = frame.img  # H,W,C_total (all preprocessed bands)
    k = len(getattr(config, "channel_list", []))
    if x.shape[2] < k:
        raise RuntimeError(f"Frame has {x.shape[2]} channels but config.channel_list requires {k}.")
    x = x[:, :, :k]  # first K inputs

    # normalize like training patches (per-channel over H,W)
    x = image_normalize(x, axis=(0, 1))
    x_t = torch.from_numpy(np.transpose(x.astype(np.float32), (2, 0, 1))).unsqueeze(0).to(device)  # 1,C,H,W

    # forward + robust logits->probs
    y_raw = _forward_with_autopad(model, x_t)
    if _is_terramind_model(model):
        prob_full = _as_probs_from_terratorch_logits_first(y_raw, num_classes=int(getattr(config, "num_classes", 1)))
    else:
        prob_full = _as_probs_from_terratorch(y_raw, num_classes=int(getattr(config, "num_classes", 1)))
    prob = _select_eval_channel(prob_full, config)  # 1,1,H,W
    prob = prob.squeeze(0).squeeze(0).clamp(0, 1).detach().cpu().numpy()  # H,W
    return prob


# -----------------------------
# Checkpoint discovery
# -----------------------------
def _find_all_checkpoints(folder: str) -> List[str]:
    pats = [
        os.path.join(folder, "*.pt"),
        os.path.join(folder, "*.weights.pt"),
        os.path.join(folder, "*.raw.weights.pt"),
    ]
    fps = []
    for p in pats:
        fps.extend(glob.glob(p))
    # unique + sort by mtime (newest last)
    fps = sorted(list(set(fps)), key=lambda f: os.stat(f).st_mtime)
    return fps


# -----------------------------
# CSV writing (append for all models)
# -----------------------------
def _append_results_row(csv_path: str, header: List[str], row: List[str]) -> None:
    write_header = not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "a") as f:
        if write_header:
            f.write(",".join(header) + "\n")
        f.write(",".join(map(str, row)) + "\n")


# -----------------------------
# Core evaluation (UNet arch)
# -----------------------------
def _evaluate_arch(config, arch: str = "unet") -> None:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- load data once ----
    frames, image_paths, test_idx = _gather_frames_and_test_indices(config)
    print(f"Testing frames: {len(test_idx)}")

    # ---- discover checkpoints in the folder (evaluate all) ----
    ckpts = _find_all_checkpoints(config.saved_models_dir)
    print(f"Found {len(ckpts)} checkpoints.\n")

    # ---- CSV path (one summary for all evaluated models) ----
    csv_path = os.path.join(config.results_dir, f"evaluation_{arch}.csv")
    header = [
        "run_name", "checkpoint_path",
        "dice_coef", "dice_loss", "IoU", "accuracy",
        "sensitivity", "specificity", "f1_score", "f_beta",
        "nominal_surface_distance", "Hausdorff_distance", "boundary_intersection_over_union",
        "elapsed",
    ]

    results_written = 0

    for ckpt_path in ckpts:
        try:
            print(f"{_C.GREEN}[EVAL]{_C.RESET} model={arch}  ckpt={ckpt_path}")
            # output dir: /results/<checkpoint basename without trailing .pt>
            base = os.path.basename(ckpt_path)
            out_dir = os.path.join(config.results_dir, base.replace(".pt", ""))
            print(f"{_C.YELLOW}[EVAL]{_C.RESET} saving masks -> {out_dir}")
            os.makedirs(out_dir, exist_ok=True)

            # ---- build/load model ----
            if arch == "unet":
                model = _load_unet_from_checkpoint(config, ckpt_path, device).eval()
            else:
                raise NotImplementedError(f"Unknown arch: {arch}")

            # ---- eval loop ----
            accum = MetricAccumulator()
            t0 = time.time()
            thr = float(getattr(config, "eval_threshold", 0.5))

            for i in tqdm(range(len(test_idx)), desc=f"Predicting ({arch})"):
                idx = test_idx[i]
                frame = frames[idx]
                im_fp = image_paths[idx]

                # predict prob map [H,W] in [0,1]
                prob = _infer_full_image(model, frame, device, config)

                # load GT and scale to [0,1] if needed
                gt = frame.annotations.astype(np.float32)
                if gt.max() > 1.5:
                    gt = gt / 255.0
                gt = np.clip(gt, 0.0, 1.0)

                # accumulate metrics
                accum.add(gt, prob)

                # save hard mask as GeoTIFF
                pred_bin = (prob >= thr).astype(np.uint8)
                with rasterio.open(im_fp) as src:
                    profile = src.profile
                    profile.update(count=1, dtype="uint8", compress="LZW")
                    out_fp = os.path.join(out_dir, os.path.basename(im_fp))
                    with rasterio.open(out_fp, "w", **profile) as dst:
                        dst.write(pred_bin, 1)

            # ---- finalize metrics ----
            metrics = accum.finalize()
            elapsed = _fmt_seconds(time.time() - t0)

            row = [
                getattr(config, "run_name", "run"),
                ckpt_path,
                f"{metrics['dice_coef']:.6f}",
                f"{metrics['dice_loss']:.6f}",
                f"{metrics['IoU']:.6f}",
                f"{metrics['accuracy']:.6f}",
                f"{metrics['sensitivity']:.6f}",
                f"{metrics['specificity']:.6f}",
                f"{metrics['f1_score']:.6f}",
                f"{metrics['f_beta']:.6f}",
                f"{metrics['nominal_surface_distance']:.6f}",
                f"{metrics['Hausdorff_distance']:.6f}",
                f"{metrics['boundary_intersection_over_union']:.6f}",
                elapsed,
            ]
            _append_results_row(csv_path, header, row)
            results_written += 1

        except Exception as exc:
            print(_col(f"Evaluation failed for {ckpt_path}: {exc}", _C.RED))

    if results_written == 0:
        print(_col("No results to write.", _C.YELLOW))
    else:
        print(_col(f"Wrote {results_written} result rows to {csv_path}", _C.GREEN))


# -----------------------------
# Public entrypoints (match training-style)
# -----------------------------
def evaluate_unet(conf) -> None:
    """
    Evaluate ALL UNet checkpoints found in config.saved_models_dir.
    Saves per-frame hard masks under config.results_dir/<checkpoint_basename>
    and appends one row per checkpoint to config.results_dir/evaluation_unet.csv
    """
    global config
    config = conf
    _evaluate_arch(config, arch="unet")
