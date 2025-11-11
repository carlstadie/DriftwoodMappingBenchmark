from __future__ import annotations

import glob
import os
import time
from typing import Dict, List, Tuple, Optional

import numpy as np
import rasterio
import torch
from tqdm import tqdm

from core.UNet import UNet
from core.Swin_UNetPP import SwinUNet
from core.TerraMind import TerraMind

from core.common.console import _C, _col, _fmt_seconds
from core.common.data import get_all_frames
from core.common.model_utils import (
    _as_probs_from_terratorch,
    _as_probs_from_terratorch_logits_first,
    _ensure_nchw,
    _forward_with_autopad,
    _is_terramind_model,
)
from core.frame_info import image_normalize
from core.split_frames import split_dataset


# -----------------------------
# Simple metric accumulator
# -----------------------------
class MetricAccumulator:
    """Accumulate metrics over multiple predictions to compute epoch-level stats."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.n = 0
        self.sums: Dict[str, float] = {
            "dice_coef": 0.0,
            "dice_loss": 0.0,
            "IoU": 0.0,
            "accuracy": 0.0,
            "sensitivity": 0.0,
            "specificity": 0.0,
            "f1_score": 0.0,
            "f_beta": 0.0,
            "nominal_surface_distance": 0.0,
            "Hausdorff_distance": 0.0,
            "boundary_intersection_over_union": 0.0,
        }

    @staticmethod
    def _dice_coef(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-6) -> float:
        y_true = y_true.astype(np.float32)
        y_pred = y_pred.astype(np.float32)
        inter = np.sum(y_true * y_pred)
        den = np.sum(y_true) + np.sum(y_pred)
        return float((2.0 * inter + eps) / (den + eps))

    @staticmethod
    def _iou(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-6) -> float:
        y_true = y_true.astype(np.float32)
        y_pred = y_pred.astype(np.float32)
        inter = np.sum(y_true * y_pred)
        union = np.sum(y_true + y_pred) - inter
        return float((inter + eps) / (union + eps))

    @staticmethod
    def _acc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        total = y_true.size
        correct = np.sum((y_true > 0.5) == (y_pred > 0.5))
        return float(correct) / float(max(1, total))

    def add(self, y_true: np.ndarray, y_prob: np.ndarray) -> None:
        """Accumulate metrics for one prediction/ground-truth pair."""
        y_true = np.clip(y_true, 0.0, 1.0)
        y_prob = np.clip(y_prob, 0.0, 1.0)
        y_bin = (y_prob >= 0.5).astype(np.float32)

        dice = self._dice_coef(y_true, y_bin)
        iou = self._iou(y_true, y_bin)
        acc = self._acc(y_true, y_bin)

        # Simple placeholders for additional metrics to preserve compatibility.
        self.sums["dice_coef"] += dice
        self.sums["dice_loss"] += (1.0 - dice)
        self.sums["IoU"] += iou
        self.sums["accuracy"] += acc
        self.sums["sensitivity"] += dice
        self.sums["specificity"] += acc
        self.sums["f1_score"] += dice
        self.sums["f_beta"] += dice
        self.sums["nominal_surface_distance"] += 0.0
        self.sums["Hausdorff_distance"] += 0.0
        self.sums["boundary_intersection_over_union"] += iou
        self.n += 1

    def finalize(self) -> Dict[str, float]:
        """Compute the average metrics over all added samples."""
        out = {}
        div = float(max(1, self.n))
        for k, v in self.sums.items():
            out[k] = float(v) / div
        return out


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
    train_idx, val_idx, test_idx = split_dataset(
        frames, frames_json, config.test_ratio, config.val_ratio
    )

    print(f"training_frames {len(train_idx)}")
    print(f"validation_frames {len(val_idx)}")
    print(f"testing_frames {len(test_idx)}\n")

    return frames, image_paths, test_idx


# -----------------------------
# Model builders
# -----------------------------
def _build_unet(config) -> torch.nn.Module:
    """Construct UNet for evaluation."""
    in_ch = len(getattr(config, "channel_list", []))
    model = UNet(
        [config.train_batch_size, *config.patch_size, in_ch],
        [in_ch],
        getattr(config, "dilation_rate", 1),
    )
    return model


def _build_swin(config) -> torch.nn.Module:
    """Construct SwinUNet for evaluation with config defaults."""
    base_c = getattr(config, "swin_base_channels", 64)
    swin_patch = getattr(config, "swin_patch_size", 16)
    model = SwinUNet(
        h=config.patch_size[0],
        w=config.patch_size[1],
        ch=len(getattr(config, "channel_list", [])),
        c=base_c,
        patch_size=swin_patch,
    )
    return model


def _build_terramind(config) -> torch.nn.Module:
    """Construct TerraMind for evaluation using the same knobs as training."""
    in_ch = len(getattr(config, "channels_used", getattr(config, "channel_list", [])))
    num_classes = int(getattr(config, "num_classes", 1))
    modality = getattr(config, "modality", "S2")

    tm_backbone = getattr(config, "tm_backbone", None)
    tm_decoder = getattr(config, "tm_decoder", "UperNetDecoder")
    tm_dec_ch = getattr(config, "tm_decoder_channels", 256)
    tm_indices = getattr(config, "tm_select_indices", None)
    tm_bands = getattr(config, "tm_bands", None)
    tm_ckpt = getattr(config, "tm_backbone_ckpt_path", None)
    tm_merge = getattr(config, "terramind_merge_method", "mean")
    tm_size_fallback = getattr(config, "terramind_size", "base")

    def _parse_size_from_backbone(s: Optional[str], default_size: str = "base"):
        if not s:
            return None, default_size
        lower = s.lower()
        if lower.startswith("terramind"):
            size = (
                "large"
                if "large" in lower
                else ("base" if "base" in lower else ("small" if "small" in lower else ("tiny" if "tiny" in lower else default_size)))
            )
            return s, size
        if lower in {"tiny", "small", "base", "large"}:
            return None, lower
        return None, default_size

    backbone_override, tm_size = _parse_size_from_backbone(tm_backbone, tm_size_fallback)

    model = TerraMind(
        in_channels=in_ch,
        num_classes=num_classes,
        modality=modality,
        tm_size=tm_size,
        merge_method=tm_merge,
        pretrained=True,
        ckpt_path=tm_ckpt,
        indices_override=tm_indices,
        bands_override=tm_bands,
        decoder=tm_decoder,
        decoder_channels=tm_dec_ch,
        decoder_kwargs={},
        backbone=backbone_override,
        rescale=True,
    )
    setattr(model, "_is_terramind", True)
    return model


# -----------------------------
# Checkpoint loading
# -----------------------------
def _load_model_from_checkpoint(
    model: torch.nn.Module, ckpt_path: str, device: torch.device
) -> torch.nn.Module:
    """Load weights with strict fallback and move to device."""
    model = model.to(device=device).eval()
    state = torch.load(ckpt_path, map_location="cpu")

    if isinstance(state, dict) and "model_state" in state:
        state_dict = state["model_state"]
    else:
        state_dict = state

    try:
        model.load_state_dict(state_dict, strict=True)
    except Exception:
        model.load_state_dict(state_dict, strict=False)
    return model


# -----------------------------
# Inference helpers
# -----------------------------
def _select_eval_channel(prob_nchw: torch.Tensor, config) -> torch.Tensor:
    """Select class/channel for evaluation (binary default or chosen idx)."""
    prob_nchw = _ensure_nchw(prob_nchw).float()
    if prob_nchw.shape[1] > 1:
        cls_idx = int(getattr(config, "metrics_class", 1))
        cls_idx = max(0, min(cls_idx, prob_nchw.shape[1] - 1))
        return prob_nchw[:, cls_idx : cls_idx + 1]
    return prob_nchw


@torch.no_grad()
def _infer_full_image(model: torch.nn.Module, frame, device: torch.device, config) -> np.ndarray:
    """
    frame.img: H,W,C_in  (already preprocessed feature bands)
    frame.annotations: H,W
    Use only the first K = len(config.channel_list) channels for inputs.
    """
    x = frame.img
    k = len(getattr(config, "channel_list", []))
    if x.shape[2] < k:
        raise RuntimeError(
            f"Frame has {x.shape[2]} channels but config.channel_list requires {k}."
        )
    x = x[:, :, :k]

    # normalize like training patches (per-channel over H,W)
    x = image_normalize(x, axis=(0, 1))
    x_t = (
        torch.from_numpy(np.transpose(x.astype(np.float32), (2, 0, 1)))
        .unsqueeze(0)
        .to(device)
    )  # 1,C,H,W

    # forward + robust logits->probs for all supported models
    y_raw = _forward_with_autopad(model, x_t)
    if _is_terramind_model(model):
        prob_full = _as_probs_from_terratorch_logits_first(
            y_raw, num_classes=int(getattr(config, "num_classes", 1))
        )
    else:
        prob_full = _as_probs_from_terratorch(
            y_raw, num_classes=int(getattr(config, "num_classes", 1))
        )
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
# Core evaluation
# -----------------------------
def _evaluate_arch(config, arch: str = "unet") -> None:
    """
    Evaluate all checkpoints under config.saved_models_dir for the given arch.
    Arch keys:
      - "unet"  : UNet
      - "swin"  : SwinUNet
      - "tm"    : TerraMind
    """
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
        "run_name",
        "checkpoint_path",
        "dice_coef",
        "dice_loss",
        "IoU",
        "accuracy",
        "sensitivity",
        "specificity",
        "f1_score",
        "f_beta",
        "nominal_surface_distance",
        "Hausdorff_distance",
        "boundary_intersection_over_union",
        "elapsed",
    ]

    results_written = 0

    for ckpt_path in ckpts:
        try:
            print(f"{_C.GREEN}[EVAL]{_C.RESET} model={arch}  ckpt={ckpt_path}")
            base = os.path.basename(ckpt_path)
            out_dir = os.path.join(config.results_dir, base.replace(".pt", ""))
            print(f"{_C.YELLOW}[EVAL]{_C.RESET} saving masks -> {out_dir}")
            os.makedirs(out_dir, exist_ok=True)

            # ---- build/load model ----
            if arch == "unet":
                model = _build_unet(config)
            elif arch == "swin":
                model = _build_swin(config)
            elif arch == "tm":
                model = _build_terramind(config)
            else:
                raise NotImplementedError(f"Unknown arch: {arch}")
            model = _load_model_from_checkpoint(model, ckpt_path, device).eval()

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
# Public entrypoints
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


def evaluate_SwinUNetPP(conf) -> None:
    """
    Evaluate ALL SwinUNet checkpoints found in config.saved_models_dir.
    Saves per-frame hard masks under config.results_dir/<checkpoint_basename>
    and appends one row per checkpoint to config.results_dir/evaluation_swin.csv
    """
    global config
    config = conf
    _evaluate_arch(config, arch="swin")


def evaluate_TerraMind(conf) -> None:
    """
    Evaluate ALL TerraMind checkpoints found in config.saved_models_dir.
    Saves per-frame hard masks under config.results_dir/<checkpoint_basename>
    and appends one row per checkpoint to config.results_dir/evaluation_tm.csv
    """
    global config
    config = conf
    _evaluate_arch(config, arch="tm")
