# evaluation.py (PyTorch) - Full-image sliding-window evaluation with BF16 + channels-last
# Evaluates all checkpoints in saved_models_dir and computes dataset-level metrics
# matching the training implementation (core.losses). Outputs georeferenced prediction
# masks and (optionally) uncertainty maps, plus a CSV summary table.
#
# UPDATE (micro-averaging + robust uncertainty):
# - Pixel-level metrics (Dice/IoU/Acc/Sens/Spec/F1/Fbeta) are now MICRO-AVERAGED by
#   accumulating a global confusion matrix over all test pixels (thresholded at eval_threshold).
# - Geometric metrics (Hausdorff, nominal surface distance, boundary IoU) remain MACRO-AVERAGED
#   across scenes, as micro-averaging is not well-defined for them.
# - MC uncertainty supports Swin stochastic depth by enabling DropPath/StochasticDepth layers
#   during inference. If drop_path=0.0 (or no stochastic layers exist), epistemic uncertainty
#   will correctly collapse towards ~0 while aleatoric remains defined.

from __future__ import annotations

import csv as csv_module
import glob
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import rasterio
import torch
import torch.nn as nn
from tqdm import tqdm

# ===== Project imports =====
from core.Swin_UNetPP import SwinUNet
from core.TerraMind import TerraMind
from core.UNet import UNet
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
from core.losses import (
    Hausdorff_distance,
    IoU,
    accuracy,
    boundary_intersection_over_union,
    dice_coef,
    dice_loss,
    f1_score,
    f_beta,
    nominal_surface_distance,
    sensitivity,
    specificity,
)
from core.split_frames import split_dataset

# ===== Fast execution defaults / mixed precision =====
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True  # auto-tune kernels

# Prefer BF16 if supported (more stable than FP16); fallback to FP16 otherwise
_AMP_DTYPE = (
    torch.bfloat16
    if (torch.cuda.is_available() and torch.cuda.is_bf16_supported())
    else torch.float16
)


# -----------------------------
# Metric accumulator (MICRO for pixel metrics, MACRO for geometric metrics)
# -----------------------------
class MetricAccumulator:
    """
    Hybrid dataset-level accumulator:
      - MICRO-average (global pixel pool) for: Dice, Dice loss, IoU, accuracy,
        sensitivity, specificity, F1, F_beta.
      - MACRO-average (per-scene mean) for geometric/boundary metrics:
        nominal_surface_distance, Hausdorff_distance, boundary_intersection_over_union.

    This addresses heterogeneous foreground prevalence across test scenes.
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
        self.reset()

        # Macro-only (geometry/boundary) metrics using training implementations
        self.macro_metric_fns = {
            "nominal_surface_distance": nominal_surface_distance,
            "Hausdorff_distance": Hausdorff_distance,
            "boundary_intersection_over_union": boundary_intersection_over_union,
        }

    def reset(self) -> None:
        # Confusion matrix for micro-averaged pixel metrics
        self.tp = 0
        self.fp = 0
        self.tn = 0
        self.fn = 0

        # Macro accumulators for geometry metrics
        self.n_macro = 0
        self.macro_sums: Dict[str, float] = {
            "nominal_surface_distance": 0.0,
            "Hausdorff_distance": 0.0,
            "boundary_intersection_over_union": 0.0,
        }

    def _to_tensors(
        self, y_true_np: np.ndarray, y_prob_np: np.ndarray
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert [H,W] numpy arrays in [0,1] to [1,1,H,W] torch tensors
        on the target device for metric computation.
        """
        y_true = torch.from_numpy(y_true_np.astype(np.float32))
        y_prob = torch.from_numpy(y_prob_np.astype(np.float32))

        if y_true.ndim == 2:
            y_true = y_true.unsqueeze(0).unsqueeze(0)
            y_prob = y_prob.unsqueeze(0).unsqueeze(0)
        elif y_true.ndim == 3 and y_true.shape[0] == 1:
            y_true = y_true.unsqueeze(0)
            y_prob = y_prob.unsqueeze(0)
        else:
            raise ValueError(f"Unexpected GT shape for metrics: {y_true.shape}")

        y_true = y_true.to(self.device, non_blocking=True)
        y_prob = y_prob.to(self.device, non_blocking=True)
        return y_true, y_prob

    @staticmethod
    def _as_float(value: Any) -> float:
        if isinstance(value, torch.Tensor):
            return float(value.detach().mean().cpu().item())
        return float(value)

    def add(self, y_true_np: np.ndarray, y_prob_np: np.ndarray) -> None:
        """
        Update:
          - micro confusion matrix using thresholded predictions
          - macro geometry metrics using training implementations
        """
        y_true_np = np.clip(y_true_np, 0.0, 1.0)
        y_prob_np = np.clip(y_prob_np, 0.0, 1.0)

        # --- MICRO: global confusion matrix over all pixels ---
        gt = (y_true_np >= 0.5)
        pr = (y_prob_np >= self.threshold)

        self.tp += int(np.logical_and(pr, gt).sum())
        self.fp += int(np.logical_and(pr, np.logical_not(gt)).sum())
        self.fn += int(np.logical_and(np.logical_not(pr), gt).sum())
        self.tn += int(np.logical_and(np.logical_not(pr), np.logical_not(gt)).sum())

        # --- MACRO: per-scene geometric metrics (not meaningful to micro-average) ---
        y_true, y_prob = self._to_tensors(y_true_np, y_prob_np)
        y_bin = (y_prob >= self.threshold).float()

        for name, fn in self.macro_metric_fns.items():
            try:
                val = fn(y_true, y_prob)
            except Exception:
                val = fn(y_true, y_bin)
            self.macro_sums[name] += self._as_float(val)

        self.n_macro += 1

    def finalize(self) -> Dict[str, float]:
        """
        Return a dict with the same keys as the CSV header expects.
        Pixel metrics are MICRO-averaged; geometry metrics are MACRO-averaged.
        """
        eps = 1e-9

        tp = float(self.tp)
        fp = float(self.fp)
        tn = float(self.tn)
        fn = float(self.fn)

        # MICRO pixel metrics
        dice = (2.0 * tp) / (2.0 * tp + fp + fn + eps)
        iou = tp / (tp + fp + fn + eps)
        acc = (tp + tn) / (tp + tn + fp + fn + eps)
        sens = tp / (tp + fn + eps)
        spec = tn / (tn + fp + eps)

        prec = tp / (tp + fp + eps)
        rec = sens
        f1 = (2.0 * prec * rec) / (prec + rec + eps)

        b2 = self.fbeta_beta ** 2
        fbeta = (1.0 + b2) * (prec * rec) / (b2 * prec + rec + eps)

        out: Dict[str, float] = {
            "dice_coef": float(dice),
            "dice_loss": float(1.0 - dice),
            "IoU": float(iou),
            "accuracy": float(acc),
            "sensitivity": float(sens),
            "specificity": float(spec),
            "f1_score": float(f1),
            "f_beta": float(fbeta),
        }

        # MACRO geometry metrics
        div = float(max(1, self.n_macro))
        out["nominal_surface_distance"] = float(
            self.macro_sums["nominal_surface_distance"] / div
        )
        out["Hausdorff_distance"] = float(self.macro_sums["Hausdorff_distance"] / div)
        out["boundary_intersection_over_union"] = float(
            self.macro_sums["boundary_intersection_over_union"] / div
        )

        return out


# -----------------------------
# Data helpers (reuse training ordering)
# -----------------------------
def _list_preprocessed_paths(preprocessed_dir: str) -> List[str]:
    """Discover and sort preprocessed .tif frames by integer stem."""
    image_paths = sorted(
        glob.glob(os.path.join(preprocessed_dir, "*.tif")),
        key=lambda f: int(os.path.basename(f)[:-4]),
    )
    return image_paths


def _gather_frames_and_test_indices(config) -> Tuple[list, list, list]:
    """Load frames once and read train/val/test split indices from JSON."""
    frames = get_all_frames(config)  # Uses config + last-band label convention
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
# Model builders (match training constructors)
# -----------------------------
def _build_unet(config) -> torch.nn.Module:
    """Construct UNet for evaluation matching training architecture."""
    in_ch = len(getattr(config, "channel_list", []))
    model = UNet(
        [config.train_batch_size, *config.patch_size, in_ch],
        [in_ch],
        getattr(config, "dilation_rate", 1),
    )
    return model


def _build_swin(config) -> torch.nn.Module:
    """Construct SwinUNet for evaluation with config defaults (match training)."""
    base_c = getattr(config, "swin_base_channels", 64)
    swin_patch = getattr(config, "swin_patch_size", 16)
    swin_window = getattr(config, "swin_window", 7)

    in_ch = len(getattr(config, "channels_used", getattr(config, "channel_list", [])))

    model = SwinUNet(
        h=config.patch_size[0],
        w=config.patch_size[1],
        ch=in_ch,
        c=base_c,
        patch_size=swin_patch,
        window_size=swin_window,
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

    def _parse_size_from_backbone(
        s: Optional[str], default_size: str = "base"
    ) -> Tuple[Optional[str], str]:
        """Parse TerraMind size token from backbone string."""
        if not s:
            return None, default_size
        lower = s.lower()
        if lower.startswith("terramind"):
            size = (
                "large"
                if "large" in lower
                else (
                    "base"
                    if "base" in lower
                    else (
                        "small"
                        if "small" in lower
                        else ("tiny" if "tiny" in lower else default_size)
                    )
                )
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
# Inference helpers (sliding-window)
# -----------------------------
def _select_eval_channel(prob_nchw: torch.Tensor, config) -> torch.Tensor:
    """
    Select class/channel for evaluation (binary default or chosen idx).

    Expects [B,C,H,W] (or convertible via _ensure_nchw).
    Returns [B,1,H,W].
    """
    prob_nchw = _ensure_nchw(prob_nchw).float()
    if prob_nchw.shape[1] > 1:
        cls_idx = int(getattr(config, "metrics_class", 1))
        cls_idx = max(0, min(cls_idx, prob_nchw.shape[1] - 1))
        return prob_nchw[:, cls_idx : cls_idx + 1]
    return prob_nchw


def _enable_stochastic_inference(model: torch.nn.Module) -> int:
    """
    Enable stochastic layers for MC inference without switching the entire model to train().

    - Enables nn.Dropout* modules.
    - Enables DropPath / StochasticDepth-style modules by name (commonly used in Swin).
      These layers typically gate behavior on module.training.

    Returns:
        Number of modules switched to train mode.
    """
    n_enabled = 0
    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
            m.train()
            n_enabled += 1
            continue

        cls_name = m.__class__.__name__
        if cls_name in {"DropPath", "StochasticDepth"}:
            m.train()
            n_enabled += 1

    return n_enabled


def _infer_full_image(
    model: torch.nn.Module, frame, device: torch.device, config
) -> np.ndarray | Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Sliding-window full-image inference that mirrors the training forward path:
      - same channel selection (channels_used / channel_list)
      - same _forward_with_autopad
      - same TerraMind / non-TerraMind logits -> probabilities decode
      - same channels_last + AMP dtype

    If config.eval_mc_dropout is False, returns a single [H,W] numpy array of probabilities.

    If config.eval_mc_dropout is True, performs stochastic inference with
    config.eval_mc_samples forward passes per patch and returns:
        (mean_prob, epistemic_uncertainty, aleatoric_uncertainty) as [H,W].
    """
    # MC controls (match evaluation loop default)
    use_mc_dropout = bool(getattr(config, "eval_mc_dropout", True))
    mc_samples = int(getattr(config, "eval_mc_samples", 8))
    mc_samples = max(1, mc_samples)

    x_full = frame.img  # [H,W,C]
    img_h, img_w = x_full.shape[:2]

    channels_used = getattr(config, "channels_used", getattr(config, "channel_list", []))
    k = len(channels_used)

    if x_full.shape[2] < k:
        raise RuntimeError(
            f"Frame has {x_full.shape[2]} channels but config.channels_used/channel_list "
            f"requires {k}."
        )

    x_full = x_full[:, :, :k]

    patch_h, patch_w = int(config.patch_size[0]), int(config.patch_size[1])

    eval_stride = getattr(config, "eval_patch_stride", None)
    if eval_stride is None:
        stride_h, stride_w = patch_h, patch_w
    else:
        if isinstance(eval_stride, int):
            stride_h = stride_w = int(eval_stride)
        else:
            stride_h, stride_w = int(eval_stride[0]), int(eval_stride[1])

    # Pad to integer number of patches
    pad_h = (patch_h - (img_h % patch_h)) if (img_h % patch_h) != 0 else 0
    pad_w = (patch_w - (img_w % patch_w)) if (img_w % patch_w) != 0 else 0

    x_padded = np.pad(
        x_full,
        ((0, pad_h), (0, pad_w), (0, 0)),
        mode="reflect",
    )
    pad_h_total, pad_w_total = img_h + pad_h, img_w + pad_w

    # Optional normalization (kept off to match your current behaviour)
    # x_padded = image_normalize(x_padded, axis=(0, 1))

    # Blending accumulators
    prob_accum = np.zeros((pad_h_total, pad_w_total), dtype=np.float32)
    weight_accum = np.zeros((pad_h_total, pad_w_total), dtype=np.float32)

    if use_mc_dropout:
        epi_accum = np.zeros((pad_h_total, pad_w_total), dtype=np.float32)
        alea_accum = np.zeros((pad_h_total, pad_w_total), dtype=np.float32)
    else:
        epi_accum = alea_accum = None  # type: ignore[assignment]

    # Use training batch size to avoid TerraTorch batch_size=1 quirks
    batch_size = int(getattr(config, "train_batch_size", 1))
    batch_size = max(1, batch_size)

    num_classes = int(getattr(config, "num_classes", 1))

    # Eval mode overall; enable stochastic layers if requested
    model.eval()
    if use_mc_dropout:
        n_stoch = _enable_stochastic_inference(model)
        if n_stoch == 0 and not getattr(model, "_warned_no_stochastic", False):
            print(
                _col(
                    "[EVAL] MC dropout requested but no Dropout/DropPath layers were enabled. "
                    "Epistemic uncertainty may be ~0 (expected if drop_path=0.0).",
                    _C.YELLOW,
                )
            )
            setattr(model, "_warned_no_stochastic", True)

    def _forward_probs(x_t: torch.Tensor) -> torch.Tensor:
        """Forward pass returning probabilities as [B,C,H,W]."""
        if _is_terramind_model(model):
            y_raw = _forward_with_autopad(model, x_t)
        else:
            y_raw = model(x_t)

        if getattr(model, "_returns_probabilities", False):
            prob_full = _ensure_nchw(y_raw).float()
        elif _is_terramind_model(model):
            prob_full = _as_probs_from_terratorch_logits_first(
                y_raw, num_classes=num_classes
            )
        else:
            prob_full = _as_probs_from_terratorch(y_raw, num_classes=num_classes)

        return prob_full

    with torch.no_grad():
        for y0 in range(0, pad_h_total - patch_h + 1, stride_h):
            y1 = y0 + patch_h
            for x0 in range(0, pad_w_total - patch_w + 1, stride_w):
                x1 = x0 + patch_w

                patch_np = x_padded[y0:y1, x0:x1, :]  # [H,W,C]
                patch_t = torch.from_numpy(
                    np.transpose(patch_np.astype(np.float32), (2, 0, 1))
                ).unsqueeze(0)  # [1,C,H,W]

                patch_t = patch_t.to(device, non_blocking=True).contiguous(
                    memory_format=torch.channels_last
                )

                if batch_size > 1:
                    x_t = (
                        patch_t.repeat(batch_size, 1, 1, 1)
                        .contiguous(memory_format=torch.channels_last)
                    )
                else:
                    x_t = patch_t

                with torch.cuda.amp.autocast(
                    enabled=torch.cuda.is_available(), dtype=_AMP_DTYPE
                ):
                    if not use_mc_dropout:
                        prob_full = _forward_probs(x_t)
                        prob_sel = _select_eval_channel(prob_full, config)

                        prob_patch = (
                            prob_sel[0:1]
                            .squeeze(0)
                            .squeeze(0)
                            .clamp(0.0, 1.0)
                            .detach()
                            .cpu()
                            .numpy()
                        )
                        prob_accum[y0:y1, x0:x1] += prob_patch
                        weight_accum[y0:y1, x0:x1] += 1.0
                    else:
                        mc_patches: List[np.ndarray] = []
                        for _ in range(mc_samples):
                            prob_full = _forward_probs(x_t)
                            prob_sel = _select_eval_channel(prob_full, config)

                            prob_patch_sample = (
                                prob_sel[0:1]
                                .squeeze(0)
                                .squeeze(0)
                                .clamp(0.0, 1.0)
                                .detach()
                                .cpu()
                                .numpy()
                            )
                            mc_patches.append(prob_patch_sample)

                        mc_stack = np.stack(mc_patches, axis=0).astype(np.float32)  # [S,H,W]
                        mean_patch = mc_stack.mean(axis=0).astype(np.float32)

                        # Aleatoric uncertainty: E[p(1-p)]
                        alea_patch = np.mean(mc_stack * (1.0 - mc_stack), axis=0).astype(
                            np.float32
                        )

                        # Epistemic uncertainty: Var[p]
                        second_moment = np.mean(mc_stack ** 2, axis=0)
                        epi_patch = (second_moment - mean_patch ** 2).astype(np.float32)

                        prob_accum[y0:y1, x0:x1] += mean_patch
                        epi_accum[y0:y1, x0:x1] += epi_patch
                        alea_accum[y0:y1, x0:x1] += alea_patch
                        weight_accum[y0:y1, x0:x1] += 1.0

    weight_accum = np.maximum(weight_accum, 1e-6)
    prob_full_padded = np.clip(prob_accum / weight_accum, 0.0, 1.0)
    prob_full = prob_full_padded[:img_h, :img_w]

    if not use_mc_dropout:
        return prob_full

    epi_full = (epi_accum / weight_accum)[:img_h, :img_w]
    alea_full = (alea_accum / weight_accum)[:img_h, :img_w]
    return prob_full, epi_full, alea_full


# -----------------------------
# Checkpoint discovery
# -----------------------------
def _find_all_checkpoints(folder: str) -> List[str]:
    """Discover all .pt checkpoint files sorted by modification time."""
    pats = [
        os.path.join(folder, "*.pt"),
        os.path.join(folder, "*.weights.pt"),
        os.path.join(folder, "*.raw.weights.pt"),
    ]
    fps: List[str] = []
    for p in pats:
        fps.extend(glob.glob(p))
    fps = sorted(list(set(fps)), key=lambda f: os.stat(f).st_mtime)
    return fps


# -----------------------------
# CSV writing (append for all models)
# -----------------------------
def _append_results_row(csv_path: str, header: List[str], row: List[str]) -> None:
    """Append a result row to CSV, creating file and header if needed."""
    write_header = not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "a") as f:
        if write_header:
            f.write(",".join(header) + "\n")
        f.write(",".join(map(str, row)) + "\n")


# -----------------------------
# Core evaluation loop
# -----------------------------
def _evaluate_arch(config, arch: str = "unet") -> None:
    """
    Evaluate all checkpoints under config.saved_models_dir for the given arch.

    For each checkpoint:
      - Loads model weights
      - Runs sliding-window inference on all test frames
      - Computes dataset-level metrics:
          * MICRO-average pixel metrics via global confusion matrix
          * MACRO-average geometric metrics across test scenes
      - Saves georeferenced prediction masks as GeoTIFFs
      - Optionally saves per-pixel epistemic/aleatoric uncertainty maps
      - Appends summary row to CSV
    """
    print(f"{_C.CYAN}Starting evaluation for architecture:{arch}{_C.RESET}")
    print(f"{_C.YELLOW}MAKE SURE TO USE FULL AREAS FROM PREPROCESSED DIR{_C.RESET}")
    print(
        f"{_C.YELLOW}CURRENTLY USING THIS PATH AS INPUT: {config.preprocessed_dir}{_C.RESET}\n"
    )

    selected_gpu = getattr(config, "selected_gpu", 0)
    if selected_gpu == -1 or not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(f"cuda:{selected_gpu}")
        torch.cuda.set_device(device)

    # ---- Load data once ----
    frames, image_paths, test_idx = _gather_frames_and_test_indices(config)
    print(f"Testing frames: {len(test_idx)}")

    # ---- Discover checkpoints ----
    ckpts = _find_all_checkpoints(config.saved_models_dir)
    print(f"Found {len(ckpts)} checkpoint(s).\n")

    # ---- CSV path ----
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
        "mean_epistemic_uncertainty",
        "mean_aleatoric_uncertainty",
        "elapsed",
    ]

    results_written = 0
    thr = float(getattr(config, "eval_threshold", 0.5))
    use_mc_dropout = bool(getattr(config, "eval_mc_dropout", True))
    fbeta_beta = float(getattr(config, "eval_fbeta_beta", 2.0))

    # ---- Evaluate each checkpoint ----
    for ckpt_path in ckpts:
        try:
            print(f"\n{'='*80}")
            print(f"{_C.GREEN}[EVAL]{_C.RESET} model={arch}  ckpt={ckpt_path}")
            base = os.path.basename(ckpt_path)
            out_dir = os.path.join(config.results_dir, base.replace(".pt", ""))
            print(f"{_C.YELLOW}[EVAL]{_C.RESET} saving masks -> {out_dir}")
            os.makedirs(out_dir, exist_ok=True)

            # ---- Build and load model ----
            if arch == "unet":
                model = _build_unet(config)
            elif arch == "swin":
                model = _build_swin(config)
            elif arch == "tm":
                model = _build_terramind(config)
            else:
                raise NotImplementedError(f"Unknown arch: {arch}")

            model = _load_model_from_checkpoint(model, ckpt_path, device).eval()

            # ---- Prediction loop ----
            accum = MetricAccumulator(
                device=device, threshold=thr, fbeta_beta=fbeta_beta
            )
            t0 = time.time()

            sum_epistemic = 0.0
            sum_aleatoric = 0.0
            unc_pixel_count = 0

            for i in tqdm(range(len(test_idx)), desc=f"Predicting ({arch})"):
                idx = test_idx[i]
                frame = frames[idx]
                im_fp = image_paths[idx]

                # Predict prob map (and uncertainty if enabled)
                if use_mc_dropout:
                    prob, epi_map, alea_map = _infer_full_image(
                        model, frame, device, config
                    )
                    sum_epistemic += float(epi_map.sum())
                    sum_aleatoric += float(alea_map.sum())
                    unc_pixel_count += int(epi_map.size)
                else:
                    prob = _infer_full_image(model, frame, device, config)

                # Load GT and scale to [0,1]
                gt = frame.annotations.astype(np.float32)
                if gt.max() > 1.5:
                    gt = gt / 255.0
                gt = np.clip(gt, 0.0, 1.0)

                # Accumulate dataset metrics (micro + macro hybrid)
                accum.add(gt, prob)

                # Save hard mask and probability mask as GeoTIFF
                pred_bin = (prob >= thr).astype(np.uint8)
                with rasterio.open(im_fp) as src:
                    base_profile = src.profile.copy()
                    base_profile.update(
                        count=1,
                        compress="LZW",
                        crs=src.crs,
                        transform=src.transform,
                        width=src.width,
                        height=src.height,
                    )

                    # Save hard mask (uint8)
                    mask_profile = base_profile.copy()
                    mask_profile["dtype"] = "uint8"
                    out_fp = os.path.join(out_dir, os.path.basename(im_fp))
                    with rasterio.open(out_fp, "w", **mask_profile) as dst:
                        dst.write(pred_bin, 1)

                    # Save probability mask (float32)
                    prob_profile = base_profile.copy()
                    prob_profile["dtype"] = "float32"
                    stem, ext = os.path.splitext(os.path.basename(im_fp))
                    prob_fp = os.path.join(out_dir, f"{stem}_prob{ext}")
                    with rasterio.open(prob_fp, "w", **prob_profile) as dst:
                        dst.write((prob * 255.0).astype(np.float32), 1)

                    if use_mc_dropout:
                        unc_profile = base_profile.copy()
                        unc_profile["dtype"] = "float32"

                        stem, ext = os.path.splitext(os.path.basename(im_fp))
                        epi_fp = os.path.join(out_dir, f"{stem}_epistemic{ext}")
                        alea_fp = os.path.join(out_dir, f"{stem}_aleatoric{ext}")

                        with rasterio.open(epi_fp, "w", **unc_profile) as dst:
                            dst.write(epi_map.astype(np.float32), 1)
                        with rasterio.open(alea_fp, "w", **unc_profile) as dst:
                            dst.write(alea_map.astype(np.float32), 1)

            # ---- Finalize metrics ----
            if use_mc_dropout and unc_pixel_count > 0:
                mean_epistemic = sum_epistemic / float(unc_pixel_count)
                mean_aleatoric = sum_aleatoric / float(unc_pixel_count)
            else:
                mean_epistemic = float("nan")
                mean_aleatoric = float("nan")

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
                f"{mean_epistemic:.6f}",
                f"{mean_aleatoric:.6f}",
                elapsed,
            ]
            _append_results_row(csv_path, header, row)
            results_written += 1

        except Exception as exc:
            print(_col(f"Evaluation failed for {ckpt_path}: {exc}", _C.RED))

    # ---- Print summary table at the end ----
    if results_written == 0:
        print(_col("No results to write.", _C.YELLOW))
    else:
        print(_col(f"\nWrote {results_written} result rows to {csv_path}", _C.GREEN))

        print(f"\n{'='*80}")
        print(_col(f"EVALUATION SUMMARY ({arch.upper()})", _C.GREEN))
        print(f"{'='*80}\n")

        if os.path.exists(csv_path):
            with open(csv_path, "r") as f:
                reader = csv_module.DictReader(f)
                rows = list(reader)

                if rows:
                    print(
                        f"{'Checkpoint':<50} {'Dice':<10} {'IoU':<10} "
                        f"{'Acc':<10} {'Hausd':<10} {'F1':<10} {'Time':<10}"
                    )
                    print("-" * 110)

                    for row in rows:
                        ckpt_name = os.path.basename(row.get("checkpoint_path", ""))[:45]
                        dice = row.get("dice_coef", "N/A")
                        iou = row.get("IoU", "N/A")
                        acc = row.get("accuracy", "N/A")
                        hausdorff_distance = row.get("Hausdorff_distance", "N/A")
                        f1 = row.get("f1_score", "N/A")
                        elapsed = row.get("elapsed", "N/A")

                        print(
                            f"{ckpt_name:<50} {dice:<10} {iou:<10} "
                            f"{acc:<10} {hausdorff_distance:<10} {f1:<10} {elapsed:<10}"
                        )

                    print(f"\nFull results saved to: {csv_path}")
        print(f"{'='*80}\n")


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
