# training.py (PyTorch) - BF16 + channels-last, EMA, progress bars, better visuals (raw | mask), robust logits->probs
import glob
import json
import os
import shutil
import time
from contextlib import contextmanager
from datetime import datetime, timedelta
from typing import Any, Dict, Iterable, Iterator, Optional, Tuple

import h5py  # kept for old .h5 checkpoints compatibility in case I have a TF model. I dont think there are any left
import numpy as np
import pandas as pd
import rasterio
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.utils as vutils
from torch.utils.data import DataLoader, IterableDataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

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

# ===== project imports =====
from core.Swin_UNetPP import SwinUNet
from core.TerraMind import TerraMind
from core.UNet import UNet
from core.common.console import _C, _col, _fmt_seconds, _format_logs_for_print
from core.common.data import create_train_val_datasets, get_all_frames
from core.common.model_utils import (
    ModelEMA,
    _as_probs_from_terratorch,
    _as_probs_from_terratorch_logits_first,
    _ensure_nchw,
    _forward_with_autopad,
    _is_terramind_model,
    set_global_seed,
)
from core.common.vis import _log_triptych_and_optional_heatmap
from core.dataset_generator import DataGenerator as Generator
from core.frame_info import FrameInfo
from core.losses import (
    Hausdorff_distance,
    IoU,
    accuracy,
    boundary_intersection_over_union,
    dice_coef,
    dice_loss,
    f1_score,
    f_beta,
    get_loss,
    nominal_surface_distance,
    sensitivity,
    specificity,
)
from core.optimizers import get_optimizer
from core.split_frames import split_dataset, summarize_positive_rates

# -----------------------------
# Global config holder
# -----------------------------
config = None  # set in train_UNet / train_SwinUNetPP etc


# -----------------------------
# Build models
# -----------------------------
def _build_model_unet() -> nn.Module:
    model = UNet(
        [config.train_batch_size, *config.patch_size, len(config.channel_list)],
        [len(config.channel_list)],
        config.dilation_rate,
    )
    return model


def _build_model_swin() -> nn.Module:
    base_c = getattr(config, "swin_base_channels", 64)
    swin_patch_size = getattr(config, "swin_patch_size", 16)
    model = SwinUNet(
        h=config.patch_size[0],
        w=config.patch_size[1],
        ch=len(getattr(config, "channels_used", config.channel_list)),
        c=base_c,
        patch_size=swin_patch_size,
    )
    return model


def _build_model_terramind() -> nn.Module:
    # ---- derive channels ----
    in_ch = len(
        getattr(config, "channels_used", getattr(config, "channel_list", []))
    )
    num_classes = int(getattr(config, "num_classes", 1))
    modality = getattr(config, "modality", "S2")

    # ---- map your tm_* config to TerraMind kwargs ----
    tm_backbone = getattr(config, "tm_backbone", None)  # e.g. "terramind_v1_large"
    tm_decoder = getattr(config, "tm_decoder", "UperNetDecoder")  # "UperNetDecoder" | "UNetDecoder" | ...
    tm_dec_ch = getattr(config, "tm_decoder_channels", 256)  # int or list[int]
    tm_indices = getattr(config, "tm_select_indices", None)  # list[int] or None
    tm_bands = getattr(config, "tm_bands", None)  # list[str] or None
    tm_ckpt = getattr(config, "tm_backbone_ckpt_path", None)
    tm_merge = getattr(config, "terramind_merge_method", "mean")
    tm_size_fallback = getattr(config, "terramind_size", "base")

    # Accept either a direct backbone name or infer size from the suffix
    def _parse_size_from_backbone(
        s: str | None, default_size: str = "base"
    ) -> tuple[str | None, str]:
        if not s:
            return None, default_size
        lower = s.lower()
        if lower.startswith("terramind"):
            size = "large" if "large" in lower else (
                "base" if "base" in lower else (
                    "small" if "small" in lower else (
                        "tiny" if "tiny" in lower else default_size
                    )
                )
            )
            return s, size
        # if it is not a full terramind id, treat it like a size token
        if lower in {"tiny", "small", "base", "large"}:
            return None, lower
        return None, default_size

    backbone_override, tm_size = _parse_size_from_backbone(
        tm_backbone, tm_size_fallback
    )

    # Optional extra decoder kwargs
    decoder_kwargs = {}

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
        decoder_kwargs=decoder_kwargs,
        backbone=backbone_override,  # direct backbone string, if provided
        rescale=True,
    )

    # mark for downstream detection even if compiled later, technically does only matter for printing
    setattr(model, "_is_terramind", True)

    # Optional: full-backbone freeze at init
    if bool(getattr(config, "tm_freeze_backbone", False)):
        for name, p in model.named_parameters():
            if "backbone" in name:
                p.requires_grad = False

    return model


class BestModelSaver:
    """Saves best model weights based on a monitored value (her minimal loss)."""

    def __init__(self, model_path: str):
        self.model_path = model_path
        self.best_val = float("inf")

    def maybe_save(self, model: nn.Module, current_val: float):
        prev = self.best_val
        if current_val < self.best_val:
            self.best_val = current_val
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            # Save (can be  EMA) weights
            torch.save(model.state_dict(), f"{self.model_path}.weights.pt")
            print(
                _col(
                    f"New best! val_loss improved {prev:.6f} -> {current_val:.6f}. "
                    f"Saved: {self.model_path}.weights.pt",
                    _C.GREEN,
                )
            )
            return True  # indicate improvement
        return False


def _create_logging(model_path: str, log_suffix: str = "") -> Tuple[BestModelSaver, str]:
    """Define logging paths and return a best-saver + log_dir."""
    log_dir = os.path.join(
        config.logs_dir, os.path.basename(model_path) + log_suffix
    )
    os.makedirs(log_dir, exist_ok=True)
    best_saver = BestModelSaver(model_path)
    return best_saver, log_dir


class MetricsCSVLogger:
    """Append all logs each epoch to a CSV in the run log dir."""

    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self.metrics_df: Optional[pd.DataFrame] = None

    def update(self, epoch: int, logs: Dict[str, Any]):
        if self.metrics_df is None:
            self.metrics_df = pd.DataFrame()
        new_row = pd.DataFrame(logs, index=[epoch])
        self.metrics_df = pd.concat([self.metrics_df, new_row], ignore_index=True)
        self.metrics_df.to_csv(self.csv_path, index=False)


class HeavyMetricsEvaluator:
    """
    Compute heavy metrics on a small validation subset at each epoch end.
    Writes to TensorBoard and merges into logs dict.
    """

    def __init__(
        self,
        val_iterable,
        log_dir: str,
        steps: int = 50,
        threshold: float = 0.5,
        device: Optional[torch.device] = None,
    ):
        self.val_iterable = val_iterable
        self.steps = steps
        self.threshold = threshold
        self.tb_writer = SummaryWriter(os.path.join(log_dir, "heavy_metrics"))
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.metric_fns = {
            "specificity": specificity,
            "sensitivity": sensitivity,
            "f_beta": f_beta,
            "f1_score": f1_score,
            "IoU": IoU,
            "nominal_surface_distance": nominal_surface_distance,
            "Hausdorff_distance": Hausdorff_distance,
            "boundary_intersection_over_union": boundary_intersection_over_union,
            "dice_loss": dice_loss,
        }

    @staticmethod
    def _as_float(v) -> float:
        try:
            if isinstance(v, torch.Tensor):
                return float(v.detach().mean().cpu().item())
            return float(v)
        except Exception:
            return float("nan")

    def run(self, model: nn.Module, epoch: int, logs: Dict[str, Any]) -> Dict[str, Any]:
        model.eval()
        accum: Dict[str, list] = {k: [] for k in self.metric_fns.keys()}

        with torch.no_grad():
            it = iter(self.val_iterable)
            for _ in range(self.steps):
                try:
                    x, y_true = next(it)
                except StopIteration:
                    break

                x = x.to(
                    self.device, non_blocking=True
                ).contiguous(memory_format=torch.channels_last)
                y_true = y_true.to(self.device, non_blocking=True)

                # TerraMind-aware probabilities
                y_pred_raw = _forward_with_autopad(model, x)
                num_classes = int(getattr(config, "num_classes", 1))
                if _is_terramind_model(model):
                    y_prob_full = _as_probs_from_terratorch_logits_first(
                        y_pred_raw, num_classes=num_classes
                    )
                else:
                    y_prob_full = _as_probs_from_terratorch(
                        y_pred_raw, num_classes=num_classes
                    )
                y_prob_full = _ensure_nchw(y_prob_full).float()
                if y_prob_full.shape[1] > 1:
                    cls_idx = int(getattr(config, "metrics_class", 1))
                    cls_idx = max(0, min(cls_idx, y_prob_full.shape[1] - 1))
                    y_prob = y_prob_full[:, cls_idx : cls_idx + 1]
                else:
                    y_prob = y_prob_full

                y_bin = (y_prob >= self.threshold).float()

                for name, fn in self.metric_fns.items():
                    try:
                        val = fn(y_true, y_prob)
                    except Exception:
                        val = fn(y_true, y_bin)
                    accum[name].append(self._as_float(val))

        for name, values in accum.items():
            if values:
                mean_val = sum(values) / len(values)
                self.tb_writer.add_scalar(name, mean_val, epoch)
                logs[f"val_{name}"] = mean_val
        self.tb_writer.flush()
        return logs


def _print_run_banner(model_key: str, log_dir: str):
    """Pretty console banner like tuning, including the loss function. Waste of space, but I forget my confg sometimes."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ema_on = bool(getattr(config, "use_ema", False))
    aug = float(getattr(config, "augmenter_strength", 1.0))
    minpos = float(getattr(config, "min_pos_frac", 0.0))
    posr = getattr(config, "pos_ratio", None)
    workers = int(getattr(config, "fit_workers", 8))
    stride = getattr(config, "patch_stride", None)
    loss_name = getattr(config, "loss_fn", "tversky")
    ab = getattr(config, "tversky_alphabeta", (0.5, 0.5))
    optimizer_name = getattr(config, "optimizer_fn", "adam")

    def _dtype_str(dt):
        if dt is torch.bfloat16:
            return "bf16"
        if dt is torch.float16:
            return "fp16"
        if dt is torch.float32:
            return "fp32"
        return str(dt)

    print("\n" + "=" * 90)
    print(
        f"[{model_key.upper()}][TRAIN]  run={getattr(config, 'run_name', 'run')}  "
        f"model_name={config.model_name}"
    )
    print(
        f"[{model_key.upper()}][TRAIN]  device={device}, "
        f"amp_dtype={_dtype_str(_AMP_DTYPE)}, channels_last=True"
    )
    print(
        f"[{model_key.upper()}][TRAIN]  ema={ema_on} "
        f"(decay={getattr(config, 'ema_decay', 0.999)})"
    )
    print(
        f"[{model_key.upper()}][TRAIN]  epochs={config.num_epochs}, "
        f"steps/epoch={config.num_training_steps}, "
        f"val_steps={config.num_validation_images}, "
        f"batch={config.train_batch_size}, workers={workers}"
    )
    print(
        f"[{model_key.upper()}][TRAIN]  patch={config.patch_size}, stride={stride}, "
        f"aug={aug}, min_pos_frac={minpos}, pos_ratio={posr}"
    )
    if hasattr(config, "swin_patch_size"):
        print(
            f"[{model_key.upper()}][TRAIN]  swin_patch="
            f"{getattr(config,'swin_patch_size',16)}, "
            f"window={getattr(config,'swin_window',4)}"
        )
    # Explicit loss + optimizer print
    if str(loss_name).lower().startswith("tversky"):
        print(
            f"[{model_key.upper()}][TRAIN]  loss={loss_name} "
            f"(alpha={ab[0]:.2f}, beta={ab[1]:.2f}), optimizer={optimizer_name}"
        )
    else:
        print(
            f"[{model_key.upper()}][TRAIN]  loss={loss_name}, optimizer={optimizer_name}"
        )
    print(f"[{model_key.upper()}][TRAIN]  logs_dir={log_dir}")
    print("=" * 90 + "\n")


# -----------------------------
# Fit loop
# -----------------------------
def _fit_model(
    model: nn.Module,
    train_iterable,
    val_iterable,
    model_path: str,
    starting_epoch: int = 0,
    log_name: str = "",
    optimizer: Optional[optim.Optimizer] = None,
    criterion: Optional[nn.Module] = None,
) -> None:
    best_saver, log_dir = _create_logging(model_path, log_suffix=log_name)

    # TensorBoard writers
    tb = SummaryWriter(log_dir)

    csv_path = os.path.join(
        config.logs_dir, f"{os.path.basename(model_path)}_metrics.csv"
    )
    csv_logger = MetricsCSVLogger(csv_path)

    # Heavy metrics evaluator
    val_eval_steps = int(getattr(config, "heavy_eval_steps", 50))
    heavy_eval = HeavyMetricsEvaluator(
        val_iterable,
        log_dir,
        steps=val_eval_steps,
        threshold=float(getattr(config, "eval_threshold", 0.5)),
    )

    # Device + memory format
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).to(memory_format=torch.channels_last)

    # ===>>> Overfit-one-batch mode: For debugging only!!!
    if getattr(config, "overfit_one_batch", False):
        print("==== Overfitting on one Batch ====")
        # get one CP batch 
        first_it = iter(train_iterable)
        first_x, first_y = next(first_it)

        # use exact same tensors every step
        def _infinite_one_batch():
            while True:
                # clone to avoid any accidental  operationss piling up
                yield first_x.clone(), first_y.clone()

        # validation also on the same batch 
        def _repeat_val(n):
            for _ in range(int(n)):
                yield first_x.clone(), first_y.clone()

        train_iterable = _infinite_one_batch()
        val_iterable = _repeat_val(getattr(config, "num_validation_images", 10))

    # EMA makes better generalisation (at least literature says so)
    use_ema = bool(getattr(config, "use_ema", False))
    ema_decay = float(getattr(config, "ema_decay", 0.999))
    ema = ModelEMA(model, decay=ema_decay) if use_ema else None
    eval_with_ema = bool(getattr(config, "eval_with_ema", False))

    # GradScaler only for FP16 (no need for BF16)
    use_fp16 = torch.cuda.is_available() and (_AMP_DTYPE is torch.float16)
    scaler = torch.cuda.amp.GradScaler(enabled=use_fp16)

    assert optimizer is not None and criterion is not None, "optimizer/criterion are required"

    # Light metrics during training for speed
    light_metric_fns = [dice_coef, accuracy]

    steps_per_epoch = int(config.num_training_steps)
    val_steps = int(getattr(config, "num_validation_images", 0))
    total_epochs = int(config.num_epochs)

    grad_accum = int(getattr(config, "steps_per_execution", 1))
    clip_norm = float(getattr(config, "clip_norm", 0.0))  # 0 disables
    global_step = 0
    model_save_interval = getattr(config, "model_save_interval", None)
    log_visuals_every = int(getattr(config, "log_visuals_every", 5))
    vis_rgb_idx = tuple(getattr(config, "vis_rgb_idx", (0, 1, 2)))

    # Verbosity knobs
    verbose = bool(getattr(config, "train_verbose", True))
    epoch_log_every = int(getattr(config, "train_epoch_log_every", 1))
    print_heavy = bool(getattr(config, "train_print_heavy", True))
    show_progress = bool(getattr(config, "show_progress", True))

    # Print run banner
    key = "unet" if log_name == "_unet" else ("tm" if log_name == "_tm" else "swin")
    _print_run_banner(key, log_dir)

    # Track the best (by val_loss)
    best_val_loss = float("inf")

    # --- training epochs ---
    for epoch in range(starting_epoch, total_epochs):
        t0 = time.time()
        model.train()
        logs: Dict[str, Any] = {}
        train_loss_accum = 0.0
        metric_accums = {fn.__name__: 0.0 for fn in light_metric_fns}

        # Iterate fixed number of steps per epoch (with progress bar)
        train_range = range(steps_per_epoch)
        if show_progress:
            train_range = tqdm(
                train_range, desc=f"Epoch {epoch+1}/{total_epochs} [train]", leave=False
            )

        train_it = iter(train_iterable)
        for step in train_range:
            try:
                x, y = next(train_it)
            except StopIteration:
                train_it = iter(train_iterable)
                x, y = next(train_it)

            # Move + channels-last
            x = x.to(
                device, non_blocking=True
            ).contiguous(memory_format=torch.channels_last)
            y = y.to(device, non_blocking=True)

            with torch.cuda.amp.autocast(
                enabled=torch.cuda.is_available(), dtype=_AMP_DTYPE
            ):
                y_pred_raw = _forward_with_autopad(model, x)
                # TerraMind-only: decode logits->probs, else use default extractor
                num_classes = int(getattr(config, "num_classes", 1))
                if _is_terramind_model(model):
                    y_prob_full = _as_probs_from_terratorch_logits_first(
                        y_pred_raw, num_classes=num_classes
                    )
                else:
                    y_prob_full = _as_probs_from_terratorch(
                        y_pred_raw, num_classes=num_classes
                    )
                y_prob_full = _ensure_nchw(y_prob_full).float()
                # For binary-style losses/metrics, pick a single class if needed
                if y_prob_full.shape[1] > 1:
                    cls_idx = int(getattr(config, "metrics_class", 1))
                    cls_idx = max(0, min(cls_idx, y_prob_full.shape[1] - 1))
                    y_prob = y_prob_full[:, cls_idx : cls_idx + 1]
                else:
                    y_prob = y_prob_full
                # warn if gradients were severed (e.g. from argmax) 
                if _is_terramind_model(model) and not y_prob.requires_grad:
                    print(
                        "WARNING: TerraMind y_prob has no grad - check decode path."
                    )

                loss = criterion(y, y_prob)

            scaler.scale(loss / grad_accum).backward()

            # Gradient step
            if (step + 1) % grad_accum == 0:
                if clip_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), max_norm=clip_norm
                    )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                if ema is not None:
                    ema.update()

            # track loss & light metrics
            train_loss_accum += float(loss.detach().cpu().item())
            for fn in light_metric_fns:
                try:
                    val = fn(y, y_prob)
                except Exception:
                    val = fn(y, (y_prob >= 0.5).float())
                metric_accums[fn.__name__] += float(
                    val.detach().mean().cpu().item()
                    if isinstance(val, torch.Tensor)
                    else float(val)
                )

            global_step += 1

            # Step-wise visuals: [ RGB | PRED | GT ]
            if log_visuals_every > 0 and ((global_step % log_visuals_every) == 0):
                try:
                    _log_triptych_and_optional_heatmap(
                        tb=tb,
                        tag_prefix="viz/train",
                        x=x,
                        y_prob=y_prob,
                        y_true=y,
                        step=global_step,
                        rgb_idx=vis_rgb_idx,
                        threshold=float(getattr(config, "eval_threshold", 0.5)),
                        cls_idx=getattr(config, "viz_class", 1),  # pick class 1 by default
                        add_heatmap=True,
                    )
                except Exception:
                    pass

            # Live postfix for progress
            if show_progress:
                avg_tr = train_loss_accum / (step + 1)
                postfix = {
                    "loss": f"{avg_tr:.4f}",
                    "dice": f"{(metric_accums['dice_coef']/(step+1)):.4f}",
                    "acc": f"{(metric_accums['accuracy']/(step+1)):.4f}",
                }
                try:
                    train_range.set_postfix(postfix)
                except Exception:
                    pass

        # Epoch-level averages
        avg_train_loss = train_loss_accum / max(1, steps_per_epoch)
        logs["loss"] = avg_train_loss
        for fn in light_metric_fns:
            logs[fn.__name__] = metric_accums[fn.__name__] / max(1, steps_per_epoch)

        # Validation (optionally with EMA weights)
        model.eval()
        val_loss_accum = 0.0
        val_metric_accums = {fn.__name__: 0.0 for fn in light_metric_fns}

        @contextmanager
        def maybe_ema_ctx():
            if eval_with_ema and ema is not None:
                with ema.use_ema_weights(model):
                    yield
            else:
                yield

        best_improved = False
        with maybe_ema_ctx():
            # Validation loop with progress bar
            val_range = range(max(1, val_steps)) if val_steps > 0 else range(0)
            if show_progress and val_steps > 0:
                val_range = tqdm(
                    val_range, desc=f"Epoch {epoch+1}/{total_epochs} [val]", leave=False
                )

            with torch.no_grad():
                val_it = iter(val_iterable)
                x_vis = None
                y_vis = None
                y_hat_vis = None  # for epoch-end viz
                val_count = 0
                for _ in val_range:
                    try:
                        x, y = next(val_it)
                    except StopIteration:
                        break
                    val_count += 1
                    x = x.to(
                        device, non_blocking=True
                    ).contiguous(memory_format=torch.channels_last)
                    y = y.to(device, non_blocking=True)
                    y_pred_raw = _forward_with_autopad(model, x)
                    # TerraMind-only decode in validation, else default
                    num_classes = int(getattr(config, "num_classes", 1))
                    if _is_terramind_model(model):
                        y_prob_full = _as_probs_from_terratorch_logits_first(
                            y_pred_raw, num_classes=num_classes
                        )
                    else:
                        y_prob_full = _as_probs_from_terratorch(
                            y_pred_raw, num_classes=num_classes
                        )
                    y_prob_full = _ensure_nchw(y_prob_full).float()
                    if y_prob_full.shape[1] > 1:
                        cls_idx = int(getattr(config, "metrics_class", 1))
                        cls_idx = max(0, min(cls_idx, y_prob_full.shape[1] - 1))
                        y_prob = y_prob_full[:, cls_idx : cls_idx + 1]
                    else:
                        y_prob = y_prob_full

                    loss = criterion(y, y_prob)
                    val_loss_accum += float(loss.detach().cpu().item())
                    for fn in light_metric_fns:
                        try:
                            v = fn(y, y_prob)
                        except Exception:
                            v = fn(y, (y_prob >= 0.5).float())
                        val_metric_accums[fn.__name__] += float(
                            v.detach().mean().cpu().item()
                            if isinstance(v, torch.Tensor)
                            else float(v)
                        )
                    # keep first batch for visualization
                    if x_vis is None:
                        x_vis, y_vis, y_hat_vis = (
                            x[:8].clone(),
                            y[:8].clone(),
                            y_prob[:8].clone(),
                        )

                denom = max(1, val_count if val_count > 0 else val_steps)
                avg_val_loss = val_loss_accum / denom
                logs["val_loss"] = avg_val_loss
                for fn in light_metric_fns:
                    logs[f"val_{fn.__name__}"] = val_metric_accums[fn.__name__] / denom

                # Heavy metrics on subset of val
                logs = heavy_eval.run(model, epoch, logs)

                # Save best (EMA context if applied)
                improved = best_saver.maybe_save(model, avg_val_loss)
                best_improved = improved
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    logs["best_val_loss"] = best_val_loss

                # Visualisation: epoch-end [ RGB | PRED | GT ]
                if x_vis is not None:
                    try:
                        _log_triptych_and_optional_heatmap(
                            tb=tb,
                            tag_prefix="viz/val",
                            x=x_vis,
                            y_prob=y_hat_vis,
                            y_true=y_vis,
                            step=epoch + 1,
                            rgb_idx=vis_rgb_idx,
                            threshold=float(getattr(config, "eval_threshold", 0.5)),
                            cls_idx=getattr(config, "viz_class", 1),
                            add_heatmap=True,
                        )
                    except Exception:
                        pass

        # If we evaluated/saved EMA weights and there was an improvement,
        # also save RAW weights snapshot
        if best_improved and eval_with_ema and ema is not None:
            try:
                torch.save(model.state_dict(), f"{model_path}.raw.weights.pt")
                print(
                    _col(
                        f"==> Also saved raw weights {model_path}.raw.weights.pt",
                        _C.GREEN,
                    )
                )
            except Exception:
                pass

        # TensorBoard scalars - group train/val into one chart per metric
        if "loss" in logs or "val_loss" in logs:
            pair = {}
            if "loss" in logs:
                pair["train"] = logs["loss"]
            if "val_loss" in logs:
                pair["val"] = logs["val_loss"]
            tb.add_scalars("loss", pair, epoch)

        for fn in light_metric_fns:
            name = fn.__name__
            train_key = name
            val_key = f"val_{name}"
            if train_key in logs or val_key in logs:
                pair = {}
                if train_key in logs:
                    pair["train"] = logs[train_key]
                if val_key in logs:
                    pair["val"] = logs[val_key]
                tb.add_scalars(name, pair, epoch)

        heavy_names = [
            "val_specificity",
            "val_sensitivity",
            "val_f_beta",
            "val_f1_score",
            "val_IoU",
            "val_nominal_surface_distance",
            "val_Hausdorff_distance",
            "val_boundary_intersection_over_union",
            "val_dice_loss",
        ]
        for name in heavy_names:
            if name in logs:
                base = name[4:] if name.startswith("val_") else name
                tb.add_scalars(base, {"val": logs[name]}, epoch)
        tb.flush()

        # Per-epoch metadata JSON
        meta_data = {
            "name": config.model_name,
            "model_path": model_path,
            "patch_size": tuple(config.patch_size),
            "channels_used": getattr(
                config, "channels_used", getattr(config, "channel_list", [])
            ),
            "resample_factor": getattr(config, "resample_factor", None),
            "frames_dir": config.preprocessed_dir,
            "train_ratio": float(f"{1 - config.val_ratio - config.test_ratio:.2f}"),
            "val_ratio": config.val_ratio,
            "test_ratio": config.test_ratio,
            "loss": config.loss_fn,
            "optimizer": config.optimizer_fn,
            "tversky_alpha": getattr(config, "tversky_alphabeta", (None, None))[0],
            "tversky_beta": getattr(config, "tversky_alphabeta", (None, None))[1],
            "batch_size": config.train_batch_size,
            "epoch_steps": config.num_training_steps,
            "val_steps": config.num_validation_images,
            "epochs_trained": f"{epoch + 1}/{config.num_epochs}",
            "total_epochs": config.num_epochs,
            "last_sensitivity": logs.get("val_sensitivity"),
            "last_specificity": logs.get("val_specificity"),
            "last_dice_coef": logs.get("val_dice_coef"),
            "last_dice_loss": logs.get("val_dice_loss"),
            "last_accuracy": logs.get("val_accuracy"),
            "last_f_beta": logs.get("val_f_beta"),
            "last_f1_score": logs.get("val_f1_score"),
            "last_IoU": logs.get("val_IoU"),
            "last_nominal_surface_distance": logs.get(
                "val_nominal_surface_distance"
            ),
            "last_Hausdorff_distance": logs.get("val_Hausdorff_distance"),
            "last_boundary_intersection_over_union": logs.get(
                "val_boundary_intersection_over_union"
            ),
            "start_time": getattr(_fit_model, "_start_time_str", None)
            or datetime.now().strftime("%d.%m.%Y %H:%M:%S"),
            "elapsed_time": None,
        }

        # Update elapsed time
        if not hasattr(_fit_model, "_start_time_dt"):
            _fit_model._start_time_dt = datetime.now()
            _fit_model._start_time_str = meta_data["start_time"]
        elapsed = datetime.now() - _fit_model._start_time_dt
        meta_data["elapsed_time"] = (
            datetime.utcfromtimestamp(0) + elapsed
        ).strftime("%H:%M:%S")

        meta_path = f"{model_path}.metadata.json"
        with open(meta_path, "w") as f:
            json.dump(meta_data, f, indent=4)

        # CSV log
        csv_logger.update(epoch, logs)

        # Optional periodic snapshot
        if model_save_interval and (epoch + 1) % int(model_save_interval) == 0:
            torch.save(model.state_dict(), f"{model_path}.epoch{epoch+1}.weights.pt")

        # ---- Verbose console print of all metrics at end of validation ----
        if verbose and (
            ((epoch + 1) % max(1, epoch_log_every) == 0)
            or (epoch == 0)
            or (epoch + 1 == total_epochs)
        ):
            lr_val = None
            try:
                lr_val = optimizer.param_groups[0].get("lr", None)
            except Exception:
                pass
            took = _fmt_seconds(time.time() - t0)
            head = f"\n Epoch {epoch+1}/{total_epochs} [{took}]"
            if lr_val is not None:
                head += f"  lr={lr_val:.2e}"
            print(head)
            print("  " + _format_logs_for_print(logs))
            if print_heavy:
                heavy_keys = [
                    k
                    for k in logs.keys()
                    if k.startswith("val_")
                    and k
                    not in ("val_loss", "val_accuracy", "val_dice_coef")
                ]
                if len(heavy_keys) > 0:
                    print(
                        "  heavy: "
                        + " | ".join(
                            [f"{k}={logs[k]:.4f}" for k in sorted(heavy_keys)]
                        )
                    )

    # End of training: export full model once
    final_export_path = f"{model_path}.pt"
    try:
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        torch.save(
            {"model_state": model.state_dict(), "config": getattr(config, "__dict__", {})},
            final_export_path,
        )
        print(_col(f"Saved final model to: {final_export_path}", _C.GREEN))
    except Exception as exc:
        print(
            _col(
                f"Warning: final model save failed ({exc}). Attempting to save weights-only.",
                _C.YELLOW,
            )
        )
        torch.save(model.state_dict(), f"{model_path}.final.weights.pt")

    print(_col("Training completed.\n", _C.GREEN))


def _prepare_model_and_logging(model_path: str) -> Tuple[Optional[str], int]:
    """
    Handle continue-from checkpoint and carry over logs.
    Returns (state_dict_path_to_load, starting_epoch).
    """
    starting_epoch = 0
    state_path = None

    if getattr(config, "continue_model_path", None):
        state_path = config.continue_model_path
        # Try to read starting epoch from the JSON meta written each epoch
        try:
            meta_json = f"{state_path}.metadata.json"
            if os.path.exists(meta_json):
                with open(meta_json, "r") as f:
                    custom_meta = json.load(f)
                starting_epoch = int(str(custom_meta["epochs_trained"]).split("/")[0])
        except Exception:
            pass

        # Copy logs forward so TB shows a continuous curve
        old_log_dir = os.path.join(
            config.logs_dir, os.path.basename(config.continue_model_path).split(".")[0]
        )
        new_log_dir = os.path.join(config.logs_dir, os.path.basename(model_path))
        if os.path.exists(old_log_dir) and not os.path.exists(new_log_dir):
            try:
                shutil.copytree(old_log_dir, new_log_dir)
            except Exception:
                pass

    return state_path, starting_epoch


# -----------------------------
# Public training functions 
# -----------------------------
def train_UNet(conf):
    """Create and train a new UNet model with fast execution and extensive logging."""
    global config
    config = conf
    print("Starting training (UNet).")
    start = time.time()

    # Optional seeding
    set_global_seed(getattr(config, "seed", None))

    # Data
    frames = get_all_frames(config)
    train_ds, val_ds, test_ds = create_train_val_datasets(frames)

    # Paths
    stamp = time.strftime("%Y%m%d-%H%M")
    model_path = os.path.join(config.saved_models_dir, f"{stamp}_{config.model_name}")

    # Prepare model / resume
    state_path, starting_epoch = _prepare_model_and_logging(model_path)

    # Build model
    model = _build_model_unet()

    # Optional compile (PyTorch 2.0+)
    if getattr(config, "use_torch_compile", False) and hasattr(torch, "compile"):
        try:
            model = torch.compile(model)  # type: ignore[attr-defined]
        except Exception:
            pass

    # Optimizer / loss
    optimizer = get_optimizer(
        config.optimizer_fn, config.num_epochs, config.num_training_steps, model
    )
    criterion = get_loss(
        config.loss_fn, getattr(config, "tversky_alphabeta", (0.5, 0.5))
    )

    # Resume if path given
    if state_path and os.path.exists(state_path):
        try:
            state = torch.load(state_path, map_location="cpu")
            if isinstance(state, dict) and "model_state" in state:
                model.load_state_dict(state["model_state"])
            elif isinstance(state, dict):
                model.load_state_dict(state)
            print(f"Loaded weights from: {state_path}")
        except Exception as exc:
            print(
                _col(
                    f"Could not load PyTorch weights from {state_path}: {exc}", _C.RED
                )
            )

    # Fit
    _fit_model(
        model,
        train_ds,
        val_ds,
        model_path,
        starting_epoch,
        log_name="_unet",
        optimizer=optimizer,
        criterion=criterion,
    )

    print(
        _col(
            f"Training completed in "
            f"{str(timedelta(seconds=time.time() - start)).split('.')[0]}.\n",
            _C.GREEN,
        )
    )


def train_SwinUNetPP(conf):
    """Create and train a new Swin-UNet model with fast execution and extensive logging."""
    global config
    config = conf
    print("Starting training (SwinUNet).")
    start = time.time()

    # Optional seeding
    set_global_seed(getattr(config, "seed", None))

    # Data
    frames = get_all_frames(config)
    train_ds, val_ds, test_ds = create_train_val_datasets(frames)

    # Paths
    stamp = time.strftime("%Y%m%d-%H%M")
    model_path = os.path.join(config.saved_models_dir, f"{stamp}_{config.model_name}")

    # Prepare model / resume
    state_path, starting_epoch = _prepare_model_and_logging(model_path)

    # Build model
    model = _build_model_swin()

    # Optional compile
    if getattr(config, "use_torch_compile", False) and hasattr(torch, "compile"):
        try:
            model = torch.compile(model)  # type: ignore[attr-defined]
        except Exception:
            pass

    # Optimizer / loss
    optimizer = get_optimizer(config.optimizer_fn, None, None, model)
    criterion = get_loss(
        config.loss_fn, getattr(config, "tversky_alphabeta", (0.5, 0.5))
    )

    # Resume if path given
    if state_path and os.path.exists(state_path):
        try:
            state = torch.load(state_path, map_location="cpu")
            if isinstance(state, dict) and "model_state" in state:
                model.load_state_dict(state["model_state"])
            elif isinstance(state, dict):
                model.load_state_dict(state)
            print(f"Loaded weights from: {state_path}")
        except Exception as exc:
            print(
                _col(
                    f"Could not load PyTorch weights from {state_path}: {exc}", _C.RED
                )
            )

    # Fit
    _fit_model(
        model,
        train_ds,
        val_ds,
        model_path,
        starting_epoch,
        log_name="_swin",
        optimizer=optimizer,
        criterion=criterion,
    )

    print(
        _col(
            f"Training completed in "
            f"{str(timedelta(seconds=time.time() - start)).split('.')[0]}.\n",
            _C.GREEN,
        )
    )


def train_TerraMind(conf):
    """Create and train a TerraMind-based segmentation model with your fast loop."""
    global config
    config = conf
    print("Starting training (TerraMind).")
    start = time.time()

    set_global_seed(getattr(config, "seed", None))

    frames = get_all_frames(config)
    train_ds, val_ds, test_ds = create_train_val_datasets(frames)

    stamp = time.strftime("%Y%m%d-%H%M")
    model_path = os.path.join(config.saved_models_dir, f"{stamp}_{config.model_name}")

    state_path, starting_epoch = _prepare_model_and_logging(model_path)

    model = _build_model_terramind()

    if getattr(config, "use_torch_compile", False) and hasattr(torch, "compile"):
        try:
            model = torch.compile(model)  # type: ignore[attr-defined]
        except Exception:
            pass

    optimizer = get_optimizer(config.optimizer_fn, None, None, model)
    criterion = get_loss(
        config.loss_fn, getattr(config, "tversky_alphabeta", (0.5, 0.5))
    )

    if state_path and os.path.exists(state_path):
        try:
            state = torch.load(state_path, map_location="cpu")
            if isinstance(state, dict) and "model_state" in state:
                model.load_state_dict(state["model_state"])
            elif isinstance(state, dict):
                model.load_state_dict(state)
            print(f"Loaded weights from: {state_path}")
        except Exception as exc:
            print(
                _col(
                    f"Could not load PyTorch weights from {state_path}: {exc}", _C.RED
                )
            )

    _fit_model(
        model,
        train_ds,
        val_ds,
        model_path,
        starting_epoch,
        log_name="_tm",
        optimizer=optimizer,
        criterion=criterion,
    )

    print(
        _col(
            f"Training completed in "
            f"{str(timedelta(seconds=time.time() - start)).split('.')[0]}.\n",
            _C.GREEN,
        )
    )
