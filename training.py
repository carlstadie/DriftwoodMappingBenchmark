# training.py (PyTorch) — BF16 + channels-last, EMA, progress bars, better visuals (raw | mask), robust logits→probs
import os
import json
import time
import glob
import shutil
from datetime import datetime, timedelta
from typing import Iterable, Iterator, Tuple, Optional, Dict, Any

import h5py  # kept for old .h5 checkpoints compatibility
import rasterio
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import IterableDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision.utils as vutils
from contextlib import contextmanager

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

# ===== Your project imports =====
from core.UNet import UNet
from core.Swin_UNetPP import SwinUNet
from core.frame_info import FrameInfo
from core.optimizers import get_optimizer
from core.split_frames import split_dataset, summarize_positive_rates
from core.dataset_generator import DataGenerator as Generator

# Metrics / losses (PyTorch)
from core.losses import (
    accuracy,
    dice_coef,
    dice_loss,
    specificity,
    sensitivity,
    f_beta,
    f1_score,
    IoU,
    nominal_surface_distance,
    Hausdorff_distance,
    boundary_intersection_over_union,
    get_loss,
)

# -----------------------------
# Global config holder (populated by train_* entrypoints)
# -----------------------------
config = None  # set in train_UNet / train_SwinUNetPP


# -----------------------------
# Utilities: seeding, EMA, autopad, visuals, pretty-print
# -----------------------------
def set_global_seed(seed: Optional[int] = None):
    """Set global RNG seeds for reproducibility. If seed is None, do nothing."""
    if seed is None:
        return
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Deterministic kernels (slower but reproducible)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class ModelEMA:
    """Exponential Moving Average of model parameters."""
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = float(decay)
        self.params = [p for p in model.parameters() if p.requires_grad]
        self.shadow = [p.detach().clone() for p in self.params]
        self.backup = None

    @torch.no_grad()
    def update(self):
        for s, p in zip(self.shadow, self.params):
            s.mul_(self.decay).add_(p.detach(), alpha=1 - self.decay)

    @contextmanager
    @torch.no_grad()
    def use_ema_weights(self, model: nn.Module):
        """Temporarily swap model params with EMA weights."""
        self.backup = [p.detach().clone() for p in self.params]
        for p, s in zip(self.params, self.shadow):
            p.copy_(s)
        try:
            yield
        finally:
            for p, b in zip(self.params, self.backup):
                p.copy_(b)
            self.backup = None


def _required_multiple(patch_size: int, window: int, levels: int) -> int:
    """Return required multiple in IMAGE pixels for Swin-like hierarchies."""
    return int(patch_size) * (int(window) * (2 ** int(levels)))


def _autopad(x: torch.Tensor, multiple: int) -> Tuple[torch.Tensor, int, int]:
    """Pad right/bottom so H,W are multiples of 'multiple'."""
    H, W = x.shape[-2], x.shape[-1]
    pad_h = (multiple - H % multiple) % multiple
    pad_w = (multiple - W % multiple) % multiple
    if pad_h or pad_w:
        x = F.pad(x, (0, pad_w, 0, pad_h))
    return x, pad_h, pad_w


def _unpad(y: torch.Tensor, pad_h: int, pad_w: int) -> torch.Tensor:
    if pad_h or pad_w:
        y = y[..., : y.shape[-2] - pad_h, : y.shape[-1] - pad_w]
    return y


def _forward_with_autopad(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Forward pass with optional autopadding for Swin models (guarded by model type)."""
    if isinstance(model, SwinUNet):
        # only Swin needs strict spatial multiples
        window = getattr(config, "swin_window", 4)
        levels = getattr(config, "swin_levels", 3)
        multiple = _required_multiple(getattr(config, "swin_patch_size", 16), window, levels)
        x, ph, pw = _autopad(x, multiple)
        y = model(x)
        return _unpad(y, ph, pw)
    else:
        return model(x)


def _fmt_seconds(s: float) -> str:
    m, s = divmod(int(s), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _ordered_metric_names() -> list:
    # preferred display order
    return [
        "loss", "val_loss",
        "dice_coef", "val_dice_coef",
        "accuracy", "val_accuracy",
        "sensitivity", "val_sensitivity",
        "specificity", "val_specificity",
        "f1_score", "val_f1_score",
        "f_beta", "val_f_beta",
        "IoU", "val_IoU",
        "dice_loss", "val_dice_loss",
        "nominal_surface_distance", "val_nominal_surface_distance",
        "Hausdorff_distance", "val_Hausdorff_distance",
        "boundary_intersection_over_union", "val_boundary_intersection_over_union",
    ]


def _format_logs_for_print(logs: Dict[str, Any]) -> str:
    keys = _ordered_metric_names()
    parts = []
    for k in keys:
        if k in logs and logs[k] is not None:
            try:
                parts.append(f"{k}={float(logs[k]):.4f}")
            except Exception:
                parts.append(f"{k}={logs[k]}")
    # append any extra metrics that weren't in preferred list
    for k, v in logs.items():
        if k not in keys:
            try:
                parts.append(f"{k}={float(v):.4f}")
            except Exception:
                parts.append(f"{k}={v}")
    return " | ".join(parts)


# --------- Console color helpers ---------
class _C:
    RED = "\033[31m"
    YELLOW = "\033[33m"
    GREEN = "\033[32m"
    RESET = "\033[0m"


def _col(s: str, color: str) -> str:
    return f"{color}{s}{_C.RESET}"


# --------- Visualization helpers (robust RGB + side-by-side panels) ---------
def _mask01(m: torch.Tensor) -> torch.Tensor:
    """
    Ensure mask is in [0,1] for visualization. Auto-scales 0/255 masks.
    """
    m = m.detach().float()
    try:
        mx = float(m.max().item())
    except Exception:
        mx = 1.0
    if mx > 1.5:  # looks like 0/255
        m = m / 255.0
    return m.clamp(0.0, 1.0)


def _rgb_from_x(
    x_nchw: torch.Tensor,
    rgb_idx: Tuple[int, int, int] = (0, 1, 2),
    use_quantiles: bool = True,
    q_lo: float = 0.02,
    q_hi: float = 0.98,
    gamma: float = 1.0,
) -> torch.Tensor:
    """
    x_nchw: [N,C,H,W] → returns [N,3,H,W] in [0,1]
    Selects channels by rgb_idx; falls back gracefully if fewer than 3 channels.
    Per-sample robust stretch using percentiles.
    """
    x = x_nchw.detach().to(dtype=torch.float32, device=x_nchw.device)
    N, C, H, W = x.shape
    idx = [i for i in rgb_idx if i < C]
    if len(idx) == 0:
        rgb = x.new_zeros((N, 3, H, W))
    elif len(idx) == 1:
        rgb = x[:, idx[0]:idx[0]+1].repeat(1, 3, 1, 1)
    elif len(idx) == 2:
        rgb = torch.zeros(N, 3, H, W, device=x.device, dtype=x.dtype)
        rgb[:, :2] = x[:, idx]
    else:
        rgb = x[:, idx[:3]]

    if use_quantiles:
        flat = rgb.view(N, 3, -1)
        try:
            lo = torch.quantile(flat, q_lo, dim=-1, method="nearest").view(N, 3, 1, 1)
            hi = torch.quantile(flat, q_hi, dim=-1, method="nearest").view(N, 3, 1, 1)
        except TypeError:
            lo = torch.quantile(flat, q_lo, dim=-1, interpolation="nearest").view(N, 3, 1, 1)
            hi = torch.quantile(flat, q_hi, dim=-1, interpolation="nearest").view(N, 3, 1, 1)
        rgb = (rgb - lo) / (hi - lo + 1e-6)
    else:
        flat = rgb.view(N, 3, -1)
        mn = flat.min(dim=-1, keepdim=True).values.view(N, 3, 1, 1)
        mx = flat.max(dim=-1, keepdim=True).values.view(N, 3, 1, 1)
        rgb = (rgb - mn) / (mx - mn + 1e-6)

    rgb = rgb.clamp_(0, 1)
    if abs(gamma - 1.0) > 1e-6:
        rgb = rgb.pow(1.0 / gamma)
    return rgb


def _mask_to_rgb(mask_n1hw: torch.Tensor, color: Tuple[float, float, float] = (1.0, 0.0, 0.0)) -> torch.Tensor:
    """
    mask_n1hw: [N,1,H,W] with values in [0,1] → [N,3,H,W] (colored mask)
    """
    m = _mask01(mask_n1hw)
    col = torch.tensor(color, device=m.device, dtype=m.dtype).view(1, 3, 1, 1)
    return (m.repeat(1, 3, 1, 1) * col).clamp(0, 1)


def _make_panel_side_by_side(x_rgb: torch.Tensor, right_rgb: torch.Tensor) -> torch.Tensor:
    """
    Concatenate horizontally: [ left=x_rgb | right=right_rgb ], both [N,3,H,W] → [N,3,H,2W]
    """
    return torch.cat([x_rgb, right_rgb], dim=-1).clamp(0, 1)

def _mask_to_two_tone(mask_n1hw: torch.Tensor,
                      threshold: float = 0.5,
                      pos: Tuple[float, float, float] = (1.0, 1.0, 0.0),   # yellow
                      neg: Tuple[float, float, float] = (0.0, 0.0, 1.0)): # blue
    """
    Convert a binary/probability mask to a two-tone RGB image:
      >= threshold -> 'pos' color (class 1), else 'neg' color (class 0).
    mask_n1hw: [N,1,H,W] in [0,1] or 0/255
    """
    m = _mask01(mask_n1hw)
    mb = (m >= threshold).float()
    pos_col = torch.tensor(pos, device=m.device, dtype=m.dtype).view(1, 3, 1, 1)
    neg_col = torch.tensor(neg, device=m.device, dtype=m.dtype).view(1, 3, 1, 1)
    return (mb * pos_col + (1.0 - mb) * neg_col).clamp(0, 1)


def _make_triptych_rgb_pred_gt(x_nchw: torch.Tensor,
                               y_pred_n1hw: torch.Tensor,
                               y_true_n1hw: torch.Tensor,
                               rgb_idx: Tuple[int, int, int] = (0, 1, 2),
                               threshold: float = 0.5,
                               pos: Tuple[float, float, float] = (1.0, 1.0, 0.0),
                               neg: Tuple[float, float, float] = (0.0, 0.0, 1.0)) -> torch.Tensor:
    """
    Build a horizontal triptych [ RGB | PRED | GT ]:
      - RGB from x channels (bands 1–3 -> indices (0,1,2))
      - PRED colored two-tone (1=yellow, 0=blue), thresholded
      - GT colored two-tone (1=yellow, 0=blue)
    Returns [N,3,H,3W].
    """
    rgb = _rgb_from_x(x_nchw, rgb_idx=rgb_idx)
    pred_prob = _ensure_probabilities(y_pred_n1hw)
    pred_rgb = _mask_to_two_tone(pred_prob, threshold=threshold, pos=pos, neg=neg)
    gt_rgb = _mask_to_two_tone(y_true_n1hw, threshold=0.5, pos=pos, neg=neg)
    return torch.cat([rgb, pred_rgb, gt_rgb], dim=-1).clamp(0, 1)


def _ensure_probabilities(y_pred: torch.Tensor) -> torch.Tensor:
    """
    Heuristic: if predictions look like logits (outside [0,1] by margin), apply sigmoid.
    Else assume probabilities and clamp to [0,1].
    """
    with torch.no_grad():
        mn = float(y_pred.min().detach().cpu())
        mx = float(y_pred.max().detach().cpu())
    if (mn < -1e-3) or (mx > 1.0 + 1e-3):
        return torch.sigmoid(y_pred)
    return y_pred.clamp(0.0, 1.0)


# -----------------------------
# Helpers: data, datasets, callbacks, heavy-metric eval
# -----------------------------
def get_all_frames(conf=None):
    """Get all pre-processed frames which will be used for training."""
    global config
    if conf is not None:
        config = conf

    if config is None:
        raise RuntimeError(
            "training.get_all_frames called without a config. "
            "Pass conf=get_config() or set training.config first."
        )

    if config.preprocessed_dir is None:
        config.preprocessed_dir = os.path.join(
            config.preprocessed_base_dir,
            sorted(os.listdir(config.preprocessed_base_dir))[-1],
        )

    image_paths = sorted(
        glob.glob(f"{config.preprocessed_dir}/*.tif"),
        key=lambda f: int(os.path.basename(f)[:-4]),
    )
    print(f"Found {len(image_paths)} input frames in {config.preprocessed_dir}")

    frames = []
    for im_path in tqdm(image_paths, desc="Processing frames"):
        preprocessed = rasterio.open(im_path).read()  # C|H|W

        image_channels = preprocessed[:-1, ...]  # C|H|W
        image_channels = np.transpose(image_channels, (1, 2, 0))  # H|W|C

        annotations = preprocessed[-1, ...]  # H|W
        frames.append(FrameInfo(image_channels, annotations))

    return frames


class TorchGeneratorDataset(IterableDataset):
    """
    Wrap a python generator of (x, y) batches and convert to NCHW tensors.
    """

    def __init__(self, py_generator: Iterable[Tuple[np.ndarray, np.ndarray]]):
        super().__init__()
        self.py_generator = py_generator

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        for x, y in self.py_generator:
            # x: (N, H, W, C) -> (N, C, H, W)
            # y: (N, H, W, 1) -> (N, 1, H, W)
            if isinstance(x, np.ndarray):
                x_t = torch.from_numpy(np.transpose(x, (0, 3, 1, 2))).to(
                    dtype=torch.float32
                )
            else:
                x_t = x.permute(0, 3, 1, 2).contiguous().to(dtype=torch.float32)

            if isinstance(y, np.ndarray):
                if y.ndim == 3:
                    y = y[..., None]
                y_t = torch.from_numpy(np.transpose(y, (0, 3, 1, 2))).to(
                    dtype=torch.float32
                )
            else:
                if y.ndim == 3:
                    y = y.unsqueeze(-1)
                y_t = y.permute(0, 3, 1, 2).contiguous().to(dtype=torch.float32)

            # --- ensure masks are in [0,1] (auto-scale 0/255 to 0/1) ---
            maxv = float(y_t.max().item()) if y_t.numel() > 0 else 1.0
            if maxv > 1.5:
                y_t = y_t / 255.0

            yield x_t, y_t


def create_train_val_datasets(frames):
    """Create training / validation / test sets and build generators / DataLoaders."""
    frames_json = os.path.join(config.preprocessed_dir, "aa_frames_list.json")
    training_frames, validation_frames, test_frames = split_dataset(
        frames, frames_json, config.test_ratio, config.val_ratio
    )

    stats = summarize_positive_rates(frames, {
        "train": training_frames,
        "val": validation_frames,
        "test": test_frames
    })
    print("\n[positive-rate % by frame]  mean | median | std | min..max  (n)")
    for k in ("train", "val", "test"):
        s = stats[k]
        print(f"  {k:>5}: {s['mean']:.3f} | {s['median']:.3f} | {s['std']:.3f} | "
              f"{s['min']:.3f}..{s['max']:.3f}  (n={s['n']})")

    # Channels
    input_channels = list(range(len(config.channel_list)))
    label_channel = len(config.channel_list)  # label directly after inputs

    # Patch size (allow tuner to override)
    patch_h, patch_w = config.patch_size
    if (
        hasattr(config, "tune_patch_h")
        and hasattr(config, "tune_patch_w")
        and config.tune_patch_h
        and config.tune_patch_w
    ):
        patch_h, patch_w = int(config.tune_patch_h), int(config.tune_patch_w)

    # H*W*(inputs+1 label)
    patch_size = [patch_h, patch_w, len(config.channel_list) + 1]

    # Generator knobs
    aug_strength = getattr(config, "augmenter_strength", 1.0)
    min_pos_frac = float(getattr(config, "min_pos_frac", 0.0))
    pos_ratio = getattr(config, "pos_ratio", None)  # None => TF-like random sampling
    stride = getattr(config, "patch_stride", None)

    # Build python generators (Albumentations path in your DataGenerator)
    train_gen = Generator(
        input_channels,
        patch_size,
        training_frames,
        frames,
        label_channel,
        augmenter="alb",
        augmenter_strength=aug_strength,
        min_pos_frac=min_pos_frac,
        pos_ratio=pos_ratio,
        stride=stride,
        weighting="area",  # TF parity: area-weighted frame sampling
    ).random_generator(config.train_batch_size)

    val_gen = Generator(
        input_channels,
        patch_size,
        validation_frames,
        frames,
        label_channel,
        augmenter=None,
        augmenter_strength=1.0,
        min_pos_frac=0.0,
        pos_ratio=None,  # TF parity for val
        stride=stride,
        weighting="area",
    ).random_generator(config.train_batch_size)

    test_gen = Generator(
        input_channels,
        patch_size,
        test_frames,
        frames,
        label_channel,
        augmenter=None,
        augmenter_strength=1.0,
        min_pos_frac=0.0,
        pos_ratio=None,
        stride=stride,
        weighting="area",
    ).random_generator(config.train_batch_size)

    # Wrap with DataLoader-like iterables
    workers = int(getattr(config, "fit_workers", 8))
    train_ds = TorchGeneratorDataset(train_gen)
    val_ds = TorchGeneratorDataset(val_gen)
    test_ds = TorchGeneratorDataset(test_gen)

    train_dl = DataLoader(
        train_ds,
        batch_size=None,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        prefetch_factor=2 if workers > 0 else None,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=None,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        prefetch_factor=2 if workers > 0 else None,
    )
    test_dl = DataLoader(
        test_ds,
        batch_size=None,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        prefetch_factor=2 if workers > 0 else None,
    )
    return train_dl, val_dl, test_dl


class BestModelSaver:
    """Saves best model weights based on a monitored value (min mode)."""

    def __init__(self, model_path: str):
        self.model_path = model_path
        self.best_val = float("inf")

    def maybe_save(self, model: nn.Module, current_val: float):
        prev = self.best_val
        if current_val < self.best_val:
            self.best_val = current_val
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            # Save (typically EMA) weights
            torch.save(model.state_dict(), f"{self.model_path}.weights.pt")
            print(_col(
                f"New best! val_loss improved {prev:.6f} --> {current_val:.6f}. "
                f"Saved: {self.model_path}.weights.pt",
                _C.GREEN
            ))
            return True  # indicate improvement
        return False


class MetricsCSVLogger:
    """Append all logs each epoch to a CSV in the run's log dir."""

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
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    def run(
        self, model: nn.Module, epoch: int, logs: Dict[str, Any]
    ) -> Dict[str, Any]:
        model.eval()
        accum: Dict[str, list] = {k: [] for k in self.metric_fns.keys()}

        with torch.no_grad():
            it = iter(self.val_iterable)
            for _ in range(self.steps):
                try:
                    x, y_true = next(it)
                except StopIteration:
                    break

                x = x.to(self.device, non_blocking=True).contiguous(memory_format=torch.channels_last)
                y_true = y_true.to(self.device, non_blocking=True)

                y_pred_raw = _forward_with_autopad(model, x)
                y_prob = _ensure_probabilities(y_pred_raw).float()
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
                # TensorBoard (PyTorch): no as_default(); just add_scalar
                self.tb_writer.add_scalar(name, mean_val, epoch)
                logs[f"val_{name}"] = mean_val
        self.tb_writer.flush()
        return logs


def _create_logging(model_path: str, log_suffix: str = "") -> Tuple[BestModelSaver, str]:
    """Define logging paths and return a best-saver + log_dir."""
    log_dir = os.path.join(config.logs_dir, os.path.basename(model_path) + log_suffix)
    os.makedirs(log_dir, exist_ok=True)
    best_saver = BestModelSaver(model_path)
    return best_saver, log_dir


def _print_run_banner(model_key: str, log_dir: str):
    """Pretty console banner like tuning, including the LOSS."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ema_on = bool(getattr(config, "use_ema", True))
    aug = float(getattr(config, "augmenter_strength", 1.0))
    minpos = float(getattr(config, "min_pos_frac", 0.0))
    posr = getattr(config, "pos_ratio", None)
    workers = int(getattr(config, "fit_workers", 8))
    stride = getattr(config, "patch_stride", None)
    loss_name = getattr(config, "loss_fn", "tversky")
    ab = getattr(config, "tversky_alphabeta", (0.5, 0.5))
    optimizer_name = getattr(config, "optimizer_fn", "adam")

    def _dtype_str(dt):
        if dt is torch.bfloat16: return "bf16"
        if dt is torch.float16:  return "fp16"
        if dt is torch.float32:  return "fp32"
        return str(dt)

    print("\n" + "=" * 90)
    print(f"[{model_key.upper()}][TRAIN]  run={getattr(config, 'run_name', 'run')}  model_name={config.model_name}")
    print(f"[{model_key.upper()}][TRAIN]  device={device}, amp_dtype={_dtype_str(_AMP_DTYPE)}, channels_last=True")
    print(f"[{model_key.upper()}][TRAIN]  ema={ema_on} (decay={getattr(config, 'ema_decay', 0.999)})")
    print(f"[{model_key.upper()}][TRAIN]  epochs={config.num_epochs}, steps/epoch={config.num_training_steps}, "
          f"val_steps={config.num_validation_images}, batch={config.train_batch_size}, workers={workers}")
    print(f"[{model_key.upper()}][TRAIN]  patch={config.patch_size}, stride={stride}, "
          f"aug={aug}, min_pos_frac={minpos}, pos_ratio={posr}")
    if hasattr(config, "swin_patch_size"):
        print(f"[{model_key.upper()}][TRAIN]  swin_patch={getattr(config,'swin_patch_size',16)}, "
              f"window={getattr(config,'swin_window',4)}")
    # Explicit loss + optimizer print
    if str(loss_name).lower().startswith("tversky"):
        print(f"[{model_key.upper()}][TRAIN]  loss={loss_name} (alpha={ab[0]:.2f}, beta={ab[1]:.2f}), optimizer={optimizer_name}")
    else:
        print(f"[{model_key.upper()}][TRAIN]  loss={loss_name}, optimizer={optimizer_name}")
    print(f"[{model_key.upper()}][TRAIN]  logs_dir={log_dir}")
    print("=" * 90 + "\n")


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

    
    # === Overfit-one-batch mode: capture first batch and repeat forever ===
    if getattr(config, "overfit_one_batch", False):

        print('==== Overfitting on one Batch ====')
        # get one CPU batch (as provided by your iterable)
        first_it = iter(train_iterable)
        first_x, first_y = next(first_it)

        # freeze data randomness: use exact same tensors every step
        def _infinite_one_batch():
            while True:
                # clone to avoid any accidental in-place ops piling up
                yield first_x.clone(), first_y.clone()

        # validation also on the same batch (repeat N times)
        def _repeat_val(n):
            for _ in range(int(n)):
                yield first_x.clone(), first_y.clone()

        train_iterable = _infinite_one_batch()
        val_iterable = _repeat_val(getattr(config, "num_validation_images", 10))

    # EMA (optional)
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
    val_steps = int(config.num_validation_images)
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
    key = "unet" if log_name == "_unet" else "swin"
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
            train_range = tqdm(train_range, desc=f"Epoch {epoch+1}/{total_epochs} [train]", leave=False)

        train_it = iter(train_iterable)
        for step in train_range:
            try:
                x, y = next(train_it)
            except StopIteration:
                train_it = iter(train_iterable)
                x, y = next(train_it)

            # Move + channels-last
            x = x.to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)
            y = y.to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available(), dtype=_AMP_DTYPE):
                y_pred_raw = _forward_with_autopad(model, x)
                y_prob = _ensure_probabilities(y_pred_raw)
                loss = criterion(y, y_prob)

            scaler.scale(loss / grad_accum).backward()

            # Gradient step
            if (step + 1) % grad_accum == 0:
                if clip_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm)
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
                    val.detach().mean().cpu().item() if isinstance(val, torch.Tensor) else float(val)
                )

            global_step += 1

            # Step-wise visuals: [ RGB | PRED | GT ]
            if log_visuals_every > 0 and ((global_step % log_visuals_every) == 0):
                try:
                    n = min(8, x.size(0))
                    panel = _make_triptych_rgb_pred_gt(
                        x[:n],
                        y_prob[:n],
                        y[:n],
                        rgb_idx=vis_rgb_idx,
                        threshold=float(getattr(config, "eval_threshold", 0.5)),
                        pos=tuple(getattr(config, "viz_pos_color", (1.0, 1.0, 0.0))),  # 1 -> yellow
                        neg=tuple(getattr(config, "viz_neg_color", (0.0, 0.0, 1.0))),  # 0 -> blue
                    )
                    grid = vutils.make_grid(panel, nrow=min(4, panel.size(0)), padding=0)
                    tb.add_image("viz/train/rgb_pred_gt", grid, global_step)
                except Exception:
                    pass

            # Live postfix for progress
            if show_progress:
                avg_tr = train_loss_accum / (step + 1)
                postfix = {
                    "loss": f"{avg_tr:.4f}",
                    "dice": f"{(metric_accums['dice_coef']/(step+1)):.4f}",
                    "acc":  f"{(metric_accums['accuracy']/(step+1)):.4f}",
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
            val_range = range(val_steps)
            if show_progress:
                val_range = tqdm(val_range, desc=f"Epoch {epoch+1}/{total_epochs} [val]", leave=False)

            with torch.no_grad():
                val_it = iter(val_iterable)
                x_vis = None; y_vis = None; y_hat_vis = None  # for epoch-end viz
                for _ in val_range:
                    try:
                        x, y = next(val_it)
                    except StopIteration:
                        break
                    x = x.to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)
                    y = y.to(device, non_blocking=True)
                    y_pred_raw = _forward_with_autopad(model, x)
                    y_prob = _ensure_probabilities(y_pred_raw).float()
                    loss = criterion(y, y_prob)
                    val_loss_accum += float(loss.detach().cpu().item())
                    for fn in light_metric_fns:
                        try:
                            v = fn(y, y_prob)
                        except Exception:
                            v = fn(y, (y_prob >= 0.5).float())
                        val_metric_accums[fn.__name__] += float(
                            v.detach().mean().cpu().item() if isinstance(v, torch.Tensor) else float(v)
                        )
                    # keep one batch for visual
                    if x_vis is None:
                        x_vis, y_vis, y_hat_vis = x[:8].clone(), y[:8].clone(), y_prob[:8].clone()

                avg_val_loss = val_loss_accum / max(1, val_steps)
                logs["val_loss"] = avg_val_loss
                for fn in light_metric_fns:
                    logs[f"val_{fn.__name__}"] = val_metric_accums[fn.__name__] / max(1, val_steps)

                # Heavy metrics on subset of val
                logs = heavy_eval.run(model, epoch, logs)

                # Save best (EMA context if applied)
                improved = best_saver.maybe_save(model, avg_val_loss)
                best_improved = improved
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    logs["best_val_loss"] = best_val_loss

                # Visualization: epoch-end [ RGB | PRED | GT ]
                if x_vis is not None:
                    try:
                        panel = _make_triptych_rgb_pred_gt(
                            x_vis,
                            y_hat_vis,
                            y_vis,
                            rgb_idx=vis_rgb_idx,
                            threshold=float(getattr(config, "eval_threshold", 0.5)),
                            pos=tuple(getattr(config, "viz_pos_color", (1.0, 1.0, 0.0))),
                            neg=tuple(getattr(config, "viz_neg_color", (0.0, 0.0, 0.5))),
                        )
                        grid = vutils.make_grid(panel, nrow=min(4, panel.size(0)), padding=0)
                        tb.add_image("viz/val/rgb_pred_gt", grid, epoch + 1)
                    except Exception:
                        pass


        # If we evaluated/saved EMA weights and there was an improvement, also save RAW weights snapshot
        if best_improved and eval_with_ema and ema is not None:
            try:
                torch.save(model.state_dict(), f"{model_path}.raw.weights.pt")
                print(_col(f"(Also saved raw weights snapshot) {model_path}.raw.weights.pt", _C.GREEN))
            except Exception as _:
                pass

        # TensorBoard scalars — group train/val into one chart per metric using add_scalars
        # Loss (train + val)
        if "loss" in logs or "val_loss" in logs:
            pair = {}
            if "loss" in logs:
                pair["train"] = logs["loss"]
            if "val_loss" in logs:
                pair["val"] = logs["val_loss"]
            tb.add_scalars("loss", pair, epoch)

        # Light metrics (e.g., dice_coef, accuracy) — grouped train/val per metric
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

        # Heavy metrics are primarily validation-only; log them grouped under the base metric name as "val"
        heavy_names = [
            "val_specificity", "val_sensitivity", "val_f_beta", "val_f1_score", "val_IoU",
            "val_nominal_surface_distance", "val_Hausdorff_distance", "val_boundary_intersection_over_union",
            "val_dice_loss"
        ]
        for name in heavy_names:
            if name in logs:
                base = name[4:] if name.startswith("val_") else name
                tb.add_scalars(base, {"val": logs[name]}, epoch)
            tb.flush()

        # Per-epoch metadata (mirrors your JSON)
        meta_data = {
            "name": config.model_name,
            "model_path": model_path,
            "patch_size": tuple(config.patch_size),
            "channels_used": getattr(config, "channels_used", getattr(config, "channel_list", [])),
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
            "last_nominal_surface_distance": logs.get("val_nominal_surface_distance"),
            "last_Hausdorff_distance": logs.get("val_Hausdorff_distance"),
            "last_boundary_intersection_over_union": logs.get("val_boundary_intersection_over_union"),
            "start_time": getattr(_fit_model, "_start_time_str", None) or datetime.now().strftime("%d.%m.%Y %H:%M:%S"),
            "elapsed_time": None,
        }

        # Update elapsed time
        if not hasattr(_fit_model, "_start_time_dt"):
            _fit_model._start_time_dt = datetime.now()
            _fit_model._start_time_str = meta_data["start_time"]
        elapsed = (datetime.now() - _fit_model._start_time_dt)
        meta_data["elapsed_time"] = (datetime.utcfromtimestamp(0) + elapsed).strftime("%H:%M:%S")

        meta_path = f"{model_path}.metadata.json"
        with open(meta_path, "w") as f:
            json.dump(meta_data, f, indent=4)

        # CSV log
        csv_logger.update(epoch, logs)

        # Optional periodic snapshot
        if model_save_interval and (epoch + 1) % int(model_save_interval) == 0:
            torch.save(model.state_dict(), f"{model_path}.epoch{epoch+1}.weights.pt")

        # ---- Verbose console print (all metrics) ----
        if verbose and (((epoch + 1) % max(1, epoch_log_every) == 0) or (epoch == 0) or (epoch + 1 == total_epochs)):
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
                heavy_keys = [k for k in logs.keys() if k.startswith("val_") and k not in ("val_loss", "val_accuracy", "val_dice_coef")]
                if len(heavy_keys) > 0:
                    print("  heavy: " + " | ".join([f"{k}={logs[k]:.4f}" for k in sorted(heavy_keys)]))

    # End of training: export full model (state dict) once
    final_export_path = f"{model_path}.pt"
    try:
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        torch.save({"model_state": model.state_dict(), "config": getattr(config, "__dict__", {})}, final_export_path)
        print(_col(f"Saved final model to: {final_export_path}", _C.GREEN))
    except Exception as exc:
        print(_col(f"Warning: final model save failed ({exc}). Attempting to save weights-only.", _C.YELLOW))
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
            config.logs_dir,
            os.path.basename(config.continue_model_path).split(".")[0],
        )
        new_log_dir = os.path.join(config.logs_dir, os.path.basename(model_path))
        if os.path.exists(old_log_dir) and not os.path.exists(new_log_dir):
            try:
                shutil.copytree(old_log_dir, new_log_dir)
            except Exception:
                pass

    return state_path, starting_epoch


# -----------------------------
# Public training functions (entrypoints preserved)
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
    frames = get_all_frames()
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
    optimizer = get_optimizer(config.optimizer_fn, config.num_epochs, config.num_training_steps, model)
    criterion = get_loss(config.loss_fn, getattr(config, "tversky_alphabeta", (0.5, 0.5)))

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
            print(_col(f"Could not load PyTorch weights from {state_path}: {exc}", _C.RED))

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

    print(_col(f"Training completed in {str(timedelta(seconds=time.time() - start)).split('.')[0]}.\n", _C.GREEN))


def train_SwinUNetPP(conf):
    """Create and train a new Swin-UNet model with fast execution and extensive logging."""
    global config
    config = conf
    print("Starting training (SwinUNet).")
    start = time.time()

    # Optional seeding
    set_global_seed(getattr(config, "seed", None))

    # Data
    frames = get_all_frames()
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
    criterion = get_loss(config.loss_fn, getattr(config, "tversky_alphabeta", (0.5, 0.5)))

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
            print(_col(f"Could not load PyTorch weights from {state_path}: {exc}", _C.RED))

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

    print(_col(f"Training completed in {str(timedelta(seconds=time.time() - start)).split('.')[0]}.\n", _C.GREEN))
