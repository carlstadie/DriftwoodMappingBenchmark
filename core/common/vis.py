# core/common/vis.py
from __future__ import annotations

from typing import Tuple, Optional, Any

import torch
import torchvision.utils as vutils
from torch.utils.tensorboard import SummaryWriter

# reuse probability helper from model_utils to avoid duplication
from .model_utils import _ensure_probabilities

# Optional runtime config (used only for default viz colors if present)
config: Any = None


def _to_cpu_image(t: torch.Tensor) -> torch.Tensor:
    """Detach → float32 → CPU so SummaryWriter.add_image always works."""
    return t.detach().to(device="cpu", dtype=torch.float32)


def _pick_vis_channel(prob: torch.Tensor, cls_idx: Optional[int] = None) -> torch.Tensor:
    """
    Ensure probs are [N,1,H,W] for visualization.
    - Binary: already [N,1,H,W] → return as is
    - Multi-class: [N,C,H,W] → pick cls_idx (default=1) or argmax mask if None
    """
    if prob.dim() == 4 and prob.size(1) > 1:
        if cls_idx is None:
            arg = prob.argmax(dim=1, keepdim=True)
            cls = 1
            return (arg == cls).float()
        return prob[:, cls_idx:cls_idx+1]
    return prob  # already single-channel


def _overlay_heatmap_on_rgb(rgb: torch.Tensor, prob: torch.Tensor, alpha: float = 0.4) -> torch.Tensor:
    """
    Overlay probability heatmap on RGB image.
    rgb: [N,3,H,W] in [0,1]
    prob: [N,1,H,W] in [0,1]
    Returns: [N,3,H,W] blended image
    """
    p = prob.clamp(0, 1)
    heat_r = p
    heat_g = 1.0 - torch.abs(p - 0.5) * 2.0
    heat_b = 1.0 - p
    heatmap_rgb = torch.cat([heat_r, heat_g, heat_b], dim=1).clamp(0, 1)
    overlay = alpha * heatmap_rgb + (1.0 - alpha) * rgb
    return overlay.clamp(0, 1)


def _mask01(m: torch.Tensor) -> torch.Tensor:
    """Ensure mask is in [0,1] for visualization. Auto-scales 0/255 masks."""
    m = m.detach().float()
    try:
        mx = float(m.max().item())
    except Exception:
        mx = 1.0
    if mx > 1.5:
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
    """mask_n1hw: [N,1,H,W] in [0,1] → [N,3,H,W] (colored mask)"""
    m = _mask01(mask_n1hw)
    col = torch.tensor(color, device=m.device, dtype=m.dtype).view(1, 3, 1, 1)
    return (m.repeat(1, 3, 1, 1) * col).clamp(0, 1)


def _make_panel_side_by_side(x_rgb: torch.Tensor, right_rgb: torch.Tensor) -> torch.Tensor:
    """Concatenate horizontally: [ left=x_rgb | right=right_rgb ]"""
    return torch.cat([x_rgb, right_rgb], dim=-1).clamp(0, 1)


def _mask_to_two_tone(
    mask_n1hw: torch.Tensor,
    threshold: float = 0.5,
    pos: Tuple[float, float, float] = (1.0, 1.0, 0.0),
    neg: Tuple[float, float, float] = (0.0, 0.0, 1.0)
) -> torch.Tensor:
    """
    Convert a binary/probability mask to a two-tone RGB image:
      >= threshold -> 'pos' color (class 1), else 'neg' color (class 0).
    """
    m = _mask01(mask_n1hw)
    mb = (m >= threshold).float()
    pos_col = torch.tensor(pos, device=m.device, dtype=m.dtype).view(1, 3, 1, 1)
    neg_col = torch.tensor(neg, device=m.device, dtype=m.dtype).view(1, 3, 1, 1)
    return (mb * pos_col + (1.0 - mb) * neg_col).clamp(0, 1)


def _make_triptych_rgb_pred_gt(
    x_nchw: torch.Tensor,
    y_pred_n1hw: torch.Tensor,
    y_true_n1hw: torch.Tensor,
    rgb_idx: Tuple[int, int, int] = (0, 1, 2),
    threshold: float = 0.5,
    pos: Tuple[float, float, float] = (1.0, 1.0, 0.0),
    neg: Tuple[float, float, float] = (0.0, 0.0, 1.0),
    overlay_heatmap: bool = False
) -> torch.Tensor:
    """
    Build a horizontal triptych [ RGB (opt heatmap) | PRED | GT ].
    Returns [N,3,H,3W].
    """
    rgb = _rgb_from_x(x_nchw, rgb_idx=rgb_idx)
    pred_prob = _ensure_probabilities(y_pred_n1hw)
    if overlay_heatmap:
        rgb = _overlay_heatmap_on_rgb(rgb, pred_prob, alpha=0.4)
    pred_rgb = _mask_to_two_tone(pred_prob, threshold=threshold, pos=pos, neg=neg)
    gt_rgb = _mask_to_two_tone(y_true_n1hw, threshold=0.5, pos=pos, neg=neg)
    return torch.cat([rgb, pred_rgb, gt_rgb], dim=-1).clamp(0, 1)


def _log_triptych_and_optional_heatmap(
    tb: SummaryWriter,
    tag_prefix: str,
    x: torch.Tensor,
    y_prob: torch.Tensor,
    y_true: torch.Tensor,
    step: int,
    rgb_idx: Tuple[int, int, int],
    threshold: float,
    cls_idx: Optional[int],
    add_heatmap: bool = True,
):
    """
    Logs: triptych [ RGB with heatmap overlay | PRED(thresholded two-tone) | GT(two-tone) ]
    Works on CPU or CUDA tensors and binary or multi-class y_prob.
    """
    n = min(8, x.size(0))
    prob_vis = _pick_vis_channel(y_prob[:n], cls_idx=cls_idx)

    # default viz colors (optionally from a runtime config if present)
    pos_col = tuple(getattr(config, "viz_pos_color", (1.0, 1.0, 0.0))) if (isinstance(config, object)) else (1.0, 1.0, 0.0)
    neg_col = tuple(getattr(config, "viz_neg_color", (0.0, 0.0, 1.0))) if (isinstance(config, object)) else (0.0, 0.0, 1.0)

    panel = _make_triptych_rgb_pred_gt(
        x[:n], prob_vis, y_true[:n],
        rgb_idx=rgb_idx,
        threshold=threshold,
        pos=pos_col,
        neg=neg_col,
        overlay_heatmap=add_heatmap,
    )
    grid = vutils.make_grid(panel, nrow=min(4, panel.size(0)), padding=0)
    tb.add_image(f"{tag_prefix}/rgb_pred_gt", _to_cpu_image(grid), step)
    tb.flush()
