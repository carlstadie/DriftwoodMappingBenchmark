# core/losses.py
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F

_EPS = 1e-6


def _ensure_nchw_1(yt: torch.Tensor, yp: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Make y_true, y_pred float tensors of shape (B, 1, H, W).
    Also resizes y_pred spatially to match y_true if needed and guards against empty tensors.
    """
    # y_true to (B,1,H,W)
    if yt.ndim == 3:
        yt = yt.unsqueeze(1)
    if yt.ndim != 4:
        raise ValueError(f"y_true must be 3D/4D, got {yt.ndim}D")
    if yt.shape[1] != 1:
        # collapse to single channel if needed (assume class 1 foreground)
        if yt.shape[1] > 1:
            yt = yt[:, 1:2]
        else:
            yt = yt[:, :1]
    yt = yt.float().clamp(0.0, 1.0)

    # y_pred to (B,1,H,W)
    if yp.ndim == 3:
        yp = yp.unsqueeze(1)
    if yp.ndim != 4:
        raise ValueError(f"y_pred must be 3D/4D, got {yp.ndim}D")

    # If yp looks like NHWC, permute
    if yp.shape[1] not in (1, yt.shape[1]) and yp.shape[-1] in (1, yt.shape[1]):
        yp = yp.permute(0, 3, 1, 2).contiguous()

    # If multi-class, use channel 1 (or argmax fallback)
    if yp.shape[1] > 1:
        cls_idx = 1 if yp.shape[1] > 1 else 0
        yp = yp[:, cls_idx:cls_idx + 1]

    # If zero channels (shouldn't happen, but be defensive)
    if yp.shape[1] == 0:
        yp = torch.zeros((yt.shape[0], 1, max(1, yp.shape[-2]), max(1, yp.shape[-1])),
                         dtype=torch.float32, device=yt.device)

    # Resize yp spatially if needed
    if yp.shape[-2:] != yt.shape[-2:]:
        if yp.numel() == 0 or yp.shape[-2] == 0 or yp.shape[-1] == 0:
            yp = torch.zeros((yt.shape[0], 1, yt.shape[-2], yt.shape[-1]),
                             dtype=torch.float32, device=yt.device)
        else:
            yp = F.interpolate(yp.float(), size=yt.shape[-2:], mode="bilinear", align_corners=False)

    # Clamp probs safely
    yp = yp.float().clamp(_EPS, 1.0 - _EPS)
    return yt, yp


# ----------------- Basic metrics -----------------
def dice_coef(y_true: torch.Tensor, y_pred: torch.Tensor, eps: float = _EPS) -> torch.Tensor:
    yt, yp = _ensure_nchw_1(y_true, y_pred)
    inter = torch.sum(yt * yp, dim=(1, 2, 3))
    den = torch.sum(yt, dim=(1, 2, 3)) + torch.sum(yp, dim=(1, 2, 3))
    dice = (2.0 * inter + eps) / (den + eps)
    return dice.mean()


def dice_loss(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    return 1.0 - dice_coef(y_true, y_pred)


def IoU(y_true: torch.Tensor, y_pred: torch.Tensor, eps: float = _EPS) -> torch.Tensor:
    yt, yp = _ensure_nchw_1(y_true, y_pred)
    inter = torch.sum(yt * yp, dim=(1, 2, 3))
    union = torch.sum(yt + yp, dim=(1, 2, 3)) - inter
    iou = (inter + eps) / (union + eps)
    return iou.mean()


def accuracy(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    yt, yp = _ensure_nchw_1(y_true, y_pred)
    yb = (yp >= 0.5).float()
    correct = torch.sum((yb == (yt >= 0.5)).float(), dim=(1, 2, 3))
    total = yt.shape[2] * yt.shape[3]
    return (correct / total).mean()


def sensitivity(y_true: torch.Tensor, y_pred: torch.Tensor, eps: float = _EPS) -> torch.Tensor:
    yt, yp = _ensure_nchw_1(y_true, y_pred)
    yb = (yp >= 0.5).float()
    tp = torch.sum(yb * yt, dim=(1, 2, 3))
    fn = torch.sum((1 - yb) * yt, dim=(1, 2, 3))
    return ((tp + eps) / (tp + fn + eps)).mean()


def specificity(y_true: torch.Tensor, y_pred: torch.Tensor, eps: float = _EPS) -> torch.Tensor:
    yt, yp = _ensure_nchw_1(y_true, y_pred)
    yb = (yp >= 0.5).float()
    tn = torch.sum((1 - yb) * (1 - yt), dim=(1, 2, 3))
    fp = torch.sum(yb * (1 - yt), dim=(1, 2, 3))
    return ((tn + eps) / (tn + fp + eps)).mean()


def f1_score(y_true: torch.Tensor, y_pred: torch.Tensor, eps: float = _EPS) -> torch.Tensor:
    d = dice_coef(y_true, y_pred, eps=eps)
    return d


def f_beta(y_true: torch.Tensor, y_pred: torch.Tensor, beta: float = 2.0, eps: float = _EPS) -> torch.Tensor:
    yt, yp = _ensure_nchw_1(y_true, y_pred)
    yb = (yp >= 0.5).float()
    tp = torch.sum(yb * yt, dim=(1, 2, 3))
    fp = torch.sum(yb * (1 - yt), dim=(1, 2, 3))
    fn = torch.sum((1 - yb) * yt, dim=(1, 2, 3))
    prec = (tp + eps) / (tp + fp + eps)
    rec = (tp + eps) / (tp + fn + eps)
    b2 = beta * beta
    fbeta = (1 + b2) * (prec * rec) / (b2 * prec + rec + eps)
    return fbeta.mean()


# ----------------- Tversky & helpers -----------------
def tversky(y_true: torch.Tensor, y_pred: torch.Tensor, alpha: float = 0.5, beta: float = 0.5, eps: float = _EPS) -> torch.Tensor:
    yt, yp = _ensure_nchw_1(y_true, y_pred)
    p1 = yp
    g1 = yt
    p0 = 1.0 - p1
    g0 = 1.0 - g1
    tp = torch.sum(p1 * g1, dim=(1, 2, 3))
    fp = torch.sum(p1 * g0, dim=(1, 2, 3))
    fn = torch.sum(p0 * g1, dim=(1, 2, 3))
    return ((tp + eps) / (tp + alpha * fp + beta * fn + eps)).mean()


def _tversky_loss(alpha: float, beta: float):
    def _fn(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        return 1.0 - tversky(y_true, y_pred, alpha=alpha, beta=beta)
    return _fn


# ---- Heavy metrics placeholders (keep names; implement simple surrogates) ----
def nominal_surface_distance(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    # Lightweight surrogate: 1 - Dice as a proxy; avoids heavy computation during frequent evals
    return dice_loss(y_true, y_pred)


def Hausdorff_distance(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    # Lightweight surrogate: 1 - IoU proxy
    return 1.0 - IoU(y_true, y_pred)


def boundary_intersection_over_union(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    # Simple proxy using IoU (without explicit boundary extraction to keep runtime low)
    return IoU(y_true, y_pred)


# ----------------- Factory -----------------
def get_loss(name: str, alphabeta=(0.5, 0.5)):
    n = (name or "").lower()
    if n.startswith("tversky"):
        a, b = alphabeta if isinstance(alphabeta, (tuple, list)) and len(alphabeta) == 2 else (0.5, 0.5)
        return _tversky_loss(float(a), float(b))
    if n in {"dice", "dice_loss"}:
        return dice_loss
    if n in {"bce"}:
        return lambda yt, yp: F.binary_cross_entropy(torch.clamp(yp.float(), _EPS, 1.0 - _EPS),
                                                     torch.clamp(yt.float(), 0.0, 1.0))
    # default
    a, b = alphabeta if isinstance(alphabeta, (tuple, list)) and len(alphabeta) == 2 else (0.5, 0.5)
    return _tversky_loss(float(a), float(b))
