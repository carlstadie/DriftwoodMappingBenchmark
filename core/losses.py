# core/losses.py
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F

_EPS = 1e-6

# --- Optional but strongly recommended for correct, stable surface distances ---
try:
    import numpy as np
    from scipy.ndimage import binary_erosion, distance_transform_edt

    _HAS_SCIPY = True
except Exception:  # pragma: no cover
    _HAS_SCIPY = False
    np = None  # type: ignore[assignment]
    binary_erosion = None  # type: ignore[assignment]
    distance_transform_edt = None  # type: ignore[assignment]


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
        yp = torch.zeros(
            (yt.shape[0], 1, max(1, yp.shape[-2]), max(1, yp.shape[-1])),
            dtype=torch.float32,
            device=yt.device,
        )

    # Resize yp spatially if needed
    if yp.shape[-2:] != yt.shape[-2:]:
        if yp.numel() == 0 or yp.shape[-2] == 0 or yp.shape[-1] == 0:
            yp = torch.zeros(
                (yt.shape[0], 1, yt.shape[-2], yt.shape[-1]),
                dtype=torch.float32,
                device=yt.device,
            )
        else:
            yp = F.interpolate(
                yp.float(), size=yt.shape[-2:], mode="bilinear", align_corners=False
            )

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
    # Note: for binary masks, Dice == F1. Here dice_coef is "soft Dice" if y_pred is probabilistic.
    return dice_coef(y_true, y_pred, eps=eps)


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
def tversky(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    alpha: float = 0.5,
    beta: float = 0.5,
    eps: float = _EPS,
) -> torch.Tensor:
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


# ----------------- Heavy metrics (distance in PIXELS, EDT-based) -----------------
def _require_scipy() -> None:
    if not _HAS_SCIPY:
        raise ImportError(
            "SciPy is required for EDT-based surface distances (HD95 / ASSD). "
            "Please install scipy (e.g. `pip install scipy`)."
        )


def _surface_np(mask_bool: "np.ndarray") -> "np.ndarray":
    """
    1-pixel surface of a binary mask (2D), using binary erosion.

    surface = mask XOR erode(mask)
    """
    if mask_bool.ndim != 2:
        raise ValueError(f"mask must be 2D, got {mask_bool.ndim}D")
    if mask_bool.sum() == 0:
        return np.zeros_like(mask_bool, dtype=bool)

    # 8-connected erosion with 3x3 structure
    er = binary_erosion(mask_bool, structure=np.ones((3, 3), dtype=bool), border_value=0)
    return np.logical_xor(mask_bool, er)


def _assd_and_hd95_pixels_2d(
    gt_bool: "np.ndarray",
    pr_bool: "np.ndarray",
) -> Tuple[float, float]:
    """
    Compute (ASSD, HD95) in PIXELS for a single 2D pair, using EDT on surfaces.

    HD95 is computed as:
      HD95 = max(p95(dist(pr_surface -> gt_surface)), p95(dist(gt_surface -> pr_surface)))
    """
    s_gt = _surface_np(gt_bool)
    s_pr = _surface_np(pr_bool)

    H, W = gt_bool.shape
    diag = float(np.sqrt(H * H + W * W))

    # Edge cases: empty surfaces
    if s_gt.sum() == 0 and s_pr.sum() == 0:
        return 0.0, 0.0
    if s_gt.sum() == 0 or s_pr.sum() == 0:
        return diag, diag

    # EDT gives distance to nearest "False" (0) pixel; so compute EDT on complement of surface.
    dt_gt = distance_transform_edt(~s_gt)
    dt_pr = distance_transform_edt(~s_pr)

    d_pr_to_gt = dt_gt[s_pr]  # distances at predicted surface pixels to nearest GT surface
    d_gt_to_pr = dt_pr[s_gt]  # distances at GT surface pixels to nearest predicted surface

    # Safety
    if d_pr_to_gt.size == 0 or d_gt_to_pr.size == 0:
        return diag, diag

    assd = 0.5 * (float(d_pr_to_gt.mean()) + float(d_gt_to_pr.mean()))
    hd95 = max(
        float(np.percentile(d_pr_to_gt, 95)),
        float(np.percentile(d_gt_to_pr, 95)),
    )
    return assd, hd95


def normalized_surface_distance(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    tau_px: float = 1.0,
    eps: float = _EPS,
) -> torch.Tensor:
    """
    Normalized Surface Distance (NSD) / Surface Dice with tolerance tau.

    Returns in [0,1], higher is better.
    NSD = ( #pred-surface points within tau of GT surface  +  #GT-surface points within tau of pred surface )
          / ( #pred-surface points + #GT-surface points )

    Edge cases (choose and document):
      - both empty => 1.0 (perfect match of "no object")
      - exactly one empty => 0.0
    """
    _require_scipy()
    yt, yp = _ensure_nchw_1(y_true, y_pred)

    yp_bin = (yp >= 0.5)
    yt_bin = (yt >= 0.5)

    vals = []
    B = int(yt.shape[0])
    for b in range(B):
        gt = yt_bin[b, 0].detach().cpu().numpy().astype(bool)
        pr = yp_bin[b, 0].detach().cpu().numpy().astype(bool)

        s_gt = _surface_np(gt)
        s_pr = _surface_np(pr)

        n_gt = int(s_gt.sum())
        n_pr = int(s_pr.sum())

        if n_gt == 0 and n_pr == 0:
            vals.append(torch.tensor(1.0, device=yt.device, dtype=yt.dtype))
            continue
        if n_gt == 0 or n_pr == 0:
            vals.append(torch.tensor(0.0, device=yt.device, dtype=yt.dtype))
            continue

        # distance to nearest surface pixel (EDT on complement of surface)
        dt_gt = distance_transform_edt(~s_gt)
        dt_pr = distance_transform_edt(~s_pr)

        # count surface points within tolerance
        tp_pr = float((dt_gt[s_pr] <= tau_px).sum())
        tp_gt = float((dt_pr[s_gt] <= tau_px).sum())

        nsd = (tp_pr + tp_gt) / (n_pr + n_gt + eps)
        vals.append(torch.tensor(nsd, device=yt.device, dtype=yt.dtype))

    return torch.stack(vals).mean()



def Hausdorff_distance(y_true: torch.Tensor, y_pred: torch.Tensor, eps: float = _EPS) -> torch.Tensor:
    """
    95th percentile Hausdorff distance (HD95) in PIXELS.
    Lower is better.

    Definition:
      HD95 = max( p95(pred_surface -> true_surface), p95(true_surface -> pred_surface) )

    Uses exact Euclidean Distance Transform (EDT) on 1-pixel surfaces.

    Edge cases:
      - both masks empty => 0
      - exactly one mask empty => image diagonal in pixels (penalty)
    """
    _require_scipy()
    yt, yp = _ensure_nchw_1(y_true, y_pred)

    yp_bin = (yp >= 0.5)
    yt_bin = (yt >= 0.5)

    vals = []
    B = int(yt.shape[0])
    for b in range(B):
        gt = yt_bin[b, 0].detach().cpu().numpy().astype(bool)
        pr = yp_bin[b, 0].detach().cpu().numpy().astype(bool)
        _, hd95 = _assd_and_hd95_pixels_2d(gt, pr)
        vals.append(torch.tensor(hd95, device=yt.device, dtype=yt.dtype))
    return torch.stack(vals).mean()


def boundary_intersection_over_union(y_true: torch.Tensor, y_pred: torch.Tensor, eps: float = _EPS) -> torch.Tensor:
    """
    Boundary IoU: IoU computed specifically on boundary regions.
    Higher is better.

    Note: this is very sensitive to 1px shifts if boundary thickness is 1 pixel.
    """
    yt, yp = _ensure_nchw_1(y_true, y_pred)

    yp_bin = (yp >= 0.5).float()
    yt_bin = (yt >= 0.5).float()

    # Keep your existing erosion-based boundary extraction (torch, GPU-friendly).
    boundary_true = _extract_boundaries(yt_bin, dilation_iters=1)
    boundary_pred = _extract_boundaries(yp_bin, dilation_iters=1)

    intersection = torch.sum(boundary_true * boundary_pred, dim=(1, 2, 3))
    union = torch.sum(boundary_true + boundary_pred, dim=(1, 2, 3)) - intersection

    boundary_iou = torch.where(
        union > eps,
        (intersection + eps) / (union + eps),
        torch.ones_like(intersection),
    )
    return boundary_iou.mean()


def _extract_boundaries(mask: torch.Tensor, dilation_iters: int = 1) -> torch.Tensor:
    """
    Extract boundaries from a binary mask using erosion (torch).
    Returns boundary pixels (edge between foreground and background).
    """
    binary_mask = (mask >= 0.5).float()
    kernel = torch.ones(1, 1, 3, 3, device=mask.device, dtype=mask.dtype)

    eroded = binary_mask
    for _ in range(max(1, int(dilation_iters))):
        eroded = F.conv2d(eroded, kernel, padding=1)
        eroded = (eroded >= 9.0).float()

    boundary = torch.clamp(binary_mask - eroded, 0.0, 1.0)
    return boundary


# ----------------- Factory -----------------
def get_loss(name: str, alphabeta=(0.5, 0.5)):
    n = (name or "").lower()
    if n.startswith("tversky"):
        a, b = alphabeta if isinstance(alphabeta, (tuple, list)) and len(alphabeta) == 2 else (0.5, 0.5)
        return _tversky_loss(float(a), float(b))
    if n in {"dice", "dice_loss"}:
        return dice_loss
    if n in {"bce"}:
        return lambda yt, yp: F.binary_cross_entropy(
            torch.clamp(yp.float(), _EPS, 1.0 - _EPS),
            torch.clamp(yt.float(), 0.0, 1.0),
        )
    # default
    a, b = alphabeta if isinstance(alphabeta, (tuple, list)) and len(alphabeta) == 2 else (0.5, 0.5)
    return _tversky_loss(float(a), float(b))
