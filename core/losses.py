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


# ---- Heavy metrics  ----
def _extract_boundaries(mask: torch.Tensor, dilation_iters: int = 1) -> torch.Tensor:
    """
    Extract boundaries from a binary mask using morphological operations.
    Returns the boundary pixels (edge between foreground and background).
    
    Args:
        mask: Binary mask of shape (B, 1, H, W) with values in [0, 1]
        dilation_iters: Number of dilation iterations for boundary thickness
    
    Returns:
        Boundary mask of shape (B, 1, H, W)
    """
    # Binarize the mask
    binary_mask = (mask >= 0.5).float()
    
    # Create erosion kernel (3x3)
    kernel = torch.ones(1, 1, 3, 3, device=mask.device, dtype=mask.dtype)
    
    # Erode the mask
    # Use padding to maintain size
    eroded = binary_mask
    for _ in range(dilation_iters):
        eroded = F.conv2d(eroded, kernel, padding=1)
        eroded = (eroded >= 9.0).float()  # All 9 neighbors must be 1
    
    # Boundary is original mask minus eroded mask
    boundary = binary_mask - eroded
    boundary = torch.clamp(boundary, 0.0, 1.0)
    
    return boundary


def _compute_distance_transform(mask: torch.Tensor, eps: float = _EPS) -> torch.Tensor:
    """
    Approximate distance transform using repeated max pooling.
    For each background pixel, approximates distance to nearest foreground pixel.
    
    Args:
        mask: Binary mask of shape (B, 1, H, W)
    
    Returns:
        Distance map of shape (B, 1, H, W)
    """
    binary_mask = (mask >= 0.5).float()
    distance_map = torch.zeros_like(binary_mask)
    
    # Invert mask to get background
    background = 1.0 - binary_mask
    
    # For pixels in the background, compute approximate distance
    current = binary_mask.clone()
    dist = 0.0
    
    # Iteratively grow the foreground region and track distance
    for d in range(1, min(mask.shape[-2], mask.shape[-1]) // 2):
        # Dilate current mask
        dilated = F.max_pool2d(
            F.pad(current, (1, 1, 1, 1), mode='constant', value=0),
            kernel_size=3,
            stride=1
        )
        
        # Find newly covered background pixels
        newly_covered = (dilated > current) * background
        
        # Update distance map for newly covered pixels
        distance_map = distance_map + newly_covered * float(d)
        
        # Update current mask
        current = dilated
        
        # Stop if all background is covered
        if newly_covered.sum() < eps:
            break
    
    return distance_map


def nominal_surface_distance(y_true: torch.Tensor, y_pred: torch.Tensor, eps: float = _EPS) -> torch.Tensor:
    """
    Normalized Surface Distance (NSD): Average distance between predicted and true boundaries.
    Lower is better. Returns normalized value in [0, 1] range.
    
    This computes the mean distance from boundary pixels of prediction to boundary pixels of ground truth,
    and vice versa, then averages both directions.
    """
    yt, yp = _ensure_nchw_1(y_true, y_pred)
    
    # Binarize predictions
    yp_bin = (yp >= 0.5).float()
    yt_bin = (yt >= 0.5).float()
    
    # Extract boundaries
    boundary_true = _extract_boundaries(yt_bin, dilation_iters=1)
    boundary_pred = _extract_boundaries(yp_bin, dilation_iters=1)
    
    # If either boundary is empty, return a penalty
    if boundary_true.sum() < eps or boundary_pred.sum() < eps:
        return torch.tensor(1.0, device=yt.device, dtype=yt.dtype)
    
    # Compute distance transforms
    dist_true = _compute_distance_transform(yt_bin)
    dist_pred = _compute_distance_transform(yp_bin)
    
    # Average distance from pred boundary to true mask
    pred_to_true_dist = (boundary_pred * dist_true).sum() / (boundary_pred.sum() + eps)
    
    # Average distance from true boundary to pred mask
    true_to_pred_dist = (boundary_true * dist_pred).sum() / (boundary_true.sum() + eps)
    
    # Average both directions and normalize by image diagonal
    max_dist = torch.sqrt(
        torch.tensor(yt.shape[-2]**2 + yt.shape[-1]**2, dtype=yt.dtype, device=yt.device)
    )
    nsd = (pred_to_true_dist + true_to_pred_dist) / (2.0 * max_dist + eps)
    
    return torch.clamp(nsd, 0.0, 1.0)


def Hausdorff_distance(y_true: torch.Tensor, y_pred: torch.Tensor, eps: float = _EPS) -> torch.Tensor:
    """
    Hausdorff Distance: Maximum distance from any boundary point to the nearest point on the other boundary.
    Lower is better. Returns normalized value in [0, 1] range.
    
    Computes max(max distance from pred to true, max distance from true to pred).
    """
    yt, yp = _ensure_nchw_1(y_true, y_pred)
    
    # Binarize predictions
    yp_bin = (yp >= 0.5).float()
    yt_bin = (yt >= 0.5).float()
    
    # Extract boundaries
    boundary_true = _extract_boundaries(yt_bin, dilation_iters=1)
    boundary_pred = _extract_boundaries(yp_bin, dilation_iters=1)
    
    # If either boundary is empty, return maximum penalty
    if boundary_true.sum() < eps or boundary_pred.sum() < eps:
        return torch.tensor(1.0, device=yt.device, dtype=yt.dtype)
    
    # Compute distance transforms
    dist_true = _compute_distance_transform(yt_bin)
    dist_pred = _compute_distance_transform(yp_bin)
    
    # Max distance from pred boundary points to true mask
    pred_to_true_max = (boundary_pred * dist_true).max()
    
    # Max distance from true boundary points to pred mask
    true_to_pred_max = (boundary_true * dist_pred).max()
    
    # Hausdorff is the maximum of these two
    hausdorff = torch.max(pred_to_true_max, true_to_pred_max)
    
    # Normalize by image diagonal
    max_dist = torch.sqrt(
        torch.tensor(yt.shape[-2]**2 + yt.shape[-1]**2, dtype=yt.dtype, device=yt.device)
    )
    hausdorff_normalized = hausdorff / (max_dist + eps)
    
    return torch.clamp(hausdorff_normalized, 0.0, 1.0)


def boundary_intersection_over_union(y_true: torch.Tensor, y_pred: torch.Tensor, eps: float = _EPS) -> torch.Tensor:
    """
    Boundary IoU: IoU computed specifically on the boundary regions.
    Higher is better. Returns value in [0, 1] range.
    
    This focuses on how well the boundaries align, ignoring the interior regions.
    """
    yt, yp = _ensure_nchw_1(y_true, y_pred)
    
    # Binarize predictions
    yp_bin = (yp >= 0.5).float()
    yt_bin = (yt >= 0.5).float()
    
    # Extract boundaries
    boundary_true = _extract_boundaries(yt_bin, dilation_iters=1)
    boundary_pred = _extract_boundaries(yp_bin, dilation_iters=1)
    
    # Compute IoU on boundaries
    intersection = torch.sum(boundary_true * boundary_pred, dim=(1, 2, 3))
    union = torch.sum(boundary_true + boundary_pred, dim=(1, 2, 3)) - intersection
    
    # Handle case where both boundaries are empty (perfect match of empty masks)
    boundary_iou = torch.where(
        union > eps,
        (intersection + eps) / (union + eps),
        torch.ones_like(intersection)
    )
    
    return boundary_iou.mean()


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
