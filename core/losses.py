# core/losses_pytorch.py
#    Edited by Sizhuo Li
#    Author: Ankit Kariryaa, University of Bremen
#
#    PyTorch rewrite aligned to the TensorFlow/Keras version

from __future__ import annotations

import torch
import torch.nn.functional as F


def get_loss(loss_fn, tversky_alpha_beta=None):
    """Wrapper to mirror TF get_loss behavior."""
    if loss_fn == "tversky":
        if tversky_alpha_beta:
            alpha, beta = tversky_alpha_beta
            def _tversky(y_true, y_pred):
                return tversky(y_true, y_pred, alpha=alpha, beta=beta)
            return _tversky
        return tversky
    elif loss_fn == "dice":
        return dice_loss
    else:
        # If a callable (or torch loss) was passed through config, just return it
        return loss_fn


# --- helpers ---

def _ensure_ch1(y: torch.Tensor) -> torch.Tensor:
    """
    Ensure shape (N, 1, H, W) by selecting channel 0 if needed.
    TF code uses y_true[..., 0] so we mimic the same in NCHW.
    """
    if y.ndim == 3:
        y = y.unsqueeze(1)
    if y.ndim != 4:
        raise ValueError(f"Expected (N, C, H, W); got {tuple(y.shape)}")
    if y.shape[1] != 1:
        y = y[:, 0:1, ...]
    return y


# --- losses / metrics (constants match TF) ---

def tversky(y_true: torch.Tensor, y_pred: torch.Tensor, alpha: float = 0.40, beta: float = 0.60):
    """
    Tversky loss (TF equivalent).
    TF uses EPSILON = 1e-5 in the denominator.
    """
    y_t = _ensure_ch1(y_true).float()
    p0 = y_pred.float()                 # prob of class
    p1 = 1.0 - p0                       # prob of not-class
    g0 = y_t
    g1 = 1.0 - y_t

    tp = torch.sum(p0 * g0)
    fp = alpha * torch.sum(p0 * g1)
    fn = beta  * torch.sum(p1 * g0)

    eps = 1.0e-5
    score = tp / (tp + fp + fn + eps)
    return 1.0 - score


def dice_coef(y_true: torch.Tensor, y_pred: torch.Tensor, smooth: float = 1.0e-7):
    """
    Dice coefficient (mirrors TF formula & smooth).
    """
    y_t = _ensure_ch1(y_true).float()
    y_p = y_pred.float()
    intersection = torch.sum(torch.abs(y_t * y_p))
    union = torch.sum(y_t) + torch.sum(y_p)
    return (2.0 * intersection + smooth) / (union + smooth)


def dice_loss(y_true: torch.Tensor, y_pred: torch.Tensor):
    """Dice loss = 1 - Dice coefficient."""
    return 1.0 - dice_coef(y_true, y_pred)


def accuracy(y_true: torch.Tensor, y_pred: torch.Tensor):
    """
    Accuracy: TF uses K.equal(K.round(y_t), K.round(y_pred)).
    Return the mean equality (scalar), same effective aggregation.
    """
    y_t = _ensure_ch1(y_true).float()
    pred_b = torch.round(y_pred.float())
    true_b = torch.round(y_t)
    return (pred_b == true_b).float().mean()


def true_positives(y_true: torch.Tensor, y_pred: torch.Tensor):
    y_t = _ensure_ch1(y_true).float()
    return torch.round(y_t * y_pred.float())


def false_positives(y_true: torch.Tensor, y_pred: torch.Tensor):
    y_t = _ensure_ch1(y_true).float()
    return torch.round((1.0 - y_t) * y_pred.float())


def true_negatives(y_true: torch.Tensor, y_pred: torch.Tensor):
    y_t = _ensure_ch1(y_true).float()
    return torch.round((1.0 - y_t) * (1.0 - y_pred.float()))


def false_negatives(y_true: torch.Tensor, y_pred: torch.Tensor):
    y_t = _ensure_ch1(y_true).float()
    return torch.round(y_t * (1.0 - y_pred.float()))


def sensitivity(y_true: torch.Tensor, y_pred: torch.Tensor):
    """Recall = TP / (TP + FN)."""
    tp = true_positives(y_true, y_pred)
    fn = false_negatives(y_true, y_pred)
    denom = torch.sum(tp) + torch.sum(fn)
    return torch.sum(tp) / (denom + 1e-12)  # tiny guard; TF has no eps here


def specificity(y_true: torch.Tensor, y_pred: torch.Tensor):
    """Specificity = TN / (TN + FP)."""
    tn = true_negatives(y_true, y_pred)
    fp = false_positives(y_true, y_pred)
    denom = torch.sum(tn) + torch.sum(fp)
    return torch.sum(tn) / (denom + 1e-12)  # tiny guard; TF has no eps here


def f_beta(y_true: torch.Tensor, y_pred: torch.Tensor, beta: float = 1):
    """
    F-beta score.
    TF uses K.epsilon() only in the final denominator; we keep tiny guards in precision/recall too
    to avoid NaNs while keeping behavior effectively identical.
    """
    tp = true_positives(y_true, y_pred)
    fp = false_positives(y_true, y_pred)
    fn = false_negatives(y_true, y_pred)

    precision = torch.sum(tp) / (torch.sum(tp) + torch.sum(fp) + 1e-12)
    recall    = torch.sum(tp) / (torch.sum(tp) + torch.sum(fn) + 1e-12)

    beta2 = float(beta) ** 2
    return (1 + beta2) * precision * recall / (beta2 * precision + recall + 1e-7)  # 1e-7 ~ K.epsilon()


def f1_score(y_true: torch.Tensor, y_pred: torch.Tensor):
    return f_beta(y_true, y_pred, beta=1)


def IoU(y_true: torch.Tensor, y_pred: torch.Tensor):
    """Intersection-over-Union."""
    tp = true_positives(y_true, y_pred)
    fp = false_positives(y_true, y_pred)
    fn = false_negatives(y_true, y_pred)
    denom = torch.sum(tp) + torch.sum(fp) + torch.sum(fn)
    return torch.sum(tp) / (denom + 1e-12)


def nominal_surface_distance(y_true: torch.Tensor, y_pred: torch.Tensor):
    """Proxy metric as in TF source."""
    tp = true_positives(y_true, y_pred)
    fp = false_positives(y_true, y_pred)
    fn = false_negatives(y_true, y_pred)
    denom = torch.sum(tp) + torch.sum(fp) + torch.sum(fn)
    return torch.sum(fp) / (denom + 1e-12)


def Hausdorff_distance(y_true: torch.Tensor, y_pred: torch.Tensor):
    """Proxy metric as in TF source."""
    tp = true_positives(y_true, y_pred)
    fp = false_positives(y_true, y_pred)
    fn = false_negatives(y_true, y_pred)
    denom = torch.sum(tp) + torch.sum(fp) + torch.sum(fn)
    return torch.sum(fp) / (denom + 1e-12)


def boundary_intersection_over_union(y_true: torch.Tensor, y_pred: torch.Tensor):
    """Boundary IoU proxy as in TF source."""
    tp = true_positives(y_true, y_pred)
    fp = false_positives(y_true, y_pred)
    fn = false_negatives(y_true, y_pred)
    denom = torch.sum(tp) + torch.sum(fp) + torch.sum(fn)
    return torch.sum(tp) / (denom + 1e-12)
