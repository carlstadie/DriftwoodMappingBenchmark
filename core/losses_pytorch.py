import torch

def get_loss(loss_fn, tversky_alpha_beta=None):
    """
    Wrapper to retrieve a loss function by name.
    If `loss_fn` is "tversky", returns a Tversky loss with optional (alpha, beta).
    If `loss_fn` is "dice", returns the Dice loss.
    Otherwise returns `loss_fn` unmodified (e.g. a built-in loss).
    """
    if loss_fn == "tversky":
        if tversky_alpha_beta is not None:
            alpha, beta = tversky_alpha_beta
            return lambda y_true, y_pred: tversky(y_true, y_pred, alpha=alpha, beta=beta)
        return tversky
    elif loss_fn == "dice":
        return dice_loss
    else:
        return loss_fn


def tversky(y_true, y_pred, alpha: float = 0.4, beta: float = 0.6):
    """
    Tversky loss for imbalanced data.
    alpha: weight of false positives
    beta:  weight of false negatives
    """
    # extract the single channel from one-hot ground truth
    y_t = y_true[..., 0].unsqueeze(-1)            # shape [...,1]
    ones = torch.ones_like(y_pred)
    p0, p1 = y_pred, ones - y_pred
    g0, g1 = y_t,    ones - y_t

    tp = (p0 * g0).sum()
    fp = alpha * (p0 * g1).sum()
    fn = beta  * (p1 * g0).sum()

    score = tp / (tp + fp + fn + 1e-5)
    return 1.0 - score


def accuracy(y_true, y_pred):
    """Element-wise accuracy (0/1) tensor."""
    y_t = y_true[..., 0].unsqueeze(-1)
    return (torch.round(y_t) == torch.round(y_pred)).float()


def dice_coef(y_true, y_pred, smooth: float = 1e-7):
    """
    Dice coefficient per sample (returns a tensor of shape [batch_size]).
    """
    y_t = y_true[..., 0].unsqueeze(-1)
    # sum over all dims except batch
    dims = tuple(range(1, y_pred.dim()))
    intersection = (y_t * y_pred).abs().sum(dim=dims)
    union = y_t.sum(dim=dims) + y_pred.sum(dim=dims)
    return (2.0 * intersection + smooth) / (union + smooth)


def dice_loss(y_true, y_pred):
    """Dice loss per sample."""
    return 1.0 - dice_coef(y_true, y_pred)


def true_positives(y_true, y_pred):
    y_t = y_true[..., 0].unsqueeze(-1)
    return torch.round(y_t * y_pred)


def false_positives(y_true, y_pred):
    y_t = y_true[..., 0].unsqueeze(-1)
    return torch.round((1.0 - y_t) * y_pred)


def true_negatives(y_true, y_pred):
    y_t = y_true[..., 0].unsqueeze(-1)
    return torch.round((1.0 - y_t) * (1.0 - y_pred))


def false_negatives(y_true, y_pred):
    y_t = y_true[..., 0].unsqueeze(-1)
    return torch.round(y_t * (1.0 - y_pred))


def sensitivity(y_true, y_pred):
    """Recall: TP / (TP + FN)"""
    tp = true_positives(y_true, y_pred).sum()
    fn = false_negatives(y_true, y_pred).sum()
    return tp / (tp + fn + 1e-5)


def specificity(y_true, y_pred):
    """Precision: TN / (TN + FP)"""
    tn = true_negatives(y_true, y_pred).sum()
    fp = false_positives(y_true, y_pred).sum()
    return tn / (tn + fp + 1e-5)


def f_beta(y_true, y_pred, beta: float = 1.0):
    """F-beta score"""
    tp = true_positives(y_true, y_pred).sum()
    fp = false_positives(y_true, y_pred).sum()
    fn = false_negatives(y_true, y_pred).sum()

    precision = tp / (tp + fp + 1e-5)
    recall    = tp / (tp + fn + 1e-5)
    bb = beta ** 2
    return (1 + bb) * precision * recall / (bb * precision + recall + 1e-5)


def f1_score(y_true, y_pred):
    """F1 = F-beta with beta=1"""
    return f_beta(y_true, y_pred, beta=1.0)


def IoU(y_true, y_pred):
    """Intersection over Union"""
    tp = true_positives(y_true, y_pred).sum()
    fp = false_positives(y_true, y_pred).sum()
    fn = false_negatives(y_true, y_pred).sum()
    return tp / (tp + fp + fn + 1e-5)


def nominal_surface_distance(y_true, y_pred):
    """Nominal surface distance (FP / (TP + FP + FN))"""
    tp = true_positives(y_true, y_pred).sum()
    fp = false_positives(y_true, y_pred).sum()
    fn = false_negatives(y_true, y_pred).sum()
    return fp / (tp + fp + fn + 1e-5)


def Hausdorff_distance(y_true, y_pred):
    """Here defined same as nominal surface distance."""
    return nominal_surface_distance(y_true, y_pred)


def boundary_intersection_over_union(y_true, y_pred):
    """Here the same as IoU."""
    return IoU(y_true, y_pred)
