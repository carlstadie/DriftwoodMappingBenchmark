# core/common/console.py
from __future__ import annotations

from typing import Any, Dict, List


# --------- Console color helpers ---------
class _C:
    RED = "\033[31m"
    YELLOW = "\033[33m"
    GREEN = "\033[32m"
    RESET = "\033[0m"


def _col(s: str, color: str) -> str:
    """Wrap string s with the provided ANSI color and reset."""
    return f"{color}{s}{_C.RESET}"


# --------- Pretty-print helpers ----------
def _fmt_seconds(s: float) -> str:
    """Format seconds as HH:MM:SS."""
    m, s = divmod(int(s), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _ordered_metric_names() -> List[str]:
    """Preferred display order for printed logs. Basically diplay all we have lol."""
    return [
        "loss",
        "val_loss",
        "dice_coef",
        "val_dice_coef",
        "accuracy",
        "val_accuracy",
        "sensitivity",
        "val_sensitivity",
        "specificity",
        "val_specificity",
        "f1_score",
        "val_f1_score",
        "f_beta",
        "val_f_beta",
        "IoU",
        "val_IoU",
        "dice_loss",
        "val_dice_loss",
        "nominal_surface_distance",
        "val_nominal_surface_distance",
        "Hausdorff_distance",
        "val_Hausdorff_distance",
        "boundary_intersection_over_union",
        "val_boundary_intersection_over_union",
    ]


def _format_logs_for_print(logs: Dict[str, Any]) -> str:
    """
    Turn a logs dict into a single readable console line using the preferred order.

    Keeps epoch summaries compact and consistent with progress-bar style output.
    """
    keys = _ordered_metric_names()
    parts = []
    for k in keys:
        if k in logs and logs[k] is not None:
            try:
                parts.append(f"{k}={float(logs[k]):.4f}")
            except Exception:
                parts.append(f"{k}={logs[k]}")
    # Append any extra metrics that weren't in the preferred list.
    for k, v in logs.items():
        if k not in keys:
            try:
                parts.append(f"{k}={float(v):.4f}")
            except Exception:
                parts.append(f"{k}={v}")
    return " | ".join(parts)
