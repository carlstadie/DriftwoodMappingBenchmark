# core/common/console.py
from __future__ import annotations

from typing import Any, Dict, List, Optional


# --------- Console color helpers ---------
class _C:
    RED = "\033[31m"
    YELLOW = "\033[33m"
    GREEN = "\033[32m"
    CYAN = "\033[36m"
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
    parts: List[str] = []
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


# --------- Search space formatting helpers ----------
def _fmt_float(v: float) -> str:
    """Format float with compact scientific notation when helpful."""
    if abs(v) >= 1e4 or (0 < abs(v) < 1e-4):
        return f"{v:.2e}"
    return f"{v:.4f}"


def _log_bar(low: float, high: float, width: int = 20) -> str:
    """
    Render a simple log-scale range hint (ASCII).

    Example:
        '1e-5 ───────────── 5e-3'
    """
    left = _fmt_float(low)
    right = _fmt_float(high)
    bar = "─" * max(3, int(width))
    return f"{left} {bar} {right}"


def _chips(vals: List[Any]) -> str:
    """Render a small set as {a, b, c}."""
    return "{%s}" % (", ".join(str(v) for v in vals))


def _hdr(title: str) -> None:
    """Print a section header bar + title + rule."""
    line = "=" * 60
    print(f"{line}\n{title}\n" + "-" * 60)


# -----------------------
# Search space printers
# -----------------------
def _print_unet_space(
    phase: str,
    conf,
    hb_data_hp: Dict[str, Any],
    hb_best: Optional[Dict[str, Any]],
    fixed: Optional[Dict[str, Any]],
) -> None:
    _hdr("UNet — SEARCH SPACE")
    print(f"Phase: {phase}")
    if phase == "HB":
        print("- Architecture")
        print(f"  - dilation_rate      : {_chips([1, 2, 4])}")
        print(f"  - layer_count        : {_chips([32, 64, 96])}")
        print(f"  - l2_weight          : {_chips([0.0, 1e-5, 1e-4])}")
        print(f"  - dropout            : {_chips([0.0, 0.1, 0.2])}")
        print("- Optimization")
        print(f"  - optimizer          : {_chips(['adam', 'adamw'])}")
        print(f"  - learning_rate (log): {_log_bar(1e-5, 5e-3)}")
        print(f"  - weight_decay (log) : {_log_bar(1e-6, 1e-2)} (only if adamw)")
        print(f"  - scheduler          : {_chips(['none', 'cosine', 'onecycle'])}")
    else:
        print("- Fixed from HB       : architecture knobs (dilation_rate, layer_count, l2_weight, dropout)")
        if hb_best is not None:
            arch = {
                k: hb_best.get(k)
                for k in ("dilation_rate", "layer_count", "l2_weight", "dropout")
                if k in hb_best
            }
            if arch:
                print(f"  - chosen             : {arch}")
        print("- Tuned               : optimizer (fixed), learning_rate, weight_decay (if adamw), scheduler")
        print(f"  - learning_rate (log): {_log_bar(1e-5, 5e-3)}")
        print(f"  - weight_decay (log) : {_log_bar(1e-6, 1e-2)} (only if adamw)")
        print(f"  - scheduler          : {_chips(['none', 'cosine', 'onecycle'])}")

    # Data for the tuning run
    ph, pw = hb_data_hp.get("patch_h"), hb_data_hp.get("patch_w")
    print("- Data (fixed for tuning run)")
    print(f"  - patch_h x patch_w  : {ph} x {pw}")
    print(f"  - augmenter_strength : {hb_data_hp.get('augmenter_strength')}")
    print(f"  - min_pos_frac       : {hb_data_hp.get('min_pos_frac')}")
    print("=" * 60)


def _print_swin_space(
    phase: str,
    conf,
    hb_data_hp: Dict[str, Any],
    hb_best: Optional[Dict[str, Any]],
    fixed: Optional[Dict[str, Any]],
) -> None:
    _hdr("Swin-UNet — SEARCH SPACE")
    ps = hb_data_hp.get("fixed", {}).get("patch_size", 4)
    ws = hb_data_hp.get("fixed", {}).get("window_size", 7)
    use_imnet = bool(getattr(conf, "use_imagenet_weights", False))

    print(f"Phase: {phase}")
    if phase == "HB":
        print("- Architecture (official Swin-style)")
        print(f"  - C (embed dim)      : {_chips([64, 96])}")
        print(f"  - drop_path          : {_chips([0.0, 0.1, 0.2, 0.3])}")
        print(f"  - patch_size         : {ps}   (fixed)")
        print(f"  - window_size        : {ws}   (fixed)")
        print("- Optimization")
        print(f"  - optimizer          : {_chips(['adam', 'adamw'])}")
        print(f"  - learning_rate (log): {_log_bar(1e-5, 5e-3)}")
        print(f"  - weight_decay (log) : {_log_bar(1e-6, 1e-2)} (only if adamw)")
        print(f"  - scheduler          : {_chips(['none', 'cosine', 'onecycle'])}")
        if use_imnet:
            print(
                _col(
                    "- Note                : use_imagenet_weights=True --> C must be 96 (Swin-T).",
                    _C.YELLOW,
                )
            )
    else:
        C_fixed = fixed.get("C") if fixed else None
        dp_fixed = fixed.get("drop_path") if fixed else None
        print("- Fixed from HB")
        if C_fixed is not None or dp_fixed is not None:
            print(f"  - C={C_fixed}, drop_path={dp_fixed}, patch_size={ps}, window_size={ws}")
        else:
            print("  - C, drop_path, patch_size, window_size")
        print("- Tuned               : optimizer (fixed), learning_rate, weight_decay (if adamw), scheduler")
        print(f"  - learning_rate (log): {_log_bar(1e-5, 5e-3)}")
        print(f"  - weight_decay (log) : {_log_bar(1e-6, 1e-2)} (only if adamw)")
        print(f"  - scheduler          : {_chips(['none', 'cosine', 'onecycle'])}")
        if use_imnet and C_fixed != 96:
            print(_col("- Note                : use_imagenet_weights=True → forcing C=96.", _C.YELLOW))

    ph, pw = hb_data_hp.get("patch_h"), hb_data_hp.get("patch_w")
    print("- Data (fixed for tuning run)")
    print(f"  - patch_h x patch_w  : {ph} x {pw} (snapped for Swin constraints)")
    print("=" * 60)


def _print_tm_space(
    phase: str,
    conf,
    hb_data_hp: Dict[str, Any],
    hb_best: Optional[Dict[str, Any]],
    fixed: Optional[Dict[str, Any]],
) -> None:
    _hdr("TerraMind — SEARCH SPACE")
    print(f"Phase: {phase}")
    if phase == "HB":
        print("- Head/Decoder")
        print(f"  - tm_decoder         : {_chips(['UNetDecoder', 'UperNetDecoder'])}")
        print(f"  - tm_decoder_channels: {_chips([128, 192, 256, 384])}")
        print(f"  - tm_head_dropout    : {_log_bar(0.0, 0.2)}")
        print(f"  - tm_freeze_backbone_epochs: {_chips([0, 1, 3, 5])}")
        print("- TerraMind-specific LRs")
        print(f"  - tm_lr_backbone (log): {_log_bar(1e-6, 3e-5)}")
        print(f"  - tm_lr_head_mult     : {_chips([5.0, 10.0])}")
        print(f"  - tm_weight_decay(log): {_log_bar(1e-6, 5e-4)}")
        print("- Global optimization")
        print(f"  - optimizer          : {_chips(['adam', 'adamw'])}")
        print(f"  - learning_rate (log): {_log_bar(1e-5, 5e-3)}")
        print(f"  - weight_decay (log) : {_log_bar(1e-6, 1e-2)} (only if adamw)")
        print(f"  - scheduler          : {_chips(['none', 'cosine', 'onecycle'])}")
    else:
        print("- Fixed from HB       : decoder choices, channels, head dropout, freeze epochs")
        if fixed:
            keep = ("tm_decoder", "tm_decoder_channels", "tm_head_dropout", "tm_freeze_backbone_epochs")
            chosen = {k: fixed[k] for k in keep if k in fixed}
            if chosen:
                print(f"  - chosen             : {chosen}")
        print("- Tuned")
        print("  - optimizer (fixed), learning_rate, weight_decay (if adamw), scheduler")
        print("  - tm_lr_backbone (log), tm_lr_head_mult, tm_weight_decay (log)")
        print(f"  - learning_rate (log): {_log_bar(1e-5, 5e-3)}")
        print(f"  - weight_decay (log) : {_log_bar(1e-6, 1e-2)} (only if adamw)")
        print(f"  - scheduler          : {_chips(['none', 'cosine', 'onecycle'])}")

    ph, pw = hb_data_hp.get("patch_h"), hb_data_hp.get("patch_w")
    print("- Data (fixed for tuning run)")
    print(f"  - patch_h x patch_w  : {ph} x {pw}")
    print("=" * 60)


# -----------------------
# Public helper
# -----------------------
def print_search_space(
    model_key: str,
    phase: str,
    conf,
    hb_data_hp: Dict[str, Any],
    hb_best: Optional[Dict[str, Any]] = None,
    fixed: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Pretty-print the tuning search space for a given model and phase.

    Args:
        model_key: 'unet' | 'swin' | 'tm'.
        phase: 'HB' or 'BO'.
        conf: config object (used for flags like use_imagenet_weights).
        hb_data_hp: dict containing data H/W and fixed patch/window (for Swin).
        hb_best: (optional) best params from HB when printing BO.
        fixed: (optional) fixed dict used for BO.

    Returns:
        None (prints to stdout).
    """
    k = (model_key or "").lower()
    p = (phase or "").upper()
    if k == "unet":
        _print_unet_space(p, conf, hb_data_hp, hb_best, fixed)
    elif k == "swin":
        _print_swin_space(p, conf, hb_data_hp, hb_best, fixed)
    else:
        _print_tm_space(p, conf, hb_data_hp, hb_best, fixed)
