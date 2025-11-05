from __future__ import annotations
import os, json, time, random
from datetime import datetime
from typing import Dict, Any, Tuple, List, Optional, Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingLR

try:
    import optuna
    from optuna.pruners import HyperbandPruner
    from optuna.samplers import TPESampler
except Exception as _e:
    raise ImportError("Optuna is required for tuning. Install with `pip install optuna`.") from _e

from training import get_all_frames, create_train_val_datasets
from core.UNet import UNet
from core.Swin_UNetPP import SwinUNet
from core.optimizers import get_optimizer  # optional project-level optimizer factory
from core.TerraMind import TerraMind  # your wrapper

# NEW: we’ll directly use the TerraTorch factory for Prithvi / SatMAE builds
from terratorch.models.encoder_decoder_factory import EncoderDecoderFactory


# -----------------------
# Utilities
# -----------------------
def _seed(seed: int = 42) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(False)  # match TF defaults

def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def _default(val, fallback):
    return val if val is not None else fallback

def _nan_to_num_torch(x: torch.Tensor, constant: float) -> torch.Tensor:
    return torch.where(torch.isfinite(x), x, torch.as_tensor(constant, dtype=x.dtype, device=x.device))

def _sanitize_pair_xy(x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    x = x.to(dtype=torch.float32)
    y = y.to(dtype=torch.float32)
    x = _nan_to_num_torch(x, 0.0).clamp_(0.0, 1.0)
    y = _nan_to_num_torch(y, 0.0).clamp_(0.0, 1.0)
    return x, y

def _sanitize_weights(w: torch.Tensor) -> torch.Tensor:
    w = w.to(dtype=torch.float32)
    w = _nan_to_num_torch(w, 1.0).clamp_(0.0, 1e6)
    return w

def _snap_hw_for_swin(H: int, W: int, patch_size: int, window_size: int, down_levels: int = 3) -> Tuple[int, int]:
    need = window_size * (2 ** down_levels)
    t_h = int(np.ceil(H / patch_size))
    t_w = int(np.ceil(W / patch_size))
    t_h_adj = int(np.ceil(t_h / need)) * need
    t_w_adj = int(np.ceil(t_w / need)) * need
    H_adj = t_h_adj * patch_size
    W_adj = t_w_adj * patch_size
    return H_adj, W_adj

def _flatten_nchw_like(y: torch.Tensor) -> torch.Tensor:
    return y.reshape(y.shape[0], -1)

def _safe_dice(y_true: torch.Tensor, y_pred: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    y_true = y_true.to(dtype=torch.float32).clamp(0.0, 1.0)
    y_pred = y_pred.to(dtype=torch.float32).clamp(0.0, 1.0)
    y_true_f = _flatten_nchw_like(y_true)
    y_pred_f = _flatten_nchw_like(y_pred)
    inter = torch.sum(y_true_f * y_pred_f, dim=1)
    den = torch.sum(y_true_f, dim=1) + torch.sum(y_pred_f, dim=1)
    dice = (2.0 * inter + eps) / (den + eps)
    return dice.mean()

def _safe_iou(y_true: torch.Tensor, y_pred: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    y_true = y_true.to(dtype=torch.float32).clamp(0.0, 1.0)
    y_pred = y_pred.to(dtype=torch.float32).clamp(0.0, 1.0)
    y_true_f = _flatten_nchw_like(y_true)
    y_pred_f = _flatten_nchw_like(y_pred)
    inter = torch.sum(y_true_f * y_pred_f, dim=1)
    union = torch.sum(y_true_f + y_pred_f, dim=1) - inter
    iou = (inter + eps) / (union + eps)
    return iou.mean()


# -----------------------
# Logging helpers (concise / readable)
# -----------------------
def _phase_banner(model_key: str, phase: str, max_epochs: int, steps: int, val_steps: int, tune_batch: int):
    print(f"\n=== [{model_key.upper()}][{phase}] epochs={max_epochs}, "
          f"steps/epoch={steps}, val_steps={val_steps}, batch={tune_batch} ===")

def _trial_banner(trial_num: int, model_key: str, hp: Dict[str, Any], data_hp: Dict[str, Any]):
    arch_bits = []
    if model_key == "unet":
        for k in ("dilation_rate", "layer_count", "l2_weight", "dropout"):
            if k in hp: arch_bits.append(f"{k}={hp[k]}")
    elif model_key == "swin":
        for k in ("C", "attn_drop", "proj_drop", "mlp_drop", "drop_path", "ss_size"):
            if k in hp: arch_bits.append(f"{k}={hp[k]}")
    else:  # TerraMind / Prithvi / SatMAE
        for k in ("fm_backbone", "tm_decoder", "tm_decoder_channels", "tm_head_dropout",
                  "tm_lr_backbone", "tm_lr_head_mult", "tm_weight_decay",
                  "tm_freeze_backbone_epochs"):
            if k in hp: arch_bits.append(f"{k}={hp[k]}")
    data_bits = f"{data_hp['patch_h']}x{data_hp['patch_w']}, aug={data_hp['augmenter_strength']}, minpos={data_hp['min_pos_frac']}"
    opt_bits = f"opt={hp['optimizer']}" if "optimizer" in hp else ""
    if "learning_rate" in hp:
        opt_bits += f", lr={hp['learning_rate']:.3g}"
    if "weight_decay" in hp:
        opt_bits += f", wd={hp['weight_decay']:.2g}"
    sch_bits = f"sched={hp.get('scheduler','none')}"
    arch_str = "; ".join(arch_bits) if arch_bits else "-"
    print(f"\n[trial {trial_num}] {opt_bits}, {sch_bits} | arch: {arch_str} | data: {data_bits}")

def _epoch_line(epoch: int, best: float, cur: float, extra: str = ""):
    msg = f"  epoch {epoch:02d}: val_dice={cur:.4f} (best={best:.4f})"
    if extra: msg += f" | {extra}"
    print(msg)

# --------- Console color helpers (match training style) ---------
class _C:
    RED = "\033[31m"
    YELLOW = "\033[33m"
    GREEN = "\033[32m"
    CYAN = "\033[36m"
    RESET = "\033[0m"
    BOLD = "\033[1m"

def _col(s: str, color: str) -> str:
    return f"{color}{s}{_C.RESET}"

def _format_val(v):
    # Short, readable numbers; keep other types as-is
    if isinstance(v, float):
        if (abs(v) > 1e4) or (0 < abs(v) < 1e-4):
            return f"{v:.2e}"
        return f"{v:.4f}"
    return str(v)

def _print_sxs_diff(prev_cfg: Dict[str, Any], curr_cfg: Dict[str, Any],
                    prev_score: Optional[float], curr_score: float) -> None:
    if prev_cfg is None:
        print("\n   baseline: no previous best to compare.")
        return
    keys = sorted(set(prev_cfg.keys()) | set(curr_cfg.keys()))
    rows = []
    for k in keys:
        a = prev_cfg.get(k, "-")
        b = curr_cfg.get(k, "-")
        rows.append((k, _format_val(a), _format_val(b), a != b))
    key_w  = max(len(k) for k, *_ in rows) if rows else 0
    prev_w = max(len(p) for _, p, _, _ in rows) if rows else 0
    curr_w = max(len(c) for _, _, c, _ in rows) if rows else 0
    print("   config diff (prev  |  curr):")
    for k, prev_s, curr_s, changed in rows:
        k_disp = _col(k, _C.YELLOW) if changed else k
        k_pad = " " * (key_w - len(k))
        mark = "  *" if changed else ""
        print(f"     - {k_disp}:{k_pad}  {prev_s:>{prev_w}}  |  {curr_s:>{curr_w}}{mark}")
    if prev_score is not None:
        delta = curr_score - prev_score
        sign = "+" if delta >= 0 else ""
        delta_str = f"{sign}{delta:.5f}"
        color = _C.GREEN if delta > 0 else (_C.RED if delta < 0 else _C.CYAN)
        print(f"\n   score: prev={prev_score:>{prev_w}.5f} | curr={curr_score:>{curr_w}.5f} ({_col(delta_str, color)})")


# -----------------------
# Impact-focused search spaces
# -----------------------
def _optimizer_space_hb(trial: optuna.Trial) -> Tuple[str, float, Optional[float]]:
    opt = trial.suggest_categorical("optimizer", ["adam", "adamw"])
    lr = trial.suggest_float("learning_rate", 1e-5, 5e-3, log=True)
    wd = None
    if opt == "adamw":
        wd = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    return opt, lr, wd

def _schedule_space_hb(trial: optuna.Trial) -> str:
    return trial.suggest_categorical("scheduler", ["none", "cosine", "onecycle"])

def _optimizer_space_bo(trial: optuna.Trial, fixed_opt: str) -> Tuple[str, float, Optional[float]]:
    lr = trial.suggest_float("learning_rate", 1e-5, 5e-3, log=True)
    wd = None
    if fixed_opt == "adamw":
        wd = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    return fixed_opt, lr, wd

def _unet_space_hb(trial: optuna.Trial) -> Tuple[int, int, float, float]:
    dilation = trial.suggest_categorical("dilation_rate", [1, 2, 4])
    layer_cnt = trial.suggest_categorical("layer_count", [32, 64, 96])
    l2w = trial.suggest_categorical("l2_weight", [0.0, 1e-5, 1e-4])
    drp = trial.suggest_categorical("dropout", [0.0, 0.1, 0.2])
    return int(dilation), int(layer_cnt), float(l2w), float(drp)

def _swin_space_hb(trial: optuna.Trial) -> Tuple[int, int, float, float, float, float]:
    C = trial.suggest_categorical("C", [48, 64, 96])
    ss_half = trial.suggest_categorical("use_shift", [0, 1])
    attn_drop = trial.suggest_categorical("attn_drop", [0.0, 0.1])
    proj_drop = trial.suggest_categorical("proj_drop", [0.0, 0.1])
    mlp_drop = trial.suggest_categorical("mlp_drop", [0.0, 0.1, 0.2])
    drop_path = trial.suggest_categorical("drop_path", [0.0, 0.1, 0.2])
    return int(C), int(ss_half), float(attn_drop), float(proj_drop), float(mlp_drop), float(drop_path)

def _data_space_hb(trial: optuna.Trial) -> Tuple[int, int, float, float]:
    valid_sizes = [32, 48, 64, 80, 96, 112, 128, 160, 192, 224,
                   256, 288, 320, 352, 384, 416, 448, 480, 512]
    patch_h = trial.suggest_categorical("patch_h", valid_sizes)
    patch_w = patch_h
    aug_str = trial.suggest_categorical("augmenter_strength", [0.0, 0.5, 1.0])
    minpos  = trial.suggest_categorical("min_pos_frac", [0.0, 0.01, 0.02])
    return int(patch_h), int(patch_w), float(aug_str), float(minpos)


# -------- TerraMind / Prithvi / SatMAE search spaces & helpers --------
def _tm_space_hb(trial: optuna.Trial):
    dec = trial.suggest_categorical("tm_decoder", ["UNetDecoder", "UperNetDecoder"])
    dec_ch = trial.suggest_categorical("tm_decoder_channels", [128, 192, 256, 384])
    head_do = trial.suggest_float("tm_head_dropout", 0.0, 0.2)
    freeze_ep = trial.suggest_categorical("tm_freeze_backbone_epochs", [0, 1, 3, 5])
    lr_backbone = trial.suggest_float("tm_lr_backbone", 1e-6, 3e-5, log=True)
    lr_head_mult = trial.suggest_categorical("tm_lr_head_mult", [5.0, 10.0])
    wd = trial.suggest_float("tm_weight_decay", 1e-6, 5e-4, log=True)
    return dec, dec_ch, head_do, freeze_ep, lr_backbone, lr_head_mult, wd

def _tm_space_bo(trial: optuna.Trial, fixed):
    lr_backbone = trial.suggest_float("tm_lr_backbone", 1e-6, 3e-5, log=True)
    lr_head_mult = trial.suggest_categorical("tm_lr_head_mult", [5.0, 10.0])
    wd = trial.suggest_float("tm_weight_decay", 1e-6, 5e-4, log=True)
    return lr_backbone, lr_head_mult, wd

def _make_two_tier_optimizer(model: nn.Module, lr_bb: float, lr_head_mult: float, wd: float, opt="adamw"):
    # Works for TerraMind AND EncoderDecoderFactory models
    assert hasattr(model, "parameters"), "Model should be nn.Module"
    bb_params = []
    if hasattr(model, "model") and hasattr(model.model, "backbone"):
        bb_params = list(model.model.backbone.parameters())
    elif hasattr(model, "backbone"):
        bb_params = list(model.backbone.parameters())
    else:
        # fallback: try to find a child named 'backbone'
        for n, m in model.named_children():
            if n == "backbone":
                bb_params = list(m.parameters())
                break

    bb_ids = {id(p) for p in bb_params}
    head_params = [p for p in model.parameters() if id(p) not in bb_ids]
    groups = [
        {"params": head_params, "lr": lr_bb * float(lr_head_mult), "weight_decay": wd},
        {"params": bb_params,   "lr": lr_bb,                      "weight_decay": wd},
    ]
    return (torch.optim.AdamW if opt=="adamw" else torch.optim.Adam)(groups)

def _set_backbone_requires_grad(model: nn.Module, requires_grad: bool):
    target = None
    if hasattr(model, "model") and hasattr(model.model, "backbone"):
        target = model.model.backbone
    elif hasattr(model, "backbone"):
        target = model.backbone
    if target is not None:
        for p in target.parameters():
            p.requires_grad = requires_grad

# Band mapping so Prithvi/SatMAE can reuse S2 embeddings for PS/AE inputs
def _bands_for_modality(modality: str, n_channels: int, explicit_bands=None):
    if explicit_bands is not None:
        return explicit_bands
    m = (modality or "S2").upper()
    if m in ("PS", "AE"):                     # 4-band: [B,G,R,NIR]
        return ["BLUE", "GREEN", "RED", "NIR_NARROW"]
    if m == "S2":
        if n_channels == 3: return ["BLUE", "GREEN", "RED"]
        if n_channels == 4: return ["BLUE", "GREEN", "RED", "NIR_NARROW"]
        return None
    return None

def _build_prithvi_or_satmae(
    backbone_name: str,
    num_classes: int,
    decoder_name: str,
    decoder_channels: int | List[int],
    img_size: Tuple[int, int],
    modality: str,
    channels_used: List[bool] | List[int] | None = None,
    bands_override: List[str] | None = None,
    rescale: bool = True,
) -> nn.Module:
    H, W = img_size
    factory = EncoderDecoderFactory()
    # S2 reuse with band subsets if needed
    n_ch = len([c for c in (channels_used or []) if c]) if isinstance(channels_used, list) else None
    bands = _bands_for_modality(modality, n_ch or 4, bands_override)
    # neck taps for 12-layer ViT-like backbones (robust default)
    necks = [
        {"name": "ReshapeTokensToImage", "remove_cls_token": False},
        {"name": "SelectIndices", "indices": [2, 5, 8, 11]},
        {"name": "LearnedInterpolateToPyramidal"},
    ]
    dec_ch = decoder_channels if isinstance(decoder_channels, list) else decoder_channels
    model = factory.build_model(
        task="segmentation",
        backbone=backbone_name,
        decoder=decoder_name,
        num_classes=num_classes,
        necks=necks,
        rescale=rescale,
        # backbone args
        backbone_pretrained=True,
        backbone_modalities=["S2L2A"],               # always reuse S2 embeddings
        backbone_merge_method="mean",
        backbone_bands={"S2L2A": bands} if bands is not None else None,
        # decoder args
        decoder_channels=dec_ch,
    )
    return model


# -----------------------
# Build / compile (optimizer + scheduler + loss)
# -----------------------
def _compile_with_optimizer(
    model: nn.Module,
    opt_name: str, lr: float, wd: Optional[float],
    scheduler_name: str, steps_per_epoch: int, max_epochs: int,
    conf
) -> Tuple[nn.Module, torch.optim.Optimizer, Any, Optional[Any], bool]:
    """
    Returns: model, optimizer, criterion, scheduler, scheduler_steps_per_batch
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Try project optimizer factory
    opt_obj: Optional[torch.optim.Optimizer] = None
    try:
        opt_obj = get_optimizer(
            _default(opt_name, getattr(conf, "optimizer_fn", "adam")),
            _default(getattr(conf, "num_epochs", max_epochs), max_epochs),
            _default(getattr(conf, "num_training_steps", steps_per_epoch), steps_per_epoch),
            model
        )
        if hasattr(opt_obj, "param_groups"):
            for g in opt_obj.param_groups:
                g["lr"] = float(lr)
                if opt_name == "adamw" and wd is not None:
                    g["weight_decay"] = float(wd)
    except Exception:
        opt_obj = None

    # Fallback optimizer construction
    if opt_obj is None or isinstance(opt_obj, str):
        params = model.parameters()
        if (opt_name or "adam").lower() == "adamw":
            opt_obj = torch.optim.AdamW(params, lr=float(lr), weight_decay=float(_default(wd, 1e-6)))
        else:
            opt_obj = torch.optim.Adam(params, lr=float(lr))

    # Loss: BCE + (1 - Dice) — robust; now auto-activate logits if needed
    def criterion(y_true, y_pred, w: Optional[torch.Tensor] = None):
        yt = _nan_to_num_torch(y_true.float(), 0.0).clamp(0.0, 1.0)
        yp = y_pred
        # logits → probs if values look out of [0,1]
        with torch.no_grad():
            mn = float(torch.min(yp).detach().cpu())
            mx = float(torch.max(yp).detach().cpu())
        if (mn < -1e-3) or (mx > 1.0 + 1e-3):
            yp = torch.sigmoid(yp)
        yp = _nan_to_num_torch(yp.float(), 0.5).clamp(1e-6, 1.0 - 1e-6)

        if w is None:
            bce_term = F.binary_cross_entropy(yp, yt)
        else:
            w_sanit = _sanitize_weights(w)
            loss_elt = F.binary_cross_entropy(yp, yt, reduction="none")
            while w_sanit.ndim < loss_elt.ndim:
                w_sanit = w_sanit.unsqueeze(-1)
            bce_term = (loss_elt * w_sanit).mean()
        return bce_term + (1.0 - _safe_dice(yt, yp))

    # Scheduler setup
    scheduler = None
    step_per_batch = False
    scheduler_name = (scheduler_name or "none").lower()
    if scheduler_name == "onecycle":
        scheduler = OneCycleLR(opt_obj, max_lr=float(lr),
                               steps_per_epoch=int(steps_per_epoch), epochs=int(max_epochs))
        step_per_batch = True
    elif scheduler_name == "cosine":
        scheduler = CosineAnnealingLR(opt_obj, T_max=int(max_epochs), eta_min=lr * 0.01)

    return model, opt_obj, criterion, scheduler, step_per_batch


# -----------------------
# Phase runner (HB / BO)
# -----------------------
def _run_phase(conf, model_key: str, phase: str, project_dir: str,
               tune_batch: int, max_epochs: int, steps: int, val_steps: int,
               executions_per_trial: int, overwrite: bool,
               hb_data_hp: Dict[str, Any]) -> Tuple[Dict[str, Any], optuna.Study]:

    frames = get_all_frames(conf)

    # cache dataloaders for (patch_h, patch_w, aug, minpos) combos (batch fixed)
    dl_cache: Dict[Tuple[int, int, float, float], Tuple[Iterable, Iterable]] = {}

    def _get_loaders_for(ph: int, pw: int, aug: float, minpos: float) -> Tuple[Iterable, Iterable]:
        key = (int(ph), int(pw), float(aug), float(minpos))
        if key in dl_cache:
            return dl_cache[key]

        # Temporarily override config to build datasets with these knobs
        old_bs = getattr(conf, "train_batch_size", None)
        old_patch = getattr(conf, "patch_size", None)
        old_aug = getattr(conf, "augmenter_strength", None)
        old_minpos = getattr(conf, "min_pos_frac", None)

        conf.train_batch_size = int(tune_batch)
        conf.patch_size = (int(ph), int(pw))
        conf.augmenter_strength = float(aug)
        conf.min_pos_frac = float(minpos)

        train_iter, val_iter, _ = create_train_val_datasets(frames)

        # Restore originals
        if old_bs is not None:       conf.train_batch_size = old_bs
        if old_patch is not None:    conf.patch_size = old_patch
        if old_aug is not None:      conf.augmenter_strength = old_aug
        if old_minpos is not None:   conf.min_pos_frac = old_minpos

        dl_cache[key] = (train_iter, val_iter)
        return train_iter, val_iter

    # Study
    direction = "maximize"
    study_name = f"{model_key}_tuning_{phase}"
    pruner = HyperbandPruner() if phase == "HB" else None
    sampler = TPESampler(seed=_default(getattr(conf, "seed", None), 42))
    _ensure_dir(project_dir)
    study = optuna.create_study(direction=direction, sampler=sampler, pruner=pruner,
                                study_name=study_name, storage=None)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = False
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    _phase_banner(model_key, phase, max_epochs, steps, val_steps, tune_batch)

    # Track best within this phase for nicer printing (does NOT affect Optuna)
    best_print = {"score": None, "cfg": None}

    def _single_execution(trial, train_iterable, val_iterable,
                          model: nn.Module, optimizer: torch.optim.Optimizer,
                          criterion, scheduler, step_per_batch: bool, freeze_ep: int = 0) -> float:
        best_val_dice = -float("inf")
        patience = 5
        no_improve = 0
        was_frozen = None

        for epoch in range(1, max_epochs + 1):
            # Optional freeze schedule (for foundation backbones)
            if freeze_ep > 0:
                if epoch <= freeze_ep and was_frozen is not True:
                    _set_backbone_requires_grad(model, False); was_frozen = True
                elif epoch == freeze_ep + 1 and was_frozen:
                    _set_backbone_requires_grad(model, True); was_frozen = False

            model.train()
            train_it = iter(train_iterable)
            for _ in range(steps):
                batch = next(train_it, None)
                if batch is None:
                    train_it = iter(train_iterable)
                    batch = next(train_it)

                if isinstance(batch, (list, tuple)) and len(batch) == 3:
                    x, y, w = batch
                else:
                    x, y = batch
                    w = None

                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                x, y = _sanitize_pair_xy(x, y)

                optimizer.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=use_amp, dtype=torch.float16):
                    out = model(x)
                    # TerraTorch ModelOutput often exposes .prediction or .logits
                    if hasattr(out, "prediction"):
                        y_pred = out.prediction
                    elif hasattr(out, "logits"):
                        y_pred = out.logits
                    else:
                        y_pred = out
                    # normalize to NCHW
                    if y_pred.ndim == 3:  # [N,H,W] -> [N,1,H,W]
                        y_pred = y_pred.unsqueeze(1)
                    loss = criterion(y, y_pred, w)

                if torch.isnan(loss) or torch.isinf(loss):
                    raise optuna.TrialPruned()

                scaler.scale(loss).backward()
                if use_amp:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()

                if scheduler is not None and step_per_batch:
                    scheduler.step()

            # Validation
            model.eval()
            val_it = iter(val_iterable)
            val_dice_accum = 0.0
            with torch.no_grad():
                for _ in range(val_steps):
                    vb = next(val_it, None)
                    if vb is None:
                        break
                    if isinstance(vb, (list, tuple)) and len(vb) == 3:
                        xv, yv, _wv = vb
                    else:
                        xv, yv = vb
                    xv = xv.to(device, non_blocking=True)
                    yv = yv.to(device, non_blocking=True)
                    xv, yv = _sanitize_pair_xy(xv, yv)
                    outv = model(xv)
                    if hasattr(outv, "prediction"):
                        yp = outv.prediction
                    elif hasattr(outv, "logits"):
                        yp = torch.sigmoid(outv.logits)
                    else:
                        yp = outv
                    if yp.ndim == 3: yp = yp.unsqueeze(1)
                    yp = yp.float().clamp(0.0, 1.0)
                    val_dice_accum += float(_safe_dice(yv, yp).cpu().item())
            val_dice = val_dice_accum / max(1, val_steps)

            if scheduler is not None and not step_per_batch:
                scheduler.step()

            trial.report(val_dice, epoch-1)
            if trial.should_prune():
                raise optuna.TrialPruned()

            if val_dice > best_val_dice + 1e-6:
                best_val_dice = val_dice
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    _epoch_line(epoch, best_val_dice, val_dice, "early-stop")
                    break

            _epoch_line(epoch, best_val_dice, val_dice)

        return float(best_val_dice)

    def objective(trial: optuna.Trial) -> float:
        trial_t0 = time.time()
        try:
            # ----- data knobs -----
            if model_key == "unet" and phase == "HB":
                patch_h, patch_w, aug_str, minpos = _data_space_hb(trial)
            else:
                patch_h = int(hb_data_hp["patch_h"])
                patch_w = int(hb_data_hp["patch_w"])
                aug_str = float(hb_data_hp["augmenter_strength"])
                minpos = float(hb_data_hp["min_pos_frac"])

            train_iterable, val_iterable = _get_loaders_for(patch_h, patch_w, aug_str, minpos)

            # ----- Foundation models: TerraMind / Prithvi / SatMAE -----
            if model_key in ("tm", "prithvi", "satmae"):
                if phase == "HB":
                    dec, dec_ch, head_do, freeze_ep, lr_bb, lr_head_mult, wd = _tm_space_hb(trial)
                    opt_name = trial.suggest_categorical("optimizer", ["adamw", "adam"])
                    scheduler_name = trial.suggest_categorical("scheduler", ["none", "cosine", "onecycle"])
                else:
                    fixed = hb_data_hp.get("fixed", {})
                    dec       = fixed.get("tm_decoder", "UperNetDecoder")
                    dec_ch    = fixed.get("tm_decoder_channels", 256)
                    head_do   = fixed.get("tm_head_dropout", 0.0)
                    freeze_ep = fixed.get("tm_freeze_backbone_epochs", 1)
                    lr_bb, lr_head_mult, wd = _tm_space_bo(trial, fixed)
                    opt_name = fixed.get("optimizer", "adamw")
                    scheduler_name = fixed.get("scheduler", "none")

                # Build model by family
                if model_key == "tm":
                    model = TerraMind(
                        in_channels=len(getattr(conf, "channels_used", getattr(conf, "channel_list", [])) or [0,1,2,3]),
                        num_classes=int(getattr(conf, "num_classes", 1)),
                        modality=getattr(conf, "modality", "S2"),
                        tm_size=str(getattr(conf, "tm_backbone", "terramind_v1_base")).split("_")[-1],   # base|large...
                        merge_method="mean",
                        pretrained=True,
                        ckpt_path=getattr(conf, "tm_backbone_ckpt_path", None),
                        indices_override=None,
                        bands_override=getattr(conf, "tm_bands", None),
                        decoder_channels=int(dec_ch) if not isinstance(dec_ch, (list, tuple)) else int(dec_ch[0]),
                        rescale=True,
                    )
                    # TerraMind wrapper exposes .model under the hood
                    # we’ll add a light dropout to the head if available
                    if hasattr(model, "model") and hasattr(model.model, "decode_head") and head_do > 0:
                        if hasattr(model.model.decode_head, "dropout"):
                            try:
                                model.model.decode_head.dropout.p = float(head_do)
                            except Exception:
                                pass
                else:
                    # Prithvi / SatMAE via EncoderDecoderFactory
                    if model_key == "prithvi":
                        fm_backbone = getattr(conf, "prithvi_backbone", "prithvi_v2_eo_300")
                    else:  # satmae
                        fm_backbone = getattr(conf, "satmae_backbone", "satmae_eo_vit_b")
                    try:
                        model = _build_prithvi_or_satmae(
                            backbone_name=fm_backbone,
                            num_classes=int(getattr(conf, "num_classes", 1)),
                            decoder_name=dec,
                            decoder_channels=([dec_ch, dec_ch, dec_ch, dec_ch] if not isinstance(dec_ch, (list, tuple)) else list(dec_ch)),
                            img_size=(patch_h, patch_w),
                            modality=getattr(conf, "modality", "S2"),
                            channels_used=getattr(conf, "channels_used", getattr(conf, "channel_list", [])),
                            bands_override=None,
                            rescale=True,
                        ).to(device)
                    except Exception as e:
                        # If the backbone is not available in this install, prune the trial gracefully
                        print(f"[{model_key}] backbone build failed ({e}); pruning trial.")
                        raise optuna.TrialPruned()

                model = model.to(device)

                optimizer = _make_two_tier_optimizer(model, lr_bb, lr_head_mult, wd, opt=opt_name)

                scheduler, step_per_batch = (None, False)
                if scheduler_name == "onecycle":
                    scheduler = torch.optim.lr_scheduler.OneCycleLR(
                        optimizer, max_lr=[g["lr"] for g in optimizer.param_groups],
                        steps_per_epoch=steps, epochs=max_epochs
                    )
                    step_per_batch = True
                elif scheduler_name == "cosine":
                    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                        optimizer, T_max=max_epochs,
                        eta_min=min(g["lr"] for g in optimizer.param_groups) * 0.01
                    )

                # loss: BCE + (1 - Dice) (criterion already handles logits)
                def criterion(y_true, y_pred, w=None):
                    return _compile_with_optimizer(model, opt_name, 1.0, 0.0, "none", 1, 1, conf)[2](y_true, y_pred, w)

                # Pretty header once per trial
                hp_line = dict(
                    optimizer=opt_name, scheduler=scheduler_name,
                    fm_backbone=(getattr(conf, "tm_backbone", None) if model_key=="tm"
                                 else getattr(conf, "prithvi_backbone", None) if model_key=="prithvi"
                                 else getattr(conf, "satmae_backbone", None)),
                    tm_decoder=dec, tm_decoder_channels=dec_ch, tm_head_dropout=head_do,
                    tm_lr_backbone=lr_bb, tm_lr_head_mult=lr_head_mult, tm_weight_decay=wd,
                    tm_freeze_backbone_epochs=freeze_ep
                )
                _trial_banner(trial.number, model_key, hp_line,
                              dict(patch_h=patch_h, patch_w=patch_w,
                                   augmenter_strength=aug_str, min_pos_frac=minpos))

                cfg_current = dict(hp_line)
                cfg_current.update(dict(
                    patch_h=patch_h, patch_w=patch_w,
                    augmenter_strength=aug_str, min_pos_frac=minpos,
                    model=model_key, phase=phase
                ))

                # Multiple executions per trial (average)
                exec_vals: List[float] = []
                for exec_idx in range(max(1, int(executions_per_trial))):
                    if exec_idx > 0:
                        # rebuild fresh model/opt/scheduler for repeats
                        if model_key == "tm":
                            model = TerraMind(
                                in_channels=len(getattr(conf, "channels_used", getattr(conf, "channel_list", [])) or [0,1,2,3]),
                                num_classes=int(getattr(conf, "num_classes", 1)),
                                modality=getattr(conf, "modality", "S2"),
                                tm_size=str(getattr(conf, "tm_backbone", "terramind_v1_base")).split("_")[-1],
                                merge_method="mean",
                                pretrained=True,
                                ckpt_path=getattr(conf, "tm_backbone_ckpt_path", None),
                                indices_override=None,
                                bands_override=getattr(conf, "tm_bands", None),
                                decoder_channels=int(dec_ch) if not isinstance(dec_ch, (list, tuple)) else int(dec_ch[0]),
                                rescale=True,
                            ).to(device)
                        else:
                            fm_backbone = (getattr(conf, "prithvi_backbone", "prithvi_v2_eo_300")
                                           if model_key=="prithvi" else
                                           getattr(conf, "satmae_backbone", "satmae_eo_vit_b"))
                            model = _build_prithvi_or_satmae(
                                backbone_name=fm_backbone,
                                num_classes=int(getattr(conf, "num_classes", 1)),
                                decoder_name=dec,
                                decoder_channels=([dec_ch, dec_ch, dec_ch, dec_ch] if not isinstance(dec_ch, (list, tuple)) else list(dec_ch)),
                                img_size=(patch_h, patch_w),
                                modality=getattr(conf, "modality", "S2"),
                                channels_used=getattr(conf, "channels_used", getattr(conf, "channel_list", [])),
                                bands_override=None,
                                rescale=True,
                            ).to(device)

                        optimizer = _make_two_tier_optimizer(model, lr_bb, lr_head_mult, wd, opt=opt_name)
                        scheduler, step_per_batch = (None, False)
                        if scheduler_name == "onecycle":
                            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                                optimizer, max_lr=[g["lr"] for g in optimizer.param_groups],
                                steps_per_epoch=steps, epochs=max_epochs
                            )
                            step_per_batch = True
                        elif scheduler_name == "cosine":
                            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                                optimizer, T_max=max_epochs,
                                eta_min=min(g["lr"] for g in optimizer.param_groups) * 0.01
                            )

                    val = _single_execution(trial, train_iterable, val_iterable,
                                            model, optimizer, criterion, scheduler, step_per_batch,
                                            freeze_ep=freeze_ep)
                    exec_vals.append(val)

                result = float(np.mean(exec_vals))
                dur = time.time() - trial_t0
                print(f"\n  --> trial {trial.number} done: val_dice={result:.5f}, time={time.strftime('%H:%M:%S', time.gmtime(dur))}")

                prev_score = best_print["score"]
                prev_cfg = best_print["cfg"]
                _print_sxs_diff(prev_cfg, cfg_current, prev_score, result)

                if (prev_score is None) or (result > prev_score + 1e-12):
                    if prev_score is None:
                        print(_col("   ✅ Baseline established (first completed trial).", _C.GREEN))
                    else:
                        delta = result - prev_score
                        print(_col(f"   ✅ NEW BEST! Improved by +{delta:.5f}", _C.GREEN))
                    best_print["score"] = result
                    best_print["cfg"] = cfg_current
                else:
                    gap = prev_score - result
                    print(_col(f"\n   ==> Off current best by {gap:.5f}", _C.CYAN))

                return result

            # ----- model knobs (original branches) -----
            trial_hps: Dict[str, Any] = {}
            if model_key == "swin":
                if phase == "HB":
                    C, ss_half, attn_d, proj_d, mlp_d, drop_p = _swin_space_hb(trial)
                    ss_size = 0 if ss_half == 0 else hb_data_hp["fixed"]["window_size"] // 2
                    ch = len(getattr(conf, "channels_used", getattr(conf, "channel_list", []))) or len(conf.channel_list)
                    model = SwinUNet(h=patch_h, w=patch_w, ch=ch, c=C,
                                     patch_size=hb_data_hp["fixed"]["patch_size"],
                                     window_size=hb_data_hp["fixed"]["window_size"],
                                     ss_size=ss_size, attn_drop=attn_d, proj_drop=proj_d,
                                     mlp_drop=mlp_d, drop_path=drop_p)
                    trial_hps.update(dict(C=C, ss_size=ss_size, attn_drop=attn_d,
                                          proj_drop=proj_d, mlp_drop=mlp_d, drop_path=drop_p))
                else:
                    fixed = hb_data_hp["fixed"]
                    ch = len(getattr(conf, "channels_used", getattr(conf, "channel_list", []))) or len(conf.channel_list)
                    model = SwinUNet(h=patch_h, w=patch_w, ch=ch,
                                     c=fixed.get("C", 64),
                                     patch_size=fixed.get("patch_size", 16),
                                     window_size=fixed.get("window_size", 4),
                                     ss_size=fixed.get("ss_size", 2),
                                     attn_drop=fixed.get("attn_drop", 0.0),
                                     proj_drop=fixed.get("proj_drop", 0.0),
                                     mlp_drop=fixed.get("mlp_drop", 0.0),
                                     drop_path=fixed.get("drop_path", 0.1))
                    trial_hps.update(dict(C=fixed.get("C", 64), ss_size=fixed.get("ss_size", 2),
                                          attn_drop=fixed.get("attn_drop", 0.0), proj_drop=fixed.get("proj_drop", 0.0),
                                          mlp_drop=fixed.get("mlp_drop", 0.0), drop_path=fixed.get("drop_path", 0.1)))
            else:  # unet
                if phase == "HB":
                    dilation, layer_cnt, l2w, drp = _unet_space_hb(trial)
                else:
                    fixed = hb_data_hp["fixed"]
                    dilation = fixed.get("dilation_rate", 1)
                    layer_cnt = fixed.get("layer_count", 64)
                    l2w = float(fixed.get("l2_weight", 1e-4))
                    drp = float(fixed.get("dropout", 0.0))
                input_shape = [tune_batch, patch_h, patch_w, len(getattr(conf, "channel_list", []))]
                model = UNet(input_shape, 1, dilation_rate=dilation, layer_count=layer_cnt,
                             l2_weight=l2w, dropout=drp)
                trial_hps.update({"dilation_rate": dilation, "layer_count": layer_cnt,
                                  "l2_weight": l2w, "dropout": drp})

            # ----- optimizer & scheduler -----
            if phase == "HB":
                opt_name, lr, wd = _optimizer_space_hb(trial)
                scheduler_name = _schedule_space_hb(trial)
            else:
                fixed_opt = hb_data_hp["fixed"]["optimizer"]
                opt_name, lr, wd = _optimizer_space_bo(trial, fixed_opt)
                scheduler_name = hb_data_hp["fixed"]["scheduler"]

            # Build/compile
            model, optimizer, criterion, scheduler, step_per_batch = _compile_with_optimizer(
                model, opt_name, lr, wd, scheduler_name, steps, max_epochs, conf
            )

            # Pretty header once per trial
            hp_line = dict(optimizer=opt_name, learning_rate=lr, scheduler=scheduler_name)
            if wd is not None: hp_line["weight_decay"] = wd
            hp_line.update(trial_hps)
            _trial_banner(trial.number, model_key, hp_line,
                          dict(patch_h=patch_h, patch_w=patch_w,
                               augmenter_strength=aug_str, min_pos_frac=minpos))

            cfg_current = dict(hp_line)
            cfg_current.update(dict(
                patch_h=patch_h, patch_w=patch_w,
                augmenter_strength=aug_str, min_pos_frac=minpos,
                model=model_key, phase=phase
            ))

            # ----- multiple executions per trial (average objective) -----
            exec_vals: List[float] = []
            for exec_idx in range(max(1, int(executions_per_trial))):
                if exec_idx > 0:
                    model, optimizer, criterion, scheduler, step_per_batch = _compile_with_optimizer(
                        UNet(input_shape, 1, dilation_rate=trial_hps.get("dilation_rate", 1),
                             layer_count=trial_hps.get("layer_count", 64),
                             l2_weight=trial_hps.get("l2_weight", 1e-4),
                             dropout=trial_hps.get("dropout", 0.0)).to(device)
                        if model_key == "unet"
                        else SwinUNet(h=patch_h, w=patch_w, ch=len(getattr(conf, "channel_list", [])),
                                      c=trial_hps.get("C", 64),
                                      patch_size=hb_data_hp["fixed"].get("patch_size", 16),
                                      window_size=hb_data_hp["fixed"].get("window_size", 4),
                                      ss_size=trial_hps.get("ss_size", 2),
                                      attn_drop=trial_hps.get("attn_drop", 0.0),
                                      proj_drop=trial_hps.get("proj_drop", 0.0),
                                      mlp_drop=trial_hps.get("mlp_drop", 0.0),
                                      drop_path=trial_hps.get("drop_path", 0.1)).to(device),
                        opt_name, lr, wd, scheduler_name, steps, max_epochs, conf
                    )

                val = _single_execution(trial, train_iterable, val_iterable,
                                        model, optimizer, criterion, scheduler, step_per_batch)
                exec_vals.append(val)

            result = float(np.mean(exec_vals))
            dur = time.time() - trial_t0
            print(f"\n  --> trial {trial.number} done: val_dice={result:.5f}, time={time.strftime('%H:%M:%S', time.gmtime(dur))}")

            prev_score = best_print["score"]
            prev_cfg = best_print["cfg"]
            _print_sxs_diff(prev_cfg, cfg_current, prev_score, result)

            if (prev_score is None) or (result > prev_score + 1e-12):
                if prev_score is None:
                    print(_col("   ✅ Baseline established (first completed trial).", _C.GREEN))
                else:
                    delta = result - prev_score
                    print(_col(f"   ✅ NEW BEST! Improved by +{delta:.5f}", _C.GREEN))
                best_print["score"] = result
                best_print["cfg"] = cfg_current
            else:
                gap = prev_score - result
                print(_col(f"\n   ==> Off current best by {gap:.5f}", _C.CYAN))

            return result

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                try: torch.cuda.empty_cache()
                except Exception: pass
                raise optuna.TrialPruned()
            raise

    n_trials = _default(getattr(conf, "tune_hb_max_trials", None), 30) if phase == "HB" \
               else _default(getattr(conf, "tune_max_trials", None), 30)

    study.optimize(objective, n_trials=int(n_trials), gc_after_trial=True)

    # Artifacts (compact)
    trials_sorted = [t for t in study.trials if t.value is not None]
    trials_sorted.sort(key=lambda t: t.value, reverse=True)

    # Persist CSV of trials
    tag = f"{model_key}_tuning"
    _ensure_dir(project_dir)
    csv_path = os.path.join(project_dir, f"{tag}_{phase}_all_trials.csv")
    hp_names = sorted({k for t in trials_sorted for k in t.params.keys()})
    lines = [",".join(["trial_id", "value"] + hp_names)]
    for t in trials_sorted:
        row = [str(t.number), "" if t.value is None else str(t.value)]
        for name in hp_names:
            row.append(str(t.params.get(name, "")))
        lines.append(",".join(row))
    with open(csv_path, "w") as f:
        f.write("\n".join(lines))

    # Short MD summary
    best_top = trials_sorted[: min(5, len(trials_sorted))]
    md_lines = [
        f"# {tag} — {phase} summary",
        f"- date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- objective: val__safe_dice (max)",
        f"- max_epochs: {max_epochs}",
        f"- steps_per_epoch: {steps}",
        f"- val_steps: {val_steps}",
        f"- tune_batch_size: {tune_batch}",
        "",
        "## top trials"
    ]
    for i, tr in enumerate(best_top, 1):
        hp_str = ", ".join([f"{k}={v}" for k, v in tr.params.items()])
        md_lines += [f"### {i}. trial {tr.number}", f"- score: {tr.value}", f"- hparams: {hp_str}", ""]
    md_lines += ["## files",
                 f"- all trials csv: {csv_path}",
                 f"- tensorboard: {project_dir}"]
    with open(os.path.join(project_dir, f"{tag}_{phase}_summary.md"), "w") as f:
        f.write("\n".join(md_lines))

    # Save best params JSON
    best_params = dict(study.best_trial.params)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    best_json_path = os.path.join(project_dir, f"{stamp}_{tag}_{phase}_best_hparams.json")
    with open(best_json_path, "w") as f:
        json.dump(best_params, f, indent=2)
    print(f"\n[{tag} {phase}] best saved to: {best_json_path}")

    return best_params, study


# -----------------------
# Chained tuner (HB --> BO)
# -----------------------
def _tune_chained(conf, model_type: str = "unet",
                  executions_per_trial: int = 1, overwrite: bool = False) -> Dict[str, Any]:
    """
    Two-stage tuning: HyperBand (broad) then Bayesian Optimization (refinement).
    Batch stays fixed; BCE+(1-Dice) loss; clipnorm=1.0; AMP off for parity/stability.
    """
    global config
    config = conf

    print("\nStarting chained tuning (HyperBand + Bayesian).")
    _seed(_default(getattr(config, "seed", None), 42))

    # Budgets
    hb_epochs = _default(getattr(config, "tune_num_epochs", None), 20)
    bo_epochs = _default(getattr(config, "tune_num_epochs_bo", None), hb_epochs)
    steps     = _default(getattr(config, "tune_steps_per_epoch", None),
                         min(100, _default(getattr(config, "num_training_steps", 100), 100)))
    val_steps = _default(getattr(config, "tune_validation_steps", None),
                         min(50, _default(getattr(config, "num_validation_images", 50), 50)))
    tune_batch = _default(getattr(config, "tune_batch_size", None),
                          min(8, _default(getattr(config, "train_batch_size", 8), 8)))

    logs_dir = _default(getattr(config, "logs_dir", "./logs"), "./logs")
    key = model_type.lower().strip()
    if key not in ("unet", "swin", "tm", "prithvi", "satmae"):
        raise ValueError(f"Unknown model_type '{model_type}'. Choose from 'unet','swin','tm','prithvi','satmae'.")

    project_dir = os.path.join(logs_dir, f"{key}_tuning")
    _ensure_dir(project_dir)

    # HB data defaults
    hb_aug    = _default(getattr(config, "augmenter_strength", None), 1.0)
    hb_minpos = _default(getattr(config, "min_pos_frac", None), 0.0)

    if key == "swin":
        fixed_ps = _default(getattr(config, "swin_patch_size", None), 8)
        fixed_ws = _default(getattr(config, "swin_window", None), 4)
        base_H, base_W = 384, 384
        hb_patch_h, hb_patch_w = _snap_hw_for_swin(base_H, base_W, fixed_ps, fixed_ws, down_levels=3)
        hb_data_hp = dict(
            patch_h=hb_patch_h,
            patch_w=hb_patch_w,
            augmenter_strength=hb_aug,
            min_pos_frac=hb_minpos,
            fixed=dict(patch_size=fixed_ps, window_size=fixed_ws),
        )
    elif key in ("tm", "prithvi", "satmae"):
        hb_patch_h, hb_patch_w = 384, 384  # multiples of 16 are fine
        hb_data_hp = dict(
            patch_h=hb_patch_h,
            patch_w=hb_patch_w,
            augmenter_strength=hb_aug,
            min_pos_frac=hb_minpos,
            fixed={},  # will be filled from HB winners for BO
        )
    else:
        hb_patch_h, hb_patch_w = 384, 384
        hb_data_hp = dict(
            patch_h=hb_patch_h,
            patch_w=hb_patch_w,
            augmenter_strength=hb_aug,
            min_pos_frac=hb_minpos,
        )

    # Phase 1: HB
    hb_best, _ = _run_phase(
        config, key, "HB", project_dir,
        tune_batch, hb_epochs, steps, val_steps,
        executions_per_trial, overwrite, hb_data_hp
    )

    # Build BO fixed set from HB winners
    if key == "unet":
        fixed = {
            "optimizer": hb_best.get("optimizer", _default(getattr(config, "optimizer_fn", "adam"), "adam")),
            "scheduler": hb_best.get("scheduler", "none"),
            "dilation_rate": hb_best.get("dilation_rate", _default(getattr(config, "dilation_rate", 1), 1)),
            "layer_count":   hb_best.get("layer_count", 64),
            "l2_weight":     float(hb_best.get("l2_weight", 1e-4)),
            "dropout":       float(hb_best.get("dropout", 0.0)),
        }
        bo_patch_h = int(hb_best.get("patch_h", hb_patch_h))
        bo_patch_w = int(hb_best.get("patch_w", hb_patch_w))
    elif key == "swin":
        fixed = {
            "optimizer": hb_best.get("optimizer", _default(getattr(config, "optimizer_fn", "adam"), "adam")),
            "scheduler": hb_best.get("scheduler", "none"),
            "C":             hb_best.get("C", _default(getattr(config, "swin_base_C", 64), 64)),
            "patch_size":    hb_data_hp["fixed"]["patch_size"],
            "window_size":   hb_data_hp["fixed"]["window_size"],
            "ss_size":       (hb_data_hp["fixed"]["window_size"] // 2) if int(hb_best.get("use_shift", 1)) == 1 else 0,
            "attn_drop":     float(hb_best.get("attn_drop", 0.0)),
            "proj_drop":     float(hb_best.get("proj_drop", 0.0)),
            "mlp_drop":      float(hb_best.get("mlp_drop", 0.0)),
            "drop_path":     float(hb_best.get("drop_path", 0.1)),
        }
        bo_patch_h, bo_patch_w = int(hb_data_hp["patch_h"]), int(hb_data_hp["patch_w"])
    else:  # tm / prithvi / satmae share the same fixed set
        fixed = {
            "optimizer": hb_best.get("optimizer", _default(getattr(config, "optimizer_fn", "adamw"), "adamw")),
            "scheduler": hb_best.get("scheduler", "none"),
            "tm_decoder":              hb_best.get("tm_decoder", "UperNetDecoder"),
            "tm_decoder_channels":     int(hb_best.get("tm_decoder_channels", 256)),
            "tm_head_dropout":         float(hb_best.get("tm_head_dropout", 0.0)),
            "tm_freeze_backbone_epochs": int(hb_best.get("tm_freeze_backbone_epochs", 1)),
        }
        bo_patch_h, bo_patch_w = int(hb_data_hp["patch_h"]), int(hb_data_hp["patch_w"])

    # Phase 2: BO
    bo_data_hp = dict(
        patch_h=bo_patch_h,
        patch_w=bo_patch_w,
        augmenter_strength=hb_aug,
        min_pos_frac=hb_minpos,
        fixed=fixed,
    )
    bo_best, _ = _run_phase(
        config, key, "BO", project_dir,
        tune_batch, bo_epochs, steps, val_steps,
        executions_per_trial, overwrite, bo_data_hp
    )

    # Merge final settings
    final = dict(fixed)
    if "learning_rate" in bo_best: final["learning_rate"] = float(bo_best["learning_rate"])
    if "weight_decay" in bo_best:  final["weight_decay"]  = float(bo_best["weight_decay"])
    final["patch_h"] = bo_patch_h
    final["patch_w"] = bo_patch_w
    final["augmenter_strength"] = float(hb_aug)
    final["min_pos_frac"] = float(hb_minpos)

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    final_path = os.path.join(project_dir, f"{stamp}_{key}_tuning_FINAL_best_hparams.json")
    with open(final_path, "w") as f:
        json.dump(final, f, indent=2)
    print("\n" + "="*60)
    print(f"[{key} tuning] FINAL merged best saved to: {final_path}")
    print(json.dumps(final, indent=2))
    print("="*60 + "\n")
    return final


# -----------------------
# Public entry points
# -----------------------
def tune_UNet(conf) -> Dict[str, Any]:
    return _tune_chained(conf, model_type="unet")

def tune_SwinUNetPP(conf) -> Dict[str, Any]:
    return _tune_chained(conf, model_type="swin")

def tune_TerraMind(conf) -> Dict[str, Any]:
    return _tune_chained(conf, model_type="tm")

# NEW
def tune_Prithvi(conf) -> Dict[str, Any]:
    # set a sensible default backbone if user hasn’t provided one
    if not hasattr(conf, "prithvi_backbone"):
        conf.prithvi_backbone = "prithvi_v2_eo_300"
    return _tune_chained(conf, model_type="prithvi")

# NEW
def tune_SatMAE(conf) -> Dict[str, Any]:
    # default; if missing in your install, trials will prune gracefully
    if not hasattr(conf, "satmae_backbone"):
        conf.satmae_backbone = "satmae_eo_vit_b"
    return _tune_chained(conf, model_type="satmae")


def apply_best_to_config(conf, best: Dict[str, Any], model_type: str) -> None:
    if "optimizer" in best:      conf.optimizer_fn = best["optimizer"]
    if "learning_rate" in best:  conf.learning_rate = best["learning_rate"]
    if "weight_decay" in best:   conf.weight_decay  = best["weight_decay"]
    if "scheduler" in best:      conf.scheduler     = best["scheduler"]

    conf.augmenter_strength = best.get("augmenter_strength", getattr(conf, "augmenter_strength", 1.0))
    conf.min_pos_frac       = best.get("min_pos_frac", getattr(conf, "min_pos_frac", 0.0))
    conf.patch_size         = [int(best.get("patch_h", conf.patch_size[0])),
                               int(best.get("patch_w", conf.patch_size[1]))]

    mt = model_type.lower()
    if mt == "unet":
        conf.dilation_rate = best.get("dilation_rate", getattr(conf, "dilation_rate", 1))
        conf.layer_count   = best.get("layer_count", getattr(conf, "layer_count", 64))
        conf.l2_weight     = best.get("l2_weight", getattr(conf, "l2_weight", 1e-4))
        conf.dropout       = best.get("dropout", getattr(conf, "dropout", 0.0))
    elif mt == "swin":
        conf.swin_base_C     = best.get("C", getattr(conf, "swin_base_C", 64))
        conf.swin_patch_size = best.get("patch_size", getattr(conf, "swin_patch_size", 16))
        conf.swin_window     = best.get("window_size", getattr(conf, "swin_window", 4))
        conf.swin_ss_size    = best.get("ss_size", getattr(conf, "swin_ss_size", 2))
        conf.swin_attn_drop  = best.get("attn_drop", getattr(conf, "swin_attn_drop", 0.0))
        conf.swin_proj_drop  = best.get("proj_drop", getattr(conf, "swin_proj_drop", 0.0))
        conf.swin_mlp_drop   = best.get("mlp_drop", getattr(conf, "swin_mlp_drop", 0.0))
        conf.swin_drop_path  = best.get("drop_path", getattr(conf, "swin_drop_path", 0.1))
    elif mt in ("tm", "prithvi", "satmae"):
        # Common decoder/opt knobs for the foundation models
        conf.tm_decoder                 = best.get("tm_decoder", getattr(conf, "tm_decoder", "UperNetDecoder"))
        conf.tm_decoder_channels        = best.get("tm_decoder_channels", getattr(conf, "tm_decoder_channels", 256))
        conf.tm_head_dropout            = best.get("tm_head_dropout", getattr(conf, "tm_head_dropout", 0.0))
        conf.tm_freeze_backbone_epochs  = best.get("tm_freeze_backbone_epochs", getattr(conf, "tm_freeze_backbone_epochs", 1))
        conf.tm_lr_backbone             = best.get("tm_lr_backbone", getattr(conf, "tm_lr_backbone", 1e-5))
        conf.tm_lr_head_mult            = best.get("tm_lr_head_mult", getattr(conf, "tm_lr_head_mult", 10.0))
        conf.tm_weight_decay            = best.get("tm_weight_decay", getattr(conf, "tm_weight_decay", 1e-5))
        # Family-specific backbone names (kept as simple attributes)
        if mt == "prithvi" and "fm_backbone" in best:
            conf.prithvi_backbone = best["fm_backbone"]
        if mt == "satmae" and "fm_backbone" in best:
            conf.satmae_backbone = best["fm_backbone"]
