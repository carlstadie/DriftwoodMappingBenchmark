import os
import json
import time
import random
from datetime import datetime
from typing import Dict, Any, Tuple, List

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, CSVLogger, TensorBoard, Callback
import keras_tuner as kt

from training import get_all_frames, create_train_val_datasets
from core.UNet import UNet
from core.Swin_UNetPP import SwinUNet
from core.losses import (accuracy, dice_coef, dice_loss, specificity, sensitivity,
                         f_beta, f1_score, IoU, nominal_surface_distance,
                         Hausdorff_distance, boundary_intersection_over_union, get_loss)
from core.optimizers import get_optimizer


# utils
def _seed(seed: int = 42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)


def _metrics(): # get list of metrics
    return [
        dice_coef, dice_loss, specificity, sensitivity, accuracy,
        f_beta, f1_score, IoU, nominal_surface_distance,
        Hausdorff_distance, boundary_intersection_over_union
    ]


def _ensure_dir(p):
    os.makedirs(p, exist_ok=True) # make directory


def _default(v, d):
    return v if v is not None else d


def _write_trials_csv(tuner: kt.engine.tuner.Tuner, out_path: str):
    """
    write all trials into a csv file
    with trial_id, score, and all hyperparameters as columns
    """
    trials = list(tuner.oracle.trials.values())
    hp_names = sorted({hp.name for hp in tuner.oracle.hyperparameters.space})
    lines = [",".join(["trial_id", "score"] + hp_names)]
    for tr in trials:
        row = [tr.trial_id, "" if tr.score is None else str(tr.score)]
        for k in hp_names:
            row.append(str(tr.hyperparameters.values.get(k, "")))
        lines.append(",".join(row))
    _ensure_dir(os.path.dirname(out_path))
    with open(out_path, "w") as f:
        f.write("\n".join(lines))


def _write_summary_md(project_dir: str, tag: str, tuner: kt.engine.tuner.Tuner,
                      phase: str, max_epochs: int, steps: int, val_steps: int, batch: int,
                      fixed: Dict[str, Any] = None, top_k: int = 5):
    
    """
    Write a markdown summary of the tuning phase, including:
    - date
    - objective
    - max epochs
    - steps per epoch
    - val steps
    - batch size
    - fixed params (if any)
    - top k trials with score and hyperparameters
    - links to all trials csv, best params json, tensorboard
    """

    best = tuner.oracle.get_best_trials(num_trials=min(top_k, len(tuner.oracle.trials)))

    lines: List[str] = []
    lines.append(f"# {tag} — {phase} summary")
    lines.append(f"- date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"- objective: val_dice_coef (max)")
    lines.append(f"- max_epochs: {max_epochs}")
    lines.append(f"- steps_per_epoch: {steps}")
    lines.append(f"- val_steps: {val_steps}")
    lines.append(f"- tune_batch_size: {batch}")

    if fixed:
        lines.append(f"- fixed params: " + ", ".join([f"{k}={v}" for k, v in fixed.items()]))
    lines.append("")
    lines.append("## top trials")

    for i, tr in enumerate(best, 1):
        lines.append(f"### {i}. trial {tr.trial_id}")
        lines.append(f"- score (val_dice_coef): {tr.score}")
        hp_str = ", ".join([f"{k}={v}" for k, v in tr.hyperparameters.values.items()])
        lines.append(f"- hparams: {hp_str}")

    lines.append("")
    lines.append("## files")
    lines.append(f"- all trials csv: {os.path.join(project_dir, f'{tag}_{phase}_all_trials.csv')}")
    lines.append(f"- best params json: {project_dir}")
    lines.append(f"- tensorboard: {project_dir}")

    with open(os.path.join(project_dir, f"{tag}_{phase}_summary.md"), "w") as f:
        f.write("\n".join(lines))


def _save_best(project_dir: str, tag: str, phase: str, best_hp: kt.HyperParameters) -> str:

    """ save best parameters to json"""

    d = {k: best_hp.get(k) for k in best_hp.values.keys()}

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    path = os.path.join(project_dir, f"{stamp}_{tag}_{phase}_best_hparams.json")

    _ensure_dir(os.path.dirname(path))

    with open(path, "w") as f:
        json.dump(d, f, indent=2)

    return path

def _round_to_valid(H, W, window_size, down_levels=3, patch_embed_stride=1):
    # ensure that for all downsampled levels: (H' % window_size == 0) and (W' % window_size == 0)
    # Start from token resolution if you have a patch embedding that reduces by stride
    h = int(np.ceil(H / patch_embed_stride))
    w = int(np.ceil(W / patch_embed_stride))
    for _ in range(down_levels + 1):
        h = int(np.ceil(h / window_size)) * window_size
        w = int(np.ceil(w / window_size)) * window_size
        # next level divides by 2 (patch merging)
        h //= 2; w //= 2
    # Rebuild input resolution
    # invert last //2 step
    for _ in range(down_levels + 1):
        h *= 2; w *= 2
    H_new = h * patch_embed_stride
    W_new = w * patch_embed_stride
    return H_new, W_new


# Data helpers (tuning-time overrides)

def _build_generators_for_tuning(frames, conf, tune_batch, patch_h=None, patch_w=None,
                                 augmenter_strength=1.0, min_pos_frac=0.0):
    # Keep originals
    old_bs = getattr(conf, "train_batch_size", None)
    old_patch = getattr(conf, "patch_size", None)
    old_aug = getattr(conf, "augmenter_strength", None)
    old_minpos = getattr(conf, "min_pos_frac", None)

    # Set overrides
    conf.train_batch_size = tune_batch
    if patch_h is not None and patch_w is not None:
        conf.patch_size = (int(patch_h), int(patch_w))   # <<<<<< ACTUAL OVERRIDE
    conf.augmenter_strength = float(augmenter_strength)
    conf.min_pos_frac = float(min_pos_frac)

    # make training.py see this config
    import training as training_mod
    training_mod.config = conf

    train_gen, val_gen, _ = create_train_val_datasets(frames)

    # restore originals
    if old_bs is not None: conf.train_batch_size = old_bs
    if old_patch is not None: conf.patch_size = old_patch
    if old_aug is not None: conf.augmenter_strength = old_aug
    if old_minpos is not None: conf.min_pos_frac = old_minpos

    return train_gen, val_gen




# Search spaces

# Phase 1 HyperBand - get broad discrete choises fast

def _optimizer_space_hb(hp: kt.HyperParameters):

    opt = hp.Choice("optimizer", ["adam", "adamw", "rmsprop"]) # discrete optimiser choise
    lr = hp.Float("learning_rate", 1e-5, 5e-3, sampling="log") # explore lr from 1e-5 to 5e-3 sampling as log
    wd = None

    if opt == "adamw":
        wd = hp.Float("weight_decay", 1e-6, 1e-2, sampling="log") # andam w add weight decay as add param

    return opt, lr, wd

def _loss_space_any(hp: kt.HyperParameters, base_loss: str, ab):

    a, b = ab if ab is not None else (0.5, 0.5) # get alpha and beta values

    if base_loss.lower() in ["tversky", "tversky_focal", "focal_tversky"]:
        a = hp.Float("tversky_alpha", 0.2, 0.8, step=0.1) # explore alpha from 0.2 to 0.8 in steps of 0.1
        b = hp.Float("tversky_beta",  0.2, 0.8, step=0.1) # same with beta

    return base_loss, (a, b)

# UNet discrete hyper params
def _unet_space_hb(hp: kt.HyperParameters):

    dilation = hp.Choice("dilation_rate", [1, 2, 4])
    layer_cnt = hp.Choice("layer_count", [32, 64, 96])
    l2w = hp.Choice("l2_weight", [0.0, 1e-5, 1e-4])
    drp = hp.Choice("dropout", [0.0, 0.1, 0.2])
    return dilation, layer_cnt, l2w, drp

# Swin discrete hypers
def _swin_space_hb(hp: kt.HyperParameters):

    base_C     = hp.Choice("C", [48, 64, 96]) # base channel for encodings
    patch_size = hp.Choice("patch_size", [8, 16]) #patch size
    window     = hp.Choice("window_size", [4, 7, 8]) #window sie devider

    # derive a simple shift choice: either none or half window
    ss_half    = hp.Choice("use_shift", [0, 1])  # 0 -> 0, 1 -> window//2
    attn_drop  = hp.Choice("attn_drop", [0.0, 0.1]) # attention dropout rate
    proj_drop  = hp.Choice("proj_drop", [0.0, 0.1]) # projection dropout rate
    mlp_drop   = hp.Choice("mlp_drop",  [0.0, 0.1, 0.2]) # MLP dropout rate
    drop_path  = hp.Choice("drop_path", [0.0, 0.1, 0.2]) # stochastic depth rate

    return base_C, patch_size, window, ss_half, attn_drop, proj_drop, mlp_drop, drop_path

# Data discrete HParams
def _data_space_hb(hp: kt.HyperParameters):

    patch_h = hp.Choice("patch_h", [256, 384, 512]) 
    patch_w = hp.Choice("patch_w", [256, 384, 512])
    aug_str = hp.Choice("augmenter_strength", [0.0, 0.5, 1.0]) # how large the fraction of augmentation should be
    minpos  = hp.Choice("min_pos_frac", [0.0, 0.01, 0.02]) # min fraction of positive pixels in a patch

    return patch_h, patch_w, aug_str, minpos

# Phase 2 baysian optimisation - refine coninous parameters, slower but more accurate
def _optimizer_space_bo(hp: kt.HyperParameters, fixed_opt: str):

    lr = hp.Float("learning_rate", 1e-5, 5e-3, sampling="log") # now just lr
    wd = None
    if fixed_opt == "adamw":
        wd = hp.Float("weight_decay", 1e-6, 1e-2, sampling="log") # same for adamw

    return fixed_opt, lr, wd

def _loss_space_bo(hp: kt.HyperParameters, base_loss: str, ab): #same with tveersky 

    # expose tversky a/b if applicable, with finer step
    a, b = ab if ab is not None else (0.5, 0.5)
    if base_loss.lower() in ["tversky", "tversky_focal", "focal_tversky"]:
        a = hp.Float("tversky_alpha", 0.2, 0.8, step=0.05)
        b = hp.Float("tversky_beta",  0.2, 0.8, step=0.05)
    return base_loss, (a, b)



# Compile helper
def _compile_with_optimizer(model, opt_name, lr, wd, conf, loss_name, a, b):
    # Try project’s optimizer factory first
    opt_obj = None
    try:
        maybe_opt = get_optimizer(
            _default(opt_name, getattr(conf, "optimizer_fn", "adam")),
            _default(getattr(conf, "num_epochs", 50), 50),
            _default(getattr(conf, "num_training_steps", 100), 100),
        )
        opt_obj = maybe_opt
    except Exception:
        opt_obj = None

    # If factory gave back a string or None, make a Keras optimizer ourselves
    if isinstance(opt_obj, str) or opt_obj is None:
        name = opt_name or "adam"
        name = name.lower()
        if name == "adamw":
            AdamW = getattr(tf.keras.optimizers, "AdamW", None)
            if AdamW is not None:
                opt_obj = AdamW(learning_rate=lr, weight_decay=_default(wd, 1e-4))
            else:
                # Fallback if this TF doesn’t have AdamW
                opt_obj = tf.keras.optimizers.Adam(learning_rate=lr)
        elif name == "rmsprop":
            opt_obj = tf.keras.optimizers.RMSprop(learning_rate=lr)
        else:
            opt_obj = tf.keras.optimizers.Adam(learning_rate=lr)
    else:
        # Got an optimizer instance from your factory: set lr / wd if available
        if hasattr(opt_obj, "learning_rate"):
            try:
                opt_obj.learning_rate = lr
            except Exception:
                pass
        if (opt_name == "adamw") and hasattr(opt_obj, "weight_decay") and wd is not None:
            try:
                opt_obj.weight_decay = wd
            except Exception:
                pass

    model.compile(optimizer=opt_obj, loss=get_loss(loss_name, (a, b)), metrics=_metrics())
    return model



# Builders — Phase 1 (HB)

def _build_unet_hb(hp: kt.HyperParameters, conf, tune_batch, patch_h, patch_w):

    opt, lr, wd = _optimizer_space_hb(hp) # get optimiser, lr and weight decay

    loss_name, (a, b) = _loss_space_any(hp, _default(getattr(conf, "loss_fn", "tversky"), "tversky"),
                                        _default(getattr(conf, "tversky_alphabeta", (0.5, 0.5)), (0.5, 0.5))) # get loss and alpha beta
    
    dilation, layer_cnt, l2w, drp = _unet_space_hb(hp) # get parameters for unet architecture

    input_shape = [tune_batch, patch_h, patch_w, len(conf.channel_list)]

    num_classes = 1

    model = UNet(input_shape, num_classes, dilation_rate=dilation, layer_count=layer_cnt,
                 l2_weight=l2w, dropout=drp) # build unet
    
    return _compile_with_optimizer(model, opt, lr, wd, conf, loss_name, a, b) #compile full model


def _build_swin_hb(hp: kt.HyperParameters, conf, tune_batch, H, W):

    opt, lr, wd = _optimizer_space_hb(hp) #again get optimiser, lr and weight decay

    loss_name, (a, b) = _loss_space_any(hp, _default(getattr(conf, "loss_fn", "tversky"), "tversky"),
                                        _default(getattr(conf, "tversky_alphabeta", (0.5, 0.5)), (0.5, 0.5))) # get loss and alpha beta
    
    C, patch_size, window, ss_half, attn_drop, proj_drop, mlp_drop, drop_path = _swin_space_hb(hp) # get architecture specific params
    ss_size = 0 if ss_half == 0 else window // 2

    H_adj, W_adj = _round_to_valid(H, W, window_size=window, down_levels=3, patch_embed_stride=patch_size) # make sure unly window sizes div by 2 exists

    model = SwinUNet(H=H_adj, W=W_adj, ch=len(conf.channels_used), C=C, patch_size=patch_size,
                     window_size=window, ss_size=ss_size, attn_drop=attn_drop, proj_drop=proj_drop,
                     mlp_drop=mlp_drop, drop_path=drop_path) # build model with all params
    
    return _compile_with_optimizer(model, opt, lr, wd, conf, loss_name, a, b) # aaaaand compile


# Builders — Phase 2 (BO)

def _build_unet_bo(hp: kt.HyperParameters, conf, tune_batch, patch_h, patch_w, fixed: Dict[str, Any]):

    opt, lr, wd = _optimizer_space_bo(hp, fixed["optimizer"])
    loss_name, (a, b) = _loss_space_bo(hp, _default(getattr(conf, "loss_fn", "tversky"), "tversky"),
                                       _default(getattr(conf, "tversky_alphabeta", (0.5, 0.5)), (0.5, 0.5)))

    input_shape = [tune_batch, patch_h, patch_w, len(conf.channel_list)]
    num_classes = 1

    model = UNet(input_shape, num_classes,
                 dilation_rate=fixed.get("dilation_rate", 1),
                 layer_count=fixed.get("layer_count", 64),
                 l2_weight=fixed.get("l2_weight", 1e-4),
                 dropout=fixed.get("dropout", 0.0))
    
    return _compile_with_optimizer(model, opt, lr, wd, conf, loss_name, a, b)


def _build_swin_bo(hp: kt.HyperParameters, conf, tune_batch, H, W, fixed: Dict[str, Any]):

    opt, lr, wd = _optimizer_space_bo(hp, fixed["optimizer"])
    loss_name, (a, b) = _loss_space_bo(hp, _default(getattr(conf, "loss_fn", "tversky"), "tversky"),
                                       _default(getattr(conf, "tversky_alphabeta", (0.5, 0.5)), (0.5, 0.5)))

    model = SwinUNet(H=H, W=W, ch=len(conf.channels_used),
                     C=fixed.get("C", 64),
                     patch_size=fixed.get("patch_size", 16),
                     window_size=fixed.get("window_size", 4),
                     ss_size=fixed.get("ss_size", 2),
                     attn_drop=fixed.get("attn_drop", 0.0),
                     proj_drop=fixed.get("proj_drop", 0.0),
                     mlp_drop=fixed.get("mlp_drop", 0.0),
                     drop_path=fixed.get("drop_path", 0.1))
    
    return _compile_with_optimizer(model, opt, lr, wd, conf, loss_name, a, b)



# One phase executor

def _run_phase(conf, model_type: str, phase: str, project_dir: str,
               tune_batch: int, max_epochs: int, steps: int, val_steps: int,
               executions_per_trial: int, overwrite: bool,
               hb_data_hp: Dict[str, Any]) -> Tuple[Dict[str, Any], kt.engine.tuner.Tuner]:

    frames = get_all_frames()
    train_gen, val_gen = _build_generators_for_tuning(
        frames, conf, tune_batch,
        patch_h=hb_data_hp["patch_h"], patch_w=hb_data_hp["patch_w"],
        augmenter_strength=hb_data_hp["augmenter_strength"],
        min_pos_frac=hb_data_hp["min_pos_frac"]
    )

    tag = f"{model_type}_tuning"
    project_name = f"{tag}_{phase}"               # <-- key change
    # (optional) keep each phase in its own subdir:
    # project_dir = os.path.join(project_dir, phase)

    _ensure_dir(project_dir)
    objective = kt.Objective("val_dice_coef", direction="max")

    # -- define builder BEFORE creating tuner --
    def builder(hp: kt.HyperParameters):
        if phase == "HB":
            patch_h = hp.Choice("patch_h", [256, 384, 512], default=hb_data_hp["patch_h"])
            patch_w = hp.Choice("patch_w", [256, 384, 512], default=hb_data_hp["patch_w"])
            if model_type == "unet":
                return _build_unet_hb(hp, conf, tune_batch, patch_h, patch_w)
            else:
                return _build_swin_hb(hp, conf, tune_batch, H=patch_h, W=patch_w)
        else:
            patch_h = hb_data_hp["patch_h"]; patch_w = hb_data_hp["patch_w"]
            if model_type == "unet":
                return _build_unet_bo(hp, conf, tune_batch, patch_h, patch_w, fixed=hb_data_hp["fixed"])
            else:
                return _build_swin_bo(hp, conf, tune_batch, H=patch_h, W=patch_w, fixed=hb_data_hp["fixed"])

    if phase == "HB":
        tuner = kt.Hyperband(
            builder, objective=objective, max_epochs=max_epochs, factor=3,
            executions_per_trial=executions_per_trial,
            directory=project_dir, project_name=project_name,
            overwrite=overwrite
        )
    else:
        tuner = kt.BayesianOptimization(
            builder, objective=objective,
            max_trials=_default(getattr(conf, "tune_max_trials", None), 30),
            executions_per_trial=executions_per_trial,
            directory=project_dir, project_name=project_name,
            overwrite=True  # <- avoid reloading HB oracle
        )

    early = EarlyStopping(monitor="val_dice_coef", mode="max", patience=5, restore_best_weights=True)
    csv_logger = CSVLogger(os.path.join(project_dir, f"{tag}_{phase}_fit_history.csv"), append=True)

    tuner.search(
        train_gen, steps_per_epoch=steps, epochs=max_epochs,
        validation_data=val_gen, validation_steps=val_steps,
        callbacks=[early, csv_logger], verbose=1, workers=1
    )

    best_hp = tuner.get_best_hyperparameters(1)[0]
    best_dict = {k: best_hp.get(k) for k in best_hp.values.keys()}

    _write_trials_csv(tuner, os.path.join(project_dir, f"{tag}_{phase}_all_trials.csv"))
    fixed = hb_data_hp.get("fixed", None)
    _write_summary_md(project_dir, tag, tuner, phase, max_epochs, steps, val_steps, tune_batch, fixed=fixed)
    best_path = _save_best(project_dir, tag, phase, best_hp)
    print(f"[{tag} {phase}] best saved to: {best_path}")

    return best_dict, tuner




# Chained tuner (HB --> BO)

def _tune_chained(conf, model_type: str = "unet",
                  executions_per_trial: int = 1, overwrite: bool = False) -> Dict[str, Any]:
    global config
    config = conf
    print("Starting chained tuning (HyperBand + Bayesian).")

    import training as training_mod
    training_mod.config=conf
    _seed(_default(getattr(config, "seed", None), 42))
    tf.config.run_functions_eagerly(False)

    hb_epochs = _default(getattr(config, "tune_num_epochs", None), 20)
    bo_epochs = _default(getattr(config, "tune_num_epochs_bo", None), hb_epochs)
    steps     = _default(getattr(config, "tune_steps_per_epoch", None),
                         min(100, _default(getattr(config, "num_training_steps", 100), 100)))
    val_steps = _default(getattr(config, "tune_validation_steps", None),
                         min(50, _default(getattr(config, "num_validation_images", 50), 50)))
    tune_batch = _default(getattr(config, "tune_batch_size", None),
                          min(8, _default(getattr(config, "train_batch_size", 8), 8)))

    logs_dir = _default(getattr(config, "logs_dir", "./logs"), "./logs")
    key = "unet" if model_type.lower() == "unet" else "swin"
    project_dir = os.path.join(logs_dir, f"{key}_tuning")
    _ensure_dir(project_dir)

    # --- Phase 1: HB (broad) ---
    # Sample initial data HPs (so different runs can vary); they’ll also be re-declared inside HB builder for logging.
    hb_patch_h = 384
    hb_patch_w = 384
    hb_aug     = _default(getattr(config, "augmenter_strength", None), 1.0)
    hb_minpos  = _default(getattr(config, "min_pos_frac", None), 0.0)
    hb_data_hp = dict(patch_h=hb_patch_h, patch_w=hb_patch_w,
                      augmenter_strength=hb_aug, min_pos_frac=hb_minpos)
    hb_best, _ = _run_phase(config, key, "HB", project_dir,
                            tune_batch, hb_epochs, steps, val_steps,
                            executions_per_trial, overwrite, hb_data_hp)

    # Determine fixed winners for BO
    fixed = {"optimizer": hb_best.get("optimizer", _default(getattr(config, "optimizer_fn", "adam"), "adam"))}
    if key == "unet":
        fixed["dilation_rate"] = hb_best.get("dilation_rate", _default(getattr(config, "dilation_rate", 1), 1))
        fixed["layer_count"]   = hb_best.get("layer_count", 64)
        fixed["l2_weight"]     = float(hb_best.get("l2_weight", 1e-4))
        fixed["dropout"]       = float(hb_best.get("dropout", 0.0))
    else:
        fixed["C"]             = hb_best.get("C", _default(getattr(config, "swin_base_C", 64), 64))
        fixed["patch_size"]    = hb_best.get("patch_size", _default(getattr(config, "swin_patch_size", 16), 16))
        fixed["window_size"]   = hb_best.get("window_size", 4)
        fixed["ss_size"]       = (fixed["window_size"] // 2) if int(hb_best.get("use_shift", 1)) == 1 else 0
        fixed["attn_drop"]     = float(hb_best.get("attn_drop", 0.0))
        fixed["proj_drop"]     = float(hb_best.get("proj_drop", 0.0))
        fixed["mlp_drop"]      = float(hb_best.get("mlp_drop", 0.0))
        fixed["drop_path"]     = float(hb_best.get("drop_path", 0.1))

    # Adopt HB-chosen data patch size for BO as well (freeze)
    bo_patch_h = int(hb_best.get("patch_h", hb_patch_h))
    bo_patch_w = int(hb_best.get("patch_w", hb_patch_w))

    # --- Phase 2: BO (refine continuous) ---
    bo_data_hp = dict(patch_h=bo_patch_h, patch_w=bo_patch_w,
                      augmenter_strength=hb_aug, min_pos_frac=hb_minpos, fixed=fixed)
    bo_best, _ = _run_phase(config, key, "BO", project_dir,
                            tune_batch, bo_epochs, steps, val_steps,
                            executions_per_trial, overwrite, bo_data_hp)

    # Merge results
    final = dict(fixed)
    if "learning_rate" in bo_best: final["learning_rate"] = float(bo_best["learning_rate"])
    if "weight_decay" in bo_best:  final["weight_decay"]  = float(bo_best["weight_decay"])
    if "tversky_alpha" in bo_best: final["tversky_alpha"] = float(bo_best["tversky_alpha"])
    if "tversky_beta"  in bo_best: final["tversky_beta"]  = float(bo_best["tversky_beta"])
    final["patch_h"] = bo_patch_h
    final["patch_w"] = bo_patch_w
    final["augmenter_strength"] = float(hb_aug)
    final["min_pos_frac"] = float(hb_minpos)

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    final_path = os.path.join(project_dir, f"{stamp}_{key}_tuning_FINAL_best_hparams.json")
    with open(final_path, "w") as f:
        json.dump(final, f, indent=2)
    print(f"[{key} tuning] FINAL merged best saved to: {final_path}")
    print(json.dumps(final, indent=2))
    return final

def tune_UNet(conf) -> Dict[str, Any]:
    return _tune_chained(conf, model_type="unet")


def tune_SwinUNetPP(conf) -> Dict[str, Any]:
    return _tune_chained(conf, model_type="swin")


def apply_best_to_config(conf, best: Dict[str, Any], model_type: str):
    # optimizer / schedule
    if "optimizer" in best:
        conf.optimizer_fn = best["optimizer"]
    if "learning_rate" in best:
        conf.learning_rate = best["learning_rate"]
    if "weight_decay" in best:
        conf.weight_decay = best["weight_decay"]

    # loss
    if "tversky_alpha" in best or "tversky_beta" in best:
        a = best.get("tversky_alpha", getattr(conf, "tversky_alphabeta", (0.5, 0.5))[0])
        b = best.get("tversky_beta",  getattr(conf, "tversky_alphabeta", (0.5, 0.5))[1])
        conf.tversky_alphabeta = (a, b)

    # data / generator knobs
    conf.augmenter_strength = best.get("augmenter_strength", getattr(conf, "augmenter_strength", 1.0))
    conf.min_pos_frac       = best.get("min_pos_frac", getattr(conf, "min_pos_frac", 0.0))
    conf.patch_size         = [int(best.get("patch_h", conf.patch_size[0])),
                               int(best.get("patch_w", conf.patch_size[1]))]

    # model-specific
    if model_type.lower() == "unet":
        conf.dilation_rate = best.get("dilation_rate", getattr(conf, "dilation_rate", 1))
        conf.layer_count   = best.get("layer_count", getattr(conf, "layer_count", 64))
        conf.l2_weight     = best.get("l2_weight", getattr(conf, "l2_weight", 1e-4))
        conf.dropout       = best.get("dropout", getattr(conf, "dropout", 0.0))
    else:
        conf.swin_base_C     = best.get("C", getattr(conf, "swin_base_C", 64))
        conf.swin_patch_size = best.get("patch_size", getattr(conf, "swin_patch_size", 16))
        conf.swin_window     = best.get("window_size", getattr(conf, "swin_window", 4))
        conf.swin_ss_size    = best.get("ss_size", getattr(conf, "swin_ss_size", 2))
        conf.swin_attn_drop  = best.get("attn_drop", getattr(conf, "swin_attn_drop", 0.0))
        conf.swin_proj_drop  = best.get("proj_drop", getattr(conf, "swin_proj_drop", 0.0))
        conf.swin_mlp_drop   = best.get("mlp_drop", getattr(conf, "swin_mlp_drop", 0.0))
        conf.swin_drop_path  = best.get("drop_path", getattr(conf, "swin_drop_path", 0.1))


