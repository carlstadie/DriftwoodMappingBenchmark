# imports
# here we bring in standard libs, tensorflow/keras, and project utilities.

import os  # filesystem paths and folders
import json  # read/write json configuration artifacts
import time  # timestamps if needed by callers
import random  # python RNG
from datetime import datetime  # readable timestamps for filenames
from typing import Dict, Any, Tuple, List  # type hints for clarity

import numpy as np  # numeric helpers
import tensorflow as tf  # deep learning backend
from tensorflow.keras.callbacks import EarlyStopping, CSVLogger, TensorBoard, Callback  # training callbacks
import keras_tuner as kt  # hyperparameter tuning

from training import get_all_frames, create_train_val_datasets  # project data entry points
from core.UNet import UNet  # UNet model
from core.Swin_UNetPP import SwinUNet  # Swin-UNet++ variant
from core.losses import (  # metrics and losses used across the project
    accuracy, dice_coef, dice_loss, specificity, sensitivity,
    f_beta, f1_score, IoU, nominal_surface_distance,
    Hausdorff_distance, boundary_intersection_over_union, get_loss
)
from core.optimizers import get_optimizer  # optional project optimizer factory


# small utilities
# here we keep tiny helpers for reproducibility, metrics lists, safe io, and defaults.

def _seed(seed: int = 42):
    """set all RNGs so runs are reproducible."""  # here we stabilize sources of randomness
    os.environ["PYTHONHASHSEED"] = str(seed)  # stabilize python hash seed
    np.random.seed(seed)  # numpy rng
    random.seed(seed)  # python rng
    tf.random.set_seed(seed)  # tensorflow rng


def _metrics():
    """return the common segmentation metrics used in logging."""  # here we centralize metric selection
    return [
        dice_coef, dice_loss, specificity, sensitivity, accuracy,
        f_beta, f1_score, IoU, nominal_surface_distance,
        Hausdorff_distance, boundary_intersection_over_union
    ]  # this keeps evaluation consistent across scripts


def _ensure_dir(path):
    os.makedirs(path, exist_ok=True)  # here we create a directory if missing (idempotent)


def _default(val, fallback):
    return val if val is not None else fallback  # here we treat None specially (0 should pass through)


def _write_trials_csv(tuner: kt.engine.tuner.Tuner, out_path: str):
    """
    write all tuner trials into a csv.                                                
    columns are trial_id, score, and each hyperparameter in the search space.           
    """
    trials = list(tuner.oracle.trials.values())  # collect trials from the oracle
    hp_names = sorted({hp.name for hp in tuner.oracle.hyperparameters.space})  # stable column order
    lines = [",".join(["trial_id", "score"] + hp_names)]  # header row
    for tr in trials:
        row = [tr.trial_id, "" if tr.score is None else str(tr.score)]  # score may be None while running
        for name in hp_names:
            row.append(str(tr.hyperparameters.values.get(name, "")))  # missing hp -> blank
        lines.append(",".join(row))  # append csv row
    _ensure_dir(os.path.dirname(out_path))  # make sure the folder exists
    with open(out_path, "w") as f:
        f.write("\n".join(lines))  # write file


def _nan_to_num(x, constant):
    """replace NaN/Inf with a finite constant."""  # here we avoid numerical blow-ups
    x = tf.where(tf.math.is_finite(x), x, tf.cast(constant, x.dtype))  # mask non-finite values
    return x  # sanitized tensor


def _sanitize_pair_xy(x, y):
    """cast and clip (x, y) to float32 and [0, 1]."""  # here we keep BCE/Dice inputs valid
    x = tf.cast(x, tf.float32)  # model inputs as float32
    y = tf.cast(y, tf.float32)  # labels as float32
    x = _nan_to_num(x, 0.0)  # replace non-finite inputs
    y = _nan_to_num(y, 0.0)  # replace non-finite labels
    x = tf.clip_by_value(x, 0.0, 1.0)  # bound to range
    y = tf.clip_by_value(y, 0.0, 1.0)  # bound to range
    return x, y  # sanitized tensors


def _sanitize_map(*args):
    """
    sanitize dataset tuples.
    we cast to float32, replace non-finite values, and clip weights to a safe range.
    """
    if len(args) == 2:
        x, y = args
        return _sanitize_pair_xy(x, y)  # clean features and labels
    if len(args) == 3:
        x, y, w = args
        x, y = _sanitize_pair_xy(x, y)  # clean features and labels
        w = tf.cast(_nan_to_num(w, 1.0), tf.float32)  # weights become finite
        w = tf.clip_by_value(w, 0.0, 1e6)  # keep weights non-negative and bounded
        return x, y, w  # clean triple
    return args  # pass through for unknown structures


def _snap_hw_for_swin(H, W, patch_size, window_size, down_levels=3):
    """
    compute an H,W so that token maps are divisible by window_size at all scales.
    we consider patch embedding and pyramid downsamples.               
    """
    need = window_size * (2 ** down_levels)  # required multiple at token resolution
    T_h = int(np.ceil(H / patch_size))  # token height
    T_w = int(np.ceil(W / patch_size))  # token width
    T_h_adj = int(np.ceil(T_h / need)) * need  # round up to a valid multiple
    T_w_adj = int(np.ceil(T_w / need)) * need  # round up to a valid multiple
    H_adj = T_h_adj * patch_size  # back to pixel space
    W_adj = T_w_adj * patch_size  # back to pixel space
    return H_adj, W_adj  # adjusted sizes


def _write_summary_md(project_dir: str, tag: str, tuner: kt.engine.tuner.Tuner,
                      phase: str, max_epochs: int, steps: int, val_steps: int, batch: int,
                      fixed: Dict[str, Any] = None, top_k: int = 5):
    """
    write a short markdown summary with settings and the top trials.                    
    this also points to csv/json/tensorboard locations.                                
    """
    best = tuner.oracle.get_best_trials(num_trials=min(top_k, len(tuner.oracle.trials)))  # get top-k trials

    lines: List[str] = []
    lines.append(f"# {tag} — {phase} summary")  # title
    lines.append(f"- date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")  # timestamp
    lines.append(f"- objective: val_dice_coef (max)")  # objective used
    lines.append(f"- max_epochs: {max_epochs}")  # training budget
    lines.append(f"- steps_per_epoch: {steps}")  # train loop size
    lines.append(f"- val_steps: {val_steps}")  # validation loop size
    lines.append(f"- tune_batch_size: {batch}")  # batch used during tuning

    if fixed:
        lines.append(f"- fixed params: " + ", ".join([f"{k}={v}" for k, v in fixed.items()]))  # frozen knobs
    lines.append("")
    lines.append("## top trials")  # section

    for i, tr in enumerate(best, 1):
        lines.append(f"### {i}. trial {tr.trial_id}")  # identifier
        lines.append(f"- score (val_dice_coef): {tr.score}")  # comparable metric
        hp_str = ", ".join([f"{k}={v}" for k, v in tr.hyperparameters.values.items()])  # hyperparameters
        lines.append(f"- hparams: {hp_str}")  # flat listing

    lines.append("")
    lines.append("## files")  # artifact pointers
    lines.append(f"- all trials csv: {os.path.join(project_dir, f'{tag}_{phase}_all_trials.csv')}")  # tabular view
    lines.append(f"- best params json: {project_dir}")  # json directory
    lines.append(f"- tensorboard: {project_dir}")  # logs directory

    with open(os.path.join(project_dir, f"{tag}_{phase}_summary.md"), "w") as f:
        f.write("\n".join(lines))  # write summary file


def _save_best(project_dir: str, tag: str, phase: str, best_hp: kt.HyperParameters) -> str:
    """save the best trial hyperparameters to a timestamped json file."""  # here we make best settings portable
    payload = {k: best_hp.get(k) for k in best_hp.values.keys()}  # extract values to a dict
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")  # unique-ish name part
    path = os.path.join(project_dir, f"{stamp}_{tag}_{phase}_best_hparams.json")  # output path
    _ensure_dir(os.path.dirname(path))  # ensure folder
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)  # pretty-print json
    return path  # let caller log the path


def _round_to_valid(H, W, window_size, down_levels=3, patch_embed_stride=1):
    """
    round up H,W so all downsampled scales are divisible by window_size.
    useful when patch embedding reduces resolution before the pyramid.
    """
    h = int(np.ceil(H / patch_embed_stride))  # token-level height
    w = int(np.ceil(W / patch_embed_stride))  # token-level width
    for _ in range(down_levels + 1):
        h = int(np.ceil(h / window_size)) * window_size  # round to multiple
        w = int(np.ceil(w / window_size)) * window_size  # round to multiple
        h //= 2; w //= 2  # mimic downsampling per level
    for _ in range(down_levels + 1):
        h *= 2; w *= 2  # invert the pyramid
    H_new = h * patch_embed_stride  # back to pixels
    W_new = w * patch_embed_stride  # back to pixels
    return H_new, W_new  # final sizes


def _patch_kerastuner_oracle():
    """
    patch the keras-tuner oracle to guard against None counters.
    this normalizes failure counters before they are compared.
    """
    try:
        from keras_tuner.engine import oracle as kt_oracle  # modern path
    except Exception:
        import kerastuner.engine.oracle as kt_oracle  # type: ignore  # legacy path

    if getattr(kt_oracle.Oracle, "_pd_safe_patached", False):
        return  # already patched

    def _safe_check(self, *args, **kwargs):
        cf = getattr(self, "_consecutive_failures", 0) or 0  # normalize to int
        try: setattr(self, "_consecutive_failures", cf)
        except Exception: pass
        limit = getattr(self, "max_consecutive_failed_trials", 3) or 3  # default limit
        try: setattr(self, "max_consecutive_failed_trials", limit)
        except Exception: pass
        if int(cf) >= int(limit):
            raise RuntimeError(f"Number of consecutive failures excceeded the limit of {int(limit)}.")  # same semantics
        return None  # match original signature

    kt_oracle.Oracle._check_consecutive_failures = _safe_check  # apply patch
    kt_oracle.Oracle._pd_safe_patached = True  # mark as patched


# data helpers (tuning-time overrides)
# here we temporarily adjust config to build datasets for tuning, then restore it.

def _build_generators_for_tuning(frames, conf, tune_batch, patch_h=None, patch_w=None,
                                 augmenter_strength=1.0, min_pos_frac=0.0):
    """
    build train/val datasets with temporary overrides (batch/patch/aug/minpos).
    we restore the original config afterwards. 
    """
    # save originals
    old_bs = getattr(conf, "train_batch_size", None)  # original batch size
    old_patch = getattr(conf, "patch_size", None)  # original patch
    old_aug = getattr(conf, "augmenter_strength", None)  # original aug strength
    old_minpos = getattr(conf, "min_pos_frac", None)  # original min-pos fraction

    # apply overrides
    conf.train_batch_size = tune_batch  # temporary batch size for tuning
    if patch_h is not None and patch_w is not None:
        conf.patch_size = (int(patch_h), int(patch_w))  # temporary patch size
    conf.augmenter_strength = float(augmenter_strength)  # temporary aug strength
    conf.min_pos_frac = float(min_pos_frac)  # temporary min-pos constraint

    # make training.py pick up updated config
    import training as training_mod  # local import to avoid circular deps
    training_mod.config = conf  # update module-level handle

    train_gen, val_gen, _ = create_train_val_datasets(frames)  # build datasets from frames

    # sanitize and prefetch
    if isinstance(train_gen, tf.data.Dataset):
        train_gen = train_gen.map(_sanitize_map, num_parallel_calls=tf.data.AUTOTUNE)  # clean batches
        train_gen = train_gen.prefetch(tf.data.AUTOTUNE)  # overlap producer/consumer
    if isinstance(val_gen, tf.data.Dataset):
        val_gen = val_gen.map(_sanitize_map, num_parallel_calls=tf.data.AUTOTUNE)  # clean val batches
        val_gen = val_gen.prefetch(tf.data.AUTOTUNE)  # reduce stalls

    # restore originals
    if old_bs is not None: conf.train_batch_size = old_bs  # put back batch size
    if old_patch is not None: conf.patch_size = old_patch  # put back patch size
    if old_aug is not None: conf.augmenter_strength = old_aug  # put back aug strength
    if old_minpos is not None: conf.min_pos_frac = old_minpos  # put back min-pos

    return train_gen, val_gen  # return tuned datasets


# search spaces
# here we define hyperparameter spaces for hyperband (discrete) and bo (continuous refinement).

def _optimizer_space_hb(hp: kt.HyperParameters):
    """discrete optimizer choices and a log-spaced learning rate; wd only for adamw."""  # here we explore broadly
    opt = hp.Choice("optimizer", ["adam", "adamw", "rmsprop"])  # algorithm family
    lr = hp.Float("learning_rate", 1e-5, 5e-3, sampling="log")  # lr across orders of magnitude
    wd = None  # initialized only if needed
    if opt == "adamw":
        wd = hp.Float("weight_decay", 1e-6, 1e-2, sampling="log")  # decoupled wd
    return opt, lr, wd  # return knobs for compile


def _loss_space_any(hp: kt.HyperParameters, base_loss: str, ab):
    """expose tversky α/β when requested; otherwise pass defaults through."""  # here we support tversky-family
    a, b = ab if ab is not None else (0.5, 0.5)  # defaults
    if base_loss.lower() in ["tversky", "tversky_focal", "focal_tversky"]:
        a = hp.Float("tversky_alpha", 0.2, 0.8, step=0.1)  # coarse grid for hb
        b = 1.0 - a # Tversky must amount to 1
    return base_loss, (a, b)  # normalized output


def _unet_space_hb(hp: kt.HyperParameters):
    """discrete unet architecture knobs for hyperband."""  # here we keep the space compact
    dilation = hp.Choice("dilation_rate", [1, 2, 4])  # receptive field
    layer_cnt = hp.Choice("layer_count", [32, 64, 96])  # base channels per stage
    l2w = hp.Choice("l2_weight", [0.0, 1e-5, 1e-4])  # kernel regularization
    drp = hp.Choice("dropout", [0.0, 0.1, 0.2])  # dropout regularization
    return dilation, layer_cnt, l2w, drp  # return architecture choices


def _swin_space_hb(hp: kt.HyperParameters):
    """discrete swin unet knobs for hyperband."""  # here we limit search for speed
    base_C     = hp.Choice("C", [48, 64, 96])  # base channels
    patch_size = hp.Choice("patch_size", [8, 16])  # patch embedding size
    window     = hp.Choice("window_size", [4, 7, 8])  # attention window
    ss_half    = hp.Choice("use_shift", [0, 1])  # 0=no shift, 1=window//2 shift
    attn_drop  = hp.Choice("attn_drop", [0.0, 0.1])  # attention dropout
    proj_drop  = hp.Choice("proj_drop", [0.0, 0.1])  # projection dropout
    mlp_drop   = hp.Choice("mlp_drop",  [0.0, 0.1, 0.2])  # mlp dropout
    drop_path  = hp.Choice("drop_path", [0.0, 0.1, 0.2])  # stochastic depth
    return base_C, patch_size, window, ss_half, attn_drop, proj_drop, mlp_drop, drop_path  # return knobs


def _data_space_hb(hp: kt.HyperParameters):
    """discrete data knobs for hyperband (patch, aug strength, min-pos)."""  # here we co-tune data with model
    patch_h = hp.Choice("patch_h", [256, 384, 512])  # crop height
    patch_w = hp.Choice("patch_w", [256, 384, 512])  # crop width
    aug_str = hp.Choice("augmenter_strength", [0.0, 0.5, 1.0])  # augmentation intensity
    minpos  = hp.Choice("min_pos_frac", [0.0, 0.01, 0.02])  # required positive fraction
    return patch_h, patch_w, aug_str, minpos  # return data choices


def _optimizer_space_bo(hp: kt.HyperParameters, fixed_opt: str):
    """continuous lr (and wd for adamw) for bo refinement."""  # here we fine-tune around hb winners
    lr = hp.Float("learning_rate", 1e-5, 5e-3, sampling="log")  # refine lr
    wd = None
    if fixed_opt == "adamw":
        wd = hp.Float("weight_decay", 1e-6, 1e-2, sampling="log")  # refine wd
    return fixed_opt, lr, wd  # keep optimizer fixed, tune scalars


def _loss_space_bo(hp: kt.HyperParameters, base_loss: str, ab):
    """finer α/β steps for tversky-like losses during bo."""  # here we allow smaller adjustments
    a, b = ab if ab is not None else (0.5, 0.5)
    if base_loss.lower() in ["tversky", "tversky_focal", "focal_tversky"]:
        a = hp.Float("tversky_alpha", 0.2, 0.8, step=0.05)  # finer grid
        b = 1.0 - a # Tversky must amount to 1
    return base_loss, (a, b)  # return loss knobs


# numerically stable tuning metrics
# here we define safe dice/iou plus a binary accuracy for sanity.

def _safe_dice(y_true, y_pred, eps=1e-6):
    """dice with clipping and nan guards; averaged over the batch."""  # here we keep the objective stable
    y_true = tf.cast(y_true, tf.float32)  # cast to float32
    y_pred = tf.cast(y_pred, tf.float32)  # cast to float32
    y_true = tf.clip_by_value(_nan_to_num(y_true, 0.0), 0.0, 1.0)  # sanitize labels
    y_pred = tf.clip_by_value(_nan_to_num(y_pred, 0.0), 0.0, 1.0)  # sanitize preds
    y_true_f = tf.reshape(y_true, [tf.shape(y_true)[0], -1])  # flatten per-sample
    y_pred_f = tf.reshape(y_pred, [tf.shape(y_pred)[0], -1])  # flatten per-sample
    inter = tf.reduce_sum(y_true_f * y_pred_f, axis=1)  # intersection per-sample
    den   = tf.reduce_sum(y_true_f, axis=1) + tf.reduce_sum(y_pred_f, axis=1)  # denominator per-sample
    dice  = (2.0 * inter + eps) / (den + eps)  # dice score
    return tf.reduce_mean(dice)  # mean over batch


def _safe_iou(y_true, y_pred, eps=1e-6):
    """iou with clipping and nan guards; averaged over the batch."""  #  we add a complementary view
    y_true = tf.cast(y_true, tf.float32)  # cast
    y_pred = tf.cast(y_pred, tf.float32)  # cast
    y_true = tf.clip_by_value(_nan_to_num(y_true, 0.0), 0.0, 1.0)  # sanitize labels
    y_pred = tf.clip_by_value(_nan_to_num(y_pred, 0.0), 0.0, 1.0)  # sanitize preds
    y_true_f = tf.reshape(y_true, [tf.shape(y_true)[0], -1])  # flatten
    y_pred_f = tf.reshape(y_pred, [tf.shape(y_pred)[0], -1])  # flatten
    inter = tf.reduce_sum(y_true_f * y_pred_f, axis=1)  # intersection
    union = tf.reduce_sum(y_true_f + y_pred_f, axis=1) - inter  # union
    iou   = (inter + eps) / (union + eps)  # iou
    return tf.reduce_mean(iou)  # mean over batch


def _tuning_metrics():
    """metrics evaluated during tuning (simple and robust)."""  #  we keep the set minimal
    return [
        tf.keras.metrics.BinaryAccuracy(name="bin_acc", threshold=0.5),  # thresholded accuracy
        _safe_dice,  # primary target proxy
        _safe_iou,   # complementary metric
    ]  # this keeps the objective aligned with selection


def _compile_with_optimizer(model, opt_name, lr, wd, conf, loss_name, a, b):
    """
    build or adapt an optimizer, then compile with a nan-safe tuning loss.
    the loss is BCE + (1 - Dice) to balance overlap and stability.  
    """
    opt_obj = None  # placeholder for optimizer

    # try the project optimizer factory first (allows schedulers etc.)
    try:
        maybe_opt = get_optimizer(
            _default(opt_name, getattr(conf, "optimizer_fn", "adam")),  # optimizer name
            _default(getattr(conf, "num_epochs", 50), 50),  # epochs for schedulers
            _default(getattr(conf, "num_training_steps", 100), 100),  # steps for schedulers
        )
        opt_obj = maybe_opt  # could be an instance or a string
    except Exception:
        opt_obj = None  # fallback to manual construction

    # build a standard optimizer if needed
    if isinstance(opt_obj, str) or opt_obj is None:
        name = (opt_name or "adam").lower()  # normalize
        if name == "adamw":
            AdamW = getattr(tf.keras.optimizers, "AdamW", None)  # tf >= 2.11
            if AdamW is not None:
                opt_obj = AdamW(learning_rate=lr, weight_decay=_default(wd, 1e-6), clipnorm=1.0)  # with clipnorm
            else:
                opt_obj = tf.keras.optimizers.Adam(learning_rate=lr, clipnorm=1.0)  # fallback if AdamW missing
        elif name == "rmsprop":
            opt_obj = tf.keras.optimizers.RMSprop(learning_rate=lr, clipnorm=1.0)  # rmsprop baseline
        else:
            opt_obj = tf.keras.optimizers.Adam(learning_rate=lr, clipnorm=1.0)  # default
    else:
        # if factory returned an object, align lr/wd where applicable
        if hasattr(opt_obj, "learning_rate"):
            try: opt_obj.learning_rate = lr  # set lr
            except Exception: pass
        if (opt_name == "adamw") and hasattr(opt_obj, "weight_decay") and wd is not None:
            try: opt_obj.weight_decay = wd  # set weight decay
            except Exception: pass
        try:
            if not hasattr(opt_obj, "clipnorm"):
                opt_obj.clipnorm = 1.0  # gradient safety cap
        except Exception:
            pass

    # define the tuning loss
    bce = tf.keras.losses.BinaryCrossentropy()  # stable BCE

    def tuning_loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)  # cast
        y_pred = tf.cast(y_pred, tf.float32)  # cast
        y_true = _nan_to_num(y_true, 0.0)  # finite labels
        y_pred = _nan_to_num(y_pred, 0.5)  # finite preds
        y_true = tf.clip_by_value(y_true, 0.0, 1.0)  # bound labels
        y_pred = tf.clip_by_value(y_pred, 1e-6, 1.0 - 1e-6)  # avoid log(0) in BCE
        return bce(y_true, y_pred) + (1.0 - _safe_dice(y_true, y_pred))  # composite objective

    model.compile(optimizer=opt_obj, loss=tuning_loss, metrics=_tuning_metrics())  # compile model
    return model  # ready for tuner


# builders — hyperband (hb)
# here we define model builders for the hb phase (discrete choices).

def _build_unet_hb(hp: kt.HyperParameters, conf, tune_batch, patch_h, patch_w):
    """unet builder for hyperband (discrete model/data/optimizer)."""  # here we explore broadly
    opt, lr, wd = _optimizer_space_hb(hp)  # optimizer knobs
    loss_name, (a, b) = _loss_space_any(
        hp, _default(getattr(conf, "loss_fn", "tversky"), "tversky"),
        _default(getattr(conf, "tversky_alphabeta", (0.5, 0.5)), (0.5, 0.5))
    )  # loss knobs
    dilation, layer_cnt, l2w, drp = _unet_space_hb(hp)  # architecture knobs

    input_shape = [tune_batch, patch_h, patch_w, len(conf.channel_list)]  # static shape for clarity
    num_classes = 1  # binary segmentation

    model = UNet(input_shape, num_classes,
                 dilation_rate=dilation, layer_count=layer_cnt,
                 l2_weight=l2w, dropout=drp)  # instantiate unet
    return _compile_with_optimizer(model, opt, lr, wd, conf, loss_name, a, b)  # compile and return


def _build_swin_hb(hp: kt.HyperParameters, conf, tune_batch, H, W, fixed_ps, fixed_ws):
    """swin builder for hyperband with fixed patch/window sizes."""  # here we keep shapes valid across trials
    opt, lr, wd = _optimizer_space_hb(hp)  # optimizer knobs
    loss_name, (a, b) = _loss_space_any(
        hp, _default(getattr(conf, "loss_fn", "tversky"), "tversky"),
        _default(getattr(conf, "tversky_alphabeta", (0.5, 0.5)), (0.5, 0.5))
    )  # loss knobs

    C         = hp.Choice("C", [48, 64, 96])  # channels
    ss_half   = hp.Choice("use_shift", [0, 1])  # shift toggle
    attn_drop = hp.Choice("attn_drop", [0.0, 0.1])  # attention dropout
    proj_drop = hp.Choice("proj_drop", [0.0, 0.1])  # projection dropout
    mlp_drop  = hp.Choice("mlp_drop",  [0.0, 0.1, 0.2])  # mlp dropout
    drop_path = hp.Choice("drop_path", [0.0, 0.1, 0.2])  # stochastic depth

    ss_size = 0 if ss_half == 0 else fixed_ws // 2  # derived shift amount

    ch = len(conf.channels_used) if hasattr(conf, "channels_used") else len(conf.channel_list)  # input channels
    model = SwinUNet(H=H, W=W, ch=ch, C=C,
                     patch_size=fixed_ps, window_size=fixed_ws, ss_size=ss_size,
                     attn_drop=attn_drop, proj_drop=proj_drop, mlp_drop=mlp_drop, drop_path=drop_path)  # instantiate
    return _compile_with_optimizer(model, opt, lr, wd, conf, loss_name, a, b)  # compile and return


# builders — bayesian optimization (bo)
# here we reuse hb winners for architecture and refine continuous parameters.

def _build_unet_bo(hp: kt.HyperParameters, conf, tune_batch, patch_h, patch_w, fixed: Dict[str, Any]):
    """unet builder for bo using fixed arch; refines lr/wd/loss params."""  # here we fine-tune around a good region
    opt, lr, wd = _optimizer_space_bo(hp, fixed["optimizer"])  # lr/wd refinement
    loss_name, (a, b) = _loss_space_bo(
        hp, _default(getattr(conf, "loss_fn", "tversky"), "tversky"),
        _default(getattr(conf, "tversky_alphabeta", (0.5, 0.5)), (0.5, 0.5))
    )  # loss refinement

    input_shape = [tune_batch, patch_h, patch_w, len(conf.channel_list)]  # data shape
    num_classes = 1  # binary

    model = UNet(input_shape, num_classes,
                 dilation_rate=fixed.get("dilation_rate", 1),
                 layer_count=fixed.get("layer_count", 64),
                 l2_weight=fixed.get("l2_weight", 1e-4),
                 dropout=fixed.get("dropout", 0.0))  # architecture fixed from hb
    return _compile_with_optimizer(model, opt, lr, wd, conf, loss_name, a, b)  # compile and return


def _build_swin_bo(hp: kt.HyperParameters, conf, tune_batch, H, W, fixed: Dict[str, Any]):
    """swin builder for bo with arch fixed; refines lr/wd/loss params."""  # here we keep shapes stable
    opt, lr, wd = _optimizer_space_bo(hp, fixed["optimizer"])  # optimizer refinement
    loss_name, (a, b) = _loss_space_bo(
        hp, _default(getattr(conf, "loss_fn", "tversky"), "tversky"),
        _default(getattr(conf, "tversky_alphabeta", (0.5, 0.5)), (0.5, 0.5))
    )  # loss refinement

    ch = len(conf.channels_used) if hasattr(conf, "channels_used") else len(conf.channel_list)  # input channels
    model = SwinUNet(H=H, W=W, ch=ch,
                     C=fixed.get("C", 64),
                     patch_size=fixed.get("patch_size", 16),
                     window_size=fixed.get("window_size", 4),
                     ss_size=fixed.get("ss_size", 2),
                     attn_drop=fixed.get("attn_drop", 0.0),
                     proj_drop=fixed.get("proj_drop", 0.0),
                     mlp_drop=fixed.get("mlp_drop", 0.0),
                     drop_path=fixed.get("drop_path", 0.1))  # architecture from hb
    return _compile_with_optimizer(model, opt, lr, wd, conf, loss_name, a, b)  # compile and return


# one-phase runner
# here we run a single tuning phase (hb or bo) with a shared objective.

def _run_phase(conf, model_type: str, phase: str, project_dir: str,
               tune_batch: int, max_epochs: int, steps: int, val_steps: int,
               executions_per_trial: int, overwrite: bool,
               hb_data_hp: Dict[str, Any]) -> Tuple[Dict[str, Any], kt.engine.tuner.Tuner]:
    """
    run one tuning phase with objective 'val__safe_dice'.                                 
    include a local oracle patch to avoid None-type errors from keras-tuner.   
    """
    # apply a local patch in case the global one wasn't called
    def _patch_kerastuner_oracle_local():
        try:
            from keras_tuner.engine import oracle as kt_oracle
        except Exception:
            import kerastuner.engine.oracle as kt_oracle  # type: ignore
        if getattr(kt_oracle.Oracle, "_pd_safe_patached", False):
            return  # already patched

        def _safe_check(self, *args, **kwargs):
            cf = getattr(self, "_consecutive_failures", 0) or 0  # normalize
            try: setattr(self, "_consecutive_failures", cf)
            except Exception: pass
            limit = getattr(self, "max_consecutive_failed_trials", 3) or 3  # default cap
            try: setattr(self, "max_consecutive_failed_trials", limit)
            except Exception: pass
            if int(cf) >= int(limit):
                raise RuntimeError(f"Number of consecutive failures excceeded the limit of {int(limit)}.")  # same semantics
            return None  # match signature

        kt_oracle.Oracle._check_consecutive_failures = _safe_check  # patch
        kt_oracle.Oracle._pd_safe_patached = True  # mark

    _patch_kerastuner_oracle_local()  # ensure patch is in place

    frames = get_all_frames()  # here we gather training frames from the project

    tag = f"{model_type}_tuning"  # base tag for artifacts
    project_name = f"{tag}_{phase}"  # subfolder per phase
    _ensure_dir(project_dir)  # ensure logs dir exists
    objective = kt.Objective("val__safe_dice", direction="max")  # maximize safe dice

    # build datasets and the model builder closure
    if model_type == "swin":
        H_fixed, W_fixed = hb_data_hp["patch_h"], hb_data_hp["patch_w"]  # fixed dims
        train_gen, val_gen = _build_generators_for_tuning(
            frames, conf, tune_batch,
            patch_h=H_fixed, patch_w=W_fixed,
            augmenter_strength=hb_data_hp["augmenter_strength"],
            min_pos_frac=hb_data_hp["min_pos_frac"]
        )  # dataset pipeline for swin

        def builder(hp: kt.HyperParameters):
            if phase == "HB":
                return _build_swin_hb(
                    hp, conf, tune_batch, H_fixed, W_fixed,
                    fixed_ps=hb_data_hp["fixed"]["patch_size"],
                    fixed_ws=hb_data_hp["fixed"]["window_size"]
                )  # hb builder
            else:
                return _build_swin_bo(
                    hp, conf, tune_batch, H_fixed, W_fixed, fixed=hb_data_hp["fixed"]
                )  # bo builder
    else:
        def builder(hp: kt.HyperParameters):
            if phase == "HB":
                patch_h = hp.Choice("patch_h", [256, 384, 512], default=hb_data_hp["patch_h"])  # let hb choose dims
                patch_w = hp.Choice("patch_w", [256, 384, 512], default=hb_data_hp["patch_w"])  # let hb choose dims
                return _build_unet_hb(hp, conf, tune_batch, patch_h, patch_w)  # hb builder
            else:
                patch_h = hb_data_hp["patch_h"]  # fix dims for bo
                patch_w = hb_data_hp["patch_w"]  # fix dims for bo
                return _build_unet_bo(hp, conf, tune_batch, patch_h, patch_w, fixed=hb_data_hp["fixed"])  # bo builder

        train_gen, val_gen = _build_generators_for_tuning(
            frames, conf, tune_batch,
            patch_h=hb_data_hp["patch_h"], patch_w=hb_data_hp["patch_w"],
            augmenter_strength=hb_data_hp["augmenter_strength"],
            min_pos_frac=hb_data_hp["min_pos_frac"]
        )  # dataset pipeline for unet

    # choose tuner by phase
    if phase == "HB":
        tuner = kt.Hyperband(
            builder, objective=objective, max_epochs=max_epochs, factor=3,  # hb schedule
            executions_per_trial=executions_per_trial,  # repeat for stability
            directory=project_dir, project_name=project_name,  # where to save logs
            overwrite=overwrite  # allow fresh hb runs
        )  # hyperband tuner
    else:
        tuner = kt.BayesianOptimization(
            builder, objective=objective,
            max_trials=_default(getattr(conf, "tune_max_trials", None), 30),  # bo budget
            executions_per_trial=executions_per_trial,  # repeat per trial
            directory=project_dir, project_name=project_name,  # where to save logs
            overwrite=True  # fresh bo run
        )  # bayesian tuner

    # normalize oracle counters defensively
    try:
        tuner.oracle._consecutive_failures = 0  # reset failure counter
        if getattr(tuner.oracle, "max_consecutive_failed_trials", None) is None:
            tuner.oracle.max_consecutive_failed_trials = 3  # set a sane limit
    except Exception:
        pass  # non-fatal

    # callbacks for training control
    early = EarlyStopping(monitor="val__safe_dice", mode="max", patience=5, restore_best_weights=True)  # here we stop early
    csv_logger = CSVLogger(os.path.join(project_dir, f"{tag}_{phase}_fit_history.csv"), append=True)  # here we log curves
    nan_killer = tf.keras.callbacks.TerminateOnNaN()  # here we abort on NaNs

    # run the search
    tuner.search(
        train_gen, steps_per_epoch=steps, epochs=max_epochs,  # train loop
        validation_data=val_gen, validation_steps=val_steps,  # validation loop
        callbacks=[early, csv_logger, nan_killer],  # stability and logging
        verbose=1, workers=1  # keep behavior deterministic-ish
    )  # launch tuner

    # collect artifacts
    best_hp = tuner.get_best_hyperparameters(1)[0]  # best trial
    best_dict = {k: best_hp.get(k) for k in best_hp.values.keys()}  # flatten to dict

    _write_trials_csv(tuner, os.path.join(project_dir, f"{tag}_{phase}_all_trials.csv"))  # csv of all trials
    fixed = hb_data_hp.get("fixed", None)  # frozen settings (if any)
    _write_summary_md(project_dir, tag, tuner, phase, max_epochs, steps, val_steps, tune_batch, fixed=fixed)  # md summary
    best_path = _save_best(project_dir, tag, phase, best_hp)  # json with best hparams
    print(f"[{tag} {phase}] best saved to: {best_path}")  # console breadcrumb

    return best_dict, tuner  # return best params for chaining


# chained tuner (hb --> bo)
# here we run hyperband first (broad), then bayesian optimization (refinement), and merge results.

def _tune_chained(conf, model_type: str = "unet",
                  executions_per_trial: int = 1, overwrite: bool = False) -> Dict[str, Any]:
    """
    run a two-stage tuning process: hyperband then bayesian optimization.
    this returns a merged dict of the final best settings. 
    """
    global config  # expose config if other modules expect it
    config = conf  # assign incoming config
    print("Starting chained tuning (HyperBand + Bayesian).")  # status message

    import training as training_mod  # late import to avoid circular deps
    training_mod.config = conf  # share config with training utilities
    _seed(_default(getattr(config, "seed", None), 42))  # set seeds for reproducibility
    tf.config.run_functions_eagerly(False)  # default graph mode

    hb_epochs = _default(getattr(config, "tune_num_epochs", None), 20)  # hb epochs
    bo_epochs = _default(getattr(config, "tune_num_epochs_bo", None), hb_epochs)  # bo epochs (defaults to hb)
    steps     = _default(getattr(config, "tune_steps_per_epoch", None),
                         min(100, _default(getattr(config, "num_training_steps", 100), 100)))  # cap steps
    val_steps = _default(getattr(config, "tune_validation_steps", None),
                         min(50, _default(getattr(config, "num_validation_images", 50), 50)))  # cap val steps
    tune_batch = _default(getattr(config, "tune_batch_size", None),
                          min(8, _default(getattr(config, "train_batch_size", 8), 8)))  # keep in-memory

    logs_dir = _default(getattr(config, "logs_dir", "./logs"), "./logs")  # logs root
    key = "unet" if model_type.lower() == "unet" else "swin"  # normalize label
    project_dir = os.path.join(logs_dir, f"{key}_tuning")  # per-model logs dir
    _ensure_dir(project_dir)  # ensure the folder exists

    # phase 1: hyperband
    hb_aug    = _default(getattr(config, "augmenter_strength", None), 1.0)  # default aug on
    hb_minpos = _default(getattr(config, "min_pos_frac", None), 0.0)  # no min-pos by default

    if key == "swin":
        # here we keep patch/window fixed during hb to avoid window misalignment
        fixed_ps = _default(getattr(config, "swin_patch_size", None), 8)  # typical: 8 or 16
        fixed_ws = _default(getattr(config, "swin_window", None), 4)  # typical: 4/7/8
        base_H = 384; base_W = 384  # nominal size
        hb_patch_h, hb_patch_w = _snap_hw_for_swin(base_H, base_W, fixed_ps, fixed_ws, down_levels=3)  # valid sizes

        hb_data_hp = dict(
            patch_h=hb_patch_h,
            patch_w=hb_patch_w,
            augmenter_strength=hb_aug,
            min_pos_frac=hb_minpos,
            fixed=dict(patch_size=fixed_ps, window_size=fixed_ws),  # frozen hb arch bits
        )  # hb data config for swin
    else:
        hb_patch_h = 384; hb_patch_w = 384  # common patch sizes for unet
        hb_data_hp = dict(
            patch_h=hb_patch_h,
            patch_w=hb_patch_w,
            augmenter_strength=hb_aug,
            min_pos_frac=hb_minpos
        )  # hb data config for unet

    hb_best, _ = _run_phase(
        config, key, "HB", project_dir,
        tune_batch, hb_epochs, steps, val_steps,
        executions_per_trial, overwrite, hb_data_hp
    )  # broad discrete search

    # build fixed set for bo from hb results
    fixed = {"optimizer": hb_best.get("optimizer", _default(getattr(config, "optimizer_fn", "adam"), "adam"))}  # optimizer
    if key == "unet":
        fixed["dilation_rate"] = hb_best.get("dilation_rate", _default(getattr(config, "dilation_rate", 1), 1))  # arch
        fixed["layer_count"]   = hb_best.get("layer_count", 64)  # arch
        fixed["l2_weight"]     = float(hb_best.get("l2_weight", 1e-4))  # regularization
        fixed["dropout"]       = float(hb_best.get("dropout", 0.0))  # regularization
        bo_patch_h = int(hb_best.get("patch_h", hb_patch_h))  # selected height
        bo_patch_w = int(hb_best.get("patch_w", hb_patch_w))  # selected width
    else:
        fixed["C"]             = hb_best.get("C", _default(getattr(config, "swin_base_C", 64), 64))  # arch
        fixed["patch_size"]    = hb_data_hp["fixed"]["patch_size"]  # keep hb fixed
        fixed["window_size"]   = hb_data_hp["fixed"]["window_size"]  # keep hb fixed
        fixed["ss_size"]       = (fixed["window_size"] // 2) if int(hb_best.get("use_shift", 1)) == 1 else 0  # derived shift
        fixed["attn_drop"]     = float(hb_best.get("attn_drop", 0.0))  # regularization
        fixed["proj_drop"]     = float(hb_best.get("proj_drop", 0.0))  # regularization
        fixed["mlp_drop"]      = float(hb_best.get("mlp_drop", 0.0))  # regularization
        fixed["drop_path"]     = float(hb_best.get("drop_path", 0.1))  # regularization
        bo_patch_h = int(hb_data_hp["patch_h"])  # keep the hb size
        bo_patch_w = int(hb_data_hp["patch_w"])  # keep the hb size

    # phase 2: bayesian optimization
    bo_data_hp = dict(
        patch_h=bo_patch_h,
        patch_w=bo_patch_w,
        augmenter_strength=hb_aug,
        min_pos_frac=hb_minpos,
        fixed=fixed
    )  # bo config

    bo_best, _ = _run_phase(
        config, key, "BO", project_dir,
        tune_batch, bo_epochs, steps, val_steps,
        executions_per_trial, overwrite, bo_data_hp
    )  # refine continuous parameters

    # merge final settings
    final = dict(fixed)  # start with fixed values
    if "learning_rate" in bo_best: final["learning_rate"] = float(bo_best["learning_rate"])  # tuned lr
    if "weight_decay" in bo_best:  final["weight_decay"]  = float(bo_best["weight_decay"])  # tuned wd
    if "tversky_alpha" in bo_best: final["tversky_alpha"] = float(bo_best["tversky_alpha"])  # tuned alpha
    if "tversky_beta"  in bo_best: final["tversky_beta"]  = float(bo_best["tversky_beta"])  # tuned beta
    final["patch_h"] = bo_patch_h  # final height
    final["patch_w"] = bo_patch_w  # final width
    final["augmenter_strength"] = float(hb_aug)  # final aug intensity
    final["min_pos_frac"] = float(hb_minpos)  # final sampling constraint

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")  # timestamp for file naming
    final_path = os.path.join(project_dir, f"{stamp}_{key}_tuning_FINAL_best_hparams.json")  # output file
    with open(final_path, "w") as f:
        json.dump(final, f, indent=2)  # write merged config
    print(f"[{key} tuning] FINAL merged best saved to: {final_path}")  # console message
    print(json.dumps(final, indent=2))  # readable dump
    return final  # return best settings


# public entry points
# here we expose simple functions for external callers.

def tune_UNet(conf) -> Dict[str, Any]:
    """run hb-->bo tuning for unet and return the final best params dict."""  # here we provide a single-call api
    return _tune_chained(conf, model_type="unet")  # delegate


def tune_SwinUNetPP(conf) -> Dict[str, Any]:
    """run hb-->bo tuning for swin-unet and return the final best params dict."""  # here we mirror the unet api
    return _tune_chained(conf, model_type="swin")  # delegate


def apply_best_to_config(conf, best: Dict[str, Any], model_type: str):
    """
    apply the best hyperparameters to a config object in-place.          
    this maps fields from the best-params dict onto the config attributes.           
    """
    # optimizer / schedule
    if "optimizer" in best:
        conf.optimizer_fn = best["optimizer"]  # set optimizer family
    if "learning_rate" in best:
        conf.learning_rate = best["learning_rate"]  # set lr
    if "weight_decay" in best:
        conf.weight_decay = best["weight_decay"]  # set wd if applicable

    # loss (tversky family)
    if "tversky_alpha" in best or "tversky_beta" in best:
        a = best.get("tversky_alpha", getattr(conf, "tversky_alphabeta", (0.5, 0.5))[0])  # keep default if missing
        b = best.get("tversky_beta",  getattr(conf, "tversky_alphabeta", (0.5, 0.5))[1])  # keep default if missing
        conf.tversky_alphabeta = (a, b)  # update tuple

    # data / generator
    conf.augmenter_strength = best.get("augmenter_strength", getattr(conf, "augmenter_strength", 1.0))  # aug intensity
    conf.min_pos_frac       = best.get("min_pos_frac", getattr(conf, "min_pos_frac", 0.0))  # min-pos constraint
    conf.patch_size         = [int(best.get("patch_h", conf.patch_size[0])),
                               int(best.get("patch_w", conf.patch_size[1]))]  # patch dims

    # model-specific
    if model_type.lower() == "unet":
        conf.dilation_rate = best.get("dilation_rate", getattr(conf, "dilation_rate", 1))  # unet arch
        conf.layer_count   = best.get("layer_count", getattr(conf, "layer_count", 64))  # unet arch
        conf.l2_weight     = best.get("l2_weight", getattr(conf, "l2_weight", 1e-4))  # regularization
        conf.dropout       = best.get("dropout", getattr(conf, "dropout", 0.0))  # regularization
    else:
        conf.swin_base_C     = best.get("C", getattr(conf, "swin_base_C", 64))  # swin arch
        conf.swin_patch_size = best.get("patch_size", getattr(conf, "swin_patch_size", 16))  # swin arch
        conf.swin_window     = best.get("window_size", getattr(conf, "swin_window", 4))  # swin arch
        conf.swin_ss_size    = best.get("ss_size", getattr(conf, "swin_ss_size", 2))  # shift size
        conf.swin_attn_drop  = best.get("attn_drop", getattr(conf, "swin_attn_drop", 0.0))  # regularization
        conf.swin_proj_drop  = best.get("proj_drop", getattr(conf, "swin_proj_drop", 0.0))  # regularization
        conf.swin_mlp_drop   = best.get("mlp_drop", getattr(conf, "swin_mlp_drop", 0.0))  # regularization
        conf.swin_drop_path  = best.get("drop_path", getattr(conf, "swin_drop_path", 0.1))  # regularization
