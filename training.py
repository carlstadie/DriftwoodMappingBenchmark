import os
import json
import time
import glob
import shutil
from datetime import datetime, timedelta

import h5py
import rasterio
import numpy as np
import pandas as pd
from tqdm import tqdm

import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, Callback, CSVLogger

# ===== Mixed precision + XLA + fast execution defaults =====
from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy("mixed_float16")
tf.config.optimizer.set_jit(True)  # XLA JIT
tf.config.run_functions_eagerly(False)
tf.config.experimental.enable_tensor_float_32_execution(True)

# ===== Your project imports =====
from core.UNet import UNet
from core.Swin_UNetPP import SwinUNet
from core.frame_info import FrameInfo
from core.optimizers import get_optimizer
from core.split_frames import split_dataset
from core.dataset_generator import DataGenerator as Generator

# Metrics / losses
from core.losses import (
    accuracy, dice_coef, dice_loss, specificity, sensitivity, f_beta, f1_score, IoU,
    nominal_surface_distance, Hausdorff_distance, boundary_intersection_over_union, get_loss
)

# -----------------------------
# Helpers: data, datasets, callbacks, heavy-metric eval
# -----------------------------

def get_all_frames():
    """Get all pre-processed frames which will be used for training."""
    # Use most recent preprocessed dir if none given
    if config.preprocessed_dir is None:
        config.preprocessed_dir = os.path.join(
            config.preprocessed_base_dir,
            sorted(os.listdir(config.preprocessed_base_dir))[-1]
        )

    # Get paths of preprocessed images
    image_paths = sorted(
        glob.glob(f"{config.preprocessed_dir}/*.tif"),
        key=lambda f: int(f.split("/")[-1][:-4])
    )
    print(f"Found {len(image_paths)} input frames in {config.preprocessed_dir}")

    frames = []
    for im_path in tqdm(image_paths, desc="Processing frames"):
        preprocessed = rasterio.open(im_path).read()  # C|H|W

        # image channels are all except last (assuming last is label; weights removed if present)
        image_channels = preprocessed[:-1, ::]                # C|H|W
        image_channels = np.transpose(image_channels, (1, 2, 0))  # H|W|C

        annotations = preprocessed[-1, ::]                    # H|W
        frames.append(FrameInfo(image_channels, annotations))

    return frames


def create_train_val_datasets(frames):
    """ Create training / validation / test sets and build generators (optionally tf.data). """
    frames_json = os.path.join(config.preprocessed_dir, "aa_frames_list.json")
    training_frames, validation_frames, test_frames = split_dataset(
        frames, frames_json, config.test_ratio, config.val_ratio
    )

    # Channels
    input_channels = list(range(len(config.channel_list)))
    label_channel = len(config.channel_list)  # label directly after inputs

    # Patch size (allow tuner to override)
    patch_h, patch_w = config.patch_size
    if hasattr(config, "tune_patch_h") and hasattr(config, "tune_patch_w") and config.tune_patch_h and config.tune_patch_w:
        patch_h, patch_w = int(config.tune_patch_h), int(config.tune_patch_w)

    # H*W*(inputs+1 label)
    patch_size = [patch_h, patch_w, len(config.channel_list) + 1]

    # Generator knobs
    aug_strength = getattr(config, "augmenter_strength", 1.0)
    min_pos_frac = getattr(config, "min_pos_frac", 0.0)

    # Build python generators
    train_gen = Generator(
        input_channels, patch_size, training_frames, frames, label_channel,
        augmenter='iaa', augmenter_strength=aug_strength, min_pos_frac=min_pos_frac
    ).random_generator(config.train_batch_size)

    val_gen = Generator(
        input_channels, patch_size, validation_frames, frames, label_channel,
        augmenter=None, augmenter_strength=1.0, min_pos_frac=0.0
    ).random_generator(config.train_batch_size)

    test_gen = Generator(
        input_channels, patch_size, test_frames, frames, label_channel,
        augmenter=None, augmenter_strength=1.0, min_pos_frac=0.0
    ).random_generator(config.train_batch_size)

    if getattr(config, "use_tf_data", True):
        # Wrap in tf.data for speed
        x_channels = len(config.channel_list)
        x_shape = (config.train_batch_size, patch_h, patch_w, x_channels)
        y_shape = (config.train_batch_size, patch_h, patch_w, 1)

        output_signature = (
            tf.TensorSpec(shape=(None, patch_h, patch_w, x_channels), dtype=tf.float32),
            tf.TensorSpec(shape=(None, patch_h, patch_w, 1), dtype=tf.float32),
        )

        def make_ds(py_gen):
            ds = tf.data.Dataset.from_generator(lambda: py_gen, output_signature=output_signature)
            # (optional) enable randomness for speed, prefetch to overlap producer/consumer
            opts = tf.data.Options()
            opts.deterministic = False
            ds = ds.with_options(opts)
            ds = ds.prefetch(tf.data.AUTOTUNE)
            return ds

        train_ds = make_ds(train_gen)
        val_ds   = make_ds(val_gen)
        test_ds  = make_ds(test_gen)
    else:
        train_ds, val_ds, test_ds = train_gen, val_gen, test_gen

    return train_ds, val_ds, test_ds


def create_callbacks(model_path, log_suffix=""):
    """Define callbacks for checkpointing, TB logging, and custom metadata."""
    # Logging dir
    log_dir = os.path.join(config.logs_dir, os.path.basename(model_path) + log_suffix)
    os.makedirs(log_dir, exist_ok=True)

    # Checkpointing: default to weights-only (fast). Allow override to save full model each time.
    save_full_each = getattr(config, "save_full_model_each_checkpoint", False)
    checkpoint = ModelCheckpoint(
        filepath=(model_path if save_full_each else f"{model_path}.weights.h5"),
        monitor='val_loss',
        verbose=1,
        save_best_only=True,
        mode='min',
        save_weights_only=(not save_full_each)
    )

    # TensorBoard (no profiling during normal runs)
    tensorboard = TensorBoard(
        log_dir=log_dir,
        histogram_freq=0,
        write_graph=True,
        write_images=False,
        profile_batch=0
    )

    class CustomMeta(Callback):
        def __init__(self):
            super().__init__()
            self.start_time = datetime.now()

        def on_epoch_end(self, epoch, logs=None):
            l = logs or {}
            # Construct meta dict (keep extensive fields)
            meta_data = {
                "name": config.model_name,
                "model_path": model_path,
                "patch_size": tuple(config.patch_size),
                "channels_used": getattr(config, "channels_used", getattr(config, "channel_list", [])),
                "resample_factor": getattr(config, "resample_factor", None),
                "frames_dir": config.preprocessed_dir,
                "train_ratio": float(f"{1-config.val_ratio-config.test_ratio:.2f}"),
                "val_ratio": config.val_ratio,
                "test_ratio": config.test_ratio,
                "loss": config.loss_fn,
                "optimizer": config.optimizer_fn,
                "tversky_alpha": getattr(config, "tversky_alphabeta", (None, None))[0],
                "tversky_beta": getattr(config, "tversky_alphabeta", (None, None))[1],
                "batch_size": config.train_batch_size,
                "epoch_steps": config.num_training_steps,
                "val_steps": config.num_validation_images,
                "epochs_trained": f"{epoch + 1}/{config.num_epochs}",
                "total_epochs": config.num_epochs,
                # last metrics (some may not be present if not tracked in model.fit)
                "last_sensitivity": float(f"{l.get('sensitivity', float('nan')):.6f}") if 'sensitivity' in l else None,
                "last_specificity": float(f"{l.get('specificity', float('nan')):.6f}") if 'specificity' in l else None,
                "last_dice_coef": float(f"{l.get('dice_coef', float('nan')):.6f}") if 'dice_coef' in l else None,
                "last_dice_loss": float(f"{l.get('dice_loss', float('nan')):.6f}") if 'dice_loss' in l else None,
                "last_accuracy": float(f"{l.get('accuracy', float('nan')):.6f}") if 'accuracy' in l else None,
                "last_f_beta": float(f"{l.get('f_beta', float('nan')):.6f}") if 'f_beta' in l else None,
                "last_f1_score": float(f"{l.get('f1_score', float('nan')):.6f}") if 'f1_score' in l else None,
                "last_IoU": float(f"{l.get('IoU', float('nan')):.6f}") if 'IoU' in l else None,
                "last_nominal_surface_distance": float(f"{l.get('nominal_surface_distance', float('nan')):.6f}") if 'nominal_surface_distance' in l else None,
                "last_Hausdorff_distance": float(f"{l.get('Hausdorff_distance', float('nan')):.6f}") if 'Hausdorff_distance' in l else None,
                "last_boundary_intersection_over_union": float(f"{l.get('boundary_intersection_over_union', float('nan')):.6f}") if 'boundary_intersection_over_union' in l else None,
                "start_time": self.start_time.strftime("%d.%m.%Y %H:%M:%S"),
                "elapsed_time": (datetime.utcfromtimestamp(0) + (datetime.now() - self.start_time)).strftime("%H:%M:%S")
            }
            meta_path = f"{model_path}.metadata.json"
            with open(meta_path, 'w') as f:
                json.dump(meta_data, f, indent=4)

            # Optional periodic extra save
            if getattr(config, "model_save_interval", None) and (epoch + 1) % config.model_save_interval == 0:
                checkpoint.on_epoch_end(epoch, logs=l)

    return [checkpoint, tensorboard, CustomMeta()], log_dir


class MetricsCSVCallback(Callback):
    """Append all logs each epoch to a CSV in the run's log dir."""
    def __init__(self, csv_path):
        super().__init__()
        self.csv_path = csv_path
        self.metrics_df = None

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        if self.metrics_df is None:
            self.metrics_df = pd.DataFrame()
        new_row = pd.DataFrame(logs, index=[epoch])
        self.metrics_df = pd.concat([self.metrics_df, new_row], ignore_index=True)
        self.metrics_df.to_csv(self.csv_path, index=False)


class EvalHeavyMetrics(Callback):
    """
    Compute heavy metrics on a small validation subset at each epoch end.
    Writes to TensorBoard and merges into Keras logs (so CSVLogger captures them).
    """
    def __init__(self, val_ds_for_eval, log_dir, steps=50, threshold=0.5):
        super().__init__()
        self.val_eval = val_ds_for_eval.take(steps) if hasattr(val_ds_for_eval, "take") else val_ds_for_eval
        self.steps = steps
        self.threshold = threshold
        self.tb_writer = tf.summary.create_file_writer(os.path.join(log_dir, "heavy_metrics"))

        # Heavy metrics to compute
        self.metric_fns = {
            "specificity": specificity,
            "sensitivity": sensitivity,
            "f_beta": f_beta,
            "f1_score": f1_score,
            "IoU": IoU,
            "nominal_surface_distance": nominal_surface_distance,
            "Hausdorff_distance": Hausdorff_distance,
            "boundary_intersection_over_union": boundary_intersection_over_union,
            "dice_loss": dice_loss,   # for reference
        }

    def _as_float(self, v):
        # convert tensor to python float
        try:
            return float(tf.reduce_mean(v).numpy())
        except Exception:
            return float(v)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        # Iterate over a small validation subset
        accum = {k: [] for k in self.metric_fns.keys()}
        seen = 0
        for batch, (x, y_true) in enumerate(self.val_eval):
            y_pred = self.model(x, training=False)
            # ensure predictions are float32 for metrics (avoid fp16 issues)
            y_pred = tf.cast(y_pred, tf.float32)
            y_true = tf.cast(y_true, tf.float32)

            # Optional binarization if metrics expect masks; many metrics can handle probs.
            y_bin = tf.cast(y_pred >= self.threshold, tf.float32)

            for name, fn in self.metric_fns.items():
                try:
                    # Try probability version first; fallback to binarized if needed
                    val = fn(y_true, y_pred)
                except Exception:
                    val = fn(y_true, y_bin)
                accum[name].append(self._as_float(val))
            seen += 1

        # Mean over evaluated batches
        with self.tb_writer.as_default():
            for name, values in accum.items():
                if values:
                    mean_val = sum(values) / len(values)
                    tf.summary.scalar(name, mean_val, step=epoch)
                    logs[f"val_{name}"] = mean_val  # join into Keras logs so CSVLogger captures it
        self.tb_writer.flush()


# -----------------------------
# Training entry points
# -----------------------------

def _build_and_compile_model_unet():
    # Build UNet
    model = UNet([config.train_batch_size, *config.patch_size, len(config.channel_list)],
                 [len(config.channel_list)], config.dilation_rate)

    # Light metrics during training for speed
    light_metrics = [dice_coef, accuracy]
    model.compile(
        optimizer=get_optimizer(config.optimizer_fn, config.num_epochs, config.num_training_steps),
        loss=get_loss(config.loss_fn, getattr(config, "tversky_alphabeta", (0.5, 0.5))),
        metrics=light_metrics,
        steps_per_execution=getattr(config, "steps_per_execution", 32)
    )
    return model


def _build_and_compile_model_swin():
    # Build SwinUNetPP (allow reducing C to speed up if desired)
    base_C = getattr(config, "swin_base_channels", 64)
    swin_patch_size = getattr(config, "swin_patch_size", 16)

    model = SwinUNet(
        H=config.patch_size[0],
        W=config.patch_size[1],
        ch=len(getattr(config, "channels_used", config.channel_list)),
        C=base_C,
        patch_size=swin_patch_size
    )

    light_metrics = [dice_coef, accuracy]
    model.compile(
        optimizer=get_optimizer(config.optimizer_fn),
        loss=get_loss(config.loss_fn, getattr(config, "tversky_alphabeta", (0.5, 0.5))),
        metrics=light_metrics,
        steps_per_execution=getattr(config, "steps_per_execution", 32)
    )
    return model


def _fit_model(model, train_ds, val_ds, model_path, starting_epoch=0, log_name=""):
    # Callbacks
    callbacks, log_dir = create_callbacks(model_path, log_suffix=log_name)

    # CSV logs (epoch-level)
    csv_path = os.path.join(config.logs_dir, f'{os.path.basename(model_path)}_metrics.csv')
    csv_logger = CSVLogger(csv_path, separator=',', append=True)

    # Heavy metrics evaluator on a subset of validation data
    val_eval_steps = getattr(config, "heavy_eval_steps", 50)
    heavy_eval_cb = EvalHeavyMetrics(val_ds, log_dir, steps=val_eval_steps, threshold=getattr(config, "eval_threshold", 0.5))

    # Choose fit inputs: tf.data or python generators
    fit_kwargs = dict(
        steps_per_epoch=config.num_training_steps,
        epochs=config.num_epochs,
        initial_epoch=starting_epoch,
        validation_data=val_ds,
        validation_steps=config.num_validation_images,
        callbacks=[*callbacks, csv_logger, heavy_eval_cb]
    )

    if getattr(config, "use_tf_data", True):
        # Best path: tf.data (already prefetched)
        model.fit(train_ds, **fit_kwargs)
    else:
        # Legacy path: python generator with multiprocessing workers
        model.fit(
            train_ds,
            **fit_kwargs,
            workers=getattr(config, "fit_workers", 8),
            use_multiprocessing=True,
            max_queue_size=getattr(config, "fit_max_queue_size", 32)
        )

    # End of training: export full SavedModel once (fast path saved weights during training)
    final_export_path = model_path if getattr(config, "save_full_model_each_checkpoint", False) else f"{model_path}.savedmodel"
    try:
        model.save(final_export_path)
    except Exception as e:
        print(f"Warning: final model.save failed ({e}). Attempting to save weights-only.")
        model.save_weights(f"{model_path}.final.weights.h5")

    print("Training completed.\n")


def _prepare_model_and_logging(model_path):
    """Handle continue-from checkpoint and carry over logs; return (model, starting_epoch, carried_log_dir?)."""
    starting_epoch = 0
    model = None

    if config.continue_model_path is not None:
        print(f"Loading pre-trained model from {config.continue_model_path} :")
        # Try to load full model (best), else load weights into a freshly built model
        try:
            model = tf.keras.models.load_model(
                config.continue_model_path,
                custom_objects={
                    'tversky': get_loss('tversky', getattr(config, "tversky_alphabeta", (0.5, 0.5))),
                    'dice_coef': dice_coef, 'dice_loss': dice_loss,
                    'accuracy': accuracy, 'specificity': specificity,
                    'sensitivity': sensitivity, 'f_beta': f_beta, 'f1': f1_score, 'IoU': IoU,
                    'nominal_surface_distance': nominal_surface_distance,
                    'Hausdorff_distance': Hausdorff_distance,
                    'boundary_intersection_over_union': boundary_intersection_over_union
                },
                compile=False
            )
        except Exception as e:
            print(f"Could not load full model ({e}). Will attempt weights-only load later.")

        # Starting epoch from metadata (if exists)
        try:
            with h5py.File(config.continue_model_path, 'r') as model_file:
                if "custom_meta" in model_file.attrs:
                    try:
                        custom_meta = json.loads(model_file.attrs["custom_meta"].decode("utf-8"))
                    except Exception:
                        custom_meta = json.loads(model_file.attrs["custom_meta"])
                    starting_epoch = int(custom_meta["epochs_trained"].split("/")[0])
        except Exception:
            pass

        # Copy logs forward so TB shows a continuous curve
        old_log_dir = os.path.join(config.logs_dir, os.path.basename(config.continue_model_path).split(".")[0])
        new_log_dir = os.path.join(config.logs_dir, os.path.basename(model_path))
        if os.path.exists(old_log_dir) and not os.path.exists(new_log_dir):
            try:
                shutil.copytree(old_log_dir, new_log_dir)
            except Exception:
                pass

    return model, starting_epoch


# -----------------------------
# Public training functions
# -----------------------------

def train_UNet(conf):
    """Create and train a new UNet model with fast execution and extensive logging."""
    global config
    config = conf
    print("Starting training (UNet).")
    start = time.time()

    # Data
    frames = get_all_frames()
    train_ds, val_ds, test_ds = create_train_val_datasets(frames)

    # Paths
    stamp = time.strftime('%Y%m%d-%H%M')
    model_path = os.path.join(config.saved_models_dir, f"{stamp}_{config.model_name}")

    # Prepare model / resume
    model, starting_epoch = _prepare_model_and_logging(model_path)

    if model is None:
        model = _build_and_compile_model_unet()
    else:
        # Re-compile after loading (use light metrics)
        model.compile(
            optimizer=get_optimizer(config.optimizer_fn, config.num_epochs, config.num_training_steps),
            loss=get_loss(config.loss_fn, getattr(config, "tversky_alphabeta", (0.5, 0.5))),
            metrics=[dice_coef, accuracy],
            steps_per_execution=getattr(config, "steps_per_execution", 32)
        )

    # Fit
    _fit_model(model, train_ds, val_ds, model_path, starting_epoch, log_name="_unet")

    print(f"Training completed in {str(timedelta(seconds=time.time() - start)).split('.')[0]}.\n")


def train_SwinUNetPP(conf):
    """Create and train a new Swin-UNet++ model with fast execution and extensive logging."""
    global config
    config = conf
    print("Starting training (SwinUNetPP).")
    start = time.time()

    # Data
    frames = get_all_frames()
    train_ds, val_ds, test_ds = create_train_val_datasets(frames)

    # Paths
    stamp = time.strftime('%Y%m%d-%H%M')
    model_path = os.path.join(config.saved_models_dir, f"{stamp}_{config.model_name}")

    # Prepare model / resume
    model, starting_epoch = _prepare_model_and_logging(model_path)

    if model is None:
        model = _build_and_compile_model_swin()
    else:
        model.compile(
            optimizer=get_optimizer(config.optimizer_fn),
            loss=get_loss(config.loss_fn, getattr(config, "tversky_alphabeta", (0.5, 0.5))),
            metrics=[dice_coef, accuracy],
            steps_per_execution=getattr(config, "steps_per_execution", 32)
        )

    # Fit
    _fit_model(model, train_ds, val_ds, model_path, starting_epoch, log_name="_swin")

    print(f"Training completed in {str(timedelta(seconds=time.time() - start)).split('.')[0]}.\n")
