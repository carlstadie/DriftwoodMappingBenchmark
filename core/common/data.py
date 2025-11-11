# core/common/data.py
from __future__ import annotations

import glob
import os
from typing import Any, Iterable, Iterator, Optional, Tuple

import numpy as np
import rasterio
import torch
from torch.utils.data import DataLoader, IterableDataset
from tqdm import tqdm

from core.dataset_generator import DataGenerator as Generator
from core.frame_info import FrameInfo
from core.split_frames import split_dataset, summarize_positive_rates

config: Any = None


def get_all_frames(conf: Optional[Any] = None):
    """
    Load all preprocessed frames from disk.

    Each .tif is expected to contain stacked channels with the last channel
    being a mask. Images are returned as HWC for compatibility with the
    downstream Generator which expects that shape.

    Centralises frame discovery and I/O so training code stays focused on
    modeling rather than file system details.

    Args:
        conf: Optional config override to set this module's global config.

    Returns:
        list[FrameInfo]: One entry per frame with (image_channels, annotations).
    """
    global config
    if conf is not None:
        config = conf

    if config is None:
        raise RuntimeError("common.data.get_all_frames called without a config.")

    # Default to the most recent preprocessed directory under the base location.
    if config.preprocessed_dir is None:
        config.preprocessed_dir = os.path.join(
            config.preprocessed_base_dir,
            sorted(os.listdir(config.preprocessed_base_dir))[-1],
        )

    # Files are numbered and sorted by integer stem, e.g., 69.tif
    image_paths = sorted(
        glob.glob(f"{config.preprocessed_dir}/*.tif"),
        key=lambda f: int(os.path.basename(f)[:-4]),
    )
    print(
        f"[DATA][LOAD] Found {len(image_paths)} input frames in "
        f"{config.preprocessed_dir}"
    )

    frames = []
    for im_path in tqdm(image_paths, desc="Processing frames"):
        # Read as CHW, then split inputs vs label channel.
        preprocessed = rasterio.open(im_path).read()  # C,H,W

        image_channels = preprocessed[:-1, ...]  # C,H,W
        image_channels = np.transpose(image_channels, (1, 2, 0))  # H,W,C

        annotations = preprocessed[-1, ...]  # H,W
        frames.append(FrameInfo(image_channels, annotations))

    return frames


class TorchGeneratorDataset(IterableDataset):
    """
    Wrap a python generator of (x, y) batches and convert to NCHW tensors.

    Notes:
        - x arrives as NHWC, y as NHW or NHWC1. Output is always NCHW for both.
        - Masks are auto-scaled from 0..255 to 0..1 when needed.
        - No shuffling here; it happens in the underlying generator.
    """

    def __init__(self, py_generator: Iterable[Tuple[np.ndarray, np.ndarray]]):
        super().__init__()
        self.py_generator = py_generator

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        for x, y in self.py_generator:
            # x: (N, H, W, C) -> (N, C, H, W)
            if isinstance(x, np.ndarray):
                x_t = torch.from_numpy(np.transpose(x, (0, 3, 1, 2))).to(
                    dtype=torch.float32
                )
            else:
                # Already a tensor (NHWC) in some flows.
                x_t = x.permute(0, 3, 1, 2).contiguous().to(dtype=torch.float32)

            # y: (N, H, W) or (N, H, W, 1) -> (N, 1, H, W)
            if isinstance(y, np.ndarray):
                if y.ndim == 3:
                    y = y[..., None]
                y_t = torch.from_numpy(np.transpose(y, (0, 3, 1, 2))).to(
                    dtype=torch.float32
                )
            else:
                if y.ndim == 3:
                    y = y.unsqueeze(-1)
                y_t = y.permute(0, 3, 1, 2).contiguous().to(dtype=torch.float32)

            # Ensure masks are in [0, 1]. Auto-scale 0..255 to 0..1 if detected.
            maxv = float(y_t.max().item()) if y_t.numel() > 0 else 1.0
            if maxv > 1.5:
                y_t = y_t / 255.0

            yield x_t, y_t


def create_train_val_datasets(frames):
    """
    Create training, validation, and test iterables built on the project Generator.

        - Train uses albumentations with area-weighted sampling.
        - Val and test are deterministic with the same patching scheme.
        - Batch comes from the Generator (iterable style).
        - DataLoader wrappers handle worker setup and CUDA pinning.

    Args:
        frames: List of FrameInfo objects from get_all_frames.

    Returns:
        Tuple[Iterable, Iterable, Iterable]: train, val, test DataLoaders.
    """
    frames_json = os.path.join(config.preprocessed_dir, "aa_frames_list.json")
    training_frames, validation_frames, test_frames = split_dataset(
        frames, frames_json, config.test_ratio, config.val_ratio
    )

    # Quick distribution summary
    stats = summarize_positive_rates(
        frames,
        {"train": training_frames, "val": validation_frames, "test": test_frames},
    )
    print(
        "\n[DATA][STATS] positive-rate % by frame - "
        "mean | median | std | min..max  (n)"
    )
    for k in ("train", "val", "test"):
        s = stats[k]
        print(
            f"  {k:>5}: {s['mean']:.3f} | {s['median']:.3f} | {s['std']:.3f} | "
            f"{s['min']:.3f}..{s['max']:.3f}  (n={s['n']})"
        )

    # Channel layout: inputs first, then label immediately after inputs.
    input_channels = list(range(len(config.channel_list)))
    label_channel = len(config.channel_list)

    # Patch size (optionally overridden by tuner fields).
    patch_h, patch_w = config.patch_size
    if (
        hasattr(config, "tune_patch_h")
        and hasattr(config, "tune_patch_w")
        and config.tune_patch_h
        and config.tune_patch_w
    ):
        patch_h, patch_w = int(config.tune_patch_h), int(config.tune_patch_w)

    # H, W, total_channels (inputs + 1 label)
    patch_size = [patch_h, patch_w, len(config.channel_list) + 1]

    # Sampling and augmentation controls.
    aug_strength = getattr(config, "augmenter_strength", 1.0)
    min_pos_frac = float(getattr(config, "min_pos_frac", 0.0))
    pos_ratio = getattr(config, "pos_ratio", None)  # None => tensorflow-like random sampling
    stride = getattr(config, "patch_stride", None)

    # Build python generators (albumentations path inside DataGenerator).
    train_gen = Generator(
        input_channels,
        patch_size,
        training_frames,
        frames,
        label_channel,
        augmenter="alb",
        augmenter_strength=aug_strength,
        min_pos_frac=min_pos_frac,
        pos_ratio=pos_ratio,
        stride=stride,
        weighting="area",  # area-weighted frame sampling to mirror original tensorflow from Ankit
    ).random_generator(config.train_batch_size)

    val_gen = Generator(
        input_channels,
        patch_size,
        validation_frames,
        frames,
        label_channel,
        augmenter=None,
        augmenter_strength=1.0,
        min_pos_frac=0.0,
        pos_ratio=None,
        stride=stride,
        weighting="area",
    ).random_generator(config.train_batch_size)

    test_gen = Generator(
        input_channels,
        patch_size,
        test_frames,
        frames,
        label_channel,
        augmenter=None,
        augmenter_strength=1.0,
        min_pos_frac=0.0,
        pos_ratio=None,
        stride=stride,
        weighting="area",
    ).random_generator(config.train_batch_size)

    # Wrap with DataLoader-like iterables (prefetch_factor only when workers > 0).
    workers = int(getattr(config, "fit_workers", 8))
    train_ds = TorchGeneratorDataset(train_gen)
    val_ds = TorchGeneratorDataset(val_gen)
    test_ds = TorchGeneratorDataset(test_gen)

    base_kwargs = dict(
        batch_size=None,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
    )
    if workers > 0:
        base_kwargs_pf = {**base_kwargs, "prefetch_factor": 2}
    else:
        base_kwargs_pf = base_kwargs

    train_dl = DataLoader(train_ds, **base_kwargs_pf)
    val_dl = DataLoader(val_ds, **base_kwargs_pf)
    test_dl = DataLoader(test_ds, **base_kwargs_pf)
    return train_dl, val_dl, test_dl
