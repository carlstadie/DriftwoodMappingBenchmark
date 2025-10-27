# core/dataset_generator.py
# Albumentations-based generator with TensorFlow parity
# - Yields (X, y) as (B, H, W, C) and (B, H, W, 1)
# - Area-weighted frame sampling
# - Optional Albumentations pipeline that keeps size

from __future__ import annotations
from typing import Tuple, List, Sequence, Dict, Any, Optional

import numpy as np
import albumentations as A


# ------------------------------
# Augmentation (Albumentations)
# ------------------------------
def alb_augmentation(patch_size: Tuple[int, int], strength: float = 1.0) -> A.Compose:
    """
    All shape-changing augs keep output size equal to the input `patch_size`.
    `strength` scales probabilities/amounts in [0..1].
    """
    s = float(np.clip(strength, 0.0, 1.0))
    H, W = int(patch_size[0]), int(patch_size[1])

    # Photometric + geometric, roughly mirroring the old imgaug recipe
    aug = A.Compose(
        [
            A.HorizontalFlip(p=0.5 * s),
            A.VerticalFlip(p=0.5 * s),

            # Crop but resize back to original size (TF parity)
            # Using RandomResizedCrop with scale <= 1.0 to emulate Crop(keep_size=True)
            A.RandomResizedCrop(height=H, width=W, scale=(max(0.0, 1.0 - 0.10 * s), 1.0), ratio=(0.9, 1.1), p=0.5 * s),

            # Photometric (no size change)
            A.GaussianBlur(blur_limit=(3, 7), p=0.30 * s),
            A.RandomBrightnessContrast(brightness_limit=0.0, contrast_limit=0.7 * s, p=0.30 * s),

            # Geometric warps
            A.PiecewiseAffine(scale=0.05 * s, p=0.30 * s),
            A.Perspective(scale=(0.0, 0.01 * s), keep_size=True, p=0.10 * s),
        ]
    )
    return aug


# ------------------------------
# Data generator
# ------------------------------
class DataGenerator:
    """Generates random or sequential patches from frames.

    Notes
    -----
    * Matches earlier TF code: frame sampling is weighted by image area.
    * Returns numpy arrays; training wraps them into PyTorch tensors.
    * If `augmenter` is 'alb'/'albumentations' (default), apply Albumentations; else no aug.
    * `min_pos_frac`, `pos_ratio`, `stride`, `weighting` are accepted for compatibility with
      newer callers but are ignored here to keep TF behavior.
    """

    def __init__(
        self,
        input_image_channel: Sequence[int],
        patch_size: Tuple[int, int],
        frame_list: Sequence[int],
        frames: Sequence[Any] | Dict[int, Any],
        annotation_channel: int,
        augmenter: Optional[str] = "alb",  # 'alb'/'albumentations' or None
        augmenter_strength: float = 1.0,
        min_pos_frac: float = 0.0,
        pos_ratio: Optional[float] = None,   # accepted but ignored (TF parity)
        stride: Optional[Tuple[int, int]] = None,  # accepted but ignored (TF parity)
        weighting: str = "area",  # accepted; only 'area' supported (TF parity)
        **_: Any,
    ) -> None:
        self.input_image_channel = list(input_image_channel)
        self.patch_size = (int(patch_size[0]), int(patch_size[1]))
        self.frame_list = list(frame_list)
        self.frames = frames
        self.annotation_channel = int(annotation_channel)
        self.augmenter = augmenter
        self.augmenter_strength = float(augmenter_strength)
        self.min_pos_frac = float(min_pos_frac)
        self.pos_ratio = None if pos_ratio is None else float(pos_ratio)

        # Prebuild Albumentations pipeline (uses fixed patch_size)
        self._alb = alb_augmentation(self.patch_size, self.augmenter_strength)

        # Compute area-weighted sampling probabilities (TF parity)
        total_area = 0.0
        areas: List[float] = []
        for i in self.frame_list:
            fr = self._frame(i)
            H, W = fr.img.shape[:2]
            a = float(H * W)
            areas.append(a)
            total_area += a
        total_area = max(total_area, 1e-6)
        self.frame_list_weights = [a / total_area for a in areas]

    # -------- public --------
    def all_sequential_patches(self, step_size: Tuple[int, int]):
        patches = []
        for fn in self.frame_list:
            frame = self._frame(fn)
            ps = frame.sequential_patches(self.patch_size, step_size)
            patches.extend(ps)
        data = np.asarray(patches)
        img = data[..., self.input_image_channel]
        ann = data[..., self.annotation_channel]
        return img, ann

    def random_patch(self, BATCH_SIZE: int):
        patches = [self._sample_one_patch() for _ in range(int(BATCH_SIZE))]
        data = np.asarray(patches)
        img = data[..., self.input_image_channel]
        ann = data[..., -1]
        return img, ann

    def random_generator(self, BATCH_SIZE: int):
        """Yield endless batches (X, y) as (B,H,W,C_in) and (B,H,W,1)."""
        use_alb = str(self.augmenter).lower() in {"alb", "albumentations"}
        while True:
            X, y = self.random_patch(BATCH_SIZE)  # X: (B,H,W,C_in), y: (B,H,W)

            # Normalize masks to (B,H,W,1), binary 0/1
            m = y
            if m.ndim == 4:
                # Could be (B, H, W, C) or (B, C, H, W)
                if m.shape[-1] == 1:
                    pass  # already (B, H, W, 1)
                elif m.shape[1] == 1 and m.shape[-1] != 1:  # (B, 1, H, W) -> (B, H, W, 1)
                    m = np.transpose(m, (0, 2, 3, 1))
                else:
                    m = m[..., :1]  # take first class channel
            elif m.ndim == 3:
                m = m[..., np.newaxis]
            else:
                raise ValueError(f"Unexpected mask shape {m.shape}; expected 3D or 4D.")

            m = (m > 0).astype(np.uint8)

            if use_alb:
                X_aug = np.empty_like(X, dtype=np.float32)
                y_aug = np.empty(m.shape[:3] + (1,), dtype=np.float32)
                for i in range(X.shape[0]):
                    res = self._alb(image=X[i].astype(np.float32), mask=m[i, ..., 0])
                    X_aug[i] = res["image"].astype(np.float32)
                    y_aug[i, ..., 0] = (res["mask"] > 0).astype(np.float32)
                X, ann = X_aug, y_aug
            else:
                ann = m.astype(np.float32)
                X = X.astype(np.float32)

            yield X, ann

    # -------- internal --------
    def _frame(self, idx: int):
        return self.frames[idx] if not isinstance(self.frames, dict) else self.frames[int(idx)]

    def _sample_one_patch(self):
        fn = int(np.random.choice(self.frame_list, p=self.frame_list_weights))
        frame = self._frame(fn)
        # Frame supplies a full (H,W,C) composite. random_patch returns an H×W crop with all bands.
        return frame.random_patch(self.patch_size)


# Backwards-compat alias used by newer training code
Generator = DataGenerator
