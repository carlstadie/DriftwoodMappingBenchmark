from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import albumentations as A
import numpy as np


# ------------------------------
# Augmentation (Albumentations)
# ------------------------------
def alb_augmentation(
    patch_size: Tuple[int, int],
    strength: float = 1.0,
) -> A.Compose:
    """
    All shape-changing augs keep output size equal to the input `patch_size`.
    `strength` scales probabilities/amounts in [0..1].
    """
    s = float(np.clip(strength, 0.0, 1.0))
    h, w = int(patch_size[0]), int(patch_size[1])

    # Photometric + geometric
    aug = A.Compose(
        [
            A.HorizontalFlip(p=0.5 * s),
            A.VerticalFlip(p=0.5 * s),
            # Crop but resize back to original size
            A.RandomResizedCrop(
                height=h,
                width=w,
                scale=(max(0.0, 1.0 - 0.10 * s), 1.0),
                ratio=(0.9, 1.1),
                p=0.5 * s,
            ),
            # Photometric 
            A.GaussianBlur(blur_limit=(3, 7), p=0.30 * s),
            A.RandomBrightnessContrast(
                brightness_limit=0.0,
                contrast_limit=0.7 * s,
                p=0.30 * s,
            ),
            # Geometric warps
            # A.PiecewiseAffine(scale=0.05 * s, p=0.30 * s),
            #A.ElasticTransform(
            #    alpha=1.0 * s,
            #    sigma=50.0 * s,
            #    alpha_affine=50.0 * s,
            #    p=0.30 * s,
            #),
            #A.Perspective(scale=(0.0, 0.01 * s), keep_size=True, p=0.10 * s),
        ]
    )
    return aug


# ------------------------------
# Data generator
# ------------------------------
class DataGenerator:
    """Generate random or sequential patches from frames.

    Notes
    -----
    * frame sampling is weighted by image area.
    * Returns numpy arrays; training wraps them into PyTorch tensors.
    * If `augmenter` is 'alb'/'albumentations', apply Albumentations;
      else no aug.
    * `min_pos_frac` and `pos_ratio` control positive/negative patch sampling:
        - `min_pos_frac`: minimum positive fraction of a patch to treat it
          as "positive" (0.0 -> any positive pixel).
        - `pos_ratio`: target fraction of positive patches in random batches.
    * `stride`, `weighting` are accepted for compatibility with newer callers.
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
        pos_ratio: Optional[float] = None,
        stride: Optional[Tuple[int, int]] = None,
        weighting: str = "area",  # accepted; only 'area' supported
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
            h, w = fr.img.shape[:2]
            a = float(h * w)
            areas.append(a)
            total_area += a
        total_area = max(total_area, 1e-6)
        self.frame_list_weights = [a / total_area for a in areas]

    # -------- public --------
    def all_sequential_patches(self, step_size: Tuple[int, int]):
        """Return all sequential patches and labels given a step size."""
        patches = []
        for fn in self.frame_list:
            frame = self._frame(fn)
            ps = frame.sequential_patches(self.patch_size, step_size)
            patches.extend(ps)
        data = np.asarray(patches)
        img = data[..., self.input_image_channel]
        ann = data[..., -1]
        return img, ann

    def random_patch(self, batch_size: int):
        """Return a single random batch (X, y)."""
        patches = [self._sample_one_patch() for _ in range(int(batch_size))]
        data = np.asarray(patches)
        img = data[..., self.input_image_channel]
        ann = data[..., -1]
        return img, ann

    def random_generator(self, batch_size: int):
        """Yield endless batches (X, y) as (B,H,W,C_in) and (B,H,W,1)."""
        use_alb = str(self.augmenter).lower() in {"alb", "albumentations"}
        while True:
            X, y = self.random_patch(batch_size)  # X: (B,H,W,C_in), y: (B,H,W)

            # Normalize masks to (B,H,W,1), binary 0/1
            m = y
            if m.ndim == 4:
                # Could be (B, H, W, C) or (B, C, H, W)
                if m.shape[-1] == 1:
                    pass  # already (B, H, W, 1)
                elif m.shape[1] == 1 and m.shape[-1] != 1:
                    # (B, 1, H, W) -> (B, H, W, 1)
                    m = np.transpose(m, (0, 2, 3, 1))
                else:
                    m = m[..., :1]  # take first class channel
            elif m.ndim == 3:
                m = m[..., np.newaxis]
            else:
                raise ValueError(
                    f"Unexpected mask shape {m.shape}; expected 3D or 4D."
                )

            m = (m > 0).astype(np.uint8)

            if use_alb:
                X_aug = np.empty_like(X, dtype=np.float32)
                y_aug = np.empty(m.shape[:3] + (1,), dtype=np.float32)
                for i in range(X.shape[0]):
                    res = self._alb(
                        image=X[i].astype(np.float32), mask=m[i, ..., 0]
                    )
                    X_aug[i] = res["image"].astype(np.float32)
                    y_aug[i, ..., 0] = (res["mask"] > 0).astype(np.float32)
                X, ann = X_aug, y_aug
            else:
                ann = m.astype(np.float32)
                X = X.astype(np.float32)

            yield X, ann

    # -------- internal --------
    def _frame(self, idx: int):
        """Return frame by index for list/dict-backed storage."""
        if not isinstance(self.frames, dict):
            return self.frames[idx]
        return self.frames[int(idx)]

    def _random_frame_patch(self) -> Tuple[np.ndarray, Optional[Tuple[int, int]]]:
        """Sample a random patch from a randomly chosen frame (area-weighted).

        Returns the padded patch along with the unpadded (h, w) size if padding was needed.
        The unpadded size is used to evaluate pos_ratio against real pixels only.
        """
        fn = int(np.random.choice(self.frame_list, p=self.frame_list_weights))
        frame = self._frame(fn)

        # Frame supplies a full (H,W,C) composite. random_patch returns an HxW crop
        # with all bands, padding if the frame is smaller than the target patch.
        patch = frame.random_patch(self.patch_size)

        h_frame, w_frame = frame.img.shape[:2]
        h_target, w_target = self.patch_size
        unpadded_slice = None
        if h_frame < h_target or w_frame < w_target:
            # Store the real data size so we ignore padded pixels when counting positives.
            unpadded_slice = (min(h_frame, h_target), min(w_frame, w_target))

        return patch, unpadded_slice

    def _is_positive_patch(
        self,
        ann: np.ndarray,
        unpadded_slice: Optional[Tuple[int, int]] = None,
    ) -> bool:
        """Return True if patch is considered 'positive' given the annotation mask.
        
        Args:
            ann: annotation array (HxW or HxWx1)
            unpadded_slice: optional (slice_h, slice_w) tuple indicating the real (unpadded) region.
                           If provided, only counts positives in the unpadded region.
        """
        # Extract unpadded region if provided
        if unpadded_slice is not None:
            h_slice, w_slice = unpadded_slice
            # Get the center crop that contains the real data
            H, W = ann.shape[:2]
            off_h = (H - h_slice) // 2
            off_w = (W - w_slice) // 2
            mask = ann[off_h : off_h + h_slice, off_w : off_w + w_slice] > 0
        else:
            mask = ann > 0
        
        if self.min_pos_frac <= 0.0:
            # Any labeled pixel makes this a positive patch
            return np.any(mask)
        # Otherwise require at least `min_pos_frac` of the pixels to be labeled
        return mask.mean() >= self.min_pos_frac

    def _sample_one_patch(self):
        """Sample a single random patch, optionally enforcing a pos/neg ratio."""
        # If no pos_ratio is set, keep original TF-style area-weighted behavior.
        if self.pos_ratio is None or not (0.0 < self.pos_ratio < 1.0):
            patch, _unpadded_slice = self._random_frame_patch()
            return patch

        want_pos = np.random.rand() < float(self.pos_ratio)
        max_tries = 50  # avoid infinite loops with very sparse labels
        last_patch = None

        for _ in range(max_tries):
            patch, unpadded_slice = self._random_frame_patch()
            last_patch = patch
            ann = patch[..., -1]
            is_pos = self._is_positive_patch(ann, unpadded_slice)

            if want_pos and is_pos:
                return patch
            if (not want_pos) and (not is_pos):
                return patch

        # Fallback: if we could not satisfy the constraint, return last sampled patch
        if last_patch is not None:
            return last_patch

        # If something went wrong, fall back to a fresh sample (discarding unpadded_slice).
        fallback_patch, _ = self._random_frame_patch()
        return fallback_patch


# Backwards-compatible alias used by newer training code
Generator = DataGenerator