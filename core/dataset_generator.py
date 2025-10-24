# core/dataset_generator.py
# Positive/negative controlled patch sampling with pre-indexed candidates
# Albumentations augmentation (shared between training/tuning)

from typing import Tuple, List, Dict, Sequence, Union, Optional
import numpy as np

try:
    import albumentations as A
    import cv2  # required by Albumentations for some ops
except ImportError as e:
    raise ImportError(
        "Albumentations is required for this dataset generator. "
        "Install with: pip install albumentations opencv-python"
    ) from e


def _to_slice_or_indices(sel: Union[int, slice, Sequence[int], None], c: int):
    """Normalize channel selection so we can index numpy arrays safely."""
    if sel is None:
        return slice(0, c)
    if isinstance(sel, (int, slice)):
        return sel
    idx = np.array(sel, dtype=int).tolist()
    return idx


def imageAugmentationWithAlbumentations(
    strength: float = 0.5, patch_size: Optional[Tuple[int, int]] = None
) -> A.Compose:
    """
    Build a moderately strong but safe augmentation policy for aerial/satellite imagery.

    strength in [0,1] scales probabilities and magnitudes.
    """
    s = float(np.clip(strength, 0.0, 1.0))

    geo_prob = 0.70 * s
    phot_prob = 0.85 * s

    # Geometric
    geo = [
        A.HorizontalFlip(p=0.5 * s),
        A.VerticalFlip(p=0.25 * s),
        A.RandomRotate90(p=0.25 * s),
        A.ShiftScaleRotate(
            shift_limit=0.05 * s, scale_limit=0.15 * s, rotate_limit=int(10 + 20 * s),
            border_mode=cv2.BORDER_REFLECT_101, p=geo_prob
        ),
        A.GridDistortion(num_steps=5, distort_limit=0.2 * s, border_mode=cv2.BORDER_REFLECT_101, p=0.20 * s),
        A.ElasticTransform(alpha=50 * s, sigma=7 * s, alpha_affine=15 * s,
                           border_mode=cv2.BORDER_REFLECT_101, p=0.10 * s),
    ]

    # Photometric (keep band semantics — avoid hue/saturation on multispectral)
    phot = [
        #A.GaussNoise(var_limit=(5.0, 25.0 * s), p=0.35 * s),
        #A.RandomBrightnessContrast(brightness_limit=0.2 * s, contrast_limit=0.2 * s, p=phot_prob),
        #A.MultiplicativeNoise(multiplier=(0.9, 1.1), per_channel=False, p=0.30 * s),
        #A.GaussianBlur(blur_limit=(3, int(max(3, 3 + 4 * s))), p=0.20 * s),
    ]

    keep_size = []
    if patch_size is not None:
        H, W = int(patch_size[0]), int(patch_size[1])
        keep_size.append(A.Resize(H, W, interpolation=cv2.INTER_LINEAR))

    return A.Compose(geo + phot + keep_size)


def _integral_image(mask: np.ndarray) -> np.ndarray:
    """Integral image for fast window sums (2D, uint32)."""
    ii = np.cumsum(np.cumsum(mask.astype(np.uint32), axis=0), axis=1)
    return ii


def _window_sum(ii: np.ndarray, top: int, left: int, h: int, w: int) -> int:
    """Sum of mask[top:top+h, left:left+w] using integral image."""
    t, l, b, r = top, left, top + h - 1, left + w - 1
    total = ii[b, r]
    if t > 0:
        total -= ii[t - 1, r]
    if l > 0:
        total -= ii[b, l - 1]
    if t > 0 and l > 0:
        total += ii[t - 1, l - 1]
    return int(total)


class DataGenerator:
    """
    Patch generator with TF-parity defaults:
      - Frame sampling weighted by image area (like TF)
      - Random patches by default (no pos/neg control)
    Set pos_ratio (e.g., 0.5) + min_pos_frac > 0 to enable controlled pos/neg sampling.
    """

    def __init__(
        self,
        input_image_channel: Union[int, slice, Sequence[int], None],
        patch_size: Tuple[int, int, int],
        frame_list: Sequence[int],
        frames: Union[List, Dict[int, object]],
        annotation_channel: int,
        augmenter: Optional[str] = "alb",
        augmenter_strength: float = 0.5,
        min_pos_frac: float = 0.0,
        pos_ratio: Optional[float] = None,
        stride: Optional[Tuple[int, int]] = None,
        max_retries: int = 20,
        weighting: str = "area",  # 'area' (TF default), 'positive', or 'uniform'
    ):
        self.patch_size = (int(patch_size[0]), int(patch_size[1]))
        self.input_image_channel = input_image_channel
        self.annotation_channel = int(annotation_channel)
        self.frame_list = list(frame_list)
        self.frames = frames
        self.augmenter = augmenter or None
        self.augmenter_strength = float(augmenter_strength)
        self.min_pos_frac = float(min_pos_frac)
        self.pos_ratio = None if pos_ratio is None else float(np.clip(pos_ratio, 0.0, 1.0))
        self.max_retries = int(max_retries)
        self.stride = stride if stride is not None else (max(1, self.patch_size[0] // 2),
                                                         max(1, self.patch_size[1] // 2))

        # Albumentations pipeline
        self._alb = imageAugmentationWithAlbumentations(self.augmenter_strength, self.patch_size)

        # Pre-compute candidate top-left coordinates for positive/negative patches per frame
        self.pos_coords: Dict[int, List[Tuple[int, int]]] = {}
        self.neg_coords: Dict[int, List[Tuple[int, int]]] = {}
        self._build_candidate_indices()

        # Frame sampling weights
        if weighting == "positive":
            # prefer frames with more positives
            weights = []
            for fn in self.frame_list:
                frame = self._get_frame(fn)
                pos = float((frame.annotations > 0).sum())
                tot = float(frame.annotations.size) + 1e-6
                weights.append((pos / tot) + 1e-6)
            w = np.asarray(weights, dtype=np.float64)
            self.frame_list_weights = (w / w.sum()).tolist()
        elif weighting == "uniform":
            self.frame_list_weights = [1.0 / len(self.frame_list)] * len(self.frame_list)
        else:
            # TF parity: weight by image area
            weights = []
            for fn in self.frame_list:
                frame = self._get_frame(fn)
                H, W, _ = frame.img.shape
                weights.append(float(H * W))
            w = np.asarray(weights, dtype=np.float64)
            self.frame_list_weights = (w / w.sum()).tolist()

    def _get_frame(self, fn):
        return self.frames[fn] if isinstance(self.frames, dict) else self.frames[int(fn)]

    def _select_inputs(self, data: np.ndarray) -> np.ndarray:
        """Select input channels from composite [C_in + label]"""
        c = data.shape[-1] - 1
        sel = _to_slice_or_indices(self.input_image_channel, c)
        return data[..., sel]

    def _ensure_mask_shape(self, m: np.ndarray) -> np.ndarray:
        """Ensure (B,H,W,1) binary float mask."""
        if m.ndim == 2:
            m = m[..., None]
        if m.ndim == 3 and m.shape[-1] != 1:
            m = m[..., :1]
        return (m > 0).astype(np.float32)

    # ---------- candidate indexing ----------
    def _build_candidate_indices(self):
        ph, pw = self.patch_size
        sh, sw = self.stride
        for fn in self.frame_list:
            frame = self._get_frame(fn)
            mask = (frame.annotations > 0).astype(np.uint8)
            H, W = mask.shape[:2]

            pos_list: List[Tuple[int, int]] = []
            neg_list: List[Tuple[int, int]] = []

            # If image is smaller than patch -> one candidate at (0,0)
            if H <= ph or W <= pw:
                pos_frac = mask.mean()
                (pos_list if pos_frac >= self.min_pos_frac else neg_list).append((0, 0))
            else:
                ii = _integral_image(mask)
                for i in range(0, H - ph + 1, sh):
                    for j in range(0, W - pw + 1, sw):
                        s = _window_sum(ii, i, j, ph, pw)
                        frac = s / float(ph * pw)
                        (pos_list if frac >= self.min_pos_frac else neg_list).append((i, j))

            # Always keep a fallback if a list ends empty
            if len(pos_list) == 0:
                pos_list.append((0, 0))
            if len(neg_list) == 0:
                neg_list.append((0, 0))

            self.pos_coords[fn] = pos_list
            self.neg_coords[fn] = neg_list

    # ---------- sampling ----------
    def _sample_patch_indexed(self, want_positive: bool):
        # Choose frame (weighted)
        idx = int(np.random.choice(len(self.frame_list), p=self.frame_list_weights))
        fn = self.frame_list[idx]
        frame = self._get_frame(fn)

        # Pick a candidate coord (positive or negative list)
        coords = self.pos_coords[fn] if (want_positive and self.min_pos_frac > 0) else self.neg_coords[fn]
        i, j = coords[np.random.randint(0, len(coords))]

        # Clamp so we never index beyond image bounds
        H, W = self.patch_size
        H_img, W_img, C_in = frame.img.shape
        i = min(i, max(0, H_img - H))
        j = min(j, max(0, W_img - W))

        # IMPORTANT: pass a 2D patch size (FrameInfo handles channels + label)
        patch = frame.getPatch(
            i, j,
            (int(H), int(W)),                    # 2-D; FrameInfo computes C_out
            (int(min(H_img, H)), int(min(W_img, W)))
        )
        return patch

    def _sample_patch_random(self):
        """TF-like: choose frame (area-weighted), then let FrameInfo do a random patch."""
        idx = int(np.random.choice(len(self.frame_list), p=self.frame_list_weights))
        fn = self.frame_list[idx]
        frame = self._get_frame(fn)
        return frame.random_patch(self.patch_size)

    def random_patch(self, batch_size: int):
        patches = []
        if self.pos_ratio is None:
            # TF parity: purely random per-frame patch
            for _ in range(batch_size):
                patches.append(self._sample_patch_random())
        else:
            # Controlled pos/neg sampling
            for _ in range(batch_size):
                want_pos = (np.random.rand() < self.pos_ratio)
                patches.append(self._sample_patch_indexed(want_pos))

        data = np.asarray(patches)
        X = self._select_inputs(data)                     # (B,H,W,Cin)
        y = data[..., -1]            # (B,H,W) -1 as last band of data is always the label !!!!!!!!!!!
        y = (y > 0).astype(np.float32)[..., None]         # (B,H,W,1)
        return X, y

    def random_generator(self, batch_size: int):
        """Yield batches of NHWC images and masks (float32)."""
        use_alb = str(self.augmenter).lower() in ("alb", "albumentations")
        while True:
            X, y = self.random_patch(batch_size)  # X: (B,H,W,Cin), y: (B,H,W,1)
            y = self._ensure_mask_shape(y)

            if use_alb:
                imgs = []
                msks = []
                for i in range(X.shape[0]):
                    aug = self._alb(image=X[i], mask=y[i, ..., 0])
                    imgs.append(aug["image"])
                    msks.append(aug["mask"])
                X = np.stack(imgs, 0).astype(np.float32)
                y = np.stack(msks, 0).astype(np.float32)[..., None]
            else:
                X = X.astype(np.float32)
                y = y.astype(np.float32)

            # Clamp/replace NaNs from earlier normalization
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

            yield X, y

    # ---------- utility ----------
    def all_sequential_patches(self, step_size: Tuple[int, int]):
        """Exhaustive patches from all frames, for debugging/eval."""
        ps = []
        for fn in self.frame_list:
            frame = self._get_frame(fn)
            ps.extend(frame.sequential_patches(self.patch_size, step_size))
        data = np.asarray(ps)
        X = self._select_inputs(data)
        y = data[..., self.annotation_channel]
        y = (y > 0).astype(np.float32)[..., None]
        return X, y
