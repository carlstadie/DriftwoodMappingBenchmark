# core/frame_info.py  (PyTorch)
#    Edited by Sizhuo Li, Carl Stadie
#    Author: Ankit Kariryaa, University of Bremen

import numpy as np


def image_normalize(im, axis=(0, 1), c=1e-8, nodata_val=None):
    """
    Normalize to zero mean and unit std along the given axis.

    If a nodata value is specified, normalise without nodata pixels and set
    them to nan.
    """
    if nodata_val is not None and np.sum(im == nodata_val) > 0:
        im = im.astype(np.float32)
        # Treat whole pixel as nodata if any band has nodata value
        im[np.any(im == nodata_val, axis=2), :] = np.nan
        return (im - np.nanmean(im, axis)) / (np.nanstd(im, axis) + c)
    else:
        return (im - im.mean(axis)) / (im.std(axis) + c)


class FrameInfo:
    """Defines a frame, includes its constituent images (inputs) and annotation."""

    def __init__(self, img, annotations, dtype=np.float32):
        """
        Args:
            img: ndarray (H, W, C_in)
            annotations: ndarray (H, W) or (H, W, 1)
            dtype: np.float32, optional
        """
        self.img = img
        self.annotations = annotations
        self.dtype = dtype

    def getPatch(
        self,
        top: int,
        left: int,
        patch_size,
        img_size,
        pad_mode: str = "reflect",
    ):
        """
        Return a composite patch (inputs + label as last channel), padded
        to `patch_size`.

        - top, left: top-left of the slice to take from the full image
        - patch_size: (H, W) or (H, W, C_out). If 2-D, C_out = C_in + 1 (label)
        - img_size: (h_slice, w_slice) actual slice size (clamped to image bounds)
        - pad_mode: 'reflect' (default) or 'constant' (zeros).
                    'reflect' looks best for aug.
        """
        if isinstance(patch_size, (list, tuple)) and len(patch_size) == 2:
            H, W = int(patch_size[0]), int(patch_size[1])
            C_out = int(self.img.shape[2]) + 1
        else:
            H, W, C_out = (
                int(patch_size[0]),
                int(patch_size[1]),
                int(patch_size[2]),
            )

        h_slice = min(int(img_size[0]), H)
        w_slice = min(int(img_size[1]), W)

        # Grab slice from source (clamped to bounds)
        img_patch = self.img[top : top + h_slice, left : left + w_slice, :]
        # Normalise per patch
        #img_patch = image_normalize(img_patch, axis=(0, 1), nodata_val=0)

        lab_patch = self.annotations[top : top + h_slice, left : left + w_slice]
        if lab_patch.ndim == 2:
            lab_patch = lab_patch[..., None]
        comb = np.concatenate([img_patch, lab_patch], axis=-1)

        # Center the slice inside the output canvas
        off_h = (H - h_slice) // 2
        off_w = (W - w_slice) // 2

        # Prepare canvas (optionally reflect-pad to avoid big black borders
        # after rot/scale aug)
        patch = np.zeros((H, W, C_out), dtype=self.img.dtype)
        patch[
            off_h : off_h + h_slice,
            off_w : off_w + w_slice,
            : comb.shape[-1],
        ] = comb

        if pad_mode == "reflect":
            # top
            if off_h > 0:
                patch[
                    :off_h,
                    off_w : off_w + w_slice,
                    : comb.shape[-1],
                ] = patch[
                    off_h : off_h + 1,
                    off_w : off_w + w_slice,
                    : comb.shape[-1],
                ][::-1, ...]
            # bottom
            if H - (off_h + h_slice) > 0:
                patch[
                    off_h + h_slice :,
                    off_w : off_w + w_slice,
                    : comb.shape[-1],
                ] = patch[
                    off_h + h_slice - 1 : off_h + h_slice,
                    off_w : off_w + w_slice,
                    : comb.shape[-1],
                ][::-1, ...]
            # left
            if off_w > 0:
                patch[:, :off_w, : comb.shape[-1]] = patch[
                    :, off_w : off_w + 1, : comb.shape[-1]
                ][:, ::-1, :]
            # right
            if W - (off_w + w_slice) > 0:
                patch[:, off_w + w_slice :, : comb.shape[-1]] = patch[
                    :, off_w + w_slice - 1 : off_w + w_slice, : comb.shape[-1]
                ][:, ::-1, :]

        return patch

    def sequential_patches(self, patch_size, step_size):
        """Return all sequential patches in this frame."""
        img_shape = self.img.shape
        x = range(0, img_shape[0] - patch_size[0], step_size[0])
        y = range(0, img_shape[1] - patch_size[1], step_size[1])
        if img_shape[0] <= patch_size[0]:
            x = [0]
        if img_shape[1] <= patch_size[1]:
            y = [0]

        ic = (min(img_shape[0], patch_size[0]), min(img_shape[1], patch_size[1]))
        xy = [(i, j) for i in x for j in y]
        img_patches = []
        for i, j in xy:
            img_patch = self.getPatch(i, j, patch_size, ic)
            img_patches.append(img_patch)
        return img_patches

    def random_patch(self, patch_size):
        """
        random crops

        centers the crop in the padded canvas instead of placing it at [0,0].
        """
        img_shape = self.img.shape
        x = (
            0
            if (img_shape[0] <= patch_size[0])
            else np.random.randint(0, img_shape[0] - patch_size[0])
        )
        y = (
            0
            if (img_shape[1] <= patch_size[1])
            else np.random.randint(0, img_shape[1] - patch_size[1])
        )
        ic = (min(img_shape[0], patch_size[0]), min(img_shape[1], patch_size[1]))
        return self.getPatch(x, y, patch_size, ic, pad_mode="reflect")
