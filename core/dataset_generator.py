# core/dataset_generator.py
from imgaug import augmenters as iaa
from imgaug import SegmentationMapsOnImage
import numpy as np

def imageAugmentationWithIAA(strength=1.0):
    """
    All shape-changing augs use keep_size=True so output has SAME H×W as input.
    `strength` scales probabilities/amounts in [0..1].
    """
    s = float(np.clip(strength, 0.0, 1.0))
    sometimes = lambda aug, p=0.5: iaa.Sometimes(p * s, aug)

    seq = iaa.Sequential(
        [
            iaa.Fliplr(0.5 * s),
            iaa.Flipud(0.5 * s),

            # Crop but resize back to original size
            sometimes(iaa.Crop(percent=(0, 0.10 * s), keep_size=True), 0.5),

            # Photometric (no size change)
            sometimes(iaa.GaussianBlur(sigma=(0, 0.30 * s)), 0.30),
            sometimes(iaa.LinearContrast((0.3, 1.2)), 0.30 * s),

            # Geometric warps; force keep_size
            # (PiecewiseAffine keeps size by design; Perspective needs keep_size=True)
            sometimes(iaa.PiecewiseAffine(scale=0.05 * s), 0.30 * s),
            sometimes(iaa.PerspectiveTransform(scale=0.01 * s, keep_size=True), 0.10 * s),
        ],
        random_order=True,
    )
    return seq



class DataGenerator:
    """Generates random or sequential patches from frames."""

    def __init__(self, input_image_channel, patch_size, frame_list, frames,
                 annotation_channel, augmenter=None, augmenter_strength=1.0, min_pos_frac=0.0):
        """
        Args:
            input_image_channel (list(int))
            patch_size (tuple(int,int))
            frame_list (list(int))
            frames (list(FrameInfo))
            annotation_channel (int)
            augmenter (str or None): 'iaa' or None
            augmenter_strength (float): 0..1 scaling of augmentation probabilities
            min_pos_frac (float): minimum fraction of positive pixels required in the label patch (0 disables)
        """
        self.input_image_channel = input_image_channel
        self.patch_size = patch_size
        self.frame_list = frame_list
        self.frames = frames
        self.annotation_channel = annotation_channel
        self.augmenter = augmenter
        self.augmenter_strength = float(augmenter_strength)
        self.min_pos_frac = float(min_pos_frac)

        total_area = sum([frames[i].img.shape[0] * frames[i].img.shape[1] for i in frame_list])
        self.frame_list_weights = [(frames[i].img.shape[0] * frames[i].img.shape[1]) / total_area for i in frame_list]

    def all_sequential_patches(self, step_size):
        patches = []
        for fn in self.frame_list:
            frame = self.frames[fn]
            ps = frame.sequential_patches(self.patch_size, step_size)
            patches.extend(ps)
        data = np.array(patches)
        img = data[..., self.input_image_channel]
        ann = data[..., self.annotation_channel]
        return (img, ann)

    def _sample_one_patch(self):
        fn = np.random.choice(self.frame_list, p=self.frame_list_weights)
        frame = self.frames[fn]
        return frame.random_patch(self.patch_size)

    def random_patch(self, BATCH_SIZE):
        patches = [self._sample_one_patch() for _ in range(BATCH_SIZE)]
        data = np.array(patches)
        img = data[..., self.input_image_channel]
        ann = data[..., self.annotation_channel]
        return (img, ann)

    def random_generator(self, BATCH_SIZE):
        seq = imageAugmentationWithIAA(getattr(self, "augmenter_strength", 1.0))
        while True:
            X, y = self.random_patch(BATCH_SIZE)  # X: (B,H,W,C_in), y: (B,H,W,1) binary 0/1

            from imgaug.augmentables.segmaps import SegmentationMapsOnImage


            if self.augmenter == 'iaa':
                seq_det = seq.to_deterministic()

                # ---- images ----
                X = seq_det.augment_images(X)

                # ---- labels: normalize to (B, H, W, 1) robustly ----
                m = y
                if m.ndim == 4:
                    # Could be (B, H, W, C) or (B, C, H, W)
                    if m.shape[-1] == 1:
                        pass  # already (B, H, W, 1)
                    elif m.shape[1] == 1 and m.shape[-1] != 1:
                        # (B, 1, H, W) -> (B, H, W, 1)
                        m = np.transpose(m, (0, 2, 3, 1))
                    else:
                        # (B, H, W, C>1) -> take first class channel
                        m = m[..., :1]
                elif m.ndim == 3:
                    # (B, H, W) -> add channel dim
                    m = m[..., np.newaxis]
                else:
                    raise ValueError(f"Unexpected mask shape {m.shape}; expected 3D or 4D.")

                # ensure binary 0/1 before augmenting
                m = (m > 0).astype(np.uint8)

                # build segmaps list
                segs = [
                    SegmentationMapsOnImage(m[i, ..., 0].astype(np.int32),
                                            shape=m[i, ..., 0].shape)
                    for i in range(m.shape[0])
                ]

                # augment
                segs_aug = seq_det.augment_segmentation_maps(segs)

                # back to (B, H, W, 1), binary uint8
                ann = np.stack([seg_aug.get_arr().astype(np.uint8) for seg_aug in segs_aug], axis=0)
                ann = (ann > 0).astype(np.float32)[..., np.newaxis]

                X = X.astype(np.float32)

                yield X, ann
            else:
                ann = (y > 0).astype(np.float32)
                if ann.ndim == 3:  # (B,H,W) -> (B,H,W,1)
                    ann = ann[..., np.newaxis]

                X = X.astype(np.float32)
                yield X, ann


