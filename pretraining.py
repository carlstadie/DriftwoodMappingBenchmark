import os
import glob
import numpy as np
import rasterio
import matplotlib.pyplot as plt
from imgaug import augmenters as iaa

def imageAugmentationWithIAA():
    sometimes = lambda aug, prob=0.5: iaa.Sometimes(prob, aug)
    return iaa.Sequential([
        #iaa.Fliplr(0.5),
        #iaa.Flipud(0.5),
        #sometimes(iaa.Crop(percent=(0, 0.1))),
        #sometimes(iaa.GaussianBlur(sigma=(0, 0.3)), 0.3),
        #sometimes(iaa.LinearContrast((0.3, 1.2)), 0.3),
        #sometimes(iaa.PiecewiseAffine(0.05), 0.3),
        #sometimes(iaa.PerspectiveTransform(0.01), 0.1)
    ], random_order=True)

class FolderDataGenerator:
    """
    On-the-fly patch generator + MIM masking directly from a folder of GeoTIFFs,
    with one-time visualization of raw RGB patches.
    """
    def __init__(self,
                 image_dir,
                 patch_size=(256,256),
                 input_image_channel=[0,1,2],
                 augmenter='iaa',
                 token_size=16,
                 mask_ratio=0.75,
                 mean=(0.485, 0.456, 0.406),
                 std=(0.229, 0.224, 0.225),
                 visualize=False,
                 vis_n=16):
        self.image_paths = sorted(glob.glob(os.path.join(image_dir, "*.tif")))
        if not self.image_paths:
            raise ValueError(f"No .tif files found in {image_dir}")
        self.patch_size = patch_size
        self.input_image_channel = input_image_channel
        self.augmenter = augmenter
        self.seq = imageAugmentationWithIAA() if augmenter=='iaa' else None

        # image-size–weighted sampling
        areas = []
        for p in self.image_paths:
            with rasterio.open(p) as src:
                areas.append(src.height * src.width)
        total = sum(areas)
        self.weights = [a/total for a in areas]

        # MIM / normalization params
        self.token_size = token_size
        self.mask_ratio = mask_ratio
        self.mean = np.array(mean).reshape(1,1,len(input_image_channel))
        self.std  = np.array(std).reshape(1,1,len(input_image_channel))

        # visualization
        self.visualize = visualize
        self.vis_n = vis_n
        self._has_visualized = False

    def random_patch(self, BATCH_SIZE):
        """Draw BATCH_SIZE random patches (H,W,C) from random TIFFs."""
        ph, pw = self.patch_size
        patches = []
        for _ in range(BATCH_SIZE):
            idx = np.random.choice(len(self.image_paths), p=self.weights)
            with rasterio.open(self.image_paths[idx]) as src:
                img = src.read()                  # (C, H, W)
                img = np.transpose(img, (1,2,0))  # → (H, W, C)
            H, W, _ = img.shape
            i = np.random.randint(0, H - ph + 1)
            j = np.random.randint(0, W - pw + 1)
            patch = img[i:i+ph, j:j+pw, :]
            patches.append(patch)
        return np.stack(patches, axis=0)  # (B, H, W, C)

    def _show_patches(self, raw_X):
        """Plot the first vis_n raw RGB patches in a grid."""
        n = min(self.vis_n, raw_X.shape[0])
        cols = int(np.sqrt(n))
        rows = int(np.ceil(n/cols))
        fig, axes = plt.subplots(rows, cols, figsize=(cols*2, rows*2))
        for idx in range(n):
            r, c = divmod(idx, cols)
            ax = axes[r, c] if rows>1 else axes[c]
            rgb = raw_X[idx][..., self.input_image_channel]
            ax.imshow(rgb[0:,:,0], cmap='gray')
            ax.axis('off')
        plt.tight_layout()
        plt.savefig("/isipd/projects/p_planetdw/data/methods_test/training_images/raw_patches.png", dpi=300)

    def random_mim_generator(self, BATCH_SIZE):
        """
        Yields tuples (X, mask) each iteration:
          X    : np.ndarray (B, H, W, 3), float32 normalized to ImageNet
          mask : np.ndarray (B, num_tokens), boolean mask for MIM
        """
        while True:
            # 1) draw raw uint8 patches
            X_raw = self.random_patch(BATCH_SIZE)



            # 3) augment
            if self.seq:
                X_aug = self.seq.to_deterministic().augment_images(X_raw.astype(np.uint8))
            else:
                X_aug = X_raw

            if self.visualize and not self._has_visualized:
                self._show_patches(X_aug)
                self._has_visualized = True

            # 4) select RGB channels & normalize
            X = X_aug[..., self.input_image_channel].astype(np.float32) / 255.0
            X = (X - self.mean) / self.std  # broadcast over H,W

            # 5) create random mask over ViT tokens
            B, H, W, _ = X.shape
            gh, gw = H // self.token_size, W // self.token_size
            num_tokens = gh * gw
            mask = (np.random.rand(B, num_tokens) < self.mask_ratio)

            yield X, mask

# ——— Example usage ———
if __name__ == "__main__":
    gen = FolderDataGenerator(
        image_dir="/isipd/projects/p_planetdw/data/methods_test/training_images/test",
        patch_size=(64,64),
        input_image_channel=[0,1,2],
        augmenter='iaa',
        token_size=16,
        mask_ratio=0.2,
        visualize=True,
        vis_n=16
    )

    loader = gen.random_mim_generator(BATCH_SIZE=64)
    images, masks = next(loader)
    # A 4×4 grid of raw RGB patches will pop up.
    # images: (64,256,256,3) float32, normalized; masks: (64,256) bool.
