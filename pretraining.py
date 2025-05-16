import os
import glob
import numpy as np
import rasterio
import matplotlib.pyplot as plt
from imgaug import augmenters as iaa

def imageAugmentationWithIAA():
    sometimes = lambda aug, prob=0.5: iaa.Sometimes(prob, aug)
    return iaa.Sequential([
        sometimes(iaa.Fliplr(0.5)),  # horizontally flip 50% of all images
        sometimes(iaa.Flipud(0.5)),  # vertically flip 20% of all images
        sometimes(iaa.Crop(percent=(0, 0.1))),  # random crops
        sometimes(iaa.GaussianBlur(sigma=(0, 0.9)), 0.9),
        sometimes(iaa.GammaContrast(gamma=0.5, per_channel=True), 0.3),
        iaa.Add(value=(-0.5,10), per_channel=True),
    ], random_order=True)

class FolderDataGenerator:
    """
    On-the-fly patch generator + MIM masking directly from a folder of GeoTIFFs,
    with one-time visualization of raw RGB patches and augmented patches with mask overlay.
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
        self._has_visualized_raw = False
        self._has_visualized_aug = False

    def random_patch(self, BATCH_SIZE):
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
            ax.imshow(rgb[0:,:,0])
            ax.axis('off')
        plt.tight_layout()
        fig.savefig("/isipd/projects/p_planetdw/data/methods_test/training_images/raw_patches.png", dpi=300)
        plt.close(fig)

    def _show_augmented_patches(self, X_aug, mask):
        """Plot first vis_n augmented patches with MIM mask overlaid."""
        B, H, W, _ = X_aug.shape
        ts = self.token_size
        gh, gw = H // ts, W // ts

        n = min(self.vis_n, B)
        cols = int(np.sqrt(n))
        rows = int(np.ceil(n/cols))
        fig, axes = plt.subplots(rows, cols, figsize=(cols*2, rows*2))

        for idx in range(n):
            r, c = divmod(idx, cols)
            ax = axes[r, c] if rows>1 else axes[c]

            # show the augmented image
            rgb = X_aug[idx][..., self.input_image_channel]
    
            ax.imshow(rgb[:,:,0])

            # build mask overlay at pixel resolution
            token_mask = mask[idx].reshape(gh, gw).astype(float)
            # upsample to HxW
            overlay = np.kron(token_mask, np.ones((ts, ts), dtype=np.float32))
            # show mask with transparency
            ax.imshow(overlay, cmap='Greys', alpha=overlay)
            ax.axis('off')

        plt.tight_layout()
        fig.savefig("/isipd/projects/p_planetdw/data/methods_test/training_images/augmented_patches_with_mask.png", dpi=300)
        plt.close(fig)

    def random_mim_generator(self, BATCH_SIZE):
        while True:
            # 1) raw patches
            X_raw = self.random_patch(BATCH_SIZE)

            # 2) augment
            if self.seq:
                X_aug = self.seq.to_deterministic().augment_images(X_raw)
            else:
                X_aug = X_raw

            # visualize raw once
            if self.visualize and not self._has_visualized_raw:
                self._show_patches(X_raw)
                self._has_visualized_raw = True

            # 3) normalize
            X = X_aug[..., self.input_image_channel].astype(np.float32) / 255.0
            X = (X - self.mean) / self.std

            # 4) MIM mask
            B, H, W, _ = X.shape
            gh, gw = H // self.token_size, W // self.token_size
            num_tokens = gh * gw
            mask = (np.random.rand(B, num_tokens) < self.mask_ratio)

            # visualize augmented + mask once
            if self.visualize and not self._has_visualized_aug:
                # note: pass the uint8 version for display
                self._show_augmented_patches(X_aug, mask)
                self._has_visualized_aug = True

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
    # This will write two files:
    #   raw_patches.png
    #   augmented_patches_with_mask.png
