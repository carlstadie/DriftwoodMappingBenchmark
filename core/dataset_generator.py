#    Author: Ankit Kariryaa, University of Bremen

from imgaug import augmenters as iaa
import numpy as np


# Sometimes(0.5, ...) applies the given augmenter in 50% of all cases,
# e.g. Sometimes(0.5, GaussianBlur(0.3)) would blur roughly every second image.
def imageAugmentationWithIAA():
    sometimes = lambda aug, prob=0.5: iaa.Sometimes(prob, aug)
    seq = iaa.Sequential([
        # Basic aug without changing any values
        iaa.Fliplr(0.5),  # horizontally flip 50% of all images
        iaa.Flipud(0.5),  # vertically flip 20% of all images
        sometimes(iaa.Crop(percent=(0, 0.1))),  # random crops
        #
        # # Gaussian blur and gamma contrast
        sometimes(iaa.GaussianBlur(sigma=(0, 0.3)), 0.3),
        # sometimes(iaa.GammaContrast(gamma=0.5, per_channel=True), 0.3),

        # iaa.CoarseDropout((0.03, 0.25), size_percent=(0.02, 0.05), per_channel=True)
        # sometimes(iaa.Multiply((0.75, 1.25), per_channel=True), 0.3),
        sometimes(iaa.LinearContrast((0.3, 1.2)), 0.3),
        # iaa.Add(value=(-0.5,0.5),per_channel=True),
        sometimes(iaa.PiecewiseAffine(0.05), 0.3),
        sometimes(iaa.PerspectiveTransform(0.01), 0.1)
    ],
        random_order=True)
    return seq


class DataGenerator:
    """The datagenerator class. Defines methods for generating patches randomly and sequentially from given frames.
    """

    def __init__(self, input_image_channel, patch_size, frame_list, frames, annotation_channel, augmenter=None):
        """Datagenerator constructor

        Args:
            input_image_channel (list(int)): Describes which channels is the image are input channels.
            patch_size (tuple(int,int)): Size of the generated patch.
            frame_list (list(int)): List containing the indexes of frames to be assigned to this generator.
            frames (list(FrameInfo)): List containing all the frames i.e. instances of the frame class.
            augmenter  (string, optional): augmenter to use. None for no augmentation and iaa for augmentations defined
            in imageAugmentationWithIAA function.
        """
        self.input_image_channel = input_image_channel
        self.patch_size = patch_size
        self.frame_list = frame_list
        self.frames = frames
        self.annotation_channel = annotation_channel
        self.augmenter = augmenter

        # Calculate area-based frame weights (to sample big frames more often, for even overall sampling coverage)
        total_area = sum([frames[i].img.shape[0] * frames[i].img.shape[1] for i in frame_list])
        self.frame_list_weights = [(frames[i].img.shape[0] * frames[i].img.shape[1])/total_area for i in frame_list]

    # Return all training and label images and weights, generated sequentially with the given step size
    def all_sequential_patches(self, step_size):
        """Generate all patches from all assigned frames sequentially.

            step_size (tuple(int,int)): Size of the step when generating frames.
            normalize (float): Probability with which a frame is normalized.
        """
        patches = []
        for fn in self.frame_list:
            frame = self.frames[fn]
            ps = frame.sequential_patches(self.patch_size, step_size)
            patches.extend(ps)
        data = np.array(patches)
        img = data[..., self.input_image_channel]
        ann = data[..., self.annotation_channel]
        return (img, ann)

    # Return a batch of training and label images, generated randomly
    def random_patch(self, BATCH_SIZE):
        """Generate patches from random location in randomly chosen frames.

        Args:
            BATCH_SIZE (int): Number of patches to generate (sampled independently).
        """
        patches = []
        for i in range(BATCH_SIZE):
            fn = np.random.choice(self.frame_list, p=self.frame_list_weights)
            frame = self.frames[fn]
            patch = frame.random_patch(self.patch_size)
            patches.append(patch)
        data = np.array(patches)

        img = data[..., self.input_image_channel]
        ann = data[..., self.annotation_channel]
        return (img, ann)
#     print("Wrote {} random patches to {} with patch size {}".format(count,write_dir,patch_size))

    # Normalization takes a probability between 0 and 1 that an image will be locally normalized.
    def random_generator(self, BATCH_SIZE):
        """Generator for random patches, yields random patches from random location in randomly chosen frames.

        Args:
            BATCH_SIZE (int): Number of patches to generate in each yield (sampled independently).
        """
        seq = imageAugmentationWithIAA()

        while True:
            X, y = self.random_patch(BATCH_SIZE)
            
            if self.augmenter == 'iaa':
                seq_det = seq.to_deterministic()
                X = seq_det.augment_images(X)
                # annotation chanel is the last channel neds to be augmented too, in case we are cropping or so
                y = seq_det.augment_images(y)
                # reassign value just to be sure
                ann =  y[...,[0]]
                ann[ann<0.5] = 0
                ann[ann>=0.5] = 1
                yield X, ann
            else:
                ann =  y[...,[0]]
                yield X, ann
