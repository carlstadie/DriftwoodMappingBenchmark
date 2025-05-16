import os
import json
import time
import glob
import shutil
from datetime import datetime, timedelta

import rasterio
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import torch.hub

from core.UNet import UNet
from core.Swin_UNetPP import SwinUNet
from core.frame_info import FrameInfo
from core.optimizers import get_optimizer
from core.split_frames import split_dataset
from core.dataset_generator import DataGenerator as Generator
from core.losses import (
    get_loss, accuracy, dice_coef, dice_loss,
    specificity, sensitivity, f_beta, f1_score,
    IoU, nominal_surface_distance,
    Hausdorff_distance, boundary_intersection_over_union
)

 #Dataloading
def get_all_frames():
    """Get all pre-processed frames which will be used for training."""
    if config.preprocessed_dir is None:
        config.preprocessed_dir = os.path.join(
            config.preprocessed_base_dir,
            sorted(os.listdir(config.preprocessed_base_dir))[-1]
        )

    image_paths = sorted(
        glob.glob(f"{config.preprocessed_dir}/*.tif"),
        key=lambda f: int(os.path.basename(f)[:-4])
    )
    print(f"Found {len(image_paths)} input frames in {config.preprocessed_dir}")

    frames = []
    for im_path in tqdm(image_paths, desc="Processing frames"):
        arr = rasterio.open(im_path).read()  # [C, H, W]
        img = np.transpose(arr[:-1], (1, 2, 0))  # [H, W, C]
        ann = arr[-1]  # [H, W]
        frames.append(FrameInfo(img, ann))
    return frames


def create_train_val_datasets(frames):
    """Create the training, validation and test generators"""
    frames_json = os.path.join(config.preprocessed_dir, "aa_frames_list.json")
    train_idx, val_idx, test_idx = split_dataset(
        frames, frames_json, config.test_ratio, config.val_ratio
    )

    input_channels = list(range(len(config.channel_list)))
    label_channel = len(config.channel_list)
    patch_size = [*config.patch_size, len(config.channel_list) + 1]

    train_gen = Generator(
        input_channels, patch_size, train_idx, frames, label_channel,
        augmenter='iaa'
    ).random_generator(config.train_batch_size)
    val_gen = Generator(
        input_channels, patch_size, val_idx, frames, label_channel,
        augmenter=None
    ).random_generator(config.train_batch_size)
    test_gen = Generator(
        input_channels, patch_size, test_idx, frames, label_channel,
        augmenter=None
    ).random_generator(config.train_batch_size)

    return train_gen, val_gen, test_gen

