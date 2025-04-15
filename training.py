import os
import json
import time
import glob
import shutil
from datetime import datetime, timedelta

import h5py
import rasterio
import numpy as np
from tqdm import tqdm

import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, Callback
from core.UNet import UNet
from core.frame_info import FrameInfo
from core.optimizers import get_optimizer
from core.split_frames import split_dataset
from core.dataset_generator import DataGenerator as Generator
from core.losses import accuracy, dice_coef, dice_loss, specificity, sensitivity, get_loss


def get_all_frames():
    """Get all pre-processed frames which will be used for training."""

    # If no specific preprocessed folder was specified, use the most recent preprocessed data
    if config.preprocessed_dir is None:
        config.preprocessed_dir = os.path.join(config.preprocessed_base_dir,
                                               sorted(os.listdir(config.preprocessed_base_dir))[-1])

    # Get paths of preprocessed images
    image_paths = sorted(glob.glob(f"{config.preprocessed_dir}/*.tif"), key=lambda f: int(f.split("/")[-1][:-4]))
    print(f"Found {len(image_paths)} input frames in {config.preprocessed_dir}")

    # Build a frame for each input image
    frames = []
    for im_path in tqdm(image_paths, desc="Processing frames"):

        # Open preprocessed image
        preprocessed = rasterio.open(im_path).read()

        # Get image channels   (last two channels are labels + weights)
        # C | H | W
        image_channels = preprocessed[:-1, ::]

        # Transpose to have channels at the end
        # C | H | W -> H | W | C
        image_channels = np.transpose(image_channels, axes=[1, 2, 0])

        # Get annotation and weight channels
        # C | H | W
        annotations = preprocessed[-1, ::]

        # Create frame with combined image, annotation, and weight bands
        frames.append(FrameInfo(image_channels, annotations))

    return frames


def create_train_val_datasets(frames):
    """ Create the training, validation and test datasets """

    # Split the dataset into training, validation and test datasets
    frames_json = os.path.join(config.preprocessed_dir, "aa_frames_list.json")
    training_frames, validation_frames, test_frames = split_dataset(frames, frames_json, config.test_ratio,
                                                                    config.val_ratio)

    # Define input and annotation channels
    input_channels = list(range(len(config.channel_list)))
    label_channel = len(config.channel_list)     # because label and weights are directly after the input channels

    # Define model patch size: Height * Width * (Input + Output) channels
    patch_size = [*config.patch_size, len(config.channel_list) + 1]  # +1 for the label channel

    # Create generators for training, validation and test data
    train_generator = Generator(input_channels, patch_size, training_frames, frames, label_channel,
                                augmenter='iaa').random_generator(
                                config.train_batch_size)
    val_generator = Generator(input_channels, patch_size, validation_frames, frames, label_channel,
                              augmenter=None).random_generator(
                              config.train_batch_size)
    test_generator = Generator(input_channels, patch_size, test_frames, frames, label_channel,
                               augmenter=None).random_generator(
                               config.train_batch_size)
    
    return train_generator, val_generator, test_generator


def create_callbacks(model_path):
    """ Define callbacks for the early stopping of training, LearningRateScheduler and model checkpointing"""

    # Add checkpoint callback to save model during training.
    checkpoint = ModelCheckpoint(model_path, monitor='val_loss', verbose=1,
                                 save_best_only=True, mode='min', save_weights_only=False)

    # Add tensorboard callback to follow training progress
    log_dir = os.path.join(config.logs_dir, os.path.basename(model_path)[:-3])
    tensorboard = TensorBoard(log_dir=log_dir, histogram_freq=0, write_graph=True, write_grads=False, embeddings_freq=0,
                              write_images=False, embeddings_layer_names=None, embeddings_metadata=None,
                              embeddings_data=None, update_freq='epoch', profile_batch='500,520')

    # Add a callback to store custom metadata in the model .h5 file
    # This allows us to later remember settings that were used when this model was trained (not only those in filename)
    # The metadata is saved at the end of every epoch to preserve info when training is ended early
    class CustomMeta(Callback):
        def __init__(self):
            super().__init__()
            self.start_time = datetime.now()

        def on_epoch_end(self, epoch, logs=None):
            # Create object with all custom metadata
            meta_data = {
                "name": config.model_name,
                "model_path": model_path,
                "patch_size": config.patch_size,
                "channels_used": config.channels_used,
                "resample_factor": config.resample_factor,
                "frames_dir": config.preprocessed_dir,
                "train_ratio": float(f"{1-config.val_ratio-config.test_ratio:.2f}"),
                "val_ratio": config.val_ratio,
                "test_ratio": config.test_ratio,
                "normalise_ratio": config.normalise_ratio,
                "loss": config.loss_fn,
                "optimizer": config.optimizer_fn,
                "tversky_alpha": config.tversky_alphabeta[0],
                "tversky_beta": config.tversky_alphabeta[1],
                "batch_size": config.train_batch_size,
                "epoch_steps": config.num_training_steps,
                "val_steps": config.num_validation_images,
                "epochs_trained": f"{epoch + 1}/{config.num_epochs}",
                "last_sensitivity": float(f"{logs['sensitivity']:.4f}"),        # could also add other metrics if needed
                "start_time": self.start_time.strftime("%d.%m.%Y %H:%M:%S"),
                "elapsed_time": (datetime.utcfromtimestamp(0) + (datetime.now() - self.start_time)).strftime("%H:%M:%S")
            }
            # Serialise to json string and inject into the .h5 model file as an attribute
            with h5py.File(model_path, "a") as file:
                file.attrs["custom_meta"] = bytes(json.dumps(meta_data), "utf-8")
            # Optionally save the model at regular intervals
            if config.model_save_interval and (epoch + 1) % config.model_save_interval == 0:
                shutil.copy(model_path, model_path.replace(".h5", f"_{epoch+1}epochs.h5"))

    return [checkpoint, tensorboard, CustomMeta()]


def train_model(conf):
    """Create and train a new model"""
    global config
    config = conf

    print("Starting training.")
    start = time.time()

    # Get all training frames
    frames = get_all_frames()

    # Split into training, validation and test datasets (proportions are set in config)
    train_generator, val_generator, test_generator = create_train_val_datasets(frames)

    # Create model name from timestamp and custom name
    model_path = os.path.join(config.saved_models_dir, f"{time.strftime('%Y%m%d-%H%M')}_{config.model_name}.h5")
    starting_epoch = 0

    # Check if we want to continue training an existing model
    if config.continue_model_path is not None:

        # Load previous model
        print(f"Loading pre-trained model from {config.continue_model_path} :")
        model = tf.keras.models.load_model(config.continue_model_path,
                                           custom_objects={'tversky': get_loss('tversky', config.tversky_alphabeta),
                                                           'dice_coef': dice_coef, 'dice_loss': dice_loss,
                                                           'accuracy': accuracy, 'specificity': specificity,
                                                           'sensitivity': sensitivity}, compile=False)

        # Get starting epoch from metadata
        with h5py.File(config.continue_model_path, 'r') as model_file:
            if "custom_meta" in model_file.attrs:
                try:
                    custom_meta = json.loads(model_file.attrs["custom_meta"].decode("utf-8"))
                except:
                    custom_meta = json.loads(model_file.attrs["custom_meta"])
                starting_epoch = int(custom_meta["epochs_trained"].split("/")[0])

        # Copy logs from previous training so that tensorboard shows combined epochs
        old_log_dir = os.path.join(config.logs_dir, os.path.basename(config.continue_model_path)[:-3])
        new_log_dir = os.path.join(config.logs_dir, os.path.basename(model_path)[:-3])
        if os.path.exists(old_log_dir):
            shutil.copytree(old_log_dir, new_log_dir)

    # Otherwise define new model
    else:
        model = UNet([config.train_batch_size, *config.patch_size, len(config.channel_list)], [len(config.channel_list)])

    # Create callbacks to be used during training
    callbacks = create_callbacks(model_path)

    # Train the model
    tf.config.run_functions_eagerly(True)
    model.compile(optimizer=get_optimizer(config.optimizer_fn), loss=get_loss(config.loss_fn, config.tversky_alphabeta),
                  metrics=[dice_coef, dice_loss, specificity, sensitivity, accuracy])
    model.fit(train_generator,
              steps_per_epoch=config.num_training_steps,
              epochs=config.num_epochs,
              initial_epoch=starting_epoch,
              validation_data=val_generator,
              validation_steps=config.num_validation_images,
              callbacks=callbacks,
              workers=1)

    print(f"Training completed in {str(timedelta(seconds=time.time() - start)).split('.')[0]}.\n")
