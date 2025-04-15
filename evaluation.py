import os
import glob
import math
import numpy as np
import rasterio
from tqdm import tqdm
import tensorflow as tf
from core.frame_info import FrameInfo
from core.optimizers import get_optimizer
from core.split_frames import split_dataset
from core.dataset_generator import DataGenerator as Generator
from core.losses import accuracy, dice_coef, dice_loss, specificity, sensitivity, get_loss


def get_all_frames(config):
    """Get all pre-processed frames which will be used for training."""

    # If no specific preprocessed folder was specified, use the most recent preprocessed data
    if config.preprocessed_dir is None:
        config.preprocessed_dir = os.path.join(
            config.preprocessed_base_dir,
            sorted(os.listdir(config.preprocessed_base_dir))[-1]
        )

    # Get paths of preprocessed images
    image_paths = sorted(
        glob.glob(f"{config.preprocessed_dir}/*.tif"),
        key=lambda f: int(f.split("/")[-1][:-4])
    )
    print(f"Found {len(image_paths)} input frames in {config.preprocessed_dir}")

    # Build a frame for each input image
    frames = []
    for im_path in tqdm(image_paths, desc="Processing frames"):
        # Open preprocessed image
        preprocessed = rasterio.open(im_path).read()

        # Get image channels (last two channels are labels + weights)
        image_channels = preprocessed[:-2, ::]

        # Transpose to have channels at the end
        image_channels = np.transpose(image_channels, axes=[1, 2, 0])

        # Get annotation and weight channels
        annotations = preprocessed[-2, ::]
        weights = preprocessed[-1, ::]

        # Create frame with combined image, annotation, and weight bands
        frames.append(FrameInfo(image_channels, annotations, weights))

    return frames



def create_train_val_datasets(frames, config):
    """Create the training, validation, and test datasets."""

    # If override set, ignore split and use all frames for everything
    if config.override_use_all_frames:
        training_frames = validation_frames = test_frames = list(range(len(frames)))
    else:
        frames_json = os.path.join(config.preprocessed_dir, "aa_frames_list.json")
        training_frames, validation_frames, test_frames = split_dataset(
            frames, frames_json, config.test_ratio, config.val_ratio
        )

    # Define input and annotation channels
    input_channels = list(range(len(config.channel_list)))
    label_channel = len(config.channel_list)     # because label and weights are directly after the input channels
    weight_channel = len(config.channel_list) + 1
    annotation_channels = [label_channel, weight_channel]

    # Define model patch size: Height * Width * (Input + Output) channels
    patch_size = [*config.patch_size, len(config.channel_list) + len(annotation_channels)]

    # Create DataGenerator instances for training, validation, and test data
    train_generator_instance = Generator(input_channels, patch_size, training_frames, frames, annotation_channels,
                                         augmenter='iaa', boundary_weight=config.boundary_weight)
    val_generator_instance = Generator(input_channels, patch_size, validation_frames, frames, annotation_channels,
                                       augmenter=None, boundary_weight=config.boundary_weight)
    test_generator_instance = Generator(input_channels, patch_size, test_frames, frames, annotation_channels,
                                        augmenter=None, boundary_weight=config.boundary_weight)

    return train_generator_instance, val_generator_instance, test_generator_instance



def evaluate_model(conf):
    """Evaluate the model based on the configuration."""
    global config
    config = conf

    # Pass config to get_all_frames
    frames = get_all_frames(config)

    # Pass both frames and config to create_train_val_datasets
    _, _, test_data_generator_instance = create_train_val_datasets(frames, config)

    # Calculate total frames directly from the DataGenerator instance
    total_frames = len(test_data_generator_instance.frame_list)

    # Calculate the number of steps
    batch_size = 32
    steps = math.ceil(total_frames / batch_size)

    # Load model
    model = tf.keras.models.load_model(
        config.evaluate_model_path,
        custom_objects={
            'dice_coef': dice_coef,
            'dice_loss': dice_loss,
            'accuracy': accuracy,
            'specificity': specificity,
            'sensitivity': sensitivity
        },
        compile=False  # Loading without compiling, so we compile after loading
    )

    # Compile the model with the same loss, optimizer, and metrics used during training
    model.compile(optimizer=get_optimizer(config.optimizer_fn),
                  loss=get_loss(config.loss_fn, config.tversky_alphabeta),
                  metrics=[dice_coef, dice_loss, specificity, sensitivity, accuracy])

    # Evaluate the model using the generator created by test_data_generator_instance
    metrics = model.evaluate(
        test_data_generator_instance.random_generator(batch_size, config.normalise_ratio), 
        steps=steps
    )

    # Metric names
    metric_names = ['Loss', 'Dice Coefficient', 'Dice Loss', 'Specificity', 'Sensitivity', 'Accuracy']

    # Print metrics with names
    print("Evaluation Metrics:")
    for name, value in zip(metric_names, metrics):
        print(f"{name}: {value:.4f}")

    # Ensure the evaluation path exists
    if not os.path.exists(config.evaluation_path):
        os.makedirs(config.evaluation_path)

    # Save metrics and model path to a file
    output_file = os.path.join(config.evaluation_path, "evaluation_results.txt")
    with open(output_file, "w") as f:
        f.write(f"Model Path: {config.evaluate_model_path}\n")
        f.write("Evaluation Metrics:\n")
        for name, value in zip(metric_names, metrics):
            f.write(f"{name}: {value:.4f}\n")

    print(f"Evaluation results saved to {output_file}")


