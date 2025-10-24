# Main script to run preprocessing, training or prediction tasks.
# Convert this to a notebook if you are using Jupyter.

# This is where you can change which config to use
import config.configUnet as configuration

# INIT
config = configuration.Configuration().validate()

# Project modules
import preprocessing
import tuning
import training
#import prediction
import postprocessing
#import evaluation

# USE PYTORCH ENV FOR THIS PROJECT

if __name__ == "__main__":
    # PREPROCESSING
    #preprocessing.preprocess_all(config)

    # HYPERPARAMETER TUNING
    best = tuning.tune_UNet(config)

    config = tuning.apply_best_to_config(config, best, model_type='unet')

    # TRAINING
    # for i in range(10):
    training.train_UNet(config)
