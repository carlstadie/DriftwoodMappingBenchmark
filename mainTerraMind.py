# Main script to run preprocessing, training or prediction tasks.
# Convert this to a notebook if you are using jupyter.

# This is where you can change which config to use
import config.configTerraMind as configuration

# INIT
config = configuration.Configuration().validate()
import preprocessing
import training
import tuning
import postprocessing

# USE PYTORCH TORCH ENV FOR THIS PROJECT


if __name__ == "__main__":

    # PREPROCESSING
    #preprocessing.preprocess_all(config)

    # TUNING (HYPERPARAMETER SEARCH)
    best = tuning.tune_TerraMind(config)

    # TRAINING
    #training.train_TerraMind(config)
    





