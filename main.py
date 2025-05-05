# Main script to run preprocessing, training or prediction tasks.
# Convert this to a notebook if you are using jupyter.

# This is where you can change which config to use
import config.config as configuration

# INIT
config = configuration.Configuration().validate()
import preprocessing
import training
import prediction
import postprocessing
import evaluation

if __name__ == "__main__":

    # PREPROCESSING
    #preprocessing.preprocess_all(config)

    # TRAINING

    # train a UNet model
    # training.train_UNet(config)

    # Strain a SwinUNet transformer model
    training.train_SwinUNetPP(config)



