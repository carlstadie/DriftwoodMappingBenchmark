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
    #for i in range(10):
    #    training.train_UNet(config)
    # training.train_UNet(config)

    # Strain a SwinUNet transformer model
    for i in range(10):
        training.train_SwinUNetPP(config)
    #training.train_SwinUNetPP(config)



