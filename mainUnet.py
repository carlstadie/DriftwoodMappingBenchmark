# Main script to run preprocessing, training or prediction tasks.
# Convert this to a notebook if you are using jupyter.

# This is where you can change which config to use
import config.configUnet as configuration

# INIT
config = configuration.Configuration().validate()
import preprocessing
import training
import prediction
import postprocessing
import evaluation
import tuning 

# USE TENSORFLOR TF2 ENV FOR THIS PROJECT


if __name__ == "__main__":

    # PREPROCESSING
    #preprocessing.preprocess_all(config)

    #HYPERPARAMETER TUNING
    best = tuning.tune_UNet(config)
    # TRAINING

    #for i in range(10):
        #training.train_UNet(config)




