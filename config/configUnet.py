import os
import warnings
import numpy as np
from osgeo import gdal


class Configuration:
    """ Configuration of all parameters used in preprocessing.py, training.py and prediction.py """
    def __init__(self):

        # --------- RUN NAME ---------
        self.run_name = 'Unet_new'                     # custom name for this run, eg resampled_x3, alpha60, new_train etc

        # ---------- PATHS -----------

        # Modality to be preprocessed

        self.modality = 'MACS'                          # 'MACS', 'PS', 'S2'

        # Path to training areas and polygons shapefiles
        self.training_data_dir = f'/isipd/projects/p_planetdw/data/methods_test/training/{self.modality}/'  # path to training data
        self.training_area_fn = 'aoi_utm_8a.gpkg'#'merged_rectangles.gpkg' 
        self.training_polygon_fn = 'dw_utm_8a.gpkg' #'merged_polygons.gpkg' 
        
        # Path to training images
        self.training_image_dir = f'/isipd/projects/p_planetdw/data/methods_test/training_images/{self.modality}/'  # path to training images
        # Output base path where all preprocessed data folders will be created, change paths depending on image modality
        self.preprocessed_base_dir = f'/isipd/projects/p_planetdw/data/methods_test/training_data/{self.modality}/'  # path to preprocessed data

        # Path to preprocessed data to use for this training
        # Preprocessed frames are a tif file per area, with bands [normalised img bands + label band]
        self.preprocessed_dir = None               # if set to None, it will use the most recent preprocessing data

        # Path to existing model to be used to continue training on [optional]
        self.continue_model_path = None 

        
        # Path where trained models and training logs will be stored
        self.saved_models_dir = '/isipd/projects/p_planetdw/data/methods_test/models'
        self.logs_dir = '/isipd/projects/p_planetdw/data/methods_test/logs'
        
       

        # ------- IMAGE CONFIG ---------
        # Image file type, used to find images for training and prediction.
        self.image_file_type = ".tif"              # supported are .tif and .jp2

        # Up-sampling factor to use during preprocessing and prediction. 1 ->no up-sampling, 2 ->double resolution, etc
        self.resample_factor = 1

        # Selection of channels to include.
        self.channels_used = [True, True, True, True]

        # ------ TRAINING CONFIG -------
        # Split of input frames into training, test and validation data   (train_ratio = 1 - test_ratio - val_ratio)
        self.test_ratio = 0.2
        self.val_ratio = 0.2

        # Model configuration
        self.patch_size = (256, 256)
        self.tversky_alphabeta = (0.5, 0.5)        # alpha is weight of false positives, beta weight of false negatives

        # Batch and epoch numbers
        self.train_batch_size = 16
        self.num_epochs = 150
        self.num_training_steps = 500
        self.num_validation_images = 50

        # --- POSTPROCESSING CONFIG ----
        self.create_polygons = True                # To polygonize the raster predictions to polygon VRT
        self.postproc_workers = 12                 # number of CPU threads for parallel processing of polygons/centroids


        # ------ ADVANCED SETTINGS ------
        # GPU selection, if you have multiple GPUS.
        # Used for both training and prediction, so use multiple config files to run on two GPUs in parallel.
        self.selected_GPU = 6 # =CUDA id, 0 is first.    -1 to disable GPU and use CPU

        # Preprocessing
        self.train_image_type = self.image_file_type           # used to find training images
        self.train_image_prefix = ''               # to filter only certain images by prefix, eg ps_
        self.preprocessing_bands = np.where(self.channels_used)[0]         # [0, 1, 2, 3] etc
        self.preprocessed_name = self.run_name
        self.rasterize_borders = False             # whether to include borders when rasterizing label polygons


        # Training
        self.loss_fn = 'tversky'                   # selection of loss function
        self.optimizer_fn = 'adaDelta'             # selection of optimizer function
        self.dilation_rate = 1                  # dilation rate for dilated convolutions, 1 is no dilation
        self.model_name = self.run_name            # this is used as saved model name (concat with timestamp)
        self.boundary_weight = 5                  # weighting applied to boundaries, (rest of image is 1)
        self.model_save_interval = None            # [optional] save model every N epochs. If None, only best is saved
        self.channel_list = self.preprocessing_bands
        self.input_shape = (self.patch_size[0], self.patch_size[1], len(self.channel_list))

        # Prediction
        self.predict_images_file_type = self.image_file_type  # used to find images to predict
        self.predict_images_prefix = ''            # to filter only certain images by prefix, eg ps_
        self.overwrite_analysed_files = False      # whether to overwrite existing files previously predicted
        self.prediction_name = self.run_name       # this is used in the prediction folder name (concat with timestamp)
        self.prediction_output_dir = None         # set dynamically at prediction time
        self.prediction_patch_size = None          # if set to None, patch size is automatically read from loaded model
        self.prediction_operator = "MAX"           # "MAX" or "MIN": used to choose value for overlapping predictions
        self.output_prefix = 'det_' + self.prediction_name + '_'
        self.output_dtype = 'bool'                 # 'bool' is smallest size, 'uint8' has nodata (255), 'float32' is raw

        # Set overall GDAL settings
        gdal.UseExceptions()                       # Enable exceptions, instead of failing silently
        gdal.SetCacheMax(32000000000)              # IO cache size in KB, used when warping/resampling. higher is better
        gdal.SetConfigOption('CPL_LOG', '/dev/null')
        warnings.filterwarnings('ignore')          # Disable warnings

        # Set up tensorflow environment variables before importing tensorflow
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'   # Hide TF logs.  [Levels: 0->DEBUG, 1->INFO, 2->WARNING, 3->ERROR]
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.selected_GPU)

    def validate(self):
        """Validate config to catch errors early, and not during or at the end of processing"""

        # Check that training data paths exist
        if not os.path.exists(self.training_data_dir):
            raise ConfigError(f"Invalid path: config.training_data_dir = {self.training_data_dir}")
        if not os.path.exists(os.path.join(self.training_data_dir, self.training_area_fn)):
            raise ConfigError(f"File not found: {os.path.join(self.training_data_dir, self.training_area_fn)}")
        if not os.path.exists(os.path.join(self.training_data_dir, self.training_polygon_fn)):
            raise ConfigError(f"File not found: {os.path.join(self.training_data_dir, self.training_polygon_fn)}")
        if not os.path.exists(self.training_image_dir):
            raise ConfigError(f"Invalid path: config.training_image_dir = {self.training_image_dir}")

        # Create required output folders if not existing
        for config_dir in ["preprocessed_base_dir", "saved_models_dir", "logs_dir"]:
            if not os.path.exists(getattr(self, config_dir)):
                try:
                    os.mkdir(getattr(self, config_dir))
                except OSError:
                    raise ConfigError(f"Unable to create folder config.{config_dir} = {getattr(self, config_dir)}")

        # Check valid output formats
        if self.predict_images_file_type not in [".tif", ".jp2"]:
            raise ConfigError("Invalid format for config.predict_images_file_type. Supported formats are .tif and .jp2")
        if self.output_dtype not in ["bool", "uint8", "float32"]:
            raise ConfigError("Invalid format for config.output_dtype: Must be one of 'bool', 'uint8' and 'float32'"
                              "\n['bool' writes as binary data for smallest file size, but no nodata values. 'uint8' "
                              "writes background as 0, trees as 1 and nodata value 255 for missing/masked areas. "
                              "'float32' writes the raw prediction values, ignoring config.prediction_threshold.] ")

        # Check that tensorflow can see the specified GPU
        import tensorflow as tf
        if not tf.compat.v1.config.list_physical_devices("GPU"):
            if int(self.selected_GPU) == -1:
                pass
            elif int(self.selected_GPU) == 0:
                raise ConfigError(f"Tensorflow cannot detect a GPU. Enable TF logging and fix the symlinks until "
                                  f"there are no more errors for .so libraries that couldn't be loaded")
            else:
                raise ConfigError(f"Tensorflow cannot detect your GPU with CUDA id {self.selected_GPU}")

        return self


class ConfigError(Exception):
    pass
    
    
