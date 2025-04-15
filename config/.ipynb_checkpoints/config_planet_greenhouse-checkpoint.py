import os
import warnings
import numpy as np
from osgeo import gdal


class Configuration:
    """ Configuration of all parameters used in preprocessing.py, training.py and prediction.py """
    def __init__(self):
        
        self.log_img = 1 # log images
        self.callbackImSave = '../log_images/'
        # --------- RUN NAME ---------
        self.run_name = 'planet_256_nonorm'                     # custom name for this run, eg resampled_x3, alpha60, new_train etc

        # ---------- PATHS -----------
        # Path to training areas and polygons shapefiles
        self.training_data_dir = '/mnt/sddb/GreenHouse/code/training/'
        self.training_area_fn = 'training_areas_v1.shp'
        self.training_polygon_fn = 'training_polys_v1.shp'

        # Path to training images
        #self.training_image_dir = '/mnt/nfs/users/Xiaoye/greenhouse/input/merged1/'
        self.training_image_dir = '/mnt/sddb/GreenHouse/code/merged/'

        # Output base path where all preprocessed data folders will be created
        self.preprocessed_base_dir = '/mnt/sddb/GreenHouse/code/preprocessed_frames/'

        # Path to preprocessed data to use for this training
        # Preprocessed frames are a tif file per area, with bands [normalised img bands + label band + boundaries band]
        self.preprocessed_dir = None               # if set to None, it will use the most recent preprocessing data

        # Path to existing model to be used to continue training on [optional]
        self.continue_model_path = '/mnt/sddb/GreenHouse/code/saved_models/20220623-1708_planet_256_nonorm.h5'            # If set to None, a new model is created

        # Path where trained models and training logs will be stored
        self.saved_models_dir = '/mnt/sddb/GreenHouse/code/saved_models/'
        self.logs_dir = '/mnt/sddb/GreenHouse/code/logs/'

        # Paths to input images to be predicted
        self.to_predict_dir = '/mnt/Users/Xiaoye/greenhouse/input/merged0/'
        #self.to_predict_dir = '/mnt/warehouse/greenhouse/merged/'
        self.to_predict_filelist = None #'/mnt/super/greenhouse/europe.txt'            # [optional] path of a file with list of image paths to be predicted

        # Output path where predictions will be saved
        self.predictions_base_dir = '/mnt/md0/GreenHouse/output/'

        # Path to model to use for this prediction [optional]
        self.trained_model_path = None #'../saved_models/20220723-2328_planet_256_nonorm.h5'             # if set to None, it will use the most recent trained model

        # Path to prediction folder to be postprocessed
        self.postprocessing_dir = None             # if set to None, it will run postprocessing on the last predictions

        # ------- IMAGE CONFIG ---------
        # Image file type, used to find images for training and prediction.
        self.image_file_type = ".jp2"              # supported are .tif and .jp2

        # Up-sampling factor to use during preprocessing and prediction. 1 ->no up-sampling, 2 ->double resolution, etc
        self.resample_factor = 1

        # Selection of channels to include. (for planet, order is blue, green, red, infrared, ndvi)
        self.channels_used = [True, True, True,True,True]


        # ------ TRAINING CONFIG -------
        # Split of input frames into training, test and validation data   (train_ratio = 1 - test_ratio - val_ratio)
        self.test_ratio = 0.2
        self.val_ratio = 0.2
        self.override_use_all_frames = True        # If True, above two ratios are ignored and train=test=val=all_frames

        # Model configuration
        self.patch_size = (256, 256)
        self.tversky_alphabeta = (0.5, 0.5)        # alpha is weight of false positives, beta weight of false negatives

        # Batch and epoch numbers
        self.train_batch_size = 32
        self.num_epochs = 1200
        self.num_training_steps = 200
        self.num_validation_images = 50

        # Normalisation ratio: probability of patches being normalised (0: don't normalise, 1: normalise all)
        self.normalise_ratio = 0

        # ----- PREDICTION CONFIG ------
        self.prediction_workers = 1                # number of prediction processes predicting image chunks in parallel
        self.prediction_gridsize = (1, 1)          # num rows/cols in which images are split for parallel processing
        self.prediction_batch_size = 200           # Depends on GPU memory, patch size and num parallel workers
        self.prediction_stride = 192               # stride = width -> no overlap, stride = width/2 -> 50 % overlap
        self.prediction_threshold = 0.5            # threshold applied when converting float predictions to binary
        self.prediction_mask_fps = []              # [optional] list of polygon mask files to limit prediction area

        # --- POSTPROCESSING CONFIG ----
        self.create_polygons = True                # To polygonize the raster predictions to polygon VRT
        self.create_centroids = True               # [needs polygons] To create centroids from polygons, with area in m2
        self.create_density_maps = True            # [needs centroids] To create tree density maps by crown area classes
        self.create_canopy_cover_maps = False       # To create canopy cover maps from raster predictions
        self.postproc_workers = 12                 # number of CPU threads for parallel processing of polygons/centroids
        self.postproc_gridsize = (8, 8)            # num rows/cols in which images are split for parallel processing
        self.canopy_resolutions = [100]            # resolutions of canopy cover maps to create, in m
        self.density_resolutions = [100]           # resolutions of density maps to create, in m
        self.area_thresholds = [20, 50, 100, 200, 300]    # thresholds of area classes used for bands in density maps, in m2

        # ------ ADVANCED SETTINGS ------
        # GPU selection, if you have multiple GPUS.
        # Used for both training and prediction, so use multiple config files to run on two GPUs in parallel.
        self.selected_GPU = 0                      # =CUDA id, 0 is first.    -1 to disable GPU and use CPU

        # Preprocessing
        self.train_image_type = self.image_file_type           # used to find training images
        self.train_image_prefix = ''               # to filter only certain images by prefix, eg ps_
        self.preprocessing_bands = np.where(self.channels_used)[0]         # [0, 1, 2, 3] etc
        self.preprocessed_name = self.run_name
        self.rasterize_borders = False             # whether to include borders when rasterizing label polygons
        self.boundary_scale = 1.2                  # scale factor to apply when finding polygon boundary weights
        self.get_json = False			     # whether to save a separate json file w/ annotation information for frames
 # =============================================================================
        # # adding density counting branch
        
        self.counting_branch = False
        self.counting_kernel_size = 5
        self.counting_kernel_sigma = 1.5
        
# =============================================================================


        # Training
        self.loss_fn = 'tversky'                   # selection of loss function
        self.optimizer_fn = 'adam'             # selection of optimizer function
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
        self.prediction_output_dir = "/mnt/md0/GreenHouse/output/20220913-1151_planet_256_nonorm/"#'../output_predictions/20220627-1040_planet_256_nonorm/'          # set dynamically at prediction time
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
        for config_dir in ["preprocessed_base_dir", "saved_models_dir", "logs_dir", "predictions_base_dir"]:
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
    
    
