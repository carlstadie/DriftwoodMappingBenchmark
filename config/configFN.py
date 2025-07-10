import os
import warnings
import numpy as np
from osgeo import gdal
import torch
import torch.nn as nn
import torch.optim as optim


class Configuration:
    """ Configuration of all parameters used in preprocessing.py, training.py and prediction.py (PyTorch edition) """
    def __init__(self):

        # --------- RUN NAME ---------
        self.run_name = 'Unet_Planet_utm8_pytorch'

        # ---------- PATHS -----------
        self.modality = 'PS'
        self.training_data_dir = f'/isipd/projects/p_planetdw/data/methods_test/training/{self.modality}/'
        self.training_area_fn = 'aoi_utm_8a.gpkg'
        self.training_polygon_fn = 'dw_utm_8p.gpkg'
        self.training_image_dir = f'/isipd/projects/p_planetdw/data/methods_test/training_images/{self.modality}/'
        self.preprocessed_base_dir = f'/isipd/projects/p_planetdw/data/methods_test/training_data/{self.modality}/'
        self.preprocessed_dir = (
            f'/isipd/projects/p_planetdw/data/methods_test/training_data/{self.modality}/'
            '20250604-0816_Unet_Planet_utm8'
        )
        self.continue_model_path = None
        self.saved_models_dir = '/isipd/projects/p_planetdw/data/methods_test/models'
        self.logs_dir = '/isipd/projects/p_planetdw/data/methods_test/logs'

        # ------- IMAGE CONFIG ---------
        self.image_file_type = ".jp2"
        self.resample_factor = 3
        self.channels_used = [True, True, True, True]

        # ------ TRAIN/VAL SPLIT -------
        self.test_ratio = 0.2
        self.val_ratio = 0.2

        # ------- MODEL / TRAINING -------
        self.patch_size = (256, 256)
        self.tversky_alphabeta = (0.7, 0.3)

        # Batch & epochs
        self.train_batch_size = 16
        self.num_epochs = 150
        self.num_training_steps = 500
        self.num_validation_images = 50

        # --- POSTPROCESSING CONFIG ----
        self.create_polygons = True
        self.postproc_workers = 12

        # ------ ADVANCED SETTINGS ------
        # GPU selection
        self.selected_gpu = 7  # CUDA device index, -1 for CPU

        # Preprocessing
        self.train_image_type = self.image_file_type
        self.train_image_prefix = ''
        self.preprocessing_bands = np.where(self.channels_used)[0]
        self.preprocessed_name = self.run_name
        self.rasterize_borders = False

        # Training specifics
        self.loss_fn = 'tversky'      # 'tversky', 'cross_entropy', etc
        self.optimizer_fn = 'adam'    # 'adam', 'sgd', etc
        self.learning_rate = 1e-3
        self.dilation_rate = 1
        self.model_name = self.run_name
        self.boundary_weight = 5
        self.model_save_interval = None
        self.channel_list = self.preprocessing_bands
        self.input_shape = (len(self.channel_list), *self.patch_size)

        # Prediction
        self.predict_images_file_type = self.image_file_type
        self.predict_images_prefix = ''
        self.overwrite_analysed_files = False
        self.prediction_name = self.run_name
        self.prediction_output_dir = None
        self.prediction_patch_size = None
        self.prediction_operator = "MAX"
        self.output_prefix = 'det_' + self.prediction_name + '_'
        self.output_dtype = 'bool'

        # GDAL settings
        gdal.UseExceptions()
        gdal.SetCacheMax(32000000000)
        gdal.SetConfigOption('CPL_LOG', '/dev/null')
        warnings.filterwarnings('ignore')

        # Seed for reproducibility
        np.random.seed(42)
        torch.manual_seed(42)

        # Set up device
        if self.selected_gpu >= 0 and torch.cuda.is_available():
            torch.cuda.set_device(self.selected_gpu)
            self.device = torch.device(f'cuda:{self.selected_gpu}')
            # cuDNN settings
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        else:
            self.device = torch.device('cpu')

    def get_loss(self) -> nn.Module:
        """Return the loss criterion."""
        if self.loss_fn == 'tversky':
            # placeholder: you'd implement your TverskyLoss class
            return TverskyLoss(alpha=self.tversky_alphabeta[0],
                               beta=self.tversky_alphabeta[1],
                               weight_boundary=self.boundary_weight)
        elif self.loss_fn == 'cross_entropy':
            return nn.CrossEntropyLoss()
        else:
            raise ValueError(f"Unknown loss function: {self.loss_fn}")

    def get_optimizer(self, model_parameters):
        """Return the optimizer."""
        if self.optimizer_fn.lower() == 'adam':
            return optim.Adam(model_parameters, lr=self.learning_rate)
        elif self.optimizer_fn.lower() == 'sgd':
            return optim.SGD(model_parameters, lr=self.learning_rate, momentum=0.9)
        else:
            raise ValueError(f"Unknown optimizer: {self.optimizer_fn}")

    def validate(self):
        """Validate config and create dirs."""
        # (same filesystem checks as before)
        required_paths = [
            self.training_data_dir,
            os.path.join(self.training_data_dir, self.training_area_fn),
            os.path.join(self.training_data_dir, self.training_polygon_fn),
            self.training_image_dir
        ]
        for p in required_paths:
            if not os.path.exists(p):
                raise ConfigError(f"Invalid or missing path: {p}")

        for attr in ["preprocessed_base_dir", "saved_models_dir", "logs_dir"]:
            d = getattr(self, attr)
            if not os.path.exists(d):
                os.makedirs(d, exist_ok=True)

        if self.predict_images_file_type not in [".tif", ".jp2"]:
            raise ConfigError("config.predict_images_file_type must be .tif or .jp2")
        if self.output_dtype not in ["bool", "uint8", "float32"]:
            raise ConfigError("config.output_dtype must be 'bool', 'uint8' or 'float32'")

        return self


class ConfigError(Exception):
    pass
