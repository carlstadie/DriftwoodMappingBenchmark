# config/configSwinUnet.py

import os
import warnings
import numpy as np
from osgeo import gdal


class Configuration:
    """
    Configuration of all parameters used in preprocessing.py, training.py
    and prediction.py (Swin-UNet variant).
    """

    def __init__(self):
        # --------- RUN NAME ---------
        self.run_name = "MACS_test"

        # ---------- PATHS -----------
        # Modality to be preprocessed (e.g. 'MACS', 'PS', 'S2', 'aerial', ...)
        self.modality = "PS"

        # Path to training areas and polygons shapefiles
        self.training_data_dir = (
            f"/isipd/projects/p_planetdw/data/methods_test/training/{self.modality}/"
        )
        self.training_area_fn = "aoi_utm_8a.gpkg"
        self.training_polygon_fn = "dw_utm_8p.gpkg"

        # Path to training images
        self.training_image_dir = (
            f"/isipd/projects/p_planetdw/data/methods_test/training_images/{self.modality}/"
        )

        # Output base path where preprocessed data folders will be created
        self.preprocessed_base_dir = (
            f"/isipd/projects/p_planetdw/data/methods_test/training_data/{self.modality}/"
        )

        # Path to preprocessed data used for this training.
        # If set to None, the most recent preprocessing data is used.
        self.preprocessed_dir = (
            "/isipd/projects/p_planetdw/data/methods_test/training_data/MACS/"
            "20250429-1208_MACS_test_utm8"
        )

        # Path to existing model to continue training on [optional]
        self.continue_model_path = None

        # Paths where trained models and training logs will be stored
        self.saved_models_dir = "/isipd/projects/p_planetdw/data/methods_test/models"
        self.logs_dir = "/isipd/projects/p_planetdw/data/methods_test/logs"

        # ------- IMAGE CONFIG ---------
        # Image file type used to find images for training and prediction.
        # Supported: .tif and .jp2
        self.image_file_type = ".tif"

        # Up-sampling factor during preprocessing/prediction.
        # 1 -> no up-sampling, 2 -> double resolution, etc.
        self.resample_factor = 1

        # Selection of channels to include (bool mask)
        self.channels_used = [True, True, True, True]

        # ------ TRAIN/VAL/TEST SPLIT ------
        # (train_ratio = 1 - test_ratio - val_ratio)
        self.test_ratio = 0.2
        self.val_ratio = 0.2

        # ------ MODEL CORE ------
        self.patch_size = (256, 256)
        # Optional tuner overrides (explicitly present so hasattr checks are true)
        self.tune_patch_h = None
        self.tune_patch_w = None
        # alpha is weight of false positives, beta weight of false negatives
        self.tversky_alphabeta = (0.5, 0.5)
        # Kept for parity with UNet (Swin ignores this)
        self.dilation_rate = 1

        # Used as saved model name (concat with timestamp)
        self.model_name = self.run_name

        # Channels actually used (indices); derived from channels_used
        self.preprocessing_bands = np.where(self.channels_used)[0]
        self.channel_list = list(self.preprocessing_bands)
        self.input_shape = (
            self.patch_size[0],
            self.patch_size[1],
            len(self.channel_list),
        )

        # ------ OPTIM / SCHED / EPOCHS ------
        self.loss_fn = "tversky"        # selection of loss function
        self.optimizer_fn = "adam"      # selection of optimizer function
        self.train_batch_size = 64
        self.num_epochs = 150
        self.num_training_steps = 500   # steps per epoch (train)
        self.num_validation_images = 50 # steps per epoch (val)

        # ------ EMA (Exponential Moving Average) ------
        self.use_ema = True
        self.ema_decay = 0.999
        self.eval_with_ema = True

        # ------ CHECKPOINTING / LOGGING ------
        # [optional] save model every N epochs. If None, only best is saved
        self.model_save_interval = None
        self.overfit_one_batch = False
        # Console verbosity / progress bars
        self.train_verbose = True
        self.train_epoch_log_every = 1   # print every N epochs
        self.train_print_heavy = True    # print heavy val metrics block
        self.show_progress = True        # tqdm bars for train/val loops
        # TensorBoard visual logging
        self.log_visuals_every = 5       # STEP interval (0 disables)
        self.vis_rgb_idx = (0, 1, 2)
        self.viz_pos_color = (1.0, 1.0, 0.0)  # class 1 -> yellow
        self.viz_neg_color = (0.0, 0.0, 1.0)  # class 0 -> blue

        # ------ AUG / SAMPLING / DATALOADER ------
        self.augmenter_strength = 0.7
        self.min_pos_frac = 0.02         # minimum fraction of positive pixels
        self.pos_ratio = 0.5             # fraction of batch from positive candidates
        self.patch_stride = None         # None -> generator default
        self.fit_workers = 8             # DataLoader workers
        self.steps_per_execution = 1     # grad accumulation steps

        # ------ EVALUATION ------
        self.eval_threshold = 0.5
        self.heavy_eval_steps = 50
        # Print positive-rate stats of splits (if you wired the helper)
        self.print_pos_stats = True

        # ------ MIXED PRECISION / COMPILE / REPRO ------
        self.use_torch_compile = False   # PyTorch 2.0+ compile()
        self.seed = None                 # int for reproducibility; None -> random
        self.clip_norm = 0.0             # gradient clipping (0 disables)

        # ------ SWIN-UNET (PP) ------
        self.swin_patch_size = 16        # patch size used by Swin stages
        self.swin_window = 4             # Swin window size for hierarchy multiple
        self.swin_levels = 3             # # of downsample levels (affects multiple)
        self.swin_base_channels = 64     # base channel width for Swin backbone
        # (Optional, picked up by training if present)
        # self.swin_ss_size = 2
        # self.swin_attn_drop = 0.0
        # self.swin_proj_drop = 0.0
        # self.swin_mlp_drop = 0.0
        # self.swin_drop_path = 0.1

        # --- POSTPROCESSING CONFIG ----
        # Polygonize raster predictions to polygon VRT
        self.create_polygons = True
        # CPU threads for polygon/centroid processing
        self.postproc_workers = 12

        # Prediction
        self.predict_images_file_type = self.image_file_type
        self.predict_images_prefix = ""
        self.overwrite_analysed_files = False
        self.prediction_name = self.run_name
        self.prediction_output_dir = None
        self.prediction_patch_size = None  # if None, read from model
        self.prediction_operator = "MAX"   # "MAX" or "MIN" for overlaps
        self.output_prefix = "det_" + self.prediction_name + "_"
        # 'bool' is smallest size, 'uint8' has nodata (255), 'float32' is raw
        self.output_dtype = "bool"

        # ------ ADVANCED SETTINGS ------
        # GPU selection. Used for both training and prediction.
        # = CUDA id, 0 is first.  -1 to disable GPU and use CPU.
        self.selected_GPU = 2

        # Preprocessing
        # used to find training images
        self.train_image_type = self.image_file_type
        # filter only certain images by prefix, eg "ps_"
        self.train_image_prefix = ""
        # used in the preprocessing folder name
        self.preprocessed_name = self.run_name
        # whether to include borders when rasterizing label polygons
        self.rasterize_borders = False

        # Set overall GDAL settings
        gdal.UseExceptions()                    # Enable exceptions
        gdal.SetCacheMax(32000000000)           # IO cache size in KB
        gdal.SetConfigOption("CPL_LOG", "/dev/null")
        warnings.filterwarnings("ignore")

        # ----- CUDA device visibility (PyTorch) -----
        # If CPU only requested, hide all GPUs. Otherwise, expose the selected one.
        if int(self.selected_GPU) == -1:
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.selected_GPU)

    def validate(self):
        """
        Validate config to catch errors early, and not during or at the end
        of processing.
        """
        # Check that training data paths exist
        if not os.path.exists(self.training_data_dir):
            raise ConfigError(
                f"Invalid path: config.training_data_dir = {self.training_data_dir}"
            )
        if not os.path.exists(os.path.join(self.training_data_dir, self.training_area_fn)):
            raise ConfigError(
                f"File not found: "
                f"{os.path.join(self.training_data_dir, self.training_area_fn)}"
            )
        if not os.path.exists(
            os.path.join(self.training_data_dir, self.training_polygon_fn)
        ):
            raise ConfigError(
                "File not found: "
                f"{os.path.join(self.training_data_dir, self.training_polygon_fn)}"
            )
        if not os.path.exists(self.training_image_dir):
            raise ConfigError(
                f"Invalid path: config.training_image_dir = {self.training_image_dir}"
            )

        # Create required output folders if not existing
        for config_dir in ["preprocessed_base_dir", "saved_models_dir", "logs_dir"]:
            target = getattr(self, config_dir)
            if not os.path.exists(target):
                try:
                    os.mkdir(target)
                except OSError as exc:
                    raise ConfigError(
                        f"Unable to create folder config.{config_dir} = {target}"
                    ) from exc

        # Check valid output formats
        if self.predict_images_file_type not in [".tif", ".jp2"]:
            raise ConfigError(
                "Invalid format for config.predict_images_file_type. "
                "Supported formats are .tif and .jp2"
            )
        if self.output_dtype not in ["bool", "uint8", "float32"]:
            raise ConfigError(
                "Invalid format for config.output_dtype: Must be one of "
                "'bool', 'uint8' and 'float32'\n"
                "['bool' writes as binary data for smallest file size, but no "
                "nodata values. 'uint8' writes background as 0, trees as 1 and "
                "nodata value 255 for missing/masked areas. 'float32' writes "
                "the raw prediction values, ignoring config.prediction_threshold.]"
            )

        # ---- GPU availability check (PyTorch) ----
        # If GPU requested (selected_GPU != -1), ensure CUDA is available.
        if int(self.selected_GPU) != -1:
            try:
                import torch
                if not torch.cuda.is_available():
                    raise ConfigError(
                        "PyTorch cannot detect a CUDA-enabled GPU. "
                        "Check your CUDA driver/toolkit installation."
                    )
                # After setting CUDA_VISIBLE_DEVICES, device 0 should be valid.
                _ = torch.cuda.get_device_name(0)
            except Exception as exc:
                raise ConfigError(
                    f"PyTorch cannot access the requested GPU with CUDA id "
                    f"{self.selected_GPU}. Environment/CUDA issue: {exc}"
                ) from exc

        return self


class ConfigError(Exception):
    pass
