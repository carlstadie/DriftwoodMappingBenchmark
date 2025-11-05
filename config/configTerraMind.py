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
        self.modality = 'PS'  # 'S2', 'PS', or 'AE'
        self.training_data_dir = f'/isipd/projects/p_planetdw/data/methods_test/training/{self.modality}/'
        self.training_area_fn = 'aoi_utm_8a.gpkg'
        self.training_polygon_fn = 'dw_utm_8p.gpkg'
        self.training_image_dir = f'/isipd/projects/p_planetdw/data/methods_test/training_images/{self.modality}/'
        self.preprocessed_base_dir = f'/isipd/projects/p_planetdw/data/methods_test/training_data/{self.modality}/'
        self.preprocessed_dir = (

            "/isipd/projects/p_planetdw/data/methods_test/training_data/MACS/"
            "20250429-1208_MACS_test_utm8"
            
        )
        self.continue_model_path = None
        self.saved_models_dir = '/isipd/projects/p_planetdw/data/methods_test/models'
        self.logs_dir = '/isipd/projects/p_planetdw/data/methods_test/logs'

        # ------- IMAGE CONFIG ---------
        self.image_file_type = ".jp2"
        self.resample_factor = 3
        self.channels_used = [True, True, True, True]  # PS/AE: 4-band expected order [B,G,R,NIR]

        # ------ TRAIN/VAL SPLIT -------
        self.test_ratio = 0.2
        self.val_ratio = 0.2

        # ------- MODEL / TRAINING -------
        self.patch_size = (256, 256)
        self.tversky_alphabeta = (0.7, 0.3)

        # Batch & epochs
        self.train_batch_size = 16
        self.num_epochs = 100
        self.num_training_steps = 100
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
        self.optimizer_fn = 'adam'    # 'adam', 'adamw', 'sgd', etc
        self.learning_rate = 1e-3
        self.dilation_rate = 1
        self.model_name = self.run_name
        self.boundary_weight = 5
        self.model_save_interval = None
        self.channel_list = self.preprocessing_bands
        self.input_shape = (len(self.channel_list), *self.patch_size)

        # ---- Optional data gen knobs (used if present) ----
        self.augmenter_strength = 1.0   # 0.0 (off), 0.5, 1.0
        self.min_pos_frac = 0.0         # hard-min fraction of positive patches
        self.pos_ratio = None           # None => random; or float in (0,1)
        self.patch_stride = None        # None => random sampling; or int
        self.fit_workers = 8            # DataLoader workers

        # ---- Training loop QoL (used if present) ----
        self.use_torch_compile = False
        self.use_ema = False
        self.ema_decay = 0.999
        self.eval_with_ema = False
        self.eval_threshold = 0.5
        self.heavy_eval_steps = 50
        self.vis_rgb_idx = (0, 1, 2)         # which bands to show as RGB
        self.viz_pos_color = (1.0, 1.0, 0.0) # yellow for class 1
        self.viz_neg_color = (0.0, 0.0, 1.0) # blue for class 0
        self.log_visuals_every = 1           # steps; 0 disables
        self.steps_per_execution = 1         # grad accumulation
        self.clip_norm = 0.0                 # 0 disables
        self.overfit_one_batch = False
        self.train_verbose = True
        self.train_epoch_log_every = 1
        self.train_print_heavy = True
        self.show_progress = True

        # ---------------- TerraMind-specific ----------------
        # These are only used when training/tuning the TerraMind model.
        self.num_classes = 1
        self.tm_backbone = 'terramind_v1_large'  # {'terramind_v1_tiny','small','base','large'}
        self.tm_decoder = 'UNetDecoder'         # or 'UperNetDecoder'
        self.tm_decoder_channels = [512, 256, 128, 64]  # UNet-style widths
        self.tm_select_indices = None           # let code pick sensible defaults
        self.tm_head_dropout = 0.0
        self.tm_bands = None  # For PS/AE 4-band, defaults to ['BLUE','GREEN','RED','NIR_NARROW'] internally
        self.tm_backbone_ckpt_path = None

        # Option A: fully freeze backbone for entire training (simple)
        self.tm_freeze_backbone = False
        # Option B: freeze backbone only for first N epochs (tuner uses this)
        self.tm_freeze_backbone_epochs = 0

        # TerraMind optim defaults (used by tuner / can be respected by your optimizer factory)
        self.tm_lr_backbone = 1e-5       # base LR for backbone
        self.tm_lr_head_mult = 10.0      # head LR = tm_lr_backbone * tm_lr_head_mult
        self.tm_weight_decay = 1e-5

        # ---- Tuning budgets (HB -> BO), used if you call tune_TerraMind(...) ----
        self.tune_num_epochs = 20
        self.tune_num_epochs_bo = 20
        self.tune_steps_per_epoch = min(100, self.num_training_steps)
        self.tune_validation_steps = min(50, self.num_validation_images)
        self.tune_batch_size = min(8, self.train_batch_size)
        self.tune_hb_max_trials = 30
        self.tune_max_trials = 30
        # ----------------------------------------------------

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
        elif self.optimizer_fn.lower() == 'adamw':
            return optim.AdamW(model_parameters, lr=self.learning_rate)
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

        # TerraMind sanity checks (only if used)
        if self.modality in ('PS', 'AE'):
            # Expect four channels selected
            if len(self.preprocessing_bands) != 4:
                raise ConfigError("For PS/AE runs, exactly 4 channels must be enabled in channels_used.")
        if any(dim % 16 != 0 for dim in self.patch_size):
            # TerraMind tokenizers use 16x16 patches; training code pads, but warn here.
            warnings.warn("patch_size is not a multiple of 16; inputs will be padded internally.", RuntimeWarning)

        return self


class ConfigError(Exception):
    pass
