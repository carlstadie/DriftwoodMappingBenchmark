# config.py
from __future__ import annotations

from typing import Dict, Optional
import numpy as np

# =============================================================================
# CONFIGURATION CONSTANTS
# =============================================================================

EVAL_LEVEL = "dataset"   # "frames" or "dataset"

# paths
UNET_AE_METRICS = (
    "/isipd/projects/p_planetdw/data/methods_test/results/UNET/AE/metrics_20260206-082747"
    f"/UNETxAE_{EVAL_LEVEL}_metrics.csv"
)
SWIN_AE_METRICS = (
    "/isipd/projects/p_planetdw/data/methods_test/results/SWIN/AE/metrics_20260206-082747"
    f"/SWINxAE_{EVAL_LEVEL}_metrics.csv"
)
TERRAMIND_AE_METRICS = (
    "/isipd/projects/p_planetdw/data/methods_test/results/TERRAMIND/AE/metrics_20260206-082747"
    f"/TERRAMINDxAE_{EVAL_LEVEL}_metrics.csv"
)

UNET_PS_METRICS = (
    "/isipd/projects/p_planetdw/data/methods_test/results/UNET/PS/metrics_20260206-082747"
    f"/UNETxPS_{EVAL_LEVEL}_metrics.csv"
)
SWIN_PS_METRICS = (
    "/isipd/projects/p_planetdw/data/methods_test/results/SWIN/PS/metrics_20260206-082747"
    f"/SWINxPS_{EVAL_LEVEL}_metrics.csv"
)
TERRAMIND_PS_METRICS = (
    "/isipd/projects/p_planetdw/data/methods_test/results/TERRAMIND/PS/metrics_20260206-082747"
    f"/TERRAMINDxPS_{EVAL_LEVEL}_metrics.csv"
)

UNET_S2_METRICS = (
    "/isipd/projects/p_planetdw/data/methods_test/results/UNET/S2/metrics_20260206-082747"
    f"/UNETxS2_{EVAL_LEVEL}_metrics.csv"
)
SWIN_S2_METRICS = (
    "/isipd/projects/p_planetdw/data/methods_test/results/SWIN/S2/metrics_20260206-082747"
    f"/SWINxS2_{EVAL_LEVEL}_metrics.csv"
)
TERRAMIND_S2_METRICS = (
    "/isipd/projects/p_planetdw/data/methods_test/results/TERRAMIND/S2/metrics_20260206-082747"
    f"/TERRAMINDxS2_{EVAL_LEVEL}_metrics.csv"
)

# metrics to analyse
METRICS = [
    "IoU",
    "dice_coef",
    "normalized_surface_distance",
    "mean_epistemic_uncertainty",
    "mean_aleatoric_uncertainty",
]

# which metrics are "higher is better"
# NOTE: "normalized_surface_distance" is usually NSD (0..1), higher is better
MAXIMIZE_METRICS = {"IoU", "dice_coef", "normalized_surface_distance"}

# metrics that are truly bounded in (0,1) and should be modeled on logit scale
# NOTE: uncertainties are positive and typically <<1, so log-scale is more appropriate
BOUNDED_01_METRICS = {"IoU", "dice_coef", "normalized_surface_distance"}

# priors for factorial effects (on transformed scale)
PRIOR_SD_LOGIT = 0.5
PRIOR_SD_LOG = 0.5

TEST = False

# sampling config
DRAWS = 2000
TUNE = 2000
CHAINS = 4
TARGET_ACCEPT = 0.9
HDI_PROB = 0.95
RANDOM_SEED = 1701

# Savage–Dickey KDE bandwidths for sensitivity analysis
SD_KDE_BANDWIDTHS = [0.1, 0.3, 0.5, "scott", "silverman"]

# overall ranking weights across metrics
METRIC_WEIGHTS: Optional[Dict[str, float]] = {
    "IoU": 0.4,
    "dice_coef": 0.15,
    "normalized_surface_distance": 0.35,
    "mean_epistemic_uncertainty": 0.05,
    "mean_aleatoric_uncertainty": 0.05,
}

# Overall multi-metric scoring mode
# Options: "rank" (robust, rank-based utility), "robust_z" (MAD-based z-score), "z" (standard z-score)
OVERALL_SCORING_MODE = "z"

# Stable ordering
DATASET_ORDER = ["AE", "PS", "S2"]
ARCH_ORDER = ["U-Net", "Swin", "Terramind"]

# Contrast baselines
BASELINE_ARCH = "U-Net"
BASELINE_DATASET = "AE"  # for dataset-within-architecture contrasts

# Baselines for interpreting factorial coefficients under EFFECTS CODING
EFFECTS_BASELINE_DATASET = DATASET_ORDER[0]  # e.g., "AE"
EFFECTS_BASELINE_ARCH = ARCH_ORDER[0]        # e.g., "U-Net"

# Practical effect thresholds (ROPE) on TRANSFORMED scale for consistency
ROPE_BY_METRIC_TRANSFORMED: Dict[str, float] = {
    "IoU": 0.05,  # on logit scale
    "dice_coef": 0.05,
    "normalized_surface_distance": 0.05,
    "mean_epistemic_uncertainty": np.log(1.05),  # 5% change on log scale
    "mean_aleatoric_uncertainty": np.log(1.05),  # 5% change on log scale
}

CSV_SPECS = [
    (UNET_AE_METRICS, "U-Net | AE", "AE", "U-Net"),
    (SWIN_AE_METRICS, "Swin U-Net | AE", "AE", "Swin"),
    (TERRAMIND_AE_METRICS, "Terramind | AE", "AE", "Terramind"),
    (UNET_PS_METRICS, "U-Net | PS", "PS", "U-Net"),
    (SWIN_PS_METRICS, "Swin U-Net | PS", "PS", "Swin"),
    (TERRAMIND_PS_METRICS, "Terramind | PS", "PS", "Terramind"),
    (UNET_S2_METRICS, "U-Net | S2", "S2", "U-Net"),
    (SWIN_S2_METRICS, "Swin U-Net | S2", "S2", "Swin"),
    (TERRAMIND_S2_METRICS, "Terramind | S2", "S2", "Terramind"),
]

GROUP_ORDER = [
    "U-Net | AE",
    "Swin U-Net | AE",
    "Terramind | AE",
    "U-Net | PS",
    "Swin U-Net | PS",
    "Terramind | PS",
    "U-Net | S2",
    "Swin U-Net | S2",
    "Terramind | S2",
]

# =============================================================================
# PLOTS
# =============================================================================
ARCH_COLORS = {
    "U-Net": "#B58AE6",
    "Swin": "#E9797A",
    "Terramind": "#F2D06B",
}

METRIC_LABELS = {
    "IoU": "IoU",
    "dice_coef": "Dice coefficient",
    "normalized_surface_distance": "Normalized Surface Distance",
    "mean_epistemic_uncertainty": "Mean epistemic uncertainty",
    "mean_aleatoric_uncertainty": "Mean aleatoric uncertainty",
}
