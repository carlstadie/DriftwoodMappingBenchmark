"""
Configuration and constants for Bayesian benchmark statistics analysis.
"""

from pathlib import Path
from typing import Dict, Optional

# =============================================================================
# PATHS
# =============================================================================

UNET_AE_METRICS = (
    r"N:\isipd\projects\p_planetdw\data\methods_test\results\UNET\AE"
    r"\evaluation_unet.csv"
)
SWIN_AE_METRICS = (
    r"N:\isipd\projects\p_planetdw\data\methods_test\results\SWIN\AE"
    r"\evaluation_swin.csv"
)
TERRAMIND_AE_METRICS = (
    r"N:\isipd\projects\p_planetdw\data\methods_test\results\Terramind\AE"
    r"\evaluation_tm.csv"
)

UNET_PS_METRICS = (
    r"N:\isipd\projects\p_planetdw\data\methods_test\results\UNET\PS"
    r"\evaluation_unet.csv"
)
SWIN_PS_METRICS = (
    r"N:\isipd\projects\p_planetdw\data\methods_test\results\SWIN\PS"
    r"\evaluation_swin.csv"
)
TERRAMIND_PS_METRICS = (
    r"N:\isipd\projects\p_planetdw\data\methods_test\results\Terramind\PS"
    r"\evaluation_tm.csv"
)

UNET_S2_METRICS = (
    r"N:\isipd\projects\p_planetdw\data\methods_test\results\UNET\S2"
    r"\evaluation_unet.csv"
)
SWIN_S2_METRICS = (
    r"N:\isipd\projects\p_planetdw\data\methods_test\results\SWIN\S2"
    r"\evaluation_swin.csv"
)
TERRAMIND_S2_METRICS = (
    r"N:\isipd\projects\p_planetdw\data\methods_test\results\Terramind\S2"
    r"\evaluation_tm.csv"
)

# CSV specifications: (path, group_label, dataset, architecture)
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

# =============================================================================
# METRICS CONFIGURATION
# =============================================================================

METRICS = [
    "IoU",
    "dice_coef",
    "normalized_surface_distance",
    "mean_epistemic_uncertainty",
    "mean_aleatoric_uncertainty",
]

# Metrics where higher values are better
MAXIMIZE_METRICS = {"IoU", "dice_coef"}

# Metrics bounded in (0,1) - use logit transformation
BOUNDED_01_METRICS = {
    "IoU",
    "dice_coef",
    "normalized_surface_distance",
    "mean_epistemic_uncertainty",
    "mean_aleatoric_uncertainty",
}

# =============================================================================
# MODEL PRIORS
# =============================================================================

# Prior standard deviations for factorial effects
PRIOR_SD_LOGIT = 0.5  # For logit-scale metrics
PRIOR_SD_LOG = 0.2    # For log-scale metrics

# =============================================================================
# SAMPLING CONFIGURATION
# =============================================================================

TEST = True  # Enable test mode with synthetic data

DRAWS = 2000
TUNE = 2000
CHAINS = 4
TARGET_ACCEPT = 0.9
HDI_PROB = 0.95
RANDOM_SEED = 1701

# Savage-Dickey KDE bandwidths for BF sensitivity analysis
SD_KDE_BANDWIDTHS = [0.1, 0.3, 0.5, "scott", "silverman"]

# =============================================================================
# ORDERING AND BASELINES
# =============================================================================

# Stable ordering for consistent results
DATASET_ORDER = ["AE", "PS", "S2"]
ARCH_ORDER = ["U-Net", "Swin", "Terramind"]

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

# Baseline groups for contrasts
BASELINE_ARCH = "U-Net"
BASELINE_DATASET = "AE"

# Baselines for effects coding interpretation
EFFECTS_BASELINE_DATASET = DATASET_ORDER[0]
EFFECTS_BASELINE_ARCH = ARCH_ORDER[0]

# =============================================================================
# ROPE (REGION OF PRACTICAL EQUIVALENCE)
# =============================================================================

# ROPE thresholds on transformed scale for consistency
ROPE_BY_METRIC_TRANSFORMED: Dict[str, float] = {
    "IoU": 0.05,
    "dice_coef": 0.05,
    "normalized_surface_distance": 0.05,
    "mean_epistemic_uncertainty": 0.05,
    "mean_aleatoric_uncertainty": 0.05,
}

# =============================================================================
# OVERALL RANKING WEIGHTS
# =============================================================================

# Optional metric weights for overall ranking (None = equal weights)
METRIC_WEIGHTS: Optional[Dict[str, float]] = None

# =============================================================================
# VISUAL STYLING (Enhanced color scheme)
# =============================================================================

# Color palettes for models (from reference)
MODEL_COLORS = {
    "U-Net": "#c27ba0",
    "Swin": "#e06666",
    "Terramind": "#ffd966",
}

# Dataset color palettes
DATASET_COLORS = {
    "AE": "#7986CB",    # Indigo
    "PS": "#81C784",    # Green  
    "S2": "#FFB74D",    # Orange
}

# Matplotlib styling parameters
PLOT_STYLE = {
    "figure.dpi": 300,
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 7,
    "axes.linewidth": 0.8,
    "axes.titleweight": "bold",
    "axes.titlepad": 8,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
    "grid.linewidth": 0.4,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}

# =============================================================================
# OUTPUT PATHS
# =============================================================================

def get_output_paths(base_dir: Path = None):
    """Get output directory paths."""
    if base_dir is None:
        base_dir = Path(__file__).resolve().parent
    
    assets_dir = base_dir / "stats_report_assets"
    report_path = base_dir / "stats_report.html"
    
    return {
        "base_dir": base_dir,
        "assets_dir": assets_dir,
        "report_path": report_path,
    }