# DriftwoodMappingBenchmark

A complete pipeline to preprocess imagery, train deep learning models, and evaluate segmentation performance for driftwood mapping across multiple modalities (Aerial, PlanetScope, Sentinel-2) and architectures (U-Net, Swin U-Net, TerraMind).

**Based on:** Stadie et al. (2025) — *Deep Learning Approaches for Driftwood Mapping along North American Coastlines*

**Workflow:**
1. **Preprocessing** → prepare GeoTIFF frame dataset from raw imagery
2. **Tuning** → hyperparameter optimization (Optuna)
3. **Training** → multi-run training (10 runs per architecture-modality combination)
4. **Evaluation** → generate predictions and performance metrics
5. **Analysis** → Bayesian hierarchical models for statistical comparison

---

## Quick Start

### Setup

This project uses **[Pixi](https://pixi.sh/)** for reproducible dependency management.

```bash
# Install dependencies and activate environment
pixi shell
```

### Training

Run a specific architecture × modality combination:

```bash
# U-Net with Aerial imagery
pixi run U-NetxAE

# Swin U-Net with PlanetScope
pixi run Swin-U-NetxPS

# TerraMind with Sentinel-2
pixi run TerramindxS2
```

Available tasks (see `pixi.toml`):
- **U-Net:** `U-NetxAE`, `U-NetxPS`, `U-NetxS2`
- **Swin U-Net:** `Swin-U-NetxAE`, `Swin-U-NetxPS`, `Swin-U-NetxS2`
- **TerraMind:** `TerramindxAE`, `TerramindxPS`, `TerramindxS2`
- **Analysis:** `pixi run analyse` (Bayesian hierarchical modeling + figures)

### Configuration

Each main script imports a config:

- `config/configUnetxAE.py` → U-Net + Aerial
- `config/configSwinUnetxPS.py` → Swin U-Net + PlanetScope
- `config/configTerraMindxS2.py` → TerraMind + Sentinel-2

Key config parameters:
- `training_image_dir` → raw raster data
- `training_gpkg` → training area polygons
- `label_gpkg` → segmentation labels
- `preprocessed_dir` → output for GeoTIFF frames

---

## Modalities

| Modality | Source | Resolution | Bands | Notes |
|----------|--------|-----------|-------|-------|
| **AE** | MACS Aerial | 10–15 cm | 4 (RGB+NIR) | Highest resolution, limited coverage |
| **PS** | PlanetScope | 3 m | 4 (RGB+NIR) | Regular global coverage |
| **S2** | Sentinel-2 L2A | 10 m | 11 + SCL | Public, open data |

---

## About the Research

This benchmark evaluates deep learning approaches for automated driftwood mapping in the Mackenzie Delta and Arctic coastlines (Canada). The study addresses two critical gaps in environmental remote sensing:

1. **Lack of systematic model-modality comparisons:** Existing work rarely compares architectures across multiple image types
2. **Insufficient statistical rigor:** Most benchmarks report single-run results without uncertainty quantification

The paper systematically evaluates:
- **Architectures:** U-Net, Swin U-Net, and TerraMind foundation model
- **Data:** Multi-sensor dataset (Aerial, PlanetScope, Sentinel-2) with ground truth from Mackenzie Delta
- **Metrics:** Overlap (IoU, Dice), boundary precision (NSD), and epistemic/aleatoric uncertainty
- **Uncertainty:** Bayesian hierarchical modeling with multiple training runs per configuration

**Key Finding:** Sensor modality dominates performance variation (~4× larger effect than architecture choice), while boundary delineation shows inverse metric rankings depending on resolution.

---

## Main Scripts

Each `main<Architecture>x<Modality>.py` follows this pattern:

```python
# 1. PREPROCESSING (uncomment to regenerate frames)
preprocessing.preprocess_all(config)

# 2. TUNING (uncomment to run hyperparameter search)
best_params = tuning.tune_UNet(config)  # or tune_SwinUNet, tune_TerraMind

# 3. TRAINING (run 10 times for robust evaluation)
for i in range(10):
    training.train_UNet(config)  # trains & saves checkpoints

# 4. EVALUATION (predict on test set, compute metrics)
evaluation.evaluate_unet(config)  # saves masks + CSV rows
```

---

## Analysis Pipeline

After training all 9 combinations (3 architectures × 3 modalities), run:

```bash
pixi run analyse
```

This executes `analysis/main.py`, which performs:

1. **Data aggregation** → Loads CSV metrics from all training runs
2. **Bayesian hierarchical modelling** → Fits multi-level effects model (PyMC) with:
   - Random intercepts per dataset & architecture
   - Uncertainty quantification (HDI 94%)
   - ROPE (region of practical equivalence) for equivalence testing
3. **Contrasts & rankings** → Architecture × dataset comparisons
4. **Figure generation** → Publication-ready plots (`analysis/figs/`)
5. **HTML report** → `analysis/stats_report.html` with interactive tables

Key analysis modules:
- `analysis/config.py` → Analysis settings (chains, draws, ROPE thresholds)
- `analysis/models_group.py` → Hierarchical model specifications
- `analysis/contrasts.py` → Posterior contrasts & comparisons
- `analysis/figs/` → Figure generation scripts (`figure2.py` … `figure8.py`)
- `analysis/reporting.py` → HTML report generation

---

## Project Structure

```
.
├── TRAINING & EVALUATION
├─ mainUnetxAE.py           # U-Net + Aerial (entry point)
├─ mainUnetxPS.py           # U-Net + PlanetScope
├─ mainUnetxS2.py           # U-Net + Sentinel-2
├─ mainSwinUnetxAE.py       # Swin U-Net + Aerial
├─ mainSwinUnetxPS.py       # Swin U-Net + PlanetScope
├─ mainSwinUnetxS2.py       # Swin U-Net + Sentinel-2
├─ mainTerraMindxAE.py      # TerraMind + Aerial
├─ mainTerraMindxPS.py      # TerraMind + PlanetScope
├─ mainTerraMindxS2.py      # TerraMind + Sentinel-2
│
├── CORE MODULES
├─ preprocessing.py         # Frame dataset preparation
├─ tuning.py                # Hyperparameter optimization (Optuna)
├─ training.py              # PyTorch training loop (AMP, EMA, mixed precision)
├─ evaluation.py            # Batch prediction & metric computation
│
├── MODELS
├─ core/
│  ├─ UNet.py               # Standard U-Net
│  ├─ Swin_UNetPP.py        # Swin U-Net++
│  ├─ TerraMind.py          # TerraMind foundation model wrapper
│  ├─ losses.py             # Dice, Tversky, weighted BCE
│  ├─ optimizers.py         # get_optimizer() factory
│  ├─ dataset_generator.py  # Iterable patch dataset
│  ├─ split_frames.py       # Train/val/test split, stratification
│  ├─ frame_info.py         # FrameInfo metadata + normalization
│  ├─ visualize.py          # Prediction visualization
│  │
│  └─ common/
│     ├─ console.py         # Colored logging, banners
│     ├─ model_utils.py     # AMP contexts, EMA, autopad
│     ├─ data.py            # Frame discovery & loading
│     └─ vis.py             # TensorBoard image logging
│
├── CONFIGURATION
├─ config/
│  ├─ configUnetxAE.py      # U-Net + Aerial config
│  ├─ configUnetxPS.py      # U-Net + PlanetScope config
│  ├─ configUnetxS2.py      # U-Net + Sentinel-2 config
│  ├─ configSwinUnetxAE.py  # Swin U-Net configs
│  ├─ configSwinUnetxPS.py  # (one per architecture-modality)
│  ├─ configSwinUnetxS2.py
│  ├─ configTerraMindxAE.py # TerraMind configs
│  ├─ configTerraMindxPS.py
│  └─ configTerraMindxS2.py
│
├── STATISTICAL ANALYSIS & FIGURES
├─ analysis/
│  ├─ main.py               # Main analysis entry point
│  ├─ config.py             # Analysis hyperparameters & ROPE
│  ├─ data_io.py            # Load/cache analysis data
│  ├─ models_group.py       # Hierarchical Bayesian models
│  ├─ models_factorial.py   # Factorial design models
│  ├─ contrasts.py          # Posterior contrasts
│  ├─ overall_ranking.py    # Architecture rankings
│  ├─ plots.py              # Generic plotting utilities
│  ├─ reporting.py          # HTML report generation
│  ├─ transforms.py         # Data transformations
│  ├─ bayes_utils.py        # PyMC utilities
│  ├─ colguides.py          # Color schemes for figures
│  │
│  ├─ figs/                 # Figure generation scripts
│  │  ├─ figure2.py         # Attention patterns
│  │  ├─ figure3.py         # Data splits
│  │  ├─ figure4.py         # Hyperparameter sensitivity
│  │  ├─ figure5.py         # Training curves
│  │  ├─ figure6.py         # Validation performance
│  │  ├─ figure7.py         # Area comparison (predicted vs observed)
│  │  ├─ figure8.py         # Model comparison
│  │  └─ html_parser.py
│  │
│  └─ result_safe/          # Cached model results
│     ├─ *xAE.csv
│     ├─ *xPS.csv
│     ├─ *xS2.csv
│     ├─ dataset_level/
│     └─ frame_level/
│
├── UTILITIES
├─ tools/
│  ├─ temp.py               # Informal tools
│  ├─ download_s2.py        # Sentinel-2 download helpers
│  ├─ download_ps.py        # PlanetScope download helpers
│  ├─ plot_losses.py        # Training curves
│  ├─ vis_results.py        # Result inspection
│  └─ *.ipynb               # Exploratory notebooks
│
├── ENVIRONMENT & METADATA
├─ pixi.toml                # Pixi manifest (dependencies + tasks)
├─ example_env.yml          # Example conda environment
├─ README.md                # This file
└─ .gitignore
```

---

## Training Outputs

Each training run generates:

| Artifact | Location | Description |
|----------|----------|-------------|
| **TensorBoard Logs** | `logs/<ARCH>/<MODALITY>/<run>/` | Training/validation scalars, images, embeddings |
| **Checkpoints** | `models/<ARCH>/<MODALITY>/` | Best model + periodic snapshots (.pt files) |
| **Predictions** | `results/<ARCH>/<MODALITY>/<timestamp>/` | GeoTIFF masks per test frame |
| **Metrics CSV** | `results/<ARCH>/<MODALITY>/evaluation_<arch>.csv` | Per-frame metrics (IoU, Dice, F1, etc.) |

### Monitor Training

```bash
tensorboard --logdir logs/ --port 6006
```

Then open http://localhost:6006 in your browser.

---

## Data Format & Assumptions

### Preprocessing Input

- **Training areas:** GeoPackage with AOI polygons
  Path: `config.training_gpkg`
- **Ground truth labels:** GeoPackage with driftwood polygons
  Path: `config.label_gpkg[<modality>]`
- **Imagery:** Raw GeoTIFFs (Sentinel-2, PlanetScope, Aerial)
  Path: `config.training_image_dir`

### Preprocessed Frame Format

Each frame is a multi-band GeoTIFF with structure:

```
Band 1..N:  Input imagery (normalized per modality)
         :  - S2: 11 bands
         :  - PS: 4 bands (RGB+NIR)
         :  - AE: 4 bands (RGB+NIR)
Band N+1:   Binary label mask (1 = driftwood, 0 = background)
Band N+2:   Boundary mask (1 = near edge, 0 = interior)
```

Stored in: `config.preprocessed_dir`

### Configuration

Each config file contains:

```python
class Configuration:
    # Data paths
    training_image_dir: Path
    training_gpkg: Path
    label_gpkg: Dict[str, Path]  # {AE, PS, S2}
    preprocessed_dir: Path

    # Training parameters
    patch_size: int
    batch_size: int
    epochs: int
    learning_rate: float

    # Model parameters
    depth: int
    initial_filters: int
    dropout: float
```

---

## Metrics

The evaluation pipeline computes:

- **Pixel-level:** IoU, Dice, F1-score, Precision, Recall
- **Instance-level:** (if applicable) AP, AR, mIoU
- **Area-based:** Predicted vs. labeled area (in m²)

All metrics are saved per frame and aggregated to CSV.

---

## Dependencies

### Key Requirements

- **PyTorch** ≥ 2.5.1 (GPU support via CUDA 12.1)
- **GeoPython stack:** rasterio, geopandas, fiona, shapely
- **ML/Segmentation:** torchvision, timm, segmentation-models-pytorch
- **Foundation models:** TerraTorch, transformers
- **Bayesian analysis:** PyMC 5+
- **Optimization:** Optuna

See `pixi.toml` for full dependency specifications.

---

## Citation

This benchmark implements the methods and multi-sensor evaluation framework from:

```bibtex
@article{Stadie2025,
  title={Deep Learning Approaches for Driftwood Mapping along North American Coastlines},
  author={Stadie, Carl and Nitze, Ingmar and Demir, Begüm and Grosse, Guido},
  journal={(in submission)},
  year={2025},
  note={Alfred Wegener Institute; Technical University Berlin; University of Potsdam}
}
```

If you use this benchmark or the DriftwoodMappingBenchmark code, please cite the paper above.

**Contact:** carl.stadie@awi.de

---

## Troubleshooting

### GPU Memory Issues

- Reduce `batch_size` in config
- Enable gradient checkpointing in model config
- Use mixed precision (enabled by default)

### Data Loading Errors

- Verify CRS alignment across input datasets
- Check GeoPackage validity: `ogrinfo <file.gpkg>`
- Ensure preprocessed frames exist before running evaluation

### Analysis Fails

- Ensure all 9 architecture-modality combinations have been trained
- Verify CSV files exist in `results/<ARCH>/<MODALITY>/`
- Check PyMC samplers have converged (Rhat < 1.01)
