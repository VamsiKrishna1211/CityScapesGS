# CityScapeGS - 3D Urban Reconstruction Pipeline

## Project Overview
CityScapeGS is an end-to-end pipeline designed for large-scale 3D urban reconstruction. It integrates data acquisition from Mapillary, sophisticated preprocessing (depth estimation, object removal, motion filtering), and a high-performance 3D Gaussian Splatting (GS) implementation based on the `gsplat` library.

The project is structured to handle the complexities of street-level imagery, including dynamic objects, varying lighting conditions, and the need for accurate camera poses and depth maps for high-quality reconstructions.

## Key Components

### 1. `custom_gaussiansplat/`
The core 3D reconstruction engine using Gaussian Splatting.
- `train.py`: Main entry point for training GS models. Supports densification, pruning, and various auxiliary losses.
- `dataset.py`: Comprehensive dataset loader supporting COLMAP, MatrixCity, and Instant-NGP formats.
- `model.py`: Definition of `GaussianModel`, managing learnable parameters (means, scales, rotations, opacities, SH coefficients, and optional semantics).
- `losses.py`: Implementation of L1, SSIM, LPIPS, depth-based losses, and scale/opacity regularization.
- `train_semantics.py`: Specialized script for fine-tuning semantic features on a pre-trained geometry model.
- `installer.sh`: Environment setup script for the GS component.

### 2. `dataset_setup/`
Tools for data ingestion and format conversion.
- `mapillary_to_cusfm_converter.py`: Converts Mapillary API data to formats compatible with CuSFM and other reconstruction tools.
- `mapillary_downloader.py`: (Deprecated but referenced) For fetching street-level imagery and metadata.

### 3. `tools/`
Preprocessing utilities to clean and enhance data before reconstruction.
- `batch_depth_prediction.py`: Batch generation of depth maps using Depth Anything V2/V3.
- `batch_object_removal_with_flow.py`: Uses SAM2 and RAFT (optical flow) to remove dynamic objects (cars, people) from sequences.
- `filter_sequence_by_motion.py`: Filters out redundant frames by analyzing camera motion between consecutive images.
- `ply_viewer.py`: Simple utility for visualizing `.ply` point clouds/GS models.

### 4. `third_party/`
External libraries integrated as submodules or included directly, such as `gsplat`, `sam2`, `Depth-Anything-3`, `colmap`, and various other GS variants (`LangSplat`, `Skyfall-GS`, etc.).

## Development Conventions

- **Environment**: Primarily Linux-based. Uses Conda for dependency management. Core reconstruction requires CUDA and `gsplat`.
- **Coding Style**:
  - **Type Hints**: Extensive use of Python type hints for clarity and IDE support.
  - **Dataclasses**: Structured data (like camera parameters and viewport info) is managed using `dataclasses`.
  - **Logging**: Uses the `logging` module combined with `rich` for enhanced console output, including tables and progress bars.
  - **Modularity**: Separation of dataset logic (`dataset.py`), model architecture (`model.py`), and training orchestration (`train.py`).
- **Configuration**: Primarily through command-line arguments using `argparse` (or specialized `train_args.py`).
- **Visualization**: Integration with multiple viewers including `nerfview`, `viser`, and `rerun`.

## Building and Running

### Installation
To set up the environment for the custom Gaussian Splatting implementation:
```bash
bash custom_gaussiansplat/installer.sh
```
This script creates a conda environment (default name: `citysplat`) and installs necessary dependencies like `torch`, `gsplat`, and `pycolmap`.

### Key Commands

**1. Training a Gaussian Splatting Model:**
```bash
python custom_gaussiansplat/train.py \
    --colmap-path /path/to/scene/sparse/0 \
    --images-path /path/to/scene/images \
    --output-dir /path/to/output
```

**2. Semantic Fine-Tuning:**
```bash
python custom_gaussiansplat/train_semantics.py \
    --checkpoint-path /path/to/model_final.pt \
    --colmap-path /path/to/scene/sparse/0 \
    --images-path /path/to/scene/images \
    --output-dir /path/to/semantic_output \
    --semantic-provider npy \
    --semantics-path /path/to/semantic_targets
```

**3. Batch Depth Prediction:**
```bash
python tools/batch_depth_prediction.py \
    --input-folder data/images \
    --output-npy-folder data/depth_npy \
    --output-png-folder data/depth_png \
    --model depth-anything/Depth-Anything-V2-Base-hf
```

**4. Object Removal:**
```bash
python tools/batch_object_removal_with_flow.py \
    --base-path data/images \
    --filtered-output data/images_clean \
    --save-masks
```

## Directory Structure Overview
- `custom_gaussiansplat/`: Main reconstruction logic.
- `dataset_setup/`: Conversion scripts for various datasets.
- `tools/`: Utility scripts for preprocessing and analysis.
- `third_party/`: External repositories and submodules.
- `data/`: (Optional/User-defined) Root for datasets and outputs.
- `scripts/`: Environment setup and build scripts.
