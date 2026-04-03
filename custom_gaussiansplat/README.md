# Custom Gaussian Splatting

This folder contains a unified, type-safe training framework for multiple Gaussian Splatting variants using an Abstract Base Class (ABC) + capability mixin pattern.

## Architecture Highlights

- **Unified Model Interface**: `BaseTrainableModel` ABC in `models/base.py` defines a shared contract for all model variants
  - **Standard 3DGS**: `GaussianModel` with `gsplat.DefaultStrategy` densification
  - **Scaffold-GS**: `ScaffoldModel` with neural Gaussians and sparse anchor optimization
  - **Type-Safe Capability Detection**: `isinstance(model, NeuralRenderingMixin)` replaces brittle `hasattr()` checks
  
- **Typed Return Values**: `RenderParams` and `NeuralGaussianOutput` dataclasses eliminate untyped dicts and tuple unpacking

- **Flattened Optimizer/Scheduler Access**: `GSOptimizers.all_optimizers()` and `GS_LR_Schedulers.all_schedulers()` iterators simplify training loops

- **Clean Imports**: All models and utilities accessible via `from models import ...` and `from gs_types import ...`

## Features

- COLMAP dataset loading (`dataset.py`)
- Densification and pruning via `gsplat.DefaultStrategy`
- Optional depth and regularization losses
- TensorBoard logging, checkpoint resume, and optional live viewers
- Standalone semantic fine-tuning (`train_semantics.py`)

## Table of Contents

- [Model Architecture](#model-architecture)
- [Requirements](#requirements)
- [Install](#install)
- [Data Layout](#data-layout)
- [Quick Start](#quick-start)
- [Training](#training)
- [Resume From Checkpoint](#resume-from-checkpoint)
- [Semantic Fine-Tuning](#semantic-fine-tuning)
- [Outputs](#outputs)
- [Useful Flags](#useful-flags)
- [Troubleshooting](#troubleshooting)
- [Additional Docs](#additional-docs)

## Model Architecture

### Folder Structure

```
custom_gaussiansplat/
├── models/                           # Model implementations (ABC + variants)
│   ├── __init__.py                   # Clean re-export facade
│   ├── base.py                       # BaseTrainableModel ABC + NeuralRenderingMixin
│   ├── gaussian.py                   # GaussianModel (standard 3DGS)
│   └── scaffold.py                   # ScaffoldModel (Scaffold-GS with neural Gaussians)
├── gs_types.py                       # Typed dataclasses + optimizer/scheduler containers
├── train.py                          # Training orchestrator using unified model interface
├── train_semantics.py                # Standalone semantic fine-tuning
├── dataset.py                        # COLMAP dataset loader
├── losses.py                         # Loss functions (depth, regularization, photometric)
└── ...                               # Other utilities
```

### Model Selection

Both models conform to `BaseTrainableModel` but have different capabilities:

- **`GaussianModel`** (default): Standard 3D Gaussian Splatting
  - Deterministic point positions (no per-view generation)
  - `gsplat.DefaultStrategy` handles densification (split/clone) and pruning
  - Smaller memory footprint

- **`ScaffoldModel`**: Scaffold-GS with neural Gaussians (when `--model-type scaffold`)
  - Sparse anchor set + MLPs generate view-dependent Gaussians
  - Implements `NeuralRenderingMixin` (detectable via `isinstance()`)
  - Higher quality for complex scenes, more memory/compute

### Import Patterns

```python
# Safe imports (updated in all training/viewer files)
from models import BaseTrainableModel, GaussianModel, ScaffoldModel, NeuralRenderingMixin
from gs_types import RenderParams, NeuralGaussianOutput, GSOptimizers, GS_LR_Schedulers

# Type-safe capability detection (replaces hasattr checks)
if isinstance(model, NeuralRenderingMixin):
    output = model.generate_neural_gaussians(cam)  # pyright knows this exists
```

### Typed Returns

- **`RenderParams`** (from `model.get_render_params()`): Typed container with `means`, `colors`, `opacities`, `scales`, `quats`, `sh_degree`, and optional `neural_opacity`/`selection_mask`.
- **`NeuralGaussianOutput`** (from `ScaffoldModel.generate_neural_gaussians()`): Typed container with generated Gaussian properties and optional training-only fields.

## Requirements

- Linux
- NVIDIA GPU with CUDA (training uses `torch.device("cuda")` in `custom_gaussiansplat/train.py`)
- COLMAP sparse reconstruction
- Python environment with PyTorch + `gsplat`

## Install

From repository root:

```bash
bash custom_gaussiansplat/installer.sh
```

This script creates a conda env (default `citysplat`) and installs the core dependencies.

If you install manually, make sure these packages are available:

- `torch`, `torchvision`
- `gsplat`
- `pycolmap`
- `imageio`, `opencv-python`, `numpy`, `scipy`
- `torchmetrics`, `tensorboard`, `rich`
- `plyfile`
- `fused-ssim`
- `simple-knn`
- Optional viewers: `nerfview`, `viser`, `rerun-sdk`

## Data Layout

`train.py` expects COLMAP sparse data, image folders, and depth folders using scale-aware naming.

Example scene layout:

```text
SCENE_ROOT/
├── sparse/
│   └── 0/
├── images/               # for scale=1
├── images_2/             # for scale=2
├── images_4/             # for scale=4
├── depths_npy/           # for scale=1
├── depths_npy_2/         # for scale=2
└── depths_npy_4/         # for scale=4
```

Important behavior:

- If you use `--scale 2`, images are resolved from `images_2` and depth from `depths_npy_2`.
- Depth directories are required by the dataset loader, even if depth loss is disabled.

## Quick Start

From repository root:

Base Gaussian training:

```bash
python custom_gaussiansplat/train.py \
    --colmap-path /path/to/scene/sparse/0 \
    --images-path /path/to/scene/images \
    --output-dir /path/to/scene/cust_gs_outputs
```

Standalone semantic fine-tuning:

```bash
python custom_gaussiansplat/train_semantics.py \
    --checkpoint-path /path/to/scene/cust_gs_outputs/model_final.pt \
    --colmap-path /path/to/scene/sparse/0 \
    --images-path /path/to/scene/images \
    --output-dir /path/to/scene/cust_gs_outputs_semantics \
    --semantic-provider npy \
    --semantics-path /path/to/semantic_targets
```

## Training

### Baseline training

```bash
python custom_gaussiansplat/train.py \
    --colmap-path /path/to/scene/sparse/0 \
    --images-path /path/to/scene/images \
    --output-dir /path/to/scene/cust_gs_outputs \
    --iterations 30000 \
    --log-interval 100 \
    --save-interval 1000 \
    --scale 1
```

### With depth losses and regularization

```bash
python custom_gaussiansplat/train.py \
    --colmap-path /path/to/scene/sparse/0 \
    --images-path /path/to/scene/images \
    --output-dir /path/to/scene/cust_gs_outputs_depth \
    --iterations 30000 \
    --enable-depth-loss \
    --depth-loss-weight 0.1 \
    --depth-loss-start-iter 1000 \
    --enable-scale-reg \
    --scale-reg-weight 0.01 \
    --enable-opacity-reg \
    --opacity-reg-weight 0.0005
```

### Enable live viewer(s)

```bash
python custom_gaussiansplat/train.py \
    --colmap-path /path/to/scene/sparse/0 \
    --images-path /path/to/scene/images \
    --output-dir /path/to/scene/cust_gs_outputs_viewer \
    --viewer \
    --viewer-port 8080
```

Optional Rerun viewer:

```bash
python custom_gaussiansplat/train.py \
    --colmap-path /path/to/scene/sparse/0 \
    --images-path /path/to/scene/images \
    --output-dir /path/to/scene/cust_gs_outputs_rerun \
    --rerun-viewer
```

## Resume From Checkpoint

```bash
python custom_gaussiansplat/train.py \
    --colmap-path /path/to/scene/sparse/0 \
    --images-path /path/to/scene/images \
    --output-dir /path/to/scene/cust_gs_outputs \
    --resume-from /path/to/scene/cust_gs_outputs/checkpoints/checkpoint_10000.pt
```

## Semantic Fine-Tuning

Semantic training is provided as a standalone stage in `custom_gaussiansplat/train_semantics.py`.

### 1) Train base geometry model

```bash
python custom_gaussiansplat/train.py \
    --colmap-path /path/to/scene/sparse/0 \
    --images-path /path/to/scene/images \
    --output-dir /path/to/scene/cust_gs_outputs
```

### 2) Fine-tune semantic features from `.npy` targets

Expected semantic file naming:

- `/path/to/semantic_targets/<image_stem>_s.npy`

Run:

```bash
python custom_gaussiansplat/train_semantics.py \
    --checkpoint-path /path/to/scene/cust_gs_outputs/model_final.pt \
    --colmap-path /path/to/scene/sparse/0 \
    --images-path /path/to/scene/images \
    --output-dir /path/to/scene/cust_gs_outputs_semantics \
    --semantic-provider npy \
    --semantics-path /path/to/semantic_targets \
    --semantics-dim 3 \
    --semantic-finetune-iters 2000
```

For runtime semantic inference, switch to `--semantic-provider runtime` and pass `--semantic-model-path`.

Runtime semantic example:

```bash
python custom_gaussiansplat/train_semantics.py \
    --checkpoint-path /path/to/scene/cust_gs_outputs/model_final.pt \
    --colmap-path /path/to/scene/sparse/0 \
    --images-path /path/to/scene/images \
    --output-dir /path/to/scene/cust_gs_outputs_semantics_runtime \
    --semantic-provider runtime \
    --semantic-model-path /path/to/model.pt \
    --semantics-dim 3
```

## Outputs

Typical output tree:

```text
OUTPUT_DIR/
├── checkpoints/
│   ├── checkpoint_1000.pt
│   ├── checkpoint_2000.pt
│   └── ...
├── tensorboard/
├── training.log
├── model_final.pt
└── final_gaussians.ply   # only if --export-ply
```

For standalone semantic fine-tuning:

```text
SEMANTIC_OUTPUT_DIR/
├── semantic_training.log
├── semantic_model_final.pt
└── tensorboard/
```

## Useful Flags

Core:

- `--iterations`, `--save-interval`, `--log-interval`
- `--scale` for `images_*` / `depths_npy_*` selection
- `--resume-from`
- `--export-ply`
- `--verbosity {0,1,2,3}`

Densification/pruning:

- `--densify-from-iter`, `--densify-until-iter`, `--densify-interval`
- `--grad-threshold`, `--max-screen-size`
- `--prune-opa`, `--grow-grad2d`, `--grow-scale3d`, `--grow-scale2d`, `--prune-scale3d`, `--prune-scale2d`
- `--opacity-reset-interval`, `--opacity-reset-value`

Regularization and depth:

- `--enable-scale-reg`, `--scale-reg-weight`
- `--enable-opacity-reg`, `--opacity-reg-weight`
- `--enable-opacity-entropy-reg`, `--opacity-entropy-reg-weight`
- `--enable-depth-loss`, `--depth-loss-weight`, `--depth-loss-start-iter`
- `--affine-invariant-depth-loss-weight`
- `--pearson-correlation-loss-weight`
- `--silog-loss-weight`
- `--ordinal-depth-loss-weight`
- `--affine-aligned-gradient-matching-loss-weight`
- `--enable-depth-smoothness-loss`, `--depth-smoothness-loss-weight`

Quality/logging:

- `--enable-lpips-loss`, `--lpips-loss-weight`, `--lpips-model`
- `--tensorboard`, `--no-tensorboard`
- `--tb-image-interval`, `--tb-histogram-interval`

Viewers:

- `--viewer`, `--viewer-port`, `--viewer-refresh-interval`
- `--rerun-viewer`

`train_semantics.py` specific:

- `--checkpoint-path`, `--colmap-path`, `--images-path`, `--output-dir`
- `--semantic-provider {npy,runtime}`
- `--semantics-path` (required for `npy` provider)
- `--semantic-model-path` (required for `runtime` provider)
- `--semantics-dim`, `--semantic-loss-weight`, `--semantic-finetune-iters`
- `--semantic-cache-enabled`, `--no-semantic-cache`
- `--semantic-image-resolution H W`
- `--lr-sh`, `--lr-semantics`, `--num-workers`, `--preload`, `--device`

See full options:

```bash
python custom_gaussiansplat/train.py --help
python custom_gaussiansplat/train_semantics.py --help
```

## Troubleshooting

- `FileNotFoundError` for depth directory:
    - Create the expected `depths_npy` (or `depths_npy_<scale>`) folder.
    - Match `--scale` with available image/depth folders.

- Viewer flag has no effect:
    - Install optional dependencies (`nerfview`, `viser`, and/or `rerun-sdk`).

- `Semantic features file not found`:
    - Ensure semantic files use `<image_stem>_s.npy` naming and that `--semantics-path` points to that folder.

- CUDA out-of-memory:
    - Reduce resolution via larger `--scale`.
    - Use fewer iterations before densification or increase pruning aggressiveness.
    - Disable optional losses (`--enable-lpips-loss` off, depth auxiliaries at `0.0`).

## Additional Docs

- `custom_gaussiansplat/readmes/USAGE.md`
- `custom_gaussiansplat/readmes/FLOATER_PREVENTION.md`
- `custom_gaussiansplat/readmes/SPHERICAL_HARMONICS.md`
- `custom_gaussiansplat/readmes/QUICK_REFERENCE.md`
