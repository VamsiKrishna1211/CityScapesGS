# Usage Guide

## Command Line Interface

The training script now supports full CLI argument parsing using argparse.

### Basic Usage

```bash
python train.py \
    --colmap-path data/boston_colmap/sparse/0 \
    --images-path data/boston/images \
    --output-dir outputs/boston_run1
```

### With PLY Export

```bash
python train.py \
    --colmap-path data/boston_colmap/sparse/0 \
    --images-path data/boston/images \
    --output-dir outputs/boston_run1 \
    --export-ply
```

### Custom Training Parameters

```bash
python train.py \
    --colmap-path data/boston_colmap/sparse/0 \
    --images-path data/boston/images \
    --output-dir outputs/boston_long_run \
    --iterations 15000 \
    --densify-from-iter 500 \
    --densify-until-iter 15000 \
    --save-interval 2000 \
    --log-interval 100
```

### Custom Learning Rates

```bash
python train.py \
    --colmap-path data/boston_colmap/sparse/0 \
    --images-path data/boston/images \
    --output-dir outputs/boston_custom_lr \
    --lr-means 0.00020 \
    --lr-scales 0.007 \
    --lr-opacities 0.08
```

### Full Custom Configuration

```bash
python train.py \
    --colmap-path data/boston_colmap/sparse/0 \
    --images-path data/boston/images \
    --output-dir outputs/boston_full_custom \
    --iterations 20000 \
    --densify-from-iter 500 \
    --densify-until-iter 18000 \
    --densify-interval 100 \
    --grad-threshold 0.0002 \
    --max-screen-size 5000 \
    --opacity-reset-interval 3000 \
    --opacity-reset-value 0.01 \
    --lr-means 0.00016 \
    --lr-scales 0.005 \
    --lr-quats 0.001 \
    --lr-opacities 0.05 \
    --lr-sh 0.0025 \
    --save-interval 1000 \
    --log-interval 100 \
    --export-ply
```

## All Available Options

### Required Arguments

| Argument | Description |
|----------|-------------|
| `--colmap-path` | Path to COLMAP sparse reconstruction (e.g., `sparse/0`) |
| `--images-path` | Path to training images directory |

### Output Options

| Argument | Default | Description |
|----------|---------|-------------|
| `--output-dir` | `./output` | Output directory for checkpoints |
| `--export-ply` | `False` | Export final model to PLY format |

### Training Parameters

| Argument | Default | Description |
|----------|---------|-------------|
| `--iterations` | `7000` | Total training iterations |
| `--save-interval` | `1000` | Save checkpoint every N iterations |
| `--log-interval` | `1` | Log progress every N iterations |

### Densification Parameters

| Argument | Default | Description |
|----------|---------|-------------|
| `--densify-from-iter` | `500` | Start densification iteration |
| `--densify-until-iter` | `15000` | Stop densification iteration |
| `--densify-interval` | `100` | Densify every N iterations |
| `--grad-threshold` | `0.0002` | Gradient threshold for densification |
| `--max-screen-size` | `5000` | Max screen size (pixels) for pruning |
| `--opacity-reset-interval` | `3000` | Reset opacity every N iterations |
| `--opacity-reset-value` | `0.01` | Opacity reset value |

### Depth & Regularization

| Argument | Default | Description |
|----------|---------|-------------|
| `--enable-depth-loss` | `False` | Enable depth supervision |
| `--depth-loss-weight` | `0.1` | Weight for depth supervision |
| `--depth-loss-start-iter` | `1000` | Start iteration for depth loss |
| `--depth-objective` | `pearson` | Depth objective (`pearson` or `silog`) |
| `--sam-loss-weight` | `0.0` | Gradient-domain SAM loss weight |
| `--enable-scale-reg` | `False` | Enable scale regularization |
| `--scale-reg-weight` | `0.01` | Scale regularization weight |
| `--enable-opacity-reg` | `False` | Enable opacity regularization |
| `--opacity-reg-weight` | `0.0005` | Opacity regularization weight |
| `--enable-opacity-entropy-reg` | `False` | Enable opacity entropy regularization |
| `--opacity-entropy-reg-weight` | `0.0001` | Opacity entropy regularization weight |

### Learning Rates

| Argument | Default | Description |
|----------|---------|-------------|
| `--lr-means` | `0.00016` | LR for positions (×5.0 internally) |
| `--lr-scales` | `0.005` | LR for scales |
| `--lr-quats` | `0.001` | LR for rotations |
| `--lr-opacities` | `0.05` | LR for opacities |
| `--lr-sh` | `0.0025` | LR for spherical harmonics |

## Help

View all options:

```bash
python train.py --help
```

## Examples by Use Case

### Quick Test (Fast Training)

```bash
python train.py \
    --colmap-path data/scene/sparse/0 \
    --images-path data/scene/images \
    --output-dir outputs/quick_test \
    --iterations 3000 \
    --save-interval 500
```

### High Quality (Long Training)

```bash
python train.py \
    --colmap-path data/scene/sparse/0 \
    --images-path data/scene/images \
    --output-dir outputs/high_quality \
    --iterations 30000 \
    --densify-until-iter 25000 \
    --save-interval 5000 \
    --export-ply
```

### Debug (Frequent Logging)

```bash
python train.py \
    --colmap-path data/scene/sparse/0 \
    --images-path data/scene/images \
    --output-dir outputs/debug_run \
    --iterations 1000 \
    --log-interval 10 \
    --save-interval 100
```

### Conservative Densification

```bash
python train.py \
    --colmap-path data/scene/sparse/0 \
    --images-path data/scene/images \
    --output-dir outputs/conservative \
    --grad-threshold 0.0005 \
    --densify-interval 200 \
    --max-screen-size 2000
```

### Aggressive Densification

```bash
python train.py \
    --colmap-path data/scene/sparse/0 \
    --images-path data/scene/images \
    --output-dir outputs/aggressive \
    --grad-threshold 0.0001 \
    --densify-interval 50 \
    --max-screen-size 8000
```

## Python API

The primary supported interface is CLI-based. For scripted execution, call the CLI from Python:

```python
import subprocess

subprocess.run([
    "python", "custom_gaussiansplat/train.py",
    "--colmap-path", "data/boston_colmap/sparse/0",
    "--images-path", "data/boston/images",
    "--output-dir", "outputs/my_run",
    "--iterations", "10000",
    "--grad-threshold", "0.0003",
    "--lr-means", "0.0002",
], check=True)
```

## Tips

1. **Start with defaults**: The default parameters work well for most scenes
2. **Adjust iterations**: Use 7000 for quick results, 15000-30000 for quality
3. **Monitor Gaussian count**: Should grow from hundreds to tens of thousands
4. **Use --export-ply**: Essential for visualizing results in viewers
5. **Adjust grad-threshold**: Lower = more aggressive densification
6. **Check GPU memory**: Densification increases memory usage over time
