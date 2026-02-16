# Custom Gaussian Splatting Implementation

## Overview
This is a clean, educational implementation of 3D Gaussian Splatting using the gsplat library (v1.0.0+).

## Files
- `model.py`: GaussianModel with densification/pruning logic
- `dataset.py`: COLMAP dataset loader
- `train.py`: Training pipeline
- `losses.py`: Loss functions (if needed)

## Key Features Implemented

### ✅ Fixed Issues
1. **Gradient Accumulation**: Proper tracking of screen-space gradients over multiple iterations
2. **Densification & Pruning**: Complete implementation with:
   - Cloning for small, high-gradient Gaussians
   - Splitting for large, high-gradient Gaussians
   - Pruning for invisible or too-large Gaussians
3. **Optimizer State Management**: Proper updating when parameters change size
4. **Scene Extent Calculation**: Computed from point cloud bounds
5. **Camera Model Support**: Handles SIMPLE_PINHOLE, PINHOLE, RADIAL, OPENCV models
6. **Model Checkpointing**: Save/load functionality with PLY export
7. **Enhanced Logging**: Track Gaussian count, losses, and training progress

### Training Pipeline
```python
from custom_gaussiansplat.train import train_pipeline

model = train_pipeline(
    colmap_path='path/to/sparse/0',
    images_path='path/to/images',
    output_dir='./output'
)
```

### Model Architecture
- **Parameters**: means, scales, quaternions, opacities, SH coefficients (DC + rest)
- **Buffers**: xyz_gradient_accum, denom, max_radii2D (for densification tracking)
- **Optimization**: Per-parameter learning rates (positions need higher LR)

### Densification Strategy
- **From iteration**: 500
- **Until iteration**: 15000 (extended for proper convergence)
- **Interval**: Every 100 iterations
- **Gradient threshold**: 0.0002 (screen-space)
- **Opacity reset**: Every 3000 iterations

## Usage

### Training
```python
from custom_gaussiansplat.train import train_pipeline

# Train from COLMAP reconstruction
model = train_pipeline(
    colmap_path='data/boston_colmap/sparse/0',
    images_path='data/boston/images',
    output_dir='./outputs/boston_run1'
)
```

### Loading Checkpoint
```python
from custom_gaussiansplat.model import GaussianModel

model = GaussianModel.load_checkpoint(
    'outputs/boston_run1/model_final.pt',
    device='cuda'
)

# Export to PLY for visualization
model.save_ply('outputs/boston_run1/gaussians.ply')
```

## Dependencies
```bash
pip install torch torchvision
pip install gsplat  # Latest version
pip install pycolmap
pip install imageio
pip install torchmetrics
pip install plyfile  # For PLY export (optional)
```

## Implementation Details

### Gradient Retention
The training loop retains gradients on `means2d` (screen-space positions) to track where the model needs more detail:
```python
means2d = meta['means2d'][0]
means2d.retain_grad()
```

### Densification Logic
1. **Accumulate gradients** over multiple frames in buffers
2. **Identify high-gradient regions** (areas with high rendering error)
3. **Clone** small Gaussians in high-gradient areas (add detail)
4. **Split** large Gaussians in high-gradient areas (subdivide)
5. **Prune** invisible or oversized Gaussians (reduce memory)

### Optimizer State Updates
When adding/removing Gaussians, optimizer state (Adam's momentum buffers) must be updated:
- **Append**: Pad with zeros for new Gaussians
- **Remove**: Filter to keep only non-pruned Gaussians

## Output Structure
```
output_dir/
├── checkpoints/
│   ├── checkpoint_1000.pt
│   ├── checkpoint_2000.pt
│   └── ...
└── model_final.pt
```

## Performance Tips
1. **GPU Memory**: Starts with ~few hundred Gaussians, grows to tens of thousands
2. **Batch Size**: Currently trains on one image at a time (memory-efficient)
3. **Iterations**: 7000 is minimal; 15000-30000 recommended for quality
4. **Learning Rates**: Position LR is 5x higher (0.00016 * 5.0)

## Known Limitations
1. No multi-GPU support (single GPU only)
2. No background color handling (assumes black background)
3. No adaptive density control (fixed thresholds)
4. No SH degree progression (starts at degree 3)

## Future Enhancements
- [ ] Add rendering function for inference
- [ ] Implement validation loop
- [ ] Add PSNR/SSIM metrics tracking
- [ ] Support for different background colors
- [ ] Adaptive density control parameters
- [ ] Multi-view consistency regularization
