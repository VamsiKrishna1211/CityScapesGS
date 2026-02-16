# Strategy System for Gaussian Densification

This module implements a modular strategy pattern for Gaussian densification and pruning, inspired by the [gsplat library](https://github.com/nerfstudio-project/gsplat).

## Architecture

The strategy system separates densification logic from the model, making it easier to:
- Experiment with different densification strategies
- Tune parameters without modifying model code
- Implement advanced features like AbsGS, MCMC, etc.

### Components

```
strategy/
├── __init__.py          # Exports Strategy, ViewportInfo, DefaultStrategy
├── base.py              # Abstract Strategy class
├── ops.py               # Helper operations (duplicate, split, remove, etc.)
├── default.py           # DefaultStrategy (original 3DGS paper)
└── README.md            # This file
```

## Quick Start

### Basic Usage (Backward Compatible)

The `GaussianModel` now uses the strategy system internally, but the API remains backward compatible:

```python
from custom_gaussiansplat.model import GaussianModel
from custom_gaussiansplat.strategy import ViewportInfo

# Create model (uses DefaultStrategy by default)
model = GaussianModel(init_points, init_colors)

# In training loop, after loss.backward()
model.densify_and_prune(
    grad_threshold=0.0002,
    scene_extent=scene_extent,
    max_screen_size=None,
    optimizers=optimizers,
    iteration=iteration,
    viewport_info=ViewportInfo(width=800, height=600, n_cameras=1)  # NEW: Required for proper gradient normalization
)
```

### Advanced Usage: Custom Strategy Configuration

```python
from custom_gaussiansplat.strategy import DefaultStrategy

# Create strategy with custom parameters
strategy = DefaultStrategy(
    prune_opa=0.005,           # Opacity pruning threshold
    grow_grad2d=0.0002,        # Screen-space gradient threshold
    grow_scale3d=0.01,         # 3D scale threshold for clone/split
    grow_scale2d=0.05,         # 2D scale threshold for split
    prune_scale3d=0.1,         # 3D scale pruning threshold
    prune_scale2d=0.15,        # 2D scale pruning threshold (high = disabled)
    refine_start_iter=500,     # Start densification
    refine_stop_iter=15_000,   # Stop densification
    reset_every=3000,          # Opacity reset interval
    refine_every=100,          # Densification frequency
    absgrad=False,             # Use absolute gradients (AbsGS mode)
    revised_opacity=False,     # Use revised opacity formula for splits
    verbose=True,              # Print detailed logs
)

# Create model with custom strategy
model = GaussianModel(init_points, init_colors, strategy=strategy)
```

### Direct Strategy Usage (Recommended for New Code)

For full control, call the strategy directly from your training loop:

```python
from custom_gaussiansplat.strategy import DefaultStrategy, ViewportInfo

# Initialize strategy
strategy = DefaultStrategy(verbose=True)

# In training loop, after loss.backward()
params = model.get_params_dict()
optimizers_dict = model.get_optimizers_dict(optimizers)

strategy.step_post_backward(
    params=params,
    optimizers=optimizers_dict,
    scene_scale=scene_extent,
    iteration=iteration,
    viewport_info=ViewportInfo(width=image.shape[1], height=image.shape[0]),
    radii=model.max_radii2D,  # Optional: for 2D scale-based refinement
)

# Update model with potentially changed parameters
model.update_params_from_dict(params)
```

## Key Features

### 1. Screen-Normalized Gradients

Gradients are now normalized by viewport size, making thresholds scale-invariant:

```python
grads[..., 0] *= viewport_info.width / 2.0
grads[..., 1] *= viewport_info.height / 2.0
```

**Migration Note**: You may need to adjust `grow_grad2d` threshold if migrating from the old system.

### 2. Covariance-Based Split Sampling

Splits now use proper Gaussian covariance sampling:

```python
# Old: samples = torch.randn() * scale
# New: samples = R @ diag(scale) @ randn()
samples = torch.einsum("nij,nj,bnj->bni", rotmat, scales, randn)
```

This creates better-oriented splits that respect the Gaussian's rotation.

### 3. Periodic Opacity Reset

Every `reset_every` iterations, opacities are clamped to prevent saturation:

```python
opacity = min(opacity, 2 * prune_opa)  # Default: max 0.01
```

Optimizer state is also zeroed to allow fresh learning.

### 4. Multi-Criteria Pruning

Gaussians are pruned based on:
- Low opacity: `< 0.005`
- Large 3D scale: `> 0.1 * scene_scale` (after first reset)
- Large 2D screen-space size: `> 0.15` (disabled by default)

### 5. AbsGS Mode (Optional)

Set `absgrad=True` to use absolute (unnormalized) gradients instead of averages:

```python
# Standard: grads = accumulated_grads / count
# AbsGS: grads = accumulated_grads
```

This can help with scenes that have varying visibility patterns.

## Migration Guide

### Old API → New API

| Old Parameter | New Strategy Parameter | Notes |
|---------------|------------------------|-------|
| `grad_threshold` | `grow_grad2d` | Now screen-normalized |
| `percent_dense * scene_extent` | `grow_scale3d * scene_scale` | `grow_scale3d=0.01` (1% of scene) |
| N/A | `grow_scale2d` | New: 2D scale split criterion |
| Hardcoded 0.005 | `prune_opa` | Configurable |
| `max_world_scale` | `prune_scale3d` | Now part of strategy |
| N/A | `reset_every` | New: periodic opacity reset |

### Breaking Changes

1. **Gradient normalization**: Old gradients were not normalized by viewport size. You may need to adjust `grow_grad2d`:
   - Old `grad_threshold=0.0002` ≈ New `grow_grad2d=0.0002` for 800x600 images
   - For different resolutions, scale accordingly

2. **Split sampling**: New covariance-based sampling creates different Gaussian distributions. Results may vary slightly.

3. **Opacity reset**: The strategy periodically resets opacity, which the old system didn't do. This can affect convergence.

### Backward Compatibility

The `densify_and_prune()` method maintains backward compatibility but logs warnings:

```python
# Still works, but prints warning:
model.densify_and_prune(
    grad_threshold=0.0002,  # [LEGACY]
    scene_extent=scene_extent,
    max_screen_size=None,   # [LEGACY]
    optimizers=optimizers,
    iteration=iteration,
    # viewport_info missing → uses default (800x600) ⚠️
)
```

## Implementation Details

### Parameter Synchronization

All operations (duplicate, split, remove) keep model parameters and optimizer states synchronized:

```python
# When duplicating Gaussians:
# 1. Duplicate parameters: torch.cat([param, param[mask]])
# 2. Duplicate optimizer state (exp_avg, exp_avg_sq): pad with zeros
# 3. Update param_groups to point to new tensors
```

### Memory Management

- Operations call `torch.cuda.empty_cache()` after major changes
- Optimizer state is cleaned when Gaussians are removed
- Buffer sizes automatically match parameter sizes

### Frustum Culling Support

The strategy handles frustum culling automatically:

```python
# If gaussian_ids provided (culling active):
self.state["grad2d"][gaussian_ids] += grad_norms  # Map visible → all

# Otherwise (no culling):
self.state["grad2d"] += grad_norms  # Direct update
```

## Performance Tips

1. **Tune `refine_every`**: More frequent refinement (lower value) = slower but better quality
2. **Adjust `grow_grad2d`**: Higher threshold = fewer Gaussians, faster training
3. **Use `absgrad=True`** for scenes with varying visibility patterns
4. **Set `refine_stop_iter`** to stop densification early if overfitting

## Future Extensions

The modular design makes it easy to add new strategies:

- **MCMC Strategy**: Stochastic densification with dead Gaussian pruning
- **Hierarchical Strategy**: Coarse-to-fine refinement
- **Adaptive Strategy**: Learn optimal thresholds during training

See `strategy/base.py` for the interface to implement.

## References

- Original 3DGS paper: https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/
- gsplat library: https://github.com/nerfstudio-project/gsplat
- AbsGS paper: https://arxiv.org/abs/2404.10484
