# DEPRECATED: Custom Strategy Implementation

**⚠️ This folder contains the old custom strategy implementation and is no longer used.**

## Why Deprecated?

This custom strategy implementation has been replaced with gsplat's built-in `DefaultStrategy` class, which provides:

- Better tested and maintained densification logic
- More efficient implementation
- Standard API that works with the broader gsplat ecosystem
- Official support and documentation

## What to Use Instead?

Use `gsplat.DefaultStrategy` directly:

```python
from gsplat import DefaultStrategy

# Initialize strategy with your desired parameters
strategy = DefaultStrategy(
    prune_opa=0.005,
    grow_grad2d=0.0002,
    grow_scale3d=0.01,
    grow_scale2d=0.05,
    prune_scale3d=0.1,
    prune_scale2d=0.15,
    refine_start_iter=500,
    refine_stop_iter=15000,
    reset_every=3000,
    refine_every=100,
    absgrad=False,
    verbose=False,
)

# Check sanity
strategy.check_sanity(params, optimizers)

# Initialize state
strategy_state = strategy.initialize_state(scene_scale=1.0)

# In training loop after loss.backward()
strategy.step_post_backward(params, optimizers, strategy_state, step, info)
```

## Migration Notes

The model.py has been updated to use gsplat's DefaultStrategy automatically. No code changes are required for existing training scripts.

Key changes:
- `model.strategy` now uses `gsplat.DefaultStrategy` instead of custom implementation
- `densify_and_prune()` delegates to gsplat's strategy system
- All densification logic is handled by gsplat (split, duplicate, prune, opacity reset)

## References

- gsplat Documentation: https://docs.gsplat.studio/
- DefaultStrategy API: https://docs.gsplat.studio/apis/strategy.html

---

**Date Deprecated:** 2026-02-12  
**Replaced By:** `gsplat.DefaultStrategy`
