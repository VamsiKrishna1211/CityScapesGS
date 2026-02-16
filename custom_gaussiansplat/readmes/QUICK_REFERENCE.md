# Quick Reference: Floater Prevention Options

## Command-Line Flags

### Opacity Reset (Always Active, Now Configurable)
```bash
--opacity-reset-interval 3000        # Reset every N iterations
--opacity-reset-value 0.01          # Opacity value to reset to (0.01-0.1)
```

### Scale Regularization
```bash
--enable-scale-reg                  # Enable scale regularization loss
--scale-reg-weight 0.01            # Weight for regularization (0.01-0.1)
```

### Aggressive Pruning
```bash
--enable-aggressive-pruning         # Enable world-space pruning
--max-world-scale 0.1              # Max scale as fraction of scene extent
```

### Visibility Tracking
```bash
--enable-visibility-tracking        # Enable multi-view consistency
--min-view-count 3                 # Minimum views required
```

### Depth Culling (Experimental)
```bash
--enable-depth-culling              # Enable near-camera culling
--near-plane-threshold 0.1         # Minimum distance in meters
```

### Spherical Harmonics
```bash
--sh-degree 3                      # SH degree: 0 (DC only), 1, 2, 3 (default: 3)
--disable-sh-rendering             # Disable SH, use DC only (faster training)
```

---

## Quick Start Presets

### Conservative (Minimal Changes)
```bash
python custom_gaussiansplat/train.py \
    --colmap-path <path> --images-path <path> \
    --output-dir outputs/conservative \
    --iterations 7000
```
- Uses only opacity reset (default behavior)
- Safe for all scene types

### Balanced (Recommended)
```bash
python custom_gaussiansplat/train.py \
    --colmap-path <path> --images-path <path> \
    --output-dir outputs/balanced \
    --iterations 7000 \
    --enable-scale-reg \
    --scale-reg-weight 0.01
```
- Adds scale regularization
- +1-2% training time
- Good for most indoor/outdoor scenes

### Aggressive (Maximum Quality)
```bash
python custom_gaussiansplat/train.py \
    --colmap-path <path> --images-path <path> \
    --output-dir outputs/aggressive \
    --iterations 7000 \
    --opacity-reset-interval 2000 \
    --opacity-reset-value 0.01 \
    --enable-scale-reg \
    --scale-reg-weight 0.05 \
    --enable-aggressive-pruning \
    --max-world-scale 0.1 \
    --enable-visibility-tracking \
    --min-view-count 4
```
- Maximum floater prevention
- +3-7% training time
- May remove thin structures

---

## Tuning Guide

### Scene is too sparse (too many Gaussians pruned)
1. Increase `--opacity-reset-value` to 0.05
2. Increase `--max-world-scale` to 0.2
3. Decrease `--min-view-count` to 2
4. Disable techniques one by one

### Still seeing floaters
1. Decrease `--opacity-reset-interval` to 2000
2. Increase `--scale-reg-weight` to 0.05
3. Enable `--enable-aggressive-pruning`
4. Enable `--enable-visibility-tracking`

### Thin structures disappearing
1. Increase `--opacity-reset-value` to 0.05
2. Disable `--enable-scale-reg`
3. Increase `--max-world-scale` to 0.2
4. Disable `--enable-aggressive-pruning`

---

## Monitoring Training

### Verbosity Levels
```bash
--verbosity 0  # QUIET - Minimal output
--verbosity 1  # NORMAL - Progress bar + summaries (default)
--verbosity 2  # VERBOSE - Detailed operations
--verbosity 3  # DEBUG - All details
```

### What to Watch For

**Verbosity 1 (Normal):**
- Configuration table shows enabled techniques
- Gaussian count changes during densification

**Verbosity 2 (Verbose):**
- Opacity reset messages
- Densification operations (clone/split/prune counts)
- Net Gaussian change per densification

**Verbosity 3 (Debug):**
- Per-frame depth culling counts (if enabled)
- Visibility tracking updates

---

## Scene-Specific Recommendations

### Indoor (extent < 20m)
```bash
--enable-scale-reg --scale-reg-weight 0.01
--enable-visibility-tracking --min-view-count 3
```

### Outdoor/Urban (extent 20-200m)
```bash
--enable-scale-reg --scale-reg-weight 0.005
--enable-aggressive-pruning --max-world-scale 0.15
```

### Aerial/Large (extent > 200m)
```bash
--enable-scale-reg --scale-reg-weight 0.002
--enable-aggressive-pruning --max-world-scale 0.2
--max-screen-size 100  # Also increase screen size limit
```

### Object-Centric (small object, extent < 5m)
```bash
--enable-scale-reg --scale-reg-weight 0.02
--enable-visibility-tracking --min-view-count 5
--opacity-reset-interval 2000
```

### Fast Training (Lower Quality, Faster Speed)
```bash
--sh-degree 0  # DC only, ~2x faster
# or
--disable-sh-rendering  # Train all SH but render with DC only
```

### High Quality View-Dependent Effects
```bash
--sh-degree 3  # Full 3rd order SH (default)
--enable-scale-reg --scale-reg-weight 0.01
```

---

## Integration with Existing Options

These new options work alongside existing training parameters:

```bash
python custom_gaussiansplat/train.py \
    # Standard options
    --colmap-path data/scene/sparse/0 \
    --images-path data/scene/images \
    --output-dir outputs/scene \
    --iterations 10000 \
    --densify-from-iter 500 \
    --densify-until-iter 15000 \
    --grad-threshold 0.0002 \
    --max-screen-size 50 \
    \
    # Floater prevention (NEW)
    --enable-scale-reg \
    --scale-reg-weight 0.01 \
    --enable-aggressive-pruning \
    --max-world-scale 0.1 \
    \
    # Learning rates
    --lr-means 0.00016 \
    --lr-scales 0.005 \
    \
    # Output
    --save-interval 1000 \
    --export-ply \
    --verbosity 2
```

---

## Troubleshooting

### Error: "All Gaussians were pruned"
- Too aggressive pruning settings
- Increase `--opacity-reset-value`
- Increase `--max-world-scale`
- Decrease `--min-view-count`

### Error: "No module named 'torch'"
- Activate your conda environment first:
```bash
conda activate <your_env_name>
```

### Training is too slow
- Disable `--enable-visibility-tracking` (saves 2-5%)
- Increase `--opacity-reset-interval` to 4000
- Use fewer techniques

### Results are blurry/low quality
- Techniques may be too aggressive
- Increase `--opacity-reset-value` to 0.05
- Disable all techniques and enable one at a time
- Check that your scene has good camera coverage

---

## Files Documentation

- **IMPLEMENTATION_SUMMARY.md** - Complete technical details
- **FLOATER_PREVENTION.md** - In-depth explanation of techniques
- **QUICK_REFERENCE.md** (this file) - Quick command reference
