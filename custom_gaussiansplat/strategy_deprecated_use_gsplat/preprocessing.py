"""
Preprocessing utilities for depth maps and other data.

Encapsulates depth alignment, normalization, and validation functions.
These were previously scattered across utils.py and train.py.
"""

import torch
from typing import Tuple, Optional


def estimate_depth_scale_shift(render_depth: torch.Tensor, prior_depth: torch.Tensor) -> Tuple[float, float]:
    """
    Estimate scale and shift to align prior (relative) depth to render (metric) depth.
    
    Uses multiple fallback methods to ensure robust alignment:
    1. Least squares (most accurate)
    2. Median/MAD based (robust to outliers)
    3. Mean/std standardization (last resort)
    
    Args:
        render_depth: [N] tensor of metric depth values from rendering
        prior_depth: [N] tensor of relative depth values from depth prior
    
    Returns:
        tuple: (scale, shift) such that metric_prior = prior_depth * scale + shift
        
    Raises:
        ValueError: If alignment fails with all methods
    """
    # Try Method 1: Least squares alignment
    try:
        return _estimate_lstsq(render_depth, prior_depth)
    except Exception as e1:
        # Try Method 2: Median-based alignment
        try:
            return _estimate_median(render_depth, prior_depth)
        except Exception as e2:
            # Try Method 3: Standardization
            try:
                return _estimate_standardize(render_depth, prior_depth)
            except Exception as e3:
                raise ValueError(
                    f"All depth alignment methods failed. "
                    f"Errors: lstsq={str(e1)}, median={str(e2)}, std={str(e3)}"
                )


def _estimate_lstsq(render_depth: torch.Tensor, prior_depth: torch.Tensor) -> Tuple[float, float]:
    """
    Least squares: solve prior * s + t = render for optimal (s, t).
    
    Most accurate when data is well-behaved with few outliers.
    """
    ones = torch.ones_like(prior_depth)
    X = torch.stack([prior_depth, ones], dim=1)  # [N, 2]
    Y = render_depth.unsqueeze(1)                # [N, 1]
    
    # Validate inputs
    if torch.isnan(X).any() or torch.isinf(X).any():
        raise ValueError("Prior depth contains NaN/Inf")
    if torch.isnan(Y).any() or torch.isinf(Y).any():
        raise ValueError("Render depth contains NaN/Inf")
    
    # Solve least squares
    solution = torch.linalg.lstsq(X, Y, rcond=1e-6).solution
    scale, shift = solution[0].item(), solution[1].item()
    
    # Validate solution
    if not torch.isfinite(solution).all():
        raise ValueError("lstsq solution contains NaN/Inf")
    if not (0.001 < abs(scale) < 1000.0):
        raise ValueError(f"Scale factor {scale:.4f} out of reasonable range")
    
    return scale, shift


def _estimate_median(render_depth: torch.Tensor, prior_depth: torch.Tensor) -> Tuple[float, float]:
    """
    Median/MAD based: robust to outliers.
    
    Uses Median Absolute Deviation for scale estimation, which is robust
    to outliers in the depth data.
    """
    render_median = torch.median(render_depth)
    prior_median = torch.median(prior_depth)
    
    # Compute MAD (Median Absolute Deviation)
    render_mad = torch.median(torch.abs(render_depth - render_median))
    prior_mad = torch.median(torch.abs(prior_depth - prior_median))
    
    # Use MAD for scale, or fall back to range if MAD is too small
    if render_mad < 1e-6 or prior_mad < 1e-6:
        render_range = render_depth.max() - render_depth.min()
        prior_range = prior_depth.max() - prior_depth.min()
        
        if render_range < 1e-6 or prior_range < 1e-6:
            raise ValueError("Depth values are nearly constant")
        
        scale = render_range / prior_range
        shift = render_median - scale * prior_median
    else:
        scale = render_mad / prior_mad
        shift = render_median - scale * prior_median
    
    return scale.item(), shift.item()


def _estimate_standardize(render_depth: torch.Tensor, prior_depth: torch.Tensor) -> Tuple[float, float]:
    """
    Z-score standardization: align to same mean and std.
    
    Last resort when both lstsq and median methods fail. Assumes data is
    somewhat normally distributed.
    """
    render_mean = render_depth.mean()
    render_std = render_depth.std()
    prior_mean = prior_depth.mean()
    prior_std = prior_depth.std()
    
    if render_std < 1e-6 or prior_std < 1e-6:
        raise ValueError("Standard deviation too small - depth is constant")
    
    # Transform: (prior - prior_mean) / prior_std * render_std + render_mean
    # Which is: prior * (render_std / prior_std) + (render_mean - prior_mean * render_std / prior_std)
    scale = render_std / prior_std
    shift = render_mean - prior_mean * scale
    
    return scale.item(), shift.item()


def validate_depth_tensors(
    render_depth: torch.Tensor,
    prior_depth: torch.Tensor,
    mask: Optional[torch.Tensor] = None
) -> Tuple[bool, str]:
    """
    Validate depth tensors for loss computation.
    
    Args:
        render_depth: [H, W] metric depth from rendering
        prior_depth: [H, W] relative depth from depth prior
        mask: [H, W] boolean mask for valid pixels (optional)
    
    Returns:
        tuple: (is_valid, message)
            - is_valid: True if tensors are valid for loss computation
            - message: Description of validation status/issues
    """
    # Check for NaN/Inf
    if not torch.isfinite(render_depth).all():
        return False, "Rendered depth contains NaN/Inf values"
    
    if not torch.isfinite(prior_depth).all():
        return False, "Prior depth contains NaN/Inf values"
    
    # Check tensor size
    if render_depth.numel() == 0 or prior_depth.numel() == 0:
        return False, "Empty depth tensors"
    
    # Check masked region if provided
    if mask is not None:
        num_valid = mask.sum().item()
        if num_valid < 10:
            return False, f"Insufficient valid pixels: only {num_valid} (need >= 10)"
    
    return True, "Valid"


def normalize_depth_for_loss(
    render_depth: torch.Tensor,
    prior_depth_metric: torch.Tensor,
    mask: torch.Tensor,
    min_valid_pixels: int = 100
) -> Tuple[torch.Tensor, torch.Tensor, bool]:
    """
    Normalize depth maps to [0, 1] range for SSIM computation.
    
    Uses valid (masked) regions to determine min/max for normalization,
    preventing invalid pixels from skewing the normalization range.
    
    Args:
        render_depth: [H, W] rendered depth map
        prior_depth_metric: [H, W] prior depth in metric space
        mask: [H, W] boolean mask for valid pixels
        min_valid_pixels: Minimum valid pixels required (default: 100)
    
    Returns:
        tuple: (render_norm, metric_norm, success)
            - render_norm: [H, W] normalized rendered depth (or zeros if failed)
            - metric_norm: [H, W] normalized metric depth (or zeros if failed)
            - success: True if normalization succeeded, False if insufficient valid pixels
    """
    num_valid = mask.sum().item()
    
    # Check if sufficient valid pixels
    if num_valid < min_valid_pixels:
        return (
            torch.zeros_like(render_depth),
            torch.zeros_like(prior_depth_metric),
            False
        )
    
    # Extract valid regions
    valid_render = render_depth[mask]
    valid_metric = prior_depth_metric[mask]
    
    # Compute min/max from valid regions
    render_min, render_max = valid_render.min(), valid_render.max()
    metric_min, metric_max = valid_metric.min(), valid_metric.max()
    
    render_range = render_max - render_min
    metric_range = metric_max - metric_min
    
    # Check if ranges are sufficient for normalization
    if render_range < 1e-6 or metric_range < 1e-6:
        return (
            torch.zeros_like(render_depth),
            torch.zeros_like(prior_depth_metric),
            False
        )
    
    # Normalize to [0, 1] using valid regions' min/max
    render_norm = (render_depth - render_min) / render_range
    metric_norm = (prior_depth_metric - metric_min) / metric_range
    
    # Clamp to [0, 1] to handle edge cases
    render_norm = torch.clamp(render_norm, 0, 1)
    metric_norm = torch.clamp(metric_norm, 0, 1)
    
    return render_norm, metric_norm, True
