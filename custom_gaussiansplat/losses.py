"""
Loss functions for Gaussian Splatting training.
"""
import torch


def scale_regularization(scales, weight=0.01, scene_extent=1.0):
    """
    Penalize large Gaussian scales to prevent floaters.
    
    Floaters often manifest as large, semi-transparent blobs. This loss
    encourages Gaussians to stay small and compact, focusing on actual geometry.
    
    Args:
        scales: Tensor of shape [N, 3] containing Gaussian scales
        weight: Regularization weight (typical range: 0.01-0.1)
        scene_extent: Scene extent for normalization (optional)
    
    Returns:
        Scalar regularization loss
    """
    # Compute max scale for each Gaussian
    max_scales = torch.max(scales, dim=1).values
    
    # Penalize scales proportional to scene extent
    # Use L2 penalty to gently discourage large scales
    normalized_scales = max_scales / scene_extent
    loss = weight * (normalized_scales ** 2).mean()
    
    return loss

def opacity_regularization(opacities, weight=0.0001):
    """
    Penalize low-opacity Gaussians to prevent them from becoming floaters.
    
    Args:
        opacities: Tensor of shape [N, 1] containing opacity values
        weight: Regularization weight (typical range: 0.0001-0.001)
    
    Returns:
        Scalar regularization loss
    """
    # Penalize very low opacities (e.g., < 0.1)
    # Use L2 penalty to gently discourage very low opacities

    return weight * opacities.mean()


def depth_regularization(depths, near_plane=0.1, weight=0.01):
    """
    Penalize Gaussians that are too close to the camera (floaters in near plane).
    
    Args:
        depths: Tensor of shape [N] containing depth values (camera-to-Gaussian distance)
        near_plane: Minimum allowed depth (typical: 0.1m for indoor, 1.0m for outdoor)
        weight: Regularization weight
    
    Returns:
        Scalar depth regularization loss
    """
    # Penalize depths below near_plane using ReLU
    # This creates a soft barrier at the near plane
    violations = torch.relu(near_plane - depths)
    loss = weight * violations.sum()
    
    return loss


def opacity_scale_regularization(opacities, scales, opacity_threshold=0.1, scale_threshold=0.1, scene_extent=1.0):
    """
    Penalize Gaussians that are both large AND transparent (classic floater signature).
    
    Logic: Floaters are often large, semi-transparent blobs. Real geometry is usually
    small and opaque. This combines both criteria.
    
    Args:
        opacities: Tensor of shape [N, 1] containing opacity values
        scales: Tensor of shape [N, 3] containing Gaussian scales
        opacity_threshold: Opacity below which a Gaussian is considered transparent
        scale_threshold: Scale threshold as fraction of scene_extent
        scene_extent: Scene extent for normalization
    
    Returns:
        Scalar regularization loss
    """
    max_scales = torch.max(scales, dim=1).values
    normalized_scales = max_scales / scene_extent
    
    # Identify large, transparent Gaussians
    is_large = normalized_scales > scale_threshold
    is_transparent = opacities.squeeze(-1) < opacity_threshold
    floater_mask = is_large & is_transparent
    
    # Penalize these Gaussians
    # We penalize both their scale and inverse opacity
    if floater_mask.any():
        scale_penalty = normalized_scales[floater_mask].sum()
        opacity_penalty = (1.0 - opacities.squeeze(-1)[floater_mask]).sum()
        loss = scale_penalty + opacity_penalty
    else:
        loss = torch.tensor(0.0, device=opacities.device)
    
    return loss


def gradient_loss(render, gt):
    """
    Calculates the L1 difference between the gradients of the rendered and GT images.
    Inputs: [H, W, C] tensors.
    """
    # 1. Compute Gradients in X and Y
    # We simply subtract adjacent pixels.
    # Diff X: (Pixel[i+1] - Pixel[i])
    # Diff Y: (Pixel[j+1] - Pixel[j])
    
    # Slice to shift images
    # [:-1, :, :] = All rows except last
    # [1:, :, :]  = All rows except first
    
    # Gradient X
    r_grad_x = render[:, 1:, :] - render[:, :-1, :]
    g_grad_x = gt[:, 1:, :] - gt[:, :-1, :]
    
    # Gradient Y
    r_grad_y = render[1:, :, :] - render[:-1, :, :]
    g_grad_y = gt[1:, :, :] - gt[:-1, :, :]
    
    # 2. L1 Loss on the Gradients
    loss_x = torch.abs(r_grad_x - g_grad_x).mean()
    loss_y = torch.abs(r_grad_y - g_grad_y).mean()
    
    return loss_x + loss_y


def convert_relative_to_metric_depth(render_depth, prior_depth, mask=None):
    """
    Convert relative depth to metric depth using scale-shift alignment.
    
    Estimates scale and shift parameters to align prior (relative) depth with
    rendered (metric) depth, then converts the full prior depth map to metric space.
    
    Args:
        render_depth: [H, W] - Metric depth from Gaussian Splatting rendering
        prior_depth: [H, W] - Relative depth from Depth Anything V2 (normalized to [0, 1])
        mask: [H, W] - Boolean mask for valid pixels (e.g., high opacity regions)
    
    Returns:
        prior_depth_metric: [H, W] - Prior depth converted to metric space
        
    Raises:
        RuntimeError: If depth data is completely invalid and cannot be processed
    """
    from utils import estimate_depth_scale_shift
    
    device = render_depth.device
    
    # Apply mask if provided
    if mask is not None:
        num_masked = mask.sum().item()
        if num_masked < 10:
            raise RuntimeError(
                f"Insufficient valid pixels for depth loss: only {num_masked} pixels with mask. "
                f"Check alpha/opacity values and depth rendering."
            )
        render_flat = render_depth[mask]
        prior_flat = prior_depth[mask]
    else:
        render_flat = render_depth.reshape(-1)
        prior_flat = prior_depth.reshape(-1)
    
    # Remove invalid values (NaN, Inf, negative/zero depths)
    valid = (
        torch.isfinite(render_flat) & 
        torch.isfinite(prior_flat) & 
        (render_flat > 1e-6) &  # Avoid near-zero depths
        (prior_flat > 1e-6)     # Avoid near-zero priors
    )
    num_valid = valid.sum().item()
    
    if num_valid < 10:
        raise RuntimeError(
            f"Insufficient valid depth values: only {num_valid}/{len(render_flat)} pixels are valid. "
            f"Rendered depth range: [{render_flat.min().item():.4f}, {render_flat.max().item():.4f}], "
            f"Prior depth range: [{prior_flat.min().item():.4f}, {prior_flat.max().item():.4f}]. "
            f"Check for NaN/Inf in depth maps."
        )
    
    render_flat = render_flat[valid]
    prior_flat = prior_flat[valid]
    
    # Estimate scale and shift to convert prior depth to metric depth
    try:
        scale, shift = estimate_depth_scale_shift(render_flat, prior_flat)
    except ValueError as e:
        raise RuntimeError(
            f"Depth alignment failed. "
            f"Valid pixels: {num_valid}, "
            f"Render depth: mean={render_flat.mean():.4f}, std={render_flat.std():.4f}, "
            f"Prior depth: mean={prior_flat.mean():.4f}, std={prior_flat.std():.4f}. "
            f"Error: {str(e)}"
        )
    
    # Convert full prior depth to metric space (preserving original shape)
    prior_depth_metric = prior_depth * scale + shift
    
    return prior_depth_metric


def depth_loss(render_depth, prior_depth, mask=None):
    """
    Depth loss with scale-shift alignment for relative depth supervision.
    
    Converts the prior (relative) depth to metric space using the rendered depth
    as reference, then computes L1 loss.
    
    Args:
        render_depth: [H, W] - Metric depth from Gaussian Splatting rendering
        prior_depth: [H, W] - Relative depth from Depth Anything V2 (normalized to [0, 1])
        mask: [H, W] - Boolean mask for valid pixels (e.g., high opacity regions)
    
    Returns:
        Scalar depth loss after converting prior to metric space
        Prior depth converted to metric space [H, W]
        
    Raises:
        RuntimeError: If depth data is completely invalid and cannot be processed
    """
    # Convert relative depth to metric space
    prior_depth_metric = convert_relative_to_metric_depth(render_depth, prior_depth, mask)
    
    # Compute L1 loss between metric depths (on valid pixels only)
    if mask is not None:
        render_flat = render_depth[mask]
        prior_metric_flat = prior_depth_metric[mask]
    else:
        render_flat = render_depth.reshape(-1)
        prior_metric_flat = prior_depth_metric.reshape(-1)
    
    # Filter out invalid values for loss computation
    valid = (
        torch.isfinite(render_flat) & 
        torch.isfinite(prior_metric_flat) & 
        (render_flat > 1e-6) &
        (prior_metric_flat > 1e-6)
    )
    
    if valid.sum() < 10:
        raise RuntimeError("Insufficient valid pixels for depth loss computation")
    
    loss = torch.abs(render_flat[valid] - prior_metric_flat[valid]).mean()
    
    return loss, prior_depth_metric
