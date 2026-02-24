"""
Metrics computation for Gaussian Splatting training.

Encapsulates metrics calculations including PSNR, SSIM, LPIPS, and utilities
for preparing tensors for metric computations.
"""

import torch
from typing import Dict, Optional, Callable, List, Tuple


def zero_loss(device: torch.device) -> torch.Tensor:
    """
    Create a zero loss tensor on the specified device.
    
    Utility to replace repeated pattern: torch.tensor(0.0, device=device)
    
    Args:
        device: Torch device (cuda or cpu)
    
    Returns:
        Scalar tensor with value 0.0
    """
    return torch.tensor(0.0, device=device)


def prepare_for_metrics(tensor: torch.Tensor, format: str = 'HWC') -> torch.Tensor:
    """
    Prepare tensor for metric computation by permuting to BCHW format if needed.
    
    Metrics functions (PSNR, SSIM, LPIPS) expect [B, C, H, W] format.
    This utility converts from [H, W, C] to [1, C, H, W].
    
    Args:
        tensor: Input tensor, expected to be [H, W, C] if format='HWC'
        format: 'HWC' to convert from (H, W, C) to (B=1, C, H, W)
    
    Returns:
        Tensor in [B, C, H, W] format suitable for metric functions
    """
    if format == 'HWC':
        # [H, W, C] -> [C, H, W] -> [1, C, H, W]
        return tensor.permute(2, 0, 1).unsqueeze(0)
    else:
        raise ValueError(f"Unsupported format: {format}")


def compute_image_metrics(
    render: torch.Tensor,
    gt_image: torch.Tensor,
    ssim_fn: Callable,
    psnr_fn: Callable,
    lpips_fn: Optional[Callable] = None,
    compute_lpips: bool = False,
) -> Dict[str, float]:
    """
    Compute image quality metrics comparing rendered image to ground truth.
    
    Computes L1, SSIM, PSNR, and optionally LPIPS metrics.
    
    Args:
        render: [H, W, 3] rendered image
        gt_image: [H, W, 3] ground truth image
        ssim_fn: SSIM metric function (e.g., StructuralSimilarityIndexMeasure)
        psnr_fn: PSNR metric function (e.g., PeakSignalNoiseRatio)
        lpips_fn: LPIPS metric function (optional)
        compute_lpips: Whether to compute LPIPS (expensive, typically skipped)
    
    Returns:
        Dictionary with keys:
            - 'l1_loss': Mean absolute error
            - 'ssim_loss': 1.0 - SSIM (higher is worse)
            - 'psnr': Peak signal-to-noise ratio
            - 'lpips': Learned perceptual loss (if compute_lpips=True, else 0.0)
    """
    # Compute L1 loss (pixel-wise absolute error)
    l1_loss = (render - gt_image).abs().mean()
    
    # Prepare tensors for metric functions (they expect [B, C, H, W])
    render_perm = prepare_for_metrics(render, format='HWC')
    gt_perm = prepare_for_metrics(gt_image, format='HWC')
    
    # Compute SSIM loss (1 - SSIM, where 1.0 indicates perfect match)
    ssim_value = ssim_fn(render_perm, gt_perm)
    ssim_loss = 1.0 - ssim_value
    
    # Compute PSNR
    psnr_value = psnr_fn(render_perm, gt_perm)
    
    # Compute LPIPS if requested (expensive, typically skipped during training)
    if compute_lpips and lpips_fn is not None:
        lpips_value = lpips_fn(render_perm, gt_perm)
    else:
        lpips_value = zero_loss(render.device)
    
    return {
        'l1_loss': l1_loss.item(),
        'ssim_loss': ssim_loss.item(),
        'psnr': psnr_value.item(),
        'lpips': lpips_value.item() if isinstance(lpips_value, torch.Tensor) else lpips_value,
    }


def prepare_depth_for_ssim(depth: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Prepare depth map for SSIM computation.
    
    Reshapes [H, W] depth to [1, 1, H, W] format required by SSIM metric function.
    
    Args:
        depth: [H, W] depth map
    
    Returns:
        tuple: (depth_prepared, height, width)
            - depth_prepared: [1, 1, H, W] for SSIM computation
            - height: H
            - width: W
    """
    height, width = depth.shape
    depth_prepared = depth.unsqueeze(0).unsqueeze(0)  # [H, W] -> [1, 1, H, W]
    return depth_prepared, height, width
