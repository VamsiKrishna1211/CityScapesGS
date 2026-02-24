"""
Rendering operations for Gaussian Splatting.

This module encapsulates all rendering-related operations:
- Camera matrix setup
- Gaussian rasterization
- Rendering output postprocessing
- Depth map extraction and normalization
"""

import torch
from typing import Tuple, Dict, Optional


def setup_camera_matrices(cam_dict: Dict, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Setup camera intrinsics and extrinsics matrices from camera dictionary.
    
    Args:
        cam_dict: Dictionary containing camera parameters with keys:
            - 'R': [3, 3] rotation matrix
            - 'T': [3] translation vector
            - 'fx', 'fy': focal lengths
            - 'cx', 'cy': principal point coordinates
        device: Torch device (cuda or cpu)
    
    Returns:
        tuple: (viewmat, K)
            - viewmat: [1, 4, 4] view matrix (camera extrinsics)
            - K: [3, 3] intrinsics matrix
    """
    # Setup view matrix from rotation and translation
    viewmat = torch.eye(4, device=device, dtype=torch.float32)
    viewmat[:3, :3] = cam_dict['R']
    viewmat[:3, 3] = cam_dict['T']
    viewmat = viewmat.unsqueeze(0)  # [1, 4, 4] for batch dimension
    
    # Setup camera intrinsics matrix K
    K = torch.tensor(
        [[cam_dict['fx'], 0, cam_dict['cx']], 
         [0, cam_dict['fy'], cam_dict['cy']], 
         [0., 0., 1.]],
        dtype=torch.float32,
        device=device
    )
    
    return viewmat, K


def rasterize_gaussians(
    model,
    viewmat: torch.Tensor,
    K: torch.Tensor,
    width: int,
    height: int,
    use_sh_rendering: bool = True,
    device: torch.device = torch.device('cuda')
) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
    """
    Rasterize Gaussian Splatting model to image.
    
    Uses gsplat's rasterization function with RGB+ED (Expected Depth) rendering mode.
    
    Args:
        model: GaussianModel instance with means, quats, scales, opacities, sh, etc.
        viewmat: [1, 4, 4] view matrix
        K: [3, 3] intrinsics matrix
        width: Image width in pixels
        height: Image height in pixels
        use_sh_rendering: If True, use full SH coefficients; if False, use DC only
        device: Torch device
    
    Returns:
        tuple: (render_output, render_alpha, meta)
            - render_output: [1, H, W, 4] tensor with RGB + ED channels
            - render_alpha: [1, H, W, 1] or [1, H, W] alpha compositing weights
            - meta: Dictionary with metadata (radii, means2d, etc.) for training
    """
    from gsplat import rasterization
    
    render_output, render_alpha, meta = rasterization(
        means=model.means,  # [N, 3]
        quats=model.quats,  # [N, 4]
        scales=model.scales,  # [N, 3]
        opacities=model.opacities.squeeze(-1),  # [N]
        colors=model.sh if use_sh_rendering else model._features_dc.squeeze(1),  # [N, K, 3] or [N, 3]
        viewmats=viewmat,  # [1, 4, 4]
        Ks=K[None, ...],  # [1, 3, 3]
        width=width,
        height=height,
        sh_degree=model.sh_degree if use_sh_rendering else None,
        render_mode="RGB+ED",
    )
    
    return render_output, render_alpha, meta


def postprocess_render_outputs(
    render_output: torch.Tensor,
    render_alpha: torch.Tensor,
    device: torch.device = torch.device('cuda')
) -> Dict[str, torch.Tensor]:
    """
    Postprocess rasterization outputs into structured format.
    
    Extracts RGB, depth, and alpha from render_output, normalizes depth by opacity,
    validates values (removes NaN/Inf), and creates mask for valid pixels.
    
    Args:
        render_output: [1, H, W, 4] tensor with RGB channels 0:3 and ED channel 3
        render_alpha: [1, H, W, 1] or [1, H, W] alpha compositing weights
        device: Torch device
    
    Returns:
        Dictionary with keys:
            - 'rgb': [H, W, 3] rendered RGB image
            - 'depth': [H, W] normalized depth map
            - 'alpha': [H, W] alpha weights
            - 'mask': [H, W] boolean mask for valid pixels (alpha > 0.5 and finite depth)
    """
    # Extract RGB and raw depth from batch dimension
    rgb = render_output[0, :, :, 0:3]  # [H, W, 3]
    render_depth_raw = render_output[0, :, :, 3]  # [H, W] - expected depth
    
    # Handle alpha dimensions (may be [H, W, 1] or [H, W])
    if render_alpha.dim() == 3:
        alpha_2d = render_alpha[0, :, :, 0]  # [H, W]
    else:
        alpha_2d = render_alpha[0]  # [H, W]
    
    # Normalize depth by alpha to get per-pixel expected depth
    # Add epsilon to avoid division by zero
    rendered_depth = render_depth_raw / (alpha_2d + 1e-6)
    
    # Replace NaN and Inf values with zeros
    rendered_depth = torch.where(
        torch.isfinite(rendered_depth),
        rendered_depth,
        torch.zeros_like(rendered_depth)
    )
    
    # Create mask for valid pixels (sufficient opacity and finite depth)
    valid_mask = (alpha_2d > 0.5) & torch.isfinite(rendered_depth) & (rendered_depth > 0)
    
    return {
        'rgb': rgb,
        'depth': rendered_depth,
        'alpha': alpha_2d,
        'mask': valid_mask,
    }
