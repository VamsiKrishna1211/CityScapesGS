"""
Loss functions for Gaussian Splatting training.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import cast

logger = logging.getLogger("cityscape_gs.model")


def _to_bhw_depth(tensor: torch.Tensor) -> torch.Tensor:
    """Normalize depth-like tensors to shape [B, H, W]."""
    if tensor.dim() == 2:
        return tensor.unsqueeze(0)
    if tensor.dim() == 3:
        if tensor.shape[-1] == 1:
            return tensor.squeeze(-1).unsqueeze(0)
        return tensor
    if tensor.dim() == 4:
        if tensor.shape[1] == 1:
            return tensor[:, 0]
        if tensor.shape[-1] == 1:
            return tensor[..., 0]
    raise ValueError(f"Expected depth tensor with shape [H,W], [B,H,W], [B,1,H,W], or [B,H,W,1], got {tuple(tensor.shape)}")


def _to_bhw_mask(mask: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """Normalize masks to shape [B, H, W] and align to ref depth tensor shape."""
    mask_bhw = _to_bhw_depth(mask).bool()
    if mask_bhw.shape != ref.shape:
        raise ValueError(f"Mask shape {tuple(mask_bhw.shape)} does not match reference shape {tuple(ref.shape)}")
    return mask_bhw


def _to_inverse_depth(depth: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Convert metric depth to inverse depth/disparity safely."""
    return 1.0 / torch.clamp(depth, min=eps)


def pearson_correlation_depth_loss(
    rendered_depth,
    prior_depth,
    mask=None,
    min_valid_pixels=100,
    eps=1e-8,
):
    """
    Scale-invariant depth/disparity supervision using signed Pearson correlation.

    This is suitable for monocular depth priors (e.g., Depth Anything V2)
    where absolute metric scale is ambiguous.

    Args:
        rendered_depth: [H, W] rendered metric depth from 3DGS
        prior_depth: [H, W] monocular prior (typically disparity / inverse depth)
        mask: Optional [H, W] boolean mask for valid pixels
        min_valid_pixels: Minimum valid samples required to compute correlation
        eps: Numerical stability epsilon

    Returns:
        loss: Scalar loss = 1 - corr
        corr: Pearson correlation in [-1, 1]

    Raises:
        RuntimeError: If insufficient valid pixels or degenerate variance
    """
    d_ren_metric = _to_bhw_depth(rendered_depth)
    d_pri = _to_bhw_depth(prior_depth)

    if d_ren_metric.shape != d_pri.shape:
        raise ValueError(f"Depth shape mismatch: rendered {tuple(d_ren_metric.shape)} vs prior {tuple(d_pri.shape)}")

    d_ren = _to_inverse_depth(d_ren_metric, eps=eps)

    finite = torch.isfinite(d_ren_metric) & torch.isfinite(d_ren) & torch.isfinite(d_pri)
    positive = d_ren_metric > eps
    valid = finite & positive
    if mask is not None:
        valid = valid & _to_bhw_mask(mask, d_ren)

    batch_size = d_ren.shape[0]
    valid_flat = valid.reshape(batch_size, -1)
    valid_counts = valid_flat.sum(dim=1)

    if (valid_counts < min_valid_pixels).any():
        raise RuntimeError(
            "Insufficient valid pixels for Pearson depth loss in batch. "
            f"min={int(valid_counts.min().item())}, required={min_valid_pixels}, "
            f"counts={valid_counts.tolist()}"
        )

    x = d_ren.reshape(batch_size, -1)
    y = d_pri.reshape(batch_size, -1)
    v = valid_flat.to(x.dtype)

    denom = valid_counts.clamp_min(1).to(x.dtype)

    sum_x = (x * v).sum(dim=1)
    sum_y = (y * v).sum(dim=1)
    mean_x = sum_x / denom
    mean_y = sum_y / denom

    x_centered = (x - mean_x[:, None]) * v
    y_centered = (y - mean_y[:, None]) * v

    var_x = (x_centered * x_centered).sum(dim=1) / denom
    var_y = (y_centered * y_centered).sum(dim=1) / denom

    if (var_x < eps).any() or (var_y < eps).any():
        raise RuntimeError(
            "Degenerate variance in Pearson depth loss for at least one batch element. "
            f"min_var_x={var_x.min().item():.4e}, min_var_y={var_y.min().item():.4e}"
        )

    cov_xy = (x_centered * y_centered).sum(dim=1) / denom
    corr = cov_xy / torch.sqrt(var_x * var_y + eps)
    corr = torch.clamp(corr, min=-1.0, max=1.0)

    loss = (1.0 - corr).mean()
    return loss, corr

def silog_depth_loss(
    rendered_depth,
    prior_depth,
    mask=None,
    min_valid_pixels=100,
    lambda_factor=1.0, # 1.0 for full scale invariance
    eps=1e-8,
):
    """
    Scale-Invariant Logarithmic (Si-Log) loss in disparity space.
    Commonly used in NYU Depth V2 and KITTI benchmarks.
    
    Args:
        rendered_depth: [H, W] rendered metric depth from 3DGS
        prior_depth: [H, W] monocular prior disparity / inverse depth
        mask: Optional [H, W] boolean mask
        lambda_factor: The scale-invariance weight (0 to 1). 
                       1.0 means fully scale-invariant.
    """
    d_ren_metric = _to_bhw_depth(rendered_depth)
    d_pri = _to_bhw_depth(prior_depth)

    if d_ren_metric.shape != d_pri.shape:
        raise ValueError(f"Depth shape mismatch: rendered {tuple(d_ren_metric.shape)} vs prior {tuple(d_pri.shape)}")

    d_ren = _to_inverse_depth(d_ren_metric, eps=eps)

    finite = torch.isfinite(d_ren_metric) & torch.isfinite(d_ren) & torch.isfinite(d_pri)
    positive = (d_ren_metric > eps) & (d_pri > eps)
    valid = finite & positive
    if mask is not None:
        valid = valid & _to_bhw_mask(mask, d_ren)

    batch_size = d_ren.shape[0]
    valid_flat = valid.reshape(batch_size, -1)
    valid_counts = valid_flat.sum(dim=1)
    valid_batches = valid_counts >= min_valid_pixels

    if not valid_batches.any():
        return torch.tensor(0.0, device=d_ren.device, requires_grad=True)

    log_ren = torch.zeros_like(d_ren)
    log_pri = torch.zeros_like(d_pri)
    log_ren[valid] = torch.log(d_ren[valid])
    log_pri[valid] = torch.log(d_pri[valid])

    diff = (log_ren - log_pri).reshape(batch_size, -1)
    diff = diff * valid_flat.to(diff.dtype)

    denom = valid_counts.clamp_min(1).to(diff.dtype)
    mse_term = (diff * diff).sum(dim=1) / denom
    scale_term = (diff.sum(dim=1) ** 2) / (denom ** 2)

    loss_per_batch = mse_term - lambda_factor * scale_term
    loss = loss_per_batch[valid_batches].mean()

    return loss * 10.0

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
    # 1. Get the max dimension of each Gaussian
    max_scales = torch.max(scales, dim=1).values
    
    # 2. Define the absolute maximum allowed size (e.g., 5% of the scene)
    # Adjust this threshold based on your specific scene size.
    size_limit = scene_extent * 0.05 
    
    # 3. Only calculate penalty for Gaussians larger than the limit
    # ReLU ensures anything smaller than the limit gets 0 penalty.
    excess_scale = torch.relu(max_scales - size_limit)
    
    # 4. CRITICAL: Use .sum(), NOT .mean()
    # This ensures a massive blob gets hit with the full weight of the penalty,
    # regardless of how many millions of tiny Gaussians exist.
    loss = weight * excess_scale.sum()
    
    return loss
    

def opacity_regularization(opacities, weight=0.0001):
    """
    Penalize high-opacity Gaussians to encourage sparsity.
    
    Args:
        opacities: Tensor of shape [N, 1] containing opacity values
        weight: Regularization weight (typical range: 0.0001-0.001)
    
    Returns:
        Scalar regularization loss
    """
    # This term pushes opacities downward on average.
    # Use together with entropy regularization when binary alpha is desired.

    return weight * opacities.mean()


def opacity_entropy_regularization(opacities, weight=0.0001, eps=1e-8):
    """
    Entropy regularization on opacity to push alpha toward binary values (0 or 1).

    This discourages many tiny semi-transparent "ghost" Gaussians and favors
    solid, geometry-consistent representations.

    Args:
        opacities: Tensor of shape [N, 1] or [N] containing opacity values in [0, 1]
        weight: Regularization weight
        eps: Numerical stability epsilon for log

    Returns:
        Scalar entropy regularization loss
    """
    alpha = opacities

    if alpha.numel() == 0:
        return torch.zeros((), device=alpha.device, dtype=alpha.dtype)

    finite_mask = torch.isfinite(alpha)
    if not finite_mask.any():
        return torch.zeros((), device=alpha.device, dtype=alpha.dtype)

    alpha = alpha[finite_mask]

    # If logits were accidentally passed instead of probabilities, map to [0, 1].
    if ((alpha < 0.0) | (alpha > 1.0)).any():
        alpha = torch.sigmoid(alpha)

    # Numerically stable binary entropy:
    # xlogy handles 0 * log(0) safely as 0 without requiring clamp.
    entropy = -(
        torch.special.xlogy(alpha, alpha)
        + torch.special.xlogy(1.0 - alpha, 1.0 - alpha)
    )

    entropy = torch.nan_to_num(entropy, nan=0.0, posinf=0.0, neginf=0.0)
    return weight * entropy.mean()


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


def gradient_loss(render, gt):
    """
    Calculates the L1 difference between the gradients of the rendered and GT images.
    Inputs: [H, W, C] tensors.
    """
    # Accept both [H, W, C] and [B, H, W, C]
    if render.dim() == 3:
        render = render.unsqueeze(0)
    if gt.dim() == 3:
        gt = gt.unsqueeze(0)

    if render.shape != gt.shape:
        raise ValueError(f"Shape mismatch in gradient_loss: render {tuple(render.shape)} vs gt {tuple(gt.shape)}")

    # Gradient X (width direction)
    r_grad_x = render[:, :, 1:, :] - render[:, :, :-1, :]
    g_grad_x = gt[:, :, 1:, :] - gt[:, :, :-1, :]

    # Gradient Y (height direction)
    r_grad_y = render[:, 1:, :, :] - render[:, :-1, :, :]
    g_grad_y = gt[:, 1:, :, :] - gt[:, :-1, :, :]

    loss_x = torch.abs(r_grad_x - g_grad_x).mean(dim=(1, 2, 3))
    loss_y = torch.abs(r_grad_y - g_grad_y).mean(dim=(1, 2, 3))

    return (loss_x + loss_y).mean()


def sharpness_aware_minimization_loss(render, gt):
    """
    Sharpness-aware reconstruction loss in image-gradient space.

    This proxy encourages rendered images to preserve high-frequency details
    by matching edge gradients with the ground truth image.

    Args:
        render: [H, W, C] rendered RGB image
        gt: [H, W, C] ground-truth RGB image

    Returns:
        Scalar sharpness-aware loss
    """
    return gradient_loss(render, gt)


def convert_relative_to_metric_depth(render_depth, prior_depth, mask=None):
    """
    Convert relative disparity/inverse-depth prior to metric depth.
    
    Estimates scale and shift in disparity space:
        inv(Z_render) ≈ s * D_prior + t
    then converts aligned disparity back to metric depth.
    
    Args:
        render_depth: [H, W] - Metric depth from Gaussian Splatting rendering
        prior_depth: [H, W] - Relative disparity / inverse depth prior
        mask: [H, W] - Boolean mask for valid pixels (e.g., high opacity regions)
    
    Returns:
        prior_depth_metric: [H, W] - Prior depth converted to metric space
        
    Raises:
        RuntimeError: If depth data is completely invalid and cannot be processed
    """
    
    d_ren = _to_bhw_depth(render_depth)
    d_pri = _to_bhw_depth(prior_depth)

    if d_ren.shape != d_pri.shape:
        raise ValueError(f"Depth shape mismatch: rendered {tuple(d_ren.shape)} vs prior {tuple(d_pri.shape)}")

    d_ren_disp = _to_inverse_depth(d_ren)

    valid = (
        torch.isfinite(d_ren)
        & torch.isfinite(d_ren_disp)
        & torch.isfinite(d_pri)
        & (d_ren > 1e-6)
        & (d_pri > 1e-6)
    )
    if mask is not None:
        valid = valid & _to_bhw_mask(mask, d_ren)

    batch_size = d_ren.shape[0]
    valid_flat = valid.reshape(batch_size, -1)
    valid_counts = valid_flat.sum(dim=1)

    if (valid_counts < 10).any():
        raise RuntimeError(
            "Insufficient valid depth values for scale-shift alignment in batch. "
            f"min_valid={int(valid_counts.min().item())}, counts={valid_counts.tolist()}"
        )

    x = d_pri.reshape(batch_size, -1)
    y = d_ren_disp.reshape(batch_size, -1)
    v = valid_flat.to(x.dtype)
    n = valid_counts.to(x.dtype).clamp_min(1.0)

    sum_x = (x * v).sum(dim=1)
    sum_y = (y * v).sum(dim=1)
    sum_xx = (x * x * v).sum(dim=1)
    sum_xy = (x * y * v).sum(dim=1)

    denom = n * sum_xx - sum_x * sum_x
    if (torch.abs(denom) < 1e-8).any():
        raise RuntimeError("Depth alignment failed due to near-zero denominator in batched least squares")

    scale = (n * sum_xy - sum_x * sum_y) / denom
    scale = torch.clamp(scale, min=1e-4)
    shift = (sum_y - scale * sum_x) / n

    aligned_disp = d_pri * scale[:, None, None] + shift[:, None, None]
    prior_depth_metric = 1.0 / torch.clamp(aligned_disp, min=1e-6)

    if render_depth.dim() == 2:
        return prior_depth_metric[0]
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
    d_ren = _to_bhw_depth(render_depth)
    prior_depth_metric = convert_relative_to_metric_depth(d_ren, prior_depth, mask)
    d_pri_metric = _to_bhw_depth(prior_depth_metric)

    valid = (
        torch.isfinite(d_ren)
        & torch.isfinite(d_pri_metric)
        & (d_ren > 1e-6)
        & (d_pri_metric > 1e-6)
    )
    if mask is not None:
        valid = valid & _to_bhw_mask(mask, d_ren)

    batch_size = d_ren.shape[0]
    valid_flat = valid.reshape(batch_size, -1)
    valid_counts = valid_flat.sum(dim=1)

    if (valid_counts < 10).any():
        raise RuntimeError("Insufficient valid pixels for depth loss computation in batch")

    diff = torch.abs(d_ren - d_pri_metric).reshape(batch_size, -1)
    diff = diff * valid_flat.to(diff.dtype)
    loss_per_batch = diff.sum(dim=1) / valid_counts.to(diff.dtype).clamp_min(1.0)
    loss = loss_per_batch.mean()

    if render_depth.dim() == 2:
        return loss, d_pri_metric[0]
    return loss, d_pri_metric

# def ssim_depth_loss(rendered_depth, prior_depth, mask, ssim_module=None):
#     """
#     Depth loss with scale-shift alignment and SSIM for relative depth supervision.
    
#     Converts the prior (relative) depth to metric space using the rendered depth
#     as reference, then computes a combination of L1 and SSIM loss.
    
#     Args:
#         render_depth: [H, W] - Metric depth from Gaussian Splatting rendering
#         prior_depth: [H, W] - Relative depth from Depth Anything V2 (Unnormalized - affine-invariant inverse depth)
#         mask: [H, W] - Boolean mask for valid pixels (e.g., high opacity regions)
#         ssim_module: Optional pre-initialized SSIM module for efficiency
    
#     Returns:
#         Scalar combined depth loss after converting prior to metric space
#         Prior depth converted to metric space [H, W]
        
#     Raises:
#         RuntimeError: If depth data is completely invalid and cannot be processed
#     """

#     rendered_inv_depth = 1.0 / (rendered_depth.flatten() + 1e-6)
#     # prior_depth_metric = convert_relative_to_metric_depth(target_inv_depth, prior_depth, mask)
#     prior_metric_depth = estimate_metric_depth_using_scale_shift(rendered_inv_depth, prior_depth, mask)
    
#     # Compute L1 loss between metric depths (on valid pixels only)
#     if mask is not None:
#         render_flat = rendered_depth[mask]
#         prior_metric_flat = prior_metric_depth[mask]
#     else:
#         render_flat = rendered_depth.reshape(-1)
#         prior_metric_flat = prior_metric_depth.reshape(-1)
    
#     # Filter out invalid values for loss computation
#     valid = (
#         torch.isfinite(render_flat) & 
#         torch.isfinite(prior_metric_flat) & 
#         (render_flat > 1e-6) &
#         (prior_metric_flat > 1e-6)
#     )
    
#     if valid.sum() < 10:
#         raise RuntimeError("Insufficient valid pixels for depth loss computation")
    
#     l1_loss = torch.abs(render_flat[valid] - prior_metric_flat[valid]).mean()
    
#     # Compute SSIM loss if module is provided
#     if ssim_module is not None:
#         # Reshape to [N, C=1, H, W] for SSIM module
#         render_ssim_input = rendered_depth.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
#         prior_ssim_input = prior_depth_metric.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        
#         ssim_score = ssim_module(render_ssim_input, prior_ssim_input)
#         ssim_loss = 1.0 - ssim_score  # Convert similarity to loss
#     else:
#         ssim_loss = torch.tensor(0.0, device=rendered_depth.device)

#     return 0.5 * l1_loss + 0.5 * ssim_loss, prior_depth_metric

# def estimate_metric_depth_using_scale_shift(inv_rendered_depth, prior_depth, mask):
#     """
#     Estimate metric depth from prior relative depth using scale and shift parameters.
    
#     Args:
#         inv_rendered_depth: [H, W] - Metric depth from Gaussian Splatting rendering
#         prior_depth: [H, W] - Relative depth from Depth Anything V2 (Unnormalized - affine-invariant inverse depth)
#         mask: [H, W] - Boolean mask for valid pixels (e.g., high opacity regions)
    
#     Returns:
#         [H, W] - Estimated metric depth
#     """

#     scale, shift = torch_ransac_linear_alignment(inv_rendered_depth, prior_depth, mask)


#     # Apply scale and shift to convert to metric space
#     estimated_inv_depth = scale * prior_depth + shift

#     # Convert back to metric space
#     estimated_metric_depth = 1.0 / (estimated_inv_depth + 1e-6)

#     return estimated_metric_depth

# def torch_ransac_linear_alignment(pred_disp, target_inv_depth, mask, iterations=1000, thresh=0.01, sample_size=100):
#     """
#     High-performance GPU RANSAC for 1D Affine Alignment (y = s*x + t).
    
#     Args:
#         pred_disp: Tensor of DA-v2 disparity [N]
#         target_inv_depth: Tensor of rendered inverse depth [N]
#         mask: Boolean tensor indicating valid pixels [N]
#         iterations: Number of RANSAC trials
#         thresh: Residual threshold for inliers
#         sample_size: Number of points to sample per iteration for the fit
#     """
#     device = pred_disp.device
#     N = pred_disp.shape[0]

#     pred_disp = pred_disp.flatten()
#     target_inv_depth = target_inv_depth.flatten()
#     mask = mask.flatten()
    
#     # Pre-filter: Remove invalid depths (zeros or inf)
#     mask = mask & (target_inv_depth > 0) & torch.isfinite(target_inv_depth)
#     x = pred_disp[mask]
#     y = target_inv_depth[mask]
#     num_pts = x.shape[0]

#     best_inlier_count = 0
#     best_params = (1.0, 0.0) # (scale, shift)

#     # Vectorized sampling: Generate all random indices at once
#     # We pick 2 points per iteration to define a line
#     idx = torch.randint(0, num_pts, (iterations, 2), device=device)
    
#     x1, x2 = x[idx[:, 0]], x[idx[:, 1]]
#     y1, y2 = y[idx[:, 0]], y[idx[:, 1]]

#     # Solve for s and t for every iteration in parallel
#     # s = (y2 - y1) / (x2 - x1)
#     # t = y1 - s * x1
#     scales = (y2 - y1) / (x2 - x1 + 1e-8)
#     shifts = y1 - scales * x1

#     # To save memory, we can loop through the iterations to check inliers
#     # or use a bit of broadcasting if memory allows. 
#     # For 1D, a simple loop over parallel batches is usually fastest.
#     for i in range(iterations):
#         s, t = scales[i], shifts[i]
        
#         # Estimate: y_hat = s*x + t
#         residuals = torch.abs((s * x + t) - y)
#         inliers = residuals < thresh
#         inlier_count = torch.sum(inliers)

#         if inlier_count > best_inlier_count:
#             best_inlier_count = inlier_count
#             best_params = (s, t)

#     # Final Refinement: Re-fit using all inliers found in the best model
#     s, t = best_params
#     final_residuals = torch.abs((s * x + t) - y)
#     final_inliers_mask = final_residuals < thresh
    
#     # Ordinary Least Squares on all inliers
#     x_final = x[final_inliers_mask]
#     y_final = y[final_inliers_mask]
    
#     # Simple linear regression formula
#     # s = Cov(x,y) / Var(x)
#     x_mean = x_final.mean()
#     y_mean = y_final.mean()
#     s_refined = torch.sum((x_final - x_mean) * (y_final - y_mean)) / torch.sum((x_final - x_mean)**2)
#     t_refined = y_mean - s_refined * x_mean

#     return s_refined, t_refined

class DepthPriorLoss(nn.Module):
    def __init__(self, lambda_l1=0.5, lambda_ssim=0.5, iterations=1000, thresh=0.01):
        super().__init__()
        self.lambda_l1 = lambda_l1
        self.lambda_ssim = lambda_ssim
        self.iterations = iterations
        self.thresh = thresh

    def forward(self, rendered_depth, prior_disparity, mask=None, ssim_module=None):
        d_ren = _to_bhw_depth(rendered_depth)
        d_pri = _to_bhw_depth(prior_disparity)
        if d_ren.shape != d_pri.shape:
            raise ValueError(f"Depth/disparity shape mismatch: rendered {tuple(d_ren.shape)} vs prior {tuple(d_pri.shape)}")
        
        # 1. Stricter Masking
        if mask is None:
            # Ignore depth=0 (uninitialized) and very far depth (sky)
            mask = (d_ren > 0.1) & (d_ren < 100.0) & torch.isfinite(d_ren)
        else:
            mask = _to_bhw_mask(mask, d_ren) & (d_ren > 0.1) & torch.isfinite(d_ren)

        # 2. Target is INVERSE depth
        target_inv_ren = 1.0 / (d_ren + 1e-6)

        # Handle empty mask case immediately to avoid NaN
        if mask.sum() < 100:
            # print(f"Warning: Very few valid pixels for depth loss ({mask.sum().item()}). Returning zero loss to avoid NaN.")
            return torch.tensor(0.0, device=d_ren.device, requires_grad=True), target_inv_ren, None


        # 3. Alignment (Keep scale/shift positive and sane)
        scale, shift = self.torch_ransac_linear_alignment(
            d_pri.detach(), 
            target_inv_ren.detach(),
            mask
        )

        # 4. DISPARITY LOSS (Much more stable than Metric Loss)
        # We compare (s * prior + t) vs (1 / render)
        aligned_prior_disp = scale * d_pri + shift
        
        # Use L1 on disparity, not metric depth!
        l1_loss = F.l1_loss(aligned_prior_disp[mask], target_inv_ren[mask])
        
        # 5. SSIM on Disparity
        ssim_loss = torch.tensor(0.0, device=d_ren.device)
        if ssim_module is not None:
            # Use disparity for SSIM
            ssim_val = ssim_module(
                target_inv_ren.unsqueeze(1),
                aligned_prior_disp.unsqueeze(1)
            )
            ssim_loss = 1.0 - ssim_val

        total_loss = self.lambda_l1 * l1_loss + self.lambda_ssim * ssim_loss
        
        # Only for visualization/return do we go back to metric
        with torch.no_grad():
            prior_metric_depth = 1.0 / torch.clamp(aligned_prior_disp, min=1e-4)
            # print(f"Depth Prior Loss: {total_loss.item():.4f}, L1: {l1_loss.item():.4f}, SSIM: {ssim_loss.item():.4f}, Scale: {scale.item():.4f}, Shift: {shift.item():.4f}, Valid Pixels: {mask.sum().item()}")

        return total_loss, target_inv_ren, prior_metric_depth

    def normalize_01(self, tensor, mask):
        m = tensor[mask]
        if m.numel() == 0: return tensor
        vmin, vmax = m.min(), m.max()
        return (tensor - vmin) / (vmax - vmin + 1e-8)

    @torch.no_grad()
    def torch_ransac_linear_alignment(self, x_raw, y_raw, mask):
        """Vectorized RANSAC for y = sx + t"""
        x = x_raw[mask]
        y = y_raw[mask]
        device = x.device
        num_pts = x.shape[0]

        if num_pts < 10: # Safety check
            return torch.tensor(1.0, device=device), torch.tensor(0.0, device=device)

        # Batch sampling
        idx = torch.randint(0, num_pts, (self.iterations, 2), device=device)
        x1, x2 = x[idx[:, 0]], x[idx[:, 1]]
        y1, y2 = y[idx[:, 0]], y[idx[:, 1]]

        scales = (y2 - y1) / (x2 - x1 + 1e-8)
        shifts = y1 - scales * x1

        # Evaluate models (Vectorized inlier counting)
        # For large images, a loop over iterations is more memory efficient than a large matrix
        best_count = -1
        best_s, best_t = 1.0, 0.0
        
        # Randomly sub-sample points for faster inlier checking if dataset is huge
        if num_pts > 50_000:
            check_idx = torch.randperm(num_pts, device=device)[:50_000]
            x_sub, y_sub = x[check_idx], y[check_idx]
        else:
            x_sub, y_sub = x, y

        dynamic_thresh = torch.median(torch.abs(y - torch.median(y))) * 0.5
        if dynamic_thresh < 1e-5: dynamic_thresh = self.thresh

        for i in range(self.iterations):
            s, t = scales[i], shifts[i]
            if s <= 0: continue
            res = torch.abs((s * x_sub + t) - y_sub)
            count = (res < dynamic_thresh).sum()
            if count > best_count:
                best_count = count
                best_s, best_t = s, t

        return best_s, best_t


class DepthPriorLossLeastSquares(nn.Module):
    def __init__(self, lambda_l1=0.5, lambda_ssim=0.5):
        super().__init__()
        self.lambda_l1 = lambda_l1
        self.lambda_ssim = lambda_ssim

    def forward(self, rendered_depth, prior_disparity, mask=None, ssim_module=None):
        # 1. Dimensions
        d_ren = _to_bhw_depth(rendered_depth)
        d_pri = _to_bhw_depth(prior_disparity)
        if d_ren.shape != d_pri.shape:
            raise ValueError(f"Depth/disparity shape mismatch: rendered {tuple(d_ren.shape)} vs prior {tuple(d_pri.shape)}")
        
        # 2. Convert Rendered Metric Depth (Z) to Rendered Disparity (1/Z)
        # Depth Anything V2 is disparity (larger = closer). We must match this format.
        disp_ren = 1.0 / (d_ren + 1e-6)

        # 3. Valid Masking
        # Ignore extremely close/far geometry or uninitialized space
        if mask is None:
            mask = (d_ren > 0.1) & (d_ren < 100.0) & torch.isfinite(d_ren)
        else:
            mask = _to_bhw_mask(mask, d_ren) & (d_ren > 0.1) & torch.isfinite(d_ren)

        # Safety Check: Prevent NaN crashes
        if mask.sum() < 100:
            return torch.tensor(0.0, device=d_ren.device, requires_grad=True), disp_ren, d_pri

        # 4. Instant Analytic Alignment (Replaces RANSAC)
        # Find s and t such that: s * prior_disparity + t ≈ disp_ren
        s, t = self.least_squares_alignment(d_pri.detach(), disp_ren.detach(), mask)

        # Create the aligned target
        aligned_prior_disp = s * d_pri + t
        
        # Detach the target so the optimizer doesn't try to change the prior to match the render
        target_disp = aligned_prior_disp.detach()

        # 5. Loss Calculation
        # L1 Loss on the disparity space
        l1_loss = F.l1_loss(disp_ren[mask], target_disp[mask])
        
        # SSIM Loss (Structure)
        ssim_loss = torch.tensor(0.0, device=d_ren.device)
        if ssim_module is not None:
            # Use shared normalization bounds to preserve alignment.
            target_valid = target_disp[mask]
            if target_valid.numel() > 0:
                vmin, vmax = target_valid.min(), target_valid.max()
                denom = vmax - vmin + 1e-8
                norm_disp_ren = ((disp_ren - vmin) / denom).clamp(0.0, 1.0).unsqueeze(1)
                norm_target_disp = ((target_disp - vmin) / denom).clamp(0.0, 1.0).unsqueeze(1)

                ssim_val = ssim_module(norm_disp_ren, norm_target_disp)
                ssim_loss = 1.0 - ssim_val

        total_loss = self.lambda_l1 * l1_loss + self.lambda_ssim * ssim_loss

        # Return inverse depths for easy visualization
        return total_loss, disp_ren, target_disp

    def normalize_01(self, tensor, mask):
        """Normalizes a tensor to [0, 1] range based on its masked min/max."""
        m = tensor[mask]
        if m.numel() == 0: return tensor
        vmin, vmax = m.min(), m.max()
        return (tensor - vmin) / (vmax - vmin + 1e-8)

    @torch.no_grad()
    def least_squares_alignment(self, prior, target, mask):
        """
        Analytically solves for s and t in: target = s * prior + t
        This is O(N) complexity and instantaneous on the GPU.
        """
        p = prior[mask]
        t = target[mask]
        
        # Calculate means
        mean_p = p.mean()
        mean_t = t.mean()
        
        # Calculate covariance and variance
        p_centered = p - mean_p
        t_centered = t - mean_t
        
        var_p = (p_centered ** 2).mean()
        cov_pt = (p_centered * t_centered).mean()
        
        # Solve for scale (s = cov(p,t) / var(p))
        if var_p < 1e-8:
            s = torch.tensor(1.0, device=prior.device)
        else:
            s = cov_pt / var_p
            
        # Physical constraint: Scale MUST be positive. 
        # If scale is negative, the math is trying to invert near/far relationships.
        s = torch.clamp(s, min=1e-4)
        
        # Solve for shift (t = mean(t) - s * mean(p))
        t = mean_t - s * mean_p
        
        return s, t


class DynamicEdgeAwareDepthSmoothnessLoss(nn.Module):
    def __init__(self, start_alpha=0.5, end_alpha=2.5, max_steps=15000):
        """Edge-aware depth smoothness loss with dynamic scheduling of the edge sensitivity parameter (alpha)."""
        super().__init__()
        
        # 1. Scheduler State Variables
        self.start_alpha = start_alpha
        self.end_alpha = end_alpha
        self.max_steps = max_steps
        self.current_step = 0
        self.current_alpha = start_alpha

        # 2. Mathematical Constants (Sobel Kernels)
        sobel_x = torch.tensor([[-1.,  0.,  1.], 
                                [-2.,  0.,  2.], 
                                [-1.,  0.,  1.]], dtype=torch.float32).view(1, 1, 3, 3)
                                
        sobel_y = torch.tensor([[-1., -2., -1.], 
                                [ 0.,  0.,  0.], 
                                [ 1.,  2.,  1.]], dtype=torch.float32).view(1, 1, 3, 3)
        
        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)

    def step(self):
        """Mimics a PyTorch LR Scheduler. Call this once per training iteration."""
        self.current_step += 1
        factor = min(1.0, self.current_step / self.max_steps)
        self.current_alpha = self.start_alpha + factor * (self.end_alpha - self.start_alpha)

    def get_last_alpha(self):
        return self.current_alpha

    def compute_loss(self, depth: torch.Tensor, image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Executes the robust edge-aware smoothness calculation strictly on valid geometry.
        
        Args:
            depth: Rendered metric depth map [B, 1, H, W]
            image: Corresponding GT RGB image [B, 3, H, W]
            mask: Boolean or Float mask of valid pixels [B, 1, H, W]
        """
        # 0. Fast Safety Check
        if mask.sum() < 100:
            return torch.tensor(0.0, device=depth.device, requires_grad=True)

        # 1. Erode the mask to prevent border artifacts from the 3x3 kernel
        # We invert the mask, run max_pool (which expands the invalid areas by 1 pixel), and invert back.
        mask_float = mask.float()
        invalid_mask = 1.0 - mask_float
        eroded_invalid = F.max_pool2d(invalid_mask, kernel_size=3, stride=1, padding=1)
        eroded_valid_mask = (1.0 - eroded_invalid) > 0.5

        # 2. Pad inputs to prevent edge artifacts (Reflection padding)
        depth_padded = F.pad(depth, (1, 1, 1, 1), mode='reflect')
        
        # 3. Calculate depth gradients using the registered buffers
        sobel_x = cast(torch.Tensor, self.sobel_x)
        sobel_y = cast(torch.Tensor, self.sobel_y)
        depth_dx = torch.abs(F.conv2d(depth_padded, sobel_x))
        depth_dy = torch.abs(F.conv2d(depth_padded, sobel_y))

        # 4. Calculate image gradients (Convert RGB to Grayscale first)
        gray_image = image.mean(dim=1, keepdim=True)
        gray_padded = F.pad(gray_image, (1, 1, 1, 1), mode='reflect')
        
        image_dx = torch.abs(F.conv2d(gray_padded, sobel_x))
        image_dy = torch.abs(F.conv2d(gray_padded, sobel_y))

        # 5. Apply Edge-Aware Weights
        weight_x = torch.exp(-self.current_alpha * image_dx)
        weight_y = torch.exp(-self.current_alpha * image_dy)

        # 6. Compute Final Weighted Tensors
        loss_x = depth_dx * weight_x
        loss_y = depth_dy * weight_y

        # 7. Strictly extract and average only the safe, eroded valid pixels
        valid_count = eroded_valid_mask.sum()
        
        if valid_count < 10:
            return torch.tensor(0.0, device=depth.device, requires_grad=True)
            
        final_loss = (loss_x[eroded_valid_mask].sum() + loss_y[eroded_valid_mask].sum()) / valid_count
        return final_loss
    
    def forward(self, depth, image, mask):
        return self.compute_loss(depth, image, mask)


class PriorGradientMatchingLoss(nn.Module):
    def __init__(self):
        """
        Matches the spatial gradients (slopes/edges) of the rendered disparity 
        directly to the unnormalized prior disparity by locally normalizing both 
        to a [0, 1] range before gradient calculation.
        """
        super().__init__()
        
        # Fixed Mathematical Constants (Sobel Kernels)
        sobel_x = torch.tensor([[-1.,  0.,  1.], 
                                [-2.,  0.,  2.], 
                                [-1.,  0.,  1.]], dtype=torch.float32).view(1, 1, 3, 3)
                                
        sobel_y = torch.tensor([[-1., -2., -1.], 
                                [ 0.,  0.,  0.], 
                                [ 1.,  2.,  1.]], dtype=torch.float32).view(1, 1, 3, 3)
        
        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)

    def compute_loss(self, render_disp: torch.Tensor, prior_disp: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Executes structural gradient matching.
        
        Args:
            render_disp: Rendered disparity map (1 / Z) [B, 1, H, W]
            prior_disp: Raw, unnormalized DA-v2 disparity map [B, 1, H, W]
            mask: Boolean or Float mask of valid geometry [B, 1, H, W]
        """
        # 0. Fast Safety Check
        if mask.sum() < 100:
            return torch.tensor(0.0, device=render_disp.device, requires_grad=True)

        # 1. Independent Min-Max Normalization to [0, 1]
        ren_valid = render_disp[mask]
        pri_valid = prior_disp[mask]
        
        # Protect against empty slices
        if ren_valid.numel() < 10 or pri_valid.numel() < 10:
            return torch.tensor(0.0, device=render_disp.device, requires_grad=True)
            
        ren_min, ren_max = ren_valid.min(), ren_valid.max()
        pri_min, pri_max = pri_valid.min(), pri_valid.max()
        
        # Normalize the full tensors (invalid regions will be excluded by the eroded mask later)
        norm_ren_disp = (render_disp - ren_min) / (ren_max - ren_min + 1e-8)
        norm_pri_disp = (prior_disp - pri_min) / (pri_max - pri_min + 1e-8)

        # 2. Erode the mask to prevent 3x3 kernel border artifacts
        # If a pixel touches the sky, its gradient is fake. We must exclude it.
        mask_float = mask.float()
        invalid_mask = 1.0 - mask_float
        eroded_invalid = F.max_pool2d(invalid_mask, kernel_size=3, stride=1, padding=1)
        eroded_valid_mask = (1.0 - eroded_invalid) > 0.5

        # 3. Reflection Padding to prevent edge ringing
        ren_padded = F.pad(norm_ren_disp, (1, 1, 1, 1), mode='reflect')
        pri_padded = F.pad(norm_pri_disp, (1, 1, 1, 1), mode='reflect')
        
        # 4. Calculate Spatial Gradients (Sobel)
        sobel_x = cast(torch.Tensor, self.sobel_x)
        sobel_y = cast(torch.Tensor, self.sobel_y)
        
        # We do NOT use absolute values yet. We need the exact directional slope.
        ren_dx = F.conv2d(ren_padded, sobel_x)
        ren_dy = F.conv2d(ren_padded, sobel_y)
        
        pri_dx = F.conv2d(pri_padded, sobel_x)
        pri_dy = F.conv2d(pri_padded, sobel_y)

        # 5. Compute the L1 difference between the gradients
        # This penalizes the rendered geometry if its normalized slope does not exactly match the prior's slope
        diff_dx = torch.abs(ren_dx - pri_dx)
        diff_dy = torch.abs(ren_dy - pri_dy)

        # 6. Extract and average over safe, eroded valid pixels
        valid_count = eroded_valid_mask.sum()
        
        if valid_count < 10:
            return torch.tensor(0.0, device=render_disp.device, requires_grad=True)
            
        final_loss = (diff_dx[eroded_valid_mask].sum() + diff_dy[eroded_valid_mask].sum()) / valid_count
        
        return final_loss
    
    def forward(self, render_disp, prior_disp, mask):
        return self.compute_loss(render_disp, prior_disp, mask)


class FastPriorGradientMatchingLoss(nn.Module):
    def __init__(self):
        """
        Matches the spatial gradients (slopes/edges) of the rendered disparity 
        directly to the unnormalized prior disparity by locally normalizing both 
        to a [0, 1] range before gradient calculation.
        """
        super().__init__()
        
        # 1. Fuse the Sobel kernels into a single tensor [Out_C, In_C, H, W]
        # Out_C = 2 (channel 0 is dx, channel 1 is dy)
        sobel_x = torch.tensor([[[-1.,  0.,  1.], 
                                 [-2.,  0.,  2.], 
                                 [-1.,  0.,  1.]]], dtype=torch.float32)
                                
        sobel_y = torch.tensor([[[-1., -2., -1.], 
                                 [ 0.,  0.,  0.], 
                                 [ 1.,  2.,  1.]]], dtype=torch.float32)
        
        # Shape: [2, 1, 3, 3]
        sobel_xy = torch.stack([sobel_x, sobel_y], dim=0)
        self.register_buffer("sobel_xy", sobel_xy)

    def compute_loss(self, render_depth: torch.Tensor, prior_disp: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Executes structural gradient matching.
        
        Args:
            render_depth: Rendered metric depth map (Z) [B, 1, H, W]
            prior_disp: Raw, unnormalized DA-v2 disparity map (1/Z) [B, 1, H, W]
            mask: Boolean mask of valid geometry [B, 1, H, W]
        """
        mask_f = mask.float()
        
        # 0. Erode mask first (using float math to avoid boolean syncs later)
        eroded_invalid = F.max_pool2d(1.0 - mask_f, kernel_size=3, stride=1, padding=1)
        eroded_mask_f = 1.0 - eroded_invalid
        
        valid_count = eroded_mask_f.sum()
        
        # Safe early exit that doesn't break the graph
        if valid_count < 10:
             return (render_depth * 0.0).sum()

        # --- THE FUNDAMENTAL FIX ---
        # Convert the linear rendered depth into inverse depth (disparity).
        # We clamp to 1e-4 to prevent ZeroDivisionError, which would flood the graph with NaNs.
        safe_render_depth = render_depth.clamp(min=1e-4)
        render_disp = 1.0 / safe_render_depth

        # 1. Min-Max Normalization WITHOUT CUDA Syncs
        # Apply normalization to the newly computed render_disp and the prior_disp
        ren_min = torch.where(mask, render_disp, torch.inf).amin(dim=(1, 2, 3), keepdim=True)
        ren_max = torch.where(mask, render_disp, -torch.inf).amax(dim=(1, 2, 3), keepdim=True)
        
        pri_min = torch.where(mask, prior_disp, torch.inf).amin(dim=(1, 2, 3), keepdim=True)
        pri_max = torch.where(mask, prior_disp, -torch.inf).amax(dim=(1, 2, 3), keepdim=True)
        
        norm_ren = (render_disp - ren_min) / (ren_max - ren_min + 1e-8)
        norm_pri = (prior_disp - pri_min) / (pri_max - pri_min + 1e-8)

        # 2. Fuse the inputs into a single batch for a single padding/conv operation
        # Shape becomes [2 * B, 1, H, W]
        combined_disp = torch.cat([norm_ren, norm_pri], dim=0)
        
        # 3. Single Pad
        padded = F.pad(combined_disp, (1, 1, 1, 1), mode='reflect')
        
        # 4. Single Fused Convolution
        # Input: [2*B, 1, H, W], Kernel: [2, 1, 3, 3] -> Output: [2*B, 2, H, W]
        # Channel 0 is dx, Channel 1 is dy
        grads = F.conv2d(padded, self.sobel_xy)
        
        # 5. Split back into render and prior
        ren_grads, pri_grads = grads.chunk(2, dim=0)
        
        # 6. Compute L1 difference across both dx and dy simultaneously
        diff_sum = torch.abs(ren_grads - pri_grads).sum(dim=1, keepdim=True)

        # 7. Apply eroded mask as a float multiplication (NO BOOLEAN INDEXING)
        final_loss = (diff_sum * eroded_mask_f).sum() / (valid_count + 1e-8)
        
        return final_loss
    
    def forward(self, render_depth, prior_disp, mask):
        return self.compute_loss(render_depth, prior_disp, mask)
