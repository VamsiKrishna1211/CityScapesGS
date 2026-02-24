"""
Loss functions for Gaussian Splatting training.
"""
import torch
from utils import estimate_depth_scale_shift
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

logger = logging.getLogger("cityscape_gs.model")

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

    print(f"Depth Loss: {loss.item():.4f}, Rendered Depth Range: [{render_flat[valid].min().item():.4f}, {render_flat[valid].max().item():.4f}], Prior Depth Range: [{prior_metric_flat[valid].min().item():.4f}, {prior_metric_flat[valid].max().item():.4f}], Valid Pixels: {valid.sum().item()}")
    
    return loss, prior_depth_metric

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
        d_ren = rendered_depth.squeeze()
        d_pri = prior_disparity.squeeze()
        
        # 1. Stricter Masking
        if mask is None:
            # Ignore depth=0 (uninitialized) and very far depth (sky)
            mask = (d_ren > 0.1) & (d_ren < 100.0) & torch.isfinite(d_ren)
        else:
            mask = mask.squeeze() & (d_ren > 0.1)

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
                target_inv_ren.view(1, 1, *d_ren.shape[-2:]), 
                aligned_prior_disp.view(1, 1, *d_pri.shape[-2:])
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
        d_ren = rendered_depth.squeeze()
        d_pri = prior_disparity.squeeze()
        
        # 2. Convert Rendered Metric Depth (Z) to Rendered Disparity (1/Z)
        # Depth Anything V2 is disparity (larger = closer). We must match this format.
        disp_ren = 1.0 / (d_ren + 1e-6)

        # 3. Valid Masking
        # Ignore extremely close/far geometry or uninitialized space
        if mask is None:
            mask = (d_ren > 0.1) & (d_ren < 100.0) & torch.isfinite(d_ren)
        else:
            mask = mask.squeeze() & (d_ren > 0.1) & torch.isfinite(d_ren)

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
            # SSIM requires [B, C, H, W] and performs best when values are 0-1
            norm_disp_ren = self.normalize_01(disp_ren, mask).view(1, 1, *d_ren.shape[-2:])
            norm_target_disp = self.normalize_01(target_disp, mask).view(1, 1, *d_pri.shape[-2:])
            
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
