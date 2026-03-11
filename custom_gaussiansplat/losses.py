"""
Loss functions for Gaussian Splatting training.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import cast

logger = logging.getLogger("cityscape_gs.model")


def _to_bchw_depth(tensor: torch.Tensor) -> torch.Tensor:
    """Normalize depth-like tensors to shape [B, H, W]."""
    if tensor.dim() == 2:
        return tensor.unsqueeze(0).unsqueeze(0)
    if tensor.dim() == 3:
        if tensor.shape[-1] == 1:
            return tensor.squeeze(-1).unsqueeze(0).unsqueeze(0)
        return tensor
    if tensor.dim() == 4:
        if tensor.shape[1] == 1:
            return tensor
        if tensor.shape[-1] == 1:
            return tensor.squeeze(-1).unsqueeze(1)
    raise ValueError(f"Expected depth tensor with shape [H,W], [B,H,W], [B,1,H,W], or [B,H,W,1], got {tuple(tensor.shape)}")


def _to_bchw_mask(mask: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """Normalize masks to shape [B, H, W] and align to ref depth tensor shape."""
    mask_bhw = _to_bchw_depth(mask).bool()
    if mask_bhw.shape != ref.shape:
        raise ValueError(f"Mask shape {tuple(mask_bhw.shape)} does not match reference shape {tuple(ref.shape)}")
    return mask_bhw


def _to_inverse_depth(depth: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Convert metric depth to inverse depth/disparity safely."""
    return 1.0 / torch.clamp(depth, min=eps)

# Entropy loss for Opacities
class EntropyLoss(nn.Module):
    def __init__(self, lambda_min: float = 1e-6, lambda_max: float = 0.5, max_steps: int = 50000):
        super().__init__()
        self.lambda_min = lambda_min
        self.lambda_max = lambda_max
        self.max_steps = max_steps
        self.current_step = 0   
    def schedule_lambda(self) -> float:
        """Schedule the lambda multiplier for the loss magnitude."""
        self.current_step += 1
        return self.lambda_min + (self.lambda_max - self.lambda_min) * (self.current_step / self.max_steps)

    def forward(self, opacities: torch.Tensor) -> torch.Tensor:
        """
        Forces Gaussians to become either fully transparent (to be pruned) 
        or fully opaque (to form solid surfaces).
        
        Variables:
            raw_opacities: Tensor [N, 1] of unactivated, learnable opacity parameters.
            weight: The lambda multiplier for the loss magnitude.
        """
        weight = self.schedule_lambda()
        # 1. Apply sigmoid to get the physical base opacity [0, 1]
        # This is exactly what the rasterizer does internally.
        base_opacities = torch.sigmoid(raw_opacities)
        
        # 2. Calculate the parabolic penalty O * (1 - O)
        # This curve peaks at 0.5 and hits 0 at bounds 0.0 and 1.0.
        entropy = base_opacities * (1.0 - base_opacities)
        
        # 3. Use .mean() to ensure the penalty doesn't scale linearly with the 
        # number of Gaussians, preventing it from overpowering the L1 image loss.
        loss = weight * entropy.mean()
        return loss


# Dinov2 Based LPIPS style loss 
class DINOv2PerceptualLoss(nn.Module):
    def __init__(self, model_size='vits14', layers=[8, 11]):
        super().__init__()
        # 1. Load the smallest model to save VRAM
        self.dino = torch.hub.load('facebookresearch/dinov2', f'dinov2_{model_size}')
        self.layers = layers
        
        # 2. Brutally enforce that this model does not train
        self.dino.eval()
        for param in self.dino.parameters():
            param.requires_grad = False
            
        # DINOv2 expects images normalized with ImageNet stats
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def extract_features(self, x):
        # Normalize input to what DINOv2 expects
        x = (x - self.mean) / self.std
        
        # We need the output of specific intermediate transformer blocks
        features = []
        
        # Forward pass returning intermediate blocks
        ret = self.dino.get_intermediate_layers(x, n=self.layers, return_class_token=False)
        for feat in ret:
            # feat is [Batch, Num_Patches, Feature_Dim]
            features.append(feat)
            
        return features

    def forward(self, pred, target):
        # pred and target should be [B, 3, H, W] bounded between [0, 1]
        
        # 1. Extract features for both (no gradients for the target)
        pred_feats = self.extract_features(pred)
        with torch.no_grad():
            target_feats = self.extract_features(target)
            
        loss = 0.0
        # 2. Compute cosine similarity loss across the chosen layers
        for pf, tf in zip(pred_feats, target_feats):
            # Normalize the vectors
            pf_norm = F.normalize(pf, p=2, dim=-1)
            tf_norm = F.normalize(tf, p=2, dim=-1)
            
            # Cosine similarity is the dot product of normalized vectors
            # We want similarity to be 1, so loss is (1 - similarity)
            cos_sim = (pf_norm * tf_norm).sum(dim=-1) 
            loss += (1.0 - cos_sim).mean()
            
        return loss / len(self.layers)


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
    d_ren_metric = _to_bchw_depth(rendered_depth)
    d_pri = _to_bchw_depth(prior_depth)

    if d_ren_metric.shape != d_pri.shape:
        raise ValueError(f"Depth shape mismatch: rendered {tuple(d_ren_metric.shape)} vs prior {tuple(d_pri.shape)}")

    d_ren = _to_inverse_depth(d_ren_metric, eps=eps)

    finite = torch.isfinite(d_ren_metric) & torch.isfinite(d_ren) & torch.isfinite(d_pri)
    positive = d_ren_metric > eps
    valid = finite & positive
    if mask is not None:
        valid = valid & _to_bchw_mask(mask, d_ren)

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
    d_ren_metric = _to_bchw_depth(rendered_depth)
    d_pri = _to_bchw_depth(prior_depth)

    if d_ren_metric.shape != d_pri.shape:
        raise ValueError(f"Depth shape mismatch: rendered {tuple(d_ren_metric.shape)} vs prior {tuple(d_pri.shape)}")

    d_ren = _to_inverse_depth(d_ren_metric, eps=eps)

    finite = torch.isfinite(d_ren_metric) & torch.isfinite(d_ren) & torch.isfinite(d_pri)
    positive = (d_ren_metric > eps) & (d_pri > eps)
    valid = finite & positive
    if mask is not None:
        valid = valid & _to_bchw_mask(mask, d_ren)

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


def opacity_entropy_regularization(raw_opacities: torch.Tensor, weight: float = 0.0001) -> torch.Tensor:
    """
    Applies Shannon entropy regularization to force opacities to 0 or 1.
    
    Args:
        raw_opacities: Tensor [N] or [N, 1] of raw, unactivated logits.
                       DO NOT pass activated probabilities.
        weight: The scalar multiplier for the loss.
    """
    # 1. Strictly apply sigmoid. We assume the input is ALWAYS logits.
    alpha = torch.sigmoid(raw_opacities)
    
    # 2. Numerically stable binary entropy using xlogy
    entropy = -(
        torch.special.xlogy(alpha, alpha) + 
        torch.special.xlogy(1.0 - alpha, 1.0 - alpha)
    )
    
    # 3. Mean reduction to decouple loss magnitude from point count
    loss = weight * entropy.mean()
    
    return loss


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
    
    d_ren = _to_bchw_depth(render_depth)
    d_pri = _to_bchw_depth(prior_depth)

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
        valid = valid & _to_bchw_mask(mask, d_ren)

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
    d_ren = _to_bchw_depth(render_depth)
    prior_depth_metric = convert_relative_to_metric_depth(d_ren, prior_depth, mask)
    d_pri_metric = _to_bchw_depth(prior_depth_metric)

    valid = (
        torch.isfinite(d_ren)
        & torch.isfinite(d_pri_metric)
        & (d_ren > 1e-6)
        & (d_pri_metric > 1e-6)
    )
    if mask is not None:
        valid = valid & _to_bchw_mask(mask, d_ren)

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


class DepthPriorLoss(nn.Module):
    def __init__(self, lambda_l1=0.5, lambda_ssim=0.5, iterations=1000, thresh=0.01):
        super().__init__()
        self.lambda_l1 = lambda_l1
        self.lambda_ssim = lambda_ssim
        self.iterations = iterations
        self.thresh = thresh

    def forward(self, rendered_depth, prior_disparity, mask=None, ssim_module=None):
        d_ren = _to_bchw_depth(rendered_depth)
        d_pri = _to_bchw_depth(prior_disparity)
        if d_ren.shape != d_pri.shape:
            raise ValueError(f"Depth/disparity shape mismatch: rendered {tuple(d_ren.shape)} vs prior {tuple(d_pri.shape)}")
        
        # 1. Stricter Masking
        if mask is None:
            # Ignore depth=0 (uninitialized) and very far depth (sky)
            mask = (d_ren > 0.1) & (d_ren < 100.0) & torch.isfinite(d_ren)
        else:
            mask = _to_bchw_mask(mask, d_ren) & (d_ren > 0.1) & torch.isfinite(d_ren)

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
        d_ren = _to_bchw_depth(rendered_depth)
        d_pri = _to_bchw_depth(prior_disparity)
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
            mask = _to_bchw_mask(mask, d_ren) & (d_ren > 0.1) & torch.isfinite(d_ren)

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
            logger.warning(f"PriorGradientMatchingLoss: Very few valid pixels ({valid_count.item()}). Returning zero loss to avoid NaN.")
            return (render_depth * 0.0).sum()

        # --- THE FUNDAMENTAL FIX ---
        # Convert the linear rendered depth into inverse depth (disparity).
        # We clamp to 1e-4 to prevent ZeroDivisionError, which would flood the graph with NaNs.
        safe_render_depth = render_depth.clamp(min=1e-4)
        render_disp = 1.0 / safe_render_depth

        # 1. Robust Percentile Normalization (5th/95th) on valid pixels
        # This is more stable than absolute min/max in the presence of outliers.
        ren_masked = torch.where(mask, render_disp, torch.nan).flatten(start_dim=1)
        pri_masked = torch.where(mask, prior_disp, torch.nan).flatten(start_dim=1)

        ren_p05 = torch.nanquantile(ren_masked, 0.05, dim=1, keepdim=True).view(-1, 1, 1, 1)
        ren_p95 = torch.nanquantile(ren_masked, 0.95, dim=1, keepdim=True).view(-1, 1, 1, 1)
        pri_p05 = torch.nanquantile(pri_masked, 0.05, dim=1, keepdim=True).view(-1, 1, 1, 1)
        pri_p95 = torch.nanquantile(pri_masked, 0.95, dim=1, keepdim=True).view(-1, 1, 1, 1)

        # Guard against all-NaN batch items and tiny percentile ranges.
        ren_p05 = torch.nan_to_num(ren_p05, nan=0.0, posinf=0.0, neginf=0.0)
        ren_p95 = torch.nan_to_num(ren_p95, nan=1.0, posinf=1.0, neginf=1.0)
        pri_p05 = torch.nan_to_num(pri_p05, nan=0.0, posinf=0.0, neginf=0.0)
        pri_p95 = torch.nan_to_num(pri_p95, nan=1.0, posinf=1.0, neginf=1.0)

        norm_ren = (render_disp - ren_p05) / (ren_p95 - ren_p05).clamp_min(1e-8)
        norm_pri = (prior_disp - pri_p05) / (pri_p95 - pri_p05).clamp_min(1e-8)

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


# class AffineInvariantDepthLoss(nn.Module):
#     def __init__(self, lambda_l1=1.0):
#         """
#         Calculates the optimal affine transformation (scale and shift) to align 
#         the prior disparity to the rendered disparity using least squares, 
#         then computes the L1 loss.
#         """
#         super().__init__()
#         self.lambda_l1 = lambda_l1

#     def forward(self, render_depth: torch.Tensor, prior_disp: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
#         d_ren = _to_bhw_depth(render_depth)
#         d_pri = _to_bhw_depth(prior_disp)
        
#         # 1. Convert Rendered Metric Depth (Z) to Disparity (1/Z) safely
#         disp_ren = _to_inverse_depth(d_ren)
        
#         # 2. Valid Masking
#         if mask is None:
#             mask_bool = (d_ren > 0.1) & (d_ren < 100.0) & torch.isfinite(d_ren)
#         else:
#             mask_bool = _to_bhw_mask(mask, d_ren) & (d_ren > 0.1) & torch.isfinite(d_ren)
            
#         mask_f = mask_bool.float()
#         valid_pixels = mask_f.sum(dim=(1, 2), keepdim=True).clamp(min=1.0)
        
#         if valid_pixels.sum() < 100:
#             logger.warning(f"Batch has too few valid pixels ({valid_pixels.sum().item()}). Skipping Affine-Invariant Depth Loss to avoid instability.")
#             return torch.tensor(0.0, device=d_ren.device, requires_grad=True)

#         # 3. Compute masked means
#         mean_r = (disp_ren * mask_f).sum(dim=(1, 2), keepdim=True) / valid_pixels
#         mean_p = (d_pri * mask_f).sum(dim=(1, 2), keepdim=True) / valid_pixels
        
#         # 4. Compute masked differences
#         diff_r = (disp_ren - mean_r) * mask_f
#         diff_p = (d_pri - mean_p) * mask_f
        
#         # 5. Compute Covariance and Variance
#         covar = (diff_r * diff_p).sum(dim=(1, 2), keepdim=True) / valid_pixels
#         var_p = (diff_p * diff_p).sum(dim=(1, 2), keepdim=True) / valid_pixels + 1e-8
        
#         # 6. Solve for optimal Scale (s) and Shift (t) analytically
#         s = covar / var_p
#         s = s.clamp(min=1e-4).detach() # Scale must be positive and detached
#         t = (mean_r - s * mean_p).detach()
        
#         # 7. Align the prior
#         aligned_prior = s * d_pri + t
        
#         # 8. Compute L1 loss on the aligned disparity
#         l1_diff = torch.abs(disp_ren - aligned_prior) * mask_f
#         loss = l1_diff.sum() / valid_pixels.sum()
        
#         return self.lambda_l1 * loss

class AffineInvariantDepthLoss(nn.Module):
    def __init__(self, lambda_l1=1.0):
        """
        Calculates the optimal affine transformation (scale and shift) to align 
        the prior disparity to the rendered disparity using least squares, 
        then computes the L1 loss. Highly optimized for VRAM efficiency.
        """
        super().__init__()
        self.lambda_l1 = lambda_l1

    def forward(self, render_depth: torch.Tensor, prior_disp: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        d_ren = _to_bchw_depth(render_depth)
        d_pri = _to_bchw_depth(prior_disp)
        
        # 1. Convert Rendered Metric Depth (Z) to Disparity (1/Z)
        disp_ren = _to_inverse_depth(d_ren)
        
        # 2. Valid Masking
        if mask is None:
            mask_bool = (d_ren > 0.1) & (d_ren < 100.0) & torch.isfinite(d_ren)
        else:
            mask_bool = _to_bchw_mask(mask, d_ren) & (d_ren > 0.1) & torch.isfinite(d_ren)
            
        valid_pixels = mask_bool.sum(dim=(1, 2), keepdim=True).clamp(min=1.0)
        
        if valid_pixels.sum() < 100:
            return torch.tensor(0.0, device=d_ren.device, requires_grad=True)

        # -------------------------------------------------------------------
        # THE MEMORY FIX: Compute Alignment Without Autograd Graph Tracking
        # -------------------------------------------------------------------
        with torch.no_grad():
            mask_f = mask_bool.float()
            
            # Calculate Raw Moments: E[X], E[Y], E[XY], E[Y^2]
            # Because this is in no_grad, the intermediates (disp_ren * mask_f) 
            # are instantly freed from VRAM once the sum is calculated.
            mean_r = (disp_ren * mask_f).sum(dim=(1, 2), keepdim=True) / valid_pixels
            mean_p = (d_pri * mask_f).sum(dim=(1, 2), keepdim=True) / valid_pixels
            
            mean_rp = (disp_ren * d_pri * mask_f).sum(dim=(1, 2), keepdim=True) / valid_pixels
            mean_pp = (d_pri * d_pri * mask_f).sum(dim=(1, 2), keepdim=True) / valid_pixels
            
            # Use moment expansion to get Variance and Covariance
            # Cov(X,Y) = E[XY] - E[X]E[Y]
            covar = mean_rp - (mean_r * mean_p)
            var_p = mean_pp - (mean_p * mean_p)
            
            # Solve for Scale (s) and Shift (t)
            s = (covar / (var_p + 1e-8)).clamp(min=1e-4)
            t = mean_r - s * mean_p
            
            # Compute the aligned prior. This target does not need gradients.
            aligned_prior = s * d_pri + t

        # -------------------------------------------------------------------
        # THE GRADIENT FIX: Calculate Loss
        # -------------------------------------------------------------------
        # disp_ren is OUTSIDE the no_grad block, so gradients will correctly 
        # flow backwards through it from this single L1 operation.
        l1_diff = torch.abs(disp_ren - aligned_prior) * mask_bool.float()
        loss = l1_diff.sum() / valid_pixels.sum()
        
        return self.lambda_l1 * loss

class FastAffineInvariantDepthLoss(nn.Module):
    def __init__(self, lambda_l1=1.0):
        """
        Calculates the optimal affine transformation to align the prior disparity 
        to the rendered disparity, heavily optimized using 1D boolean extraction.
        """
        super().__init__()
        self.lambda_l1 = lambda_l1

    def forward(self, render_depth: torch.Tensor, prior_disp: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        # Normalize input ranks so boolean indexing always operates on [B, H, W].

        d_ren = _to_bchw_depth(render_depth)
        d_pri = _to_bchw_depth(prior_disp)

        if d_ren.shape != d_pri.shape:
            raise ValueError(
                f"Depth/disparity shape mismatch: rendered {tuple(d_ren.shape)} vs prior {tuple(d_pri.shape)}"
            )

        # 1. Convert Rendered Metric Depth (Z) to Disparity (1/Z) safely
        # clamp_min prevents ZeroDivisionError NaNs before they happen
        disp_ren = 1.0 / d_ren.clamp_min(1e-6)
        
        # 2. Valid Masking
        if mask is None:
            mask_bool = (d_ren > 0.1) & (d_ren < 100.0) & torch.isfinite(d_ren)
        else:
            mask_bool = _to_bchw_mask(mask, d_ren) & (d_ren > 0.1) & torch.isfinite(d_ren)
            
        # 3. THE FUNDAMENTAL FIX: 1D Boolean Extraction
        # Extract ONLY the valid pixels into contiguous 1D arrays. 
        # This instantly drops memory footprint and bypasses all float-mask zero-multiplications.
        logger.debug(f"AffineInvariantDepthLoss: Extracting {mask_bool.shape} valid pixels for alignment.")
        valid_r = disp_ren[mask_bool]
        valid_p = d_pri[mask_bool]
        
        if valid_r.numel() < 100:
            return torch.tensor(0.0, device=d_ren.device, requires_grad=True)

        # -------------------------------------------------------------------
        # Fast Alignment (No Autograd Tracking)
        # -------------------------------------------------------------------
        with torch.no_grad():
            # Raw Moments on 1D arrays (Blazing fast)
            mean_r = valid_r.mean()
            mean_p = valid_p.mean()
            
            # Variance and Covariance
            mean_rp = (valid_r * valid_p).mean()
            mean_pp = (valid_p * valid_p).mean()
            
            covar = mean_rp - (mean_r * mean_p)
            var_p = mean_pp - (mean_p * mean_p)
            
            # Solve for Scale (s) and Shift (t)
            s = (covar / (var_p + 1e-8)).clamp(min=1e-4)
            t = mean_r - s * mean_p
            
            # Compute the aligned prior exclusively for the 1D valid pixels
            aligned_prior_valid = s * valid_p + t

        # -------------------------------------------------------------------
        # Memory-Efficient Gradient Calculation
        # -------------------------------------------------------------------
        # Autograd only caches the tiny 1D array of valid pixels, saving massive VRAM.
        # No zero-multiplications required.
        loss = torch.abs(valid_r - aligned_prior_valid).mean()
        
        return self.lambda_l1 * loss


class PearsonCorrelationLoss(nn.Module):
    def __init__(self):
        """
        Maximizes the linear correlation between rendered disparity and prior disparity.
        Inherently invariant to scale and shift.
        """
        super().__init__()

    def forward(self, render_depth: torch.Tensor, prior_disp: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        d_ren = _to_bchw_depth(render_depth)
        d_pri = _to_bchw_depth(prior_disp)
        disp_ren = _to_inverse_depth(d_ren)
        
        if mask is None:
            mask_bool = (d_ren > 0.1) & torch.isfinite(d_ren)
        else:
            mask_bool = _to_bchw_mask(mask, d_ren) & (d_ren > 0.1) & torch.isfinite(d_ren)
            
        mask_f = mask_bool.float()
        valid_pixels = mask_f.sum(dim=(1, 2), keepdim=True).clamp(min=1.0)
        
        if valid_pixels.detach().sum() < 100:
            logger.warning(f"Batch has too few valid pixels ({valid_pixels.sum().item()}). Skipping Pearson correlation loss to avoid instability.")
            return torch.tensor(0.0, device=d_ren.device, requires_grad=True)

        valid_r = disp_ren[mask_bool]
        valid_p = d_pri[mask_bool]    
        
        with torch.no_grad():
            mean_r = valid_r.mean()
            mean_p = valid_p.mean()
            
            diff_r = valid_r - mean_r
            diff_p = valid_p - mean_p
            
            covar = (diff_r * diff_p).mean()
            std_r = torch.sqrt((diff_r * diff_r).mean() + 1e-8)
            std_p = torch.sqrt((diff_p * diff_p).mean() + 1e-8)
        
        pearson_r = covar / (std_r * std_p + 1e-8)
        
        # Minimize (1 - r)
        loss = (1.0 - pearson_r).mean()
        return loss


class SILogLoss(nn.Module):
    def __init__(self, variance_focus: float = 0.85):
        """
        Scale-Invariant Logarithmic Error.
        variance_focus (lambda): 1.0 means fully scale-invariant. 0.85 retains slight scale awareness.
        """
        super().__init__()
        self.variance_focus = variance_focus

    def forward(self, render_depth: torch.Tensor, prior_disp: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        d_ren = _to_bchw_depth(render_depth)
        d_pri = _to_bchw_depth(prior_disp)
        disp_ren = _to_inverse_depth(d_ren)
        
        if mask is None:
            mask_bool = (d_ren > 0.1) & (d_pri > 1e-6) & torch.isfinite(d_ren)
        else:
            mask_bool = _to_bchw_mask(mask, d_ren) & (d_ren > 0.1) & (d_pri > 1e-6) & torch.isfinite(d_ren)
            
        if mask_bool.sum() < 100:
            logger.warning(f"Batch has too few valid pixels ({mask_bool.sum().item()}). Skipping SI-Log loss to avoid instability.")
            return torch.tensor(0.0, device=d_ren.device, requires_grad=True)
            
        r_valid = disp_ren[mask_bool].clamp(min=1e-6)
        p_valid = d_pri[mask_bool].clamp(min=1e-6)
        
        log_diff = torch.log(r_valid) - torch.log(p_valid)
        
        mean_sq_err = (log_diff ** 2).mean()
        sq_mean_err = (log_diff.mean()) ** 2
        
        loss = mean_sq_err - self.variance_focus * sq_mean_err
        return torch.sqrt(loss.clamp(min=1e-6))


class OrdinalDepthLoss(nn.Module):
    def __init__(self, num_samples: int = 5000, margin: float = 0.05):
        """Enforces relative depth ordering by randomly sampling pixel pairs."""
        super().__init__()
        self.num_samples = num_samples
        self.margin = margin

    def forward(self, render_depth: torch.Tensor, prior_disp: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        d_ren = _to_bchw_depth(render_depth)
        d_pri = _to_bchw_depth(prior_disp)
        disp_ren = _to_inverse_depth(d_ren)
        
        B = d_ren.shape[0]
        if mask is None:
            mask_bool = (d_ren > 0.1) & torch.isfinite(d_ren)
        else:
            mask_bool = _to_bchw_mask(mask, d_ren) & (d_ren > 0.1) & torch.isfinite(d_ren)

        loss = torch.tensor(0.0, device=d_ren.device)
        valid_batches = 0

        for b in range(B):
            valid_indices = torch.nonzero(mask_bool[b], as_tuple=False)
            n_valid = valid_indices.shape[0]
            
            if n_valid < self.num_samples * 2:
                continue
                
            idx = torch.randint(0, n_valid, (self.num_samples, 2), device=d_ren.device)
            idx_i, idx_j = valid_indices[idx[:, 0]], valid_indices[idx[:, 1]]
            
            p_i = d_pri[b, idx_i[:, 0], idx_i[:, 1]]
            p_j = d_pri[b, idx_j[:, 0], idx_j[:, 1]]
            r_i = disp_ren[b, idx_i[:, 0], idx_i[:, 1]]
            r_j = disp_ren[b, idx_j[:, 0], idx_j[:, 1]]
            
            # 1 if i is closer than j in prior
            R_ij = torch.sign(p_i - p_j)
            valid_pairs_mask = torch.abs(p_i - p_j) > 1e-3
            
            # Margin ranking
            ranking_loss = torch.relu(-R_ij * (r_i - r_j) + self.margin)
            loss += (ranking_loss * valid_pairs_mask).sum() / (valid_pairs_mask.sum() + 1e-8)
            valid_batches += 1
            
        return loss / max(valid_batches, 1)


class AffineAlignedGradientMatchingLoss(nn.Module):
    def __init__(self):
        """
        Solves the scale/shift mismatch mathematically via least squares, 
        THEN computes the spatial gradients. This completely avoids the fragile 
        percentile/min-max normalizations.
        """
        super().__init__()
        sobel_x = torch.tensor([[[-1.,  0.,  1.], 
                                 [-2.,  0.,  2.], 
                                 [-1.,  0.,  1.]]], dtype=torch.float32)
        sobel_y = torch.tensor([[[-1., -2., -1.], 
                                 [ 0.,  0.,  0.], 
                                 [ 1.,  2.,  1.]]], dtype=torch.float32)
        self.register_buffer("sobel_xy", torch.stack([sobel_x, sobel_y], dim=0))

    def forward(self, render_depth: torch.Tensor, prior_disp: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        d_ren = _to_bchw_depth(render_depth)
        d_pri = _to_bchw_depth(prior_disp)
        disp_ren = _to_inverse_depth(d_ren)

        mask_bool = _to_bchw_mask(mask, d_ren) & (d_ren > 0.1) & torch.isfinite(d_ren)
        mask_f = mask_bool.float().unsqueeze(1)
        disp_ren_bchw = disp_ren.unsqueeze(1)
        d_pri_bchw = d_pri.unsqueeze(1)

        eroded_invalid = F.max_pool2d(1.0 - mask_f, kernel_size=3, stride=1, padding=1)
        eroded_mask_f = 1.0 - eroded_invalid
        valid_count = eroded_mask_f.sum()

        if valid_count < 100:
            logger.warning(f"Batch has too few valid pixels ({valid_count.item()}). Skipping gradient loss to avoid instability.")
            return torch.tensor(0.0, device=d_ren.device, requires_grad=True)

        valid_r = disp_ren_bchw[mask_bool]
        valid_p = d_pri_bchw[mask_bool]

        valid_pixels = mask_f.sum(dim=(1, 2, 3), keepdim=True).clamp_min(1.0)
        with torch.no_grad():
            # Raw Moments on 1D arrays (Blazing fast)
            mean_r = valid_r.mean()
            mean_p = valid_p.mean()
            
            # Variance and Covariance
            mean_rp = (valid_r * valid_p).mean()
            mean_pp = (valid_p * valid_p).mean()
            
            covar = mean_rp - (mean_r * mean_p)
            var_p = mean_pp - (mean_p * mean_p)

            s = (covar / var_p).clamp(min=1e-4).detach()
            t = (mean_r - s * mean_p).detach()

            aligned_prior_valid = s * valid_p + t

        combined_disp = torch.cat([valid_r, aligned_prior_valid], dim=0)
        padded = F.pad(combined_disp, (1, 1, 1, 1), mode='reflect')
        grads = F.conv2d(padded, self.sobel_xy)
        ren_grads, pri_grads = grads.chunk(2, dim=0)

        diff_sum = torch.abs(ren_grads - pri_grads).sum(dim=1, keepdim=True)
        return (diff_sum * eroded_mask_f).sum() / valid_count
