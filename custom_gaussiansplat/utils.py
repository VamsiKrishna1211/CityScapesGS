"""
Utility functions for 3D Gaussian Splatting training.
"""
import torch
import numpy as np
from rich.table import Table
from rich import box


class ShuffledCameraSampler:
    """
    Camera sampler that ensures more uniform distribution across all cameras.
    Shuffles all camera indices and iterates through them before reshuffling.
    This reduces repetition and ensures all cameras are seen before any repeats.
    """
    def __init__(self, num_cameras):
        self.num_cameras = num_cameras
        self.indices = None
        self.current_idx = 0
        self._reshuffle()
    
    def _reshuffle(self):
        """Shuffle the camera indices."""
        self.indices = torch.randperm(self.num_cameras).tolist()
        self.current_idx = 0
    
    def next(self):
        """Get next camera index."""
        if self.current_idx >= self.num_cameras:
            self._reshuffle()
        
        idx = self.indices[self.current_idx]
        self.current_idx += 1
        return idx
    
    def __iter__(self):
        return self
    
    def __next__(self):
        return self.next()


def create_metrics_table(iteration, loss_val, l1_val, ssim_val, lpips_val, psnr_val, num_gauss, phase, total_iterations, verbose=False):
    """
    Create a Rich table displaying training metrics.
    
    Args:
        iteration: Current iteration number
        loss_val: Total loss value
        l1_val: L1 loss value
        ssim_val: SSIM loss value
        lpips_val: LPIPS loss value
        psnr_val: PSNR value
        num_gauss: Number of Gaussians
        phase: Training phase (e.g., "Densification", "Refinement")
        total_iterations: Total number of iterations
        verbose: If True, include additional metrics like GPU memory
    
    Returns:
        Rich Table object
    """
    table = Table(show_header=False, box=box.SIMPLE, padding=(0, 1))
    table.add_column("Metric", style="cyan", width=18)
    table.add_column("Value", style="yellow")
    
    table.add_row("Iteration", f"{iteration:,} / {total_iterations:,}")
    table.add_row("Loss", f"{loss_val:.6f}")
    table.add_row("L1 Loss", f"{l1_val:.6f}")
    table.add_row("SSIM Loss", f"{ssim_val:.6f}")
    table.add_row("LPIPS Loss", f"{lpips_val:.6f}")
    table.add_row("PSNR", f"{psnr_val:.2f} dB")
    table.add_row("Gaussians", f"{num_gauss:,}")
    table.add_row("Phase", phase)
    
    if verbose and torch.cuda.is_available():
        # Add memory info for verbose mode
        mem_allocated = torch.cuda.memory_allocated() / 1024**3
        mem_reserved = torch.cuda.memory_reserved() / 1024**3
        table.add_row("GPU Memory", f"{mem_allocated:.2f} / {mem_reserved:.2f} GB")
    
    return table


def create_viewer_render_fn(model, device, sh_degree):
    """
    Create a render function for the nerfview viewer.
    
    Args:
        model: GaussianModel instance
        device: torch device
        sh_degree: Spherical harmonics degree
    
    Returns:
        Callable render function for viewer
    """
    from gsplat import rasterization
    
    def render_fn(camera_state, render_tab_state):
        """Callable function for the viewer."""
        with torch.no_grad():
            # Get resolution based on preview mode
            if render_tab_state.preview_render:
                W = render_tab_state.render_width
                H = render_tab_state.render_height
            else:
                W = render_tab_state.viewer_width
                H = render_tab_state.viewer_height
            
            c2w = camera_state.c2w
            K = camera_state.get_K((W, H))
            
            # Convert camera parameters
            c2w = torch.from_numpy(c2w).float().to(device)
            K = torch.from_numpy(K).float().to(device)
            
            # Setup camera matrices for gsplat
            viewmat = torch.linalg.inv(c2w)
            
            # Render
            try:
                render, alpha, _ = rasterization(
                    means=model.means,
                    quats=model.quats,
                    scales=model.scales,
                    opacities=model.opacities.squeeze(-1),
                    colors=model.sh,
                    viewmats=viewmat[None, ...],
                    Ks=K[None, ...],
                    width=W,
                    height=H,
                    sh_degree=sh_degree,
                )
                # Clamp colors to [0, 1] range
                render_rgb = torch.clamp(render[0, ..., 0:3], 0, 1)
                render_np = (render_rgb.detach().cpu().numpy() * 255).astype(np.uint8)
                return render_np
            except Exception as e:
                # Return a placeholder if rendering fails
                return np.zeros((H, W, 3), dtype=np.uint8)
    
    return render_fn


def format_phase_description(phase, current_loss, current_l1, current_ssim, current_lpips, current_psnr, current_scale_loss, num_gaussians):
    """
    Format a compact phase description for progress bar.
    
    Args:
        phase: Training phase string
        current_loss: Current total loss
        current_l1: Current L1 loss
        current_ssim: Current SSIM loss
        current_lpips: Current LPIPS loss
        current_psnr: Current PSNR value
        current_scale_loss: Current scale regularization loss
        num_gaussians: Current number of Gaussians
    
    Returns:
        Formatted string
    """
    return (
        f"[cyan]{phase}[/cyan] │ "
        f"Loss: [yellow]{current_loss:.4f}[/yellow] │ "
        f"L1: {current_l1:.4f} │ "
        f"SSIM: {current_ssim:.4f} │ "
        # f"LPIPS: {current_lpips:.6f} │ "
        f"PSNR: {current_psnr:.2f} dB │ "
        # f"Scale Reg: {current_scale_loss:.6f} │ "
        f"GS: [green]{num_gaussians:,}[/green]"
    )


def format_iteration_log(iteration, current_loss, current_l1, current_ssim, current_lpips, current_psnr, current_scale_loss, num_gaussians, iters_per_sec=None):
    """
    Format a compact one-line iteration log.
    
    Args:
        iteration: Current iteration number
        current_loss: Current total loss
        current_l1: Current L1 loss
        current_ssim: Current SSIM loss
        num_gaussians: Current number of Gaussians
        iters_per_sec: Iterations per second (optional)
    
    Returns:
        Formatted string
    """
    log_str = (
        f"[cyan]Iter {iteration:,}[/cyan] │ "
        f"Loss: [yellow]{current_loss:.4f}[/yellow] │ "
        f"L1: {current_l1:.4f} │ "
        f"SSIM: {current_ssim:.4f} │ "
        # f"LPIPS: {current_lpips:.6f} │ "
        f"PSNR: {current_psnr:.2f} dB │ "
        # f"Scale Reg: {current_scale_loss:.6f} │ "
        f"Gaussians: [green]{num_gaussians:,}[/green]"
    )
    
    if iters_per_sec is not None:
        log_str += f" │ {iters_per_sec:.1f} it/s"
    
    return log_str


def estimate_depth_scale_shift(render_depth, prior_depth):
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


def _estimate_lstsq(render_depth, prior_depth):
    """
    Least squares: solve prior * s + t = render for optimal (s, t).
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


def _estimate_median(render_depth, prior_depth):
    """
    Median/MAD based: robust to outliers.
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


def _estimate_standardize(render_depth, prior_depth):
    """
    Z-score standardization: align to same mean and std.
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
