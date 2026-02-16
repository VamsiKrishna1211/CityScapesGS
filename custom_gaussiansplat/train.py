import torch
from torch.utils.data import DataLoader
from gsplat import rasterization, DefaultStrategy
from torchmetrics.image import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio, LearnedPerceptualImagePatchSimilarity, MultiScaleStructuralSimilarityIndexMeasure
import os
from pathlib import Path
import argparse
import time
import numpy as np
import logging
from typing import Tuple

try:
    import nerfview
    import viser
    VIEWER_AVAILABLE = True
except ImportError:
    VIEWER_AVAILABLE = False

from rich.console import Console
from rich.logging import RichHandler
from rich.progress import Progress, SpinnerColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn, TimeRemainingColumn, TextColumn
from rich.table import Table
from rich.panel import Panel
from rich.live import Live
from rich.layout import Layout
from rich import box

from dataset import ColmapDataset
from model import GaussianModel, inverse_sigmoid
from gs_types import GSOptimizers
from utils import create_viewer_render_fn, format_phase_description
import losses
from logger import GaussianSplattingLogger
from losses import depth_loss



def setup_logger(verbosity: int, output_dir: Path) -> logging.Logger:
    """Setup logging with RichHandler.
    
    Args:
        verbosity: 0=QUIET, 1=NORMAL, 2=VERBOSE, 3=DEBUG
        output_dir: Directory to save log file
    
    Returns:
        Configured logger instance
    """
    # Map verbosity to logging levels
    level_map = {
        0: logging.WARNING,   # QUIET: only warnings and errors
        1: logging.INFO,      # NORMAL: info and above
        2: logging.DEBUG,     # VERBOSE: debug and above
        3: logging.DEBUG,     # DEBUG: everything
    }
    
    log_level = level_map.get(verbosity, logging.INFO)
    
    # Create logger
    logger = logging.getLogger("cityscape_gs.train")
    logger.setLevel(log_level)
    logger.handlers.clear()  # Clear any existing handlers
    
    # Rich console handler for terminal output
    console_handler = RichHandler(
        rich_tracebacks=True,
        markup=True,
        show_time=True,
        show_path=verbosity >= 3,  # Show file path only in DEBUG mode
    )
    console_handler.setLevel(log_level)
    logger.addHandler(console_handler)
    
    # File handler for persistent logs
    log_file = output_dir / "training.log"
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setLevel(logging.DEBUG)  # Always log everything to file
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    return logger


def train_pipeline(
    colmap_path,
    images_path,
    output_dir='./output',
    iterations=7000,
    densify_from_iter=500,
    densify_until_iter=15000,
    densify_interval=100,
    opacity_reset_interval=3000,
    opacity_reset_value=0.01,
    grad_threshold=0.02,
    max_screen_size=20,
    lr_means=0.00016,
    lr_scales=0.005,
    lr_quats=0.001,
    lr_opacities=0.05,
    lr_sh=0.0025,
    save_interval=1000,
    log_interval=500,
    verbosity=1,
    enable_viewer=False,
    viewer_port=8080,

    # TensorBoard logging
    enable_tensorboard=True,
    tensorboard_image_interval=500,
    tensorboard_histogram_interval=1000,
    # Floater prevention options
    enable_scale_reg=False,
    scale_reg_weight=0.01,
    enable_opacity_reg=False,
    opacity_reg_weight=0.0001,

    # Spherical harmonics options
    sh_degree=3,
    use_sh_rendering=True,

    # Depth supervision options
    enable_depth_loss=False,
    depth_loss_weight=0.1,
    depth_loss_start_iter=1000,
):
    """
    Train a 3D Gaussian Splatting model.
    
    Args:
        colmap_path: Path to COLMAP sparse reconstruction directory
        images_path: Path to training images directory
        output_dir: Output directory for checkpoints and results
        iterations: Total number of training iterations
        densify_from_iter: Start densification at this iteration
        densify_until_iter: Stop densification at this iteration
        densify_interval: Densify every N iterations
        opacity_reset_interval: Reset opacity every N iterations
        opacity_reset_value: Opacity value to reset to (0.01-0.1)
        grad_threshold: Gradient threshold for densification
        max_screen_size: Maximum screen size in pixels for pruning
        lr_means: Learning rate for Gaussian positions
        lr_scales: Learning rate for Gaussian scales
        lr_quats: Learning rate for Gaussian rotations
        lr_opacities: Learning rate for Gaussian opacities
        lr_sh: Learning rate for spherical harmonics
        save_interval: Save checkpoint every N iterations
        log_interval: Log progress every N iterations
        verbosity: Verbosity level (0=QUIET, 1=NORMAL, 2=VERBOSE, 3=DEBUG)
        enable_viewer: Enable interactive 3D viewer during training
        viewer_port: Port for the viewer server
        
        Floater Prevention Options:
        enable_scale_reg: Enable scale regularization loss
        scale_reg_weight: Weight for scale regularization (0.01-0.1)
        enable_depth_culling: Enable depth-based culling of near-camera Gaussians
        near_plane_threshold: Minimum camera-to-Gaussian distance in meters
        enable_aggressive_pruning: Enable world-space scale pruning
        max_world_scale: Max scale as fraction of scene extent (e.g., 0.1 = 10%)
        enable_visibility_tracking: Enable multi-view consistency checking
        min_view_count: Minimum number of views a Gaussian must be visible in
        
        Spherical Harmonics Options:
        sh_degree: Degree of spherical harmonics (0-3, where 0=DC only, 3=full 3rd order)
        use_sh_rendering: Whether to use SH during rendering (if False, uses DC component only)
        
        Depth Supervision Options:
        enable_depth_loss: Enable depth supervision from Depth Anything V2
        depth_loss_weight: Weight for depth loss (0.05-0.2 recommended)
        depth_loss_start_iter: Start applying depth loss after this many iterations
        
        TensorBoard Options:
        enable_tensorboard: Enable TensorBoard logging
        tensorboard_image_interval: Log images every N iterations
        tensorboard_histogram_interval: Log histograms every N iterations
    """
    device = torch.device('cuda')
    
    # Initialize Rich console for UI elements (tables, panels, progress bars)
    console = Console()
    
    # Verbosity levels: 0=QUIET, 1=NORMAL, 2=VERBOSE, 3=DEBUG
    VERBOSITY = verbosity
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Setup structured logging with RichHandler
    logger = setup_logger(verbosity, output_path)
    logger.info("[bold cyan]🚀 Starting 3D Gaussian Splatting training pipeline[/bold cyan]")
    checkpoint_dir = output_path / 'checkpoints'
    checkpoint_dir.mkdir(exist_ok=True)
    
    # Initialize TensorBoard logger
    tb_logger = GaussianSplattingLogger(
        log_dir=str(output_path / 'tensorboard'),
        enabled=enable_tensorboard
    )
    
    if enable_tensorboard and VERBOSITY >= 1:
        logger.info(f"[green]📊 TensorBoard:[/green] Logging to run [cyan]{tb_logger.run_name}[/cyan]")
        logger.info(f"[dim]Run: tensorboard --logdir={output_path / 'tensorboard'}[/dim]")
    
    # Setup
    dataset = ColmapDataset(colmap_path, images_path, 'cuda')  # Pass as string
    model = GaussianModel(
        dataset.init_points, 
        dataset.init_colors,
        sh_degree=sh_degree, 
        console=console
    ).to(device)
    scene_extent = dataset.scene_extent
    
    # Initialize gsplat's DefaultStrategy for densification and pruning
    from gsplat import DefaultStrategy
    strategy = DefaultStrategy(
        prune_opa=0.005,
        grow_grad2d=grad_threshold,
        
        # Threshold for "Clone vs Split" decision
        # 0.01 is standard. 0.05 was too high (protected blobs).
        grow_scale3d=0.01,
        
        # Threshold for "Screen Space Split"
        # 0.03 = 3% of screen. Small enough for detail, big enough to prevent explosion.
        grow_scale2d=0.03,
        
        # Threshold for "Big 3D Pruning" (The Blob Killer)
        # 0.01 = 1% of scene. Good for killing sky blobs.
        # 0.001 was too harsh (killed walls).
        prune_scale3d=0.01,
        
        # Threshold for "Big 2D Pruning"
        # 0.10 = 10% of screen. If it covers 10% of the view, kill it.
        prune_scale2d=0.10,
        
        # Enable the 2D checks
        refine_scale2d_stop_iter=densify_until_iter, 
        
        # Standard mappings
        refine_start_iter=densify_from_iter,
        refine_stop_iter=densify_until_iter,
        reset_every=opacity_reset_interval,
        refine_every=densify_interval,
        pause_refine_after_reset=0,
        absgrad=False,
        revised_opacity=False,
        verbose=(verbosity >= 2),
    )
    
    # Initialize strategy state
    strategy_state = strategy.initialize_state(scene_scale=scene_extent)
    
    # Intelligent auto-adjustment of max_screen_size based on scene extent
    # For large scenes (extent > 100), use larger max_screen_size
    original_max_screen_size = max_screen_size
    if max_screen_size == 20:  # Only auto-adjust if using default
        if scene_extent > 300:
            max_screen_size = 200
            auto_adjusted = True
        elif scene_extent > 150:
            max_screen_size = 100
            auto_adjusted = True
        elif scene_extent > 75:
            max_screen_size = 50
            auto_adjusted = True
        else:
            auto_adjusted = False
    else:
        auto_adjusted = False
    
    # Display initialization info
    if VERBOSITY >= 1:
        init_info = Table.grid(padding=(0, 2))
        init_info.add_column(style="cyan", justify="right")
        init_info.add_column(style="green")
        init_info.add_row("Initial Gaussians:", f"{len(model._means):,}")
        init_info.add_row("Scene Extent:", f"{scene_extent:.3f}")
        init_info.add_row("Training Images:", f"{len(dataset)}")
        init_info.add_row("Total Iterations:", f"{iterations:,}")
        init_info.add_row("Device:", str(device))
        init_info.add_row("SH Degree:", f"{sh_degree} ({'Enabled' if use_sh_rendering else 'DC only'})")
        
        # Show auto-adjustment info
        if auto_adjusted:
            init_info.add_row(
                "[blue] Auto-adjusted:[/blue]",
                f"[blue]max_screen_size {original_max_screen_size} → {max_screen_size} (for large scene)[/blue]"
            )
        
        # Add warning for large scenes with manual small max_screen_size
        if scene_extent > 200 and max_screen_size < 50 and not auto_adjusted:
            init_info.add_row(
                "[yellow]⚠ Warning:[/yellow]",
                f"[yellow]Large scene + small max_screen_size ({max_screen_size}) may cause aggressive pruning[/yellow]"
            )
        
        panel = Panel(
            init_info,
            title="[bold blue]🚀 Model Initialized[/bold blue]",
            border_style="blue",
            box=box.ROUNDED
        )
        console.print(panel)
        console.print()
    
    # Log hyperparameters to TensorBoard
    if enable_tensorboard:
        hparams = {
            'iterations': iterations,
            'lr_means': lr_means,
            'lr_scales': lr_scales,
            'lr_quats': lr_quats,
            'lr_opacities': lr_opacities,
            'lr_sh': lr_sh,
            'densify_from_iter': densify_from_iter,
            'densify_until_iter': densify_until_iter,
            'densify_interval': densify_interval,
            'grad_threshold': grad_threshold,
            'max_screen_size': max_screen_size,
            'opacity_reset_interval': opacity_reset_interval,
            'opacity_reset_value': opacity_reset_value,
            'sh_degree': sh_degree,
            'use_sh_rendering': use_sh_rendering,
            'scene_extent': scene_extent,
            'enable_scale_reg': enable_scale_reg,
            'scale_reg_weight': scale_reg_weight if enable_scale_reg else 0.0,
        }
        tb_logger.log_hyperparameters(hparams)
    
    # Initialize viewer if requested
    viewer = None
    server = None
    if enable_viewer:
        if not VIEWER_AVAILABLE:
            logger.warning("[yellow]⚠ Warning:[/yellow] nerfview not available. Install with: pip install nerfview")
            logger.info("[yellow]Continuing training without viewer...[/yellow]")
        else:
            # Create render function for viewer
            render_fn = create_viewer_render_fn(model, device, model.sh_degree)
            
            # Initialize viewer
            server = viser.ViserServer(port=viewer_port, verbose=False)
            viewer = nerfview.Viewer(
                server=server,
                render_fn=render_fn,
                mode="training",
            )
            if VERBOSITY >= 1:
                logger.info(f"[green]📺 Viewer started:[/green] http://localhost:{viewer_port}")
    
    # SSIM Loss (Standard Library)
    # ssim = StructuralSimilarityIndexMeasure(data_range=(0, 1.0)).to(device)
    ssim = MultiScaleStructuralSimilarityIndexMeasure(
                # data_range=(0, 1.0), 
                betas=(0.2, 0.2, 0.2, 0.2, 0.2),
                # compute_on_cpu=True
            ).to(device)
    depth_ssim = StructuralSimilarityIndexMeasure(
                    compute_on_cpu=True
                )

    lpips = LearnedPerceptualImagePatchSimilarity(net_type='squeeze', normalize=True).to(device)

    psnr = PeakSignalNoiseRatio(data_range=(0, 1.0)).to(device)
    
    # Optimizer
    # Create optimizers using model's factory method
    optimizers = model.create_optimizers(
        lr_means=lr_means,
        lr_scales=lr_scales,
        lr_quats=lr_quats,
        lr_opacities=lr_opacities,
        lr_sh=lr_sh,
        means_lr_multiplier=5.0,
    )
    
    # NOTE: Learning rate schedulers are currently disabled but can be re-enabled for future experiments
    # To enable:
    # 1. Add imports: from torch.optim.lr_scheduler import CosineAnnealingLR
    #                 from gs_types import GS_LR_Schedulers
    # 2. Uncomment the following:
    # schedulers = GS_LR_Schedulers.create_schedulers(
    #     optimizers,
    #     step_size=iterations,  # T_max: full cosine period
    #     gamma=0.1,             # eta_min can be adjusted
    # )
    # 3. Uncomment scheduler.step() in the training loop (search for "Learning rate schedulers")
    
    if VERBOSITY >= 2:
        logger.debug("[cyan]📈 Optimizers:[/cyan] Adam optimizers initialized for all parameters")

    ITERATIONS = iterations
    DENSIFY_FROM_ITER = densify_from_iter
    DENSIFY_UNTIL_ITER = densify_until_iter
    DENSIFY_INTERVAL = densify_interval
    OPACITY_RESET_INTERVAL = opacity_reset_interval

    OPACITY_LAMBDA = opacity_reg_weight
    SCALE_LAMBDA = scale_reg_weight

    # Setup progress bar
    if VERBOSITY >= 1:
        progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=console,
            expand=False
        )
        progress.start()
        task = progress.add_task("[cyan]Training...", total=ITERATIONS)
    
    # Training metrics for display
    current_loss = 0.0
    current_l1 = 0.0
    current_ssim = 0.0
    iteration_start_time = time.time()
    
    # Create DataLoader for efficient data loading
    # Using pin_memory for faster GPU transfer, num_workers=2 for parallel loading
    # batch_size=1 to handle varying image sizes and maintain numerical stability
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,  # Shuffle for better training coverage
        num_workers=0,  # Parallel data loading (adjust based on CPU cores)
        pin_memory=True,  # Pin memory for faster CUDA transfer
        collate_fn=dataset.collate_fn,  # Custom collate to handle camera dict + image
        # persistent_workers=True,  # Keep workers alive between epochs
        # prefetch_factor=2,  # Prefetch 2 batches per worker
    )
    
    # Create iterator that cycles through the dataloader indefinitely
    dataloader_iter = iter(dataloader)
    
    for i in range(ITERATIONS):
        # Handle viewer pause state
        if viewer is not None:
            while viewer.state == "paused":
                time.sleep(0.01)
            viewer.lock.acquire()
        
        # Track training timing for viewer
        step_start_time = time.perf_counter()
        
        # 1. Get Batch - using DataLoader for efficient loading
        try:
            cam, gt_image, depth_tensor = next(dataloader_iter)
        except StopIteration:
            # Reset iterator when epoch ends (seamless cycling)
            dataloader_iter = iter(dataloader)
            cam, gt_image, depth_tensor = next(dataloader_iter)
        
        # Move to GPU with non_blocking for async transfer (data already pinned)
        # Note: Camera dict values are already on CUDA from dataset, gt_image needs transfer
        if not gt_image.is_cuda:
            gt_image = gt_image.to(device, non_blocking=True)
        
        # Move depth to GPU if available
        if depth_tensor is not None and not depth_tensor.is_cuda:
            depth_tensor = depth_tensor.to(device, non_blocking=True)

        # 2. Setup Camera Matrices
        viewmat = torch.eye(4, device=device, dtype=torch.float32)
        viewmat[:3, :3] = cam['R']
        viewmat[:3, 3] = cam['T']
        viewmat = viewmat.unsqueeze(0)  # [1, 4, 4] for batch dimension
        
        # Camera intrinsics matrix K
        K = torch.tensor(
            [[cam['fx'], 0, cam['cx']], 
             [0, cam['fy'], cam['cy']], 
             [0., 0., 1.]], 
            dtype=torch.float32,
            device=device
        )
        
        # 3. Rasterize (v1.0.0 API - single call)
        # gsplat now handles projection, sorting, and blending in one call
        # SH coefficients are passed directly, no need for manual conversion
        # Conditional SH rendering: if use_sh_rendering is False, use DC only (sh_degree=None)
        render_output, render_alpha, meta = rasterization(
            means=model.means,  # [N, 3]
            quats=model.quats,  # [N, 4]
            scales=model.scales,  # [N, 3]
            opacities=model.opacities.squeeze(-1),  # [N]
            colors=model.sh if use_sh_rendering else model._features_dc.squeeze(1),  # [N, K, 3] or [N, 3]
            viewmats=viewmat,  # [1, 4, 4]
            Ks=K[None, ...],  # [1, 3, 3]
            width=cam['width'],
            height=cam['height'],
            sh_degree=model.sh_degree if use_sh_rendering else None,
            render_mode="RGB+ED",
            # absgrad=True,
            # sparse_grad=False
        )

        
        # Extract results from batch dimension
        render = render_output[0, :, :, 0:3]  # [H, W, 3]
        alpha = render_alpha[0]  # [H, W, 1] or [H, W]
        
        # Extract expected depth (ED) from render_output channel 3
        # This gives depth-weighted by opacity: sum(depth * alpha) / sum(alpha)
        render_depth_raw = render_output[0, :, :, 3]  # [H, W]
        
        # Normalize by alpha to get expected depth per pixel
        # Add epsilon to avoid division by zero
        if alpha.dim() == 3:
            alpha_2d = alpha[:, :, 0]
        else:
            alpha_2d = alpha
        render_depth_map = render_depth_raw / (alpha_2d + 1e-6)
        
        # Replace any NaN or Inf values with zeros
        render_depth_map = torch.where(
            torch.isfinite(render_depth_map), 
            render_depth_map, 
            torch.zeros_like(render_depth_map)
        )
        
        # Create mask for valid pixels (sufficient opacity and finite depth)
        depth_mask = (alpha_2d > 0.5) & torch.isfinite(render_depth_map) & (render_depth_map > 0)

        # Get means2d and radii for densification tracking
        # Safety check: if no Gaussians, we can't proceed
        if len(model._means) == 0:
            logger.error("[bold red]❌ Error:[/bold red] No Gaussians remaining in model! Training cannot continue.")
            if VERBOSITY >= 1:
                progress.stop()
            raise RuntimeError(
                f"Model has no Gaussians at iteration {i}. Training cannot continue. "
                f"This indicates overly aggressive pruning."
            )
        

        # Get params and optimizers for strategy (step_pre_backward)
        params = model.get_params_dict()
        optimizers_dict = model.get_optimizers_dict(optimizers)

        # Retain gradients as step_pre_backward needs to access them for densification
        strategy.step_pre_backward(
            params, optimizers_dict, strategy_state, i, meta
        )

        # 4. Loss Calculation
        # Combine L1 and SSIM
        # Rearrange for SSIM: [H, W, C] -> [1, C, H, W]
        render_perm = render.permute(2, 0, 1).unsqueeze(0)
        gt_perm = gt_image.permute(2, 0, 1).unsqueeze(0)

        l1_loss = (render - gt_image).abs().mean()
        ssim_loss = 1.0 - ssim(render_perm, gt_perm)
        # lpips_loss = lpips(render_perm, gt_perm)
        lpips_loss = torch.tensor(0.0, device=device)
        psnr_value = psnr(render_perm, gt_perm)
        
        # Depth loss (conditional) - using SSIM on metric-aligned depth
        if enable_depth_loss and depth_tensor is not None and i >= depth_loss_start_iter:
            # Validate depth tensor before computing loss
            if torch.isfinite(depth_tensor).all() and depth_tensor.numel() > 0:
                try:
                    # Convert relative depth to metric depth using scale-shift alignment
                    metric_depth = losses.convert_relative_to_metric_depth(render_depth_map, depth_tensor, depth_mask)
                    
                    # Normalize depths to [0, 1] range for SSIM computation
                    # Use valid (masked) regions to determine min/max for normalization
                    if depth_mask.sum() > 100:  # Ensure sufficient valid pixels
                        valid_render = render_depth_map[depth_mask]
                        valid_metric = metric_depth[depth_mask]
                        
                        # Normalize both depth maps to [0, 1] using their valid ranges
                        render_min, render_max = valid_render.min(), valid_render.max()
                        metric_min, metric_max = valid_metric.min(), valid_metric.max()
                        
                        # Avoid division by zero
                        render_range = render_max - render_min
                        metric_range = metric_max - metric_min
                        
                        if render_range > 1e-6 and metric_range > 1e-6:
                            render_norm = (render_depth_map - render_min) / render_range
                            metric_norm = (metric_depth - metric_min) / metric_range
                            
                            # Clamp to [0, 1] to handle edge cases
                            render_norm = torch.clamp(render_norm, 0, 1)
                            metric_norm = torch.clamp(metric_norm, 0, 1)
                            
                            # Reshape for SSIM: [H, W] -> [1, 1, H, W]
                            render_depth_ssim = render_norm.unsqueeze(0).unsqueeze(0)
                            metric_depth_ssim = metric_norm.unsqueeze(0).unsqueeze(0)
                            
                            # Compute SSIM-based depth loss (1 - SSIM)
                            # Higher SSIM = better match, so we minimize (1 - SSIM)
                            depth_ssim_value = depth_ssim(render_depth_ssim, metric_depth_ssim)
                            d_loss = 1.0 - depth_ssim_value
                            
                            # Log depth statistics periodically
                            if i % (log_interval * 10) == 0 and VERBOSITY >= 2:
                                logger.debug(
                                    f"[cyan]Depth Stats:[/cyan] "
                                    f"Rendered: [{render_min:.3f}, {render_max:.3f}], "
                                    f"Metric Prior: [{metric_min:.3f}, {metric_max:.3f}], "
                                    f"Valid pixels: {depth_mask.sum()}, "
                                    f"SSIM: {depth_ssim_value.item():.4f}, Loss: {d_loss.item():.6f}"
                                )
                        else:
                            # Insufficient depth range
                            d_loss = torch.tensor(0.0, device=device)
                            if i % (log_interval * 10) == 0 and VERBOSITY >= 2:
                                logger.debug("[yellow]⚠ Skipping depth loss: insufficient depth range[/yellow]")
                    else:
                        # Insufficient valid pixels
                        d_loss = torch.tensor(0.0, device=device)
                        if i % (log_interval * 10) == 0 and VERBOSITY >= 2:
                            logger.debug(f"[yellow]⚠ Skipping depth loss: only {depth_mask.sum()} valid pixels[/yellow]")
                            
                except RuntimeError as e:
                    # Depth loss computation failed - log detailed error but continue training
                    d_loss = torch.tensor(0.0, device=device)
                    metric_depth = torch.zeros_like(render_depth_map)  # Placeholder
                    if i % (log_interval * 10) == 0 or i < 100:  # Log frequently at start
                        logger.warning(
                            f"[yellow]⚠ Depth loss failed at iteration {i}:[/yellow] {str(e)}\n"
                            f"[dim]Continuing without depth supervision for this iteration.[/dim]"
                        )
                        if VERBOSITY >= 3:
                            logger.exception(e)  # Log stack trace only in DEBUG mode
            else:
                d_loss = torch.tensor(0.0, device=device)
                metric_depth = torch.zeros_like(render_depth_map)  # Placeholder
                if i % (log_interval * 10) == 0 and VERBOSITY >= 2:
                    logger.debug("[yellow]⚠ Skipping depth loss: invalid depth tensor (NaN/Inf)[/yellow]")
        else:
            d_loss = torch.tensor(0.0, device=device)
            metric_depth = torch.zeros_like(render_depth_map)  # Placeholder for logging consistency
        
        # Combine losses
        loss = 0.4 * l1_loss + 0.6 * ssim_loss
        if enable_depth_loss and depth_tensor is not None and i >= depth_loss_start_iter:
            loss = loss + depth_loss_weight * d_loss
        
        # Add scale regularization if enabled (floater prevention technique #2)
        if enable_scale_reg:
            scale_reg_loss = losses.scale_regularization(
                model.scales, 
                weight=SCALE_LAMBDA, 
                scene_extent=scene_extent
            )
            loss = loss + scale_reg_loss
        else:
            scale_reg_loss = torch.tensor(0.0, device=device)

        if enable_opacity_reg:
            opacity_reg_loss = losses.opacity_regularization(
                model.opacities, 
                weight=OPACITY_LAMBDA
            )
            loss = loss + opacity_reg_loss
        else:
            opacity_reg_loss = torch.tensor(0.0, device=device)
        
        # Update current metrics
        current_loss = loss.item()
        current_l1 = l1_loss.item()
        current_ssim = ssim_loss.item()
        current_lpips = lpips_loss.item()
        current_psnr = psnr_value.item()
        current_scale_reg = scale_reg_loss.item()
        current_opacity_reg = opacity_reg_loss.item()
        current_depth_loss = d_loss.item()
        
        # Log metrics to TensorBoard
        if enable_tensorboard and i % log_interval == 0:
            # Prepare losses dict
            losses_dict = {
                'total_loss': current_loss,
                'l1_loss': current_l1,
                'ssim_loss': current_ssim,
                'lpips_loss': current_lpips,
                'scale_reg_loss': current_scale_reg,
                'opacity_reg_loss': current_opacity_reg,
            }
            if enable_depth_loss and i >= depth_loss_start_iter:
                losses_dict['depth_loss'] = current_depth_loss
            
            tb_logger.log_losses(**losses_dict, step=i)
            tb_logger.log_quality_metrics(
                psnr=current_psnr,
                ssim_loss=current_ssim,
                lpips=current_lpips,
                step=i
            )
            tb_logger.log_model_stats(
                num_gaussians=len(model._means),
                max_radii=model.max_radii2D,
                step=i
            )
            
            # Log system metrics (CPU, RAM, GPU usage, power, etc.)
            tb_logger.log_system_metrics(step=i)
        
        # Log images periodically
        if enable_tensorboard and i % tensorboard_image_interval == 0:
            tb_logger.log_images(
                rendered=render.detach(),
                ground_truth=gt_image,
                alpha=alpha.detach(),
                rendered_depth=render_depth_map.detach(),
                gt_depth=metric_depth.detach(),
                step=i
            )
        
        # console.print(f"Info Radii {meta['radii'].shape} Means2D {meta['means2d'].shape}")
        loss.backward()

        # Densify and prune using strategy
        gaussians_before = len(model._means)
        
        # Get params and optimizers for strategy (step_post_backward)
        params = model.get_params_dict()
        optimizers_dict = model.get_optimizers_dict(optimizers)
        
        # Call strategy's step_post_backward to handle densification/pruning
        strategy.step_post_backward(
            params=params,
            optimizers=optimizers_dict,
            state=strategy_state,
            step=i,
            info=meta,
            packed=True
        )
        
        # Update model parameters from strategy (they may have changed due to split/duplicate/prune)
        model.update_params_from_dict(params)
        
        gaussians_after = len(model._means)
        
        # Clear CUDA cache after densification/pruning
        torch.cuda.empty_cache()

        
        # 6. Densification Step (The Gaussian Splatting Magic)
        if i >= DENSIFY_FROM_ITER and i <= DENSIFY_UNTIL_ITER:
            if i % DENSIFY_INTERVAL == 0:
                
                # Log densification event to TensorBoard
                if enable_tensorboard:
                    tb_logger.log_densification_event(gaussians_before, gaussians_after, step=i)
                
                # Safety check: ensure we don't remove all Gaussians
                if gaussians_after == 0:
                    logger.error("[bold red]⚠ Critical:[/bold red] All Gaussians were pruned! Pruning is too aggressive.")
                    if VERBOSITY >= 1:
                        progress.stop()
                    raise RuntimeError(
                        f"All Gaussians were removed during densification at iteration {i}. "
                        f"Try adjusting pruning parameters: increase --max-screen-size (current: {max_screen_size}) "
                        f"or modify --grad-threshold (current: {grad_threshold})."
                    )
                                
                # Log densification event
                if VERBOSITY >= 2:
                    change = gaussians_after - gaussians_before
                    logger.debug(
                        f"[yellow]⚡ Densification:[/yellow] [dim]{gaussians_before:,} → {gaussians_after:,} ({change:+,})[/dim]"
                    )
                elif VERBOSITY >= 1:
                    if gaussians_after != gaussians_before:
                        logger.info(f"[dim]Gaussians: {gaussians_after:,}[/dim]")
            
            # Periodically reset opacity to prevent floaters (technique #1 - already implemented)
            if i > 0 and i % OPACITY_RESET_INTERVAL == 0:
                model._opacities.data = torch.clamp(
                    model._opacities.data,
                    max=inverse_sigmoid(torch.tensor(opacity_reset_value, device=device))
                )
                if VERBOSITY >= 2:
                    logger.debug(f"[magenta]🔄 Opacity reset:[/magenta] [dim]max {opacity_reset_value}[/dim]")
        
        # Log histograms periodically
        if enable_tensorboard and i % tensorboard_histogram_interval == 0 and i > 0:
            tb_logger.log_gaussian_histograms(model, step=i)

        # Optimizer step
        for opt in optimizers.__dict__.values():
            opt.step()
            opt.zero_grad()
        
        # NOTE: Learning rate schedulers currently disabled (see initialization section above)
        # To enable, uncomment:
        # for sched in schedulers.__dict__.values():
        #     sched.step()
        
        # Log learning rates
        if enable_tensorboard and i % log_interval == 0:
            tb_logger.log_learning_rates(optimizers, step=i)
        
        # Update viewer
        if viewer is not None:
            viewer.lock.release()
            step_time = time.perf_counter() - step_start_time
            num_train_rays_per_step = cam['width'] * cam['height']
            num_train_steps_per_sec = 1.0 / step_time if step_time > 0 else 0
            num_train_rays_per_sec = num_train_rays_per_step * num_train_steps_per_sec
            # Update the viewer state
            viewer.render_tab_state.num_train_rays_per_sec = num_train_rays_per_sec
            # Update the scene
            viewer.update(i, num_train_rays_per_step)
        
        # Update progress bar with loss values
        if VERBOSITY >= 1:
            num_gaussians = len(model._means)
            phase = "Densification" if DENSIFY_FROM_ITER <= i <= DENSIFY_UNTIL_ITER else "Refinement"
            desc = format_phase_description(phase, current_loss, current_l1, current_ssim, current_lpips, current_psnr, current_scale_reg, num_gaussians)
            progress.update(task, advance=1, description=desc)

        
        # Save checkpoints
        if i % save_interval == 0 and i > 0:
            checkpoint_path = checkpoint_dir / f'checkpoint_{i}.pt'
            torch.save({
                'iteration': i,
                'model_state_dict': model.state_dict(),
                'optimizers_state_dict': {key: opt.state_dict() for key, opt in optimizers.__dict__.items()},
                'loss': loss.item(),
            }, checkpoint_path)
            if VERBOSITY >= 1:
                logger.info(f"[green]💾 Checkpoint saved:[/green] [dim]{checkpoint_path.name}[/dim]")
    
    # Complete viewer training mode
    if viewer is not None:
        viewer.complete()
        if VERBOSITY >= 1:
            logger.info("[green]✓ Training complete.[/green] Viewer remains active for inspection.")
    
    # Stop progress bar
    if VERBOSITY >= 1:
        progress.stop()
    
    # Log final metrics to TensorBoard
    if enable_tensorboard:
        final_metrics = {
            'final/loss': current_loss,
            'final/psnr': current_psnr,
            'final/gaussians': len(model._means),
        }
        tb_logger.log_hyperparameters({}, final_metrics)
        tb_logger.close()
        if VERBOSITY >= 1:
            logger.info(f"[green]✓ TensorBoard logs saved:[/green] [cyan]{tb_logger.run_name}[/cyan]")
    
    # Save final model
    final_path = output_path / 'model_final.pt'
    torch.save({
        'iteration': ITERATIONS,
        'model_state_dict': model.state_dict(),
        'scene_extent': scene_extent,
        'num_gaussians': len(model._means),
    }, final_path)
    
    # Display completion summary
    if VERBOSITY >= 1:
        console.print()
        summary_grid = Table.grid(padding=(0, 2))
        summary_grid.add_column(style="cyan", justify="right")
        summary_grid.add_column(style="green")
        summary_grid.add_row("Final Loss:", f"{current_loss:.6f}")
        summary_grid.add_row("Final L1:", f"{current_l1:.6f}")
        summary_grid.add_row("Final SSIM:", f"{current_ssim:.6f}")
        summary_grid.add_row("Final Gaussians:", f"{len(model._means):,}")
        summary_grid.add_row("Model Saved:", str(final_path.name))
        
        summary_panel = Panel(
            summary_grid,
            title="[bold green]✅ Training Complete![/bold green]",
            border_style="green",
            box=box.ROUNDED
        )
        console.print(summary_panel)
    
    # Keep viewer alive if enabled
    if viewer is not None and VERBOSITY >= 1:
        logger.info("[blue]ℹ Viewer is still running.[/blue] Press Ctrl+C to exit.")
    
    return model, viewer

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train a 3D Gaussian Splatting model from COLMAP reconstruction',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument(
        '--colmap-path',
        type=str,
        required=True,
        help='Path to COLMAP sparse reconstruction directory (e.g., sparse/0)'
    )
    parser.add_argument(
        '--images-path',
        type=str,
        required=True,
        help='Path to training images directory'
    )
    
    # Output options
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./output',
        help='Output directory for checkpoints and results'
    )
    
    # Training parameters
    parser.add_argument(
        '--iterations',
        type=int,
        default=7000,
        help='Total number of training iterations'
    )
    parser.add_argument(
        '--save-interval',
        type=int,
        default=1000,
        help='Save checkpoint every N iterations'
    )
    parser.add_argument(
        '--log-interval',
        type=int,
        default=1,
        help='Log progress every N iterations'
    )
    
    # Densification parameters
    parser.add_argument(
        '--densify-from-iter',
        type=int,
        default=500,
        help='Start densification at this iteration'
    )
    parser.add_argument(
        '--densify-until-iter',
        type=int,
        default=15000,
        help='Stop densification at this iteration'
    )
    parser.add_argument(
        '--densify-interval',
        type=int,
        default=100,
        help='Densify every N iterations'
    )
    parser.add_argument(
        '--grad-threshold',
        type=float,
        default=0.0002,
        help='Gradient threshold for densification'
    )
    parser.add_argument(
        '--max-screen-size',
        type=int,
        default=5000,
        help='Maximum screen size in pixels for pruning (increase for large scenes, e.g., 100-200)'
    )
    parser.add_argument(
        '--opacity-reset-interval',
        type=int,
        default=3000,
        help='Reset opacity every N iterations'
    )
    parser.add_argument(
        '--opacity-reset-value',
        type=float,
        default=0.01,
        help='Opacity value to reset to (0.01-0.1, lower = more aggressive floater removal)'
    )
    
    # Floater Prevention Options
    parser.add_argument(
        '--enable-scale-reg',
        action='store_true',
        help='Enable scale regularization loss to penalize large Gaussians'
    )
    parser.add_argument(
        '--scale-reg-weight',
        type=float,
        default=0.01,
        help='Weight for scale regularization loss (0.01-0.1)'
    )
    parser.add_argument(
        '--enable-opacity-reg',
        action='store_true',
        help='Enable opacity regularization loss to encourage sparsity and prevent floaters'
    )
    parser.add_argument(
        "--opacity-reg-weight",
        type=float,
        default=0.0005,
        help="Weight for opacity regularization loss (0.0001-0.001)"
    )
    
    # Spherical Harmonics options
    parser.add_argument(
        '--sh-degree',
        type=int,
        default=3,
        choices=[0, 1, 2, 3],
        help='Degree of spherical harmonics (0=DC only, 1-3=higher order). Default: 3'
    )
    parser.add_argument(
        '--disable-sh-rendering',
        action='store_true',
        help='Disable spherical harmonics during rendering (use DC component only for faster training)'
    )
    
    # Depth supervision options
    parser.add_argument(
        '--enable-depth-loss',
        action='store_true',
        help='Enable depth supervision from Depth Anything V2 depth maps'
    )
    parser.add_argument(
        '--depth-loss-weight',
        type=float,
        default=0.1,
        help='Weight for depth loss (0.05-0.2 recommended)'
    )
    parser.add_argument(
        '--depth-loss-start-iter',
        type=int,
        default=1000,
        help='Start applying depth loss after this many iterations'
    )
    
    # Learning rates
    parser.add_argument(
        '--lr-means',
        type=float,
        default=0.00016,
        help='Base learning rate for Gaussian positions (multiplied by 5.0)'
    )
    parser.add_argument(
        '--lr-scales',
        type=float,
        default=0.005,
        help='Learning rate for Gaussian scales'
    )
    parser.add_argument(
        '--lr-quats',
        type=float,
        default=0.001,
        help='Learning rate for Gaussian rotations'
    )
    parser.add_argument(
        '--lr-opacities',
        type=float,
        default=0.05,
        help='Learning rate for Gaussian opacities'
    )
    parser.add_argument(
        '--lr-sh',
        type=float,
        default=0.0025,
        help='Learning rate for spherical harmonics'
    )
    
    # Export options
    parser.add_argument(
        '--export-ply',
        action='store_true',
        help='Export final model to PLY format'
    )
    
    # Verbosity options
    parser.add_argument(
        '--verbosity',
        type=int,
        default=1,
        choices=[0, 1, 2, 3],
        help='Verbosity level: 0=QUIET, 1=NORMAL, 2=VERBOSE, 3=DEBUG'
    )
    
    # Viewer options
    parser.add_argument(
        '--viewer',
        action='store_true',
        help='Enable interactive 3D viewer during training'
    )
    parser.add_argument(
        '--viewer-port',
        type=int,
        default=8080,
        help='Port for the viewer server'
    )
    
    # TensorBoard options
    parser.add_argument(
        '--tensorboard',
        action='store_true',
        default=True,
        help='Enable TensorBoard logging (default: enabled)'
    )
    parser.add_argument(
        '--no-tensorboard',
        action='store_false',
        dest='tensorboard',
        help='Disable TensorBoard logging'
    )
    parser.add_argument(
        '--tb-image-interval',
        type=int,
        default=500,
        help='Log images to TensorBoard every N iterations'
    )
    parser.add_argument(
        '--tb-histogram-interval',
        type=int,
        default=1000,
        help='Log histograms to TensorBoard every N iterations'
    )
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    console = Console()
    
    # Display configuration
    if args.verbosity >= 1:
        console.print()
        console.rule("[bold cyan]3D Gaussian Splatting Training[/bold cyan]", style="cyan")
        console.print()
        
        config_table = Table(title="Configuration", box=box.ROUNDED, show_header=False)
        config_table.add_column("Parameter", style="cyan", width=20)
        config_table.add_column("Value", style="yellow")
        
        config_table.add_row("COLMAP Path", args.colmap_path)
        config_table.add_row("Images Path", args.images_path)
        config_table.add_row("Output Directory", args.output_dir)
        config_table.add_row("Iterations", f"{args.iterations:,}")
        config_table.add_row("Densify From", f"{args.densify_from_iter:,}")
        config_table.add_row("Densify Until", f"{args.densify_until_iter:,}")
        config_table.add_row("Verbosity", ["QUIET", "NORMAL", "VERBOSE", "DEBUG"][args.verbosity])
        
        # Add SH configuration
        sh_status = "Disabled (DC only)" if args.disable_sh_rendering else f"Degree {args.sh_degree}"
        config_table.add_row("Spherical Harmonics", sh_status)
        
        # Add depth supervision configuration
        if args.enable_depth_loss:
            depth_config = f"Enabled (weight={args.depth_loss_weight}, start_iter={args.depth_loss_start_iter})"
            config_table.add_row("Depth Supervision", depth_config)
        
        # Show enabled floater prevention techniques
        floater_techniques = []
        if args.enable_scale_reg:
            floater_techniques.append(f"Scale Reg (λ={args.scale_reg_weight})")
        if args.enable_opacity_reg:
            floater_techniques.append(f"Opacity Reg (λ={args.opacity_reg_weight})")
        if floater_techniques:
            config_table.add_row("Floater Prevention", ", ".join(floater_techniques))
        
        console.print(config_table)
        console.print()
    
    # Setup logger for main block
    logger = setup_logger(args.verbosity, Path(args.output_dir))
    
    # Train model
    result = train_pipeline(
        colmap_path=args.colmap_path,
        images_path=args.images_path,
        output_dir=args.output_dir,
        iterations=args.iterations,
        densify_from_iter=args.densify_from_iter,
        densify_until_iter=args.densify_until_iter,
        densify_interval=args.densify_interval,
        opacity_reset_interval=args.opacity_reset_interval,
        opacity_reset_value=args.opacity_reset_value,
        grad_threshold=args.grad_threshold,
        max_screen_size=args.max_screen_size,
        lr_means=args.lr_means,
        lr_scales=args.lr_scales,
        lr_quats=args.lr_quats,
        lr_opacities=args.lr_opacities,
        lr_sh=args.lr_sh,
        save_interval=args.save_interval,
        log_interval=args.log_interval,
        verbosity=args.verbosity,
        enable_viewer=args.viewer,
        viewer_port=args.viewer_port,
        # TensorBoard options
        enable_tensorboard=args.tensorboard,
        tensorboard_image_interval=args.tb_image_interval,
        tensorboard_histogram_interval=args.tb_histogram_interval,
        # Floater prevention options
        enable_scale_reg=args.enable_scale_reg,
        scale_reg_weight=args.scale_reg_weight,
        enable_opacity_reg=args.enable_opacity_reg,
        opacity_reg_weight=args.opacity_reg_weight,
        # Spherical harmonics options
        sh_degree=args.sh_degree,
        use_sh_rendering=not args.disable_sh_rendering,
        # Depth supervision options
        enable_depth_loss=args.enable_depth_loss,
        depth_loss_weight=args.depth_loss_weight,
        depth_loss_start_iter=args.depth_loss_start_iter,
    )
    
    # Handle return values (model or model+viewer)
    if isinstance(result, tuple):
        model, viewer = result
    else:
        model = result
        viewer = None
    
    # Export to PLY if requested
    if args.export_ply:
        ply_path = Path(args.output_dir) / 'final_gaussians.ply'
        model.save_ply(str(ply_path))
        if args.verbosity >= 1:
            logger.info(f"[green]📦 Exported to PLY:[/green] {ply_path}")
    
    if args.verbosity >= 1:
        console.print()
        final_info = Table.grid(padding=(0, 1))
        final_info.add_column(style="cyan")
        final_info.add_row(f"📁 Output: [yellow]{args.output_dir}[/yellow]")
        final_info.add_row(f"📊 Checkpoints: Every [yellow]{args.save_interval:,}[/yellow] iterations")
        final_info.add_row(f"🎨 Final model: [yellow]{args.output_dir}/model_final.pt[/yellow]")
        console.print(final_info)
        console.print()
    
    # Keep viewer alive if enabled
    if viewer is not None:
        if args.verbosity >= 1:
            logger.info("[blue]📺 Viewer running...[/blue] Press Ctrl+C to exit.")
        try:
            time.sleep(1000000)
        except KeyboardInterrupt:
            if args.verbosity >= 1:
                logger.info("[yellow]Viewer closed.[/yellow]")
