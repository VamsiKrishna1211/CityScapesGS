import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

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
from typing import Tuple, Optional

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
from gs_types import GSOptimizers, GS_LR_Schedulers
from utils import create_viewer_render_fn, format_phase_description, add_cameras_to_viewer
import losses
from logger import GaussianSplattingLogger
from losses import depth_loss
from torch.optim.lr_scheduler import CosineAnnealingLR
from concurrent.futures import ThreadPoolExecutor


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
    
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
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
    
    # Checkpoint options
    resume_checkpoint=None,

    # Semantic segmentation options (not implemented in this snippet, but placeholders for future extension)
    train_semantics=False,
    sem_dataset_path: Optional[Path] = None,
    sem_dim=None,
    sem_loss_weight=1.0,
    sem_start_iter=15000,
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
        
        Checkpoint Options:
        resume_checkpoint: Path to checkpoint file to resume training from
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

    # Offload heavy logging tasks to a background thread to keep GPU busy
    # max_workers=1 ensures sequential logging and prevents excessive memory usage
    executor = ThreadPoolExecutor(max_workers=1)
    
    if enable_tensorboard and VERBOSITY >= 1:
        logger.info(f"[green]📊 TensorBoard:[/green] Logging to run [cyan]{tb_logger.run_name}[/cyan]")
        logger.info(f"[dim]Run: tensorboard --logdir={output_path / 'tensorboard'}[/dim]")
    

    # Setup
    dataset = ColmapDataset(colmap_path, 
                            images_path, 
                            train_semantics, 
                            sem_dataset_path,
                            device=device
                            )
    
    # Check if resuming from checkpoint - if so, load it once and reuse
    checkpoint = None
    start_iteration = 0
    if resume_checkpoint is not None:
        checkpoint_path = Path(resume_checkpoint)
        if not checkpoint_path.exists():
            logger.error(f"[bold red]❌ Checkpoint not found:[/bold red] {checkpoint_path}")
            raise FileNotFoundError(f"Checkpoint file does not exist: {checkpoint_path}")
        
        logger.info(f"[cyan]📂 Loading checkpoint:[/cyan] {checkpoint_path.name}")
        
        # Load checkpoint once - will be reused for both model and optimizer restoration
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        
        # Create model from checkpoint state (preserves densified Gaussians)
        # We need to create a model with the right size, so we use dummy data
        # The state_dict will replace all parameters with the checkpoint values
        num_gaussians_in_ckpt = checkpoint['model_state_dict']['_means'].shape[0]
        dummy_points = torch.zeros((num_gaussians_in_ckpt, 3), device=device)
        dummy_colors = torch.zeros((num_gaussians_in_ckpt, 3), device=device)
        
        model = GaussianModel(
            dummy_points,
            dummy_colors,
            sh_degree=sh_degree,
            train_semantics=train_semantics,
            feature_dim=sem_dim,
            console=console
        ).to(device)
        
        # Remove view_count from checkpoint if present (it's just a tracking buffer)
        # This avoids size mismatch errors when resuming from checkpoints
        state_dict = checkpoint['model_state_dict']
        if 'view_count' in state_dict:
            del state_dict['view_count']
        
        # Load the checkpoint state
        model.load_state_dict(state_dict, strict=False)
        
        # Initialize view_count buffer to match the loaded model size
        model.view_count = torch.zeros(len(model._means), device=device)
        
        start_iteration = checkpoint.get('iteration', 0) + 1
        
        if VERBOSITY >= 1:
            logger.info(f"[green]✓ Loaded model with {num_gaussians_in_ckpt:,} Gaussians[/green]")
            logger.info(f"[green]  Resuming from iteration {checkpoint['iteration']:,}[/green]")
            if 'loss' in checkpoint:
                logger.info(f"[green]  Previous loss: {checkpoint['loss']:.6f}[/green]")
    else:
        # Fresh training - initialize from COLMAP point cloud
        test = torch.zeros(1, device=device)  # Warm up CUDA context to prevent first-iteration lag

        model = GaussianModel(
            dataset.init_points, 
            dataset.init_colors,
            sh_degree=sh_degree, 
            console=console
        ).to(device)
    test = torch.zeros(1, device=device)  # Warm up CUDA context to prevent first-iteration lag

    
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
    strategy_state = strategy.initialize_state(scene_scale=float(scene_extent))
    
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

            # Add all training cameras and their image thumbnails to the scene
            # Camera frustums are computed from COLMAP world-to-camera extrinsics.
            # Frustum scale is set relative to the scene extent for consistency.
            # cam_frustum_scale = max(0.005, float(scene_extent) * 0.001)
            # num_cams_added, cam_frustum_handles = add_cameras_to_viewer(
            #     server,
            #     dataset,
            #     scale=cam_frustum_scale,
            #     add_images=True,
            #     max_image_size=128,
            # )
            # if VERBOSITY >= 1:
            #     logger.info(f"[green]✓ Added {num_cams_added} camera frustums to viewer[/green]")

            # # GUI toggle for training camera visibility
            # with server.gui.add_folder("Training Cameras"):
            #     cam_toggle = server.gui.add_checkbox(
            #         "Show Cameras", initial_value=True
            #     )

            #     @cam_toggle.on_update
            #     def _toggle_cameras(event: viser.GuiUpdateEvent) -> None:
            #         for h in cam_frustum_handles:
            #             h.visible = event.target.value
    

    # SSIM Loss (Standard Library)
    # ssim = StructuralSimilarityIndexMeasure(data_range=(0, 1.0))
    ssim = MultiScaleStructuralSimilarityIndexMeasure(
        # data_range=(0, 1.0),
        # betas: weights per scale (scale 1=finest/original → scale 5=coarsest)
        # Default (0.0448, 0.2856, 0.3001, 0.2363, 0.1333) underweights fine details.
        # Increased betas[0] & betas[1] to emphasize fine-scale features (text, logos).
        betas=(0.35, 0.30, 0.20, 0.10, 0.05),
        sigma=1.0,       # Sharper gaussian kernel → better fine-detail sensitivity
        normalize='relu',
    ).to(device)

    depth_ssim = MultiScaleStructuralSimilarityIndexMeasure(
        # compute_on_cpu=True
        # betas=(0.3, 0.3, 0.3, 0.3, 0.3),
    ).to(device)
    depth_loss = losses.DepthPriorLossLeastSquares()

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
    
    # Enable learning rate schedulers for improved convergence
    # schedulers = GS_LR_Schedulers.create_schedulers(
    #     optimizers,
    #     step_size=iterations,  # T_max: full cosine period
    #     gamma=0.1,             # Not used in CosineAnnealingLR but kept for interface
    # )
    
    if VERBOSITY >= 2:
        logger.debug("[cyan]📈 Optimizers & Schedulers:[/cyan] Adam + CosineAnnealingLR initialized")

    # Restore optimizer states if resuming from checkpoint (reuse already-loaded checkpoint)
    if checkpoint is not None and 'optimizers_state_dict' in checkpoint:
        for key, opt in optimizers.__dict__.items():
            if key in checkpoint['optimizers_state_dict']:
                opt.load_state_dict(checkpoint['optimizers_state_dict'][key])
        
        if VERBOSITY >= 1:
            logger.info(f"[green]✓ Restored optimizer states[/green]")

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
        task = progress.add_task("[cyan]Training...", total=ITERATIONS, completed=start_iteration)
    
    # Training metrics for display
    current_loss = 0.0
    current_l1 = 0.0
    current_ssim = 0.0
    current_lpips = 0.0
    current_psnr = 0.0
    current_scale_reg = 0.0
    current_opacity_reg = 0.0
    current_depth_loss = 0.0
    iteration_start_time = time.time()
    
    # Create DataLoader for efficient data loading
    # Using pin_memory for faster GPU transfer, num_workers=4 for parallel loading
    # batch_size=1 to handle varying image sizes and maintain numerical stability
    import multiprocessing
    num_workers = min(0, multiprocessing.cpu_count())
    
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,  # Shuffle for better training coverage
        num_workers=num_workers,  # Parallel data loading
        # pin_memory=True,  # Pin memory for faster CUDA transfer
        collate_fn=dataset.collate_fn,  # Custom collate to handle camera dict + image
        persistent_workers=True if num_workers > 0 else False,  # Keep workers alive between epochs
        prefetch_factor=4 if num_workers > 0 else None,  # Prefetch 2 batches per worker
    )
    
    # Create iterator that cycles through the dataloader indefinitely
    dataloader_iter = iter(dataloader)
    
    for step in range(start_iteration, ITERATIONS):
        # Handle viewer pause state
        if viewer is not None:
            while viewer.state == "paused":
                time.sleep(0.01)
            viewer.lock.acquire()
        
        # Track training timing for viewer
        step_start_time = time.perf_counter()
        
        # 1. Get Batch - using DataLoader for efficient loading
        try:
            cam, gt_image, depth_tensor = next(dataloader_iter) # Depth tensor may be None & if returned, is the raw value without normalization (handled in loss function)
        except StopIteration:
            # Reset iterator when epoch ends (seamless cycling)
            dataloader_iter = iter(dataloader)
            cam, gt_image, depth_tensor = next(dataloader_iter)

        torch.cuda.synchronize()
        
        # Move to GPU with non_blocking for async transfer (data already pinned)
        # Note: Camera dict values are already on CUDA from dataset, gt_image needs transfer
        if not gt_image.is_cuda:
            gt_image = gt_image.to(device)
        

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

        if train_semantics:
            sem_output, sem_alpha, meta = rasterization(
                means=model.means,  # [N, 3]
                quats=model.quats,  # [N, 4]
                scales=model.scales,  # [N, 3]
                opacities=model.opacities.squeeze(-1),  # [N]
                colors=model._features_sem.squeeze(1),  # [N, num_classes]
                viewmats=viewmat,  # [1, 4, 4]
                Ks=K[None, ...],  # [1, 3, 3]
                width=cam['width'],
                height=cam['height'],
                sh_degree=None,
                render_mode="RGB",
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
        

        # Get means2d and radii for densification tracking
        # Safety check: if no Gaussians, we can't proceed
        if len(model._means) == 0:
            logger.error("[bold red]❌ Error:[/bold red] No Gaussians remaining in model! Training cannot continue.")
            if VERBOSITY >= 1:
                progress.stop()
            raise RuntimeError(
                f"Model has no Gaussians at iteration {step}. Training cannot continue. "
                f"This indicates overly aggressive pruning."
            )
        

        # Get params and optimizers for strategy (step_pre_backward)
        params = model.get_params_dict()
        optimizers_dict = model.get_optimizers_dict(optimizers)

        # Retain gradients as step_pre_backward needs to access them for densification
        strategy.step_pre_backward(
            params, optimizers_dict, strategy_state, step, meta
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
        
        # Depth loss (conditional)
        inv_rendered_depth, inv_prior_depth = None, None # For potential future use in visualization or advanced loss functions
        if enable_depth_loss and depth_tensor is not None and step >= depth_loss_start_iter:

            # Move depth to GPU if available
            if depth_tensor is not None and not depth_tensor.is_cuda:
                depth_tensor = depth_tensor.to(device, non_blocking=True)
            # Replace any NaN or Inf values with zeros
            render_depth_map = torch.where(
                torch.isfinite(render_depth_map), 
                render_depth_map, 
                torch.zeros_like(render_depth_map)
            )
            
            # Create mask for valid pixels (sufficient opacity and finite depth)
            depth_mask = (alpha_2d > 0.5) & torch.isfinite(render_depth_map) & (render_depth_map > 0) & (render_depth_map < scene_extent * 0.8)

            try:
                # Use the built-in depth loss function which handles alignment and L1
                # d_loss, metric_depth = losses.depth_loss(render_depth_map, depth_tensor, depth_mask)
                d_loss, inv_rendered_depth, inv_prior_depth = depth_loss(
                    render_depth_map, 
                    depth_tensor.detach(),  # Detach GT depth to prevent gradients
                    depth_mask,
                    ssim_module=depth_ssim
                )
                # logger.debug(f"[cyan]Depth Loss ({step}):[/cyan] {d_loss.item():.6f}")
                # logger.debug(f"[cyan]RenderedDepth Mask Range ({step}):[/cyan] {render_depth_map.min().item():.4f} to {render_depth_map.max().item():.4f}")
                # logger.debug(f"[cyan]GT Depth Mask Range ({step}):[/cyan] {depth_tensor.min().item():.4f} to {depth_tensor.max().item():.4f}")
                # logger.debug(f"[cyan]Metric Depth Range ({step}):[/cyan] {metric_depth.min().item():.4f} to {metric_depth.max().item():.4f}")
                
                # Periodically log depth stats without causing syncs
                # if step % (log_interval * 10) == 0 and VERBOSITY >= 3:
                #     logger.debug(f"[cyan]Depth Loss ({step}):[/cyan] {d_loss.item():.6f}")
            except RuntimeError as e:
                d_loss = torch.tensor(0.0, device=device)
                # metric_depth = torch.zeros_like(render_depth_map, device=device)
                # if step % (log_interval * 10) == 0:
                logger.debug(f"[yellow]⚠ Depth loss skipped:[/yellow] {str(e)}")
        else:
            d_loss = torch.tensor(0.0, device=device)
            # metric_depth = torch.zeros_like(render_depth_map, device=device)
        
        # Combine losses
        loss = 0.4 * l1_loss + 0.6 * ssim_loss
        if enable_depth_loss and depth_tensor is not None and step >= depth_loss_start_iter:
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
        
        
        # Update metrics only periodically for logging to avoid excessive CPU-GPU synchronization
        if step % log_interval == 0 or step == ITERATIONS - 1:
            current_loss = loss.item()
            current_l1 = l1_loss.item()
            current_ssim = ssim_loss.item()
            current_lpips = lpips_loss.item()
            current_psnr = psnr_value.item()
            current_scale_reg = scale_reg_loss.item()
            current_opacity_reg = opacity_reg_loss.item()
            current_depth_loss = d_loss.item()
            
            # Log metrics to TensorBoard
            if enable_tensorboard:
                # Prepare losses dict
                losses_dict = {
                    'total_loss': current_loss,
                    'l1_loss': current_l1,
                    'ssim_loss': current_ssim,
                    'lpips_loss': current_lpips,
                    'scale_reg_loss': current_scale_reg,
                    'opacity_reg_loss': current_opacity_reg,
                }
                if enable_depth_loss and step >= depth_loss_start_iter:
                    losses_dict['depth_loss'] = current_depth_loss
                
                tb_logger.log_losses(**losses_dict, step=step)
                tb_logger.log_quality_metrics(
                    psnr=current_psnr,
                    ssim_loss=current_ssim,
                    lpips=current_lpips,
                    step=step
                )
                tb_logger.log_model_stats(
                    num_gaussians=len(model._means),
                    max_radii=meta['radii'],
                    step=step
                )
                
                # Log system metrics (CPU, RAM, GPU usage, power, etc.)
                tb_logger.log_system_metrics(step=step)
        
        # Update progress bar every iteration for smooth visual feedback
        if VERBOSITY >= 1:
            num_gaussians = len(model._means)
            phase = "Densification" if DENSIFY_FROM_ITER <= step <= DENSIFY_UNTIL_ITER else "Refinement"
            desc = format_phase_description(step, phase, current_loss, current_l1, current_ssim, current_lpips, current_psnr, current_scale_reg, num_gaussians)
            progress.update(task, advance=1, description=desc)

        # Log images periodically
        if enable_tensorboard and step % tensorboard_image_interval == 0:
            # We offload image logging to a background thread.
            # Passing detached GPU tensors is safe as the thread will handle the CPU transfer
            # (synchronization point) without stalling the main training loop.
            executor.submit(
                tb_logger.log_images,
                rendered=render.detach(),
                ground_truth=gt_image.detach(),
                alpha=alpha.detach(),
                inv_rendered_depth=inv_rendered_depth if inv_rendered_depth is not None else None,
                inv_prior_depth=inv_prior_depth if inv_prior_depth is not None else None,
                step=step
            )
            # tb_logger.log_images(
            #     rendered=render.detach(),
            #     ground_truth=gt_image.detach(),
            #     alpha=alpha.detach(),
            #     inv_rendered_depth=inv_rendered_depth if inv_rendered_depth is not None else None,
            #     inv_prior_depth=inv_prior_depth if inv_prior_depth is not None else None,
            #     step=step
            # )
        # 5. Optimization Steps
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
            step=step,
            info=meta,
            packed=True
        )
        
        # Update model parameters from strategy (they may have changed due to split/duplicate/prune)
        model.update_params_from_dict(params)
        gaussians_after = len(model._means)
        
        # Clear CUDA cache occasionally after densification
        if step % (DENSIFY_INTERVAL * 5) == 0:
            torch.cuda.empty_cache()

        # Handle manual densification events for logging and opacity resets
        if step >= DENSIFY_FROM_ITER and step <= DENSIFY_UNTIL_ITER:
            if step % DENSIFY_INTERVAL == 0:
                if enable_tensorboard:
                    tb_logger.log_densification_event(gaussians_before, gaussians_after, step=step)
                
                if gaussians_after == 0:
                    logger.error("[bold red]⚠ Critical:[/bold red] All Gaussians were pruned!")
                    if VERBOSITY >= 1: progress.stop()
                    raise RuntimeError(f"All Gaussians removed at iteration {step}")
            
            # Periodically reset opacity to prevent floaters
            if step > 0 and step % OPACITY_RESET_INTERVAL == 0:
                model._opacities.data = torch.clamp(
                    model._opacities.data,
                    max=inverse_sigmoid(torch.tensor(opacity_reset_value, device=device))
                )

        # Log histograms periodically
        if enable_tensorboard and step > 0 and step % tensorboard_histogram_interval == 0:
            tb_logger.log_gaussian_histograms(model, step=step)

        # Optimizer step
        for opt in optimizers.__dict__.values():
            opt.step()
            opt.zero_grad(set_to_none=True)
        
        # Step learning rate schedulers
        # for sched in schedulers.__dict__.values():
        #     sched.step()
        
        # Log learning rates
        if enable_tensorboard and step % log_interval == 0:
            tb_logger.log_learning_rates(optimizers, step=step)
        
        # Update viewer
        if viewer is not None:
            viewer.lock.release()
            step_time = time.perf_counter() - step_start_time
            num_train_rays_per_step = cam['width'] * cam['height']
            num_train_steps_per_sec = 1.0 / step_time if step_time > 0 else 0
            num_train_rays_per_sec = num_train_rays_per_step * num_train_steps_per_sec
            viewer.render_tab_state.num_train_rays_per_sec = num_train_rays_per_sec
            viewer.update(step, num_train_rays_per_step)

        # Save checkpoints
        if step % save_interval == 0 and step > 0:
            checkpoint_path = checkpoint_dir / f'checkpoint_{step}.pt'
            torch.save({
                'iteration': step,
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

    # Wait for background logging tasks to complete
    if VERBOSITY >= 1:
        logger.info("[dim]⌛ Waiting for background logging tasks to finish...[/dim]")
    executor.shutdown(wait=True)
    
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
    
    # Checkpoint options
    parser.add_argument(
        '--resume-from',
        type=str,
        default=None,
        help='Path to checkpoint file to resume training from (e.g., output/checkpoints/checkpoint_1000.pt)'
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
    parser.add_argument(
        '--train-semantics',
        action='store_true',
        help='Enable training with semantic features',
        default=False
    )
    parser.add_argument(
        '--sem-dim',
        type=int,
        default=None,
        help='Number of semantic classes (only relevant if --train-semantics is enabled)'
    )
    parser.add_argument(
        '--sem-loss-weight',
        type=float,
        default=1.0,
        help='Weight for semantic loss (only relevant if --train-semantics is enabled)'
    )
    parser.add_argument(
        '--sem-start-iter',
        type=str,
        default=15000,
        help='Start applying semantic loss after this many iterations (only relevant if --train-semantics is enabled)'
    )
    parser.add_argument(
        '--sem-dataset-path',
        type=str,
        default=None,
        help='Path to semantic segmentation dataset (only relevant if --train-semantics is enabled)'
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
        # Checkpoint options
        resume_checkpoint=args.resume_from,

        # Semantic training options
        train_semantics=args.train_semantics,
        sem_dim=args.sem_dim,
        sem_loss_weight=args.sem_loss_weight,
        sem_start_iter=args.sem_start_iter,
        sem_dataset_path=args.sem_dataset_path
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
