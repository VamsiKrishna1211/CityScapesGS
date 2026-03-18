import logging
import os
import time
from pathlib import Path
from typing import Tuple, Union

import numpy as np

# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import torch
import torch.nn.functional as F
from gsplat import DefaultStrategy, rasterization
from torch.utils.data import DataLoader
from torchmetrics.image import (
    LearnedPerceptualImagePatchSimilarity,
    MultiScaleStructuralSimilarityIndexMeasure,
    PeakSignalNoiseRatio,
    StructuralSimilarityIndexMeasure,
)

try:
    import nerfview
    import viser
    VIEWER_AVAILABLE = True
except ImportError:
    VIEWER_AVAILABLE = False

from concurrent.futures import ThreadPoolExecutor

import losses
from dataset import create_dataset

# Modules
from fused_ssim import fused_ssim
from gs_types import GS_LR_Schedulers, GSOptimizers
from logger import GaussianSplattingLogger
from model import GaussianModel, inverse_sigmoid
from rich import box
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.logging import RichHandler
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table
from torch.optim.lr_scheduler import CosineAnnealingLR
from train_args import TrainConfig, parse_args
from utils import format_phase_description
from viewer_sync import ViewerParamSync

torch.backends.cudnn.enabled = True  # Enable cuDNN auto-tuner for better performance on fixed-size inputs
torch.backends.cudnn.allow_tf32 = True  # Allow TF32 for cuDNN (can speed up on Ampere GPUs)
torch.backends.cudnn.benchmark = True  # Enable cuDNN benchmark mode for optimal performance
torch.backends.cuda.matmul.allow_tf32 = True  # Allow TF32 for matrix multiplications (can speed up on Ampere GPUs)

ACTIVE_PROGRESS = None


def _log_interval_metrics_async(
    tb_logger: GaussianSplattingLogger,
    losses_snapshot: dict,
    psnr_snapshot: float,
    ssim_snapshot: float,
    lpips_snapshot: float,
    num_gaussians_snapshot: int,
    max_radii_snapshot,
    step_snapshot: int,
):
    """Log per-interval TensorBoard metrics in a background thread using immutable snapshots."""
    tb_logger.log_losses(**losses_snapshot, step=step_snapshot)
    tb_logger.log_quality_metrics(
        psnr=psnr_snapshot,
        ssim_loss=ssim_snapshot,
        lpips=lpips_snapshot,
        step=step_snapshot,
    )
    tb_logger.log_model_stats(
        num_gaussians=num_gaussians_snapshot,
        max_radii=max_radii_snapshot,
        step=step_snapshot,
    )


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


def train_pipeline(config: TrainConfig):
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

        Sharpness-aware Minimization Options:
        sam_loss_weight: Weight for sharpness-aware loss in gradient space
        
        TensorBoard Options:
        enable_tensorboard: Enable TensorBoard logging
        tensorboard_image_interval: Log images every N iterations
        tensorboard_histogram_interval: Log histograms every N iterations
        
        Checkpoint Options:
        resume_checkpoint: Path to checkpoint file to resume training from
    """
    required_cfg = config.required
    output_cfg = config.output
    training_cfg = config.training
    densification_cfg = config.densification
    floater_cfg = config.floater_prevention
    sh_cfg = config.sh
    depth_cfg = config.depth
    lr_cfg = config.learning_rates
    runtime_cfg = config.runtime
    checkpoint_cfg = config.checkpoint
    viewer_cfg = config.viewer
    tensorboard_cfg = config.tensorboard

    device = torch.device('cuda')
    
    # Initialize Rich console for UI elements (tables, panels, progress bars)
    console = Console()
    
    # Verbosity levels: 0=QUIET, 1=NORMAL, 2=VERBOSE, 3=DEBUG
    VERBOSITY = runtime_cfg.verbosity
    
    # Create output directory
    output_path = Path(output_cfg.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Setup structured logging with RichHandler
    logger = setup_logger(runtime_cfg.verbosity, output_path)
    logger.info("[bold cyan]🚀 Starting 3D Gaussian Splatting training pipeline[/bold cyan]")
    checkpoint_dir = output_path / 'checkpoints'
    checkpoint_dir.mkdir(exist_ok=True)

    # Setup
    semantics_path = (
        Path(semantics_cfg.semantics_path)
        if semantics_cfg.semantics_path is not None and not isinstance(semantics_cfg.semantics_path, Path)
        else semantics_cfg.semantics_path
    )
    raw_semantics_resolution = semantics_cfg.semantic_image_resolution
    if isinstance(raw_semantics_resolution, list):
        if len(raw_semantics_resolution) != 2:
            raise ValueError(
                "semantic_image_resolution must contain exactly two values [height, width] when provided as a list."
            )
        semantics_resolution: Union[Tuple[int, int], int, None] = (
            int(raw_semantics_resolution[0]),
            int(raw_semantics_resolution[1]),
        )
    elif isinstance(raw_semantics_resolution, tuple):
        if len(raw_semantics_resolution) != 2:
            raise ValueError(
                "semantic_image_resolution must contain exactly two values (height, width) when provided as a tuple."
            )
        semantics_resolution = (
            int(raw_semantics_resolution[0]),
            int(raw_semantics_resolution[1]),
        )
    else:
        semantics_resolution = raw_semantics_resolution

    if required_cfg.dataset_type == "colmap":
        dataset = create_dataset(
            dataset_type="colmap",
            colmap_path=required_cfg.colmap_path,
            image_dir=required_cfg.images_path,
            device=device,
            image_scale=required_cfg.scale,
            train_semantics=semantics_cfg.train_semantics,
            semantics_dim=semantics_cfg.semantics_dim,
            semantics_path=semantics_path,
            semantics_resolution=semantics_resolution,
            scene_extent_margin=1.5,  # Add margin to scene extent to prevent aggressive pruning of boundary Gaussians
            require_depth=depth_cfg.enable_depth_loss,
        )
    elif required_cfg.dataset_type == "instant-ngp":
        dataset = create_dataset(
            dataset_type="instant-ngp",
            transforms_path=required_cfg.transforms_path,
            image_dir=required_cfg.images_path,
            point_cloud_path=required_cfg.point_cloud_path,
            device=device,
            image_scale=required_cfg.scale,
            train_semantics=semantics_cfg.train_semantics,
            semantics_dim=semantics_cfg.semantics_dim,
            semantics_path=semantics_path,
            semantics_resolution=semantics_resolution,
            scene_extent_margin=1.5,
            require_depth=depth_cfg.enable_depth_loss,
        )
    else:
        raise ValueError(f"Unsupported dataset_type: {required_cfg.dataset_type}")
    # Enable preload path so image/depth disk I/O does not happen inside the training loop.
    if training_cfg.preload:
        logger.info("[yellow]Preloading enabled:[/yellow] loading all images into RAM before training...")
        dataset.preload_all_data()
    
    # Check if resuming from checkpoint - if so, load it once and reuse
    checkpoint = None
    start_iteration = 0
    tb_resume_run_name = None
    tb_purge_step = None
    if checkpoint_cfg.resume_from is not None:
        checkpoint_path = Path(checkpoint_cfg.resume_from)
        if not checkpoint_path.exists():
            logger.error(f"[bold red]❌ Checkpoint not found:[/bold red] {checkpoint_path}")
            raise FileNotFoundError(f"Checkpoint file does not exist: {checkpoint_path}")
        
        logger.info(f"[cyan]📂 Loading checkpoint:[/cyan] {checkpoint_path.name}")

        model, checkpoint = GaussianModel.resume_from_checkpoint(
            checkpoint_path=checkpoint_path,
            device=str(device),
            sh_degree=sh_cfg.sh_degree,
            console=console,
            strict=False,
        )
        model.to(device)
        
        start_iteration = checkpoint.get('iteration', 0) + 1
        tb_resume_run_name = checkpoint.get('tensorboard_run_name', None)
        if tb_resume_run_name is not None:
            tb_purge_step = start_iteration
        
        if VERBOSITY >= 1:
            logger.info(f"[green]✓ Loaded model with {len(model._means):,} Gaussians[/green]")
            logger.info(f"[green]  Resuming from iteration {checkpoint['iteration']:,}[/green]")
            if 'loss' in checkpoint:
                logger.info(f"[green]  Previous loss: {checkpoint['loss']:.6f}[/green]")
    else:
        # Fresh training - initialize from COLMAP point cloud

        model = GaussianModel(
            dataset.init_points, 
            dataset.init_colors,
            sh_degree=sh_cfg.sh_degree,
            console=console
        ).to(device)

    # Initialize TensorBoard logger (supports resume using checkpoint metadata)
    tb_logger = GaussianSplattingLogger(
        log_dir=str(output_path / 'tensorboard'),
        enabled=tensorboard_cfg.tensorboard,
        run_name=tb_resume_run_name,
        purge_step=tb_purge_step,
    )

    # Offload heavy logging tasks to a background thread to keep GPU busy
    # max_workers=1 ensures sequential logging and prevents excessive memory usage
    executor = ThreadPoolExecutor(max_workers=16)

    if tensorboard_cfg.tensorboard and VERBOSITY >= 1:
        if tb_resume_run_name is not None:
            logger.info(
                f"[green]📊 TensorBoard:[/green] Resuming run [cyan]{tb_logger.run_name}[/cyan] "
                f"from step [yellow]{start_iteration:,}[/yellow]"
            )
        else:
            logger.info(f"[green]📊 TensorBoard:[/green] Logging to run [cyan]{tb_logger.run_name}[/cyan]")
        logger.info(f"[dim]Project: {tb_logger.run_name}[/dim]")
        logger.info(f"[dim]Run: tensorboard --logdir={output_path / 'tensorboard'}[/dim]")

    
    scene_extent = dataset.scene_extent
    

    # Initialize gsplat's DefaultStrategy for densification and pruning
    strategy = DefaultStrategy(
        # prune_opa=0.005,  
        
        # # Keep gradient threshold standard, but use the dynamic scheduler we built 
        # # to choke it off later in training.
        # grow_grad2d=0.0001,  
        
        # # --- THE CLONING FIX ---
        # # Drop this aggressively. Force the model to SPLIT (push apart) 
        # # rather than CLONE (stack in the same spot).
        # grow_scale3d=0.001, 
        
        # # --- THE DETAIL RECOVERY ---
        # # Drop back to 0.01 (1% of screen). Let the model build fine, sharp details.
        # grow_scale2d=0.01,
        
        # # --- THE BLOB KILLER ---
        # # Keep this strict to kill anything that gets massive.
        # prune_scale3d=0.01,
        
        # prune_scale2d=0.20,

        prune_opa=densification_cfg.prune_opa, # default 0.005 - drop anything that contributes very little to the render (opacity < 0.5% per Gaussian)
        grow_grad2d=densification_cfg.grow_grad2d, # default 0.0001 - add new Gaussians in areas with high 2D image gradients (indicates missed details)
        grow_scale3d=densification_cfg.grow_scale3d, # default 0.02 - add new Gaussians if existing ones are too small in world space (indicates underfitting/clumping)
        grow_scale2d=densification_cfg.grow_scale2d, # default 0.01 - add new Gaussians if existing ones are too small on screen (indicates missed fine details)
        prune_scale3d=densification_cfg.prune_scale3d, # default 0.02 - remove Gaussians that become too large in world space (indicates floaters/clumps)
        prune_scale2d=densification_cfg.prune_scale2d, # default 0.20 - remove Gaussians that become too large on screen (indicates floaters/blobs)
        
        # Lock in the intervals. Do not densify at iteration 0.
        refine_start_iter=densification_cfg.densify_from_iter,
        refine_stop_iter=densification_cfg.densify_until_iter,
        refine_every=densification_cfg.densify_interval,
        reset_every=densification_cfg.opacity_reset_interval,
        
        pause_refine_after_reset=0,
        absgrad=False,
        revised_opacity=False,
        verbose=(runtime_cfg.verbosity >= 2),
    )
    
    # Initialize strategy state
    strategy_state = strategy.initialize_state(scene_scale=float(scene_extent))
    
    # Intelligent auto-adjustment of max_screen_size based on scene extent
    # For large scenes (extent > 100), use larger max_screen_size
    max_screen_size = densification_cfg.max_screen_size
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
        init_info.add_row("Total Iterations:", f"{training_cfg.iterations:,}")
        init_info.add_row("Device:", str(device))
        init_info.add_row("SH Degree:", f"{sh_cfg.sh_degree} ({'Enabled' if not sh_cfg.disable_sh_rendering else 'DC only'})")
        
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
    if tensorboard_cfg.tensorboard:
        hparams = {
            'iterations': training_cfg.iterations,
            'save_interval': training_cfg.save_interval,
            'log_interval': training_cfg.log_interval,
            'num_workers': training_cfg.num_workers,
            'lr_means': lr_cfg.lr_means,
            'lr_scales': lr_cfg.lr_scales,
            'lr_quats': lr_cfg.lr_quats,
            'lr_opacities': lr_cfg.lr_opacities,
            'lr_sh': lr_cfg.lr_sh,
            'densify_from_iter': densification_cfg.densify_from_iter,
            'densify_until_iter': densification_cfg.densify_until_iter,
            'densify_interval': densification_cfg.densify_interval,
            'grad_threshold': densification_cfg.grad_threshold,
            'prune_opa': densification_cfg.prune_opa,
            'grow_grad2d': densification_cfg.grow_grad2d,
            'grow_scale3d': densification_cfg.grow_scale3d,
            'grow_scale2d': densification_cfg.grow_scale2d,
            'prune_scale3d': densification_cfg.prune_scale3d,
            'prune_scale2d': densification_cfg.prune_scale2d,
            'max_screen_size': max_screen_size,
            'opacity_reset_interval': densification_cfg.opacity_reset_interval,
            'opacity_reset_value': densification_cfg.opacity_reset_value,
            'sh_degree': sh_cfg.sh_degree,
            'use_sh_rendering': not sh_cfg.disable_sh_rendering,
            'enable_depth_loss': depth_cfg.enable_depth_loss,
            'depth_loss_weight': depth_cfg.depth_loss_weight,
            'depth_loss_start_iter': depth_cfg.depth_loss_start_iter,
            'sam_loss_weight': depth_cfg.sam_loss_weight,
            'affine_invariant_depth_loss_weight': depth_cfg.affine_invariant_depth_loss_weight,
            'pearson_correlation_loss_weight': depth_cfg.pearson_correlation_loss_weight,
            'silog_loss_weight': depth_cfg.silog_loss_weight,
            'ordinal_depth_loss_weight': depth_cfg.ordinal_depth_loss_weight,
            'affine_aligned_gradient_matching_loss_weight': depth_cfg.affine_aligned_gradient_matching_loss_weight,
            'enable_depth_smoothness_loss': depth_cfg.enable_depth_smoothness_loss,
            'depth_smoothness_start_alpha': depth_cfg.depth_smoothness_start_alpha,
            'depth_smoothness_end_alpha': depth_cfg.depth_smoothness_end_alpha,
            'depth_smoothness_max_steps': depth_cfg.depth_smoothness_max_steps if depth_cfg.depth_smoothness_max_steps is not None else training_cfg.iterations,
            'depth_smoothness_loss_weight': depth_cfg.depth_smoothness_loss_weight,
            'scene_extent': scene_extent,
            'enable_scale_reg': floater_cfg.enable_scale_reg,
            'scale_reg_weight': floater_cfg.scale_reg_weight if floater_cfg.enable_scale_reg else 0.0,
            'enable_opacity_reg': floater_cfg.enable_opacity_reg,
            'opacity_reg_weight': floater_cfg.opacity_reg_weight if floater_cfg.enable_opacity_reg else 0.0,
            'enable_opacity_entropy_reg': floater_cfg.enable_opacity_entropy_reg,
            'opacity_entropy_reg_weight': floater_cfg.opacity_entropy_reg_weight if floater_cfg.enable_opacity_entropy_reg else 0.0,
            'viewer': viewer_cfg.viewer,
            'viewer_port': viewer_cfg.viewer_port,
            'viewer_refresh_interval': viewer_cfg.viewer_refresh_interval,
            'tensorboard_enabled': tensorboard_cfg.tensorboard,
            'tb_image_interval': tensorboard_cfg.tb_image_interval,
            'tb_histogram_interval': tensorboard_cfg.tb_histogram_interval,
            'resume_from_checkpoint': checkpoint_cfg.resume_from is not None,
        }
        tb_logger.log_hyperparameters(hparams)
    
    # Initialize viewer if requested
    viewer = None
    server = None
    viewer_param_sync = None
    if viewer_cfg.viewer:
        if not VIEWER_AVAILABLE:
            logger.warning("[yellow]⚠ Warning:[/yellow] nerfview not available. Install with: pip install nerfview")
            logger.info("[yellow]Continuing training without viewer...[/yellow]")
        else:
            viewer_param_sync = ViewerParamSync(
                model=model,
                device=device,
                disable_sh_rendering=sh_cfg.disable_sh_rendering,
                refresh_interval=viewer_cfg.viewer_refresh_interval,
            )
            
            # Initialize viewer
            server = viser.ViserServer(port=viewer_cfg.viewer_port, verbose=False)
            viewer = nerfview.Viewer(
                server=server,
                render_fn=viewer_param_sync.render_fn,
                mode="training",
            )
            if VERBOSITY >= 1:
                logger.info(f"[green]📺 Viewer started:[/green] http://localhost:{viewer_cfg.viewer_port}")

    # SSIM Loss (Standard Library)
    # Using fused SSIM implementation for better performance and stability on GPU

    # LPIPS Loss (Learned Perceptual Image Patch Similarity)
    if training_cfg.enable_lpips_loss:
        try:
            lpips = LearnedPerceptualImagePatchSimilarity(net_type='vgg', normalize=True).to(device)
            if VERBOSITY >= 1:
                logger.info("[green]✓ LPIPS loss enabled[/green]")
        except Exception as e:
            logger.error(f"[bold red]❌ Failed to initialize LPIPS loss:[/bold red] {e}")
            logger.info("[yellow]Continuing training without LPIPS loss...[/yellow]")
            lpips = None
    else:
        lpips = None
    # lpips = LearnedPerceptualImagePatchSimilarity(net_type='vgg', normalize=True).to(device)

    psnr = PeakSignalNoiseRatio(data_range=(0, 1.0)).to(device)

    # depth_smoothness_loss = losses.DynamicEdgeAwareDepthSmoothnessLoss(
    #     start_alpha=depth_cfg.depth_smoothness_start_alpha,
    #     end_alpha=depth_cfg.depth_smoothness_end_alpha,
    #     max_steps= training_cfg.iterations if depth_cfg.depth_smoothness_max_steps is None else depth_cfg.depth_smoothness_max_steps,  # Avoid scheduling if weight is 0
    # ).to(device)
    depth_smoothness_loss = losses.FastPriorGradientMatchingLoss().to(device)
    affine_invariant_depth_loss = losses.FastAffineInvariantDepthLoss().to(device)
    pearson_correlation_loss = losses.PearsonCorrelationLoss().to(device)
    silog_depth_loss = losses.SILogLoss().to(device)
    ordinal_depth_loss = losses.OrdinalDepthLoss().to(device)
    affine_aligned_gradient_matching_loss = losses.AffineAlignedGradientMatchingLoss().to(device)
    
    # Optimizer
    # Create optimizers using model's factory method
    optimizers = model.create_optimizers(
        lr_means=lr_cfg.lr_means,
        lr_scales=lr_cfg.lr_scales,
        lr_quats=lr_cfg.lr_quats,
        lr_opacities=lr_cfg.lr_opacities,
        lr_sh=lr_cfg.lr_sh,
        means_lr_multiplier=5.0,
    )
    
    # Enable learning rate schedulers for improved convergence
    schedulers = GS_LR_Schedulers.create_schedulers(
        optimizers,
        enabled_lrs=GS_LR_Schedulers(
            means=True,
        ),
        step_size=training_cfg.iterations,  # T_max: full cosine period
        gamma=0.1,             # Not used in CosineAnnealingLR but kept for interface
    )
    
    if VERBOSITY >= 2:
        logger.debug("[cyan]📈 Optimizers & Schedulers:[/cyan] Adam + CosineAnnealingLR initialized")

    # Restore optimizer states if resuming from checkpoint (reuse already-loaded checkpoint)
    if checkpoint is not None and 'optimizers_state_dict' in checkpoint:
        for key, opt in optimizers.__dict__.items():
            if opt is not None and key in checkpoint['optimizers_state_dict']:
                opt.load_state_dict(checkpoint['optimizers_state_dict'][key])
        
        if VERBOSITY >= 1:
            logger.info(f"[green]✓ Restored optimizer states[/green]")

    ITERATIONS = training_cfg.iterations
    DENSIFY_FROM_ITER = densification_cfg.densify_from_iter
    DENSIFY_UNTIL_ITER = densification_cfg.densify_until_iter
    DENSIFY_INTERVAL = densification_cfg.densify_interval
    OPACITY_RESET_INTERVAL = densification_cfg.opacity_reset_interval

    OPACITY_LAMBDA = floater_cfg.opacity_reg_weight
    OPACITY_ENTROPY_LAMBDA = floater_cfg.opacity_entropy_reg_weight
    SCALE_LAMBDA = floater_cfg.scale_reg_weight

    global ACTIVE_PROGRESS
    ACTIVE_PROGRESS = None

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
        ACTIVE_PROGRESS = progress
        task = progress.add_task("[cyan]Training...", total=training_cfg.iterations, completed=start_iteration)
    
    # Training metrics for display
    current_loss = 0.0
    current_l1 = 0.0
    current_ssim = 0.0
    current_lpips = 0.0
    current_psnr = 0.0
    current_scale_reg = 0.0
    current_opacity_reg = 0.0
    current_opacity_entropy_reg = 0.0
    current_depth_loss = 0.0
    current_depth_corr = 0.0
    current_sam_loss = 0.0
    current_affine_invariant_depth_loss = 0.0
    current_pearson_correlation_loss = 0.0
    current_silog_loss = 0.0
    current_ordinal_depth_loss = 0.0
    current_affine_aligned_gradient_matching_loss = 0.0
    smoothness_loss = torch.tensor(0.0, device=device)
    iteration_start_time = time.time()
    
    # Create DataLoader for efficient data loading
    # Using pin_memory for faster GPU transfer, num_workers=4 for parallel loading
    # batch_size=1 to handle varying image sizes and maintain numerical stability
    import multiprocessing
    max_workers = multiprocessing.cpu_count()
    num_workers = max(0, min(training_cfg.num_workers, max_workers))

    if num_workers > 0:
        if VERBOSITY >= 1:
            logger.info(
                "[yellow]Preloading enabled:[/yellow] forcing DataLoader num_workers to 0 "
                "to avoid duplicating preloaded RAM across worker processes."
            )
        num_workers = 0
    
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
    
    for step in range(start_iteration, training_cfg.iterations):
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
        if gt_image.dim() == 3:
            gt_image = gt_image.unsqueeze(0)

        if depth_tensor is not None and not depth_tensor.is_cuda:
            depth_tensor = depth_tensor.to(device)
        

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
        render_output, render_alpha, render_meta = rasterization(
            means=model.means,  # [N, 3]
            quats=model.quats,  # [N, 4]
            scales=model.scales,  # [N, 3]
            opacities=model.opacities.squeeze(-1),  # [N]
            colors=model.sh if not sh_cfg.disable_sh_rendering else model._features_dc.squeeze(1),  # [N, K, 3] or [N, 3]
            viewmats=viewmat,  # [1, 4, 4]
            Ks=K[None, ...],  # [1, 3, 3]
            width=cam['width'],
            height=cam['height'],
            sh_degree=model.sh_degree if not sh_cfg.disable_sh_rendering else None,
            render_mode="RGB+ED",
            # absgrad=True,
            # sparse_grad=False
        )

        
        # Keep batch dimension for batch-compatible losses
        render = render_output[..., 0:3]  # [B, H, W, 3]
        alpha = render_alpha  # [B, H, W, 1] or [B, H, W]
        
        # Extract expected depth (ED) from render_output channel 3
        # This gives depth-weighted by opacity: sum(depth * alpha) / sum(alpha)
        render_depth_raw = render_output[..., 3]  # [B, H, W]
        
        # Normalize by alpha to get expected depth per pixel
        # Add epsilon to avoid division by zero
        if alpha.dim() == 4:
            alpha_2d = alpha[..., 0]
        else:
            alpha_2d = alpha
        render_depth_map = render_depth_raw / (alpha_2d + 1e-6) # [B, H, W] - this is the expected depth map that we can compare against depth supervision if enabled

        # Create mask for valid pixels (sufficient opacity + finite values)
        depth_mask = (
            (alpha_2d > 0.5)
            & torch.isfinite(render_depth_map)
            & (render_depth_map > 0)
        )
        depth_mask_bchw = depth_mask.unsqueeze(0)  # [B, 1, H, W] for loss functions that expect channel dim

        render_depth_map_bchw = render_depth_map.unsqueeze(0) # [B, 1, H, W] for loss functions that expect channel dim
        depth_tensor_bchw = depth_tensor.unsqueeze(0).unsqueeze(1).to(device) if depth_tensor is not None else None # [B, 1, H, W] for loss functions that expect channel dim
        

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
            params, optimizers_dict, strategy_state, step, render_meta
        )

        # 4. Loss Calculation
        # Combine L1 and SSIM
        # Rearrange for SSIM: [B, H, W, C] -> [B, C, H, W]
        render_perm = render.permute(0, 3, 1, 2)
        gt_perm = gt_image.permute(0, 3, 1, 2)

        l1_loss = (render - gt_image).abs().mean()
        # ssim_loss = 1.0 - ssim(render_perm, gt_perm)
        ssim_loss = 1.0 - fused_ssim(
            render_perm, gt_perm
        )

        # LPIPS LOSS with random patch
        h, w = render_perm.shape[2:]
        if lpips is not None:
            if h >= 1024 and w >= 1024:
                patch_size = 1024
                top = torch.randint(0, h - patch_size, (1,))
                left = torch.randint(0, w - patch_size, (1,))

                render_patch = render_perm[:, :, top:top+patch_size, left:left+patch_size].clamp(0.0, 1.0) # Using clamp because bilinear interpolation can produce slight out-of-range values which cause LPIPS to return NaN error
                gt_patch = gt_perm[:, :, top:top+patch_size, left:left+patch_size].clamp(0.0, 1.0) # Using clamp because bilinear interpolation can produce slight out-of-range values which cause LPIPS to return NaN or throw error
            else: #use the original images if smaller than 1024
                render_patch = render_perm
                gt_patch = gt_perm

            lpips_loss = lpips(render_patch, gt_patch)
        else:
            lpips_loss = torch.tensor(0.0, device=device)
        # render_lpips = F.interpolate(render_perm, size=512, mode='bilinear', align_corners=False)
        # gt_lpips = F.interpolate(gt_perm, size=512, mode='bilinear', align_corners=False)
        # lpips_loss = lpips(render_lpips, gt_lpips)
        # lpips_loss = torch.tensor(0.0, device=device)
        psnr_value = psnr(render_perm, gt_perm)

        loss = 0.8 * l1_loss + 0.2 * ssim_loss
        if lpips_loss is not None and lpips_loss > 0.0:
            loss = loss + training_cfg.lpips_loss_weight * lpips_loss  # Add LPIPS with a smaller weight to avoid overpowering the main losses

        # if depth_cfg.enable_depth_smoothness_loss and depth_tensor is not None:
        #     # gt_image_bchw = gt_image.permute(0, 3, 1, 2)  # [B, C, H, W]
        #     gt_depth_bchw = depth_tensor.unsqueeze(0).unsqueeze(1).to(device) if depth_tensor is not None else None  # [B, 1, H, W]
        #     render_depth_map_bchw = render_depth_map.unsqueeze(0)  # [B, 1, H, W]
        #     smoothness_loss = depth_smoothness_loss(render_depth_map_bchw, gt_depth_bchw, mask=depth_mask_bchw)
        #     loss = loss + depth_cfg.depth_smoothness_loss_weight * smoothness_loss
        # else:
        #     smoothness_loss = torch.tensor(0.0, device=device)

        # Depth loss (conditional, scale-invariant for monocular depth priors)
        inv_rendered_depth, inv_prior_depth = None, None
        depth_corr = torch.tensor(0.0, device=device)
        affine_invariant_d_loss = torch.tensor(0.0, device=device)
        pearson_module_d_loss = torch.tensor(0.0, device=device)
        silog_d_loss = torch.tensor(0.0, device=device)
        ordinal_d_loss = torch.tensor(0.0, device=device)
        affine_aligned_grad_d_loss = torch.tensor(0.0, device=device)

        if depth_tensor is not None:
            if depth_cfg.affine_invariant_depth_loss_weight > 0.0:
                affine_invariant_d_loss = affine_invariant_depth_loss(render_depth_map_bchw, depth_tensor_bchw, depth_mask_bchw)
                loss = loss + depth_cfg.affine_invariant_depth_loss_weight * affine_invariant_d_loss

            if depth_cfg.pearson_correlation_loss_weight > 0.0:
                pearson_module_d_loss = pearson_correlation_loss(render_depth_map_bchw, depth_tensor_bchw, depth_mask_bchw)
                loss = loss + depth_cfg.pearson_correlation_loss_weight * pearson_module_d_loss

            if depth_cfg.silog_loss_weight > 0.0:
                silog_d_loss = silog_depth_loss(render_depth_map_bchw, depth_tensor_bchw, depth_mask_bchw)
                loss = loss + depth_cfg.silog_loss_weight * silog_d_loss

            if depth_cfg.ordinal_depth_loss_weight > 0.0:
                ordinal_d_loss = ordinal_depth_loss(render_depth_map_bchw, depth_tensor_bchw, depth_mask_bchw)
                loss = loss + depth_cfg.ordinal_depth_loss_weight * ordinal_d_loss

            if depth_cfg.affine_aligned_gradient_matching_loss_weight > 0.0:
                affine_aligned_grad_d_loss = affine_aligned_gradient_matching_loss(render_depth_map_bchw, depth_tensor_bchw, depth_mask_bchw)
                loss = loss + depth_cfg.affine_aligned_gradient_matching_loss_weight * affine_aligned_grad_d_loss
                    

        if depth_cfg.enable_depth_loss and depth_tensor is not None and step >= depth_cfg.depth_loss_start_iter:

            # Move depth to GPU if available
            if depth_tensor is not None and not depth_tensor.is_cuda:
                depth_tensor = depth_tensor.to(device, non_blocking=True)
            # Replace any NaN or Inf values with zeros
            render_depth_map = torch.where(
                torch.isfinite(render_depth_map), 
                render_depth_map, 
                torch.zeros_like(render_depth_map)
            )
            
            if depth_tensor.dim() == 2:
                depth_tensor = depth_tensor.unsqueeze(0)
            elif depth_tensor.dim() == 4 and depth_tensor.shape[-1] == 1:
                depth_tensor = depth_tensor[..., 0]


            try:
                # d_loss, depth_corr = losses.pearson_correlation_depth_loss(
                #     render_depth_map, 
                #     depth_tensor.detach(),
                #     depth_mask
                # )
                # d_loss = losses.silog_depth_loss(
                #     render_depth_map, 
                #     depth_tensor.detach(),
                #     depth_mask
                # )
                current_depth_corr = depth_corr.mean().item() if torch.isfinite(depth_corr).all() else 0.0
                inv_rendered_depth = render_depth_map.detach()
                inv_prior_depth = depth_tensor.detach()

            except RuntimeError as e:
                d_loss = torch.tensor(0.0, device=device)
                current_depth_corr = 0.0
                logger.debug(f"[yellow]⚠ Depth loss skipped:[/yellow] {str(e)}")
        else:
            d_loss = torch.tensor(0.0, device=device)
            current_depth_corr = 0.0

        # Sharpness-aware minimization loss (gradient-domain detail preservation)
        if depth_cfg.sam_loss_weight > 0.0:
            sam_loss = losses.gradient_loss(render, gt_image)
            loss = loss + depth_cfg.sam_loss_weight * sam_loss
        else:
            sam_loss = torch.tensor(0.0, device=device)
        
        # Add scale regularization if enabled (floater prevention technique #2)
        if floater_cfg.enable_scale_reg:
            scale_reg_loss = losses.scale_regularization(
                model.scales, 
                weight=floater_cfg.scale_reg_weight, 
                scene_extent=float(scene_extent)
            )
            loss = loss + scale_reg_loss
        else:
            scale_reg_loss = torch.tensor(0.0, device=device)

        if floater_cfg.enable_opacity_reg:
            opacity_reg_loss = losses.opacity_regularization(
                model.opacities, 
                weight=floater_cfg.opacity_reg_weight
            )
            loss = loss + opacity_reg_loss
        else:
            opacity_reg_loss = torch.tensor(0.0, device=device)

        if floater_cfg.enable_opacity_entropy_reg and step >= floater_cfg.opacity_entropy_reg_start_iter and step <= floater_cfg.opacity_entropy_reg_end_iter:
            opacity_entropy_reg_loss = losses.opacity_entropy_regularization(
                model.opacities,
                weight=floater_cfg.opacity_entropy_reg_weight
            )
            loss = loss + opacity_entropy_reg_loss
        else:
            opacity_entropy_reg_loss = torch.tensor(0.0, device=device)


        # Update metrics only periodically for logging to avoid excessive CPU-GPU synchronization
        if step % training_cfg.log_interval == 0 or step == training_cfg.iterations - 1:
            current_loss = loss.item()
            current_l1 = l1_loss.item()
            current_ssim = ssim_loss.item()
            current_lpips = lpips_loss.item()
            current_psnr = psnr_value.item()
            current_depth_smoothness_loss = smoothness_loss.item() if depth_cfg.enable_depth_smoothness_loss else torch.tensor(0.0).item()
            current_scale_reg = scale_reg_loss.item()
            current_opacity_reg = opacity_reg_loss.item()
            current_opacity_entropy_reg = opacity_entropy_reg_loss.item()
            current_depth_loss = d_loss.item()
            current_affine_invariant_depth_loss = affine_invariant_d_loss.item()
            current_pearson_correlation_loss = pearson_module_d_loss.item()
            current_silog_loss = silog_d_loss.item()
            current_ordinal_depth_loss = ordinal_d_loss.item()
            current_affine_aligned_gradient_matching_loss = affine_aligned_grad_d_loss.item()
            current_sam_loss = sam_loss.item()
            
            # Log metrics to TensorBoard
            if tensorboard_cfg.tensorboard:
                # Prepare losses dict
                losses_dict = {
                    'total_loss': current_loss,
                    'l1_loss': current_l1,
                    'ssim_loss': current_ssim,
                    'lpips_loss': current_lpips,
                    'scale_reg_loss': current_scale_reg,
                    'opacity_reg_loss': current_opacity_reg,
                    'opacity_entropy_reg_loss': current_opacity_entropy_reg,
                }
                if depth_cfg.enable_depth_smoothness_loss:
                    losses_dict['depth_smoothness_loss'] = current_depth_smoothness_loss
                if depth_cfg.enable_depth_loss and step >= depth_cfg.depth_loss_start_iter:
                    losses_dict['depth_loss'] = current_depth_loss
                    losses_dict['depth_corr'] = current_depth_corr
                if depth_cfg.affine_invariant_depth_loss_weight > 0.0:
                    losses_dict['affine_invariant_depth_loss'] = current_affine_invariant_depth_loss
                if depth_cfg.pearson_correlation_loss_weight > 0.0:
                    losses_dict['pearson_correlation_loss'] = current_pearson_correlation_loss
                if depth_cfg.silog_loss_weight > 0.0:
                    losses_dict['silog_loss'] = current_silog_loss
                if depth_cfg.ordinal_depth_loss_weight > 0.0:
                    losses_dict['ordinal_depth_loss'] = current_ordinal_depth_loss
                if depth_cfg.affine_aligned_gradient_matching_loss_weight > 0.0:
                    losses_dict['affine_aligned_gradient_matching_loss'] = current_affine_aligned_gradient_matching_loss
                if depth_cfg.sam_loss_weight > 0.0:
                    losses_dict['sam_loss'] = current_sam_loss

                # Snapshot logging inputs so loop-variable updates cannot race with async logging.
                losses_snapshot = dict(losses_dict)
                max_radii_snapshot = render_meta.get('radii', None)
                if isinstance(max_radii_snapshot, torch.Tensor):
                    max_radii_snapshot = max_radii_snapshot.detach().clone()

                executor.submit(
                    _log_interval_metrics_async,
                    tb_logger=tb_logger,
                    losses_snapshot=losses_snapshot,
                    psnr_snapshot=float(current_psnr),
                    ssim_snapshot=float(current_ssim),
                    lpips_snapshot=float(current_lpips),
                    num_gaussians_snapshot=int(len(model._means)),
                    max_radii_snapshot=max_radii_snapshot,
                    step_snapshot=int(step),
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
        if tensorboard_cfg.tensorboard and step % tensorboard_cfg.tb_image_interval == 0:
            # We offload image logging to a background thread.
            # Passing detached GPU tensors is safe as the thread will handle the CPU transfer
            # (synchronization point) without stalling the main training loop.
            executor.submit(
                tb_logger.log_images,
                rendered=render[0].detach(),
                ground_truth=gt_image[0].detach(),
                alpha=alpha[0].detach() if alpha.dim() >= 3 else alpha.detach(),
                rendered_depth_map=render_depth_map[0].detach(),
                inv_rendered_depth=inv_rendered_depth[0] if inv_rendered_depth is not None else None,
                inv_prior_depth=depth_tensor.squeeze(0).detach() if depth_tensor is not None else None,
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

        if not loss.isfinite():
            current_loss = loss.item()
            current_l1 = l1_loss.item()
            current_ssim = ssim_loss.item()
            current_lpips = lpips_loss.item()
            current_psnr = psnr_value.item()
            current_scale_reg = scale_reg_loss.item()
            current_opacity_reg = opacity_reg_loss.item()
            current_opacity_entropy_reg = opacity_entropy_reg_loss.item()
            current_depth_loss = d_loss.item()
            current_affine_invariant_depth_loss = affine_invariant_d_loss.item()
            current_pearson_correlation_loss = pearson_module_d_loss.item()
            current_silog_loss = silog_d_loss.item()
            current_ordinal_depth_loss = ordinal_d_loss.item()
            current_affine_aligned_gradient_matching_loss = affine_aligned_grad_d_loss.item()
            current_sam_loss = sam_loss.item()

            logger.warning(f"[yellow]⚠ Loss is NaN or too large at step {step}[/yellow]: {loss.item()}")
            logger.warning(f"[yellow]   Current Metrics:[/yellow] loss={current_loss:.4f}, l1={current_l1:.4f}, ssim_loss={current_ssim:.4f}, lpips={current_lpips:.4f}, psnr={current_psnr:.2f}, scale_reg={current_scale_reg:.6f}, opacity_reg={current_opacity_reg:.6f}, opacity_entropy_reg={current_opacity_entropy_reg:.6f}, depth_loss={current_depth_loss:.4f}, affine_invariant_depth_loss={current_affine_invariant_depth_loss:.4f}, pearson_correlation_loss={current_pearson_correlation_loss:.4f}, silog_loss={current_silog_loss:.4f}, ordinal_depth_loss={current_ordinal_depth_loss:.4f}, affine_aligned_gradient_matching_loss={current_affine_aligned_gradient_matching_loss:.4f}, sam_loss={current_sam_loss:.4f}")
            if VERBOSITY >= 1: progress.stop()
            raise RuntimeError(f"Loss is NaN or too large at step {step}")
        loss.backward()

        # Densify and prune using strategy
        gaussians_before = len(model._means)
        
        # Get params and optimizers for strategy (step_post_backward)
        params = model.get_params_dict()
        optimizers_dict = model.get_optimizers_dict(optimizers)
        
        # Call strategy's step_post_backward to handle densification/pruning
        # Guard against missing grad metadata (e.g., if a future branch overwrites meta)
        # grad_key = getattr(strategy, 'key_for_gradient', 'means2d')
        # grad_tensor = render_meta.get(grad_key, None)
        # if grad_tensor is not None and grad_tensor.grad is not None:
        strategy.step_post_backward(
            params=params,
            optimizers=optimizers_dict,
            state=strategy_state,
            step=step,
            info=render_meta,
            packed=True
        )
        # elif VERBOSITY >= 2:
        #     logger.warning(
        #         f"[yellow]⚠ Skipping densification at step {step}:[/yellow] "
        #         f"missing gradient for meta['{grad_key}']"
        #     )
        
        # Update model parameters from strategy (they may have changed due to split/duplicate/prune)
        model.update_params_from_dict(params)
        gaussians_after = len(model._means)
        
        # Clear CUDA cache occasionally after densification
        if step % (densification_cfg.densify_interval * 1) == 0:
            torch.cuda.empty_cache()

        # Handle manual densification events for logging and opacity resets
        if step >= densification_cfg.densify_from_iter and step <= densification_cfg.densify_until_iter:
            if step % DENSIFY_INTERVAL == 0:
                if tensorboard_cfg.tensorboard:
                    tb_logger.log_densification_event(gaussians_before, gaussians_after, step=step)
                
                if gaussians_after == 0:
                    logger.error("[bold red]⚠ Critical:[/bold red] All Gaussians were pruned!")
                    if VERBOSITY >= 1: progress.stop()
                    raise RuntimeError(f"All Gaussians removed at iteration {step}")
            
            # Periodically reset opacity to prevent floaters
            # if step > 0 and step % densification_cfg.opacity_reset_interval == 0:
            #     model._opacities.data = torch.clamp(
            #         model._opacities.data,
            #         max=inverse_sigmoid(torch.tensor(densification_cfg.opacity_reset_value, device=device))
            #     )

        # Log histograms periodically
        if tensorboard_cfg.tensorboard and step > 0 and step % tensorboard_cfg.tb_histogram_interval == 0:
            tb_logger.log_gaussian_histograms(model, step=step)

        # Optimizer step
        for opt in optimizers.__dict__.values():
            if opt is None:
                continue
            opt.step()
            opt.zero_grad(set_to_none=True)
        
        # Step learning rate schedulers
        for sched in schedulers.__dict__.values():
            if sched is not None:
                sched.step()
        
        # Log learning rates
        if tensorboard_cfg.tensorboard and step % training_cfg.log_interval == 0:
            tb_logger.log_learning_rates(optimizers, step=step)
        
        # Update viewer
        if viewer is not None:
            if viewer_param_sync is not None:
                viewer_param_sync.refresh_if_needed(step)
            viewer.lock.release()
            step_time = time.perf_counter() - step_start_time
            num_train_rays_per_step = cam['width'] * cam['height']
            num_train_steps_per_sec = 1.0 / step_time if step_time > 0 else 0
            num_train_rays_per_sec = num_train_rays_per_step * num_train_steps_per_sec
            viewer.render_tab_state.num_train_rays_per_sec = num_train_rays_per_sec
            viewer.update(step, num_train_rays_per_step)

        # Save checkpoints
        if step % training_cfg.save_interval == 0 and step > 0:
            checkpoint_path = checkpoint_dir / f'checkpoint_{step}.pt'
            torch.save({
                'iteration': step,
                'model_state_dict': model.state_dict(),
                'optimizers_state_dict': {key: opt.state_dict() for key, opt in optimizers.__dict__.items() if opt is not None},
                'loss': loss.item(),
                'tensorboard_run_name': tb_logger.run_name if tensorboard_cfg.tensorboard else None,
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
        ACTIVE_PROGRESS = None

    # Wait for background logging tasks to complete
    if VERBOSITY >= 1:
        logger.info("[dim]⌛ Waiting for background logging tasks to finish...[/dim]")
    executor.shutdown(wait=True)
    
    # Log final metrics to TensorBoard
    if tensorboard_cfg.tensorboard:
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
        'iteration': training_cfg.iterations,
        'model_state_dict': model.state_dict(),
        'scene_extent': scene_extent,
        'num_gaussians': len(model._means),
        'tensorboard_run_name': tb_logger.run_name if tensorboard_cfg.tensorboard else None,
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

if __name__ == "__main__":
    config = parse_args()
    required_cfg = config.required
    output_cfg = config.output
    training_cfg = config.training
    densification_cfg = config.densification
    floater_cfg = config.floater_prevention
    sh_cfg = config.sh
    depth_cfg = config.depth
    export_cfg = config.export
    runtime_cfg = config.runtime
    
    console = Console()
    
    # Display configuration
    if runtime_cfg.verbosity >= 1:
        console.print()
        console.rule("[bold cyan]3D Gaussian Splatting Training[/bold cyan]", style="cyan")
        console.print()
        
        config_table = Table(title="Configuration", box=box.ROUNDED, show_header=False)
        config_table.add_column("Parameter", style="cyan", width=20)
        config_table.add_column("Value", style="yellow")
        
        config_table.add_row("Dataset Type", required_cfg.dataset_type)
        if required_cfg.dataset_type == "colmap":
            config_table.add_row("COLMAP Path", str(required_cfg.colmap_path))
            config_table.add_row("Images Path", str(required_cfg.images_path))
        elif required_cfg.dataset_type == "instant-ngp":
            config_table.add_row("Transforms Path", str(required_cfg.transforms_path))
            config_table.add_row("Images Path", str(required_cfg.images_path) if required_cfg.images_path else "auto")
            if required_cfg.point_cloud_path:
                config_table.add_row("Point Cloud Path", str(required_cfg.point_cloud_path))
        config_table.add_row("Output Directory", output_cfg.output_dir)
        config_table.add_row("Iterations", f"{training_cfg.iterations:,}")
        config_table.add_row("Densify From", f"{densification_cfg.densify_from_iter:,}")
        config_table.add_row("Densify Until", f"{densification_cfg.densify_until_iter:,}")
        config_table.add_row("Verbosity", ["QUIET", "NORMAL", "VERBOSE", "DEBUG"][runtime_cfg.verbosity])
        
        # Add SH configuration
        sh_status = "Disabled (DC only)" if sh_cfg.disable_sh_rendering else f"Degree {sh_cfg.sh_degree}"
        config_table.add_row("Spherical Harmonics", sh_status)
        
        # Add depth supervision configuration
        if depth_cfg.enable_depth_loss:
            depth_config = f"Enabled (weight={depth_cfg.depth_loss_weight}, start_iter={depth_cfg.depth_loss_start_iter})"
            config_table.add_row("Depth Supervision", depth_config)
            extra_depth_losses = []
            if depth_cfg.affine_invariant_depth_loss_weight > 0.0:
                extra_depth_losses.append(f"AffineInvariant(λ={depth_cfg.affine_invariant_depth_loss_weight})")
            if depth_cfg.pearson_correlation_loss_weight > 0.0:
                extra_depth_losses.append(f"PearsonClass(λ={depth_cfg.pearson_correlation_loss_weight})")
            if depth_cfg.silog_loss_weight > 0.0:
                extra_depth_losses.append(f"SILog(λ={depth_cfg.silog_loss_weight})")
            if depth_cfg.ordinal_depth_loss_weight > 0.0:
                extra_depth_losses.append(f"Ordinal(λ={depth_cfg.ordinal_depth_loss_weight})")
            if depth_cfg.affine_aligned_gradient_matching_loss_weight > 0.0:
                extra_depth_losses.append(f"AffineAlignedGrad(λ={depth_cfg.affine_aligned_gradient_matching_loss_weight})")
            if depth_cfg.enable_depth_smoothness_loss and depth_cfg.depth_smoothness_loss_weight > 0.0:
                extra_depth_losses.append(f"FastPriorGrad(λ={depth_cfg.depth_smoothness_loss_weight})")
            if extra_depth_losses:
                config_table.add_row("Depth Aux Losses", ", ".join(extra_depth_losses))
        if depth_cfg.sam_loss_weight > 0:
            config_table.add_row("SAM Loss", f"Enabled (weight={depth_cfg.sam_loss_weight})")
        
        # Show enabled floater prevention techniques
        floater_techniques = []
        if floater_cfg.enable_scale_reg:
            floater_techniques.append(f"Scale Reg (λ={floater_cfg.scale_reg_weight})")
        if floater_cfg.enable_opacity_reg:
            floater_techniques.append(f"Opacity Reg (λ={floater_cfg.opacity_reg_weight})")
        if floater_cfg.enable_opacity_entropy_reg:
            floater_techniques.append(f"Opacity Entropy Reg (λ={floater_cfg.opacity_entropy_reg_weight})")
        if floater_techniques:
            config_table.add_row("Floater Prevention", ", ".join(floater_techniques))
        
        console.print(config_table)
        console.print()
    
    # Setup logger for main block
    logger = setup_logger(runtime_cfg.verbosity, Path(output_cfg.output_dir))
    
    # Train model
    try:
        result = train_pipeline(config)
    except Exception:
        if ACTIVE_PROGRESS is not None:
            try:
                ACTIVE_PROGRESS.stop()
            except Exception:
                pass
            ACTIVE_PROGRESS = None
        raise
    
    # Handle return values (model or model+viewer)
    if isinstance(result, tuple):
        model, viewer = result
    else:
        model = result
        viewer = None
    
    # Export to PLY if requested
    if export_cfg.export_ply:
        ply_path = Path(output_cfg.output_dir) / 'final_gaussians.ply'
        model.save_ply(str(ply_path))
        if runtime_cfg.verbosity >= 1:
            logger.info(f"[green]📦 Exported to PLY:[/green] {ply_path}")
    
    if runtime_cfg.verbosity >= 1:
        console.print()
        final_info = Table.grid(padding=(0, 1))
        final_info.add_column(style="cyan")
        final_info.add_row(f"📁 Output: [yellow]{output_cfg.output_dir}[/yellow]")
        final_info.add_row(f"📊 Checkpoints: Every [yellow]{training_cfg.save_interval:,}[/yellow] iterations")
        final_info.add_row(f"🎨 Final model: [yellow]{output_cfg.output_dir}/model_final.pt[/yellow]")
        console.print(final_info)
        console.print()
    
    # Keep viewer alive if enabled
    if viewer is not None:
        if runtime_cfg.verbosity >= 1:
            logger.info("[blue]📺 Viewer running...[/blue] Press Ctrl+C to exit.")
        try:
            time.sleep(1000000)
        except KeyboardInterrupt:
            if runtime_cfg.verbosity >= 1:
                logger.info("[yellow]Viewer closed.[/yellow]")
