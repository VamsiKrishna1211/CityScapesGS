"""
TensorBoard logger for 3D Gaussian Splatting training.

This module provides a clean interface for logging training metrics,
losses, images, and model statistics to TensorBoard.
"""

import logging
import secrets
import warnings
from pathlib import Path
from typing import Dict, Optional

import torch
from torch.utils.tensorboard import SummaryWriter

logger = logging.getLogger("cityscape_gs.logger")


def configure_app_logger(
    verbosity: int,
    output_dir: Path,
    *,
    log_filename: str = "training.log",
    logger_name: str = "cityscape_gs.train",
) -> logging.Logger:
    """Configure application logging for the whole cityscape_gs namespace.

    Handlers are attached to the parent namespace logger so child loggers like
    ``cityscape_gs.dataset`` and ``cityscape_gs.model`` propagate correctly.
    """
    from rich.logging import RichHandler

    level_map = {
        0: logging.WARNING,
        1: logging.INFO,
        2: logging.DEBUG,
        3: logging.DEBUG,
    }
    log_level = level_map.get(verbosity, logging.INFO)

    namespace_logger = logging.getLogger("cityscape_gs")
    namespace_logger.setLevel(logging.DEBUG)
    namespace_logger.propagate = False
    namespace_logger.handlers.clear()

    console_handler = RichHandler(
        rich_tracebacks=True,
        markup=True,
        show_time=True,
        show_path=verbosity >= 3,
    )
    console_handler.setLevel(log_level)
    namespace_logger.addHandler(console_handler)

    output_dir.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(output_dir / log_filename, mode="w")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    namespace_logger.addHandler(file_handler)

    app_logger = logging.getLogger(logger_name)
    app_logger.setLevel(logging.DEBUG)
    app_logger.handlers.clear()
    app_logger.propagate = True
    return app_logger

# Optional imports for system metrics
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    warnings.warn("psutil not available. Install with: pip install psutil")

try:
    import pynvml
    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False
    warnings.warn("pynvml not available. Install with: pip install nvidia-ml-py3")


class GaussianSplattingLogger:
    """TensorBoard logger for 3D Gaussian Splatting training."""
    
    # Name generator components (adjective + noun pattern)
    ADJECTIVES = [
        'ancient', 'azure', 'blazing', 'bold', 'brave', 'bright', 'calm', 'clever',
        'cosmic', 'crystal', 'daring', 'divine', 'dynamic', 'eager', 'elegant', 'epic',
        'fancy', 'fierce', 'flaming', 'flying', 'frosty', 'gentle', 'glowing', 'golden',
        'graceful', 'grand', 'happy', 'heroic', 'hidden', 'humble', 'iron', 'jolly',
        'keen', 'legendary', 'lively', 'lucky', 'lunar', 'magic', 'mighty', 'mystic',
        'noble', 'peaceful', 'proud', 'quantum', 'quiet', 'radiant', 'rapid', 'robust',
        'royal', 'sacred', 'sharp', 'shining', 'silent', 'silver', 'sleek', 'smooth',
        'solar', 'solid', 'sparkling', 'stellar', 'swift', 'thunder', 'tranquil', 'trusting',
        'twilight', 'valiant', 'vibrant', 'vigilant', 'violet', 'vivid', 'wandering', 'wild',
        'wise', 'zealous'
    ]
    
    NOUNS = [
        'aurora', 'beacon', 'blade', 'blaze', 'breeze', 'canyon', 'cascade', 'castle',
        'cloud', 'comet', 'compass', 'cosmos', 'crest', 'crown', 'crystal', 'dagger',
        'delta', 'dragon', 'eagle', 'echo', 'eclipse', 'ember', 'falcon', 'flame',
        'forest', 'fortress', 'galaxy', 'glacier', 'hawk', 'horizon', 'island', 'jewel',
        'knight', 'lance', 'lightning', 'lion', 'lotus', 'meadow', 'meteor', 'moon',
        'mountain', 'nebula', 'nova', 'oasis', 'ocean', 'orbit', 'peak', 'phoenix',
        'planet', 'prism', 'pulse', 'quest', 'raven', 'ridge', 'river', 'sapphire',
        'shadow', 'shield', 'sphere', 'spirit', 'star', 'storm', 'summit', 'sunrise',
        'sunset', 'sword', 'temple', 'thunder', 'titan', 'torch', 'tower', 'valley',
        'vertex', 'wave', 'whisper', 'wind', 'wolf', 'zenith'
    ]
    
    @staticmethod
    def generate_run_name() -> str:
        """
        Generate a unique, memorable name for this training run.
        
        Format: adjective-noun-XXXX
        Example: cosmic-phoenix-a1b2
        
        Returns:
            Unique run name string
        """
        # Use OS entropy so run naming is unaffected by global random seeds
        # used for deterministic training.
        adjective = secrets.choice(GaussianSplattingLogger.ADJECTIVES)
        noun = secrets.choice(GaussianSplattingLogger.NOUNS)
        nonce = secrets.token_hex(2)
        return f"{adjective}-{noun}-{nonce}"
    
    def __init__(self, log_dir: str, enabled: bool = True, run_name: Optional[str] = None, purge_step: Optional[int] = None):
        """
        Initialize TensorBoard logger.
        
        Args:
            log_dir: Base directory to save TensorBoard logs
            enabled: Whether logging is enabled (default: True)
            run_name: Optional custom run name. If None, generates a unique name automatically
            purge_step: Optional step to purge stale events when resuming an existing run
        """
        self.enabled = enabled
        self.purge_step = purge_step
        if self.enabled:
            # Generate unique run name if not provided
            self.run_name = run_name if run_name else self.generate_run_name()
            
            # Create run-specific subdirectory
            self.log_dir = Path(log_dir) / self.run_name
            if self.log_dir.exists() and run_name is None:
                # Extremely rare, but avoid accidental writer reuse.
                self.run_name = f"{self.run_name}-{secrets.token_hex(2)}"
                self.log_dir = Path(log_dir) / self.run_name
            self.log_dir.mkdir(parents=True, exist_ok=True)
            
            self.writer = SummaryWriter(str(self.log_dir), purge_step=purge_step)
            
            # Initialize NVML for GPU monitoring (once at startup)
            self._nvml_initialized = False
            if PYNVML_AVAILABLE:
                try:
                    pynvml.nvmlInit()
                    self._nvml_initialized = True
                    self._gpu_count = pynvml.nvmlDeviceGetCount()
                except Exception:
                    self._nvml_initialized = False
        else:
            self.writer = None
            self.run_name = None
            self._nvml_initialized = False
    
    def log_losses(self, total_loss: float, 
                   l1_loss: float, 
                   ssim_loss: float, 
                   lpips_loss: float,
                   depth_smoothness_loss: Optional[float] = None,
                   scale_reg_loss: Optional[float] = None,
                   opacity_reg_loss: Optional[float] = None,
                   opacity_entropy_reg_loss: Optional[float] = None,
                   depth_loss: Optional[float] = None,
                   depth_corr: Optional[float] = None,
                   affine_invariant_depth_loss: Optional[float] = None,
                   pearson_correlation_loss: Optional[float] = None,
                   silog_loss: Optional[float] = None,
                   ordinal_depth_loss: Optional[float] = None,
                   affine_aligned_gradient_matching_loss: Optional[float] = None,
                   metric_depth_normal_loss: Optional[float] = None,
                   dn_splatter_normal_loss: Optional[float] = None,
                   sam_loss: Optional[float] = None,
                   semantic_loss: Optional[float] = None,
                   step: int = 0):
        """
        Log all loss components.
        
        Args:
            total_loss: Combined total loss
            l1_loss: L1 reconstruction loss
            ssim_loss: SSIM loss component
            lpips_loss: LPIPS perceptual loss
            depth_smoothness_loss: Optional depth smoothness loss
            scale_reg_loss: Scale regularization loss
            opacity_reg_loss: Optional opacity regularization loss
            opacity_entropy_reg_loss: Optional opacity entropy regularization loss
            depth_loss: Optional depth supervision loss
            depth_corr: Optional signed Pearson depth correlation in [-1, 1]
            affine_invariant_depth_loss: Optional affine-invariant depth loss
            pearson_correlation_loss: Optional Pearson-correlation depth loss module value
            silog_loss: Optional SI-Log depth loss
            ordinal_depth_loss: Optional ordinal depth ranking loss
            affine_aligned_gradient_matching_loss: Optional affine-aligned gradient matching loss
            metric_depth_normal_loss: Optional metric depth normal loss
            dn_splatter_normal_loss: Optional DN-Splatter normal loss
            sam_loss: Optional sharpness-aware minimization loss
            semantic_loss: Optional semantic reconstruction loss
            step: Current training step
        """
        if not self.enabled:
            return
        
        self.writer.add_scalar('Loss/Total', total_loss, step)
        self.writer.add_scalar('Loss/L1', l1_loss, step)
        self.writer.add_scalar('Loss/SSIM', ssim_loss, step)
        if lpips_loss > 0:
            self.writer.add_scalar('Loss/LPIPS', lpips_loss, step)
        if depth_smoothness_loss is not None and depth_smoothness_loss > 0:
            self.writer.add_scalar('Loss/DepthSmoothness', depth_smoothness_loss, step)
        if scale_reg_loss is not None and scale_reg_loss > 0:
            self.writer.add_scalar('Loss/ScaleRegularization', scale_reg_loss, step)
        if opacity_reg_loss is not None and opacity_reg_loss > 0:
            self.writer.add_scalar('Loss/OpacityRegularization', opacity_reg_loss, step)
        if opacity_entropy_reg_loss is not None and opacity_entropy_reg_loss > 0:
            self.writer.add_scalar('Loss/OpacityEntropyRegularization', opacity_entropy_reg_loss, step)
        if depth_loss is not None and depth_loss > 0:
            self.writer.add_scalar('Loss/Depth', depth_loss, step)
        if depth_corr is not None:
            self.writer.add_scalar('Depth/PearsonCorr', depth_corr, step)
        if affine_invariant_depth_loss is not None and affine_invariant_depth_loss > 0:
            self.writer.add_scalar('Loss/AffineInvariantDepth', affine_invariant_depth_loss, step)
        if pearson_correlation_loss is not None and pearson_correlation_loss > 0:
            self.writer.add_scalar('Loss/PearsonCorrelationDepth', pearson_correlation_loss, step)
        if silog_loss is not None and silog_loss > 0:
            self.writer.add_scalar('Loss/SILog', silog_loss, step)
        if ordinal_depth_loss is not None and ordinal_depth_loss > 0:
            self.writer.add_scalar('Loss/OrdinalDepth', ordinal_depth_loss, step)
        if affine_aligned_gradient_matching_loss is not None and affine_aligned_gradient_matching_loss > 0:
            self.writer.add_scalar('Loss/AffineAlignedGradientMatching', affine_aligned_gradient_matching_loss, step)
        if metric_depth_normal_loss is not None and metric_depth_normal_loss > 0:
            self.writer.add_scalar('Loss/MetricDepthNormal', metric_depth_normal_loss, step)
        if dn_splatter_normal_loss is not None:
            self.writer.add_scalar('Loss/DNSplatterNormal', dn_splatter_normal_loss, step)
        if sam_loss is not None and sam_loss > 0:
            self.writer.add_scalar('Loss/SAM', sam_loss, step)
        if semantic_loss is not None and semantic_loss > 0:
            self.writer.add_scalar('Loss/Semantic', semantic_loss, step)
    
    def log_quality_metrics(self, psnr: float, ssim_loss: float, lpips: float, step: int):
        """
        Log image quality metrics.
        
        Args:
            psnr: Peak Signal-to-Noise Ratio
            ssim_loss: SSIM loss (will be converted to SSIM score)
            lpips: Learned Perceptual Image Patch Similarity
            step: Current training iteration
        """
        if not self.enabled:
            return
        
        self.writer.add_scalar('Quality/PSNR', psnr, step)
        # Convert SSIM loss (1 - SSIM) back to SSIM score for better readability
        self.writer.add_scalar('Quality/SSIM', 1.0 - ssim_loss, step)
        if lpips > 0:
            self.writer.add_scalar('Quality/LPIPS', lpips, step)

    def log_model_stats(self, num_gaussians: int, max_radii: Optional[torch.Tensor] = None, step: int = 0):
        """
        Log model statistics.
        
        Args:
            num_gaussians: Current number of Gaussians in the model
            max_radii: Optional tensor of maximum radii for statistics
            step: Current training iteration
        """
        if not self.enabled:
            return
        
        self.writer.add_scalar('Model/NumGaussians', num_gaussians, step)
        
        if max_radii is not None:
            # Ensure max_radii is float before computing mean
            self.writer.add_scalar('Model/AvgMaxRadii', max_radii.float().mean().item(), step)

    def log_pointcloud_mesh(
        self,
        points: torch.Tensor,
        colors: Optional[torch.Tensor] = None,
        tag: str = 'Scene/InitialPointCloud',
        step: int = 0,
        max_points: int = 500000,
    ) -> None:
        """Log a 3D point cloud to TensorBoard via SummaryWriter.add_mesh.

        Args:
            points: Point positions as [N, 3] tensor.
            colors: Optional RGB colors as [N, 3], values in [0, 1] or [0, 255].
            tag: TensorBoard tag namespace.
            step: Global step used for this mesh event.
            max_points: Optional cap to keep event size manageable.
        """
        if not self.enabled:
            return

        if points is None or points.numel() == 0:
            return

        try:
            pts = points.detach().float().cpu()
            cols = colors.detach().float().cpu() if colors is not None else None

            if pts.dim() != 2 or pts.shape[1] != 3:
                logger.warning(f"Skipping TensorBoard point cloud log: expected [N,3], got {tuple(pts.shape)}")
                return

            n = pts.shape[0]
            if n > max_points:
                idx = torch.randperm(n)[:max_points]
                pts = pts[idx]
                if cols is not None and cols.shape[0] == n:
                    cols = cols[idx]

            if cols is not None:
                if cols.dim() != 2 or cols.shape[1] != 3 or cols.shape[0] != pts.shape[0]:
                    cols = None
                else:
                    cols = cols.clamp(0.0, 1.0)

            # add_mesh expects [B, N, 3]
            pts_batched = pts.unsqueeze(0)
            cols_batched = cols.unsqueeze(0) if cols is not None else None
            self.writer.add_mesh(tag, vertices=pts_batched, colors=cols_batched, global_step=step)
        except Exception as e:
            logger.warning(f"Failed to log point cloud mesh to TensorBoard: {e}")
    
    def log_images(self, rendered: torch.Tensor, ground_truth: torch.Tensor,
                   alpha: Optional[torch.Tensor] = None,
                   rendered_depth_map: Optional[torch.Tensor] = None,
                   inv_rendered_depth: Optional[torch.Tensor] = None,
                   inv_prior_depth: Optional[torch.Tensor] = None,
                   step: int = 0):
        """
        Log rendered and ground truth images with optional alpha mask.
        
        Args:
            rendered: Rendered image [H, W, 3]
            ground_truth: Ground truth image [H, W, 3]
            alpha: Optional alpha mask [H, W, 1]
            inv_rendered_depth: Optional inverse rendered depth map [H, W]
            inv_prior_depth: Optional inverse prior depth map [H, W]
            step: Current training iteration
        """
        try:
            if not self.enabled:
                return
            
            # Convert from [H, W, 3] to [3, H, W] for tensorboard
            rendered_tb = rendered.permute(2, 0, 1).clamp(0, 1)
            gt_tb = ground_truth.permute(2, 0, 1).clamp(0, 1)
            
            self.writer.add_image('Images/Rendered', rendered_tb, step)
            self.writer.add_image('Images/GroundTruth', gt_tb, step)
            
            # Log absolute difference map
            diff = torch.abs(rendered - ground_truth).mean(dim=2, keepdim=True)  # [H, W, 1]
            diff_tb = diff.permute(2, 0, 1).clamp(0, 1)
            self.writer.add_image('Images/AbsoluteDifference', diff_tb, step)

            # Log rendered depth map if provided
            if rendered_depth_map is not None:
                norm_rendered_depth = (rendered_depth_map - rendered_depth_map.min()) / (rendered_depth_map.max() - rendered_depth_map.min() + 1e-8)
                self.writer.add_image('Images/RenderedDepthMap', norm_rendered_depth.unsqueeze(0), step)
            
            # Log alpha mask if provided
            if alpha is not None:
                alpha_tb = alpha.permute(2, 0, 1).clamp(0, 1)
                self.writer.add_image('Images/AlphaMask', alpha_tb, step)
            
            # Log depth maps if provided
            if inv_rendered_depth is not None:
                # rendered_depth_tb = rendered_depth.unsqueeze(0).clamp(0, 1)  # [1, H, W]
                norm_rendered_depth = (inv_rendered_depth - inv_rendered_depth.min()) / (inv_rendered_depth.max() - inv_rendered_depth.min() + 1e-8)
                self.writer.add_image('Images/RenderedDepth', norm_rendered_depth.unsqueeze(0), step)
            if inv_prior_depth is not None:
                # gt_depth_tb = gt_depth.unsqueeze(0).clamp(0, 1)  # [1, H, W]
                norm_gt_depth = (inv_prior_depth - inv_prior_depth.min()) / (inv_prior_depth.max() - inv_prior_depth.min() + 1e-8)
                self.writer.add_image('Images/GroundTruthDepth', norm_gt_depth.unsqueeze(0), step)
        except Exception as e:
            logger.warning(f"Failed to log depth images: {e}")
    
    def log_gaussian_histograms(self, model, step: int):
        """
        Log histograms of Gaussian parameters and their gradients.

        Gradient logging is model-agnostic: core params come from
        model.get_params_dict() (BaseTrainableModel contract), and extra
        params (MLPs, language features) come from model.iter_extra_optimizers().
        Gradients are available here because logging runs after loss.backward()
        but before optimizer.zero_grad().

        Args:
            model: Any BaseTrainableModel instance
            step: Current training iteration
        """
        if not self.enabled:
            return

        # Log parameter distributions
        self.writer.add_histogram('Parameters/Scales', model.scales.detach().cpu().flatten(), step)
        self.writer.add_histogram('Parameters/Opacities', model.opacities.detach().cpu().flatten(), step)
        self.writer.add_histogram('Parameters/Positions', model.means.detach().cpu().flatten(), step)

        # Log scale statistics per axis
        scales_cpu = model.scales.detach().cpu()
        for i, axis in enumerate(['X', 'Y', 'Z']):
            self.writer.add_histogram(f'Scales/{axis}', scales_cpu[:, i], step)

        # --- Gradient histograms ---
        # Core geometric/feature params (keys normalised across all model types)
        for name, param in model.get_params_dict().items():
            if param.grad is not None:
                self.writer.add_histogram(
                    f'Gradients/{name}', param.grad.detach().cpu().flatten(), step
                )

        # Extra params: MLPs, language features, etc. — aggregated per optimizer group
        for name, opt in model.iter_extra_optimizers():
            grads = [
                p.grad.detach().cpu().flatten()
                for group in opt.param_groups
                for p in group['params']
                if p.grad is not None
            ]
            if grads:
                self.writer.add_histogram(
                    f'Gradients/extra/{name}', torch.cat(grads), step
                )
    
    def log_learning_rates(self, optimizers, step: int, extra_optimizers=None):
        """
        Log current learning rates for all optimizers.

        Args:
            optimizers: GSOptimizers instance with all parameter optimizers
            step: Current training iteration
            extra_optimizers: Optional iterator of (name, optimizer) tuples for
                model-owned extras (e.g. MLP optimizers in ScaffoldModel). If None
                or exhausted, it is silently skipped.
        """
        if not self.enabled:
            return

        for name, optimizer in optimizers.all_optimizers():
            lr = optimizer.param_groups[0]['lr']
            self.writer.add_scalar(f'LearningRate/{name}', lr, step)

        if extra_optimizers is not None:
            for name, optimizer in extra_optimizers:
                lr = optimizer.param_groups[0]['lr']
                self.writer.add_scalar(f'LearningRate/{name}', lr, step)
    
    def log_training_phase(self, phase: str, step: int):
        """
        Log training phase as text.
        
        Args:
            phase: Phase name (e.g., 'Densification', 'Refinement')
            step: Current training iteration
        """
        if not self.enabled:
            return
        
        self.writer.add_text('Training/Phase', phase, step)
    
    def log_densification_event(self, gaussians_before: int, gaussians_after: int, step: int):
        """
        Log densification events.
        
        Args:
            gaussians_before: Number of Gaussians before densification
            gaussians_after: Number of Gaussians after densification
            step: Current training iteration
        """
        if not self.enabled:
            return
        
        change = gaussians_after - gaussians_before
        self.writer.add_scalar('Events/DensificationChange', change, step)
        self.writer.add_scalar('Events/GaussiansAdded', max(0, change), step)
        self.writer.add_scalar('Events/GaussiansPruned', max(0, -change), step)
    
    def log_hyperparameters(self, hparams: Dict, metrics: Optional[Dict] = None):
        """
        Log hyperparameters and optional final metrics.
        
        Args:
            hparams: Dictionary of hyperparameters
            metrics: Optional dictionary of final metrics
        """
        if not self.enabled:
            return
        
        if metrics is None:
            metrics = {}
        
        self.writer.add_hparams(hparams, metrics)
    
    def log_system_metrics(self, step: int):
        """
        Log system-level metrics including CPU, RAM, GPU usage, and power consumption.
        
        Requires:
            - psutil: pip install psutil
            - pynvml: pip install nvidia-ml-py3
        
        Args:
            step: Current training iteration
        """
        if not self.enabled:
            return
        
        # CPU and RAM metrics (via psutil)
        if PSUTIL_AVAILABLE:
            try:
                # CPU metrics
                cpu_percent = psutil.cpu_percent(interval=0.1)
                cpu_count = psutil.cpu_count(logical=True)
                cpu_count_physical = psutil.cpu_count(logical=False)
                
                self.writer.add_scalar('System/CPU/Usage_Percent', cpu_percent, step)
                self.writer.add_scalar('System/CPU/Count_Logical', cpu_count, step)
                self.writer.add_scalar('System/CPU/Count_Physical', cpu_count_physical, step)
                
                # Per-core CPU usage (optional, can be verbose)
                # cpu_per_core = psutil.cpu_percent(interval=0.1, percpu=True)
                # for i, usage in enumerate(cpu_per_core):
                #     self.writer.add_scalar(f'System/CPU/Core_{i}/Percent', usage, step)
                
                # CPU frequency
                try:
                    cpu_freq = psutil.cpu_freq()
                    if cpu_freq:
                        self.writer.add_scalar('System/CPU/Freq_Current_MHz', cpu_freq.current, step)
                        self.writer.add_scalar('System/CPU/Freq_Max_MHz', cpu_freq.max, step)
                except Exception:
                    pass
                
                # RAM metrics
                memory = psutil.virtual_memory()
                self.writer.add_scalar('System/RAM/Used_GB', memory.used / (1024**3), step)
                self.writer.add_scalar('System/RAM/Available_GB', memory.available / (1024**3), step)
                self.writer.add_scalar('System/RAM/Usage_Percent', memory.percent, step)
                self.writer.add_scalar('System/RAM/Total_GB', memory.total / (1024**3), step)
                
                # Swap memory
                swap = psutil.swap_memory()
                self.writer.add_scalar('System/Swap/Used_GB', swap.used / (1024**3), step)
                self.writer.add_scalar('System/Swap/Total_GB', swap.total / (1024**3), step)
                self.writer.add_scalar('System/Swap/Usage_Percent', swap.percent, step)
                
                # Disk I/O
                try:
                    disk_io = psutil.disk_io_counters()
                    if disk_io:
                        self.writer.add_scalar('System/Disk/Read_GB', disk_io.read_bytes / (1024**3), step)
                        self.writer.add_scalar('System/Disk/Write_GB', disk_io.write_bytes / (1024**3), step)
                        self.writer.add_scalar('System/Disk/Read_Count', disk_io.read_count, step)
                        self.writer.add_scalar('System/Disk/Write_Count', disk_io.write_count, step)
                except Exception:
                    pass
                
                # Network I/O
                try:
                    net_io = psutil.net_io_counters()
                    if net_io:
                        self.writer.add_scalar('System/Network/Sent_GB', net_io.bytes_sent / (1024**3), step)
                        self.writer.add_scalar('System/Network/Recv_GB', net_io.bytes_recv / (1024**3), step)
                except Exception:
                    pass
                    
            except Exception as e:
                # Silently skip if psutil metrics fail
                pass
        
        # GPU metrics (via pynvml for NVIDIA GPUs)
        if self._nvml_initialized:
            try:
                for i in range(self._gpu_count):
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                    
                    # GPU name (log once or rarely)
                    try:
                        gpu_name = pynvml.nvmlDeviceGetName(handle)
                        if isinstance(gpu_name, bytes):
                            gpu_name = gpu_name.decode('utf-8')
                        # Only log at step 0 to avoid cluttering
                        if step == 0:
                            self.writer.add_text(f'System/GPU_{i}_Name', gpu_name, step)
                    except Exception:
                        pass
                    
                    # GPU utilization
                    try:
                        utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                        self.writer.add_scalar(f'System/GPU_{i}/Usage_Percent', utilization.gpu, step)
                        self.writer.add_scalar(f'System/GPU_{i}/Memory_Usage_Percent', utilization.memory, step)
                    except Exception:
                        pass
                    
                    # GPU memory
                    try:
                        memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                        self.writer.add_scalar(f'System/GPU_{i}/Memory_Used_GB', memory_info.used / (1024**3), step)
                        self.writer.add_scalar(f'System/GPU_{i}/Memory_Free_GB', memory_info.free / (1024**3), step)
                        self.writer.add_scalar(f'System/GPU_{i}/Memory_Total_GB', memory_info.total / (1024**3), step)
                    except Exception:
                        pass
                    
                    # GPU temperature
                    try:
                        temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                        self.writer.add_scalar(f'System/GPU_{i}/Temperature_C', temp, step)
                    except Exception:
                        pass
                    
                    # GPU power consumption
                    try:
                        power_mw = pynvml.nvmlDeviceGetPowerUsage(handle)
                        power_w = power_mw / 1000.0  # Convert milliwatts to watts
                        self.writer.add_scalar(f'System/GPU_{i}/Power_W', power_w, step)
                    except Exception:
                        pass
                    
                    # GPU power limit (for reference)
                    try:
                        power_limit_mw = pynvml.nvmlDeviceGetPowerManagementLimit(handle)
                        power_limit_w = power_limit_mw / 1000.0
                        self.writer.add_scalar(f'System/GPU_{i}/Power_Limit_W', power_limit_w, step)
                    except Exception:
                        pass
                    
                    # GPU clock speeds
                    try:
                        clock_graphics = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_GRAPHICS)
                        clock_sm = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM)
                        clock_memory = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_MEM)
                        self.writer.add_scalar(f'System/GPU_{i}/Clock_Graphics_MHz', clock_graphics, step)
                        self.writer.add_scalar(f'System/GPU_{i}/Clock_SM_MHz', clock_sm, step)
                        self.writer.add_scalar(f'System/GPU_{i}/Clock_Memory_MHz', clock_memory, step)
                    except Exception:
                        pass
                    
                    # GPU fan speed
                    try:
                        fan_speed = pynvml.nvmlDeviceGetFanSpeed(handle)
                        self.writer.add_scalar(f'System/GPU_{i}/Fan_Percent', fan_speed, step)
                    except Exception:
                        pass
                    
                    # GPU performance state (P-state: P0=max performance, P12=min)
                    try:
                        pstate = pynvml.nvmlDeviceGetPerformanceState(handle)
                        self.writer.add_scalar(f'System/GPU_{i}/Performance_State', pstate, step)
                    except Exception:
                        pass
                        
            except Exception as e:
                # Silently skip if NVML metrics fail
                pass
        
        # PyTorch CUDA memory metrics (complementary to NVML)
        if torch.cuda.is_available():
            try:
                for i in range(torch.cuda.device_count()):
                    # Memory allocated by PyTorch
                    allocated = torch.cuda.memory_allocated(i) / (1024**3)
                    reserved = torch.cuda.memory_reserved(i) / (1024**3)
                    max_allocated = torch.cuda.max_memory_allocated(i) / (1024**3)
                    max_reserved = torch.cuda.max_memory_reserved(i) / (1024**3)
                    
                    self.writer.add_scalar(f'System/PyTorch_GPU_{i}/Allocated_GB', allocated, step)
                    self.writer.add_scalar(f'System/PyTorch_GPU_{i}/Reserved_GB', reserved, step)
                    self.writer.add_scalar(f'System/PyTorch_GPU_{i}/Max_Allocated_GB', max_allocated, step)
                    self.writer.add_scalar(f'System/PyTorch_GPU_{i}/Max_Reserved_GB', max_reserved, step)
            except Exception:
                pass
    
    def close(self):
        """Close the TensorBoard writer and flush all pending events."""
        if self.enabled and self.writer:
            self.writer.close()
        
        # Shutdown NVML if it was initialized
        if self._nvml_initialized:
            try:
                pynvml.nvmlShutdown()
                self._nvml_initialized = False
            except Exception:
                pass

    def flush(self):
        """Flush pending TensorBoard events to disk."""
        if self.enabled and self.writer:
            self.writer.flush()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensures writer is closed."""
        self.close()
