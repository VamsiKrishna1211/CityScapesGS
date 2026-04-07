# ruff: noqa: E402

import importlib.util
import logging
import os

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional, Tuple, cast


def _prepend_env_path(var_name: str, path_value: str) -> None:
    current = os.environ.get(var_name, "")
    parts = [p for p in current.split(os.pathsep) if p]
    if path_value not in parts:
        os.environ[var_name] = (
            f"{path_value}{os.pathsep}{current}" if current else path_value
        )


def _configure_cuda_jit_env() -> None:
    """Make CUDA headers discoverable for torch JIT extensions in conda envs."""
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if not conda_prefix:
        return

    conda_root = Path(conda_prefix)
    if (conda_root / "bin" / "nvcc").exists():
        os.environ.setdefault("CUDA_HOME", str(conda_root))

    include_candidates = [
        conda_root / "targets" / "x86_64-linux" / "include",
        conda_root / "include",
    ]
    for include_dir in include_candidates:
        if (include_dir / "cuda_runtime.h").exists():
            _prepend_env_path("CPATH", str(include_dir))
            _prepend_env_path("CPLUS_INCLUDE_PATH", str(include_dir))
            break

    lib_candidates = [
        conda_root / "targets" / "x86_64-linux" / "lib",
        conda_root / "lib",
    ]
    for lib_dir in lib_candidates:
        if lib_dir.exists():
            _prepend_env_path("LIBRARY_PATH", str(lib_dir))
            _prepend_env_path("LD_LIBRARY_PATH", str(lib_dir))


_configure_cuda_jit_env()

import time

import torch
import torch.nn.functional as F
from gsplat import DefaultStrategy, rasterization  # type: ignore[import-untyped]
from torch.utils.data import DataLoader
from torchmetrics.image import (
    LearnedPerceptualImagePatchSimilarity,
    PeakSignalNoiseRatio,
)

VIEWER_AVAILABLE = (
    importlib.util.find_spec("nerfview") is not None
    and importlib.util.find_spec("viser") is not None
)

import numpy as np

RERUN_AVAILABLE = importlib.util.find_spec("rerun") is not None

import losses
from dataset import (
    CameraData,
    ColmapDataset,
    InstantNGPDataset,
    MatrixCityDataset,
    create_dataset,
)
from fused_ssim import fused_ssim  # type: ignore[import-untyped]
from gs_types import GS_LR_Schedulers, GSOptimizers, RenderParams
from logger import GaussianSplattingLogger, configure_app_logger
from model_factory import ModelFactory
from models import (
    BaseTrainableModel,
    NeuralRenderingMixin,
)
from ns_viewer import NSReplicaViewer, NSReplicaViewerConfig
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table
from train_args import DepthConfig, FloaterPreventionConfig, TrainConfig, parse_args
from utils import format_phase_description, seed_everything
from viewer_sync import ViewerParamSync

seed_everything(42)


torch.backends.cudnn.enabled = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True

# Allows dynamism on parms
torch._dynamo.config.force_parameter_static_shapes = False

ACTIVE_PROGRESS: Optional[Progress] = None


# ---------------------------------------------------------------------------
# Small utility helpers
# ---------------------------------------------------------------------------


def _build_viewmat(cam: dict, device: torch.device) -> torch.Tensor:
    """Construct a [1, 4, 4] view matrix from a camera dict."""
    viewmat = torch.eye(4, device=device, dtype=torch.float32)
    viewmat[:3, :3] = cam["R"]
    viewmat[:3, 3] = cam["T"]
    return viewmat.unsqueeze(0)


def _build_intrinsics(cam: dict, device: torch.device) -> torch.Tensor:
    """Construct a [3, 3] intrinsics matrix from a camera dict."""
    return torch.tensor(
        [[cam["fx"], 0.0, cam["cx"]],
         [0.0, cam["fy"], cam["cy"]],
         [0.0, 0.0, 1.0]],
        dtype=torch.float32, device=device,
    )

@torch.no_grad()
def _prepare_gt_image(
    gt_image: torch.Tensor,
    device: torch.device,
    target_h: Optional[int] = None,
    target_w: Optional[int] = None,
) -> torch.Tensor:
    """Move gt_image to device, normalize shape, and optionally resize to target HW.

    Returns tensor in [B, H, W, C] layout.
    """
    if not gt_image.is_cuda:
        gt_image = gt_image.to(device)
    if gt_image.dim() == 3:
        gt_image = gt_image.unsqueeze(0)

    if target_h is not None and target_w is not None:
        h, w = gt_image.shape[1], gt_image.shape[2]
        if h != target_h or w != target_w:
            gt_bchw = gt_image.permute(0, 3, 1, 2)
            gt_bchw = F.interpolate(
                gt_bchw,
                size=(target_h, target_w),
                mode="bilinear",
                align_corners=False,
            )
            gt_image = gt_bchw.permute(0, 2, 3, 1)
    return gt_image

@torch.no_grad()
def _prepare_depth_tensor(
    depth_tensor: Optional[torch.Tensor],
    device: torch.device,
    target_h: Optional[int] = None,
    target_w: Optional[int] = None,
) -> Optional[torch.Tensor]:
    """Move depth tensor to device, normalize dims, and optionally resize to target HW.

    Returns depth in [B, H, W] layout when available.
    """
    if depth_tensor is None:
        return None
    if not depth_tensor.is_cuda:
        depth_tensor = depth_tensor.to(device)

    if depth_tensor.dim() == 2:
        depth_tensor = depth_tensor.unsqueeze(0)
    elif depth_tensor.dim() == 4 and depth_tensor.shape[-1] == 1:
        depth_tensor = depth_tensor[..., 0]

    if target_h is not None and target_w is not None and depth_tensor.dim() == 3:
        h, w = depth_tensor.shape[-2], depth_tensor.shape[-1]
        if h != target_h or w != target_w:
            depth_bchw = depth_tensor.unsqueeze(1)
            depth_bchw = F.interpolate(
                depth_bchw,
                size=(target_h, target_w),
                mode="nearest",
            )
            depth_tensor = depth_bchw[:, 0]

    return depth_tensor


def _optimizer_has_any_grad(optimizer: torch.optim.Optimizer) -> bool:
    """Return True when at least one parameter in this optimizer has a gradient.

    Fast path: most optimizers in this codebase own a single large tensor
    (for example means/scales/quats/opacities/features). In that case we do
    a single grad check and avoid iterating parameter lists.
    """
    groups = optimizer.param_groups
    if len(groups) == 1:
        params = groups[0].get("params", [])
        if len(params) == 1:
            return params[0].grad is not None

    for group in groups:
        params = group.get("params", [])
        for param in params:
            if param.grad is not None:
                return True
    return False


def setup_logger(verbosity: int, output_dir: Path) -> logging.Logger:
    """Setup logging with RichHandler.

    Args:
        verbosity: 0=QUIET, 1=NORMAL, 2=VERBOSE, 3=DEBUG
        output_dir: Directory to save log file

    Returns:
        Configured logger instance
    """
    return configure_app_logger(
        verbosity=verbosity,
        output_dir=output_dir,
        log_filename="training.log",
        logger_name="cityscape_gs.train",
    )


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass
class RenderOutput:
    """Bundles all outputs from a single rasterization call."""
    render: torch.Tensor           # [B, H, W, 3]
    alpha: torch.Tensor            # [B, H, W, 1]
    depth_map: torch.Tensor        # [B, H, W]
    depth_mask: torch.Tensor       # [B, H, W]
    depth_mask_bchw: torch.Tensor  # [B, 1, H, W]
    depth_map_bchw: torch.Tensor   # [B, 1, H, W]
    render_perm: torch.Tensor      # [B, C, H, W]
    gt_perm: torch.Tensor          # [B, C, H, W]
    meta: dict = field(default_factory=dict)


@dataclass
class LossResult:
    """Bundles loss tensor and per-component metrics for logging."""
    total_loss: torch.Tensor
    metrics: dict  # str -> float, all individual loss/metric values
    inv_rendered_depth: Optional[torch.Tensor] = None
    inv_prior_depth: Optional[torch.Tensor] = None


# ---------------------------------------------------------------------------
# Rasterizer — encapsulates gsplat rasterization + depth extraction
# ---------------------------------------------------------------------------


class Rasterizer:
    """Encapsulates gsplat rasterization and post-processing.

    Handles the rasterization call, depth map extraction, alpha masking,
    and tensor permutations for loss computation.
    """

    def __init__(
        self,
        model: BaseTrainableModel,
        sh_cfg,
        packed: bool = False,
        absgrad: bool = False,
    ) -> None:
        self.model = model
        self.sh_cfg = sh_cfg
        self.packed = packed
        self.absgrad = absgrad

    def render(self, cam: dict[str, Any], gt_image: torch.Tensor,
               device: torch.device, lod: Optional[int] = None) -> RenderOutput:
        """Rasterize the scene for a single camera view.

        Args:
            cam: Camera dict with R, T, fx, fy, cx, cy, width, height.
            gt_image: Ground-truth image [B, H, W, C] (already on device).
            device: Target device.
            lod: Optional LoD level to render.

        Returns:
            RenderOutput with all tensors needed for loss and logging.
        """
        viewmat = _build_viewmat(cam, device)
        K = _build_intrinsics(cam, device)

        rp: RenderParams = self.model.get_render_params(cam, self.sh_cfg, is_training=True, lod=lod)

        render_output, render_alpha, render_meta = rasterization(
            means=rp.means,
            quats=rp.quats,
            scales=rp.scales,
            opacities=rp.opacities.squeeze(-1),
            colors=rp.colors,
            viewmats=viewmat,
            Ks=K[None, ...],
            width=cam["width"],
            height=cam["height"],
            sh_degree=rp.sh_degree,
            packed=self.packed,
            absgrad=self.absgrad,
            render_mode="RGB+ED",
            camera_model="pinhole"
        )

        # Forward any model-specific meta fields (e.g. Scaffold-GS neural_opacity).
        if rp.neural_opacity is not None:
            render_meta["neural_opacity"] = rp.neural_opacity
        if rp.selection_mask is not None:
            render_meta["selection_mask"] = rp.selection_mask

        render = render_output[..., 0:3]             # [B, H, W, 3]
        alpha = render_alpha                          # [B, H, W, 1]
        render_depth_raw = render_output[..., 3]      # [B, H, W]

        alpha_2d = alpha[..., 0] if alpha.dim() == 4 else alpha
        depth_map = render_depth_raw / (alpha_2d + 1e-6)

        depth_mask = (
            (alpha_2d > 0.5)
            & torch.isfinite(depth_map)
            & (depth_map > 0)
        )

        render_perm = render.permute(0, 3, 1, 2)     # [B, C, H, W]
        gt_perm = gt_image.permute(0, 3, 1, 2)

        return RenderOutput(
            render=render,
            alpha=alpha,
            depth_map=depth_map,
            depth_mask=depth_mask,
            depth_mask_bchw=depth_mask.unsqueeze(1),
            depth_map_bchw=depth_map.unsqueeze(1),
            render_perm=render_perm,
            gt_perm=gt_perm,
            meta=render_meta,
        )


# ---------------------------------------------------------------------------
# LossComputer — consolidates all loss modules and per-step computation
# ---------------------------------------------------------------------------


class LossComputer:
    """Consolidates all loss modules and their per-step computation.

    Loss modules are created once at init rather than scattered as free
    variables, and the conditional loss-accumulation logic is unified in
    a single ``compute()`` method.
    """

    def __init__(self, training_cfg, depth_cfg: DepthConfig, floater_cfg: FloaterPreventionConfig, device: torch.device,
                 verbosity: int = 1, logger: Optional[logging.Logger] = None) -> None:
        self.training_cfg = training_cfg
        self.depth_cfg = depth_cfg
        self.floater_cfg = floater_cfg
        self.device = device
        self.logger = logger
        self.scene_extent: float = 1.0

        # Quality metric
        self.psnr = PeakSignalNoiseRatio(data_range=(0, 1.0)).to(device)

        # LPIPS loss (optional)
        self.lpips = None
        if training_cfg.enable_lpips_loss:
            try:
                self.lpips = LearnedPerceptualImagePatchSimilarity(
                    net_type="vgg", normalize=True,
                ).to(device)
                if verbosity >= 1 and logger:
                    logger.info("[green]✓ LPIPS loss enabled[/green]")
            except Exception as e:
                if logger:
                    logger.error(f"[bold red]❌ Failed to initialize LPIPS:[/bold red] {e}")

        # Depth loss modules (created unconditionally, only used when weights > 0)
        self.depth_smoothness_loss = losses.FastPriorGradientMatchingLoss().to(device)
        self.affine_invariant_depth_loss = losses.FastAffineInvariantDepthLoss().to(device)
        self.pearson_correlation_loss = losses.PearsonCorrelationLoss().to(device)
        self.silog_depth_loss = losses.SILogLoss().to(device)
        self.ordinal_depth_loss = losses.OrdinalDepthLoss().to(device)
        self.affine_aligned_gradient_matching_loss = losses.AffineAlignedGradientMatchingLoss().to(device)
        self.metric_depth_normal_loss = losses.MetricNormalLoss().to(device)
        self.dn_splatter_normal_loss = losses.DNSplatterNormalLoss(
            lambda_weight=depth_cfg.dn_splatter_normal_loss_weight,
            tv_weight=depth_cfg.dn_splatter_normal_tv_weight,
        ).to(device)

    # -- internal helpers ---------------------------------------------------

    def _compute_lpips(self, render_perm: torch.Tensor,
                       gt_perm: torch.Tensor) -> torch.Tensor:
        """LPIPS with random-patch cropping for large images."""
        if self.lpips is None:
            return torch.tensor(0.0, device=self.device)

        h, w = render_perm.shape[2:]
        if h >= 1024 and w >= 1024:
            ps = 1024
            top = torch.randint(0, h - ps, (1,))
            left = torch.randint(0, w - ps, (1,))
            rp = render_perm[:, :, top:top + ps, left:left + ps].clamp(0.0, 1.0)
            gp = gt_perm[:, :, top:top + ps, left:left + ps].clamp(0.0, 1.0)
        else:
            rp, gp = render_perm, gt_perm
        return self.lpips(rp, gp)

    def _compute_depth_losses(
        self, depth_map_bchw: torch.Tensor,
        depth_tensor: Optional[torch.Tensor],
        depth_mask_bchw: torch.Tensor,
        cam_data: Optional[CameraData] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """Compute all active depth-related losses.

        Returns:
            (accumulated_loss, metrics_dict)
        """
        dcfg = self.depth_cfg
        # Use zeros_like to inherit dtype from render output (important for mixed precision)
        acc = torch.zeros(1, device=self.device, dtype=depth_map_bchw.dtype).squeeze()
        m: dict = {}

        if depth_tensor is None:
            return acc, m

        depth_tensor_bchw = depth_tensor.unsqueeze(0).unsqueeze(1).to(self.device)

        depth_loss_items = [
            ("affine_invariant_depth_loss", dcfg.enable_affine_invariant_depth_loss, dcfg.affine_invariant_depth_loss_weight,
             self.affine_invariant_depth_loss),
            ("pearson_correlation_loss", dcfg.enable_pearson_correlation_loss, dcfg.pearson_correlation_loss_weight,
             self.pearson_correlation_loss),
            ("silog_loss", dcfg.enable_silog_loss, dcfg.silog_loss_weight, self.silog_depth_loss),
            ("ordinal_depth_loss", dcfg.enable_ordinal_depth_loss, dcfg.ordinal_depth_loss_weight,
             self.ordinal_depth_loss),
            ("affine_aligned_gradient_matching_loss", dcfg.enable_affine_aligned_gradient_matching_loss,
             dcfg.affine_aligned_gradient_matching_loss_weight,
             self.affine_aligned_gradient_matching_loss),
            ("metric_depth_normal_loss", dcfg.enable_metric_depth_normal_loss, dcfg.metric_depth_normal_loss_weight,
             self.metric_depth_normal_loss),
            ("dn_splatter_normal_loss", dcfg.enable_dn_splatter_normal_loss, 1.0,
             self.dn_splatter_normal_loss),
        ]

        for name, enabled, weight, module in depth_loss_items:
            if enabled and weight > 0.0:
                try:
                    val = module(depth_map_bchw, depth_tensor_bchw, depth_mask_bchw)
                    acc = acc + weight * val
                    m[name] = val.item()
                except TypeError:
                    val = module(depth_map_bchw, depth_tensor_bchw, depth_mask_bchw, cam_data)
                    acc = acc + weight * val
                    m[name] = val.item()

        return acc, m

    def _compute_regularization(
        self, model: BaseTrainableModel, step: int,
    ) -> Tuple[torch.Tensor, dict]:
        """Compute floater-prevention regularization losses."""
        fcfg = self.floater_cfg
        # Use zeros with dtype from model parameters (important for mixed precision)
        acc = torch.zeros(1, device=self.device, dtype=model.opacities.dtype).squeeze()
        m: dict = {}

        if fcfg.enable_scale_reg:
            v = losses.scale_regularization(
                model.scales, weight=fcfg.scale_reg_weight,
                scene_extent=float(self.scene_extent),
            )
            acc = acc + v
            m["scale_reg_loss"] = v.item()
        else:
            m["scale_reg_loss"] = 0.0

        if fcfg.enable_opacity_reg:
            v = losses.opacity_regularization(model.opacities, weight=fcfg.opacity_reg_weight)
            acc = acc + v
            m["opacity_reg_loss"] = v.item()
        else:
            m["opacity_reg_loss"] = 0.0

        if (fcfg.enable_opacity_entropy_reg
                and step >= fcfg.opacity_entropy_reg_start_iter
                and step <= fcfg.opacity_entropy_reg_end_iter):
            v = losses.opacity_entropy_regularization(
                model.opacities, weight=fcfg.opacity_entropy_reg_weight,
            )
            acc = acc + v
            m["opacity_entropy_reg_loss"] = v.item()
        else:
            m["opacity_entropy_reg_loss"] = 0.0

        return acc, m

    # -- main entry-point ---------------------------------------------------

    def compute(
        self,
        render_out: RenderOutput,
        gt_image: torch.Tensor,
        depth_tensor: Optional[torch.Tensor],
        model: BaseTrainableModel,
        cam_data: CameraData,
        step: int,
    ) -> LossResult:
        """Compute all active losses for a single training step.

        Returns:
            LossResult with total loss tensor and per-component metrics dict.
        """
        dcfg = self.depth_cfg
        render_perm = render_out.render_perm
        gt_perm = render_out.gt_perm

        # Core photometric losses
        l1_loss = (render_out.render - gt_image).abs().mean()
        ssim_loss = 1.0 - fused_ssim(render_perm, gt_perm)
        lpips_loss = self._compute_lpips(render_perm, gt_perm)
        psnr_value = self.psnr(render_perm, gt_perm)

        loss = 0.8 * l1_loss + 0.2 * ssim_loss
        if self.lpips is not None:
            loss = loss + self.training_cfg.lpips_loss_weight * lpips_loss

        metrics = {
            "total_loss": 0.0,  # filled at end
            "l1_loss": l1_loss.item(),
            "ssim_loss": ssim_loss.item(),
            "lpips_loss": lpips_loss.item(),
            "psnr": psnr_value.item(),
        }

        # Depth losses
        depth_acc, depth_m = self._compute_depth_losses(
            render_out.depth_map_bchw, depth_tensor, render_out.depth_mask_bchw, cam_data
        )
        loss = loss + depth_acc
        metrics.update(depth_m)

        # Legacy depth loss (pearson + silog combined)
        inv_rendered_depth = None
        inv_prior_depth = None
        depth_corr = torch.tensor(0.0, device=self.device)
        d_loss = torch.tensor(0.0, device=self.device)

        if dcfg.enable_depth_loss and depth_tensor is not None and step >= dcfg.depth_loss_start_iter:
            if not depth_tensor.is_cuda:
                depth_tensor = depth_tensor.to(self.device, non_blocking=True)
            render_depth = torch.where(
                torch.isfinite(render_out.depth_map),
                render_out.depth_map,
                torch.zeros_like(render_out.depth_map),
            )
            dt = depth_tensor
            if dt.dim() == 2:
                dt = dt.unsqueeze(0)
            elif dt.dim() == 4 and dt.shape[-1] == 1:
                dt = dt[..., 0]
            try:
                current_depth_corr = depth_corr.mean().item() if torch.isfinite(depth_corr).all() else 0.0
                inv_rendered_depth = render_depth.detach()
                inv_prior_depth = dt.detach()
            except RuntimeError:
                current_depth_corr = 0.0
        else:
            current_depth_corr = 0.0

        metrics["depth_loss"] = d_loss.item()
        metrics["depth_corr"] = current_depth_corr

        # SAM loss (gradient-domain detail preservation)
        if dcfg.sam_loss_weight > 0.0:
            sam_loss = losses.gradient_loss(render_out.render, gt_image)
            loss = loss + dcfg.sam_loss_weight * sam_loss
            metrics["sam_loss"] = sam_loss.item()
        else:
            metrics["sam_loss"] = 0.0

        # Regularization losses
        reg_acc, reg_m = self._compute_regularization(model, step)
        loss = loss + reg_acc
        metrics.update(reg_m)

        metrics["total_loss"] = loss.item()

        return LossResult(
            total_loss=loss,
            metrics=metrics,
            inv_rendered_depth=inv_rendered_depth,
            inv_prior_depth=inv_prior_depth,
        )


# ---------------------------------------------------------------------------
# TrainingLogger — consolidates Rich progress + async TensorBoard dispatching
# ---------------------------------------------------------------------------


class TrainingLogger:
    """Consolidates Rich progress bars and async TensorBoard logging.

    All interval checks and background-thread dispatching are internal,
    keeping the training loop clean.
    """

    def __init__(
        self,
        tb_logger: GaussianSplattingLogger,
        console: Console,
        logger: logging.Logger,
        verbosity: int,
        total_iterations: int,
        start_iteration: int,
        log_interval: int,
        tb_image_interval: int,
        tb_histogram_interval: int,
        tensorboard_enabled: bool,
        densify_from: int,
        densify_until: int,
    ) -> None:
        self.tb_logger = tb_logger
        self.console = console
        self.logger = logger
        self.verbosity = verbosity
        self.total_iterations = total_iterations
        self.log_interval = log_interval
        self.tb_image_interval = tb_image_interval
        self.tb_histogram_interval = tb_histogram_interval
        self.tensorboard_enabled = tensorboard_enabled
        self.densify_from = densify_from
        self.densify_until = densify_until

        # Background executor for async logging
        self.executor = ThreadPoolExecutor(max_workers=16)

        # Rich progress bar
        self.progress: Optional[Progress] = None
        self.task_id: Optional[TaskID] = None
        global ACTIVE_PROGRESS
        if verbosity >= 1:
            self.progress = Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
                console=console,
                expand=False,
            )
            self.progress.start()
            ACTIVE_PROGRESS = self.progress
            self.task_id = self.progress.add_task(
                "[cyan]Training...", total=total_iterations,
                completed=start_iteration,
            )

    def _build_losses_dict(self, metrics: dict, step: int) -> dict:
        """Build the losses dict expected by tb_logger.log_losses."""
        d = {
            "total_loss": metrics["total_loss"],
            "l1_loss": metrics["l1_loss"],
            "ssim_loss": metrics["ssim_loss"],
            "lpips_loss": metrics["lpips_loss"],
            "scale_reg_loss": metrics["scale_reg_loss"],
            "opacity_reg_loss": metrics["opacity_reg_loss"],
            "opacity_entropy_reg_loss": metrics["opacity_entropy_reg_loss"],
        }
        # Only include optional depth keys when active (non-zero).
        # dn_splatter_normal_loss is always forwarded when present so TensorBoard
        # shows 0.0 during early training rather than an absent series.
        optional_keys = [
            "depth_loss", "depth_corr", "depth_smoothness_loss",
            "affine_invariant_depth_loss", "pearson_correlation_loss",
            "silog_loss", "ordinal_depth_loss",
            "affine_aligned_gradient_matching_loss", "metric_depth_normal_loss",
            "sam_loss", "dn_splatter_normal_loss",
        ]
        for k in optional_keys:
            if k in metrics and metrics[k] != 0.0:
                d[k] = metrics[k]
        return d

    def log_step(
        self,
        step: int,
        loss_result: LossResult,
        model: BaseTrainableModel,
        render_out: RenderOutput,
        gt_image: torch.Tensor,
        depth_tensor: Optional[torch.Tensor],
    ) -> None:
        """Handle progress-bar update + periodic TB metric/image/histogram logging."""
        m = loss_result.metrics

        # Update progress bar every iteration
        if self.progress is not None:
            if self.task_id is None:
                raise RuntimeError("Training progress task is not initialized")
            num_gaussians = len(model.means)
            phase = ("Densification"
                     if self.densify_from <= step <= self.densify_until
                     else "Refinement")
            desc = format_phase_description(
                step, phase, m["total_loss"], m["l1_loss"], m["ssim_loss"],
                m["lpips_loss"], m["psnr"], m["scale_reg_loss"],
                num_gaussians,
                count_label=model.count_label,
            )
            self.progress.update(self.task_id, advance=1, description=desc)

        # Periodic TensorBoard metric logging
        if step % self.log_interval == 0 or step == self.total_iterations - 1:
            if self.tensorboard_enabled:
                losses_dict = self._build_losses_dict(m, step)
                losses_snapshot = dict(losses_dict)
                max_radii_snapshot = render_out.meta.get("radii", None)
                if isinstance(max_radii_snapshot, torch.Tensor):
                    max_radii_snapshot = max_radii_snapshot.detach().clone()

                self.executor.submit(
                    _log_interval_metrics_async,
                    tb_logger=self.tb_logger,
                    losses_snapshot=losses_snapshot,
                    psnr_snapshot=float(m["psnr"]),
                    ssim_snapshot=float(m["ssim_loss"]),
                    lpips_snapshot=float(m["lpips_loss"]),
                    num_gaussians_snapshot=int(len(model.means)),
                    max_radii_snapshot=max_radii_snapshot,
                    step_snapshot=int(step),
                )
                self.tb_logger.log_system_metrics(step=step)
                # Keep event files up-to-date even if training stops early.
                self.tb_logger.flush()

        # Periodic image logging
        if (self.tensorboard_enabled
                and step % self.tb_image_interval == 0):
            self.executor.submit(
                self.tb_logger.log_images,
                rendered=render_out.render[0].detach(),
                ground_truth=gt_image[0].detach(),
                alpha=(render_out.alpha[0].detach()
                       if render_out.alpha.dim() >= 3
                       else render_out.alpha.detach()),
                rendered_depth_map=render_out.depth_map[0].detach(),
                inv_rendered_depth=(loss_result.inv_rendered_depth[0]
                                    if loss_result.inv_rendered_depth is not None
                                    else None),
                inv_prior_depth=(depth_tensor.squeeze(0).detach()
                                 if depth_tensor is not None else None),
                step=step,
            )

        # Periodic histogram logging
        if (self.tensorboard_enabled
                and step > 0
                and step % self.tb_histogram_interval == 0):
            self.tb_logger.log_gaussian_histograms(model, step=step)

    def log_densification(self, before: int, after: int, step: int) -> None:
        if self.tensorboard_enabled:
            self.tb_logger.log_densification_event(before, after, step=step)

    def log_learning_rates(self, optimizers: GSOptimizers, step: int, extra_optimizers=None) -> None:
        if self.tensorboard_enabled and step % self.log_interval == 0:
            self.tb_logger.log_learning_rates(optimizers, step=step, extra_optimizers=extra_optimizers)

    def finalize(self, final_metrics: dict) -> None:
        """Stop progress bar, shutdown executor, close TensorBoard."""
        global ACTIVE_PROGRESS
        if self.progress is not None:
            self.progress.stop()
            ACTIVE_PROGRESS = None

        if self.verbosity >= 1:
            self.logger.info("[dim]⌛ Waiting for background logging tasks to finish...[/dim]")
        self.executor.shutdown(wait=True)

        if self.tensorboard_enabled:
            self.tb_logger.log_hyperparameters({}, final_metrics)
            self.tb_logger.close()
            if self.verbosity >= 1:
                self.logger.info(
                    f"[green]✓ TensorBoard logs saved:[/green] "
                    f"[cyan]{self.tb_logger.run_name}[/cyan]"
                )


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
    """Log per-interval TensorBoard metrics in a background thread."""
    tb_logger.log_losses(**losses_snapshot, step=step_snapshot)
    tb_logger.log_quality_metrics(
        psnr=psnr_snapshot, ssim_loss=ssim_snapshot,
        lpips=lpips_snapshot, step=step_snapshot,
    )
    tb_logger.log_model_stats(
        num_gaussians=num_gaussians_snapshot,
        max_radii=max_radii_snapshot, step=step_snapshot,
    )


# ---------------------------------------------------------------------------
# GaussianSplatTrainer — orchestrates the full training loop
# ---------------------------------------------------------------------------


class GaussianSplatTrainer:
    """Main training orchestrator for 3D Gaussian Splatting.

    Analogous to ``SemanticTrainer`` in ``train_semantics.py``: encapsulates
    model setup, strategy, data loading, viewer, and the training loop
    so that callers only need to call :meth:`run`.
    """

    def __init__(self, config: TrainConfig, device: torch.device) -> None:
        self.config = config
        self.device = device

        # Unpack config groups for convenience
        self.required_cfg = config.required
        self.output_cfg = config.output
        self.training_cfg = config.training
        self.densification_cfg = config.densification
        self.floater_cfg = config.floater_prevention
        self.sh_cfg = config.sh
        self.depth_cfg = config.depth
        self.lr_cfg = config.learning_rates
        self.runtime_cfg = config.runtime
        self.checkpoint_cfg = config.checkpoint
        self.viewer_cfg = config.viewer
        self.tensorboard_cfg = config.tensorboard

        self.console = Console()
        self.VERBOSITY = self.runtime_cfg.verbosity

        # Low VRAM optimizations: Initialize gradient scaler for mixed precision training
        # Note: TF32 is already enabled globally at module level (lines 107-110)
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.training_cfg.use_low_vram)

        # Will be initialized in setup methods
        self.model: Optional[BaseTrainableModel] = None
        self.dataset: Optional[ColmapDataset | InstantNGPDataset | MatrixCityDataset] = None
        self.logger: Optional[logging.Logger] = None
        self.tb_logger: Optional[GaussianSplattingLogger] = None
        self.training_logger: Optional[TrainingLogger] = None
        self.rasterizer: Optional[Rasterizer] = None
        self.loss_computer: Optional[LossComputer] = None
        self.optimizers: Optional[GSOptimizers] = None
        self.schedulers: Optional[GS_LR_Schedulers] = None
        self.strategy: Optional[Any] = None  # DefaultStrategy or ScaffoldStrategy
        self.strategy_state: Optional[dict[str, Any]] = None
        self.viewer: Optional[Any] = None
        self.server: Optional[Any] = None
        self.viewer_param_sync: Optional[ViewerParamSync] = None
        self.rerun_viewer: Optional[Any] = None
        self.start_iteration = 0
        self.scene_extent = 0.0

        self._dataloader: Optional[DataLoader] = None
        self._dataloader_iter: Optional[Any] = None

    # -- setup methods ------------------------------------------------------

    def _setup_logger(self) -> None:
        output_path = Path(self.output_cfg.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        self.logger = setup_logger(self.VERBOSITY, output_path)
        self.logger.info("[bold cyan]🚀 Starting 3D Gaussian Splatting training pipeline[/bold cyan]")
        
        # Log low VRAM mode status
        if self.training_cfg.use_low_vram:
            self.logger.info(
                "[yellow]⚡ Low VRAM mode enabled:[/yellow] Using mixed precision (FP16/AMP), "
                "aggressive cache clearing, and gradient scaling"
            )
        
        self.checkpoint_dir = output_path / "checkpoints"
        self.checkpoint_dir.mkdir(exist_ok=True)

    def _setup_dataset(self) -> None:
        if self.logger is None:
            raise RuntimeError("Logger must be initialized before dataset setup")

        if self.required_cfg.dataset_type == "colmap":
            self.dataset = create_dataset(
                dataset_type="colmap",
                colmap_path=self.required_cfg.colmap_path,
                image_dir=self.required_cfg.images_path,
                depth_dir=self.required_cfg.depths_path,
                device=self.device,
                image_scale=self.required_cfg.scale,
                scene_extent_margin=1.5,  # Add margin to scene extent to prevent aggressive pruning of boundary Gaussians
                require_depth=self.depth_cfg.enable_depth_loss,
            )
        elif self.required_cfg.dataset_type == "instant-ngp":
            self.dataset = create_dataset(
                dataset_type="instant-ngp",
                transforms_path=self.required_cfg.transforms_path,
                image_dir=self.required_cfg.images_path,
                depth_dir=self.required_cfg.depths_path,
                point_cloud_path=self.required_cfg.point_cloud_path,
                device=self.device,
                image_scale=self.required_cfg.scale,
                scene_extent_margin=1.5,
                require_depth=self.depth_cfg.enable_depth_loss,
            )
        elif self.required_cfg.dataset_type == "matrixcity":
            self.dataset = create_dataset(
                dataset_type="matrixcity",
                matrixcity_paths=self.required_cfg.matrixcity_paths,
                matrixcity_depth_paths=self.required_cfg.matrixcity_depth_paths,
                matrixcity_pointcloud_paths=self.required_cfg.matrixcity_pointcloud_paths,
                matrixcity_max_init_points=self.required_cfg.matrixcity_max_init_points,
                device=self.device,
                image_scale=self.required_cfg.scale,
                scene_extent_margin=1.5,
                require_depth=self.depth_cfg.enable_depth_loss,
            )
        else:
            raise ValueError(f"Unsupported dataset type: {self.required_cfg.dataset_type}")
        
        if self.training_cfg.preload:
            self.logger.info("[yellow]Preloading enabled:[/yellow] loading all images into RAM...")
            self.dataset.preload_all_data()
            
        self.scene_extent = self.dataset.scene_extent

    def _setup_model(self) -> None:
        if self.logger is None:
            raise RuntimeError("Logger must be initialized before model setup")
        if self.dataset is None:
            raise RuntimeError("Dataset must be initialized before model setup")

        checkpoint = None
        tb_resume_run_name = None
        tb_purge_step = None

        if self.checkpoint_cfg.resume_from is not None:
            checkpoint_path = Path(self.checkpoint_cfg.resume_from)
            if not checkpoint_path.exists():
                raise FileNotFoundError(f"Checkpoint file does not exist: {checkpoint_path}")
            self.logger.info(f"[cyan]📂 Loading checkpoint:[/cyan] {checkpoint_path.name}")

            scaffold_kwargs = asdict(self.config.model.scaffold) if self.config.model.model_type == "scaffold" else {}
            self.model, checkpoint = ModelFactory.resume(
                model_type=self.config.model.model_type,
                checkpoint_path=checkpoint_path,
                device=self.device,
                sh_degree=self.sh_cfg.sh_degree,
                console=self.console,
                **scaffold_kwargs,
            )
            self.model.set_appearance(len(self.dataset))

            self.model.to(self.device)
            self.start_iteration = checkpoint.get("iteration", 0) + 1
            tb_resume_run_name = checkpoint.get("tensorboard_run_name", None)
            if tb_resume_run_name is not None:
                tb_purge_step = self.start_iteration
            if self.VERBOSITY >= 1:
                num_pts = len(self.model.means)
                self.logger.info(f"[green]✓ Loaded model with {num_pts:,} {self.model.point_name}[/green]")
                self.logger.info(f"[green]  Resuming from iteration {checkpoint['iteration']:,}[/green]")
        else:
            scaffold_kwargs = asdict(self.config.model.scaffold) if self.config.model.model_type == "scaffold" else {}
            self.model = ModelFactory.create(
                model_type=self.config.model.model_type,
                init_points=self.dataset.init_points,
                init_colors=self.dataset.init_colors,
                sh_degree=self.sh_cfg.sh_degree,
                console=self.console,
                **scaffold_kwargs,
            ).to(self.device)
            self.model.set_appearance(len(self.dataset))

        self._checkpoint = checkpoint

        # Determine experiment name
        experiment_name = self.output_cfg.experiment_name
        if experiment_name is None and tb_resume_run_name is not None:
            # Use the checkpoint's run name if resuming and no explicit name
            experiment_name = tb_resume_run_name
        elif experiment_name is not None and tb_resume_run_name is not None:
            # Both provided: CLI takes precedence (warn if different)
            if experiment_name != tb_resume_run_name:
                if self.VERBOSITY >= 1:
                    self.logger.warning(
                        f"[yellow]⚡ Experiment name mismatch:[/yellow] "
                        f"CLI specifies '{experiment_name}' but checkpoint has '{tb_resume_run_name}'. "
                        f"Using CLI value."
                    )

        # Check for existing experiment directory
        output_path = Path(self.output_cfg.output_dir)
        tensorboard_dir = output_path / "tensorboard"

        if experiment_name is not None:
            experiment_dir = tensorboard_dir / experiment_name
            if experiment_dir.exists():
                if self.checkpoint_cfg.resume_from is None:
                    # Not resuming, but experiment exists → error
                    raise ValueError(
                        f"Experiment '{experiment_name}' already exists at {experiment_dir}. "
                        f"Use --resume-from to continue training, or choose a different experiment name."
                    )
                # Resuming: allow continuation (TensorBoard purge_step will handle it)
                if self.VERBOSITY >= 1:
                    self.logger.info(
                        f"[cyan]📂 Resuming existing experiment:[/cyan] {experiment_name}"
                    )

        # TensorBoard logger
        self.tb_logger = GaussianSplattingLogger(
            log_dir=str(output_path / "tensorboard"),
            enabled=self.tensorboard_cfg.tensorboard,
            run_name=experiment_name,
            purge_step=tb_purge_step,
        )
        if self.tensorboard_cfg.tensorboard and self.VERBOSITY >= 1:
            if experiment_name is not None:
                self.logger.info(
                    f"[green]📊 TensorBoard:[/green] Resuming run [cyan]{self.tb_logger.run_name}[/cyan] "
                    f"from step [yellow]{self.start_iteration:,}[/yellow]"
                )
            else:
                self.logger.info(f"[green]📊 TensorBoard:[/green] Logging to run [cyan]{self.tb_logger.run_name}[/cyan]")
            self.logger.info(f"[dim]Run: tensorboard --logdir={output_path / 'tensorboard'}[/dim]")
            self.logger.info(f"[dim]Run directory: {self.tb_logger.log_dir}[/dim]")

    def _setup_strategy(self) -> None:
        dcfg = self.densification_cfg

        if self.model is None:
            raise RuntimeError("Model must be initialized before strategy setup")
        if self.dataset is None:
            raise RuntimeError("Dataset must be initialized before strategy setup")

        # Use isinstance() for type-safe capability detection instead of string checks
        if isinstance(self.model, NeuralRenderingMixin):
            from strategy_scaffold import ScaffoldStrategy
            self.strategy = ScaffoldStrategy(
                model=self.model,
                densify_from_iter=dcfg.densify_from_iter,
                densify_until_iter=dcfg.densify_until_iter,
                densify_interval=dcfg.densify_interval,
                grad_threshold=dcfg.grad_threshold,
                prune_opa=dcfg.prune_opa,
                verbose=(self.VERBOSITY >= 2),
            )
            self.strategy_state = self.strategy.initialize_state()
            if self.VERBOSITY >= 1 and self.logger is not None:
                self.logger.info("[cyan]🏗️ Scaffold-GS:[/cyan] Using ScaffoldStrategy for anchor-based densification")
        else:
            self.strategy = DefaultStrategy(
                prune_opa=dcfg.prune_opa,
                grow_grad2d=dcfg.grow_grad2d,
                grow_scale3d=dcfg.grow_scale3d,
                grow_scale2d=dcfg.grow_scale2d,
                prune_scale3d=dcfg.prune_scale3d,
                prune_scale2d=dcfg.prune_scale2d,
                refine_start_iter=dcfg.densify_from_iter,
                refine_stop_iter=dcfg.densify_until_iter,
                refine_every=dcfg.densify_interval,
                reset_every=dcfg.opacity_reset_interval,
                pause_refine_after_reset=0,
                absgrad=self.config.densification.absgrad,
                revised_opacity=False,
                verbose=(self.VERBOSITY >= 2),
            )
            self.strategy_state = self.strategy.initialize_state(
                scene_scale=float(self.scene_extent),
            )

        # Auto-adjust max_screen_size for large scenes
        mss = dcfg.max_screen_size
        original_mss = mss
        auto_adjusted = False
        if mss == 20:
            if self.scene_extent > 300:
                mss = 200
                auto_adjusted = True
            elif self.scene_extent > 150:
                mss = 100
                auto_adjusted = True
            elif self.scene_extent > 75:
                mss = 50
                auto_adjusted = True
        self.max_screen_size = mss

        # Display init info
        if self.VERBOSITY >= 1:
            init_info = Table.grid(padding=(0, 2))
            init_info.add_column(style="cyan", justify="right")
            init_info.add_column(style="green")
            init_info.add_row("Initial Gaussians:", f"{len(self.model.means):,}")
            init_info.add_row("Scene Extent:", f"{self.scene_extent:.3f}")
            init_info.add_row("Training Images:", f"{len(self.dataset)}")
            init_info.add_row("Total Iterations:", f"{self.training_cfg.iterations:,}")
            init_info.add_row("Device:", str(self.device))
            sh_label = f"{self.sh_cfg.sh_degree} ({'Enabled' if not self.sh_cfg.disable_sh_rendering else 'DC only'})"
            init_info.add_row("SH Degree:", sh_label)
            if auto_adjusted:
                init_info.add_row(
                    "[blue] Auto-adjusted:[/blue]",
                    f"[blue]max_screen_size {original_mss} → {mss} (for large scene)[/blue]",
                )
            if self.scene_extent > 200 and mss < 50 and not auto_adjusted:
                init_info.add_row(
                    "[yellow]⚠ Warning:[/yellow]",
                    f"[yellow]Large scene + small max_screen_size ({mss}) may cause aggressive pruning[/yellow]",
                )
            panel = Panel(init_info, title="[bold blue]🚀 Model Initialized[/bold blue]",
                          border_style="blue", box=box.ROUNDED)
            self.console.print(panel)
            self.console.print()

    def _setup_optimizers(self) -> None:
        if self.model is None:
            raise RuntimeError("Model must be initialized before optimizer setup")
        if self.strategy is None:
            raise RuntimeError("Strategy must be initialized before optimizer setup")
        if self.logger is None:
            raise RuntimeError("Logger must be initialized before optimizer setup")

        lr = self.lr_cfg
        self.optimizers = self.model.create_optimizers(
            lr_means=lr.lr_means, lr_scales=lr.lr_scales,
            lr_quats=lr.lr_quats, lr_opacities=lr.lr_opacities,
            lr_sh=lr.lr_sh, lr_semantics=lr.lr_semantics,
            means_lr_multiplier=5.0,
        )
        self.schedulers = self.model.create_schedulers(
            self.optimizers,
            iterations=self.training_cfg.iterations,
        )
        if self.VERBOSITY >= 2:
            self.logger.debug("[cyan]📈 Optimizers & Schedulers:[/cyan] Adam + CosineAnnealingLR initialized")

        # Validate strategy <-> params <-> optimizers wiring once optimizers exist.
        self.strategy.check_sanity(
            self.model.get_params_dict(),
            self.model.get_optimizers_dict(self.optimizers),
        )

        # Restore optimizer states from checkpoint
        if self._checkpoint is not None and "optimizers_state_dict" in self._checkpoint:
            saved_states = self._checkpoint["optimizers_state_dict"]
            for name, opt in self.optimizers.all_optimizers():
                if name in saved_states:
                    opt.load_state_dict(saved_states[name])
            self.model.load_extra_optimizer_states(self._checkpoint)
            if self.VERBOSITY >= 1:
                self.logger.info("[green]✓ Restored optimizer states[/green]")

    def _setup_components(self) -> None:
        """Initialize Rasterizer, LossComputer, TrainingLogger."""
        if self.model is None:
            raise RuntimeError("Model must be initialized before component setup")
        if self.tb_logger is None:
            raise RuntimeError("TensorBoard logger must be initialized before component setup")
        if self.logger is None:
            raise RuntimeError("App logger must be initialized before component setup")
        
        if self.strategy is None:
            raise RuntimeError("Strategy must be initialized before component setup")

        self.rasterizer = Rasterizer(
            self.model,
            self.sh_cfg,
            packed=False,
            absgrad=self.config.densification.absgrad,
        )
        self.loss_computer = LossComputer(
            self.training_cfg, self.depth_cfg, self.floater_cfg,
            self.device, self.VERBOSITY, self.logger,
        )
        self.loss_computer.scene_extent = self.scene_extent

        self.training_logger = TrainingLogger(
            tb_logger=self.tb_logger,
            console=self.console,
            logger=self.logger,
            verbosity=self.VERBOSITY,
            total_iterations=self.training_cfg.iterations,
            start_iteration=self.start_iteration,
            log_interval=self.training_cfg.log_interval,
            tb_image_interval=self.tensorboard_cfg.tb_image_interval,
            tb_histogram_interval=self.tensorboard_cfg.tb_histogram_interval,
            tensorboard_enabled=self.tensorboard_cfg.tensorboard,
            densify_from=self.densification_cfg.densify_from_iter,
            densify_until=self.densification_cfg.densify_until_iter,
        )

    def _setup_viewer(self) -> None:
        if self.logger is None:
            raise RuntimeError("Logger must be initialized before viewer setup")
        if self.model is None:
            raise RuntimeError("Model must be initialized before viewer setup")

        if self.viewer_cfg.rerun_viewer:
            if not RERUN_AVAILABLE:
                self.logger.warning("[yellow]⚠ Warning:[/yellow] rerun-sdk not available. Ignoring --rerun-viewer flag.")
            else:
                try:
                    from rerun_viewer import RerunViewer
                except ImportError:
                    self.logger.warning("[yellow]⚠ Warning:[/yellow] rerun_viewer module not available. Ignoring --rerun-viewer flag.")
                    return

                self.rerun_viewer = RerunViewer(
                    model=self.model,
                    disable_sh_rendering=self.sh_cfg.disable_sh_rendering,
                    refresh_interval=self.viewer_cfg.viewer_refresh_interval,
                )
                self.rerun_viewer.init()
                if self.VERBOSITY >= 1:
                    self.logger.info("[green]📺 Rerun Viewer started.[/green]")

        if not self.viewer_cfg.viewer:
            return
        if not VIEWER_AVAILABLE:
            self.logger.warning("[yellow]⚠ Warning:[/yellow] nerfview not available.")
            return
        nerfview_mod = importlib.import_module("nerfview")
        viser_mod = importlib.import_module("viser")
        if self.dataset is None:
            raise RuntimeError("Dataset must be initialized before viewer setup")
        model = self.model
        self.viewer_param_sync = ViewerParamSync(
            model=model, device=self.device,
            disable_sh_rendering=self.sh_cfg.disable_sh_rendering,
            refresh_interval=self.viewer_cfg.viewer_refresh_interval,
        )
        self.server = viser_mod.ViserServer(port=self.viewer_cfg.viewer_port, verbose=False)
        server = self.server
        assert server is not None
        if self.viewer_cfg.viewer_backend == "nerfview":
            self.viewer = nerfview_mod.Viewer(
                server=server,
                render_fn=self.viewer_param_sync.render_fn,
                mode="training",
            )
        else:
            replica_cfg = NSReplicaViewerConfig(
                add_training_cameras=self.viewer_cfg.viewer_add_training_cameras,
                camera_frustum_scale=self.viewer_cfg.viewer_camera_frustum_scale,
                image_policy=self.viewer_cfg.viewer_image_policy,
                image_cache_size=self.viewer_cfg.viewer_image_cache_size,
                max_thumbnail_size=self.viewer_cfg.viewer_max_thumbnail_size,
            )
            self.viewer = NSReplicaViewer(
                server=server,
                render_fn=self.viewer_param_sync.render_fn,
                dataset=self.dataset,
                config=replica_cfg,
            )
        
        # Add LoD slider if levels > 1
        lod_offsets = cast(list[int], model.lod_offsets)
        if len(lod_offsets) > 1:
            self.lod_slider = server.gui.add_slider(
                "Display LoD",
                min=0,
                max=len(lod_offsets) - 1,
                step=1,
                initial_value=0,
            )
            @self.lod_slider.on_update
            def _(_) -> None:
                if self.viewer is not None:
                    self.viewer.render_tab_state.lod = self.lod_slider.value

        # Add anchor point cloud toggle for Scaffold-GS models
        if isinstance(model, NeuralRenderingMixin):
            self.show_anchors_checkbox = server.gui.add_checkbox(
                "Show Anchors",
                initial_value=False,
            )
            self._anchor_cloud_handle: Optional[Any] = None

            @self.show_anchors_checkbox.on_update
            def _(event) -> None:
                if self.viewer_param_sync is None or self.server is None:
                    return
                enabled = self.show_anchors_checkbox.value
                self.viewer_param_sync.show_anchors = enabled
                if enabled:
                    # Create point cloud visualization of anchors
                    pts = model.means.detach().cpu().numpy().astype(np.float32)
                    # Use light gray color for anchors
                    gray = np.full((len(pts), 3), 180, dtype=np.uint8)
                    self._anchor_cloud_handle = server.scene.add_point_cloud(
                        name="/scaffold/anchors",
                        points=pts,
                        colors=gray,
                        point_size=0.015,
                    )
                else:
                    # Remove the anchor point cloud when toggled off
                    if self._anchor_cloud_handle is not None:
                        self._anchor_cloud_handle.remove()
                        self._anchor_cloud_handle = None

        # Add Gaussian visibility toggle
        self.hide_gaussians_checkbox = server.gui.add_checkbox(
            "Hide Gaussians",
            initial_value=False,
        )

        @self.hide_gaussians_checkbox.on_update
        def _(event) -> None:
            if self.viewer_param_sync is not None:
                self.viewer_param_sync.hide_gaussians = self.hide_gaussians_checkbox.value

        if self.VERBOSITY >= 1:
            self.logger.info(
                f"[green]📺 Viewer started:[/green] http://localhost:{self.viewer_cfg.viewer_port} "
                f"[dim](backend={self.viewer_cfg.viewer_backend})[/dim]"
            )

    def _setup_dataloader(self) -> DataLoader:
        if self.dataset is None:
            raise RuntimeError("Dataset must be initialized before creating DataLoader")
        if self.logger is None:
            raise RuntimeError("Logger must be initialized before creating DataLoader")

        import multiprocessing
        max_workers = multiprocessing.cpu_count()
        num_workers = max(0, min(self.training_cfg.num_workers, max_workers))
        if num_workers > 0:
            if self.VERBOSITY >= 1:
                self.logger.info(
                    "[yellow]Preloading enabled:[/yellow] forcing DataLoader num_workers to 0 "
                    "to avoid duplicating preloaded RAM across worker processes."
                )
            num_workers = 0

        self._dataloader = DataLoader(
            self.dataset, batch_size=1, shuffle=True,
            num_workers=num_workers,
            collate_fn=self.dataset.collate_fn,
            persistent_workers=True if num_workers > 0 else False,
            prefetch_factor=4 if num_workers > 0 else None,
        )
        self._dataloader_iter = iter(self._dataloader)
        return self._dataloader

    def _log_hyperparameters(self) -> None:
        """Log all hyperparameters to TensorBoard."""
        if not self.tensorboard_cfg.tensorboard:
            return
        if self.tb_logger is None:
            raise RuntimeError("TensorBoard logger must be initialized before logging hyperparameters")

        dcfg = self.densification_cfg
        hparams = {
            "iterations": self.training_cfg.iterations,
            "save_interval": self.training_cfg.save_interval,
            "log_interval": self.training_cfg.log_interval,
            "lr_means": self.lr_cfg.lr_means,
            "lr_scales": self.lr_cfg.lr_scales,
            "lr_quats": self.lr_cfg.lr_quats,
            "lr_opacities": self.lr_cfg.lr_opacities,
            "lr_sh": self.lr_cfg.lr_sh,
            "densify_from_iter": dcfg.densify_from_iter,
            "densify_until_iter": dcfg.densify_until_iter,
            "densify_interval": dcfg.densify_interval,
            "grad_threshold": dcfg.grad_threshold,
            "prune_opa": dcfg.prune_opa,
            "grow_grad2d": dcfg.grow_grad2d,
            "grow_scale3d": dcfg.grow_scale3d,
            "grow_scale2d": dcfg.grow_scale2d,
            "prune_scale3d": dcfg.prune_scale3d,
            "prune_scale2d": dcfg.prune_scale2d,
            "max_screen_size": self.max_screen_size,
            "opacity_reset_interval": dcfg.opacity_reset_interval,
            "sh_degree": self.sh_cfg.sh_degree,
            "use_sh_rendering": not self.sh_cfg.disable_sh_rendering,
            "enable_depth_loss": self.depth_cfg.enable_depth_loss,
            "depth_loss_weight": self.depth_cfg.depth_loss_weight,
            "scene_extent": self.scene_extent,
            "enable_scale_reg": self.floater_cfg.enable_scale_reg,
            "enable_opacity_reg": self.floater_cfg.enable_opacity_reg,
            "tensorboard_enabled": self.tensorboard_cfg.tensorboard,
            "resume_from_checkpoint": self.checkpoint_cfg.resume_from is not None,
        }
        self.tb_logger.log_hyperparameters(hparams)

    def _log_initial_pointcloud(self) -> None:
        """Log initial scene point cloud to TensorBoard before training loop starts."""
        if not self.tensorboard_cfg.tensorboard:
            return
        if self.tb_logger is None or self.dataset is None:
            return

        # Use dataset initialization cloud for a stable pre-training scene preview.
        self.tb_logger.log_pointcloud_mesh(
            points=self.dataset.init_points,
            colors=self.dataset.init_colors,
            tag="Scene/InitialPointCloud",
            step=self.start_iteration,
        )
        self.tb_logger.flush()

    # -- per-step helpers ---------------------------------------------------

    def _next_batch(self) -> Tuple:
        if self._dataloader_iter is None or self._dataloader is None:
            raise RuntimeError("DataLoader is not initialized")

        try:
            return next(self._dataloader_iter)
        except StopIteration:
            self._dataloader_iter = iter(self._dataloader)
            return next(self._dataloader_iter)

    def _train_step(self, step: int) -> LossResult:
        """Execute a single training iteration."""
        if self._dataloader_iter is None:
            raise RuntimeError("DataLoader iterator is not initialized")
        if self.model is None:
            raise RuntimeError("Model is not initialized")
        if self.rasterizer is None:
            raise RuntimeError("Rasterizer is not initialized")
        if self.strategy is None or self.strategy_state is None:
            raise RuntimeError("Strategy is not initialized")
        if self.optimizers is None:
            raise RuntimeError("Optimizers are not initialized")
        if self.schedulers is None:
            raise RuntimeError("Schedulers are not initialized")
        if self.training_logger is None:
            raise RuntimeError("Training logger is not initialized")
        if self.logger is None:
            raise RuntimeError("Logger is not initialized")

        cam, gt_image, depth_tensor = self._next_batch()
        torch.cuda.synchronize()

        target_h = int(cam["height"])
        target_w = int(cam["width"])
        gt_image = _prepare_gt_image(
            gt_image.detach(),
            self.device,
            target_h=target_h,
            target_w=target_w,
        )
        # depth_tensor = _prepare_depth_tensor(
        #     depth_tensor.detach() if depth_tensor is not None else None,
        #     self.device,
        #     target_h=target_h,
        #     target_w=target_w,
        # )

        # Safety check
        if len(self.model.means) == 0:
            raise RuntimeError(
                f"Model has no Gaussians at iteration {step}. Training cannot continue."
            )

        # Render and compute loss (with optional mixed precision)
        with torch.cuda.amp.autocast(enabled=self.training_cfg.use_low_vram):
            render_out = self.rasterizer.render(cam, gt_image, self.device)

            if self.sh_cfg.disable_sh_rendering and step == self.start_iteration and self.VERBOSITY >= 1:
                dc_rgb = self.model.dc_rgb.detach()
                self.logger.debug(
                    "[cyan]DC-only RGB stats:[/cyan] "
                    f"min={dc_rgb.min().item():.4f}, "
                    f"max={dc_rgb.max().item():.4f}, "
                    f"mean={dc_rgb.mean().item():.4f}"
                )

            params = self.model.get_params_dict()
            optimizers_dict = self.model.get_optimizers_dict(self.optimizers)
            self.strategy.step_pre_backward(
                params, optimizers_dict, self.strategy_state, step, render_out.meta,
            )

            # Compute loss
            if self.model is None:
                raise RuntimeError("Model is not initialized before computing loss")
            if self.loss_computer is None:
                raise RuntimeError("LossComputer is not initialized before computing loss")
            loss_result = self.loss_computer.compute(
                render_out, gt_image, depth_tensor, self.model, cam, step,
            )

        # Validate loss (outside autocast for accurate checking)
        if not loss_result.total_loss.isfinite():
            m = loss_result.metrics
            self.logger.warning(
                f"[yellow]⚠ Loss is NaN at step {step}[/yellow]: "
                f"total={m['total_loss']:.4f}, l1={m['l1_loss']:.4f}, "
                f"ssim={m['ssim_loss']:.4f}"
            )
            
            # Print ALL loss components for debugging
            self.logger.error("[red]═══ NaN Loss Debug Info ═══[/red]")
            self.logger.error(f"[red]Step: {step}[/red]")
            self.logger.error("[red]All loss components:[/red]")
            for key, value in sorted(m.items()):
                is_finite = "✓" if isinstance(value, (int, float)) and torch.isfinite(torch.tensor(value)) else "✗ NaN/Inf"
                dtype_info = ""
                # Try to get dtype if it's a recent tensor
                if key == "total_loss" and hasattr(loss_result.total_loss, 'dtype'):
                    dtype_info = f" (dtype: {loss_result.total_loss.dtype})"
                self.logger.error(f"  {key:40s}: {value:12.6f} {is_finite}{dtype_info}")
            self.logger.error("[red]═══════════════════════════[/red]")
            
            if self.VERBOSITY >= 1 and self.training_logger.progress:
                self.training_logger.progress.stop()
            raise RuntimeError(f"Loss is NaN or too large at step {step}")

        # Backward pass (with gradient scaling for mixed precision)
        if self.training_cfg.use_low_vram:
            self.scaler.scale(loss_result.total_loss).backward()
        else:
            loss_result.total_loss.backward()

        # Strategy post-backward (densification/pruning) — unified for all model types.
        gaussians_before = len(self.model.means)
        self.strategy.step_post_backward(
            params=params, optimizers=optimizers_dict,
            state=self.strategy_state, step=step,
            info=render_out.meta, packed=self.rasterizer.packed,
        )
        # NOTE: Do NOT call update_params_from_dict here - it must happen after optimizer.step()
        # to avoid in-place operation errors during backward pass
        gaussians_after = len(params["means"])

        # Clear CUDA cache periodically (more aggressive in low VRAM mode)
        if self.training_cfg.use_low_vram:
            if step % 50 == 0:
                torch.cuda.empty_cache()
        else:
            if step % (self.densification_cfg.densify_interval * 1) == 0:
                torch.cuda.empty_cache()

        # Handle densification events
        dcfg = self.densification_cfg
        if step >= dcfg.densify_from_iter and step <= dcfg.densify_until_iter:
            if step % dcfg.densify_interval == 0:
                self.training_logger.log_densification(gaussians_before, gaussians_after, step)
                if gaussians_after == 0:
                    raise RuntimeError(f"All Gaussians removed at iteration {step}")

        # Logging
        self.training_logger.log_step(
            step, loss_result, self.model, render_out, gt_image, depth_tensor,
        )

        # Optimizer step (with gradient scaling for mixed precision)
        if self.training_cfg.use_low_vram:
            # With GradScaler and multiple optimizers, we need to:
            # 1. Step each optimizer with scaler (it unscales internally)
            # 2. Update scaler once at the end
            did_scaled_step = False
            for _name, opt in self.optimizers.all_optimizers():
                if _optimizer_has_any_grad(opt):
                    self.scaler.step(opt)
                    did_scaled_step = True
                opt.zero_grad(set_to_none=True)
            for _name, opt in self.model.iter_extra_optimizers():
                if _optimizer_has_any_grad(opt):
                    self.scaler.step(opt)
                    did_scaled_step = True
                opt.zero_grad(set_to_none=True)

            # Update scaler only if at least one scaled step happened.
            if did_scaled_step:
                self.scaler.update()
        else:
            for _name, opt in self.optimizers.all_optimizers():
                opt.step()
                opt.zero_grad(set_to_none=True)
            for _name, opt in self.model.iter_extra_optimizers():
                opt.step()
                opt.zero_grad(set_to_none=True)

        # LR schedulers
        for _name, sched in self.schedulers.all_schedulers():
            sched.step()
        for _name, sched in self.model.iter_extra_schedulers():
            sched.step()

        # Update model parameters from strategy-modified dict (after optimizer step)
        # This must happen AFTER optimizer.step() to avoid in-place operation errors
        self.model.update_params_from_dict(params)

        # Learning rate logging (includes MLP/extra optimizers for scaffold-type models)
        self.training_logger.log_learning_rates(self.optimizers, step, extra_optimizers=self.model.iter_extra_optimizers())

        # Viewer update
        if self.viewer is not None:
            if self.viewer_param_sync is not None:
                self.viewer_param_sync.refresh_if_needed(step)
            self.viewer.lock.release()
            step_time = time.perf_counter() - self._step_start_time
            num_rays = cam["width"] * cam["height"]
            self.viewer.render_tab_state.num_train_rays_per_sec = (
                num_rays / step_time if step_time > 0 else 0
            )
            self.viewer.update(step, num_rays)
            
        if self.rerun_viewer is not None:
            self.rerun_viewer.update(step)

        return loss_result

    def _save_checkpoint(self, step: int, loss: float) -> None:
        if self.model is None:
            raise RuntimeError("Model is not initialized")
        if self.optimizers is None:
            raise RuntimeError("Optimizers are not initialized")
        if self.logger is None:
            raise RuntimeError("Logger is not initialized")

        tb_run_name = None
        if self.tensorboard_cfg.tensorboard:
            if self.tb_logger is None:
                raise RuntimeError("TensorBoard logger is not initialized")
            tb_run_name = self.tb_logger.run_name

        checkpoint_path = self.checkpoint_dir / f"checkpoint_{step}.pt"
        # Prepare optimizer state dict (flattened with all_optimizers)
        optimizers_state = {name: opt.state_dict() for name, opt in self.optimizers.all_optimizers()}
        extra_optimizers_state = self.model.get_extra_optimizer_states()

        # Convert TrainConfig to dict for serialization
        train_config_dict = asdict(self.training_cfg)

        torch.save({
            "iteration": step,
            "model_state_dict": self.model.state_dict(),
            "optimizers_state_dict": optimizers_state,
            "extra_optimizers_state_dict": extra_optimizers_state,
            "loss": loss,
            "tensorboard_run_name": tb_run_name,
            "train_config": train_config_dict,
        }, checkpoint_path)
        if self.VERBOSITY >= 1:
            self.logger.info(f"[green]💾 Checkpoint saved:[/green] [dim]{checkpoint_path.name}[/dim]")

    # -- main entry-point ---------------------------------------------------

    def run(self) -> Tuple[BaseTrainableModel, object]:
        """Execute the full training pipeline.

        Returns:
            Tuple of (trained model, viewer or None).
        """
        # Setup
        self._setup_logger()
        self._setup_dataset()
        self._setup_model()
        self._setup_strategy()
        self._setup_optimizers()
        self._setup_components()
        self._setup_viewer()
        self._setup_dataloader()
        self._log_hyperparameters()
        self._log_initial_pointcloud()

        if self.model is None:
            raise RuntimeError("Model failed to initialize")
        if self.training_logger is None:
            raise RuntimeError("Training logger failed to initialize")
        if self.logger is None:
            raise RuntimeError("Logger failed to initialize")

        last_loss_result = None
        training_completed = False

        try:
            for step in range(self.start_iteration, self.training_cfg.iterations):
                # Handle viewer pause
                if self.viewer is not None:
                    while self.viewer.state == "paused":
                        time.sleep(0.01)
                    self.viewer.lock.acquire()

                self._step_start_time = time.perf_counter()
                last_loss_result = self._train_step(step)

                # Save checkpoints
                if step % self.training_cfg.save_interval == 0 and step > 0:
                    self._save_checkpoint(step, last_loss_result.metrics["total_loss"])

            training_completed = True
        finally:
            final_metrics = {}
            if last_loss_result is not None:
                final_metrics = {
                    "final/loss": last_loss_result.metrics["total_loss"],
                    "final/psnr": last_loss_result.metrics["psnr"],
                    "final/gaussians": len(self.model.means),
                }
            self.training_logger.finalize(final_metrics)

        # Viewer complete
        if self.viewer is not None and training_completed:
            self.viewer.complete()
            if self.VERBOSITY >= 1:
                self.logger.info("[green]✓ Training complete.[/green] Viewer remains active.")

        # Save final model
        output_path = Path(self.output_cfg.output_dir)
        
        # Compute LoD levels if requested before saving
        if self.config.lod.num_levels > 1:
            self.logger.info(f"[cyan]🏔️ Computing {self.config.lod.num_levels} LoD levels...[/cyan]")
            self.model.compute_lods(
                num_levels=self.config.lod.num_levels,
                factor=self.config.lod.reduction_factor,
                optimizers=self.optimizers
            )

        final_path = output_path / "model_final.pt"
        # Convert TrainConfig to dict for serialization
        train_config_dict = asdict(self.training_cfg)

        torch.save({
            "iteration": self.training_cfg.iterations,
            "model_state_dict": self.model.state_dict(),
            "scene_extent": self.scene_extent,
            "num_gaussians": len(self.model.means),
            "lod_offsets": self.model.lod_offsets,
            "tensorboard_run_name": (
                self.tb_logger.run_name if (self.tensorboard_cfg.tensorboard and self.tb_logger is not None) else None
            ),
            "train_config": train_config_dict,
        }, final_path)

        # Display summary
        if self.VERBOSITY >= 1 and last_loss_result is not None:
            m = last_loss_result.metrics
            self.console.print()
            summary_grid = Table.grid(padding=(0, 2))
            summary_grid.add_column(style="cyan", justify="right")
            summary_grid.add_column(style="green")
            summary_grid.add_row("Final Loss:", f"{m['total_loss']:.6f}")
            summary_grid.add_row("Final L1:", f"{m['l1_loss']:.6f}")
            summary_grid.add_row("Final SSIM:", f"{m['ssim_loss']:.6f}")
            summary_grid.add_row("Final Gaussians:", f"{len(self.model.means):,}")
            summary_grid.add_row("Model Saved:", str(final_path.name))
            panel = Panel(summary_grid, title="[bold green]✅ Training Complete![/bold green]",
                          border_style="green", box=box.ROUNDED)
            self.console.print(panel)

        if self.viewer is not None and self.VERBOSITY >= 1:
            self.logger.info("[blue]ℹ Viewer is still running.[/blue] Press Ctrl+C to exit.")

        return self.model, self.viewer


# ---------------------------------------------------------------------------
# Public API — backward-compatible wrapper
# ---------------------------------------------------------------------------


def train_pipeline(config: TrainConfig):
    """Train a 3D Gaussian Splatting model.

    This is a thin backward-compatible wrapper around
    :class:`GaussianSplatTrainer`, preserving the existing call-site
    signature used by ``__main__`` and ``test_training.py``.

    Args:
        config: Full training configuration.

    Returns:
        Tuple of (trained GaussianModel, viewer or None).
    """
    device = torch.device("cuda")
    trainer = GaussianSplatTrainer(config, device)
    return trainer.run()


# ---------------------------------------------------------------------------
# Standalone entry-point
# ---------------------------------------------------------------------------

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

        config_table.add_row("COLMAP Path", required_cfg.colmap_path)
        config_table.add_row("Dataset Type", required_cfg.dataset_type)
        config_table.add_row("Images Path", required_cfg.images_path)
        if required_cfg.dataset_type == "instant-ngp":
            config_table.add_row("Transforms Path", required_cfg.transforms_path)
        if required_cfg.dataset_type == "matrixcity":
            config_table.add_row("MatrixCity Blocks", str(len(required_cfg.matrixcity_paths or [])))
            config_table.add_row("MatrixCity Paths", ", ".join(required_cfg.matrixcity_paths or []))
            if required_cfg.matrixcity_depth_paths:
                config_table.add_row("MatrixCity Depth Paths", ", ".join(required_cfg.matrixcity_depth_paths))
        config_table.add_row("Output Directory", output_cfg.output_dir)
        config_table.add_row("Iterations", f"{training_cfg.iterations:,}")
        config_table.add_row("Densify From", f"{densification_cfg.densify_from_iter:,}")
        config_table.add_row("Densify Until", f"{densification_cfg.densify_until_iter:,}")
        config_table.add_row("Verbosity", ["QUIET", "NORMAL", "VERBOSE", "DEBUG"][runtime_cfg.verbosity])

        sh_status = "Disabled (DC only)" if sh_cfg.disable_sh_rendering else f"Degree {sh_cfg.sh_degree}"
        config_table.add_row("Spherical Harmonics", sh_status)

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

        floater_techniques = []
        if floater_cfg.enable_scale_reg:
            floater_techniques.append(f"Scale Reg (λ={floater_cfg.scale_reg_weight})")
        if floater_cfg.enable_opacity_reg:
            floater_techniques.append(f"Opacity Reg (λ={floater_cfg.opacity_reg_weight})")
        if floater_cfg.enable_opacity_entropy_reg:
            floater_techniques.append(f"Opacity Entropy Reg (λ={floater_cfg.opacity_entropy_reg_weight})")
        if floater_techniques:
            config_table.add_row("Floater Prevention", ", ".join(floater_techniques))

        if config.lod.num_levels > 1:
            lod_info = f"{config.lod.num_levels} levels (factor {config.lod.reduction_factor})"
            config_table.add_row("Level of Detail", lod_info)

        console.print(config_table)
        console.print()

    # Setup logger for main block
    logger = setup_logger(runtime_cfg.verbosity, Path(output_cfg.output_dir))

    # Train model
    try:
        result = train_pipeline(config)
    except KeyboardInterrupt:
        if ACTIVE_PROGRESS is not None:
            try:
                ACTIVE_PROGRESS.stop()
            except Exception:
                pass
        if runtime_cfg.verbosity >= 1:
            logger.info("[yellow]Training interrupted by user.[/yellow]")
        raise
    except Exception:
        if ACTIVE_PROGRESS is not None:
            try:
                ACTIVE_PROGRESS.stop()
            except Exception:
                pass
        raise

    # Handle return values
    if isinstance(result, tuple):
        model, viewer = result
    else:
        model = result
        viewer = None

    # Export to PLY if requested
    if export_cfg.export_ply:
        ply_path = Path(output_cfg.output_dir) / "final_gaussians.ply"
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
