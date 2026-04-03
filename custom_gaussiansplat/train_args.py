import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Generic, Optional, Tuple, Type, TypeVar, Union


@dataclass
class RequiredConfig:
    dataset_type: str
    colmap_path: Optional[str]
    transforms_path: Optional[str]
    images_path: Optional[str]
    depths_path: Optional[str]
    matrixcity_paths: Optional[list[str]]
    matrixcity_depth_paths: Optional[list[str]]
    matrixcity_pointcloud_paths: Optional[list[str]]
    matrixcity_max_init_points: int
    point_cloud_path: Optional[str]
    scale: int


@dataclass
class OutputConfig:
    output_dir: str


@dataclass
class TrainingConfig:
    iterations: int
    save_interval: int
    log_interval: int
    num_workers: int
    preload: bool
    enable_lpips_loss: bool
    lpips_loss_weight: float
    use_low_vram: bool
    lpips_model: str = "vgg"  # Could be made configurable if desired, but VGG is a good default for perceptual similarity


@dataclass
class ModelConfig:
    model_type: str  # "gaussian" or "scaffold"
    # Scaffold-GS specific
    feat_dim: int
    n_offsets: int
    voxel_size: float
    update_depth: int
    update_init_factor: int
    update_hierachy_factor: int
    use_feat_bank: bool
    appearance_dim: int
    add_opacity_dist: bool
    add_cov_dist: bool
    add_color_dist: bool


@dataclass
class DensificationConfig:
    densify_from_iter: int
    densify_until_iter: int
    densify_interval: int
    grad_threshold: float
    prune_opa: float
    grow_grad2d: float
    grow_scale3d: float
    grow_scale2d: float
    prune_scale3d: float
    prune_scale2d: float
    max_screen_size: int
    opacity_reset_interval: int
    opacity_reset_value: float
    absgrad: bool


@dataclass
class FloaterPreventionConfig:
    enable_scale_reg: bool
    scale_reg_weight: float
    enable_opacity_reg: bool
    opacity_reg_weight: float
    enable_opacity_entropy_reg: bool
    opacity_entropy_reg_weight: float
    opacity_entropy_reg_start_iter: int
    opacity_entropy_reg_end_iter: int


@dataclass
class SHConfig:
    sh_degree: int
    disable_sh_rendering: bool


@dataclass
class SemanticsConfig:
    train_semantics: bool
    semantics_path: Optional[Path]
    semantics_dim: int
    semantic_image_resolution: Optional[Tuple[int, int]]
    semantic_loss_weight: float
    semantic_finetune_iters: int
    semantic_provider: str
    semantic_model_path: Optional[Path]
    semantic_cache_enabled: bool


@dataclass
class DepthConfig:
    enable_depth_loss: bool
    depth_loss_weight: float
    depth_loss_start_iter: int
    sam_loss_weight: float
    enable_affine_invariant_depth_loss: bool
    affine_invariant_depth_loss_weight: float
    enable_pearson_correlation_loss: bool
    pearson_correlation_loss_weight: float
    enable_silog_loss: bool
    silog_loss_weight: float
    enable_ordinal_depth_loss: bool
    ordinal_depth_loss_weight: float
    enable_affine_aligned_gradient_matching_loss: bool
    affine_aligned_gradient_matching_loss_weight: float
    enable_depth_smoothness_loss: bool
    depth_smoothness_start_alpha: float
    depth_smoothness_end_alpha: float
    depth_smoothness_max_steps: int # If not used, then it will default to total iterations
    depth_smoothness_loss_weight: float
    enable_metric_depth_normal_loss: bool
    metric_depth_normal_loss_weight: float


@dataclass
class LearningRateConfig:
    lr_means: float
    lr_scales: float
    lr_quats: float
    lr_opacities: float
    lr_sh: float
    lr_semantics: Optional[float]


@dataclass
class ExportConfig:
    export_ply: bool


@dataclass
class RuntimeConfig:
    verbosity: int


@dataclass
class CheckpointConfig:
    resume_from: Optional[str]


@dataclass
class ViewerConfig:
    viewer: bool
    viewer_port: int
    viewer_refresh_interval: int = 100  # How often to refresh viewer with latest model params (in iterations)
    rerun_viewer: bool = False
    viewer_backend: str = "ns-replica"  # ns-replica | nerfview
    viewer_image_policy: str = "lazy"  # lazy | preload
    viewer_image_cache_size: int = 256
    viewer_max_thumbnail_size: int = 128
    viewer_add_training_cameras: bool = True
    viewer_camera_frustum_scale: float = 0.1


@dataclass
class TensorBoardConfig:
    tensorboard: bool
    tb_image_interval: int
    tb_histogram_interval: int


@dataclass
class LODConfig:
    num_levels: int
    reduction_factor: int


@dataclass
class TrainConfig:
    raw: argparse.Namespace
    required: RequiredConfig
    output: OutputConfig
    model: ModelConfig
    training: TrainingConfig
    densification: DensificationConfig
    floater_prevention: FloaterPreventionConfig
    lod: LODConfig
    sh: SHConfig
    semantics: SemanticsConfig
    depth: DepthConfig
    learning_rates: LearningRateConfig
    export: ExportConfig
    runtime: RuntimeConfig
    checkpoint: CheckpointConfig
    viewer: ViewerConfig
    tensorboard: TensorBoardConfig


T = TypeVar("T")


@dataclass(frozen=True)
class ArgSpec:
    flags: Tuple[str, ...]
    dest: str
    help: str
    arg_type: Optional[Type[Any]] = None
    default: Any = None
    required: bool = False
    action: Optional[str] = None
    choices: Optional[Tuple[Any, ...]] = None
    nargs: Optional[Union[int, str]] = None

    def add_to_group(self, group: argparse._ArgumentGroup) -> None:
        kwargs: Dict[str, Any] = {
            "dest": self.dest,
            "help": self.help,
        }
        if self.action is not None:
            kwargs["action"] = self.action
        elif self.arg_type is not None:
            kwargs["type"] = self.arg_type

        if self.default is not None or self.action in {"store_true", "store_false"}:
            kwargs["default"] = self.default
        if self.required:
            kwargs["required"] = True
        if self.choices is not None:
            kwargs["choices"] = self.choices
        if self.nargs is not None:
            kwargs["nargs"] = self.nargs

        group.add_argument(*self.flags, **kwargs)


@dataclass(frozen=True)
class ArgGroupDef(Generic[T]):
    key: str
    title: str
    config_cls: Type[T]
    specs: Tuple[ArgSpec, ...]
    transform: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None


def _add_group_to_parser(parser: argparse.ArgumentParser, group_def: ArgGroupDef[Any]) -> None:
    group = parser.add_argument_group(group_def.title)
    for spec in group_def.specs:
        spec.add_to_group(group)


def _build_group_config(flat_args: argparse.Namespace, group_def: ArgGroupDef[T]) -> T:
    dataclass_fields = getattr(group_def.config_cls, "__dataclass_fields__", {})
    values: Dict[str, Any] = {
        field_name: getattr(flat_args, field_name)
        for field_name in dataclass_fields.keys()
    }
    if group_def.transform is not None:
        values = group_def.transform(values)
    return group_def.config_cls(**values)


def _normalize_semantics_values(values: Dict[str, Any]) -> Dict[str, Any]:
    path_value = values.get("semantics_path")
    values["semantics_path"] = Path(path_value) if path_value is not None else None

    model_path_value = values.get("semantic_model_path")
    values["semantic_model_path"] = Path(model_path_value) if model_path_value is not None else None

    raw_resolution = values.get("semantic_image_resolution")
    if isinstance(raw_resolution, list):
        if len(raw_resolution) != 2:
            raise ValueError("--semantic-image-resolution must contain exactly two values: height width")
        values["semantic_image_resolution"] = (int(raw_resolution[0]), int(raw_resolution[1]))
    elif isinstance(raw_resolution, tuple):
        if len(raw_resolution) != 2:
            raise ValueError("--semantic-image-resolution must contain exactly two values: height width")
        values["semantic_image_resolution"] = (int(raw_resolution[0]), int(raw_resolution[1]))

    if values.get("semantics_dim", 0) <= 0:
        raise ValueError("--semantics-dim must be a positive integer")

    if values.get("semantic_finetune_iters", 0) <= 0:
        raise ValueError("--semantic-finetune-iters must be a positive integer")

    if values.get("train_semantics", False):
        provider = values.get("semantic_provider")
        if provider == "npy" and values.get("semantics_path") is None:
            raise ValueError("--semantics-path is required when --semantic-provider=npy and --train-semantics is enabled")
        if provider == "runtime":
            model_path = values.get("semantic_model_path")
            if model_path is None:
                raise ValueError("--semantic-model-path is required when --semantic-provider=runtime and --train-semantics is enabled")
            if not model_path.exists():
                raise ValueError(f"semantic model path does not exist: {model_path}")

    return values


def _normalize_required_values(values: Dict[str, Any]) -> Dict[str, Any]:
    dataset_type = str(values.get("dataset_type", "colmap")).strip().lower().replace("_", "-")
    if dataset_type not in {"colmap", "instant-ngp", "matrixcity"}:
        raise ValueError(f"Unsupported --dataset-type '{dataset_type}'. Supported: colmap, instant-ngp, matrixcity")
    values["dataset_type"] = dataset_type

    matrixcity_paths = values.get("matrixcity_paths")
    matrixcity_depth_paths = values.get("matrixcity_depth_paths")

    if matrixcity_paths is not None:
        values["matrixcity_paths"] = [str(path) for path in matrixcity_paths]
    if matrixcity_depth_paths is not None:
        values["matrixcity_depth_paths"] = [str(path) for path in matrixcity_depth_paths]

    if dataset_type == "matrixcity":
        if not values.get("matrixcity_paths"):
            raise ValueError(
                "--matrixcity-path must be provided at least once when --dataset-type=matrixcity"
            )
        if values.get("matrixcity_depth_paths") and (
            len(values["matrixcity_depth_paths"]) != len(values["matrixcity_paths"])
        ):
            raise ValueError(
                "--matrixcity-depth-path must be provided either zero times or exactly "
                "once per --matrixcity-path"
            )

    # Backward compatibility: allow using --colmap-path as transforms input for instant-ngp.
    if dataset_type == "instant-ngp" and values.get("transforms_path") is None and values.get("colmap_path") is not None:
        values["transforms_path"] = values["colmap_path"]

    return values


REQUIRED_GROUP = ArgGroupDef(
    key="required",
    title="Required Arguments",
    config_cls=RequiredConfig,
    transform=_normalize_required_values,
    specs=(
        ArgSpec(
            flags=("--dataset-type",),
            dest="dataset_type",
            arg_type=str,
            default="colmap",
            choices=("colmap", "instant-ngp", "matrixcity"),
            help="Dataset format to load",
        ),
        ArgSpec(
            flags=("--colmap-path",),
            dest="colmap_path",
            arg_type=str,
            required=False,
            help="Path to COLMAP sparse reconstruction directory (required for --dataset-type colmap)",
        ),
        ArgSpec(
            flags=("--transforms-path",),
            dest="transforms_path",
            arg_type=str,
            required=False,
            help="Path to Instant-NGP transform file or scene directory containing transform.json/transforms.json (required for --dataset-type instant-ngp)",
        ),
        ArgSpec(
            flags=("--matrixcity-path",),
            dest="matrixcity_paths",
            action="append",
            required=False,
            help="MatrixCity block root path. Repeat this flag for multiple blocks (for example: .../aerial/train/block_1 --matrixcity-path .../aerial/train/block_2).",
        ),
        ArgSpec(
            flags=("--images-path",),
            dest="images_path",
            arg_type=str,
            required=False,
            help="Path to training images directory (required for colmap, optional for instant-ngp if transforms folder has images/)",
        ),
        ArgSpec(
            flags=("--depths-path",),
            dest="depths_path",
            arg_type=str,
            default=None,
            help="Optional explicit depth directory override. Supports .npy and .exr depth maps.",
        ),
        ArgSpec(
            flags=("--matrixcity-depth-path",),
            dest="matrixcity_depth_paths",
            action="append",
            required=False,
            help="Optional MatrixCity depth root per block. Repeat in the same order as --matrixcity-path, or omit to auto-resolve per block.",
        ),
        ArgSpec(
            flags=("--matrixcity-pointcloud-path",),
            dest="matrixcity_pointcloud_paths",
            action="append",
            required=False,
            help="Optional MatrixCity point cloud root, and the cound need not match the number of blocks. If provided, the point cloud will be used",
        ),
        ArgSpec(
            flags=("--matrixcity-max-init-points",),
            dest="matrixcity_max_init_points",
            arg_type=int,
            default=300000,
            help="Maximum number of initialization points used from MatrixCity point clouds after merge (default: 300000).",
        ),
        ArgSpec(
            flags=("--point-cloud-path",),
            dest="point_cloud_path",
            arg_type=str,
            default=None,
            help="Optional point cloud for instant-ngp init (.npy/.pth/.pt/.ply). Shapes: [N,3] xyz or [N,6] xyzrgb, .pth dict {'xyz','rgb'}, or PLY vertex x/y/z with optional RGB.",
        ),
        ArgSpec(
            flags=("-s", "--scale"),
            dest="scale",
            arg_type=int,
            default=1,
            help="Image downscale factor to use (1=images, 2=images_2, 4=images_4, etc.)",
        ),
    ),
)

OUTPUT_GROUP = ArgGroupDef(
    key="output",
    title="Output Options",
    config_cls=OutputConfig,
    specs=(
        ArgSpec(
            flags=("--output-dir",),
            dest="output_dir",
            arg_type=str,
            default="./output",
            help="Output directory for checkpoints and results",
        ),
    ),
)

MODEL_GROUP = ArgGroupDef(
    key="model",
    title="Model Options",
    config_cls=ModelConfig,
    specs=(
        ArgSpec(flags=("--model-type",), dest="model_type", arg_type=str, default="gaussian", choices=("gaussian", "scaffold"), help="Model architecture to use"),
        ArgSpec(flags=("--feat-dim",), dest="feat_dim", arg_type=int, default=32, help="Scaffold-GS: anchor feature dimension"),
        ArgSpec(flags=("--n-offsets",), dest="n_offsets", arg_type=int, default=10, help="Scaffold-GS: number of Gaussians per anchor"),
        ArgSpec(flags=("--voxel-size",), dest="voxel_size", arg_type=float, default=0.01, help="Scaffold-GS: initial voxel size for anchors"),
        ArgSpec(flags=("--update-depth",), dest="update_depth", arg_type=int, default=3, help="Scaffold-GS: anchor growing depth"),
        ArgSpec(flags=("--update-init-factor",), dest="update_init_factor", arg_type=int, default=100, help="Scaffold-GS: anchor growing factor"),
        ArgSpec(flags=("--update-hierachy-factor",), dest="update_hierachy_factor", arg_type=int, default=4, help="Scaffold-GS: anchor growing hierarchy factor"),
        ArgSpec(flags=("--use-feat-bank",), dest="use_feat_bank", action="store_true", help="Scaffold-GS: use feature bank for view-dependency"),
        ArgSpec(flags=("--appearance-dim",), dest="appearance_dim", arg_type=int, default=32, help="Scaffold-GS: appearance embedding dimension"),
        ArgSpec(flags=("--add-opacity-dist",), dest="add_opacity_dist", action="store_true", help="Scaffold-GS: include distance in opacity MLP"),
        ArgSpec(flags=("--add-cov-dist",), dest="add_cov_dist", action="store_true", help="Scaffold-GS: include distance in covariance MLP"),
        ArgSpec(flags=("--add-color-dist",), dest="add_color_dist", action="store_true", help="Scaffold-GS: include distance in color MLP"),
    ),
)

TRAINING_GROUP = ArgGroupDef(
    key="training",
    title="Training Parameters",
    config_cls=TrainingConfig,
    specs=(
        ArgSpec(flags=("--iterations",), dest="iterations", arg_type=int, default=7000, help="Total number of training iterations"),
        ArgSpec(flags=("--save-interval",), dest="save_interval", arg_type=int, default=1000, help="Save checkpoint every N iterations"),
        ArgSpec(flags=("--log-interval",), dest="log_interval", arg_type=int, default=1, help="Log progress every N iterations"),
        ArgSpec(flags=("--num-workers",), dest="num_workers", arg_type=int, default=0, help="Number of worker threads for data loading"),
        ArgSpec(flags=("--preload",), dest="preload", action="store_true", help="Preload all images into RAM before training (can speed up training but requires more memory)"),
        ArgSpec(flags=("--use-low-vram",), dest="use_low_vram", action="store_true", default=False, help="Enable low VRAM optimizations: mixed precision training (FP16/AMP), aggressive cache clearing, and gradient scaling. Reduces memory usage by ~30-40%% with minimal impact on quality. Consider disabling --enable-lpips-loss for additional savings."),
        ArgSpec(flags=("--enable-lpips-loss",), dest="enable_lpips_loss", action="store_true", help="Enable LPIPS loss for perceptual similarity (requires additional dependencies and GPU memory)"),
        ArgSpec(flags=("--lpips-loss-weight",), dest="lpips_loss_weight", arg_type=float, default=0.2, help="Weight for LPIPS loss (0.01-0.4 recommended if enabled), default: 0.2"),
        ArgSpec(flags=("--lpips-model",), dest="lpips_model", arg_type=str, default="vgg", choices=("vgg", "alex", "squeeze"), help="Model to use for LPIPS loss (default: vgg)"),
    ),
)

DENSIFICATION_GROUP = ArgGroupDef(
    key="densification",
    title="Densification Parameters",
    config_cls=DensificationConfig,
    specs=(
        ArgSpec(flags=("--densify-from-iter",), dest="densify_from_iter", arg_type=int, default=500, help="Start densification at this iteration"),
        ArgSpec(flags=("--densify-until-iter",), dest="densify_until_iter", arg_type=int, default=15000, help="Stop densification at this iteration"),
        ArgSpec(flags=("--densify-interval",), dest="densify_interval", arg_type=int, default=100, help="Densify every N iterations"),
        ArgSpec(flags=("--grad-threshold",), dest="grad_threshold", arg_type=float, default=0.0002, help="Gradient threshold for densification"),
        ArgSpec(flags=("--prune-opa",), dest="prune_opa", arg_type=float, default=0.005, help="Opacity threshold for pruning low-opacity Gaussians"),
        ArgSpec(flags=("--grow-grad2d",), dest="grow_grad2d", arg_type=float, default=0.0001, help="2D gradient threshold for densification growth"),
        ArgSpec(flags=("--grow-scale3d",), dest="grow_scale3d", arg_type=float, default=0.05, help="3D scale threshold controlling split vs clone behavior"),
        ArgSpec(flags=("--grow-scale2d",), dest="grow_scale2d", arg_type=float, default=0.01, help="2D screen-space scale threshold for growth"),
        ArgSpec(flags=("--prune-scale3d",), dest="prune_scale3d", arg_type=float, default=0.02, help="3D scale threshold for pruning oversized Gaussians"),
        ArgSpec(flags=("--prune-scale2d",), dest="prune_scale2d", arg_type=float, default=0.20, help="2D screen-space scale threshold for pruning"),
        ArgSpec(flags=("--max-screen-size",), dest="max_screen_size", arg_type=int, default=5000, help="Maximum screen size in pixels for pruning (increase for large scenes, e.g., 100-200)"),
        ArgSpec(flags=("--opacity-reset-interval",), dest="opacity_reset_interval", arg_type=int, default=3000, help="Reset opacity every N iterations"),
        ArgSpec(flags=("--opacity-reset-value",), dest="opacity_reset_value", arg_type=float, default=0.01, help="Opacity value to reset to (0.01-0.1, lower = more aggressive floater removal)"),
        ArgSpec(flags=("--absgrad",), dest="absgrad", action="store_true", default=False, help="Use absolute gradients for densification decisions (standard Gaussian model only)"),
    ),
)

FLOATER_PREVENTION_GROUP = ArgGroupDef(
    key="floater_prevention",
    title="Floater Prevention Options",
    config_cls=FloaterPreventionConfig,
    specs=(
        ArgSpec(flags=("--enable-scale-reg",), dest="enable_scale_reg", action="store_true", help="Enable scale regularization loss to penalize large Gaussians"),
        ArgSpec(flags=("--scale-reg-weight",), dest="scale_reg_weight", arg_type=float, default=0.01, help="Weight for scale regularization loss (0.01-0.1)"),
        ArgSpec(flags=("--enable-opacity-reg",), dest="enable_opacity_reg", action="store_true", help="Enable opacity regularization loss to encourage sparsity and prevent floaters"),
        ArgSpec(flags=("--opacity-reg-weight",), dest="opacity_reg_weight", arg_type=float, default=0.0005, help="Weight for opacity regularization loss (0.0001-0.001)"),
        ArgSpec(flags=("--enable-opacity-entropy-reg",), dest="enable_opacity_entropy_reg", action="store_true", help="Enable entropy regularization on opacity to push alpha toward 0/1 and reduce ghost Gaussians"),
        ArgSpec(flags=("--opacity-entropy-reg-weight",), dest="opacity_entropy_reg_weight", arg_type=float, default=0.0001, help="Weight for opacity entropy regularization loss (typical: 1e-5 to 1e-3)"),
        ArgSpec(flags=("--opacity-entropy-reg-start-iter",), dest="opacity_entropy_reg_start_iter", arg_type=int, default=5000, help="Iteration to start opacity entropy regularization (default: 15000)"),
        ArgSpec(flags=("--opacity-entropy-reg-end-iter",), dest="opacity_entropy_reg_end_iter", arg_type=int, default=30000, help="Iteration to end opacity entropy regularization (default: 30000)"),
    ),
)

SH_GROUP = ArgGroupDef(
    key="sh",
    title="Spherical Harmonics Options",
    config_cls=SHConfig,
    specs=(
        ArgSpec(flags=("--sh-degree",), dest="sh_degree", arg_type=int, default=3, choices=(0, 1, 2, 3), help="Degree of spherical harmonics (0=DC only, 1-3=higher order). Default: 3"),
        ArgSpec(flags=("--disable-sh-rendering",), dest="disable_sh_rendering", action="store_true", help="Disable spherical harmonics during rendering (use DC component only for faster training)"),
    ),
)

SEMANTICS_GROUP = ArgGroupDef(
    key="semantics",
    title="Training Semantics",
    config_cls=SemanticsConfig,
    transform=_normalize_semantics_values,
    specs=(
        ArgSpec(flags=("--train-semantics",), dest="train_semantics", action="store_true", default=False, help="Enable post-training semantic fine-tuning stage"),
        ArgSpec(flags=("--semantics-path",), dest="semantics_path", arg_type=str, default=None, help="Path to semantic targets directory for npy provider"),
        ArgSpec(flags=("--semantics-dim",), dest="semantics_dim", arg_type=int, default=3, help="Dimensionality of semantic Gaussian features"),
        ArgSpec(flags=("--semantic-image-resolution",), dest="semantic_image_resolution", arg_type=int, nargs=2, default=(1080, 1620), help="Semantic supervision reference resolution: height width"),
        ArgSpec(flags=("--semantic-loss-weight",), dest="semantic_loss_weight", arg_type=float, default=1.0, help="Weight for semantic supervision loss during semantic fine-tuning"),
        ArgSpec(flags=("--semantic-finetune-iters",), dest="semantic_finetune_iters", arg_type=int, default=2000, help="Number of post-training semantic fine-tuning iterations"),
        ArgSpec(flags=("--semantic-provider",), dest="semantic_provider", arg_type=str, default="npy", choices=("npy", "runtime"), help="Semantic supervision provider backend"),
        ArgSpec(flags=("--semantic-model-path",), dest="semantic_model_path", arg_type=str, default=None, help="Path to TorchScript or PyTorch model for runtime semantic inference"),
        ArgSpec(flags=("--semantic-cache-enabled",), dest="semantic_cache_enabled", action="store_true", default=False, help="Enable in-memory semantic target caching for runtime provider"),
        ArgSpec(flags=("--no-semantic-cache",), dest="semantic_cache_enabled", action="store_false", help="Disable in-memory semantic target caching"),
    ),
)

DEPTH_GROUP = ArgGroupDef(
    key="depth",
    title="Depth Supervision Options",
    config_cls=DepthConfig,
    specs=(
        ArgSpec(flags=("--enable-depth-loss",), dest="enable_depth_loss", action="store_true", help="Enable depth supervision from Depth Anything V2 depth maps"),
        ArgSpec(flags=("--depth-loss-weight",), dest="depth_loss_weight", arg_type=float, default=0.0, help="Weight for depth loss (0.05-0.2 recommended)"),
        ArgSpec(flags=("--depth-loss-start-iter",), dest="depth_loss_start_iter", arg_type=int, default=1000, help="Start applying depth loss after this many iterations"),
        ArgSpec(flags=("--sam-loss-weight",), dest="sam_loss_weight", arg_type=float, default=0.0, help="Weight for sharpness-aware minimization loss in gradient space (0.0 disables)"),
        ArgSpec(flags=("--enable-affine-invariant-depth-loss",), dest="enable_affine_invariant_depth_loss", action="store_true", help="Enable affine-invariant depth loss"),
        ArgSpec(flags=("--affine-invariant-depth-loss-weight",), dest="affine_invariant_depth_loss_weight", arg_type=float, default=0.0, help="Weight for AffineInvariantDepthLoss (0.0 disables)"),
        ArgSpec(flags=("--enable-pearson-correlation-loss",), dest="enable_pearson_correlation_loss", action="store_true", help="Enable Pearson correlation depth loss"),
        ArgSpec(flags=("--pearson-correlation-loss-weight",), dest="pearson_correlation_loss_weight", arg_type=float, default=0.0, help="Weight for PearsonCorrelationLoss module (0.0 disables)"),
        ArgSpec(flags=("--enable-silog-loss",), dest="enable_silog_loss", action="store_true", help="Enable scale-invariant log depth loss"),
        ArgSpec(flags=("--silog-loss-weight",), dest="silog_loss_weight", arg_type=float, default=0.0, help="Weight for SILogLoss (0.0 disables)"),
        ArgSpec(flags=("--enable-ordinal-depth-loss",), dest="enable_ordinal_depth_loss", action="store_true", help="Enable ordinal depth ranking loss"),
        ArgSpec(flags=("--ordinal-depth-loss-weight",), dest="ordinal_depth_loss_weight", arg_type=float, default=0.0, help="Weight for OrdinalDepthLoss (0.0 disables)"),
        ArgSpec(flags=("--enable-affine-aligned-gradient-matching-loss",), dest="enable_affine_aligned_gradient_matching_loss", action="store_true", help="Enable affine-aligned gradient matching loss"),
        ArgSpec(flags=("--affine-aligned-gradient-matching-loss-weight",), dest="affine_aligned_gradient_matching_loss_weight", arg_type=float, default=0.0, help="Weight for AffineAlignedGradientMatchingLoss (0.0 disables)"),
        ArgSpec(flags=("--enable-depth-smoothness-loss",), dest="enable_depth_smoothness_loss", action="store_true", help="Enable edge-aware depth smoothness loss to regularize depth maps and reduce noise"),
        ArgSpec(flags=("--depth-smoothness-start-alpha",), dest="depth_smoothness_start_alpha", arg_type=float, default=0.5, help="Starting alpha value for edge-aware depth smoothness loss (lower = more edge-sensitive)"),
        ArgSpec(flags=("--depth-smoothness-end-alpha",), dest="depth_smoothness_end_alpha", arg_type=float, default=2.5, help="Ending alpha value for edge-aware depth smoothness loss (higher = less edge-sensitive)"),
        ArgSpec(flags=("--depth-smoothness-max-steps",), dest="depth_smoothness_max_steps", arg_type=int, default=None, help="Number of steps over which to schedule alpha for depth smoothness loss (defaults to total iterations if not set)"),
        ArgSpec(flags=("--depth-smoothness-loss-weight",), dest="depth_smoothness_loss_weight", arg_type=float, default=0.0, help="Weight for depth smoothness loss (0.01-0.1 recommended)"),
        ArgSpec(flags=("--enable-metric-depth-normal-loss",), dest="enable_metric_depth_normal_loss", action="store_true", help="Enable metric depth normal loss"),
        ArgSpec(flags=("--metric-depth-normal-loss-weight",), dest="metric_depth_normal_loss_weight", arg_type=float, default=0.1, help="Weight for metric depth normal loss (0.01-0.1 recommended)"),
    ),
)

LEARNING_RATE_GROUP = ArgGroupDef(
    key="learning_rates",
    title="Learning Rates",
    config_cls=LearningRateConfig,
    specs=(
        ArgSpec(flags=("--lr-means",), dest="lr_means", arg_type=float, default=0.00016, help="Base learning rate for Gaussian positions (multiplied by 5.0)"),
        ArgSpec(flags=("--lr-scales",), dest="lr_scales", arg_type=float, default=0.005, help="Learning rate for Gaussian scales"),
        ArgSpec(flags=("--lr-quats",), dest="lr_quats", arg_type=float, default=0.001, help="Learning rate for Gaussian rotations"),
        ArgSpec(flags=("--lr-opacities",), dest="lr_opacities", arg_type=float, default=0.05, help="Learning rate for Gaussian opacities"),
        ArgSpec(flags=("--lr-sh",), dest="lr_sh", arg_type=float, default=0.0025, help="Learning rate for spherical harmonics"),
        ArgSpec(flags=("--lr-semantics",), dest="lr_semantics", arg_type=float, default=None, help="Learning rate for semantic Gaussian features (defaults to --lr-sh)"),
    ),
)

EXPORT_GROUP = ArgGroupDef(
    key="export",
    title="Export Options",
    config_cls=ExportConfig,
    specs=(
        ArgSpec(flags=("--export-ply",), dest="export_ply", action="store_true", help="Export final model to PLY format"),
    ),
)

RUNTIME_GROUP = ArgGroupDef(
    key="runtime",
    title="Verbosity Options",
    config_cls=RuntimeConfig,
    specs=(
        ArgSpec(flags=("--verbosity",), dest="verbosity", arg_type=int, default=1, choices=(0, 1, 2, 3), help="Verbosity level: 0=QUIET, 1=NORMAL, 2=VERBOSE, 3=DEBUG"),
    ),
)

CHECKPOINT_GROUP = ArgGroupDef(
    key="checkpoint",
    title="Checkpoint Options",
    config_cls=CheckpointConfig,
    specs=(
        ArgSpec(flags=("--resume-from",), dest="resume_from", arg_type=str, default=None, help="Path to checkpoint file to resume training from (e.g., output/checkpoints/checkpoint_1000.pt)"),
    ),
)

VIEWER_GROUP = ArgGroupDef(
    key="viewer",
    title="Viewer Options",
    config_cls=ViewerConfig,
    specs=(
        ArgSpec(flags=("--viewer",), dest="viewer", action="store_true", help="Enable interactive 3D viewer during training"),
        ArgSpec(flags=("--viewer-port",), dest="viewer_port", arg_type=int, default=8080, help="Port for the viewer server"),
        ArgSpec(flags=("--viewer-refresh-interval",), dest="viewer_refresh_interval", arg_type=int, default=100, help="How often to refresh viewer with latest model params (in iterations)"),
        ArgSpec(flags=("--rerun-viewer",), dest="rerun_viewer", action="store_true", help="Enable Rerun viewer during training"),
        ArgSpec(flags=("--viewer-backend",), dest="viewer_backend", arg_type=str, default="ns-replica", choices=("ns-replica", "nerfview"), help="Viewer backend implementation"),
        ArgSpec(flags=("--viewer-image-policy",), dest="viewer_image_policy", arg_type=str, default="lazy", choices=("lazy", "preload"), help="How viewer camera thumbnails are loaded"),
        ArgSpec(flags=("--viewer-image-cache-size",), dest="viewer_image_cache_size", arg_type=int, default=256, help="LRU cache size for lazy thumbnail loading"),
        ArgSpec(flags=("--viewer-max-thumbnail-size",), dest="viewer_max_thumbnail_size", arg_type=int, default=128, help="Max dimension for camera thumbnail images"),
        ArgSpec(flags=("--viewer-add-training-cameras",), dest="viewer_add_training_cameras", action="store_true", default=True, help="Display training cameras as frustums in viewer"),
        ArgSpec(flags=("--no-viewer-add-training-cameras",), dest="viewer_add_training_cameras", action="store_false", help="Disable training camera frustums in viewer"),
        ArgSpec(flags=("--viewer-camera-frustum-scale",), dest="viewer_camera_frustum_scale", arg_type=float, default=0.1, help="Scale factor for training camera frustums"),
    ),
)

LOD_GROUP = ArgGroupDef(
    key="lod",
    title="Level of Detail (LoD) Options",
    config_cls=LODConfig,
    specs=(
        ArgSpec(flags=("--num-lod-levels",), dest="num_levels", arg_type=int, default=1, help="Number of Level of Detail (LoD) levels to compute (1 = all, >1 enables LoD)"),
        ArgSpec(flags=("--lod-reduction-factor",), dest="reduction_factor", arg_type=int, default=4, help="Reduction factor between LoD levels (e.g., 4 means 4x fewer Gaussians per level)"),
    ),
)

TENSORBOARD_GROUP = ArgGroupDef(
    key="tensorboard",
    title="TensorBoard Options",
    config_cls=TensorBoardConfig,
    specs=(
        ArgSpec(flags=("--tensorboard",), dest="tensorboard", action="store_true", default=True, help="Enable TensorBoard logging (default: enabled)"),
        ArgSpec(flags=("--no-tensorboard",), dest="tensorboard", action="store_false", help="Disable TensorBoard logging"),
        ArgSpec(flags=("--tb-image-interval",), dest="tb_image_interval", arg_type=int, default=500, help="Log images to TensorBoard every N iterations"),
        ArgSpec(flags=("--tb-histogram-interval",), dest="tb_histogram_interval", arg_type=int, default=1000, help="Log histograms to TensorBoard every N iterations"),
    ),
)

ARG_GROUP_DEFS: Tuple[ArgGroupDef[Any], ...] = (
    REQUIRED_GROUP,
    OUTPUT_GROUP,
    MODEL_GROUP,
    TRAINING_GROUP,
    DENSIFICATION_GROUP,
    FLOATER_PREVENTION_GROUP,
    LOD_GROUP,
    SH_GROUP,
    SEMANTICS_GROUP,
    DEPTH_GROUP,
    LEARNING_RATE_GROUP,
    EXPORT_GROUP,
    RUNTIME_GROUP,
    CHECKPOINT_GROUP,
    VIEWER_GROUP,
    TENSORBOARD_GROUP,
)


def parse_args() -> TrainConfig:
    """Parse command line arguments and return strongly typed grouped config."""
    parser = argparse.ArgumentParser(
        description="Train a 3D Gaussian Splatting model from COLMAP, Instant-NGP, or MatrixCity style datasets",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    for group_def in ARG_GROUP_DEFS:
        _add_group_to_parser(parser, group_def)

    flat_args = parser.parse_args()

    required_cfg = _build_group_config(flat_args, REQUIRED_GROUP)
    if required_cfg.dataset_type == "colmap":
        if not required_cfg.colmap_path:
            parser.error("--colmap-path is required when --dataset-type=colmap")
        if not required_cfg.images_path:
            parser.error("--images-path is required when --dataset-type=colmap")
    elif required_cfg.dataset_type == "instant-ngp":
        if not required_cfg.transforms_path:
            parser.error("--transforms-path is required when --dataset-type=instant-ngp")
    elif required_cfg.dataset_type == "matrixcity":
        if not required_cfg.matrixcity_paths:
            parser.error("--matrixcity-path is required when --dataset-type=matrixcity")
        if required_cfg.matrixcity_depth_paths and (
            len(required_cfg.matrixcity_depth_paths) != len(required_cfg.matrixcity_paths)
        ):
            parser.error(
                "--matrixcity-depth-path must be provided either zero times or exactly "
                "once per --matrixcity-path"
            )

    return TrainConfig(
        raw=flat_args,
        required=required_cfg,
        output=_build_group_config(flat_args, OUTPUT_GROUP),
        model=_build_group_config(flat_args, MODEL_GROUP),
        training=_build_group_config(flat_args, TRAINING_GROUP),
        densification=_build_group_config(flat_args, DENSIFICATION_GROUP),
        floater_prevention=_build_group_config(flat_args, FLOATER_PREVENTION_GROUP),
        lod=_build_group_config(flat_args, LOD_GROUP),
        sh=_build_group_config(flat_args, SH_GROUP),
        semantics=_build_group_config(flat_args, SEMANTICS_GROUP),
        depth=_build_group_config(flat_args, DEPTH_GROUP),
        learning_rates=_build_group_config(flat_args, LEARNING_RATE_GROUP),
        export=_build_group_config(flat_args, EXPORT_GROUP),
        runtime=_build_group_config(flat_args, RUNTIME_GROUP),
        checkpoint=_build_group_config(flat_args, CHECKPOINT_GROUP),
        viewer=_build_group_config(flat_args, VIEWER_GROUP),
        tensorboard=_build_group_config(flat_args, TENSORBOARD_GROUP),
    )


__all__ = [
    "TrainConfig",
    "parse_args",
]
