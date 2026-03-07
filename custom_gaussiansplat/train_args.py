import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Generic, Optional, Tuple, Type, TypeVar, Union


@dataclass
class RequiredConfig:
    colmap_path: str
    images_path: str
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


@dataclass
class FloaterPreventionConfig:
    enable_scale_reg: bool
    scale_reg_weight: float
    enable_opacity_reg: bool
    opacity_reg_weight: float
    enable_opacity_entropy_reg: bool
    opacity_entropy_reg_weight: float


@dataclass
class SHConfig:
    sh_degree: int
    disable_sh_rendering: bool


@dataclass
class SemanticsConfig:
    train_semantics: bool
    semantics_path: Optional[Path]
    semantics_dim: int
    semantic_image_resolution: Union[Tuple[int, int], int, None]
    semantic_start_iter: int
    semantic_loss_weight: float


@dataclass
class DepthConfig:
    enable_depth_loss: bool
    depth_loss_weight: float
    depth_loss_start_iter: int
    sam_loss_weight: float
    affine_invariant_depth_loss_weight: float
    pearson_correlation_loss_weight: float
    silog_loss_weight: float
    ordinal_depth_loss_weight: float
    affine_aligned_gradient_matching_loss_weight: float
    enable_depth_smoothness_loss: bool
    depth_smoothness_start_alpha: float
    depth_smoothness_end_alpha: float
    depth_smoothness_max_steps: int # If not used, then it will default to total iterations
    depth_smoothness_loss_weight: float


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


@dataclass
class TensorBoardConfig:
    tensorboard: bool
    tb_image_interval: int
    tb_histogram_interval: int


@dataclass
class TrainConfig:
    raw: argparse.Namespace
    required: RequiredConfig
    output: OutputConfig
    training: TrainingConfig
    densification: DensificationConfig
    floater_prevention: FloaterPreventionConfig
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

    raw_resolution = values.get("semantic_image_resolution")
    if isinstance(raw_resolution, list):
        if len(raw_resolution) != 2:
            raise ValueError("--semantic-image-resolution must contain exactly two values: height width")
        values["semantic_image_resolution"] = (int(raw_resolution[0]), int(raw_resolution[1]))
    elif isinstance(raw_resolution, tuple):
        if len(raw_resolution) != 2:
            raise ValueError("--semantic-image-resolution must contain exactly two values: height width")
        values["semantic_image_resolution"] = (int(raw_resolution[0]), int(raw_resolution[1]))

    return values


REQUIRED_GROUP = ArgGroupDef(
    key="required",
    title="Required Arguments",
    config_cls=RequiredConfig,
    specs=(
        ArgSpec(
            flags=("--colmap-path",),
            dest="colmap_path",
            arg_type=str,
            required=True,
            help="Path to COLMAP sparse reconstruction directory (e.g., sparse/0)",
        ),
        ArgSpec(
            flags=("--images-path",),
            dest="images_path",
            arg_type=str,
            required=True,
            help="Path to training images directory",
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

TRAINING_GROUP = ArgGroupDef(
    key="training",
    title="Training Parameters",
    config_cls=TrainingConfig,
    specs=(
        ArgSpec(flags=("--iterations",), dest="iterations", arg_type=int, default=7000, help="Total number of training iterations"),
        ArgSpec(flags=("--save-interval",), dest="save_interval", arg_type=int, default=1000, help="Save checkpoint every N iterations"),
        ArgSpec(flags=("--log-interval",), dest="log_interval", arg_type=int, default=1, help="Log progress every N iterations"),
        ArgSpec(flags=("--num-workers",), dest="num_workers", arg_type=int, default=0, help="Number of worker threads for data loading"),
        ArgSpec(flags=("--preload",), dest="preload", action="store_true", help="Preload all images into RAM before training (can speed up training but requires more memory)")
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
        ArgSpec(flags=("--train-semantics",), dest="train_semantics", action="store_true", default=False, help="Whether to use training semantics for densification (default: False)"),
        ArgSpec(flags=("--semantics-path",), dest="semantics_path", arg_type=str, default=None, help="Path to training semantics file (e.g., output/semantics/semantics.pt)"),
        ArgSpec(flags=("--semantics-dim",), dest="semantics_dim", arg_type=int, default=3, help="Dimensionality of semantics features (default: 3)"),
        ArgSpec(flags=("--semantic-image-resolution",), dest="semantic_image_resolution", arg_type=int, nargs=2, default=(1080, 1620), help="Resolution to render semantic maps for training semantics: height width"),
        ArgSpec(flags=("--semantic-start-iter",), dest="semantic_start_iter", arg_type=int, default=20000, help="Start applying training semantics after this many iterations"),
        ArgSpec(flags=("--semantic-loss-weight",), dest="semantic_loss_weight", arg_type=float, default=1.0, help="Weight for semantic supervision loss (0.0 disables semantic loss)"),
    ),
)

DEPTH_GROUP = ArgGroupDef(
    key="depth",
    title="Depth Supervision Options",
    config_cls=DepthConfig,
    specs=(
        ArgSpec(flags=("--enable-depth-loss",), dest="enable_depth_loss", action="store_true", help="Enable depth supervision from Depth Anything V2 depth maps"),
        ArgSpec(flags=("--depth-loss-weight",), dest="depth_loss_weight", arg_type=float, default=0.1, help="Weight for depth loss (0.05-0.2 recommended)"),
        ArgSpec(flags=("--depth-loss-start-iter",), dest="depth_loss_start_iter", arg_type=int, default=1000, help="Start applying depth loss after this many iterations"),
        ArgSpec(flags=("--sam-loss-weight",), dest="sam_loss_weight", arg_type=float, default=0.0, help="Weight for sharpness-aware minimization loss in gradient space (0.0 disables)"),
        ArgSpec(flags=("--affine-invariant-depth-loss-weight",), dest="affine_invariant_depth_loss_weight", arg_type=float, default=0.0, help="Weight for AffineInvariantDepthLoss (0.0 disables)"),
        ArgSpec(flags=("--pearson-correlation-loss-weight",), dest="pearson_correlation_loss_weight", arg_type=float, default=0.0, help="Weight for PearsonCorrelationLoss module (0.0 disables)"),
        ArgSpec(flags=("--silog-loss-weight",), dest="silog_loss_weight", arg_type=float, default=0.0, help="Weight for SILogLoss (0.0 disables)"),
        ArgSpec(flags=("--ordinal-depth-loss-weight",), dest="ordinal_depth_loss_weight", arg_type=float, default=0.0, help="Weight for OrdinalDepthLoss (0.0 disables)"),
        ArgSpec(flags=("--affine-aligned-gradient-matching-loss-weight",), dest="affine_aligned_gradient_matching_loss_weight", arg_type=float, default=0.0, help="Weight for AffineAlignedGradientMatchingLoss (0.0 disables)"),
        ArgSpec(flags=("--enable-depth-smoothness-loss",), dest="enable_depth_smoothness_loss", action="store_true", help="Enable edge-aware depth smoothness loss to regularize depth maps and reduce noise"),
        ArgSpec(flags=("--depth-smoothness-start-alpha",), dest="depth_smoothness_start_alpha", arg_type=float, default=0.5, help="Starting alpha value for edge-aware depth smoothness loss (lower = more edge-sensitive)"),
        ArgSpec(flags=("--depth-smoothness-end-alpha",), dest="depth_smoothness_end_alpha", arg_type=float, default=2.5, help="Ending alpha value for edge-aware depth smoothness loss (higher = less edge-sensitive)"),
        ArgSpec(flags=("--depth-smoothness-max-steps",), dest="depth_smoothness_max_steps", arg_type=int, default=None, help="Number of steps over which to schedule alpha for depth smoothness loss (defaults to total iterations if not set)"),
        ArgSpec(flags=("--depth-smoothness-loss-weight",), dest="depth_smoothness_loss_weight", arg_type=float, default=0.1, help="Weight for depth smoothness loss (0.01-0.1 recommended)")
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
    TRAINING_GROUP,
    DENSIFICATION_GROUP,
    FLOATER_PREVENTION_GROUP,
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
        description="Train a 3D Gaussian Splatting model from COLMAP reconstruction",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    for group_def in ARG_GROUP_DEFS:
        _add_group_to_parser(parser, group_def)

    flat_args = parser.parse_args()

    return TrainConfig(
        raw=flat_args,
        required=_build_group_config(flat_args, REQUIRED_GROUP),
        output=_build_group_config(flat_args, OUTPUT_GROUP),
        training=_build_group_config(flat_args, TRAINING_GROUP),
        densification=_build_group_config(flat_args, DENSIFICATION_GROUP),
        floater_prevention=_build_group_config(flat_args, FLOATER_PREVENTION_GROUP),
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
