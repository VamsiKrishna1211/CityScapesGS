"""Argument parsing for standalone semantic fine-tuning (train_semantics.py).

Shared dataset/training/loss flags are reused from train_args so every
flag is defined exactly once.  Only semantics-specific and standalone-specific
flags live here.
"""
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from train_args import (
    DENSIFICATION_GROUP,
    DEPTH_GROUP,
    FLOATER_PREVENTION_GROUP,
    LEARNING_RATE_GROUP,
    MODEL_GROUP,
    REQUIRED_GROUP,
    RUNTIME_GROUP,
    SH_GROUP,
    TENSORBOARD_GROUP,
    TRAINING_GROUP,
    VIEWER_GROUP,
    ArgSpec,
    DensificationConfig,
    DepthConfig,
    FloaterPreventionConfig,
    LearningRateConfig,
    ModelConfig,
    RequiredConfig,
    RuntimeConfig,
    SHConfig,
    TensorBoardConfig,
    TrainingConfig,
    ViewerConfig,
    _add_group_to_parser,
    _build_group_config,
)

# ---------------------------------------------------------------------------
# SemanticsConfig
# ---------------------------------------------------------------------------


@dataclass
class SemanticsConfig:
    semantics_path: Optional[Path]           # npy provider only
    semantics_dim: int                       # bottleneck / feature dim
    semantic_image_resolution: Optional[Tuple[int, int]]
    semantic_loss_weight: float
    semantic_finetune_iters: int
    semantic_provider: str                   # "npy" | "dino_encoded" | "runtime"
    semantic_model_path: Optional[Path]      # runtime provider only
    semantic_cache_enabled: bool
    dino_encoded_dir: Optional[Path] = None  # dino_encoded provider
    finetune_params: Optional[List[str]] = None  # which param groups to train


def _normalize_semantics_values(values: Dict[str, Any]) -> Dict[str, Any]:
    path_value = values.get("semantics_path")
    values["semantics_path"] = Path(path_value) if path_value is not None else None

    model_path_value = values.get("semantic_model_path")
    values["semantic_model_path"] = (
        Path(model_path_value) if model_path_value is not None else None
    )

    raw_resolution = values.get("semantic_image_resolution")
    if isinstance(raw_resolution, (list, tuple)):
        if len(raw_resolution) != 2:
            raise ValueError(
                "--semantic-image-resolution must be exactly two values: height width"
            )
        values["semantic_image_resolution"] = (
            int(raw_resolution[0]), int(raw_resolution[1])
        )

    if values.get("semantics_dim", 0) <= 0:
        raise ValueError("--semantics-dim must be a positive integer")
    if values.get("semantic_finetune_iters", 0) <= 0:
        raise ValueError("--semantic-finetune-iters must be a positive integer")

    return values


# ---------------------------------------------------------------------------
# Composed standalone config
# ---------------------------------------------------------------------------


@dataclass
class SemanticStandaloneConfig:
    # ── Standalone-only ───────────────────────────────────────────────────────
    checkpoint_path: Path
    output_dir: Path
    device: str
    # "auto" detects model type from checkpoint keys; "scaffold" overrides.
    model_type: str
    # ScaffoldModel per-anchor language-feature dim (internal representation)
    lang_feat_dim: int
    # Semantic learning rate (overrides learning_rates.lr_sh when set)
    lr_semantics: Optional[float]

    # ── Shared sub-configs (reused from train_args.py) ────────────────────────
    required: RequiredConfig
    semantics: SemanticsConfig
    training: TrainingConfig
    learning_rates: LearningRateConfig        # lr_sh lives here; also lr_means etc.
    depth: DepthConfig                        # full depth loss suite
    floater_prevention: FloaterPreventionConfig
    sh: SHConfig                              # needed for Rasterizer init
    tensorboard: TensorBoardConfig
    viewer: ViewerConfig
    runtime: RuntimeConfig
    model: ModelConfig                        # model architecture config (scaffold params)
    densification: DensificationConfig        # densification strategy


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


def parse_semantic_args() -> SemanticStandaloneConfig:
    parser = argparse.ArgumentParser(
        description="Standalone semantic fine-tuning for Gaussian Splatting",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Reuse shared group definitions verbatim ───────────────────────────────
    _add_group_to_parser(parser, REQUIRED_GROUP)
    _add_group_to_parser(parser, TRAINING_GROUP)
    # MODEL_GROUP uses "gaussian" | "scaffold", but semantics needs "auto" support
    # So we manually add model arguments from MODEL_GROUP with modified model-type
    mod = parser.add_argument_group(MODEL_GROUP.title)
    for spec in MODEL_GROUP.specs:
        if spec.dest == "model_type":
            # Override --model-type to support "auto" 
            mod.add_argument(*spec.flags, type=str, default="auto",
                           choices=("auto", "gaussian", "scaffold"),
                           help="'auto' detects from checkpoint keys (_anchor → scaffold)")
        else:
            spec.add_to_group(mod)
    _add_group_to_parser(parser, DENSIFICATION_GROUP)         # densification strategy
    _add_group_to_parser(parser, LEARNING_RATE_GROUP)         # includes --lr-sh
    _add_group_to_parser(parser, DEPTH_GROUP)
    _add_group_to_parser(parser, FLOATER_PREVENTION_GROUP)
    _add_group_to_parser(parser, SH_GROUP)
    _add_group_to_parser(parser, VIEWER_GROUP)
    _add_group_to_parser(parser, TENSORBOARD_GROUP)
    _add_group_to_parser(parser, RUNTIME_GROUP)

    # ── Semantics ─────────────────────────────────────────────────────────────
    sem = parser.add_argument_group("Semantic Fine-Tuning")
    sem.add_argument("--semantics-path", type=str, default=None,
                     help="Path to semantic targets directory (npy provider)")
    sem.add_argument("--semantics-dim", type=int, default=4,
                     help="Output/supervision dimensionality (must match DINO bottleneck, e.g. 4 or 32)")
    sem.add_argument("--semantic-image-resolution", type=int, nargs=2, default=None,
                     metavar=("HEIGHT", "WIDTH"),
                     help="Override rendered resolution for semantic supervision")
    sem.add_argument("--semantic-loss-weight", type=float, default=1.0,
                     help="Weight applied to the semantic supervision loss")
    sem.add_argument("--semantic-finetune-iters", type=int, default=2000,
                     help="Number of semantic fine-tuning iterations")
    sem.add_argument("--semantic-provider", type=str, default="dino_encoded",
                     choices=("npy", "runtime", "dino_encoded"),
                     help="Semantic supervision provider backend")
    sem.add_argument("--semantic-model-path", type=str, default=None,
                     help="TorchScript/PyTorch model path (runtime provider only)")
    sem.add_argument("--semantic-cache-enabled", action="store_true", default=False,
                     help="Cache semantic targets in RAM (speeds up repeated epochs)")
    sem.add_argument("--no-semantic-cache", dest="semantic_cache_enabled",
                     action="store_false", help="Disable in-memory semantic target caching")
    sem.add_argument("--dino-encoded-dir", type=Path, default=None,
                     help="Directory of compressed DINO .pt files "
                          "(required when --semantic-provider=dino_encoded)")
    sem.add_argument("--finetune-params", type=str, nargs="+",
                     default=["semantics"],
                     choices=("semantics", "anchor_features", "mlp_geo", "geometry", "appearance"),
                     help="Param groups to unfreeze during fine-tuning. "
                          "Adding geometry groups also enables photometric + depth losses. "
                          "Default: semantics only (original behaviour).")

    # ── Standalone options ────────────────────────────────────────────────────
    sa = parser.add_argument_group("Standalone Options")
    sa.add_argument("--checkpoint-path", type=Path, required=True,
                    help="Geometry-only ScaffoldModel checkpoint (.pt) to fine-tune")
    sa.add_argument("--output-dir", type=Path, required=True,
                    help="Output directory for the semantic checkpoint and logs")
    sa.add_argument("--device", type=str, default="cuda")

    # ── Scaffold language-feature params ──────────────────────────────────────
    scf = parser.add_argument_group("Scaffold Language Features")
    scf.add_argument("--lang-feat-dim", type=int, default=4,
                     help="Per-anchor internal language feature dimension "
                          "(can differ from --semantics-dim; features are projected to semantics_dim for supervision)")

    # ── Semantic learning rate override ───────────────────────────────────────
    lr = parser.add_argument_group("Semantic Learning Rate")
    lr.add_argument("--lr-semantics", type=float, default=None,
                    help="Learning rate for semantic parameters (defaults to --lr-sh)")

    flat = parser.parse_args()

    # ── Validation ────────────────────────────────────────────────────────────
    if flat.dataset_type == "colmap":
        if flat.colmap_path is None:
            parser.error("--colmap-path is required when --dataset-type=colmap")
        if flat.images_path is None:
            parser.error("--images-path is required when --dataset-type=colmap")
    elif flat.dataset_type == "matrixcity":
        if not flat.matrixcity_paths:
            parser.error("--matrixcity-path is required when --dataset-type=matrixcity")

    if flat.semantic_provider == "npy" and flat.semantics_path is None:
        parser.error("--semantics-path is required when --semantic-provider=npy")
    if flat.semantic_provider == "runtime" and flat.semantic_model_path is None:
        parser.error("--semantic-model-path is required when --semantic-provider=runtime")
    if flat.semantic_provider == "dino_encoded" and flat.dino_encoded_dir is None:
        parser.error("--dino-encoded-dir is required when --semantic-provider=dino_encoded")

    sem_values = _normalize_semantics_values({
        "semantics_path": flat.semantics_path,
        "semantic_model_path": flat.semantic_model_path,
        "semantic_image_resolution": flat.semantic_image_resolution,
        "semantics_dim": flat.semantics_dim,
        "semantic_finetune_iters": flat.semantic_finetune_iters,
        "semantic_provider": flat.semantic_provider,
        "semantic_cache_enabled": flat.semantic_cache_enabled,
        "semantic_loss_weight": flat.semantic_loss_weight,
        "dino_encoded_dir": flat.dino_encoded_dir,
        "finetune_params": flat.finetune_params,
    })

    return SemanticStandaloneConfig(
        checkpoint_path=flat.checkpoint_path,
        output_dir=flat.output_dir,
        device=flat.device,
        model_type=flat.model_type,
        lang_feat_dim=flat.lang_feat_dim,
        lr_semantics=flat.lr_semantics,
        required=_build_group_config(flat, REQUIRED_GROUP),
        semantics=SemanticsConfig(**sem_values),
        training=_build_group_config(flat, TRAINING_GROUP),
        learning_rates=_build_group_config(flat, LEARNING_RATE_GROUP),
        depth=_build_group_config(flat, DEPTH_GROUP),
        floater_prevention=_build_group_config(flat, FLOATER_PREVENTION_GROUP),
        sh=_build_group_config(flat, SH_GROUP),
        viewer=_build_group_config(flat, VIEWER_GROUP),
        tensorboard=_build_group_config(flat, TENSORBOARD_GROUP),
        runtime=_build_group_config(flat, RUNTIME_GROUP),
        model=_build_group_config(flat, MODEL_GROUP),
        densification=_build_group_config(flat, DENSIFICATION_GROUP),
    )
