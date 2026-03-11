import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class SemanticStandaloneConfig:
    checkpoint_path: Path
    colmap_path: str
    images_path: str
    output_dir: Path
    scale: int
    semantic_provider: str
    semantics_path: Optional[Path]
    semantic_model_path: Optional[Path]
    semantics_dim: int
    semantic_loss_weight: float
    semantic_finetune_iters: int
    semantic_cache_enabled: bool
    semantic_image_resolution: Optional[tuple[int, int]]
    lr_sh: float
    lr_semantics: Optional[float]
    num_workers: int
    preload: bool
    log_interval: int
    tensorboard: bool
    device: str


def parse_semantic_args() -> SemanticStandaloneConfig:
    parser = argparse.ArgumentParser(
        description="Standalone semantic-only training for Gaussian Splatting",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint-path", type=Path, required=True, help="Base model checkpoint (.pt) path")
    parser.add_argument("--colmap-path", type=str, required=True, help="COLMAP sparse path")
    parser.add_argument("--images-path", type=str, required=True, help="Images directory")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory")
    parser.add_argument("-s", "--scale", type=int, default=1, help="Image scale factor")

    parser.add_argument("--semantic-provider", type=str, default="npy", choices=("npy", "runtime"))
    parser.add_argument("--semantics-path", type=Path, default=None, help="Semantic npy directory for npy provider")
    parser.add_argument("--semantic-model-path", type=Path, default=None, help="TorchScript/PyTorch model path for runtime provider")
    parser.add_argument("--semantics-dim", type=int, default=3)
    parser.add_argument("--semantic-loss-weight", type=float, default=1.0)
    parser.add_argument("--semantic-finetune-iters", type=int, default=2000)
    parser.add_argument("--semantic-cache-enabled", action="store_true", default=False)
    parser.add_argument("--no-semantic-cache", dest="semantic_cache_enabled", action="store_false")
    parser.add_argument("--semantic-image-resolution", type=int, nargs=2, default=(1080, 1620), help="height width")

    parser.add_argument("--lr-sh", type=float, default=0.0025)
    parser.add_argument("--lr-semantics", type=float, default=None)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--preload", action="store_true", default=False)
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--tensorboard", action="store_true", default=True)
    parser.add_argument("--no-tensorboard", dest="tensorboard", action="store_false")
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    if args.semantics_dim <= 0:
        raise ValueError("--semantics-dim must be positive")
    if args.semantic_finetune_iters <= 0:
        raise ValueError("--semantic-finetune-iters must be positive")
    if args.semantic_provider == "npy" and args.semantics_path is None:
        raise ValueError("--semantics-path is required when --semantic-provider=npy")
    if args.semantic_provider == "runtime" and args.semantic_model_path is None:
        raise ValueError("--semantic-model-path is required when --semantic-provider=runtime")

    resolution = tuple(args.semantic_image_resolution) if args.semantic_image_resolution is not None else None

    return SemanticStandaloneConfig(
        checkpoint_path=args.checkpoint_path,
        colmap_path=args.colmap_path,
        images_path=args.images_path,
        output_dir=args.output_dir,
        scale=args.scale,
        semantic_provider=args.semantic_provider,
        semantics_path=args.semantics_path,
        semantic_model_path=args.semantic_model_path,
        semantics_dim=args.semantics_dim,
        semantic_loss_weight=args.semantic_loss_weight,
        semantic_finetune_iters=args.semantic_finetune_iters,
        semantic_cache_enabled=args.semantic_cache_enabled,
        semantic_image_resolution=resolution,
        lr_sh=args.lr_sh,
        lr_semantics=args.lr_semantics,
        num_workers=args.num_workers,
        preload=args.preload,
        log_interval=args.log_interval,
        tensorboard=args.tensorboard,
        device=args.device,
    )
