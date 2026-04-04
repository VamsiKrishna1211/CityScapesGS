"""Standalone viser viewer for inspecting a saved Gaussian Splatting model.

Usage:
    python custom_gaussiansplat/view_model.py --checkpoint path/to/ckpt.pth
    python custom_gaussiansplat/view_model.py --checkpoint path/to/ckpt.pth --model-type scaffold
    python custom_gaussiansplat/view_model.py --checkpoint path/to/ckpt.pth --port 8081
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

import numpy as np
import torch

try:
    import nerfview
    import viser
    VIEWER_AVAILABLE = True
except ImportError:
    VIEWER_AVAILABLE = False

from model_factory import ModelFactory
from models import NeuralRenderingMixin
from viewer_sync import ViewerParamSync


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize a saved Gaussian Splatting checkpoint with viser.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to the saved model checkpoint (.pth file).",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="gaussian",
        choices=("gaussian", "scaffold"),
        help="Model type stored in the checkpoint.",
    )
    parser.add_argument(
        "--sh-degree",
        type=int,
        default=3,
        help="Spherical harmonics degree used during training.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Port for the viser web server.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to load the model on.",
    )
    parser.add_argument(
        "--disable-sh",
        action="store_true",
        help="Render using only DC (zeroth-order SH) colour instead of full SH.",
    )
    # Scaffold-GS specific kwargs forwarded to ModelFactory.resume
    parser.add_argument("--feat-dim", type=int, default=32)
    parser.add_argument("--n-offsets", type=int, default=10)
    parser.add_argument("--voxel-size", type=float, default=0.01)
    parser.add_argument("--update-depth", type=int, default=3)
    parser.add_argument("--update-init-factor", type=int, default=100)
    parser.add_argument("--update-hierachy-factor", type=int, default=4)
    parser.add_argument("--use-feat-bank", action="store_true")
    parser.add_argument("--appearance-dim", type=int, default=32)
    parser.add_argument("--add-opacity-dist", action="store_true")
    parser.add_argument("--add-cov-dist", action="store_true")
    parser.add_argument("--add-color-dist", action="store_true")
    return parser.parse_args()


def _scaffold_kwargs(args: argparse.Namespace) -> dict:
    return dict(
        feat_dim=args.feat_dim,
        n_offsets=args.n_offsets,
        voxel_size=args.voxel_size,
        update_depth=args.update_depth,
        update_init_factor=args.update_init_factor,
        update_hierachy_factor=args.update_hierachy_factor,
        use_feat_bank=args.use_feat_bank,
        appearance_dim=args.appearance_dim,
        add_opacity_dist=args.add_opacity_dist,
        add_cov_dist=args.add_cov_dist,
        add_color_dist=args.add_color_dist,
    )


def main() -> None:
    args = parse_args()

    if not VIEWER_AVAILABLE:
        print("Error: nerfview and/or viser are not installed. Install them with:")
        print("  pip install nerfview viser")
        sys.exit(1)

    checkpoint_path: Path = args.checkpoint
    if not checkpoint_path.exists():
        print(f"Error: checkpoint not found: {checkpoint_path}")
        sys.exit(1)

    device = torch.device(args.device)
    print(f"Loading {args.model_type} model from {checkpoint_path} on {device} ...")

    scaffold_kw = _scaffold_kwargs(args) if args.model_type == "scaffold" else {}
    model, _ = ModelFactory.resume(
        model_type=args.model_type,
        checkpoint_path=checkpoint_path,
        device=device,
        sh_degree=args.sh_degree,
        **scaffold_kw,
    )
    model.eval()

    n_gaussians = len(model.means)
    print(f"Loaded model: {n_gaussians:,} Gaussians")

    # Wire up the render callback (reuses the same sync helper as the trainer)
    viewer_sync = ViewerParamSync(
        model=model,
        device=device,
        disable_sh_rendering=args.disable_sh,
        refresh_interval=1,  # static model — no need to poll
    )

    server = viser.ViserServer(port=args.port, verbose=False)

    # "rendering" mode — no training-specific UI (pause/step/rays-per-sec)
    viewer = nerfview.Viewer(
        server=server,
        render_fn=viewer_sync.render_fn,
        mode="rendering",
    )

    # LoD slider (only shown for hierarchical GaussianModel with multiple levels)
    lod_offsets = getattr(model, "lod_offsets", [n_gaussians])
    if len(lod_offsets) > 1:
        lod_slider = server.gui.add_slider(
            "Display LoD",
            min=0,
            max=len(lod_offsets) - 1,
            step=1,
            initial_value=0,
        )
        viewer_sync.lod_slider = lod_slider

    # Anchor point cloud toggle for Scaffold-GS
    anchor_cloud_handle = None
    if isinstance(model, NeuralRenderingMixin):
        show_anchors_checkbox = server.gui.add_checkbox("Show Anchors", initial_value=False)

        @show_anchors_checkbox.on_update
        def _(event) -> None:
            nonlocal anchor_cloud_handle
            enabled = show_anchors_checkbox.value
            viewer_sync.show_anchors = enabled
            if enabled:
                pts = model.anchors.detach().cpu().numpy().astype(np.float32)
                gray = np.full((len(pts), 3), 180, dtype=np.uint8)
                anchor_cloud_handle = server.scene.add_point_cloud(
                    name="/scaffold/anchors",
                    points=pts,
                    colors=gray,
                    point_size=0.015,
                )
            else:
                if anchor_cloud_handle is not None:
                    anchor_cloud_handle.remove()
                anchor_cloud_handle = None

    hide_checkbox = server.gui.add_checkbox("Hide Gaussians", initial_value=False)

    @hide_checkbox.on_update
    def _(event) -> None:
        viewer_sync.hide_gaussians = hide_checkbox.value

    print(f"Viewer ready: http://localhost:{args.port}")
    print("Press Ctrl+C to exit.")

    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
