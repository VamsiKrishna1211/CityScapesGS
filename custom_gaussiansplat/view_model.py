"""Standalone viser viewer for inspecting a saved Gaussian Splatting model.

Usage:
    python custom_gaussiansplat/view_model.py --checkpoint path/to/ckpt.pth
    python custom_gaussiansplat/view_model.py --checkpoint path/to/ckpt.pth --model-type scaffold
    python custom_gaussiansplat/view_model.py --checkpoint path/to/ckpt.pth --port 8081
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys
import time
from pathlib import Path
from typing import Any

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

import numpy as np
import torch
import torch.nn.functional as F
from model_factory import ModelFactory
from models import NeuralRenderingMixin
from viewer_sync import ViewerParamSync

nerfview: Any = None
viser: Any = None
try:
    nerfview = importlib.import_module("nerfview")
    viser = importlib.import_module("viser")
    VIEWER_AVAILABLE = True
except ImportError:
    VIEWER_AVAILABLE = False


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
    parser.add_argument(
        "--show-anchor-similarity",
        action="store_true",
        help="Color Scaffold-GS anchors by feature similarity (supports cluster-aware mode).",
    )
    parser.add_argument(
        "--anchor-similarity-mode",
        type=str,
        default="cluster",
        choices=("mean", "anchor", "cluster"),
        help="Reference used for anchor-feature similarity visualization.",
    )
    parser.add_argument(
        "--anchor-similarity-index",
        type=int,
        default=0,
        help="Reference anchor index when using anchor-based similarity mode.",
    )
    parser.add_argument(
        "--anchor-kmeans-clusters",
        type=int,
        default=3,
        help="Number of K-means clusters for anchor feature clustering (paper-style).",
    )
    parser.add_argument(
        "--anchor-kmeans-iters",
        type=int,
        default=30,
        help="Number of K-means iterations.",
    )
    parser.add_argument(
        "--anchor-cluster-visualization",
        type=str,
        default="label",
        choices=("label", "similarity"),
        help="Cluster display style: discrete labels or cosine similarity to assigned centroid.",
    )
    parser.add_argument(
        "--anchor-visible-labels",
        type=str,
        default="",
        help="Comma-separated cluster labels to display (e.g. '0,2,4'). Empty shows all.",
    )
    parser.add_argument(
        "--anchor-point-size",
        type=float,
        default=0.015,
        help="Point size used for anchor cloud rendering.",
    )
    parser.add_argument(
        "--only-show-anchors",
        action="store_true",
        help="Hide Gaussian rendering and show only Scaffold anchors in the viewer.",
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


def _normalize_features(features: torch.Tensor) -> torch.Tensor:
    return F.normalize(features, dim=-1)


def _similarity_scores(features: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    normalized_features = _normalize_features(features)
    normalized_reference = F.normalize(reference.unsqueeze(0), dim=-1)
    return (normalized_features * normalized_reference).sum(dim=-1)


def _scores_to_colors(scores: torch.Tensor) -> np.ndarray:
    """Map cosine similarity scores in [-1, 1] to a blue-white-red ramp."""
    scores = scores.clamp(-1.0, 1.0)
    normalized = (scores + 1.0) * 0.5

    low = torch.tensor([0.160, 0.447, 0.698], device=scores.device, dtype=scores.dtype)
    mid = torch.tensor([0.965, 0.965, 0.965], device=scores.device, dtype=scores.dtype)
    high = torch.tensor([0.698, 0.094, 0.169], device=scores.device, dtype=scores.dtype)

    colors = torch.empty((scores.shape[0], 3), device=scores.device, dtype=scores.dtype)
    left_mask = normalized <= 0.5
    right_mask = ~left_mask

    if left_mask.any():
        t = (normalized[left_mask] * 2.0).unsqueeze(-1)
        colors[left_mask] = low * (1.0 - t) + mid * t
    if right_mask.any():
        t = ((normalized[right_mask] - 0.5) * 2.0).unsqueeze(-1)
        colors[right_mask] = mid * (1.0 - t) + high * t

    return (colors.clamp(0.0, 1.0).detach().cpu().numpy() * 255).astype(np.uint8)


def _anchor_similarity_colors(model, similarity_mode: str, similarity_anchor_index: int) -> np.ndarray:
    features = model.features_dc.detach()
    if features.dim() == 3:
        features = features.squeeze(1)

    if features.numel() == 0:
        return np.full((0, 3), 180, dtype=np.uint8)

    if similarity_mode == "anchor":
        index = int(np.clip(similarity_anchor_index, 0, features.shape[0] - 1))
        reference = features[index]
    else:
        reference = features.mean(dim=0)

    scores = _similarity_scores(features, reference)
    return _scores_to_colors(scores)


def _kmeans_cluster_features(
    features: torch.Tensor,
    num_clusters: int,
    num_iters: int,
    verbose: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Simple K-means on normalized anchor features."""
    n = int(features.shape[0])
    if n == 0:
        empty_labels = torch.empty((0,), dtype=torch.long, device=features.device)
        empty_centroids = torch.empty((0, features.shape[1]), dtype=features.dtype, device=features.device)
        return empty_labels, empty_centroids

    k = max(1, min(int(num_clusters), n))
    x = _normalize_features(features)

    if verbose:
        print(f"[KMeans] Step 1/3: setup done (num_points={n}, k={k}, iters={max(1, int(num_iters))})")

    perm = torch.randperm(n, device=x.device)
    centroids = x[perm[:k]].contiguous()

    labels = torch.zeros((n,), dtype=torch.long, device=x.device)
    max_iters = max(1, int(num_iters))
    for i in range(max_iters):
        if verbose:
            print(f"[KMeans] Step 2/3: iteration {i + 1}/{max_iters}")
        dists = torch.cdist(x, centroids)
        labels = torch.argmin(dists, dim=1)

        new_centroids = torch.zeros_like(centroids)
        counts = torch.zeros((k, 1), dtype=x.dtype, device=x.device)
        new_centroids.index_add_(0, labels, x)
        counts.index_add_(0, labels, torch.ones((n, 1), dtype=x.dtype, device=x.device))

        empty_mask = counts.squeeze(1) == 0
        if empty_mask.any():
            refill_idx = torch.randperm(n, device=x.device)[: int(empty_mask.sum().item())]
            new_centroids[empty_mask] = x[refill_idx]
            counts[empty_mask] = 1.0

        centroids = F.normalize(new_centroids / counts, dim=-1)

    if verbose:
        print("[KMeans] Step 3/3: finished clustering")

    return labels, centroids


def _labels_to_colors(labels: torch.Tensor) -> np.ndarray:
    palette = torch.tensor(
        [
            [228, 26, 28],
            [55, 126, 184],
            [77, 175, 74],
            [152, 78, 163],
            [255, 127, 0],
            [166, 86, 40],
            [247, 129, 191],
            [153, 153, 153],
        ],
        dtype=torch.uint8,
        device=labels.device,
    )
    return palette[labels % palette.shape[0]].detach().cpu().numpy()


def _anchor_cluster_colors(
    model,
    num_clusters: int,
    num_iters: int,
    visualization: str,
    verbose: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    features = model.features_dc.detach()
    if features.dim() == 3:
        features = features.squeeze(1)

    if features.numel() == 0:
        return np.full((0, 3), 180, dtype=np.uint8), np.full((0,), -1, dtype=np.int64)

    labels, centroids = _kmeans_cluster_features(
        features,
        num_clusters=num_clusters,
        num_iters=num_iters,
        verbose=verbose,
    )
    if visualization == "label":
        return _labels_to_colors(labels), labels.detach().cpu().numpy().astype(np.int64)

    feat_norm = _normalize_features(features)
    scores = (feat_norm * centroids[labels]).sum(dim=-1)
    return _scores_to_colors(scores), labels.detach().cpu().numpy().astype(np.int64)


def _parse_visible_labels(spec: str) -> set[int] | None:
    text = (spec or "").strip()
    if text == "":
        return None

    labels: set[int] = set()
    for token in text.split(","):
        token = token.strip()
        if token == "":
            continue
        try:
            value = int(token)
        except ValueError:
            continue
        if value >= 0:
            labels.add(value)

    return labels if labels else None


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

    if args.only_show_anchors and not isinstance(model, NeuralRenderingMixin):
        print("Warning: --only-show-anchors is only supported for Scaffold models; ignoring it.")
        args.only_show_anchors = False

    # Wire up the render callback (reuses the same sync helper as the trainer)
    viewer_sync = ViewerParamSync(
        model=model,
        device=device,
        disable_sh_rendering=args.disable_sh,
        refresh_interval=1,  # static model — no need to poll
    )

    server = viser.ViserServer(port=args.port, verbose=False)

    # "rendering" mode — no training-specific UI (pause/step/rays-per-sec)
    _viewer = nerfview.Viewer(
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
        show_anchors_checkbox = server.gui.add_checkbox(
            "Show Anchors", initial_value=args.only_show_anchors
        )
        viewer_sync.show_anchors = show_anchors_checkbox.value
        if args.only_show_anchors:
            viewer_sync.hide_gaussians = True

        # Live cluster-label filtering controls (created lazily once labels are known).
        cluster_label_checkboxes: dict[int, Any] = {}
        cached_cluster_colors: np.ndarray | None = None
        cached_cluster_labels: np.ndarray | None = None

        initial_visible_labels = _parse_visible_labels(args.anchor_visible_labels)

        similarity_index_slider = None
        if args.show_anchor_similarity and args.anchor_similarity_mode == "anchor":
            similarity_index_slider = server.gui.add_slider(
                "Similarity Anchor Index",
                min=0,
                max=max(0, n_gaussians - 1),
                step=1,
                initial_value=min(args.anchor_similarity_index, max(0, n_gaussians - 1)),
            )

        # Run cluster K-means once at startup and reuse labels/colors afterwards.
        if args.show_anchor_similarity and args.anchor_similarity_mode == "cluster":
            if args.anchor_cluster_visualization != "label":
                print(
                    "[Viewer] Cluster mode uses label visualization only when caching startup K-means; "
                    "overriding --anchor-cluster-visualization to 'label'."
                )
            print("[Viewer] Running startup K-means for anchor labels...")
            cached_cluster_colors, cached_cluster_labels = _anchor_cluster_colors(
                model,
                num_clusters=args.anchor_kmeans_clusters,
                num_iters=args.anchor_kmeans_iters,
                visualization="label",
                verbose=True,
            )
            print("[Viewer] Startup K-means complete.")

        def _anchor_colors() -> tuple[np.ndarray, np.ndarray | None]:
            if not args.show_anchor_similarity:
                pts = model.anchors.detach().cpu().numpy().astype(np.float32)
                return np.full((len(pts), 3), 180, dtype=np.uint8), None

            current_mode = args.anchor_similarity_mode
            current_index = args.anchor_similarity_index
            if current_mode == "anchor" and similarity_index_slider is not None:
                current_index = int(similarity_index_slider.value)
            if current_mode == "cluster":
                if cached_cluster_colors is None or cached_cluster_labels is None:
                    # Fallback path if startup cache was skipped for any reason.
                    fallback_colors, fallback_labels = _anchor_cluster_colors(
                        model,
                        num_clusters=args.anchor_kmeans_clusters,
                        num_iters=args.anchor_kmeans_iters,
                        visualization="label",
                        verbose=True,
                    )
                    return fallback_colors, fallback_labels
                return cached_cluster_colors, cached_cluster_labels
            return _anchor_similarity_colors(model, current_mode, current_index), None

        def _refresh_anchor_cloud() -> None:
            nonlocal anchor_cloud_handle
            if anchor_cloud_handle is not None:
                anchor_cloud_handle.remove()
                anchor_cloud_handle = None

            if not show_anchors_checkbox.value:
                return

            pts = model.anchors.detach().cpu().numpy().astype(np.float32)
            colors, labels = _anchor_colors()

            if labels is not None:
                label_array = np.asarray(labels, dtype=np.int64)

                # Build live checkboxes once for each discovered cluster label.
                if len(cluster_label_checkboxes) == 0:
                    unique_labels = sorted(np.unique(label_array).tolist())
                    for lbl in unique_labels:
                        is_visible = (
                            True
                            if initial_visible_labels is None
                            else (int(lbl) in initial_visible_labels)
                        )
                        cb = server.gui.add_checkbox(
                            f"Show Cluster {int(lbl)}",
                            initial_value=is_visible,
                        )

                        @cb.on_update
                        def _on_cluster_visibility_update(event) -> None:
                            if show_anchors_checkbox.value:
                                _refresh_anchor_cloud()

                        cluster_label_checkboxes[int(lbl)] = cb

                selected_labels = {
                    int(lbl)
                    for lbl, cb in cluster_label_checkboxes.items()
                    if cb.value
                }
                if selected_labels is not None:
                    keep = np.isin(label_array, np.array(sorted(selected_labels), dtype=np.int64))
                    pts = pts[keep]
                    colors = colors[keep]

            anchor_cloud_handle = server.scene.add_point_cloud(
                name="/scaffold/anchors",
                points=pts,
                colors=colors,
                point_size=max(1e-4, float(args.anchor_point_size)),
            )

        @show_anchors_checkbox.on_update
        def _on_show_anchors_update(event) -> None:
            viewer_sync.show_anchors = show_anchors_checkbox.value
            _refresh_anchor_cloud()

        if similarity_index_slider is not None:
            @similarity_index_slider.on_update
            def _on_similarity_index_update(event) -> None:
                if show_anchors_checkbox.value:
                    _refresh_anchor_cloud()

        # In only-anchor mode, rendering is blank by design, so seed the
        # anchor point cloud immediately instead of waiting for GUI interaction.
        if show_anchors_checkbox.value:
            _refresh_anchor_cloud()

    hide_checkbox = server.gui.add_checkbox("Hide Gaussians", initial_value=False)
    if args.only_show_anchors:
        hide_checkbox.value = True
        viewer_sync.hide_gaussians = True

    @hide_checkbox.on_update
    def _on_hide_gaussians_update(event) -> None:
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
