from __future__ import annotations

import importlib
import importlib.util
import logging
from dataclasses import dataclass, field
from typing import Any, List, Optional

import numpy as np
import torch
from gsplat import rasterization
from models import (
    BaseTrainableModel,
    GaussianModel,
    NeuralRenderingMixin,
    ScaffoldModel,
)


@dataclass
class _ViewerRenderCache:
    means: torch.Tensor
    quats: torch.Tensor
    scales: torch.Tensor
    opacities: torch.Tensor
    colors: torch.Tensor
    sh_degree: Optional[int]
    lod_offsets: List[int]


class ViewerParamSync:
    """Synchronize model params for viewer rendering at a fixed interval."""

    def __init__(
        self,
        model: BaseTrainableModel | NeuralRenderingMixin | ScaffoldModel | GaussianModel,
        device: torch.device,
        disable_sh_rendering: bool,
        refresh_interval: int = 100,
    ) -> None:
        self.model: BaseTrainableModel | NeuralRenderingMixin | ScaffoldModel | GaussianModel = model
        self.device = device
        self.disable_sh_rendering = disable_sh_rendering
        self.refresh_interval = max(1, int(refresh_interval))
        self._cache: Optional[_ViewerRenderCache] = None
        self._last_refresh_step = -1
        self.lod_slider: Optional[Any] = None # Optional viser slider for LoD control
        self.show_anchors: bool = False  # Toggle flag for anchor point cloud visualization
        self.hide_gaussians: bool = False  # Toggle flag to hide Gaussian rendering
        self.refresh(step=0, force=True)

    @torch.no_grad()
    def refresh(self, step: int, force: bool = False) -> None:
        if not force and self._last_refresh_step >= 0 and (step - self._last_refresh_step) < self.refresh_interval:
            return

        if isinstance(self.model, NeuralRenderingMixin):
            # Neural models (e.g. ScaffoldModel) generate Gaussians dynamically per
            # camera — there are no stored quats/means to cache.  render_fn handles
            # them by calling generate_neural_gaussians directly.
            self._last_refresh_step = step
            return

        colors = (
            self.model.dc_rgb.squeeze(1)
            if self.disable_sh_rendering
            else self.model.sh
        )

        self._cache = _ViewerRenderCache(
            means=self.model.means.detach(),
            quats=self.model.quats.detach(),
            scales=self.model.scales.detach(),
            opacities=self.model.opacities.squeeze(-1).detach(),
            colors=colors.detach(),
            sh_degree=None if self.disable_sh_rendering else self.model.sh_degree,
            lod_offsets=getattr(self.model, "lod_offsets", [len(self.model.means)]),
        )
        self._last_refresh_step = step

    def refresh_if_needed(self, step: int) -> None:
        self.refresh(step=step, force=False)

    @torch.no_grad()
    def render_fn(self, camera_state, render_tab_state):
        if self._cache is None and not isinstance(self.model, NeuralRenderingMixin):
            width = render_tab_state.render_width if render_tab_state.preview_render else render_tab_state.viewer_width
            height = render_tab_state.render_height if render_tab_state.preview_render else render_tab_state.viewer_height
            return np.zeros((height, width, 3), dtype=np.uint8)

        if render_tab_state.preview_render:
            width = render_tab_state.render_width
            height = render_tab_state.render_height
        else:
            width = render_tab_state.viewer_width
            height = render_tab_state.viewer_height

        # Return blank frame if Gaussians are hidden
        if self.hide_gaussians:
            return np.zeros((height, width, 3), dtype=np.uint8)

        c2w = torch.from_numpy(camera_state.c2w).float().to(self.device)
        K = torch.from_numpy(camera_state.get_K((width, height))).float().to(self.device)
        viewmat = torch.linalg.inv(c2w)

        if isinstance(self.model, NeuralRenderingMixin):
            # Scaffold-GS dynamic rendering path — model implements neural Gaussian generation
            if self.show_anchors:
                # Anchor overlay is shown via viser scene; return blank render
                return np.zeros((height, width, 3), dtype=np.uint8)

            cam = {
                "camera_center": c2w[:3, 3],
                "uid": 0,
                "width": width,
                "height": height,
            }
            out = self.model.generate_neural_gaussians(cam, is_training=False)
            means = out.means
            quats = out.quats
            scales = out.scales
            opacities = out.opacities.squeeze(-1)
            colors = out.colors
            sh_degree = None
        else:
            # Standard GaussianModel cached path
            # Handle LoD slicing
            lod = 0
            if self.lod_slider is not None:
                lod = int(self.lod_slider.value)
            else:
                lod = getattr(render_tab_state, "lod", 0)
            
            offsets = self._cache.lod_offsets
            if lod < 0 or lod >= len(offsets):
                end_idx = offsets[-1]
            else:
                idx = len(offsets) - 1 - lod
                end_idx = offsets[idx]
                
            means = self._cache.means[:end_idx]
            quats = self._cache.quats[:end_idx]
            scales = self._cache.scales[:end_idx]
            opacities = self._cache.opacities[:end_idx]
            colors = self._cache.colors[:end_idx]
            sh_degree = self._cache.sh_degree

        try:
            # Background colors - randomly select one from the list
            # backgrounds_list = torch.tensor([
            #     [1.0, 1.0, 1.0],  # White
            #     [0.0, 0.0, 0.0],  # Black
            #     [0.5, 0.5, 0.5],  # Gray
            #     [0.2, 0.2, 0.2],  # Dark gray
            # ], device=means.device, dtype=means.dtype)

            # Randomly select one background, shape: [3, 1]
            # background = backgrounds_list[torch.randint(0, backgrounds_list.size(0), (1,))]

            render, _, _ = rasterization(
                means=means,
                quats=quats,
                scales=scales,
                opacities=opacities,
                colors=colors,
                viewmats=viewmat[None, ...],
                Ks=K[None, ...],
                width=width,
                height=height,
                sh_degree=sh_degree,
                # backgrounds=background,
            )
            render_rgb = torch.clamp(render[0, ..., 0:3], 0, 1)
            return (render_rgb.detach().cpu().numpy() * 255).astype(np.uint8)
        except Exception:
            return np.zeros((height, width, 3), dtype=np.uint8)


# ──────────────────────────────────────────────────────────────────────────────
# Shared training-viewer setup
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class TrainingViewerBundle:
    """All viewer handles returned by setup_training_viewer.

    Fields are None when the viewer is disabled or packages are unavailable.
    """
    viewer: Optional[Any] = None
    server: Optional[Any] = None
    viewer_param_sync: Optional[ViewerParamSync] = None
    rerun_viewer: Optional[Any] = None


def setup_training_viewer(
    model: BaseTrainableModel,
    device: torch.device,
    viewer_cfg: Any,
    verbosity: int = 1,
    logger: Optional[logging.Logger] = None,
) -> TrainingViewerBundle:
    """Create and wire up the viser/nerfview (and optional rerun) viewer for training.

    Shared by SemanticTrainer and the main Trainer so viewer logic lives in one place.
    Returns a TrainingViewerBundle; all fields are None when viewer is disabled.
    """
    bundle = TrainingViewerBundle()

    _rerun_available = importlib.util.find_spec("rerun") is not None
    _viewer_available = (
        importlib.util.find_spec("nerfview") is not None
        and importlib.util.find_spec("viser") is not None
    )

    if viewer_cfg and getattr(viewer_cfg, "rerun_viewer", False):
        if not _rerun_available:
            if logger:
                logger.warning("[yellow]⚠ Warning:[/yellow] rerun-sdk not available.")
        else:
            try:
                from rerun_viewer import RerunViewer  # type: ignore[import-untyped]
                bundle.rerun_viewer = RerunViewer(
                    model=model,
                    disable_sh_rendering=False,
                    refresh_interval=viewer_cfg.viewer_refresh_interval,
                )
                bundle.rerun_viewer.init()
                if verbosity >= 1 and logger:
                    logger.info("[green]📺 Rerun Viewer started.[/green]")
            except ImportError:
                if logger:
                    logger.warning("[yellow]⚠ Warning:[/yellow] rerun_viewer module not available.")

    if not viewer_cfg or not getattr(viewer_cfg, "viewer", False):
        return bundle
    if not _viewer_available:
        if logger:
            logger.warning("[yellow]⚠ Warning:[/yellow] nerfview not available.")
        return bundle

    nerfview_mod = importlib.import_module("nerfview")
    viser_mod = importlib.import_module("viser")

    bundle.viewer_param_sync = ViewerParamSync(
        model=model,
        device=device,
        disable_sh_rendering=False,
        refresh_interval=viewer_cfg.viewer_refresh_interval,
    )
    bundle.server = viser_mod.ViserServer(port=viewer_cfg.viewer_port, verbose=False)
    bundle.viewer = nerfview_mod.Viewer(
        server=bundle.server,
        render_fn=bundle.viewer_param_sync.render_fn,
        mode="training",
    )

    if isinstance(model, NeuralRenderingMixin):
        show_anchors_cb = bundle.server.gui.add_checkbox("Show Anchors", initial_value=False)
        bundle.viewer_param_sync.show_anchors = show_anchors_cb.value
        _anchor_handle: list = [None]  # mutable cell so the callback can update it

        @show_anchors_cb.on_update
        def _on_show_anchors(event: Any) -> None:
            if bundle.viewer_param_sync is None or bundle.server is None:
                return
            bundle.viewer_param_sync.show_anchors = show_anchors_cb.value
            if show_anchors_cb.value:
                pts = model.means.detach().cpu().numpy().astype("float32")
                gray = np.full((len(pts), 3), 180, dtype="uint8")
                _anchor_handle[0] = bundle.server.scene.add_point_cloud(
                    name="/scaffold/anchors", points=pts, colors=gray, point_size=0.015,
                )
            else:
                if _anchor_handle[0] is not None:
                    _anchor_handle[0].remove()
                    _anchor_handle[0] = None

    hide_cb = bundle.server.gui.add_checkbox("Hide Gaussians", initial_value=False)

    @hide_cb.on_update
    def _on_hide_gaussians(event: Any) -> None:
        if bundle.viewer_param_sync is not None:
            bundle.viewer_param_sync.hide_gaussians = hide_cb.value

    if verbosity >= 1 and logger:
        logger.info(
            f"[green]📺 Viewer started:[/green] http://localhost:{viewer_cfg.viewer_port}"
        )

    return bundle
