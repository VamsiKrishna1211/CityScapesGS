from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional

import numpy as np
import torch
from gsplat import rasterization
from models.base import BaseTrainableModel, NeuralRenderingMixin


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
        model,
        device: torch.device,
        disable_sh_rendering: bool,
        refresh_interval: int = 100,
    ) -> None:
        self.model = model
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
        if self._cache is None:
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
            )
            render_rgb = torch.clamp(render[0, ..., 0:3], 0, 1)
            return (render_rgb.detach().cpu().numpy() * 255).astype(np.uint8)
        except Exception:
            return np.zeros((height, width, 3), dtype=np.uint8)
