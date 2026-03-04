from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from gsplat import rasterization


@dataclass
class _ViewerRenderCache:
    means: torch.Tensor
    quats: torch.Tensor
    scales: torch.Tensor
    opacities: torch.Tensor
    colors: torch.Tensor
    sh_degree: Optional[int]


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
        self.refresh(step=0, force=True)

    @torch.no_grad()
    def refresh(self, step: int, force: bool = False) -> None:
        if not force and self._last_refresh_step >= 0 and (step - self._last_refresh_step) < self.refresh_interval:
            return

        colors = (
            self.model._features_dc.squeeze(1)
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

        c2w = torch.from_numpy(camera_state.c2w).float().to(self.device)
        K = torch.from_numpy(camera_state.get_K((width, height))).float().to(self.device)
        viewmat = torch.linalg.inv(c2w)

        try:
            render, _, _ = rasterization(
                means=self._cache.means,
                quats=self._cache.quats,
                scales=self._cache.scales,
                opacities=self._cache.opacities,
                colors=self._cache.colors,
                viewmats=viewmat[None, ...],
                Ks=K[None, ...],
                width=width,
                height=height,
                sh_degree=self._cache.sh_degree,
            )
            render_rgb = torch.clamp(render[0, ..., 0:3], 0, 1)
            return (render_rgb.detach().cpu().numpy() * 255).astype(np.uint8)
        except Exception:
            return np.zeros((height, width, 3), dtype=np.uint8)
