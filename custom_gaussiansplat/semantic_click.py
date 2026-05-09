"""Click-to-query per-pixel semantic similarity overlay for viser viewers.

"Enable Semantic Select" checkbox gates all click interception.  When off,
viser's default left-click-to-orbit is fully restored.  When on, clicking
a pixel in the rendered scene extracts that pixel's rendered semantic feature
and overlays a cosine-similarity heatmap on every subsequent frame.

"Sem Multi-select" keeps accumulating queries across clicks (union).

Viser API: server.scene.on_pointer_event('click') / remove_pointer_callback().
"""
from __future__ import annotations

from typing import Any, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from models import NeuralRenderingMixin, SemanticsMixin


# ── Colour utility ────────────────────────────────────────────────────────────

def apply_jet_colormap(values: np.ndarray) -> np.ndarray:
    """Map float values in [0, 1] to jet colormap RGB uint8, shape [..., 3]."""
    values = np.clip(values, 0.0, 1.0)
    r = np.clip(1.5 - np.abs(4.0 * values - 3.0), 0.0, 1.0)
    g = np.clip(1.5 - np.abs(4.0 * values - 2.0), 0.0, 1.0)
    b = np.clip(1.5 - np.abs(4.0 * values - 1.0), 0.0, 1.0)
    return (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)


# ── Camera helpers ────────────────────────────────────────────────────────────

def build_semantic_cam(
    camera_state: Any, width: int, height: int, device: torch.device
) -> dict:
    """Convert a nerfview CameraState to the cam dict expected by render_semantics()."""
    c2w = torch.from_numpy(camera_state.c2w).float().to(device)
    K = torch.from_numpy(camera_state.get_K((width, height))).float().to(device)
    viewmat = torch.linalg.inv(c2w)
    return {
        "R": viewmat[:3, :3],
        "T": viewmat[:3, 3],
        "fx": float(K[0, 0].item()),
        "fy": float(K[1, 1].item()),
        "cx": float(K[0, 2].item()),
        "cy": float(K[1, 2].item()),
        "width": width,
        "height": height,
        "camera_center": c2w[:3, 3],
        "uid": 0,
    }


def _cam_from_viser_client(
    client: Any, width: int, height: int, device: torch.device
) -> dict:
    """Build a cam dict from a viser CameraHandle (available in ScenePointerEvent.client)."""
    wxyz = np.array(client.camera.wxyz, dtype=np.float64)  # [w, x, y, z]
    pos = np.array(client.camera.position, dtype=np.float64)  # [3]

    w, x, y, z = wxyz
    R_c2w = np.array([
        [1 - 2*(y*y + z*z),   2*(x*y - w*z),       2*(x*z + w*y)],
        [2*(x*y + w*z),       1 - 2*(x*x + z*z),   2*(y*z - w*x)],
        [2*(x*z - w*y),       2*(y*z + w*x),       1 - 2*(x*x + y*y)],
    ])
    c2w = np.eye(4)
    c2w[:3, :3] = R_c2w
    c2w[:3, 3] = pos

    fov_y = float(client.camera.fov)  # vertical fov in radians
    fy = height / (2.0 * np.tan(fov_y / 2.0))
    fx = fy
    cx, cy = width / 2.0, height / 2.0

    c2w_t = torch.from_numpy(c2w).float().to(device)
    viewmat = torch.linalg.inv(c2w_t)
    return {
        "R": viewmat[:3, :3],
        "T": viewmat[:3, 3],
        "fx": fx, "fy": fy, "cx": cx, "cy": cy,
        "width": width, "height": height,
        "camera_center": c2w_t[:3, 3],
        "uid": 0,
    }


# ── Interactor ────────────────────────────────────────────────────────────────

class SemanticClickInteractor:
    """Per-pixel semantic similarity query driven by viewer clicks.

    Workflow:
      1. User enables "Enable Semantic Select".
      2. User clicks a pixel in the rendered scene.
      3. render_semantics() is called at the click-time camera view; the feature
         at the clicked pixel is extracted and stored as a query.
      4. Every rendered frame computes max cosine similarity of all pixels
         against the stored query features and overlays a heatmap.
      5. Clicking again (single mode) replaces the query; "Sem Multi-select"
         accumulates queries for a union highlight.
    """

    def __init__(self, model: Any, device: torch.device, server: Any) -> None:
        self.model = model
        self.device = device
        self.server = server
        self.threshold: float = 0.5
        self._alpha: float = 0.6
        self._show_mask: bool = False
        self._select_enabled: bool = False
        self._multi_select: bool = False
        self._query_feats: List[torch.Tensor] = []  # each [semantics_dim]

    # ── GUI ───────────────────────────────────────────────────────────────────

    def attach_gui(self) -> None:
        enable_cb = self.server.gui.add_checkbox("Enable Semantic Select", initial_value=False)

        @enable_cb.on_update
        def _on_enable(event: Any) -> None:
            self._select_enabled = bool(enable_cb.value)
            if self._select_enabled:
                self._register_click_handler()
            else:
                self.server.scene.remove_pointer_callback()
                self.clear()

        multi_cb = self.server.gui.add_checkbox("Sem Multi-select", initial_value=False)

        @multi_cb.on_update
        def _on_multi(event: Any) -> None:
            self._multi_select = bool(multi_cb.value)

        threshold_sl = self.server.gui.add_slider(
            "Sim Threshold", min=0.0, max=1.0, step=0.01, initial_value=self.threshold,
        )

        @threshold_sl.on_update
        def _on_threshold(event: Any) -> None:
            self.threshold = float(threshold_sl.value)

        alpha_sl = self.server.gui.add_slider(
            "Sem Blend Alpha", min=0.0, max=1.0, step=0.05, initial_value=self._alpha,
        )

        @alpha_sl.on_update
        def _on_alpha(event: Any) -> None:
            self._alpha = float(alpha_sl.value)

        mask_cb = self.server.gui.add_checkbox("Sem Hard Mask", initial_value=False)

        @mask_cb.on_update
        def _on_mask(event: Any) -> None:
            self._show_mask = bool(mask_cb.value)

        clear_btn = self.server.gui.add_button("Clear Similarity")

        @clear_btn.on_click
        def _on_clear(event: Any) -> None:
            self.clear()

    # ── Click handling ────────────────────────────────────────────────────────

    def attach_click_handler(self) -> None:
        """Pre-register handler (no-op guard); actual registration happens on enable."""

    def _register_click_handler(self) -> None:
        @self.server.scene.on_pointer_event("click")
        def handle_click(event: Any) -> None:
            self._on_click(event)

    def _on_click(self, event: Any) -> None:
        if event.ray_origin is None or not event.screen_pos:
            return

        u, v = event.screen_pos[0]  # normalized [0, 1]
        client = event.client

        W = max(int(client.camera.image_width), 64)
        H = max(int(client.camera.image_height), 64)
        cam = _cam_from_viser_client(client, W, H, self.device)

        try:
            with torch.no_grad():
                _, feat_map = self.model.render_semantics(cam, self.device)
            # feat_map: [H, W, semantics_dim]
            px = min(int(u * feat_map.shape[1]), feat_map.shape[1] - 1)
            py = min(int(v * feat_map.shape[0]), feat_map.shape[0] - 1)
            feat = feat_map[py, px].detach()
        except Exception:
            return

        if self._multi_select:
            self._query_feats.append(feat)
        else:
            self._query_feats = [feat]

    # ── Render wrapper ────────────────────────────────────────────────────────

    def make_render_fn(self, base_fn: Any) -> Any:
        """Return a render_fn that wraps base_fn with a live similarity heatmap.

        The wrapped fn has the same signature: (camera_state, render_tab_state) -> ndarray.
        When no queries are set or select is disabled, the base RGB is returned unchanged.
        """
        interactor = self

        def render_fn(camera_state: Any, render_tab_state: Any) -> np.ndarray:
            rgb = base_fn(camera_state, render_tab_state)
            if not interactor._select_enabled or not interactor._query_feats:
                return rgb

            H, W = rgb.shape[:2]
            cam = build_semantic_cam(camera_state, W, H, interactor.device)
            try:
                with torch.no_grad():
                    _, feat_map = interactor.model.render_semantics(cam, interactor.device)
            except Exception:
                return rgb

            # [H*W, D] cosine similarity against all queries (max = union)
            feat_flat = feat_map.reshape(-1, feat_map.shape[-1])
            feat_norm = F.normalize(feat_flat, dim=-1)
            max_sims = torch.full((H * W,), -1.0, device=interactor.device)
            for q in interactor._query_feats:
                q_norm = F.normalize(q.unsqueeze(0), dim=-1)
                sims = (feat_norm * q_norm).sum(dim=-1)
                max_sims = torch.maximum(max_sims, sims)

            sim_map = max_sims.reshape(H, W).cpu().numpy()
            # Map cosine sim [-1,1] → [0,1]
            sim_norm = (sim_map + 1.0) * 0.5

            alpha = interactor._alpha
            if interactor._show_mask:
                mask = (sim_norm > interactor.threshold)[..., None].astype(np.float32)
                return (rgb * (1.0 - alpha) + mask * 255.0 * alpha).clip(0, 255).astype(np.uint8)
            else:
                heatmap = apply_jet_colormap(sim_norm)
                return (rgb * (1.0 - alpha) + heatmap * alpha).clip(0, 255).astype(np.uint8)

        return render_fn

    # ── Misc ──────────────────────────────────────────────────────────────────

    def clear(self) -> None:
        self._query_feats = []


# ── Factory ───────────────────────────────────────────────────────────────────

def setup_semantic_click(
    model: Any,
    device: torch.device,
    server: Any,
) -> Optional[SemanticClickInteractor]:
    """Create a SemanticClickInteractor and attach its GUI if the model supports it.

    Returns the interactor (caller wraps the render_fn via make_render_fn),
    or None if the model is not a semantic NeuralRenderingMixin.
    """
    if not (isinstance(model, NeuralRenderingMixin) and isinstance(model, SemanticsMixin)):
        return None
    interactor = SemanticClickInteractor(model, device, server)
    interactor.attach_gui()
    return interactor
