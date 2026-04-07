from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import nerfview
import numpy as np
from scipy.spatial.transform import Rotation as ScipyRotation

from .image_loader import ImagePolicy, ThumbnailStore

logger = logging.getLogger("cityscape_gs.ns_viewer.replica_viewer")


@dataclass(frozen=True)
class NSReplicaViewerConfig:
    add_training_cameras: bool = True
    camera_frustum_scale: float = 0.1
    image_policy: str = "lazy"  # lazy | preload
    image_cache_size: int = 256
    max_thumbnail_size: int = 128


class NSReplicaViewer:
    """Standalone viewer wrapper designed to mirror nerfview.Viewer behavior.

    This adapter keeps trainer integration minimal by exposing lock/state/
    render_tab_state/update/complete while remaining independent from
    nerfstudio internals.
    """

    def __init__(
        self,
        *,
        server,
        render_fn,
        dataset,
        config: NSReplicaViewerConfig,
    ) -> None:

        self._server = server
        self._dataset = dataset
        self._cfg = config
        self._inner = nerfview.Viewer(server=server, render_fn=render_fn, mode="training")
        self._training_camera_handles = []
        self._training_cameras_created = False
        self._show_training_cameras = None

        # if getattr(dataset, "cameras", None):
        #     self._install_training_camera_toggle_gui()
        #     if self._cfg.add_training_cameras:
        #         self._add_training_cameras()

    @property
    def lock(self):
        return self._inner.lock

    @property
    def state(self):
        return self._inner.state

    @property
    def render_tab_state(self):
        return self._inner.render_tab_state

    def update(self, step: int, num_rays: int) -> None:
        self._inner.update(step, num_rays)

    def complete(self) -> None:
        self._inner.complete()

    def _install_training_camera_toggle_gui(self) -> None:
        self._show_training_cameras = self._server.gui.add_checkbox(
            "Show Training Cameras",
            initial_value=self._cfg.add_training_cameras,
        )
        checkbox = self._show_training_cameras

        @checkbox.on_update
        def _(_event) -> None:
            self.set_training_cameras_visible(checkbox.value)

    def set_training_cameras_visible(self, visible: bool) -> None:
        if visible and not self._training_cameras_created:
            self._add_training_cameras()

        for handle in self._training_camera_handles:
            handle.visible = visible

    def _add_training_cameras(self) -> None:
        if self._training_cameras_created:
            return

        image_paths = [cam["image_path"] for cam in self._dataset.cameras]
        thumbs = ThumbnailStore(
            image_paths,
            ImagePolicy(
                mode=self._cfg.image_policy,
                max_image_size=self._cfg.max_thumbnail_size,
                cache_size=self._cfg.image_cache_size,
            ),
        )

        for i, cam_data in enumerate(self._dataset.cameras):
            r_w2c = cam_data["R"].detach().cpu().numpy()
            t_w2c = cam_data["T"].detach().cpu().numpy()

            r_c2w = r_w2c.T
            position = (-r_c2w @ t_w2c).astype(np.float32)

            qxyzw = ScipyRotation.from_matrix(r_c2w).as_quat()
            wxyz = np.array([qxyzw[3], qxyzw[0], qxyzw[1], qxyzw[2]], dtype=np.float64)

            fov_y = float(2.0 * np.arctan(cam_data["height"] / (2.0 * cam_data["fy"])))
            aspect = float(cam_data["width"]) / float(cam_data["height"])

            handle = self._server.scene.add_camera_frustum(
                name=f"/training_cameras/cam_{i:04d}",
                fov=fov_y,
                aspect=aspect,
                scale=self._cfg.camera_frustum_scale,
                color=(160, 220, 255),
                image=thumbs.get(i),
                wxyz=wxyz,
                position=position,
            )
            self._attach_camera_click_callback(handle)
            self._training_camera_handles.append(handle)

        self._training_cameras_created = True

        # Respect initial visibility state from config/GUI.
        initial_visible = True
        if self._show_training_cameras is not None:
            initial_visible = self._show_training_cameras.value
        self.set_training_cameras_visible(initial_visible)

    def _attach_camera_click_callback(self, handle) -> None:
        def on_click_callback(event) -> None:
            # Prefer the clicking client for behavior parity with nerfstudio.
            client = getattr(event, "client", None)
            if client is not None:
                with client.atomic():
                    client.camera.position = event.target.position
                    client.camera.wxyz = event.target.wxyz
                return

            # Fallback for older callback/event variants: update all clients.
            for other_client in self._server.get_clients().values():
                other_client.camera.position = handle.position
                other_client.camera.wxyz = handle.wxyz

        handle.on_click(on_click_callback)
