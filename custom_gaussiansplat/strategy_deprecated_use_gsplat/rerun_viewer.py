import torch
import numpy as np

try:
    import rerun as rr
    RERUN_AVAILABLE = True
except ImportError:
    RERUN_AVAILABLE = False


class RerunViewer:
    """Synchronize model params for viewer rendering in Rerun at a fixed interval."""

    def __init__(
        self,
        model,
        disable_sh_rendering: bool,
        refresh_interval: int = 100,
    ) -> None:
        if not RERUN_AVAILABLE:
            raise ImportError("rerun-sdk is not installed. Please install it to use RerunViewer.")
        self.model = model
        self.disable_sh_rendering = disable_sh_rendering
        self.refresh_interval = max(1, int(refresh_interval))
        self._last_refresh_step = -1

    def init(self):
        """Initializes Rerun and spawns the viewer."""
        rr.init("CityScapesGS", spawn=True)
        # 3D points in Gaussian Splatting usually assume y-down for camera, Rerun uses X-Forward, Y-Right, Z-Down by default for images but we can define coordinate system
        rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, timeless=True)

    @torch.no_grad()
    def update(self, step: int, force: bool = False) -> None:
        """Logs the Gaussians to Rerun."""
        if not force and self._last_refresh_step >= 0 and (step - self._last_refresh_step) < self.refresh_interval:
            return

        rr.set_time_sequence("step", step)

        means = self.model.means.detach().cpu().numpy()
        # Scale is exponentiated in GaussianModel if log-scale is used
        scales = self.model.scales.detach().cpu().numpy()
        opacities = (self.model.opacities if hasattr(self.model, 'opacities') else getattr(self.model, '_opacities')).squeeze(-1).detach().cpu().numpy()

        colors = (
            self.model._features_dc.squeeze(1)
            if self.disable_sh_rendering
            else self.model.sh
        )
        
        sh_C0 = 0.28209479177387814
        # Extract DC component or use directly
        if colors.dim() == 3:
            # SH features: [N, D, 3] where D is number of SH bases
            colors_dc = colors[:, 0, :]
        else:
            # Just DC features: [N, 3]
            colors_dc = colors

        colors_rgb = (colors_dc.detach().cpu().numpy() * sh_C0 + 0.5).clip(0, 1)

        # We take the maximum scale along axes as a proxy for radius in world space
        radii = scales.max(axis=1) if scales.shape[1] == 3 else scales.squeeze()

        # Combine with opacity for RGBA
        rgba = np.concatenate([colors_rgb, np.expand_dims(opacities, axis=1).clip(0, 1)], axis=1)
        # Assuming Rerun 0.15+, rgba components should be [0-1] or [0-255]. Rerun handles [0-1] floats usually, or [0-255] uint8.
        # It's safer to provide RGBA as uint8 [0, 255]
        rgba_uint8 = (rgba * 255.0).astype(np.uint8)

        rr.log("world/gaussians", rr.Points3D(positions=means, radii=radii, colors=rgba_uint8))

        self._last_refresh_step = step
