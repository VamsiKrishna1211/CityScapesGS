import torch.nn as nn
from gsplat import spherical_harmonics
import torch
import logging
from gs_types import Parameters, GSOptimizers
from typing import Optional, Dict, Tuple

# Module-level logger
logger = logging.getLogger("cityscape_gs.model")

# ViewportInfo dataclass for backward compatibility
from dataclasses import dataclass

from simple_knn._C import distCUDA2

@dataclass
class ViewportInfo:
    """Information about the current viewport/camera for gradient normalization."""
    width: int
    height: int
    n_cameras: int = 1

def inverse_sigmoid(x):
    return torch.log(x / (1 - x))

class GaussianModel(nn.Module):
    def __init__(self, init_points: torch.Tensor, 
                 init_colors: torch.Tensor, 
                 sh_degree=3,
                 train_semantics=False,
                 semantics_dim=3,
                 console=None):
        super().__init__()
        num_points = init_points.shape[0]
        self.sh_degree = sh_degree
        self.console = console

        if len(init_points) == 0:
            raise ValueError("Initialization point cloud is empty. Please provide a valid COLMAP point cloud for initialization.")
        # --- Learnable Parameters ---
        # 1. Position: The center of the gaussian
        self._means = nn.Parameter(init_points)
        
        # 2. Scale: Log-space to ensure positivity
        # Initialize as isotropic small spheres
        # dist_to_nearest = torch.ones(num_points, device=init_points.device) * 0.1
        # Use KNN to find distance to nearest neighbor for better initialization
        torch.zeros(1, device=init_points.device)  # Warm up CUDA context to prevent first-iteration lag
        with torch.no_grad():
            # distCUDA2 expects a single tensor and returns a 1D tensor of mean squared
            # distances per point. Take sqrt to get distance and clamp small values.
            dist_to_nearest = torch.sqrt(distCUDA2(init_points.contiguous()))
            torch.zeros(1, device=init_points.device)  # Warm up CUDA context to prevent first-iteration lag
            logger.debug(f"Distance to nearest neighbor stats: min={dist_to_nearest.min().item():.6f}, max={dist_to_nearest.max().item():.6f}, mean={dist_to_nearest.mean().item():.6f}")

            dist_to_nearest[dist_to_nearest < 1e-5] = 1e-5  # Avoid zero distances
        
        self._scales = nn.Parameter(torch.log(dist_to_nearest.unsqueeze(1).repeat(1, 3)))
        
        # 3. Rotation: Quaternion (w, x, y, z)
        # Initialize as identity
        quats = torch.zeros(num_points, 4, device=init_points.device) + 10e-6
        quats[:, 0] = 1.0 
        self._quats = nn.Parameter(quats)
        
        # 4. Opacity: Logit-space (inverse sigmoid)
        # Initialize to 0.1 opacity
        self._opacities = nn.Parameter(inverse_sigmoid(0.1 * torch.ones(num_points, 1, device=init_points.device)))
        
        # 5. Color: Spherical Harmonics
        # Features DC (0th order) + Features Rest (Higher order)
        # We fuse RGB into the DC component for easier initialization
        self._features_dc = nn.Parameter(init_colors.unsqueeze(1)) # [N, 1, 3]
        self._features_rest = nn.Parameter(torch.zeros(num_points, (sh_degree + 1)**2 - 1, 3, device=init_points.device)) # [N, D, 3]

        if train_semantics:
            # 6. Optional Semantic Features (e.g., for semantic segmentation)
            if semantics_dim is None or semantics_dim <= 0:
                raise ValueError("semantics_dim must be a positive integer when train_semantics is True.")
            self._features_semantics = nn.Parameter(torch.zeros(num_points, semantics_dim, device=init_points.device)) # [N, semantics_dim]
        
        # Visibility tracking buffer for multi-view consistency (floater prevention)
        self.register_buffer('view_count', torch.zeros(num_points, device=init_points.device))

    @property
    def means(self): return self._means
    
    @property
    def scales(self): return torch.exp(self._scales)
    
    @property
    def quats(self): return torch.nn.functional.normalize(self._quats)
    
    @property
    def opacities(self): return torch.sigmoid(self._opacities)
    
    @property
    def sh(self): return torch.cat([self._features_dc, self._features_rest], dim=1)

    @property
    def semantics(self):
        if hasattr(self, '_features_semantics'):
            return self._features_semantics
        else:
            raise ValueError("Semantic features are not enabled. Please enable them by setting train_semantics=True in the config.")
    
    def add_view_count(self, alpha_contributions, threshold=0.5):
        """
        Track which Gaussians contribute significantly to rendered pixels.
        Used for multi-view consistency checking.
        
        Args:
            alpha_contributions: Per-Gaussian alpha contribution to pixels [N]
            threshold: Minimum alpha contribution to count as "visible"
        """
        visible_mask = alpha_contributions > threshold
        self.view_count[visible_mask] += 1

    def get_params_dict(self) -> Dict[str, nn.Parameter]:
        """Get model parameters as dictionary for strategy interface.
        
        Returns:
            Dictionary mapping parameter names to Parameter tensors
        """
        params = {
            "means": self._means,
            "scales": self._scales,
            "quats": self._quats,
            "opacities": self._opacities,
            "features_dc": self._features_dc,
            "features_rest": self._features_rest,
        }
        if hasattr(self, '_features_semantics'):
            params["features_semantics"] = self._features_semantics
        return params
    
    def get_optimizers_dict(self, optimizers: GSOptimizers) -> Dict[str, torch.optim.Optimizer]:
        """Convert GSOptimizers to dictionary for strategy interface.
        
        Args:
            optimizers: GSOptimizers named tuple containing all optimizers
            
        Returns:
            Dictionary mapping parameter names to optimizers
        """
        optimizers_dict = {
            "means": optimizers.means,
            "scales": optimizers.scales,
            "quats": optimizers.quats,
            "opacities": optimizers.opacities,
            "features_dc": optimizers.features_dc,
            "features_rest": optimizers.features_rest,
        }
        if optimizers.features_semantics is not None:
            optimizers_dict["features_semantics"] = optimizers.features_semantics
        return optimizers_dict
    
    def update_params_from_dict(self, params: Dict[str, nn.Parameter]):
        """Update model parameters from dictionary after strategy operations.
        
        This is necessary because strategy.step_post_backward() modifies the params
        dictionary in-place, potentially replacing Parameter objects when Gaussians
        are split, duplicated, or pruned.
        
        Args:
            params: Dictionary containing updated parameters from strategy
        """
        self._means = params["means"]
        self._scales = params["scales"]
        self._quats = params["quats"]
        self._opacities = params["opacities"]
        self._features_dc = params["features_dc"]
        self._features_rest = params["features_rest"]
        if "features_semantics" in params:
            self._features_semantics = params["features_semantics"]
    
    def create_optimizers(
        self,
        lr_means: float = 0.00016,
        lr_scales: float = 0.005,
        lr_quats: float = 0.001,
        lr_opacities: float = 0.05,
        lr_sh: float = 0.0025,
        lr_semantics: float | None = None,
        means_lr_multiplier: float = 5.0,
    ) -> GSOptimizers:
        """Create optimizers for all model parameters.
        
        Args:
            lr_means: Base learning rate for Gaussian positions
            lr_scales: Learning rate for Gaussian scales
            lr_quats: Learning rate for Gaussian rotations
            lr_opacities: Learning rate for Gaussian opacities
            lr_sh: Learning rate for spherical harmonics coefficients
            lr_semantics: Learning rate for semantic features (defaults to lr_sh)
            means_lr_multiplier: Multiplier for means learning rate (default: 5.0)
            
        Returns:
            GSOptimizers: Named tuple containing optimizers for each parameter group
        """
        if lr_semantics is None:
            lr_semantics = lr_sh

        semantics_optimizer = None
        if hasattr(self, '_features_semantics'):
            semantics_optimizer = torch.optim.Adam([self._features_semantics], lr=lr_semantics)

        return GSOptimizers(
            means=torch.optim.Adam([self._means], lr=lr_means * means_lr_multiplier),
            scales=torch.optim.Adam([self._scales], lr=lr_scales),
            quats=torch.optim.Adam([self._quats], lr=lr_quats),
            opacities=torch.optim.Adam([self._opacities], lr=lr_opacities),
            features_dc=torch.optim.Adam([self._features_dc], lr=lr_sh),
            features_rest=torch.optim.Adam([self._features_rest], lr=lr_sh*0.1),
            features_semantics=semantics_optimizer,
        )
    
    def construct_list_of_attributes(self):
        """Build the list of PLY attribute names matching the original 3DGS convention."""
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        for i in range(self._features_dc.shape[1] * self._features_dc.shape[2]):
            l.append(f'f_dc_{i}')
        for i in range(self._features_rest.shape[1] * self._features_rest.shape[2]):
            l.append(f'f_rest_{i}')
        l.append('opacity')
        for i in range(self._scales.shape[1]):
            l.append(f'scale_{i}')
        for i in range(self._quats.shape[1]):
            l.append(f'rot_{i}')
        return l

    def save_ply(self, path):
        """Save Gaussians to PLY format compatible with the original 3DGS project.

        All parameters are saved in their **raw pre-activation** form so that
        the original 3DGS ``load_ply`` (which applies its own activations) can
        consume them directly.

        SH features are transposed before flattening to match the original
        column-major ordering: ``[N, K, 3] -> transpose -> [N, 3, K] -> flatten``.
        """
        try:
            import numpy as np
            from plyfile import PlyData, PlyElement
        except ImportError:
            logger.warning("[yellow]⚠ Warning:[/yellow] plyfile not installed. Install with: pip install plyfile")
            return

        import os
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)

        # Convert to numpy — all values are RAW (pre-activation)
        xyz = self._means.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        # Transpose SH features to match original 3DGS column ordering:
        # Original stores [N, C, SH] and transposes to [N, SH, C] before flatten
        # Our storage is [N, SH, C], so we transpose to [N, C, SH] then flatten
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        # Raw pre-activation values (NOT sigmoid/exp/normalize)
        opacities = self._opacities.detach().cpu().numpy()
        scale = self._scales.detach().cpu().numpy()
        rotation = self._quats.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        attributes = np.concatenate(
            (xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1
        )
        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        elements[:] = list(map(tuple, attributes))

        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)
        logger.info(f"[green]✓ Saved {xyz.shape[0]:,} Gaussians to[/green] {path}")

    def to_original_gs_state_dict(self) -> dict:
        """Return a dict of model tensors keyed by the original 3DGS parameter names.

        This maps the custom parameter names to the original ones:
        - ``_means``      → ``_xyz``
        - ``_scales``     → ``_scaling``
        - ``_quats``      → ``_rotation``
        - ``_opacities``  → ``_opacity``
        - ``_features_dc``  (same)
        - ``_features_rest`` (same)

        All values are raw (pre-activation), exactly as the original 3DGS
        ``capture()`` would return them.
        """
        return {
            '_xyz': self._means.detach(),
            '_features_dc': self._features_dc.detach(),
            '_features_rest': self._features_rest.detach(),
            '_scaling': self._scales.detach(),
            '_rotation': self._quats.detach(),
            '_opacity': self._opacities.detach(),
        }

    def export_to_original_gs_checkpoint(
        self,
        path: str,
        iteration: int = 0,
        spatial_lr_scale: float = 1.0,
    ):
        """Save the model as an original 3DGS checkpoint (``.pth``).

        The file is written in the exact format that the original 3DGS
        ``train.py`` produces::

            torch.save((capture_tuple, iteration), path)

        The ``capture_tuple`` mirrors ``GaussianModel.capture()`` and can be
        consumed by the original ``GaussianModel.restore()``.

        Because this model has no original-style optimizer, placeholder values
        are used for ``max_radii2D``, ``xyz_gradient_accum``, ``denom``, and
        ``optimizer_state_dict``.  These are only needed if you intend to
        **resume training** in the original codebase (and you would call
        ``training_setup`` + ``restore`` anyway, which rebuilds them).

        Args:
            path: Output file path (e.g. ``chkpnt30000.pth``).
            iteration: Iteration number to embed in the checkpoint.
            spatial_lr_scale: Scene-extent value used by the original LR
                scheduler.  Defaults to 1.0 (safe default; override with
                the actual ``scene.cameras_extent`` if you have it).
        """
        import os
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)

        num_pts = self._means.shape[0]
        device = self._means.device

        capture_tuple = (
            self.sh_degree,                                    # active_sh_degree
            self._means.detach(),                              # _xyz
            self._features_dc.detach(),                        # _features_dc
            self._features_rest.detach(),                      # _features_rest
            self._scales.detach(),                             # _scaling
            self._quats.detach(),                              # _rotation
            self._opacities.detach(),                          # _opacity
            torch.zeros(num_pts, device=device),               # max_radii2D (placeholder)
            torch.zeros(num_pts, 1, device=device),            # xyz_gradient_accum (placeholder)
            torch.zeros(num_pts, 1, device=device),            # denom (placeholder)
            {},                                                # optimizer state_dict (empty placeholder)
            spatial_lr_scale,                                  # spatial_lr_scale
        )

        torch.save((capture_tuple, iteration), path)
        logger.info(
            f"[green]✓ Exported original 3DGS checkpoint:[/green] {path} "
            f"({num_pts:,} Gaussians, iter {iteration})"
        )

    @classmethod
    def load_ply(
        cls,
        path,
        sh_degree=3,
        device='cuda',
        console=None,
    ):
        """Load a PLY file saved by the original 3DGS project into a GaussianModel.

        This handles the inverse of the original ``save_ply``:
        - SH features are un-transposed back to ``[N, K, 3]``.
        - Raw pre-activation values are assigned directly to internal parameters.

        Args:
            path: Path to PLY file.
            sh_degree: Maximum SH degree. If None, inferred from the PLY file.
            device: Target device.
            console: Optional rich console for display.

        Returns:
            A populated GaussianModel instance.
        """
        import numpy as np
        from plyfile import PlyData
        import torch.nn as nn

        plydata = PlyData.read(path)
        el = plydata.elements[0]

        # --- Positions ---
        xyz = np.stack(
            (np.asarray(el["x"]), np.asarray(el["y"]), np.asarray(el["z"])),
            axis=1,
        )  # [N, 3]
        num_pts = xyz.shape[0]

        # --- Opacity (raw logit) ---
        opacities = np.asarray(el["opacity"])[..., np.newaxis]  # [N, 1]

        # --- SH DC features ---
        # Original stores [N, 3, 1] in the PLY (transposed from [N, 1, 3]).
        # We read into [N, 3, 1] then transpose back to [N, 1, 3].
        features_dc = np.zeros((num_pts, 3, 1))
        features_dc[:, 0, 0] = np.asarray(el["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(el["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(el["f_dc_2"])

        # --- SH rest features ---
        extra_f_names = sorted(
            [p.name for p in el.properties if p.name.startswith("f_rest_")],
            key=lambda x: int(x.split("_")[-1]),
        )
        if sh_degree is None:
            # Infer SH degree: total SH coeffs (including DC) = (degree+1)^2
            total_sh = len(extra_f_names) // 3 + 1  # +1 for DC
            sh_degree = int(total_sh**0.5) - 1

        features_extra = np.zeros((num_pts, len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(el[attr_name])
        # Original saves as [N, 3, K] (after transpose). We reshape to [N, 3, K]
        # then transpose back to [N, K, 3] which is our internal layout.
        num_rest_coeffs = (sh_degree + 1) ** 2 - 1
        features_extra = features_extra.reshape((num_pts, 3, num_rest_coeffs))  # [N, 3, K]

        # --- Scales (raw log) ---
        scale_names = sorted(
            [p.name for p in el.properties if p.name.startswith("scale_")],
            key=lambda x: int(x.split("_")[-1]),
        )
        scales = np.zeros((num_pts, len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(el[attr_name])

        # --- Rotations (raw quaternion) ---
        rot_names = sorted(
            [p.name for p in el.properties if p.name.startswith("rot")],
            key=lambda x: int(x.split("_")[-1]),
        )
        rots = np.zeros((num_pts, len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(el[attr_name])

        # --- Build the model ---
        # Use dummy init points/colors — we'll overwrite all parameters
        dummy_points = torch.zeros((num_pts, 3), device=device)
        dummy_colors = torch.zeros((num_pts, 3), device=device)
        model = cls(
            dummy_points,
            dummy_colors,
            sh_degree=sh_degree,
            console=console,
        ).to(device)

        # Assign raw parameters directly
        model._means = nn.Parameter(
            torch.tensor(xyz, dtype=torch.float32, device=device)
        )
        # Transpose DC: [N, 3, 1] -> [N, 1, 3] to match our internal layout
        model._features_dc = nn.Parameter(
            torch.tensor(features_dc, dtype=torch.float32, device=device)
            .transpose(1, 2)
            .contiguous()
        )
        # Transpose rest: [N, 3, K] -> [N, K, 3] to match our internal layout
        model._features_rest = nn.Parameter(
            torch.tensor(features_extra, dtype=torch.float32, device=device)
            .transpose(1, 2)
            .contiguous()
        )
        model._opacities = nn.Parameter(
            torch.tensor(opacities, dtype=torch.float32, device=device)
        )
        model._scales = nn.Parameter(
            torch.tensor(scales, dtype=torch.float32, device=device)
        )
        model._quats = nn.Parameter(
            torch.tensor(rots, dtype=torch.float32, device=device)
        )
        # Reset view count buffer to match new number of points
        model.view_count = torch.zeros(num_pts, device=device)

        logger.info(
            f"[green]✓ Loaded {num_pts:,} Gaussians from PLY:[/green] {path}"
        )
        return model

    @classmethod
    def from_original_gs_checkpoint(
        cls,
        checkpoint_path,
        device='cuda',
        console=None,
    ):
        """Load an original 3DGS checkpoint (``.pth``) into a custom GaussianModel.

        The original checkpoint format is::

            torch.save((capture_tuple, iteration), path)

        where ``capture_tuple`` is the output of ``GaussianModel.capture()``:
        ``(active_sh_degree, _xyz, _features_dc, _features_rest, _scaling,
        _rotation, _opacity, max_radii2D, xyz_gradient_accum, denom,
        opt_dict, spatial_lr_scale)``.

        This method maps the original parameter names to the custom ones and
        returns a fully populated model.

        Args:
            checkpoint_path: Path to ``.pth`` file.
            device: Target device.
            console: Optional rich console.

        Returns:
            A tuple ``(model, iteration)`` where ``model`` is the custom
            GaussianModel and ``iteration`` is the training iteration from
            the checkpoint.
        """
        import torch.nn as nn

        data = torch.load(checkpoint_path, map_location=device, weights_only=False)
        capture_tuple, iteration = data

        (
            active_sh_degree,
            _xyz,
            _features_dc,
            _features_rest,
            _scaling,
            _rotation,
            _opacity,
            _max_radii2D,
            _xyz_gradient_accum,
            _denom,
            _opt_dict,
            _spatial_lr_scale,
        ) = capture_tuple

        sh_degree = active_sh_degree
        num_pts = _xyz.shape[0]

        # Build model with dummy data
        dummy_points = torch.zeros((num_pts, 3), device=device)
        dummy_colors = torch.zeros((num_pts, 3), device=device)
        model = cls(
            dummy_points,
            dummy_colors,
            sh_degree=sh_degree,
            console=console,
        ).to(device)

        # Map original param names → custom param names
        model._means = nn.Parameter(_xyz.to(device))
        model._features_dc = nn.Parameter(_features_dc.to(device))
        model._features_rest = nn.Parameter(_features_rest.to(device))
        model._scales = nn.Parameter(_scaling.to(device))
        model._quats = nn.Parameter(_rotation.to(device))
        model._opacities = nn.Parameter(_opacity.to(device))
        model.view_count = torch.zeros(num_pts, device=device)

        logger.info(
            f"[green]✓ Loaded original 3DGS checkpoint:[/green] {checkpoint_path} "
            f"({num_pts:,} Gaussians, iter {iteration})"
        )
        return model, iteration

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path,
        device='cuda',
        sh_degree=None,
        train_semantics=False,
        semantics_dim=3,
        console=None,
        strict=False,
        return_checkpoint=False,
    ):
        """Construct a GaussianModel from a training checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file.
            device: Target device.
            sh_degree: Optional SH degree override. If None, inferred from checkpoint.
            train_semantics: Whether to initialize semantic features if not inferable.
            semantics_dim: Semantic feature dimension fallback.
            console: Optional rich console.
            strict: Whether to enforce strict state-dict loading.
            return_checkpoint: If True, returns (model, checkpoint_dict).

        Returns:
            model or (model, checkpoint)
        """
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if 'model_state_dict' not in checkpoint:
            raise KeyError(f"Checkpoint missing 'model_state_dict': {checkpoint_path}")

        state_dict = dict(checkpoint['model_state_dict'])

        if '_means' not in state_dict:
            raise KeyError("Checkpoint state_dict missing '_means'")

        num_gaussians = state_dict['_means'].shape[0]

        if sh_degree is None:
            features_rest = state_dict.get('_features_rest', None)
            if features_rest is not None:
                sh_degree = int((features_rest.shape[1] + 1) ** 0.5) - 1
            else:
                sh_degree = 3

        if '_features_semantics' in state_dict:
            train_semantics = True
            semantics_dim = state_dict['_features_semantics'].shape[1]

        dummy_points = torch.zeros((num_gaussians, 3), device=device)
        dummy_colors = torch.zeros((num_gaussians, 3), device=device)
        model = cls(
            dummy_points,
            dummy_colors,
            sh_degree=sh_degree,
            train_semantics=train_semantics,
            semantics_dim=semantics_dim,
            console=console,
        ).to(device)

        if 'view_count' in state_dict:
            del state_dict['view_count']

        model.load_state_dict(state_dict, strict=strict)
        model.view_count = torch.zeros(len(model._means), device=device)

        logger.info(f"[green]✓ Loaded model from[/green] {checkpoint_path}")
        if 'iteration' in checkpoint:
            logger.debug(f"[dim]Checkpoint iteration: {checkpoint['iteration']}[/dim]")
        logger.debug(f"[dim]Number of Gaussians: {num_gaussians:,}[/dim]")

        if return_checkpoint:
            return model, checkpoint
        return model

    @classmethod
    def load_checkpoint(cls, checkpoint_path, device='cuda'):
        """Load model from checkpoint (backward-compatible helper)."""
        return cls.from_checkpoint(
            checkpoint_path=checkpoint_path,
            device=device,
            return_checkpoint=False,
        )

    @classmethod
    def resume_from_checkpoint(
        cls,
        checkpoint_path,
        device='cuda',
        sh_degree=None,
        train_semantics=False,
        semantics_dim=3,
        console=None,
        strict=False,
    ) -> Tuple["GaussianModel", Dict]:
        """Load model and checkpoint dict for training resume workflows."""
        return cls.from_checkpoint(
            checkpoint_path=checkpoint_path,
            device=device,
            sh_degree=sh_degree,
            train_semantics=train_semantics,
            semantics_dim=semantics_dim,
            console=console,
            strict=strict,
            return_checkpoint=True,
        )
