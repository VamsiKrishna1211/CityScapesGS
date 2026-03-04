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
        return None
    
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
    
    def save_ply(self, path):
        """Save Gaussians to PLY format for visualization."""
        
        try:
            import numpy as np
            from plyfile import PlyData, PlyElement
        except ImportError:
            logger.warning("[yellow]⚠ Warning:[/yellow] plyfile not installed. Install with: pip install plyfile")
            return
        
        # Convert to numpy
        xyz = self._means.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().cpu().numpy().reshape(xyz.shape[0], -1)
        f_rest = self._features_rest.detach().cpu().numpy().reshape(xyz.shape[0], -1)
        opacities = self.opacities.detach().cpu().numpy()
        scale = self.scales.detach().cpu().numpy()
        rotation = self.quats.detach().cpu().numpy()
        
        # Prepare attributes
        dtype_full = [(attribute, 'f4') for attribute in ['x', 'y', 'z', 'nx', 'ny', 'nz']]
        dtype_full += [(attribute, 'f4') for attribute in ['f_dc_0', 'f_dc_1', 'f_dc_2']]
        dtype_full += [(attribute, 'f4') for attribute in 
                       [f'f_rest_{i}' for i in range(f_rest.shape[1])]]
        dtype_full += [('opacity', 'f4')]
        dtype_full += [(attribute, 'f4') for attribute in ['scale_0', 'scale_1', 'scale_2']]
        dtype_full += [(attribute, 'f4') for attribute in ['rot_0', 'rot_1', 'rot_2', 'rot_3']]
        
        attributes = np.concatenate(
            (xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1
        )
        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        elements[:] = list(map(tuple, attributes))
        
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)
        logger.info(f"[green]✓ Saved {xyz.shape[0]:,} Gaussians to[/green] {path}")
    
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
