import torch.nn as nn
from gsplat import spherical_harmonics
import torch
from gs_types import Parameters, GSOptimizers
from strategy import DefaultStrategy, ViewportInfo
from typing import Optional, Dict

def inverse_sigmoid(x):
    return torch.log(x / (1 - x))

class GaussianModel(nn.Module):
    def __init__(self, init_points, init_colors, sh_degree=3, console=None, strategy=None):
        super().__init__()
        num_points = init_points.shape[0]
        self.sh_degree = sh_degree
        self.console = console
        
        # Initialize strategy (defaults to DefaultStrategy if not provided)
        if strategy is None:
            strategy = DefaultStrategy(verbose=False)
        self.strategy = strategy

        # --- Learnable Parameters ---
        # 1. Position: The center of the gaussian
        self._means = nn.Parameter(init_points)
        
        # 2. Scale: Log-space to ensure positivity
        # Initialize as isotropic small spheres
        dist_to_nearest = torch.ones(num_points, device=init_points.device) * 0.1 
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
        self._features_rest = nn.Parameter(torch.zeros(num_points, (sh_degree + 1)**2 - 1, 3, device=init_points.device))
        
        # --- Gradient Accumulation Buffers (for densification) ---
        # These track gradients over multiple iterations to decide when to densify/prune
        self.register_buffer('xyz_gradient_accum', torch.zeros(num_points, 1, device=init_points.device))
        self.register_buffer('denom', torch.zeros(num_points, 1, device=init_points.device))
        self.register_buffer('max_radii2D', torch.zeros(num_points, device=init_points.device))
        
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
    
    def add_densification_stats(self, viewspace_point_tensor, update_filter, gaussian_ids=None):
        """
        [DEPRECATED] Use strategy.step_post_backward() instead.
        
        Legacy method kept for backward compatibility.
        Accumulates gradient statistics in the old buffer format.
        
        Args:
            viewspace_point_tensor: Screen-space positions with gradients [N_visible, 2]
            update_filter: Boolean mask for which of the visible Gaussians to update [N_visible]
            gaussian_ids: Optional indices mapping visible Gaussians to model Gaussians [N_visible]
                         If None, assumes viewspace_point_tensor covers all model Gaussians
        """
        if gaussian_ids is not None:
            # Frustum culling is active - map visible Gaussians to full model
            visible_grads = torch.norm(
                viewspace_point_tensor.grad[update_filter, :2], dim=-1, keepdim=True
            )
            # Get indices of Gaussians to update in the full model
            full_model_indices = gaussian_ids[update_filter]
            self.xyz_gradient_accum[full_model_indices] += visible_grads
            self.denom[full_model_indices] += 1
        else:
            # No culling - update_filter applies directly to full model
            self.xyz_gradient_accum[update_filter] += torch.norm(
                viewspace_point_tensor.grad[update_filter, :2], dim=-1, keepdim=True
            )
            self.denom[update_filter] += 1
    
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
    
    def reset_densification_stats(self):
        """Reset accumulated gradient statistics."""
        self.xyz_gradient_accum.zero_()
        self.denom.zero_()
        self.max_radii2D.zero_()
        self.view_count.zero_()

    def get_params_dict(self) -> Dict[str, nn.Parameter]:
        """Get parameters as dictionary for strategy interface."""
        return {
            "means": self._means,
            "scales": self._scales,
            "quats": self._quats,
            "opacities": self._opacities,
            "features_dc": self._features_dc,
            "features_rest": self._features_rest,
        }
    
    def get_optimizers_dict(self, optimizers: GSOptimizers) -> Dict[str, torch.optim.Optimizer]:
        """Convert GSOptimizers to dictionary for strategy interface."""
        return {
            "means": optimizers.means,
            "scales": optimizers.scales,
            "quats": optimizers.quats,
            "opacities": optimizers.opacities,
            "features_dc": optimizers.features_dc,
            "features_rest": optimizers.features_rest,
        }
    
    def update_params_from_dict(self, params: Dict[str, nn.Parameter]):
        """Update model parameters from dictionary after strategy operations."""
        self._means = params["means"]
        self._scales = params["scales"]
        self._quats = params["quats"]
        self._opacities = params["opacities"]
        self._features_dc = params["features_dc"]
        self._features_rest = params["features_rest"]
    
    def densify_and_prune(self, grad_threshold, scene_extent, max_screen_size, optimizers: GSOptimizers, 
                         verbosity=1, max_world_scale=None, min_view_count=None, iteration=0,
                         viewport_info: Optional[ViewportInfo] = None):
        """
        Densify and prune Gaussians using the strategy system.
        
        This method now delegates to the configured strategy (DefaultStrategy by default).
        For backward compatibility, it supports the old parameter names but maps them
        to the strategy's configuration.
        
        Args:
            grad_threshold: [LEGACY] Gradient threshold (mapped to strategy's grow_grad2d)
            scene_extent: Scene extent for normalization
            max_screen_size: [LEGACY] Maximum screen size in pixels
            optimizers: Optimizer group
            verbosity: Logging verbosity level
            max_world_scale: [LEGACY] Max world-space scale (as fraction of scene_extent)
            min_view_count: [LEGACY] Minimum view count for multi-view consistency
            iteration: Current training iteration
            viewport_info: Camera/viewport information (width, height, n_cameras)
                          If None, uses default values (may affect gradient normalization)
        
        Note: To use the new strategy system properly, call strategy.step_post_backward()
              directly from your training loop after loss.backward().
        """
        # Use default viewport info if not provided (for backward compatibility)
        if viewport_info is None:
            viewport_info = ViewportInfo(width=800, height=600, n_cameras=1)
            if verbosity >= 1:
                print("  ⚠️  Warning: viewport_info not provided, using default (800x600).")
                print("     For proper gradient normalization, pass ViewportInfo to densify_and_prune()")
        
        # Override strategy verbosity based on function parameter
        old_verbose = self.strategy.verbose
        self.strategy.verbose = (verbosity >= 2)
        
        # Get params and optimizers as dicts for strategy interface
        params = self.get_params_dict()
        optimizers_dict = self.get_optimizers_dict(optimizers)
        
        # Call strategy's refinement logic
        self.strategy.step_post_backward(
            params=params,
            optimizers=optimizers_dict,
            scene_scale=scene_extent,
            iteration=iteration,
            viewport_info=viewport_info,
            radii=self.max_radii2D,
        )
        
        # Update model parameters from strategy (they may have changed)
        self.update_params_from_dict(params)
        
        # Restore original verbosity
        self.strategy.verbose = old_verbose
        
        # Legacy: Apply additional pruning criteria not in default strategy
        # (for backward compatibility with your custom pruning logic)
        if min_view_count is not None and iteration > 2000:
            mask_prune = self.view_count < min_view_count
            if mask_prune.any():
                n_pruned = mask_prune.sum().item()
                params = self.get_params_dict()
                optimizers_dict = self.get_optimizers_dict(optimizers)
                
                from strategy import ops
                params = ops.remove(params, optimizers_dict, self.strategy.state, mask_prune)
                self.update_params_from_dict(params)
                
                if verbosity >= 2:
                    print(f"  🗑️  Pruned {n_pruned:,} Gaussians (low view count)")
        
        torch.cuda.empty_cache() 
    
    def _append_params(self, mask, new_means, new_scales, optimizers: GSOptimizers, repeat=1):
        # Implementation Detail:
        # You must cat() new tensors to the existing parameters
        # AND manually update the optimizer's internal state (exp_avg, exp_avg_sq)
        # to match the new tensor sizes.
        
        # Clone other attributes from the masked Gaussians
        new_quats = self._quats[mask].repeat(repeat, 1)
        new_opacities = self._opacities[mask].repeat(repeat, 1)
        new_features_dc = self._features_dc[mask].repeat(repeat, 1, 1)
        new_features_rest = self._features_rest[mask].repeat(repeat, 1, 1)
        
        # Store old parameters for optimizer state transfer
        old_params = Parameters(
            means=self._means,
            scales=self._scales,
            quats=self._quats,
            opacities=self._opacities,
            features_dc=self._features_dc,
            features_rest=self._features_rest
        )
        
        # Concatenate to existing parameters
        self._means = nn.Parameter(torch.cat([self._means, new_means], dim=0))
        self._scales = nn.Parameter(torch.cat([self._scales, new_scales], dim=0))
        self._quats = nn.Parameter(torch.cat([self._quats, new_quats], dim=0))
        self._opacities = nn.Parameter(torch.cat([self._opacities, new_opacities], dim=0))
        self._features_dc = nn.Parameter(torch.cat([self._features_dc, new_features_dc], dim=0))
        self._features_rest = nn.Parameter(torch.cat([self._features_rest, new_features_rest], dim=0))
        
        # Update buffers for gradient accumulation
        n_new = new_means.shape[0]
        self.xyz_gradient_accum = torch.cat([
            self.xyz_gradient_accum,
            torch.zeros(n_new, 1, device=new_means.device)
        ], dim=0)
        self.denom = torch.cat([
            self.denom,
            torch.zeros(n_new, 1, device=new_means.device)
        ], dim=0)
        self.max_radii2D = torch.cat([
            self.max_radii2D,
            torch.zeros(n_new, device=new_means.device)
        ], dim=0)
        self.view_count = torch.cat([
            self.view_count,
            torch.zeros(n_new, device=new_means.device)
        ], dim=0)
        
        # Map old params to new params and new values
        param_map = Parameters(
            means=(self._means, new_means),
            scales=(self._scales, new_scales),
            quats=(self._quats, new_quats),
            opacities=(self._opacities, new_opacities),
            features_dc=(self._features_dc, new_features_dc),
            features_rest=(self._features_rest, new_features_rest)
        )

        for opt in optimizers.__dict__.values():
            for group in opt.param_groups:
                for i, param in enumerate(group['params']):
                    # Use identity check to avoid tensor comparison
                    for attr_name in ['means', 'scales', 'quats', 'opacities', 'features_dc', 'features_rest']:
                        if param is old_params[attr_name]:
                            new_param, new_vals = param_map[attr_name]
                            # Replace parameter in param group
                            group['params'][i] = new_param
                            
                            # Update optimizer state
                            if param in opt.state:
                                old_state = opt.state.pop(param)
                                new_state = {}
                                
                                # Transfer and extend state
                                for key, value in old_state.items():
                                    if isinstance(value, torch.Tensor):
                                        # Pad with zeros for the new parameters (ensure same device)
                                        new_state[key] = torch.cat([
                                            value,
                                            torch.zeros_like(new_vals, device=value.device)
                                        ], dim=0)
                                    else:
                                        new_state[key] = value
                                
                                opt.state[new_param] = new_state
                            break
    
    def save_ply(self, path):
        """Save Gaussians to PLY format for visualization."""
        try:
            import numpy as np
            from plyfile import PlyData, PlyElement
        except ImportError:
            print("Warning: plyfile not installed. Install with: pip install plyfile")
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
        print(f"Saved {xyz.shape[0]} Gaussians to {path}")
    
    @classmethod
    def load_checkpoint(cls, checkpoint_path, device='cuda'):
        """Load model from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Create a dummy model to get structure (will be overwritten)
        dummy_points = torch.zeros((1, 3), device=device)
        dummy_colors = torch.zeros((1, 3), device=device)
        model = cls(dummy_points, dummy_colors)
        
        # Load state dict
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        
        print(f"Loaded model from {checkpoint_path}")
        if 'iteration' in checkpoint:
            print(f"Checkpoint iteration: {checkpoint['iteration']}")
        if 'num_gaussians' in checkpoint:
            print(f"Number of Gaussians: {checkpoint['num_gaussians']}")
        
        return model
    
    def _remove_params(self, mask, optimizers: GSOptimizers):
        """
        Remove Gaussians based on a boolean mask.
        mask: True for Gaussians to REMOVE
        """
        # Create keep mask (inverse of remove mask)
        keep_mask = ~mask
        
        # Store old parameters for optimizer state transfer
        old_params = Parameters(
            means=self._means,
            scales=self._scales,
            quats=self._quats,
            opacities=self._opacities,
            features_dc=self._features_dc,
            features_rest=self._features_rest
        )
        
        # Filter parameters to keep only non-pruned Gaussians
        self._means = nn.Parameter(self._means[keep_mask])
        self._scales = nn.Parameter(self._scales[keep_mask])
        self._quats = nn.Parameter(self._quats[keep_mask])
        self._opacities = nn.Parameter(self._opacities[keep_mask])
        self._features_dc = nn.Parameter(self._features_dc[keep_mask])
        self._features_rest = nn.Parameter(self._features_rest[keep_mask])
        
        # Update buffers for gradient accumulation
        self.xyz_gradient_accum = self.xyz_gradient_accum[keep_mask]
        self.denom = self.denom[keep_mask]
        self.max_radii2D = self.max_radii2D[keep_mask]
        self.view_count = self.view_count[keep_mask]
        
        # Map old params to new params
        param_map = Parameters(
            means=self._means,
            scales=self._scales,
            quats=self._quats,
            opacities=self._opacities,
            features_dc=self._features_dc,
            features_rest=self._features_rest
        )
        
        # Update optimizer
        for opt in optimizers.__dict__.values():
            new_param_list = []
            for group in opt.param_groups:
                new_group_params = []
                for param in group['params']:
                    # Use identity check to avoid tensor comparison
                    if any(param is old_param for old_param in old_params.__dict__.values()):
                        # Find which parameter this is
                        for attr_name, old_param in old_params.__dict__.items():
                            if param is old_param:
                                new_param = param_map[attr_name]
                                new_group_params.append(new_param)
                                break
                    else:
                        new_group_params.append(param)
                group['params'] = new_group_params
                new_param_list.extend(new_group_params)
            
            # Clean up optimizer state to remove pruned parameters
            keys_to_remove = []
            for key in opt.state.keys():
                # Use identity check to avoid tensor comparison
                if not any(key is param for param in new_param_list):
                    keys_to_remove.append(key)
            for key in keys_to_remove:
                del opt.state[key]
