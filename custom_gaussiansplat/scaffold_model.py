import logging
import os
from functools import reduce
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from plyfile import PlyData, PlyElement
from simple_knn._C import distCUDA2
from torch_scatter import scatter_max

from gs_types import GSOptimizers, Parameters

# Module-level logger
logger = logging.getLogger("cityscape_gs.scaffold_model")

def inverse_sigmoid(x):
    return torch.log(x / (1 - x))

class Embedding(nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.embedding = torch.nn.Embedding(in_dim, out_dim)

    def forward(self, in_tensor: torch.Tensor) -> torch.Tensor:
        return self.embedding(in_tensor)

class ScaffoldModel(nn.Module):
    def __init__(self, 
                 init_points: torch.Tensor,
                 feat_dim: int = 32,
                 n_offsets: int = 10,
                 voxel_size: float = 0.01,
                 update_depth: int = 3,
                 update_init_factor: int = 100,
                 update_hierachy_factor: int = 4,
                 use_feat_bank: bool = False,
                 appearance_dim: int = 32,
                 add_opacity_dist: bool = False,
                 add_cov_dist: bool = False,
                 add_color_dist: bool = False,
                 sh_degree: int = 3,
                 console=None):
        super().__init__()
        self.console = console
        self.feat_dim = feat_dim
        self.n_offsets = n_offsets
        self.voxel_size = voxel_size
        self.update_depth = update_depth
        self.update_init_factor = update_init_factor
        self.update_hierachy_factor = update_hierachy_factor
        self.use_feat_bank = use_feat_bank
        self.appearance_dim = appearance_dim
        self.add_opacity_dist = add_opacity_dist
        self.add_cov_dist = add_cov_dist
        self.add_color_dist = add_color_dist
        self.sh_degree = sh_degree

        # --- Parameters ---
        self._anchor = nn.Parameter(torch.empty(0))
        self._offset = nn.Parameter(torch.empty(0))
        self._anchor_feat = nn.Parameter(torch.empty(0))
        self._scaling = nn.Parameter(torch.empty(0))
        self._rotation = nn.Parameter(torch.empty(0))
        self._opacity = nn.Parameter(torch.empty(0))
        
        # --- Buffers for densification ---
        self.register_buffer('opacity_accum', torch.empty(0))
        self.register_buffer('offset_gradient_accum', torch.empty(0))
        self.register_buffer('offset_denom', torch.empty(0))
        self.register_buffer('anchor_denom', torch.empty(0))
        self.register_buffer('max_radii2D', torch.empty(0))
        
        # LoD compatibility (Scaffold-GS doesn't use LoD but needs this for unified API)
        self.lod_offsets = [0]  # Will be updated in create_from_pcd

        # --- MLPs ---
        self.opacity_dist_dim = 1 if self.add_opacity_dist else 0
        self.mlp_opacity = nn.Sequential(
            nn.Linear(feat_dim + 3 + self.opacity_dist_dim, feat_dim),
            nn.ReLU(True),
            nn.Linear(feat_dim, n_offsets),
            nn.Tanh()
        )

        self.cov_dist_dim = 1 if self.add_cov_dist else 0
        self.mlp_cov = nn.Sequential(
            nn.Linear(feat_dim + 3 + self.cov_dist_dim, feat_dim),
            nn.ReLU(True),
            nn.Linear(feat_dim, 7 * self.n_offsets),
        )

        self.color_dist_dim = 1 if self.add_color_dist else 0
        self.mlp_color = nn.Sequential(
            nn.Linear(feat_dim + 3 + self.color_dist_dim + self.appearance_dim, feat_dim),
            nn.ReLU(True),
            nn.Linear(feat_dim, 3 * self.n_offsets),
            nn.Sigmoid()
        )

        if self.use_feat_bank:
            self.mlp_feature_bank = nn.Sequential(
                nn.Linear(3 + 1, feat_dim),
                nn.ReLU(True),
                nn.Linear(feat_dim, 3),
                nn.Softmax(dim=1)
            )
        
        self.embedding_appearance = None

        if init_points is not None and len(init_points) > 0:
            self.create_from_pcd(init_points)

    def set_appearance(self, num_cameras: int):
        if self.appearance_dim > 0:
            self.embedding_appearance = Embedding(num_cameras, self.appearance_dim).to(self._anchor.device)

    @property
    def anchors(self): return self._anchor

    @property
    def means(self): return self._anchor
    
    @property
    def anchor_feats(self): return self._anchor_feat

    @property
    def scales(self): return torch.exp(self._scaling)
    
    @property
    def quats(self): return torch.nn.functional.normalize(self._rotation)
    
    @property
    def opacities(self): return torch.sigmoid(self._opacity)

    @property
    def features_dc(self): return self._anchor_feat

    @property
    def features_rest(self): return self._offset

    @property
    def dc_rgb(self):
        # Return a neutral gray color for anchors [N, 1, 3]
        return torch.ones((self._anchor.shape[0], 1, 3), device=self._anchor.device) * 0.5

    @property
    def sh(self):
        # Return DC-only SH (gray) [N, 1, 3]
        return torch.ones((self._anchor.shape[0], 1, 3), device=self._anchor.device) * 0.5

    def get_params_dict(self) -> Dict[str, nn.Parameter]:
        return {
            "anchor": self._anchor,
            "scaling": self._scaling,
            "rotation": self._rotation,
            "opacity": self._opacity,
            "anchor_feat": self._anchor_feat,
            "offset": self._offset,
        }

    def get_optimizers_dict(self, optimizers: GSOptimizers) -> Dict[str, torch.optim.Optimizer]:
        d = {
            "anchor": optimizers.means,
            "scaling": optimizers.scales,
            "rotation": optimizers.quats,
            "opacity": optimizers.opacities,
            "anchor_feat": optimizers.features_dc,
            "offset": optimizers.features_rest,
        }
        if optimizers.extra:
            d.update(optimizers.extra)
        return d

    def update_params_from_dict(self, params: Dict[str, nn.Parameter]):
        self._anchor = params["anchor"]
        self._scaling = params["scaling"]
        self._rotation = params["rotation"]
        self._opacity = params["opacity"]
        self._anchor_feat = params["anchor_feat"]
        self._offset = params["offset"]

    def create_from_pcd(self, points: torch.Tensor):
        num_init_points = points.shape[0]
        
        if self.voxel_size <= 0:
            init_dist = distCUDA2(points.contiguous()).float()
            median_dist, _ = torch.kthvalue(init_dist, int(init_dist.shape[0]*0.5))
            self.voxel_size = torch.sqrt(median_dist).item()
            logger.info(f"Auto-calculated voxel_size: {self.voxel_size}")

        # Voxelization sample (simplified)
        points_np = points.detach().cpu().numpy()
        np.random.shuffle(points_np)
        points_np = np.unique(np.round(points_np / self.voxel_size), axis=0) * self.voxel_size
        fused_point_cloud = torch.tensor(points_np).float().to(points.device)
        num_points = fused_point_cloud.shape[0]
        
        logger.info(f"Number of anchors after voxelization: {num_points}")

        offsets = torch.zeros((num_points, self.n_offsets, 3), device=points.device)
        anchors_feat = torch.zeros((num_points, self.feat_dim), device=points.device)
        
        dist2 = torch.clamp_min(distCUDA2(fused_point_cloud.contiguous()).float(), 1e-7)
        scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 6)
        
        rots = torch.zeros((num_points, 4), device=points.device)
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((num_points, 1), device=points.device))

        self._anchor = nn.Parameter(fused_point_cloud)
        self._offset = nn.Parameter(offsets)
        self._anchor_feat = nn.Parameter(anchors_feat)
        self._scaling = nn.Parameter(scales)
        self._rotation = nn.Parameter(rots)
        self._opacity = nn.Parameter(opacities)
        
        self.max_radii2D = torch.zeros(num_points, device=points.device)
        self.opacity_accum = torch.zeros((num_points, 1), device=points.device)
        self.anchor_denom = torch.zeros((num_points, 1), device=points.device)
        self.offset_gradient_accum = torch.zeros((num_points * self.n_offsets, 1), device=points.device)
        self.offset_denom = torch.zeros((num_points * self.n_offsets, 1), device=points.device)
        
        # Update LoD info for compatibility
        self.lod_offsets = [num_points]

    def generate_neural_gaussians(self, viewpoint_camera, visible_mask=None, is_training=True):
        if visible_mask is None:
            visible_mask = torch.ones(self._anchor.shape[0], dtype=torch.bool, device=self._anchor.device)
        
        feat = self._anchor_feat[visible_mask]
        anchor = self._anchor[visible_mask]
        grid_offsets = self._offset[visible_mask]
        grid_scaling = self.scales[visible_mask]

        # Get view properties for anchor
        # Compute camera center on the correct device
        camera_center = viewpoint_camera["camera_center"]
        if not isinstance(camera_center, torch.Tensor):
            camera_center = torch.tensor(camera_center, device=anchor.device, dtype=anchor.dtype)
        elif camera_center.device != anchor.device:
            camera_center = camera_center.to(device=anchor.device, dtype=anchor.dtype)
        
        ob_view = anchor - camera_center
        ob_dist = ob_view.norm(dim=1, keepdim=True)
        ob_view = ob_view / (ob_dist + 1e-10)

        if self.use_feat_bank:
            cat_view = torch.cat([ob_view, ob_dist], dim=1)
            bank_weight = self.mlp_feature_bank(cat_view).unsqueeze(dim=1) # [n, 1, 3]
            
            # Simplified multi-res feat (matching Scaffold-GS)
            feat = feat.unsqueeze(dim=-1)
            feat = feat[:, ::4, :1].repeat([1, 4, 1]) * bank_weight[:, :, :1] + \
                   feat[:, ::2, :1].repeat([1, 2, 1]) * bank_weight[:, :, 1:2] + \
                   feat[:, ::1, :1] * bank_weight[:, :, 2:]
            feat = feat.squeeze(dim=-1)

        cat_local_view = torch.cat([feat, ob_view, ob_dist], dim=1)
        cat_local_view_wodist = torch.cat([feat, ob_view], dim=1)
        
        if self.embedding_appearance is not None:
            camera_indices = torch.ones_like(cat_local_view[:, 0], dtype=torch.long) * viewpoint_camera["uid"]
            appearance = self.embedding_appearance(camera_indices)

        # Get offset's opacity
        if self.add_opacity_dist:
            neural_opacity = self.mlp_opacity(cat_local_view)
        else:
            neural_opacity = self.mlp_opacity(cat_local_view_wodist)

        neural_opacity = neural_opacity.reshape([-1, 1])
        mask = (neural_opacity > 0.0).view(-1)
        opacity = neural_opacity[mask]

        # Get offset's color
        if self.embedding_appearance is not None:
            color_input = torch.cat([cat_local_view if self.add_color_dist else cat_local_view_wodist, appearance], dim=1)
        else:
            color_input = cat_local_view if self.add_color_dist else cat_local_view_wodist
        
        color = self.mlp_color(color_input).reshape([-1, 3])

        # Get offset's covariance
        if self.add_cov_dist:
            scale_rot = self.mlp_cov(cat_local_view)
        else:
            scale_rot = self.mlp_cov(cat_local_view_wodist)
        
        scale_rot = scale_rot.reshape([-1, 7])
        offsets = grid_offsets.view([-1, 3])

        # We need to apply the mask to all generated Gaussians
        # But wait, gsplat/rasterizer expects ALL Gaussians. 
        # Actually, Scaffold-GS filters them out to save computation.
        
        # We'll use a trick: concatenated indexing for speed
        scaling_anchor = torch.cat([grid_scaling, anchor], dim=-1) # [N, 6+3]
        scaling_anchor_repeated = scaling_anchor.unsqueeze(1).repeat(1, self.n_offsets, 1).view(-1, 9)
        
        # Apply mask
        masked_scaling_anchor = scaling_anchor_repeated[mask]
        masked_color = color[mask]
        masked_scale_rot = scale_rot[mask]
        masked_offsets = offsets[mask]

        s_repeat = masked_scaling_anchor[:, :6]
        a_repeat = masked_scaling_anchor[:, 6:9]

        # Post-process
        scaling = s_repeat[:, 3:] * torch.sigmoid(masked_scale_rot[:, :3])
        rot = torch.nn.functional.normalize(masked_scale_rot[:, 3:7])
        
        # Local offsets scaled by anchor scaling
        actual_offsets = masked_offsets * s_repeat[:, :3]
        xyz = a_repeat + actual_offsets

        if is_training:
            return xyz, masked_color, opacity, scaling, rot, neural_opacity, mask
        else:
            return xyz, masked_color, opacity, scaling, rot

    def training_statis(self, viewspace_point_tensor, opacity, update_filter, offset_selection_mask, anchor_visible_mask):
        with torch.no_grad():
            temp_opacity = opacity.clone().view(-1).detach()
            temp_opacity[temp_opacity < 0] = 0
            temp_opacity = temp_opacity.view([-1, self.n_offsets])
            
            self.opacity_accum[anchor_visible_mask] += temp_opacity.sum(dim=1, keepdim=True)
            self.anchor_denom[anchor_visible_mask] += 1

            # Update neural gaussian stats
            # anchor_visible_mask is [N], we need to expand it to [N*k]
            expanded_anchor_mask = anchor_visible_mask.unsqueeze(1).repeat(1, self.n_offsets).view(-1)
            
            # Create combined mask: which neural gaussians (offsets) to track
            # offset_selection_mask tells us which offsets were selected from visible anchors
            combined_mask = torch.zeros(self.offset_gradient_accum.shape[0], dtype=torch.bool, device=self.offset_gradient_accum.device)
            combined_mask[expanded_anchor_mask] = offset_selection_mask
            
            # update_filter is visibility filter (radii > 0) from rasterization
            # We want to update only those that are both selected AND visible
            final_mask = combined_mask.clone()
            final_mask[combined_mask] = update_filter
            
            if viewspace_point_tensor.grad is not None:
                grad_norm = torch.norm(viewspace_point_tensor.grad[update_filter, :2], dim=-1, keepdim=True)
                self.offset_gradient_accum[final_mask] += grad_norm
                self.offset_denom[final_mask] += 1

    def adjust_anchor(self, check_interval=100, success_threshold=0.8, grad_threshold=0.0002, min_opacity=0.005, optimizers=None):
        # Growing
        grads = self.offset_gradient_accum / (self.offset_denom + 1e-10)
        grads[grads.isnan()] = 0.0
        grads_norm = torch.norm(grads, dim=-1)
        offset_mask = (self.offset_denom > check_interval * success_threshold * 0.5).squeeze(1)
        
        self.anchor_growing(grads_norm, grad_threshold, offset_mask, optimizers)
        
        # Pruning
        prune_mask = (self.opacity_accum < min_opacity * self.anchor_denom).squeeze(1)
        anchors_mask = (self.anchor_denom > check_interval * success_threshold).squeeze(1)
        prune_mask = torch.logical_and(prune_mask, anchors_mask)
        
        if prune_mask.any():
            self.prune_anchor(prune_mask, optimizers)
            
        # Reset accumulation buffers for remaining/new anchors
        num_anchors = self._anchor.shape[0]
        self.opacity_accum = torch.zeros((num_anchors, 1), device=self._anchor.device)
        self.anchor_denom = torch.zeros((num_anchors, 1), device=self._anchor.device)
        self.max_radii2D = torch.zeros(num_anchors, device=self._anchor.device)
        self.offset_denom = torch.zeros((num_anchors * self.n_offsets, 1), device=self._anchor.device)
        self.offset_gradient_accum = torch.zeros((num_anchors * self.n_offsets, 1), device=self._anchor.device)

    def anchor_growing(self, grads, threshold, offset_mask, optimizers=None):
        init_length = self._anchor.shape[0] * self.n_offsets
        for i in range(self.update_depth):
            cur_threshold = threshold * ((self.update_hierachy_factor // 2)**i)
            candidate_mask = (grads >= cur_threshold)
            candidate_mask = torch.logical_and(candidate_mask, offset_mask)
            
            rand_mask = torch.rand_like(candidate_mask.float()) > (0.5**(i + 1))
            candidate_mask = torch.logical_and(candidate_mask, rand_mask)
            
            # Handle growth for subsequent iterations in the loop
            curr_num_gaussians = self._anchor.shape[0] * self.n_offsets
            if curr_num_gaussians > init_length:
                candidate_mask = torch.cat([candidate_mask, torch.zeros(curr_num_gaussians - init_length, dtype=torch.bool, device=self._anchor.device)], dim=0)

            if not candidate_mask.any():
                continue

            # Calculate new anchor positions
            all_xyz = self._anchor.unsqueeze(1) + self._offset * self.scales[:, :3].unsqueeze(1)
            selected_xyz = all_xyz.view(-1, 3)[candidate_mask]
            
            size_factor = self.update_init_factor // (self.update_hierachy_factor**i)
            cur_size = self.voxel_size * size_factor
            
            selected_grid_coords = torch.round(selected_xyz / cur_size).int()
            selected_grid_coords_unique, inverse_indices = torch.unique(selected_grid_coords, return_inverse=True, dim=0)
            
            # Filter existing anchors
            existing_grid_coords = torch.round(self._anchor / cur_size).int()
            
            # Efficient duplication check
            # For simplicity in this implementation, we use a slower but safer check
            # In production, a hash table or voxel grid would be better.
            new_mask = torch.ones(selected_grid_coords_unique.shape[0], dtype=torch.bool, device=self._anchor.device)
            for chunk in torch.split(existing_grid_coords, 10000):
                matches = (selected_grid_coords_unique.unsqueeze(1) == chunk).all(-1).any(-1)
                new_mask = torch.logical_and(new_mask, ~matches)
            
            candidate_anchor = selected_grid_coords_unique[new_mask].float() * cur_size
            
            if candidate_anchor.shape[0] > 0:
                self.add_anchors(candidate_anchor, candidate_mask, inverse_indices, new_mask, cur_size, optimizers)

    def add_anchors(self, new_anchors, candidate_mask, inverse_indices, new_mask, cur_size, optimizers=None):
        num_new = new_anchors.shape[0]
        device = self._anchor.device
        
        new_scaling = torch.log(torch.ones((num_new, 6), device=device) * cur_size)
        new_rotation = torch.zeros((num_new, 4), device=device)
        new_rotation[:, 0] = 1.0
        new_opacities = inverse_sigmoid(0.1 * torch.ones((num_new, 1), device=device))
        
        # Feature inheritance
        inherited_feat = self._anchor_feat.unsqueeze(1).repeat(1, self.n_offsets, 1).view(-1, self.feat_dim)[candidate_mask]
        new_feat = scatter_max(inherited_feat, inverse_indices.unsqueeze(1).expand(-1, inherited_feat.size(1)), dim=0)[0][new_mask]
        
        new_offsets = torch.zeros((num_new, self.n_offsets, 3), device=device)

        # Update parameters and optimizers
        d = {
            "anchor": new_anchors,
            "scaling": new_scaling,
            "rotation": new_rotation,
            "anchor_feat": new_feat,
            "offset": new_offsets,
            "opacity": new_opacities,
        }
        
        self._anchor = nn.Parameter(torch.cat([self._anchor, d["anchor"]], dim=0))
        self._scaling = nn.Parameter(torch.cat([self._scaling, d["scaling"]], dim=0))
        self._rotation = nn.Parameter(torch.cat([self._rotation, d["rotation"]], dim=0))
        self._anchor_feat = nn.Parameter(torch.cat([self._anchor_feat, d["anchor_feat"]], dim=0))
        self._offset = nn.Parameter(torch.cat([self._offset, d["offset"]], dim=0))
        self._opacity = nn.Parameter(torch.cat([self._opacity, d["opacity"]], dim=0))

        # Update buffers
        self.opacity_accum = torch.cat([self.opacity_accum, torch.zeros((num_new, 1), device=device)], dim=0)
        self.anchor_denom = torch.cat([self.anchor_denom, torch.zeros((num_new, 1), device=device)], dim=0)

        if optimizers is not None:
            self.update_optimizers_after_growth(d, optimizers)

    def update_optimizers_after_growth(self, new_data, optimizers):
        # Implementation of cat_tensors_to_optimizer logic
        for name, tensor in new_data.items():
            opt = getattr(optimizers, name, None)
            if opt is None: continue
            
            param = opt.param_groups[0]['params'][0]
            stored_state = opt.state.get(param, None)
            
            if stored_state is not None:
                stored_state["exp_avg"] = torch.cat([stored_state["exp_avg"], torch.zeros_like(tensor)], dim=0)
                stored_state["exp_avg_sq"] = torch.cat([stored_state["exp_avg_sq"], torch.zeros_like(tensor)], dim=0)
                
                del opt.state[param]
                # The parameter itself is already updated in self._anchor etc.
                # We just need to point the optimizer to the new Parameter object
                new_param = getattr(self, f"_{name}" if name == "anchor" or name == "offset" or name == "anchor_feat" else f"_{name}")
                if name == "anchor": new_param = self._anchor
                elif name == "offset": new_param = self._offset
                elif name == "anchor_feat": new_param = self._anchor_feat
                elif name == "scaling": new_param = self._scaling
                elif name == "rotation": new_param = self._rotation
                elif name == "opacity": new_param = self._opacity
                
                opt.param_groups[0]['params'][0] = new_param
                opt.state[new_param] = stored_state
            else:
                new_param = getattr(self, f"_{name}") # Simplified
                opt.param_groups[0]['params'][0] = new_param

    def prune_anchor(self, mask, optimizers=None):
        valid_mask = ~mask
        self._anchor = nn.Parameter(self._anchor[valid_mask])
        self._offset = nn.Parameter(self._offset[valid_mask])
        self._anchor_feat = nn.Parameter(self._anchor_feat[valid_mask])
        self._scaling = nn.Parameter(self._scaling[valid_mask])
        self._rotation = nn.Parameter(self._rotation[valid_mask])
        self._opacity = nn.Parameter(self._opacity[valid_mask])
        
        self.opacity_accum = self.opacity_accum[valid_mask]
        self.anchor_denom = self.anchor_denom[valid_mask]

        if optimizers is not None:
            self.update_optimizers_after_pruning(valid_mask, optimizers)

    def update_optimizers_after_pruning(self, valid_mask, optimizers):
        for name in ["anchor", "offset", "anchor_feat", "scaling", "rotation", "opacity"]:
            opt = getattr(optimizers, name, None)
            if opt is None: continue
            
            param = opt.param_groups[0]['params'][0]
            stored_state = opt.state.get(param, None)
            
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][valid_mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][valid_mask]
                
                del opt.state[param]
                new_param = getattr(self, f"_{name}" if name in ["anchor", "offset", "anchor_feat"] else f"_{name}")
                if name == "anchor": new_param = self._anchor
                opt.param_groups[0]['params'][0] = new_param
                opt.state[new_param] = stored_state

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
        # MLPs learning rate
        lr_mlp = 0.001 

        extra_optimizers = {
            "mlp_opacity": torch.optim.Adam(self.mlp_opacity.parameters(), lr=lr_mlp),
            "mlp_cov": torch.optim.Adam(self.mlp_cov.parameters(), lr=lr_mlp),
            "mlp_color": torch.optim.Adam(self.mlp_color.parameters(), lr=lr_mlp),
        }

        if self.use_feat_bank:
            extra_optimizers["mlp_feature_bank"] = torch.optim.Adam(self.mlp_feature_bank.parameters(), lr=lr_mlp)
        
        if self.embedding_appearance is not None:
            extra_optimizers["embedding_appearance"] = torch.optim.Adam(self.embedding_appearance.parameters(), lr=lr_mlp)

        return GSOptimizers(
            means=torch.optim.Adam([self._anchor], lr=lr_means * means_lr_multiplier),
            scales=torch.optim.Adam([self._scaling], lr=lr_scales),
            quats=torch.optim.Adam([self._rotation], lr=lr_quats),
            opacities=torch.optim.Adam([self._opacity], lr=lr_opacities),
            features_dc=torch.optim.Adam([self._anchor_feat], lr=lr_sh),
            features_rest=torch.optim.Adam([self._offset], lr=lr_sh * 0.1), # Using rest for offset
            features_semantics=None,
            extra=extra_optimizers
        )

    def save_ply(self, path):
        # Custom save logic for Scaffold-GS
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        
        anchor = self._anchor.detach().cpu().numpy()
        num_pts = anchor.shape[0]
        normals = np.zeros_like(anchor)
        
        # We'll save a subset of parameters to be compatible with standard loaders if possible,
        # but realistically this needs a custom loader.
        # For now, let's save the anchor data.
        
        dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
                 ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4')]
        
        elements = np.empty(num_pts, dtype=dtype)
        attributes = np.concatenate((anchor, normals), axis=1)
        elements[:] = list(map(tuple, attributes))
        
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)
        logger.info(f"Saved {num_pts} anchors to {path}")

    def save_checkpoints(self, path):
        # Save MLPs and embeddings
        torch.save({
            'opacity_mlp': self.mlp_opacity.state_dict(),
            'cov_mlp': self.mlp_cov.state_dict(),
            'color_mlp': self.mlp_color.state_dict(),
            'feature_bank_mlp': self.mlp_feature_bank.state_dict() if self.use_feat_bank else None,
            'appearance': self.embedding_appearance.state_dict() if self.embedding_appearance is not None else None,
            'model_state_dict': self.state_dict()
        }, path)

    def load_checkpoint(self, path):
        checkpoint = torch.load(path)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.mlp_opacity.load_state_dict(checkpoint['opacity_mlp'])
        self.mlp_cov.load_state_dict(checkpoint['cov_mlp'])
        self.mlp_color.load_state_dict(checkpoint['color_mlp'])
        if self.use_feat_bank and checkpoint['feature_bank_mlp'] is not None:
            self.mlp_feature_bank.load_state_dict(checkpoint['feature_bank_mlp'])
        if self.embedding_appearance is not None and checkpoint['appearance'] is not None:
            self.embedding_appearance.load_state_dict(checkpoint['appearance'])
    
    def compute_lods(self, num_levels: int = 1, factor: int = 4, optimizers=None):
        """
        Scaffold-GS doesn't support LoD in the traditional sense, but this method
        is provided for API compatibility with GaussianModel.
        """
        if num_levels > 1:
            logger.warning(
                f"[yellow]⚠️  Scaffold-GS does not support LoD levels. "
                f"Requested {num_levels} levels will be ignored. "
                f"All anchors remain at single level.[/yellow]"
            )
        # Keep single-level representation
        self.lod_offsets = [len(self._anchor)]

