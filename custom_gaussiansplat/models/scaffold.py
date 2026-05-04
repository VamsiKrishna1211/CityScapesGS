import logging
import os
from functools import reduce
from typing import Dict, Iterator, Optional, Tuple

import numpy as np
import tinycudann as tcnn
import torch
import torch.nn as nn
import torch.nn.functional as F
from gs_types import GS_LR_Schedulers, GSOptimizers, NeuralGaussianOutput, RenderParams
from plyfile import PlyData, PlyElement
from simple_knn._C import distCUDA2
from torch.utils.checkpoint import checkpoint as gradient_checkpoint
from torch_scatter import scatter_max

from .base import BaseTrainableModel, NeuralRenderingMixin

logger = logging.getLogger("cityscape_gs.models.scaffold")

torch.set_float32_matmul_precision('high')
torch.backends.cudnn.enabled = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True

# Allows dynamism on parms
torch._dynamo.config.force_parameter_static_shapes = False


# ---------------------------------------------------------------------------
# Optimization helpers
# ---------------------------------------------------------------------------

def find_duplicates_voxel_grid(
    coords: torch.Tensor, resolution: float = 1.0
) -> torch.Tensor:
    """Find duplicate coordinates using a spatial hash grid.

    Returns a boolean mask where True = keep (first occurrence), False = duplicate.
    Complexity: O(n log n) vs O(n²) for naive nested loop comparison.

    Args:
        coords: [N, 3] tensor of 3D coordinates
        resolution: voxel size for grouping

    Returns:
        Boolean mask [N] where True = keep, False = duplicate
    """
    # Quantize coordinates to grid cells
    cell_keys = torch.floor(coords / resolution).long()

    # Hash each cell to a single integer (linearize the 3D grid)
    # Use large primes for hashing to minimize collisions
    hash_keys = (
        cell_keys[:, 0].long() * 73856093
        ^ cell_keys[:, 1].long() * 19349663
        ^ cell_keys[:, 2].long() * 83492791
    ) % (1 << 30)

    # Sort by hash to group same cells
    sort_idx = torch.argsort(hash_keys)
    sorted_keys = hash_keys[sort_idx]

    # Find first occurrence of each unique value
    is_first = torch.ones(len(coords), dtype=torch.bool, device=coords.device)
    is_first[1:] = sorted_keys[1:] != sorted_keys[:-1]

    # Unsort to restore original order
    keep_mask = torch.zeros(len(coords), dtype=torch.bool, device=coords.device)
    keep_mask[sort_idx] = is_first

    return keep_mask


def normalize_safe(x: torch.Tensor, dim: int = -1, eps: float = 1e-10) -> torch.Tensor:
    """Normalize along dim with numerical safety."""
    norm = x.norm(dim=dim, keepdim=True)
    return x / (norm + eps)

def inverse_sigmoid(x: torch.Tensor) -> torch.Tensor:
    return torch.log(x / (1 - x))


class Embedding(nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.embedding = torch.nn.Embedding(in_dim, out_dim)

    def forward(self, in_tensor: torch.Tensor) -> torch.Tensor:
        return self.embedding(in_tensor)


class GaussianFourierFeatureMapping(nn.Module):
    """Projects view directions into a high-dim Fourier feature space.

    Overcomes MLP spectral bias for high-frequency view-dependent effects.
    The frequency matrix B is a non-trainable buffer — it moves to the right
    device with the model but is never updated by the optimizer.
    """

    def __init__(self, input_dim: int = 3, num_frequencies: int = 64, scale: float = 5.0) -> None:
        super().__init__()
        B = torch.randn(num_frequencies, input_dim) * scale
        self.register_buffer("B", B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_proj = (2.0 * np.pi * x) @ self.B.T  # type: ignore[operator]
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1).contiguous()


class GeometricMultiHeadMLP(nn.Module):
    """Single-call multi-head module for opacity, covariance, and color outputs.

    Eliminates runtime branching by pre-building input selector functions at init time.
    This removes Python-side conditional overhead from the forward path entirely.

    OPTIMIZATION: Supports gradient checkpointing for memory efficiency.
    """

    def __init__(
        self,
        feat_dim: int,
        fourier_embed_dim: int,
        n_offsets: int,
        appearance_dim: int,
        add_opacity_dist: bool,
        add_cov_dist: bool,
        add_color_dist: bool,
        use_gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__()

        # OPTIMIZATION: Enable gradient checkpointing for memory savings
        self.use_gradient_checkpointing = use_gradient_checkpointing

        # Pre-build selector functions—choice is made once at init, not per-forward
        self._opacity_select = lambda with_dist, wodist: with_dist if add_opacity_dist else wodist
        self._cov_select = lambda with_dist, wodist: with_dist if add_cov_dist else wodist
        self._color_select = lambda with_dist, wodist: with_dist if add_color_dist else wodist

        # Track whether color head expects appearance to avoid forward-time checks
        self._append_appearance = appearance_dim > 0

        # opacity_in_dim = feat_dim + fourier_embed_dim + 3 + (1 if add_opacity_dist else 0)
        # cov_in_dim = feat_dim + fourier_embed_dim + 3 + (1 if add_cov_dist else 0)
        # color_in_dim = feat_dim + fourier_embed_dim + 3 + (1 if add_color_dist else 0) + appearance_dim

        opacity_in_dim = feat_dim + fourier_embed_dim
        cov_in_dim = feat_dim + fourier_embed_dim
        color_in_dim = feat_dim + fourier_embed_dim

        # opacity_in_dim = fourier_embed_dim + 3 + (1 if add_opacity_dist else 0)
        # cov_in_dim = fourier_embed_dim + 3 + (1 if add_cov_dist else 0)
        # color_in_dim = fourier_embed_dim + 3 + (1 if add_color_dist else 0) + appearance_dim

        # opacity_in_dim = feat_dim + 3 + (1 if add_opacity_dist else 0)
        # cov_in_dim = feat_dim + 3 + (1 if add_cov_dist else 0)
        # color_in_dim = feat_dim + 3 + (1 if add_color_dist else 0) + appearance_dim


        self.opacity_head = nn.Sequential(
            nn.Linear(opacity_in_dim, feat_dim),
            nn.ReLU(True),
            nn.Linear(feat_dim, n_offsets),
            nn.Tanh(),
        )

        self.cov_head = nn.Sequential(
            nn.Linear(cov_in_dim, feat_dim),
            nn.ReLU(True),
            nn.Linear(feat_dim, 7 * n_offsets),
        )

        self.color_head = nn.Sequential(
            nn.Linear(color_in_dim, feat_dim),
            nn.ReLU(True),
            nn.Linear(feat_dim, 3 * n_offsets),
            nn.Sigmoid(),
        )

    def forward(
        self,
        cat_local_view: torch.Tensor,
        cat_local_view_wodist: torch.Tensor,
        appearance: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # All branching is eliminated—selectors are pre-computed closures
        opacity_input = self._opacity_select(cat_local_view, cat_local_view_wodist)
        color_input = self._color_select(cat_local_view, cat_local_view_wodist)
        cov_input = self._cov_select(cat_local_view, cat_local_view_wodist)

        if self._append_appearance and appearance is not None:
            color_input = torch.cat([color_input, appearance], dim=1)

        # OPTIMIZATION: Use gradient checkpointing for memory savings
        if self.use_gradient_checkpointing and self.training:
            neural_opacity = gradient_checkpoint(self.opacity_head, opacity_input).contiguous().view(-1, 1)
            color = gradient_checkpoint(self.color_head, color_input).contiguous().view(-1, 3)
            scale_rot = gradient_checkpoint(self.cov_head, cov_input).contiguous().view(-1, 7)
        else:
            neural_opacity = self.opacity_head(opacity_input).contiguous().view(-1, 1)
            color = self.color_head(color_input).contiguous().view(-1, 3)
            scale_rot = self.cov_head(cov_input).contiguous().view(-1, 7)

        return neural_opacity, color, scale_rot

class GaussianFourierFeatureMappingTcnn(nn.Module):
    """Projects view directions using tcnn's optimized Frequency encoding.
    
    tcnn has built-in Fourier encoding that's CUDA-fused and faster than
    manual projection + sin/cos.
    Note: tcnn Frequency encoding doesn't support scale parameter directly,
    so we apply scaling manually in forward().
    """

    def __init__(self, input_dim: int = 3, num_frequencies: int = 64, scale: float = 5.0) -> None:
        super().__init__()
        self.scale = scale
        self.encoding = tcnn.Encoding(
            n_input_dims=input_dim,
            encoding_config={
                "otype": "Frequency",
                "n_frequencies": num_frequencies,
            },
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply scale manually (tcnn doesn't support scale parameter natively)
        # Frequency encoding outputs 2*n_frequencies dims (sin+cos)
        return self.encoding(x * self.scale)


class GeometricMultiHeadMLPTcnn(nn.Module):
    """Multi-head model using tcnn.Network for CUDA-optimized inference.
    
    Key benefits:
      - FullyFusedMLP: all layers fused into single CUDA kernel
      - ~5-10x faster than nn.Sequential on typical configs
      - Lower memory footprint due to kernel fusion
      - Automatic half-precision support
    
    Constraint: FullyFusedMLP requires n_neurons ∈ {16, 32, 64, 128}.
    If feat_dim doesn't match, automatically selects closest valid value
    or falls back to CutlassMLP (flexible but slower).
    """

    def __init__(
        self,
        feat_dim: int,
        fourier_embed_dim: int,
        n_offsets: int,
        appearance_dim: int,
        add_opacity_dist: bool,
        add_cov_dist: bool,
        add_color_dist: bool,
        n_neurons: Optional[int] = None,
    ) -> None:
        super().__init__()

        # Selector functions: same as before
        self._opacity_select = lambda with_dist, wodist: with_dist if add_opacity_dist else wodist
        self._cov_select = lambda with_dist, wodist: with_dist if add_cov_dist else wodist
        self._color_select = lambda with_dist, wodist: with_dist if add_color_dist else wodist

        self._append_appearance = appearance_dim > 0

        opacity_in_dim = feat_dim + fourier_embed_dim + (1 if add_opacity_dist else 0)
        cov_in_dim = feat_dim + fourier_embed_dim + (1 if add_cov_dist else 0)
        color_in_dim = feat_dim + fourier_embed_dim + (1 if add_color_dist else 0) + appearance_dim

        # Determine n_neurons: FullyFusedMLP only accepts {16, 32, 64, 128}
        if n_neurons is None:
            # Auto-select closest valid value
            valid_neurons = [16, 32, 64, 128]
            n_neurons = min(valid_neurons, key=lambda x: abs(x - feat_dim))
        
        use_fully_fused = n_neurons in [16, 32, 64, 128]
        mlp_type = "FullyFusedMLP" if use_fully_fused else "CutlassMLP"
        n_hidden_layers = 1  # 2-layer network: input→hidden→output

        # tcnn.Network config: correct parameter names per documentation
        network_cfg_base = {
            "otype": mlp_type,
            "activation": "ReLU",
            "n_neurons": n_neurons,
            "n_hidden_layers": n_hidden_layers,
        }

        self.opacity_head = tcnn.Network(
            n_input_dims=opacity_in_dim,
            n_output_dims=n_offsets,
            network_config={**network_cfg_base, "output_activation": "Tanh"},
        )

        self.cov_head = tcnn.Network(
            n_input_dims=cov_in_dim,
            n_output_dims=7 * n_offsets,
            network_config={**network_cfg_base, "output_activation": "None"},
        )

        self.color_head = tcnn.Network(
            n_input_dims=color_in_dim,
            n_output_dims=3 * n_offsets,
            network_config={**network_cfg_base, "output_activation": "Sigmoid"},
        )

    def forward(
        self,
        cat_local_view: torch.Tensor,
        cat_local_view_wodist: torch.Tensor,
        appearance: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # tcnn.Network expects inputs in [B, input_dim]; outputs [B, output_dim]
        # tcnn kernels are pre-optimized; no need for .contiguous() or torch.compile
        # Use .reshape() instead of .view() since tcnn output may not be contiguous
        
        opacity_input = self._opacity_select(cat_local_view, cat_local_view_wodist)
        neural_opacity = self.opacity_head(opacity_input).reshape(-1, 1)

        color_input = self._color_select(cat_local_view, cat_local_view_wodist)
        if self._append_appearance and appearance is not None:
            color_input = torch.cat([color_input, appearance], dim=1)
        color = self.color_head(color_input).reshape(-1, 3)

        cov_input = self._cov_select(cat_local_view, cat_local_view_wodist)
        scale_rot = self.cov_head(cov_input).reshape(-1, 7)
        
        return neural_opacity, color, scale_rot


class ScaffoldModel(BaseTrainableModel, NeuralRenderingMixin):
    """Scaffold-GS: sparse anchor points + MLPs generate neural Gaussians per view.

    Language/semantic features are handled by SemanticScaffoldModel (a subclass).
    This class is responsible only for visual reconstruction.
    """

    def __init__(
        self,
        init_points: torch.Tensor | None = None,
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
        fourier_freqs: int = 256,
        fourier_scale: float = 6.05,
        lr_offset: float = 0.01,
        lr_mlp_opacity: float = 0.002,
        lr_mlp_cov: float = 0.004,
        lr_mlp_color: float = 0.008,
        lr_appearance: float = 0.05,
        use_gradient_checkpointing: bool = False,
        console=None,
    ):
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
        self._sh_degree = sh_degree
        self.lr_offset = lr_offset
        self.lr_mlp_opacity = lr_mlp_opacity
        self.lr_mlp_cov = lr_mlp_cov
        self.lr_mlp_color = lr_mlp_color
        self.lr_appearance = lr_appearance
        self.use_gradient_checkpointing = use_gradient_checkpointing

        self._extra_optimizers: Dict[str, torch.optim.Optimizer] = {}
        self._extra_schedulers: Dict[str, torch.optim.lr_scheduler.LRScheduler] = {}

        # Pre-allocated buffer capacity tracking for densification
        # These track the maximum allocated size to avoid repeated reallocation
        self._buffer_capacity: int = 0

        # --- Learnable anchor parameters ---
        self._anchor: nn.Parameter = nn.Parameter(torch.empty(0))       # anchor positions
        self._offset: nn.Parameter = nn.Parameter(torch.empty(0))       # per-anchor offsets
        self._anchor_feat: nn.Parameter = nn.Parameter(torch.empty(0))  # per-anchor features
        self._scaling: nn.Parameter = nn.Parameter(torch.empty(0))      # anchor scales
        self._opacity: nn.Parameter = nn.Parameter(torch.empty(0))      # anchor opacities

        # --- Densification buffers ---
        self.register_buffer("opacity_accum", torch.empty(0))
        self.register_buffer("offset_gradient_accum", torch.empty(0))
        self.register_buffer("offset_denom", torch.empty(0))
        self.register_buffer("anchor_denom", torch.empty(0))
        self.register_buffer("max_radii2D", torch.empty(0))

        # Single-level LoD placeholder (Scaffold-GS doesn't use LoD).
        self.lod_offsets = [0]

        # --- Fourier view embedder (shared across geometric MLPs) ---
        self.fourier_embedder: nn.Module = GaussianFourierFeatureMapping(
            input_dim=3, num_frequencies=fourier_freqs, scale=fourier_scale
        )
        self.fourier_embedder_dist = GaussianFourierFeatureMapping(
            input_dim=4, num_frequencies=fourier_freqs, scale=fourier_scale
        )
        # Don't compile tcnn encodings—they're already CUDA-fused
        # tcnn Frequency encoding outputs: n_input_dims * 2 * n_frequencies
        # With 3D input and fourier_freqs: 3 * 2 * fourier_freqs
        fourier_embed_dim = 2 * fourier_freqs
        self.fourier_embed_dim = fourier_embed_dim

        # --- Geometric multi-head MLP (opacity + cov + color in one call) ---
        self.mlp_geo_heads_raw = GeometricMultiHeadMLP(
            feat_dim=feat_dim,
            fourier_embed_dim=fourier_embed_dim,
            n_offsets=n_offsets,
            appearance_dim=appearance_dim,
            add_opacity_dist=add_opacity_dist,
            add_cov_dist=add_cov_dist,
            add_color_dist=add_color_dist,
            use_gradient_checkpointing=use_gradient_checkpointing,
        )
        # tcnn.Network is already kernel-fused; no need for torch.compile
        self.mlp_geo_heads = self.mlp_geo_heads_raw

        self.mlp_feature_bank: Optional[nn.Module] = None
        if use_feat_bank:
            self.mlp_feature_bank = torch.compile(  # type: ignore[assignment]
                nn.Sequential(
                    nn.Linear(fourier_embed_dim + 1, feat_dim),
                    nn.ReLU(True),
                    nn.Linear(feat_dim, 3),
                    nn.Softmax(dim=1),
                )
            )

        self.embedding_appearance: Optional[Embedding] = None

        if init_points is not None and len(init_points) > 0:
            self.create_from_pcd(init_points)

    # ── Appearance embedding (set after dataset is loaded) ───────────────────

    def set_appearance(self, num_cameras: int) -> None:
        if self.appearance_dim > 0:
            self.embedding_appearance = Embedding(num_cameras, self.appearance_dim).to(
                self._anchor.device
            )

    # ── Geometric properties (required by BaseTrainableModel) ────────────────

    @property
    def anchors(self) -> torch.Tensor:
        return self._anchor

    @property
    def means(self) -> torch.Tensor:
        return self._anchor

    @property
    def anchor_feats(self) -> torch.Tensor:
        return self._anchor_feat

    @property
    def scales(self) -> torch.Tensor:
        return torch.exp(self._scaling)

    @property
    def quats(self) -> torch.Tensor | None:
        # return F.normalize(self._rotation, dim=-1)
        return None  # Scaffold-GS does not use rotation; return None or identity as appropriate in downstream code.

    @property
    def opacities(self) -> torch.Tensor:
        return torch.sigmoid(self._opacity)

    @property
    def sh_degree(self) -> int:
        return self._sh_degree

    @property
    def point_name(self) -> str:
        return "anchors"

    @property
    def count_label(self) -> str:
        return "Anchors"

    @property
    def features_dc(self) -> torch.Tensor:
        return self._anchor_feat

    @property
    def features_rest(self) -> torch.Tensor:
        return self._offset

    @property
    def dc_rgb(self) -> torch.Tensor:
        return torch.ones((self._anchor.shape[0], 1, 3), device=self._anchor.device) * 0.5

    @property
    def sh(self) -> torch.Tensor:
        return torch.ones((self._anchor.shape[0], 1, 3), device=self._anchor.device) * 0.5

    # ── Parameter dict contract (required by BaseTrainableModel) ─────────────

    def get_params_dict(self) -> Dict[str, nn.Parameter]:
        return {
            "means": self._anchor,
            "scales": self._scaling,
            "opacities": self._opacity,
            "features_dc": self._anchor_feat,
            "features_rest": self._offset,
        }

    def get_optimizers_dict(
        self, optimizers: GSOptimizers
    ) -> Dict[str, torch.optim.Optimizer]:
        d: Dict[str, torch.optim.Optimizer] = {
            "means": optimizers.means,
            "scales": optimizers.scales,
            "opacities": optimizers.opacities,
            "features_dc": optimizers.features_dc,
            "features_rest": optimizers.features_rest,
        }
        return d

    def update_params_from_dict(self, params: Dict[str, nn.Parameter]) -> None:
        self._anchor = params["means"]
        self._scaling = params["scales"]
        self._opacity = params["opacities"]
        self._anchor_feat = params["features_dc"]
        self._offset = params["features_rest"]

    # ── Point cloud initialization ────────────────────────────────────────────

    def create_from_pcd(self, points: torch.Tensor) -> None:
        if self.voxel_size <= 0:
            init_dist = distCUDA2(points.contiguous()).float()
            median_dist, _ = torch.kthvalue(init_dist, int(init_dist.shape[0] * 0.5))
            self.voxel_size = torch.sqrt(median_dist).item()
            logger.info(f"Auto-calculated voxel_size: {self.voxel_size}")

        points_np = points.detach().cpu().numpy()
        np.random.shuffle(points_np)
        points_np = np.unique(np.round(points_np / self.voxel_size), axis=0) * self.voxel_size
        fused_point_cloud = torch.tensor(points_np, dtype=torch.float32, device=points.device)
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
        torch.nn.init.xavier_uniform_(self._anchor_feat)
        self._scaling = nn.Parameter(scales)
        self._opacity = nn.Parameter(opacities)

        self.max_radii2D = torch.zeros(num_points, device=points.device)
        self.opacity_accum = torch.zeros((num_points, 1), device=points.device)
        self.anchor_denom = torch.zeros((num_points, 1), device=points.device)
        self.offset_gradient_accum = torch.zeros(
            (num_points * self.n_offsets, 1), device=points.device
        )
        self.offset_denom = torch.zeros(
            (num_points * self.n_offsets, 1), device=points.device
        )
        self.lod_offsets = [num_points]

        # OPTIMIZATION: Invalidate params cache after initialization
        self._invalidate_params_cache()

    # ── Core neural Gaussian generation ──────────────────────────────────────

    def generate_neural_gaussians(
        self,
        cam: dict,
        visible_mask: Optional[torch.Tensor] = None,
        is_training: bool = True,
    ) -> NeuralGaussianOutput:
        if visible_mask is None:
            visible_mask = torch.ones(
                self._anchor.shape[0], dtype=torch.bool, device=self._anchor.device
            )

        feat = self._anchor_feat[visible_mask]
        anchor = self._anchor[visible_mask]
        grid_offsets = self._offset[visible_mask]
        grid_scaling = self.scales[visible_mask]

        # --- View direction and distance ---
        camera_center = cam["camera_center"]
        if not isinstance(camera_center, torch.Tensor):
            camera_center = torch.tensor(camera_center, device=anchor.device, dtype=anchor.dtype)
        elif camera_center.device != anchor.device:
            camera_center = camera_center.to(device=anchor.device, dtype=anchor.dtype)

        ob_view = anchor - camera_center
        ob_dist = ob_view.norm(dim=1, keepdim=True)
        ob_view = normalize_safe(ob_view, dim=1)

        # embedded_feat = self.fourier_embedder(feat)
        embedded_view = self.fourier_embedder(ob_view)
        embedded_view_dist = self.fourier_embedder_dist(torch.cat([ob_view, ob_dist], dim=1))

        # --- Optional feature bank (multi-resolution anchor feat) ---
        if self.mlp_feature_bank is not None:
            cat_view = torch.cat([embedded_view, ob_dist], dim=1)
            bank_weight = self.mlp_feature_bank(cat_view)  # [N_vis, 3]
            f1 = feat[:, ::4].repeat(1, 4)
            f2 = feat[:, ::2].repeat(1, 2)
            f3 = feat
            feat = (
                f1 * bank_weight[:, 0:1]
                + f2 * bank_weight[:, 1:2]
                + f3 * bank_weight[:, 2:3]
            )

        cat_local_view = torch.cat([feat, embedded_view_dist], dim=1).contiguous()
        cat_local_view_wodist = torch.cat([feat, embedded_view], dim=1).contiguous()

        # --- Appearance embedding ---
        appearance: Optional[torch.Tensor] = None
        if self.embedding_appearance is not None and self.appearance_dim > 0:
            camera_indices = (
                torch.ones_like(cat_local_view[:, 0], dtype=torch.long)
                * cam["uid"]
            )
            appearance = self.embedding_appearance(camera_indices)

        # --- Fused geometric heads (opacity + color + covariance) ---
        neural_opacity, color, scale_rot = self.mlp_geo_heads(
            cat_local_view, cat_local_view_wodist, appearance
        )
        # Flatten and create boolean mask for filtering positive opacities
        neural_opacity_flat = neural_opacity.view(-1)
        mask = neural_opacity_flat > 0.0
        mask = mask.to(dtype=torch.bool, device=neural_opacity_flat.device)
        opacity = neural_opacity_flat[mask].view(-1, 1)

        offsets = grid_offsets.contiguous().view(-1, 3)

        # Gather anchor-level properties for each selected offset
        indices = (
            torch.arange(anchor.shape[0], device=anchor.device)
            .unsqueeze(1)
            .expand(-1, self.n_offsets)
            .reshape(-1)
        )
        valid_indices = indices[mask]

        s_repeat = grid_scaling[valid_indices]
        a_repeat = anchor[valid_indices]
        masked_color = color[mask]
        masked_scale_rot = scale_rot[mask]
        masked_offsets = offsets[mask]

        scaling = s_repeat[:, 3:] * torch.sigmoid(masked_scale_rot[:, :3])
        rot = F.normalize(masked_scale_rot[:, 3:7], dim=-1)
        xyz = masked_offsets * s_repeat[:, :3] + a_repeat

        if is_training:
            return NeuralGaussianOutput(
                means=xyz,
                colors=masked_color,
                opacities=opacity,
                scales=scaling,
                quats=rot,
                neural_opacity=neural_opacity,
                selection_mask=mask,
            )
        else:
            return NeuralGaussianOutput(
                means=xyz,
                colors=masked_color,
                opacities=opacity,
                scales=scaling,
                quats=rot,
            )

    # ── Densification statistics ──────────────────────────────────────────────

    def update_training_stats(
        self,
        viewspace_point_tensor: torch.Tensor,
        opacity: torch.Tensor,
        update_filter: torch.Tensor,
        offset_selection_mask: torch.Tensor,
        anchor_visible_mask: torch.Tensor,
    ) -> None:
        with torch.no_grad():
            temp_opacity = opacity.clone().view(-1).detach()
            temp_opacity[temp_opacity < 0] = 0
            temp_opacity = temp_opacity.view(-1, self.n_offsets)

            self.opacity_accum[anchor_visible_mask] += temp_opacity.sum(dim=1, keepdim=True)
            self.anchor_denom[anchor_visible_mask] += 1

            expanded_anchor_mask = (
                anchor_visible_mask.unsqueeze(1).repeat(1, self.n_offsets).view(-1)
            )
            combined_mask = torch.zeros(
                self.offset_gradient_accum.shape[0],
                dtype=torch.bool,
                device=self.offset_gradient_accum.device,
            )
            combined_mask[expanded_anchor_mask] = offset_selection_mask

            final_mask = combined_mask.clone()
            final_mask[combined_mask] = update_filter

            if viewspace_point_tensor.grad is not None:
                grad = viewspace_point_tensor.grad
                if grad.dim() == 3:
                    grad = grad.squeeze(0)
                grad_norm = torch.norm(grad[update_filter, :2], dim=-1, keepdim=True)
                self.offset_gradient_accum[final_mask] += grad_norm
                self.offset_denom[final_mask] += 1

    def adjust_anchor(
        self,
        check_interval: int = 100,
        success_threshold: float = 0.8,
        grad_threshold: float = 0.0002,
        min_opacity: float = 0.005,
        optimizers: Optional[GSOptimizers] = None,
    ) -> None:
        grads = self.offset_gradient_accum / (self.offset_denom + 1e-10)
        grads[grads.isnan()] = 0.0
        grads_norm = torch.norm(grads, dim=-1)
        offset_mask = (self.offset_denom > check_interval * success_threshold * 0.5).squeeze(1)

        self.anchor_growing(grads_norm, grad_threshold, offset_mask, optimizers)

        # Reset and grow offset_denom to exact size (matches original Scaffold-GS)
        self.offset_denom[offset_mask] = 0
        padding_offset_denom = torch.zeros(
            [self._anchor.shape[0] * self.n_offsets - self.offset_denom.shape[0], 1],
            dtype=torch.int32,
            device=self.offset_denom.device,
        )
        self.offset_denom = torch.cat([self.offset_denom, padding_offset_denom], dim=0)

        # Reset and grow offset_gradient_accum to exact size (matches original Scaffold-GS)
        self.offset_gradient_accum[offset_mask] = 0
        padding_offset_gradient_accum = torch.zeros(
            [self._anchor.shape[0] * self.n_offsets - self.offset_gradient_accum.shape[0], 1],
            dtype=torch.int32,
            device=self.offset_gradient_accum.device,
        )
        self.offset_gradient_accum = torch.cat(
            [self.offset_gradient_accum, padding_offset_gradient_accum], dim=0
        )

        prune_mask = (self.opacity_accum < min_opacity * self.anchor_denom).squeeze(1)
        anchors_mask = (self.anchor_denom > check_interval * success_threshold).squeeze(1)
        prune_mask = torch.logical_and(prune_mask, anchors_mask)

        # Properly resize offset_denom after pruning (matches original Scaffold-GS)
        offset_denom = self.offset_denom.view(-1, self.n_offsets)[~prune_mask]
        offset_denom = offset_denom.view(-1, 1)
        del self.offset_denom
        self.offset_denom = offset_denom

        # Properly resize offset_gradient_accum after pruning (matches original Scaffold-GS)
        offset_gradient_accum = self.offset_gradient_accum.view(-1, self.n_offsets)[~prune_mask]
        offset_gradient_accum = offset_gradient_accum.view(-1, 1)
        del self.offset_gradient_accum
        self.offset_gradient_accum = offset_gradient_accum

        n_visited = int(anchors_mask.sum())
        if n_visited > 0:
            self.opacity_accum[anchors_mask] = torch.zeros(
                (n_visited, 1), device=self._anchor.device
            ).float()
            self.anchor_denom[anchors_mask] = torch.zeros(
                (n_visited, 1), device=self._anchor.device
            ).float()

        self.opacity_accum = self.opacity_accum[~prune_mask]
        self.anchor_denom = self.anchor_denom[~prune_mask]

        if prune_mask.any():
            self.prune_anchor(prune_mask, optimizers)

        # Memory management (matches original Scaffold-GS)
        torch.cuda.empty_cache()

        self.max_radii2D = torch.zeros(self._anchor.shape[0], device=self._anchor.device)

        # OPTIMIZATION: Invalidate params cache once after all structural changes
        self._invalidate_params_cache()

    def anchor_growing(
        self,
        grads: torch.Tensor,
        threshold: float,
        offset_mask: torch.Tensor,
        optimizers: Optional[GSOptimizers] = None,
    ) -> None:
        init_length = self._anchor.shape[0] * self.n_offsets
        for i in range(self.update_depth):
            cur_threshold = threshold * ((self.update_hierachy_factor // 2) ** i)
            candidate_mask = (grads >= cur_threshold)
            candidate_mask = torch.logical_and(candidate_mask, offset_mask)

            rand_mask = torch.rand_like(candidate_mask.float()) > (0.5 ** (i + 1))
            rand_mask = rand_mask.to(self._anchor.device)
            candidate_mask = torch.logical_and(candidate_mask, rand_mask)

            length_inc = self._anchor.shape[0] * self.n_offsets - init_length
            if length_inc == 0:
                if i > 0:
                    continue
            else:
                candidate_mask = torch.cat(
                    [
                        candidate_mask,
                        torch.zeros(length_inc, dtype=torch.bool, device=self._anchor.device),
                    ],
                    dim=0,
                )

            if not candidate_mask.any():
                continue

            all_xyz = self._anchor.unsqueeze(1) + self._offset * self.scales[:, :3].unsqueeze(1)
            selected_xyz = all_xyz.view(-1, 3)[candidate_mask]

            size_factor = self.update_init_factor // (self.update_hierachy_factor ** i)
            cur_size = self.voxel_size * size_factor

            selected_grid_coords = torch.round(selected_xyz / cur_size).int()
            selected_grid_coords_unique, inverse_indices = torch.unique(
                selected_grid_coords, return_inverse=True, dim=0
            )

            grid_coords = torch.round(self._anchor / cur_size).int()
            # OPTIMIZATION: Efficient O(n log n) deduplication using voxel grid hashing
            # instead of O(n²) nested loop comparisons
            all_grid_coords = torch.cat([grid_coords, selected_grid_coords_unique], dim=0)
            keep_mask = find_duplicates_voxel_grid(all_grid_coords, resolution=1.0)
            n_existing = grid_coords.shape[0]
            remove_duplicates = keep_mask[n_existing:]  # True = keep, False = duplicate

            candidate_anchor = selected_grid_coords_unique[remove_duplicates] * cur_size

            if candidate_anchor.shape[0] > 0:
                self.add_anchors(
                    candidate_anchor, candidate_mask, inverse_indices, remove_duplicates,
                    cur_size, optimizers,
                )

    def add_anchors(
        self,
        new_anchors: torch.Tensor,
        candidate_mask: torch.Tensor,
        inverse_indices: torch.Tensor,
        new_mask: torch.Tensor,
        cur_size: float,
        optimizers: Optional[GSOptimizers] = None,
    ) -> None:
        num_new = new_anchors.shape[0]
        device = self._anchor.device

        new_scaling = torch.log(torch.ones((num_new, 6), device=device) * cur_size)
        new_rotation = torch.zeros((num_new, 4), device=device)
        new_rotation[:, 0] = 1.0
        new_opacities = inverse_sigmoid(0.1 * torch.ones((num_new, 1), device=device))

        # Geometric feature inheritance via scatter_max
        inherited_feat = (
            self._anchor_feat.unsqueeze(1)
            .repeat(1, self.n_offsets, 1)
            .view(-1, self.feat_dim)[candidate_mask]
        )
        new_feat = scatter_max(
            inherited_feat,
            inverse_indices.unsqueeze(1).expand(-1, inherited_feat.size(1)),
            dim=0,
        )[0][new_mask]

        new_offsets = torch.zeros((num_new, self.n_offsets, 3), device=device)

        # Keys match GSOptimizers field names for correct optimizer state updates.
        d: Dict[str, torch.Tensor] = {
            "means": new_anchors,
            "scales": new_scaling,
            "quats": new_rotation,
            "features_dc": new_feat,
            "features_rest": new_offsets,
            "opacities": new_opacities,
        }

        self._anchor = nn.Parameter(torch.cat([self._anchor, d["means"]], dim=0))
        self._scaling = nn.Parameter(torch.cat([self._scaling, d["scales"]], dim=0))
        self._anchor_feat = nn.Parameter(torch.cat([self._anchor_feat, d["features_dc"]], dim=0))
        self._offset = nn.Parameter(torch.cat([self._offset, d["features_rest"]], dim=0))
        self._opacity = nn.Parameter(torch.cat([self._opacity, d["opacities"]], dim=0))

        self.opacity_accum = torch.cat(
            [self.opacity_accum, torch.zeros((num_new, 1), device=device)], dim=0
        )
        self.anchor_denom = torch.cat(
            [self.anchor_denom, torch.zeros((num_new, 1), device=device)], dim=0
        )

        if optimizers is not None:
            self.update_optimizers_after_growth(d, optimizers)

    # ── Optimizer / pruning helpers ───────────────────────────────────────────

    # Mapping: unified GSOptimizers key → internal parameter attribute name.
    # SemanticScaffoldModel overrides this dict to add "features_semantics".
    _PARAM_ATTR: Dict[str, str] = {
        "means": "_anchor",
        "scales": "_scaling",
        "opacities": "_opacity",
        "features_dc": "_anchor_feat",
        "features_rest": "_offset",
    }

    def _get_opt(
        self, optimizers: GSOptimizers, name: str
    ) -> Optional[torch.optim.Optimizer]:
        if not hasattr(optimizers, name):
            return None
        return getattr(optimizers, name)

    def update_optimizers_after_growth(
        self, new_data: Dict[str, torch.Tensor], optimizers: GSOptimizers
    ) -> None:
        """Extend Adam moment tensors and re-point optimizers to grown Parameters."""
        for name, tensor in new_data.items():
            opt = self._get_opt(optimizers, name)
            if opt is None:
                continue

            param = opt.param_groups[0]["params"][0]
            stored_state = opt.state.get(param, None)
            attr = self._PARAM_ATTR[name]
            new_param = getattr(self, attr)

            if stored_state is not None:
                stored_state["exp_avg"] = torch.cat(
                    [stored_state["exp_avg"], torch.zeros_like(tensor)], dim=0
                )
                stored_state["exp_avg_sq"] = torch.cat(
                    [stored_state["exp_avg_sq"], torch.zeros_like(tensor)], dim=0
                )
                del opt.state[param]
                opt.param_groups[0]["params"][0] = new_param
                opt.state[new_param] = stored_state
            else:
                opt.param_groups[0]["params"][0] = new_param

    def prune_anchor(
        self, mask: torch.Tensor, optimizers: Optional[GSOptimizers] = None
    ) -> None:
        valid_mask = ~mask
        self._anchor = nn.Parameter(self._anchor[valid_mask])
        self._offset = nn.Parameter(self._offset[valid_mask])
        self._anchor_feat = nn.Parameter(self._anchor_feat[valid_mask])
        self._scaling = nn.Parameter(self._scaling[valid_mask])
        self._opacity = nn.Parameter(self._opacity[valid_mask])

        if optimizers is not None:
            self.update_optimizers_after_pruning(valid_mask, optimizers)

    def update_optimizers_after_pruning(
        self, valid_mask: torch.Tensor, optimizers: GSOptimizers
    ) -> None:
        """Slice Adam moment tensors and re-point optimizers after anchor pruning."""
        for name, attr in self._PARAM_ATTR.items():
            opt = self._get_opt(optimizers, name)
            if opt is None:
                continue

            param = opt.param_groups[0]["params"][0]
            stored_state = opt.state.get(param, None)

            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][valid_mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][valid_mask]
                del opt.state[param]
                new_param = getattr(self, attr)
                opt.param_groups[0]["params"][0] = new_param
                opt.state[new_param] = stored_state

    # ── Scheduler creation ───────────────────────────────────────────────────

    def create_schedulers(self, optimizers: "GSOptimizers", iterations: int) -> "GS_LR_Schedulers":
        """Scaffold-GS-specific LR schedulers, calibrated to official final LR targets.

        Named-field schedule:
          features_rest (_offset): 0.01 → 0.0001 (100×, mirrors official offset_lr_final).
          All others (means, scales, opacities, features_dc): not scheduled —
            anchors are near-fixed; anchor features / scales / rotation converge
            without decay at these modest LRs.

        MLP extras (per-head eta_min calibrated to official final values):
          mlp_opacity : 0.002 → 2e-5  (official mlp_opacity_lr_final  = 0.00002)
          mlp_cov     : NOT scheduled  (official keeps constant at 0.004)
          mlp_color   : 0.008 → 5e-5  (official mlp_color_lr_final    = 0.00005)
          mlp_feature_bank / embedding_appearance / mlp_language:
            modest 100× decay if present.
        """
        CosineAnn = torch.optim.lr_scheduler.CosineAnnealingLR

        # --- named fields ---
        # _offset (features_rest): 0.01 → 0.0001, mirrors official offset_lr_final.
        offset_sched = CosineAnn(
            optimizers.features_rest, T_max=iterations, eta_min=1e-4
        )

        # --- extra (MLP) fields ---
        # eta_min values calibrated to official Scaffold-GS final LR targets.
        # mlp_cov is intentionally skipped — official holds it constant at 0.004.
        _eta_min: Dict[str, float] = {
            "mlp_opacity": 2e-5,  # official mlp_opacity_lr_final = 0.00002
            "mlp_color":   5e-5,  # official mlp_color_lr_final   = 0.00005
        }

        extra_scheds: Dict[str, torch.optim.lr_scheduler.LRScheduler] = {}
        for name, opt in self._extra_optimizers.items():
            if name == "mlp_cov":
                continue  # no scheduler — matches official constant schedule
            init_lr: float = opt.param_groups[0]["lr"]
            eta: float = _eta_min[name] if name in _eta_min else init_lr * 0.01
            extra_scheds[name] = CosineAnn(opt, T_max=iterations, eta_min=eta)

        self._extra_schedulers = extra_scheds

        return GS_LR_Schedulers(
            means=False,
            scales=False,
            quats=False,
            opacities=False,
            features_dc=False,
            features_rest=offset_sched,
            features_semantics=False,
        )

    # ── Optimizer creation ────────────────────────────────────────────────────

    def create_optimizers(
        self,
        lr_means: float = 0.00016,
        lr_scales: float = 0.005,
        lr_quats: float = 0.001,
        lr_opacities: float = 0.05,
        lr_sh: float = 0.0075,
        lr_semantics: Optional[float] = None,
        means_lr_multiplier: float = 5.0,
    ) -> GSOptimizers:
        extra_optimizers: Dict[str, torch.optim.Optimizer] = {
            "mlp_opacity": torch.optim.Adam(
                self.mlp_geo_heads_raw.opacity_head.parameters(), lr=self.lr_mlp_opacity
            ),
            "mlp_cov": torch.optim.Adam(self.mlp_geo_heads_raw.cov_head.parameters(), lr=self.lr_mlp_cov),
            "mlp_color": torch.optim.Adam(
                self.mlp_geo_heads_raw.color_head.parameters(), lr=self.lr_mlp_color
            ),
        }

        if self.mlp_feature_bank is not None:
            # Official featurebank lr_init = 0.01; use color-head LR as proxy (similar complexity).
            extra_optimizers["mlp_feature_bank"] = torch.optim.Adam(
                self.mlp_feature_bank.parameters(), lr=self.lr_mlp_color
            )

        if self.embedding_appearance is not None and self.appearance_dim > 0:
            extra_optimizers["embedding_appearance"] = torch.optim.Adam(
                self.embedding_appearance.parameters(), lr=self.lr_appearance
            )

        self._extra_optimizers = extra_optimizers
        self._extra_schedulers = {}

        return GSOptimizers(
            means=torch.optim.Adam([self._anchor], lr=lr_means * means_lr_multiplier),
            scales=torch.optim.Adam([self._scaling], lr=lr_scales),
            opacities=torch.optim.Adam([self._opacity], lr=lr_opacities),
            features_dc=torch.optim.Adam([self._anchor_feat], lr=lr_sh),
            features_rest=torch.optim.Adam([self._offset], lr=self.lr_offset),
            features_semantics=None,
        )

    def iter_extra_optimizers(self) -> Iterator[Tuple[str, torch.optim.Optimizer]]:
        for name, opt in self._extra_optimizers.items():
            if opt is not None:
                yield name, opt

    def iter_extra_schedulers(self) -> Iterator[Tuple[str, torch.optim.lr_scheduler.LRScheduler]]:
        for name, sched in self._extra_schedulers.items():
            if sched is not None:
                yield name, sched

    def get_extra_optimizer_states(self) -> Dict[str, dict]:
        return {name: opt.state_dict() for name, opt in self._extra_optimizers.items()}

    def load_extra_optimizer_states(self, checkpoint: Dict[str, object]) -> None:
        if not self._extra_optimizers:
            return
        source = checkpoint.get("extra_optimizers_state_dict")
        if not isinstance(source, dict):
            legacy = checkpoint.get("optimizers_state_dict")
            source = legacy if isinstance(legacy, dict) else {}
        for name, opt in self._extra_optimizers.items():
            state = source.get(name) if isinstance(source, dict) else None
            if state is not None:
                opt.load_state_dict(state)

    # ── Persistence ───────────────────────────────────────────────────────────

    def save_ply(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        anchor = self._anchor.detach().cpu().numpy()
        num_pts = anchor.shape[0]
        normals = np.zeros_like(anchor)
        dtype = [("x", "f4"), ("y", "f4"), ("z", "f4"), ("nx", "f4"), ("ny", "f4"), ("nz", "f4")]
        elements = np.empty(num_pts, dtype=dtype)
        attributes = np.concatenate((anchor, normals), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, "vertex")
        PlyData([el]).write(path)
        logger.info(f"Saved {num_pts} anchors to {path}")

    def save_checkpoints(self, path: str) -> None:
        # model_state_dict captures all nn.Parameters and sub-module weights including
        # _anchor_lang_feat, language_codebook, codebook_proj, and compiled MLP weights.
        torch.save(
            {
                "opacity_mlp": self.mlp_geo_heads_raw.opacity_head.state_dict(),
                "cov_mlp": self.mlp_geo_heads_raw.cov_head.state_dict(),
                "color_mlp": self.mlp_geo_heads_raw.color_head.state_dict(),
                "feature_bank_mlp": (
                    self.mlp_feature_bank.state_dict()
                    if self.mlp_feature_bank is not None
                    else None
                ),
                "appearance": (
                    self.embedding_appearance.state_dict()
                    if self.embedding_appearance is not None
                    else None
                ),
                "model_state_dict": self.state_dict(),
            },
            path,
        )

    def load_checkpoint(self, path: str) -> None:
        checkpoint = torch.load(path, weights_only=False)
        self.load_state_dict(checkpoint["model_state_dict"], strict=False)
        if checkpoint.get("opacity_mlp") is not None:
            self.mlp_geo_heads_raw.opacity_head.load_state_dict(checkpoint["opacity_mlp"])
        if checkpoint.get("cov_mlp") is not None:
            self.mlp_geo_heads_raw.cov_head.load_state_dict(checkpoint["cov_mlp"])
        if checkpoint.get("color_mlp") is not None:
            self.mlp_geo_heads_raw.color_head.load_state_dict(checkpoint["color_mlp"])
        if self.mlp_feature_bank is not None and checkpoint.get("feature_bank_mlp") is not None:
            self.mlp_feature_bank.load_state_dict(checkpoint["feature_bank_mlp"])
        if self.embedding_appearance is not None and checkpoint.get("appearance") is not None:
            self.embedding_appearance.load_state_dict(checkpoint["appearance"])
    # ── Unified render API ────────────────────────────────────────────────────

    def get_render_params(
        self, cam: dict, sh_cfg=None, is_training: bool = True, lod: Optional[int] = None
    ) -> RenderParams:
        out = self.generate_neural_gaussians(cam=cam, is_training=is_training)
        return RenderParams(
            means=out.means,
            colors=out.colors,
            opacities=out.opacities,
            scales=out.scales,
            quats=out.quats,
            sh_degree=None,
            neural_opacity=out.neural_opacity,
            selection_mask=out.selection_mask,
        )

    # ── LoD (compatibility stub) ──────────────────────────────────────────────

    def compute_lods(
        self, num_levels: int = 1, factor: int = 4, optimizers: Optional[GSOptimizers] = None
    ) -> None:
        if num_levels > 1:
            logger.warning(
                "[yellow]⚠️  Scaffold-GS does not support LoD levels. "
                f"Requested {num_levels} levels will be ignored.[/yellow]"
            )
        self.lod_offsets = [len(self._anchor)]
