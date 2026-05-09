"""SemanticScaffoldModel — ScaffoldModel with per-anchor patch-feature learning.

Architecture
------------
Each anchor owns a ``lang_feat_dim``-dim latent (``_anchor_lang_feat``).
``mlp_language`` predicts per-Gaussian view-conditional offsets in that latent
space.  The rendered [H, W, D] feature map is supervised against compressed
DINO patch features (dino_encoded provider) with a patch-grid MSE loss.

Only ``train_semantics.py`` instantiates this class.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from gs_types import GSOptimizers, NeuralGaussianOutput, RenderOutput, RenderParams
from gsplat import rasterization
from torch_scatter import scatter_max

from .base import SemanticsMixin
from .scaffold import ScaffoldModel, normalize_safe

logger = logging.getLogger("cityscape_gs.models.semantic_scaffold")


def _build_viewmat(cam: dict, device: torch.device) -> torch.Tensor:
    viewmat = torch.eye(4, device=device, dtype=torch.float32)
    viewmat[:3, :3] = cam["R"]
    viewmat[:3, 3] = cam["T"]
    return viewmat.unsqueeze(0)


def _build_K(cam: dict, device: torch.device) -> torch.Tensor:
    return torch.tensor(
        [[cam["fx"], 0.0, cam["cx"]], [0.0, cam["fy"], cam["cy"]], [0.0, 0.0, 1.0]],
        dtype=torch.float32, device=device,
    )

class CrossModalGaussianAttention(nn.Module):
    def __init__(self, color_dim: int = 3, lang_dim: int = 32):
        super().__init__()
        # Project color up to lang_dim so both tokens live in same space
        self.color_proj = nn.Linear(color_dim, lang_dim)

        # Standard Q, K, V projections — operate on the 2-token sequence
        self.q_proj = nn.Linear(lang_dim, lang_dim)
        self.k_proj = nn.Linear(lang_dim, lang_dim)
        self.v_proj = nn.Linear(lang_dim, lang_dim)
        self.out_proj = nn.Linear(lang_dim, lang_dim)

        self.scale = lang_dim ** -0.5

    def forward(self, colors: torch.Tensor, lang: torch.Tensor) -> torch.Tensor:
        # colors: [M, 3],  lang: [M, D]
        M, D = lang.shape

        # Both modalities projected to same space
        c_embed = self.color_proj(colors)         # [M, D]

        # Stack into 2-token sequence: token-0 = color, token-1 = language
        tokens = torch.stack([c_embed, lang], dim=1)   # [M, 2, D]

        Q = self.q_proj(tokens)   # [M, 2, D]
        K = self.k_proj(tokens)   # [M, 2, D]
        V = self.v_proj(tokens)   # [M, 2, D]

        # Attention weights: [M, 2, 2] — each token attends to both tokens
        attn_weights = F.softmax(
            (Q @ K.transpose(-1, -2)) * self.scale, dim=-1
        )                         # [M, 2, 2]

        attended = attn_weights @ V               # [M, 2, D]

        # Extract only the language token's attended output
        lang_attended = attended[:, 1, :]         # [M, D]  — the semantic token's output

        # Residual: preserve original language feature, only add the refined delta
        lang_refined = lang + self.out_proj(lang_attended)  # [M, D]

        return lang_refined

class SemanticScaffoldModel(ScaffoldModel, SemanticsMixin):
    """ScaffoldModel extended with per-anchor language feature learning.

    Added parameters over the base ScaffoldModel:
    - ``_anchor_lang_feat [N, lang_feat_dim]`` — per-anchor language feature latents.
    - ``mlp_language``  — view-conditional per-Gaussian offsets in latent space.
    - ``mlp_lang_proj``  — projects per-Gaussian features from lang_feat_dim to semantics_dim.

    Supervision: MSE on the rendered [H, W, semantics_dim] feature map against
    DINO patch features (bottleneck-dim == semantics_dim).
    """

    # Include semantic anchor param in _PARAM_ATTR so the inherited
    # grow/prune optimizer-update helpers handle it automatically.
    _PARAM_ATTR: Dict[str, str] = {
        **ScaffoldModel._PARAM_ATTR,
        "features_semantics": "_anchor_lang_feat",
    }

    def __init__(
        self,
        init_points: Optional[torch.Tensor],
        # ── All standard ScaffoldModel args ──
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
        console=None,
        # ── Semantic-specific args ──
        lang_feat_dim: int = 32,
        semantics_dim: int = 4,
        *args,
        **kwargs,
    ):
        # Must be set before super().__init__ because ScaffoldModel conditionally
        # calls self.create_from_pcd(init_points), which dispatches to our override
        # and reads self.lang_feat_dim.  When init_points=None (fine-tuning path),
        # create_from_pcd is never called; _init_lang_feat() runs in from_pretrained_base.
        self.lang_feat_dim = lang_feat_dim
        self._semantics_dim = semantics_dim

        super().__init__(
            init_points=init_points,
            feat_dim=feat_dim,
            n_offsets=n_offsets,
            voxel_size=voxel_size,
            update_depth=update_depth,
            update_init_factor=update_init_factor,
            update_hierachy_factor=update_hierachy_factor,
            use_feat_bank=use_feat_bank,
            appearance_dim=appearance_dim,
            add_opacity_dist=add_opacity_dist,
            add_cov_dist=add_cov_dist,
            add_color_dist=add_color_dist,
            sh_degree=sh_degree,
            fourier_freqs=fourier_freqs,
            fourier_scale=fourier_scale,
            lr_offset=lr_offset,
            lr_mlp_opacity=lr_mlp_opacity,
            lr_mlp_cov=lr_mlp_cov,
            lr_mlp_color=lr_mlp_color,
            lr_appearance=lr_appearance,
            console=console,
        )

        # Empty placeholder — replaced by _init_lang_feat(N) once anchor count is known.
        self._anchor_lang_feat: nn.Parameter = nn.Parameter(torch.empty(0))

        # [anchor_lang_feat(D) | ob_view(3)] → D * n_offsets view-conditional offsets.
        self.mlp_language: nn.Module = nn.Sequential(
            nn.Linear(lang_feat_dim + self.fourier_embed_dim, lang_feat_dim),
            nn.ReLU(True),
            nn.Linear(lang_feat_dim, lang_feat_dim * n_offsets),
            nn.Tanh(),
        )

        # Project per-Gaussian language features from lang_feat_dim to semantics_dim
        # for supervision against DINO bottleneck features.
        # self.mlp_lang_proj: nn.Module = nn.Linear(lang_feat_dim, semantics_dim)
        self.mlp_lang_proj: nn.Module = nn.Sequential(
            nn.Linear(lang_feat_dim, lang_feat_dim),
            nn.ReLU(True),
            nn.Linear(lang_feat_dim, semantics_dim),
        )

        self.cross_modal_attn = CrossModalGaussianAttention(
            color_dim=3,
            lang_dim=semantics_dim
        )

    # ── Language feature initialisation ──────────────────────────────────────

    def _init_lang_feat(self, num_anchors: int, device: torch.device) -> None:
        """Allocate _anchor_lang_feat to zeros for ``num_anchors`` anchors.

        Called by create_from_pcd (fresh training) and from_pretrained_base
        (fine-tuning from a geometry-only checkpoint).
        """
        self._anchor_lang_feat = nn.Parameter(
            torch.zeros((num_anchors, self.lang_feat_dim), device=device)
        )

    def create_from_pcd(self, points: torch.Tensor) -> None:
        super().create_from_pcd(points)
        self._init_lang_feat(self._anchor.shape[0], points.device)

    # ── Neural Gaussian generation (adds language feature path) ──────────────

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

        # --- Language features (semantic extension) ---
        lang_feat = self._anchor_lang_feat[visible_mask]  # [N_vis, D]
        lang_input = torch.cat([lang_feat, embedded_view], dim=1)  # [N_vis, D + fourier_embed_dim]
        lang_offsets = self.mlp_language(lang_input)  # [N_vis, D*k]
        lang_offsets = lang_offsets.view(-1, self.lang_feat_dim)  # [N_vis*k, D]

        lang_feat_rep = (
            lang_feat.unsqueeze(1)
            .expand(-1, self.n_offsets, -1)
            .reshape(-1, self.lang_feat_dim)
        )
        per_gaussian_lang = lang_feat_rep + lang_offsets  # [N_vis*k, D]
        masked_lang = per_gaussian_lang[mask]  # [M, D] — apply same selection mask

        if is_training:
            return NeuralGaussianOutput(
                means=xyz,
                colors=masked_color,
                opacities=opacity,
                scales=scaling,
                quats=rot,
                neural_opacity=neural_opacity,
                selection_mask=mask,
                language_features=masked_lang,
            )
        else:
            return NeuralGaussianOutput(
                means=xyz,
                colors=masked_color,
                opacities=opacity,
                scales=scaling,
                quats=rot,
                language_features=masked_lang,
            )

    # ── SemanticsMixin interface ──────────────────────────────────────────────

    @property
    def semantics_dim(self) -> int:
        return self._semantics_dim

    @property
    def provider_semantics_dim(self) -> int:
        """Provider returns bottleneck-dim features == semantics_dim (DINO bottleneck)."""
        return self._semantics_dim

    @property
    def anchor_lang_feat(self) -> torch.Tensor:
        return self._anchor_lang_feat

    @property
    def geometry_param_group_names(self) -> frozenset:
        return frozenset({"geometry", "mlp_geo", "anchor_features", "appearance"})

    def get_finetune_param_groups(self) -> Dict[str, List[nn.Parameter]]:
        groups: Dict[str, List[nn.Parameter]] = {
            "semantics": (
                [self._anchor_lang_feat]
                + list(self.mlp_language.parameters())
                + list(self.mlp_lang_proj.parameters())
                + list(self.cross_modal_attn.parameters())
            ),
            "anchor_features": [self._anchor_feat],
            "mlp_geo":         list(self.mlp_geo_heads_raw.parameters()),
            "geometry":        [self._anchor, self._scaling, self._opacity, self._offset],
        }
        if self.embedding_appearance is not None:
            groups["appearance"] = list(self.embedding_appearance.parameters())
        return groups

    def render_semantics(
        self,
        cam: dict,
        device: torch.device,
        detach_geometry: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Rasterise RGB + language features jointly.

        Args:
            detach_geometry: When True (default) geometry tensors are detached
                so only language parameters receive gradients.  Set to False
                for joint geometry + semantic training.

        Returns:
            rgb_pred:  [H, W, 3]
            feat_pred: [H, W, semantics_dim] — projected to match DINO bottleneck
        """

        viewmat = _build_viewmat(cam, device)
        K = _build_K(cam, device)

        out = self.generate_neural_gaussians(cam, is_training=True)
        torch.cuda.synchronize()
        assert out.language_features is not None

        colors = out.colors.detach() if detach_geometry else out.colors
        # Project language features from lang_feat_dim to semantics_dim for supervision
        # lang_feat_proj = self.mlp_lang_proj(out.language_features)  # [M, semantics_dim]
        # colors_with_lang = torch.cat([colors, lang_feat_proj], dim=-1)

        lang_feat_proj = self.mlp_lang_proj(out.language_features)  # [M, D]
        lang_feat_proj = self.cross_modal_attn(colors, lang_feat_proj)  # [M, D]  ← insert here
        colors_with_lang = torch.cat([colors, lang_feat_proj], dim=-1)

        means   = out.means.detach()   if detach_geometry else out.means
        quats   = out.quats.detach()   if detach_geometry else out.quats
        scales  = out.scales.detach()  if detach_geometry else out.scales
        opacities = out.opacities.squeeze(-1).detach() if detach_geometry else out.opacities.squeeze(-1)

        rendered, _, _ = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors_with_lang,
            viewmats=viewmat,
            Ks=K[None, ...],
            width=cam["width"],
            height=cam["height"],
            sh_degree=None,
            render_mode="RGB",
        )

        rendered = rendered[0].float()  # [H, W, 3 + semantics_dim]
        return rendered[..., :3], rendered[..., 3:]

    def get_semantic_trainable_params(self) -> List[nn.Parameter]:
        params: List[nn.Parameter] = [self._anchor_lang_feat]
        params.extend(self.mlp_language.parameters())
        params.extend(self.mlp_lang_proj.parameters())   # ← also missing
        params.extend(self.cross_modal_attn.parameters()) # ← missing
        return params

    def setup_semantic_training(self, semantics_cfg: object, device: torch.device) -> None:
        """No-op — dino_encoded features are pre-compressed; no setup needed."""
        logger.info(
            "SemanticScaffoldModel: dino_encoded mode, lang_feat_dim=%d. "
            "No pre-training setup required.",
            self.lang_feat_dim,
        )

    def prepare_target(self, raw_target: torch.Tensor, device: torch.device) -> torch.Tensor:
        """Identity transform — dino_encoded features are already bottleneck-dim."""
        return raw_target.float()

    # ── Densification: grow / prune _anchor_lang_feat alongside geometry ──────

    def add_anchors(
        self,
        new_anchors: torch.Tensor,
        candidate_mask: torch.Tensor,
        inverse_indices: torch.Tensor,
        new_mask: torch.Tensor,
        cur_size: float,
        optimizers: Optional[GSOptimizers] = None,
    ) -> None:
        """Grow _anchor_lang_feat when new anchors are added."""
        super().add_anchors(
            new_anchors, candidate_mask, inverse_indices, new_mask, cur_size,
            optimizers=None,
        )

        num_new = new_anchors.shape[0]
        device = self._anchor.device

        inherited_lang = (
            self._anchor_lang_feat.unsqueeze(1)
            .repeat(1, self.n_offsets, 1)
            .view(-1, self.lang_feat_dim)[candidate_mask]
        )
        new_lang = scatter_max(
            inherited_lang,
            inverse_indices.unsqueeze(1).expand(-1, inherited_lang.size(1)),
            dim=0,
        )[0][new_mask]
        self._anchor_lang_feat = nn.Parameter(
            torch.cat([self._anchor_lang_feat, new_lang], dim=0)
        )

        if optimizers is not None:
            new_data: Dict[str, torch.Tensor] = {
                "means": new_anchors,
                "scales": torch.log(torch.ones((num_new, 6), device=device) * cur_size),
                "opacities": torch.zeros((num_new, 1), device=device),
                "features_dc": torch.zeros((num_new, self.feat_dim), device=device),
                "features_rest": torch.zeros((num_new, self.n_offsets, 3), device=device),
                "features_semantics": new_lang,
            }
            self.update_optimizers_after_growth(new_data, optimizers)

    def prune_anchor(
        self,
        mask: torch.Tensor,
        optimizers: Optional[GSOptimizers] = None,
    ) -> None:
        valid_mask = ~mask
        super().prune_anchor(mask, optimizers=None)
        self._anchor_lang_feat = nn.Parameter(self._anchor_lang_feat[valid_mask])
        if optimizers is not None:
            self.update_optimizers_after_pruning(valid_mask, optimizers)

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
        base_opts = super().create_optimizers(
            lr_means=lr_means, lr_scales=lr_scales, lr_quats=lr_quats,
            lr_opacities=lr_opacities, lr_sh=lr_sh, lr_semantics=None,
            means_lr_multiplier=means_lr_multiplier,
        )
        lr_lang = lr_semantics if lr_semantics is not None else lr_sh
        lang_feat_opt = torch.optim.Adam([self._anchor_lang_feat], lr=lr_lang)
        self._extra_optimizers["mlp_language"] = torch.optim.Adam(
            list(self.mlp_language.parameters())
            + list(self.mlp_lang_proj.parameters())
            + list(self.cross_modal_attn.parameters()),
            lr=self.lr_mlp_color,
        )
        return GSOptimizers(
            means=base_opts.means,
            scales=base_opts.scales,
            quats=base_opts.quats,
            opacities=base_opts.opacities,
            features_dc=base_opts.features_dc,
            features_rest=base_opts.features_rest,
            features_semantics=lang_feat_opt,
        )

    # ── Persistence ───────────────────────────────────────────────────────────

    def save_checkpoints(self, path: str) -> None:
        torch.save(
            {
                "opacity_mlp": self.mlp_geo_heads_raw.opacity_head.state_dict(),
                "cov_mlp": self.mlp_geo_heads_raw.cov_head.state_dict(),
                "color_mlp": self.mlp_geo_heads_raw.color_head.state_dict(),
                "feature_bank_mlp": (
                    self.mlp_feature_bank.state_dict()
                    if self.mlp_feature_bank is not None else None
                ),
                "appearance": (
                    self.embedding_appearance.state_dict()
                    if self.embedding_appearance is not None else None
                ),
                "language_mlp":    self.mlp_language.state_dict(),
                "language_proj":   self.mlp_lang_proj.state_dict(),
                "cross_modal_attn": self.cross_modal_attn.state_dict(),
                "lang_feat_dim": self.lang_feat_dim,
                "semantics_dim": self._semantics_dim,
                "model_state_dict": self.state_dict(),
            },
            path,
        )

    def load_checkpoint(self, path: str) -> None:
        checkpoint = torch.load(path, weights_only=False)
        self.load_state_dict(checkpoint["model_state_dict"], strict=False)
        if checkpoint.get("opacity_mlp"):
            self.mlp_geo_heads_raw.opacity_head.load_state_dict(checkpoint["opacity_mlp"])
        if checkpoint.get("cov_mlp"):
            self.mlp_geo_heads_raw.cov_head.load_state_dict(checkpoint["cov_mlp"])
        if checkpoint.get("color_mlp"):
            self.mlp_geo_heads_raw.color_head.load_state_dict(checkpoint["color_mlp"])
        if self.mlp_feature_bank is not None and checkpoint.get("feature_bank_mlp"):
            self.mlp_feature_bank.load_state_dict(checkpoint["feature_bank_mlp"])
        if self.embedding_appearance is not None and checkpoint.get("appearance"):
            self.embedding_appearance.load_state_dict(checkpoint["appearance"])
        if checkpoint.get("language_mlp"):
            self.mlp_language.load_state_dict(checkpoint["language_mlp"])
        if checkpoint.get("language_proj"):
            self.mlp_lang_proj.load_state_dict(checkpoint["language_proj"])
        if checkpoint.get("cross_modal_attn"):
            self.cross_modal_attn.load_state_dict(checkpoint["cross_modal_attn"])

    # ── Checkpoint loaders ────────────────────────────────────────────────────

    @classmethod
    def from_pretrained_base(
        cls,
        checkpoint_path: Path,
        device: torch.device,
        lang_feat_dim: int = 32,
        semantics_dim: int = 4,
        console=None,
        **scaffold_kwargs,
    ) -> "SemanticScaffoldModel":
        """Load a geometry-only ScaffoldModel checkpoint and add language features.

        ``_anchor_lang_feat`` is initialised to zeros; ``mlp_language`` and
        ``mlp_lang_proj`` weights are randomly initialised.  Call train_semantics
        to fine-tune them.

        Args:
            lang_feat_dim: Per-anchor language feature dimension
            semantics_dim: Output dimension for DINO bottleneck supervision
        """
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        state_dict = checkpoint.get("model_state_dict", {})

        if "_anchor" not in state_dict:
            raise RuntimeError(
                f"Checkpoint {checkpoint_path} has no '_anchor' key. "
                "Make sure you are loading a valid ScaffoldModel checkpoint."
            )
        num_anchors = state_dict["_anchor"].shape[0]

        model = cls(
            init_points=None,
            lang_feat_dim=lang_feat_dim,
            semantics_dim=semantics_dim,
            console=console,
            **scaffold_kwargs,
        )
        model.to(device)

        if "embedding_appearance.embedding.weight" in state_dict:
            num_cameras = state_dict["embedding_appearance.embedding.weight"].shape[0]
            model.set_appearance(num_cameras)
            if model.embedding_appearance is not None:
                model.embedding_appearance.to(device)

        # Replace param data directly to handle shape differences (strict=False
        # silently skips shape mismatches on in-place copy, leaving params empty).
        # Skip language MLP params — they're handled below with shape validation.
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in state_dict and not name.startswith(("mlp_language", "mlp_lang_proj")):
                    param.data = state_dict[name].to(device)
            for name, buf in model.named_buffers():
                if name in state_dict:
                    buf.data = state_dict[name].to(device)

        model.load_state_dict(state_dict, strict=False)
        model.lod_offsets = [num_anchors]
        model._init_lang_feat(num_anchors, device)

        # Reinitialize mlp_language if checkpoint had a different fourier_embed_dim.
        # Use weight.shape[1] (actual tensor shape) not in_features (stale int attribute).
        expected_input_size = lang_feat_dim + model.fourier_embed_dim
        actual_input_size = model.mlp_language[0].weight.shape[1]
        if actual_input_size != expected_input_size:
            logger.warning(
                "Reinitializing mlp_language: fourier_embed_dim changed. "
                "Expected input size %d, got %d.",
                expected_input_size, actual_input_size,
            )
            model.mlp_language = nn.Sequential(
                nn.Linear(expected_input_size, lang_feat_dim),
                nn.ReLU(True),
                nn.Linear(lang_feat_dim, lang_feat_dim * model.n_offsets),
                nn.Tanh(),
            ).to(device)

        logger.info(
            "SemanticScaffoldModel loaded from %s (%d anchors, lang_feat_dim=%d)",
            checkpoint_path, num_anchors, lang_feat_dim,
        )
        return model

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: Path,
        device: torch.device,
        console=None,
    ) -> "SemanticScaffoldModel":
        """Load a SemanticScaffoldModel checkpoint (produced by a previous semantic run)."""
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        state_dict = checkpoint.get("model_state_dict", {})
        lang_feat_dim = checkpoint.get("lang_feat_dim", 32)
        # For backward compat: if semantics_dim is not in checkpoint, use lang_feat_dim
        semantics_dim = checkpoint.get("semantics_dim", lang_feat_dim)

        model = cls(init_points=None, lang_feat_dim=lang_feat_dim, semantics_dim=semantics_dim, console=console)
        model.to(device)

        if "embedding_appearance.embedding.weight" in state_dict:
            num_cameras = state_dict["embedding_appearance.embedding.weight"].shape[0]
            model.set_appearance(num_cameras)
            if model.embedding_appearance is not None:
                model.embedding_appearance.to(device)

        with torch.no_grad():
            for name, param in model.named_parameters():
                # Skip language MLP params — handled below with shape validation
                if name in state_dict and not name.startswith(("mlp_language", "mlp_lang_proj")):
                    param.data = state_dict[name].to(device)
            for name, buf in model.named_buffers():
                if name in state_dict:
                    buf.data = state_dict[name].to(device)

        model.load_state_dict(state_dict, strict=False)

        if model._anchor.numel() > 0:
            model.lod_offsets = [model._anchor.shape[0]]

        # Load language MLP if available and shapes match
        if checkpoint.get("language_mlp"):
            expected_input_size = lang_feat_dim + model.fourier_embed_dim
            actual_input_size = checkpoint["language_mlp"].get(
                "0.weight", torch.zeros(1, 1)
            ).shape[1] if "0.weight" in checkpoint["language_mlp"] else None
            if actual_input_size == expected_input_size:
                model.mlp_language.load_state_dict(checkpoint["language_mlp"])
            else:
                logger.warning(
                    "Skipping mlp_language weights: fourier_embed_dim mismatch. "
                    "Expected input %d, checkpoint had %d. Reinitializing.",
                    expected_input_size, actual_input_size,
                )

        if checkpoint.get("language_proj"):
            actual = checkpoint["language_proj"].get("2.weight", torch.zeros(1, 1)).shape[0]
            if actual == semantics_dim:
                model.mlp_lang_proj.load_state_dict(checkpoint["language_proj"])
            else:
                logger.warning(
                    "Skipping mlp_lang_proj: semantics_dim mismatch (%d vs %d).",
                    actual, semantics_dim,
                )

        if checkpoint.get("cross_modal_attn"):
            actual = checkpoint["cross_modal_attn"].get("color_proj.weight", torch.zeros(1, 1)).shape[0]
            if actual == semantics_dim:
                model.cross_modal_attn.load_state_dict(checkpoint["cross_modal_attn"])
            else:
                logger.warning(
                    "Skipping cross_modal_attn: semantics_dim mismatch (%d vs %d).",
                    actual, semantics_dim,
                )

        logger.info(
            "SemanticScaffoldModel (semantic ckpt) loaded from %s", checkpoint_path
        )
        return model
