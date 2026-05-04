"""SemanticGaussianModel — GaussianModel with per-Gaussian semantic features.

Separates semantic responsibility from the base GaussianModel so that
train.py never touches semantic code.  Only train_semantics.py instantiates
this class.

Typical workflow
----------------
1. train.py trains a GaussianModel checkpoint (visual only, no semantics).
2. train_semantics.py calls SemanticGaussianModel.from_pretrained_base() to
   load that checkpoint and extend it with _features_semantics (zero-init).
3. train_semantics.py's SemanticTrainer freezes geometry and fine-tunes
   _features_semantics via the SemanticsMixin interface.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from gs_types import GSOptimizers
from .base import SemanticsMixin
from .gaussian import GaussianModel

logger = logging.getLogger("cityscape_gs.models.semantic_gaussian")


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


class SemanticGaussianModel(GaussianModel, SemanticsMixin):
    """GaussianModel extended with a per-Gaussian semantic feature field.

    All geometry (means, scales, quats, opacities, SH) is inherited unchanged
    from GaussianModel.  This class adds ``_features_semantics [N, semantics_dim]``
    and implements the SemanticsMixin interface so that SemanticTrainer can
    operate on it without knowing the model type.
    """

    def __init__(
        self,
        init_points: torch.Tensor,
        init_colors: torch.Tensor,
        sh_degree: int = 3,
        semantics_dim: int = 3,
        console=None,
    ):
        super().__init__(
            init_points=init_points,
            init_colors=init_colors,
            sh_degree=sh_degree,
            console=console,
        )
        if semantics_dim <= 0:
            raise ValueError(f"semantics_dim must be positive, got {semantics_dim}")
        self._semantics_dim = semantics_dim
        self._features_semantics = nn.Parameter(
            torch.zeros(init_points.shape[0], semantics_dim, device=init_points.device)
        )

    # ── SemanticsMixin interface ──────────────────────────────────────────────

    @property
    def semantics_dim(self) -> int:
        return self._semantics_dim

    def render_semantics(
        self,
        cam: dict,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Rasterise geometric + semantic features jointly.

        Geometry tensors are detached so only ``_features_semantics`` receives
        gradients.  RGB and semantic channels are concatenated into a single
        rasterisation call for efficiency.

        Returns:
            rgb_pred:      [H, W, 3]
            semantic_pred: [H, W, semantics_dim]
        """
        from gsplat import rasterization

        viewmat = _build_viewmat(cam, device)
        K = _build_K(cam, device)

        # DC component in SH-coefficient space → detach so geometry gets no grad
        rgb_dc = self._features_dc.squeeze(1).detach()       # [N, 3]
        colors = torch.cat([rgb_dc, self._features_semantics], dim=-1)  # [N, 3+D]

        out, _, _ = rasterization(
            means=self.means.detach(),
            quats=self.quats.detach(),
            scales=self.scales.detach(),
            opacities=self.opacities.squeeze(-1).detach(),
            colors=colors,
            viewmats=viewmat,
            Ks=K[None, ...],
            width=cam["width"],
            height=cam["height"],
            sh_degree=None,
            render_mode="RGB",
        )

        rendered = out[0].float()  # [H, W, 3+D]
        return rendered[..., :3], rendered[..., 3:]

    def get_semantic_trainable_params(self) -> List[nn.Parameter]:
        return [self._features_semantics]

    # ── Densification support (reorder semantic features with geometry) ───────

    def reorder_gaussians(
        self, indices: torch.Tensor, optimizers: Optional[GSOptimizers] = None
    ) -> None:
        """Reorder geometry AND semantic features by the given index permutation."""
        super().reorder_gaussians(indices, optimizers)
        self._features_semantics.data = self._features_semantics.data[indices]

    def get_params_dict(self) -> Dict[str, nn.Parameter]:
        params = super().get_params_dict()
        params["features_semantics"] = self._features_semantics
        return params

    def get_optimizers_dict(self, optimizers: GSOptimizers) -> Dict[str, torch.optim.Optimizer]:
        d = super().get_optimizers_dict(optimizers)
        if optimizers.features_semantics is not None:
            d["features_semantics"] = optimizers.features_semantics
        return d

    def update_params_from_dict(self, params: Dict[str, nn.Parameter]) -> None:
        super().update_params_from_dict(params)
        if "features_semantics" in params:
            self._features_semantics = params["features_semantics"]

    def create_optimizers(
        self,
        lr_means: float = 0.00016,
        lr_scales: float = 0.005,
        lr_quats: float = 0.001,
        lr_opacities: float = 0.05,
        lr_sh: float = 0.0025,
        lr_semantics: Optional[float] = None,
        means_lr_multiplier: float = 5.0,
    ) -> GSOptimizers:
        opts = super().create_optimizers(
            lr_means=lr_means, lr_scales=lr_scales, lr_quats=lr_quats,
            lr_opacities=lr_opacities, lr_sh=lr_sh, lr_semantics=lr_semantics,
            means_lr_multiplier=means_lr_multiplier,
        )
        lr_sem = lr_semantics if lr_semantics is not None else lr_sh
        return GSOptimizers(
            means=opts.means,
            scales=opts.scales,
            quats=opts.quats,
            opacities=opts.opacities,
            features_dc=opts.features_dc,
            features_rest=opts.features_rest,
            features_semantics=torch.optim.Adam([self._features_semantics], lr=lr_sem),
        )

    # ── Checkpoint helpers ────────────────────────────────────────────────────

    @classmethod
    def from_pretrained_base(
        cls,
        checkpoint_path: Path,
        device: torch.device,
        semantics_dim: int = 3,
        sh_degree: Optional[int] = None,
        console=None,
        strict: bool = False,
    ) -> "SemanticGaussianModel":
        """Load a visual-only GaussianModel checkpoint and extend it with semantic features.

        The semantic field ``_features_semantics`` is zero-initialised; the
        subsequent fine-tuning loop will train it from scratch.

        Args:
            checkpoint_path: Path to a GaussianModel ``.pt`` checkpoint.
            device:          Target device.
            semantics_dim:   Dimension of the semantic feature field to add.
            sh_degree:       Optional SH degree override (inferred from checkpoint if None).
            console:         Optional rich console.
            strict:          Whether to enforce strict state_dict loading.

        Returns:
            SemanticGaussianModel with geometry loaded and semantics zero-initialised.
        """
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if "model_state_dict" not in checkpoint:
            raise KeyError(f"Checkpoint missing 'model_state_dict': {checkpoint_path}")

        state_dict = dict(checkpoint["model_state_dict"])
        if "_means" not in state_dict:
            raise KeyError("Checkpoint state_dict missing '_means'")

        num_pts = state_dict["_means"].shape[0]

        if sh_degree is None:
            fr = state_dict.get("_features_rest")
            sh_degree = int((fr.shape[1] + 1) ** 0.5) - 1 if fr is not None else 3

        dummy = torch.zeros((num_pts, 3), device=device)
        model = cls(
            init_points=dummy,
            init_colors=dummy,
            sh_degree=sh_degree,
            semantics_dim=semantics_dim,
            console=console,
        ).to(device)

        # Remove stale semantic state from old-format checkpoints (if any)
        state_dict.pop("_features_semantics", None)
        state_dict.pop("view_count", None)

        model.load_state_dict(state_dict, strict=strict)
        model.view_count = torch.zeros(num_pts, device=device)

        logger.info(
            "[green]✓ SemanticGaussianModel loaded from[/green] %s (%d Gaussians, semantics_dim=%d)",
            checkpoint_path, num_pts, semantics_dim,
        )
        return model

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: Path,
        device: torch.device,
        console=None,
        strict: bool = False,
    ) -> "SemanticGaussianModel":
        """Load a SemanticGaussianModel checkpoint (produced by a previous semantic run).

        Args:
            checkpoint_path: Path to a SemanticGaussianModel ``.pt`` checkpoint.
            device:          Target device.
            console:         Optional rich console.
            strict:          Whether to enforce strict state_dict loading.

        Returns:
            Fully populated SemanticGaussianModel.
        """
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        state_dict = dict(checkpoint["model_state_dict"])

        num_pts = state_dict["_means"].shape[0]
        fr = state_dict.get("_features_rest")
        sh_degree = int((fr.shape[1] + 1) ** 0.5) - 1 if fr is not None else 3

        sem = state_dict.get("_features_semantics")
        if sem is None:
            raise KeyError(
                f"Checkpoint {checkpoint_path} has no '_features_semantics'. "
                "Use from_pretrained_base() to create a new semantic model from a visual checkpoint."
            )
        semantics_dim = sem.shape[1]

        dummy = torch.zeros((num_pts, 3), device=device)
        model = cls(
            init_points=dummy,
            init_colors=dummy,
            sh_degree=sh_degree,
            semantics_dim=semantics_dim,
            console=console,
        ).to(device)

        state_dict.pop("view_count", None)
        model.load_state_dict(state_dict, strict=strict)
        model.view_count = torch.zeros(num_pts, device=device)

        logger.info(
            "[green]✓ SemanticGaussianModel loaded from[/green] %s (%d Gaussians, semantics_dim=%d)",
            checkpoint_path, num_pts, semantics_dim,
        )
        return model
