"""Base class and capability mixins for all trainable Gaussian Splatting models."""

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Dict, Optional

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from gs_types import GSOptimizers, GS_LR_Schedulers, NeuralGaussianOutput, RenderParams

logger = logging.getLogger("cityscape_gs.models.base")


# ─────────────────────────────────────────────────────────────────────────────
# Capability Mixins (scalable pattern for optional features)
# ─────────────────────────────────────────────────────────────────────────────


class NeuralRenderingMixin(ABC):
    """Mixin for models that generate Gaussians dynamically per-view.

    Example: Scaffold-GS generates neural Gaussians from anchors on-the-fly.

    Usage: isinstance(model, NeuralRenderingMixin) for type-safe feature detection.
    """

    @abstractmethod
    def generate_neural_gaussians(
        self,
        cam: dict,
        visible_mask: Optional[torch.Tensor] = None,
        is_training: bool = True,
    ) -> "NeuralGaussianOutput":
        """Generate dynamic Gaussians for this camera view.

        Args:
            cam: Camera dict with keys: camera_center, uid, width, height
            visible_mask: Optional mask of which anchors are visible [N_anchors]
            is_training: If True, return training-specific fields (neural_opacity, selection_mask)

        Returns:
            NeuralGaussianOutput with generated Gaussian parameters
        """
        ...


# Future mixin stubs for other capabilities (add as needed):
# class LoDAwareMixin(ABC):
#     """Mixin for models with Level-of-Detail support."""
#     ...
#
# class SemanticsMixin(ABC):
#     """Mixin for models with semantic feature learning."""
#     ...


# ─────────────────────────────────────────────────────────────────────────────
# Base Trainable Model ABC
# ─────────────────────────────────────────────────────────────────────────────


class BaseTrainableModel(nn.Module, ABC):
    """Unified contract for all trainable Gaussian Splatting model variants.

    Subclasses implement the geometric rendering contract (means, scales, etc.)
    and training interface (optimizers, schedulers, densification).

    Capability mixins (e.g. NeuralRenderingMixin) are inherited by subclasses
    to declare support for model-specific features.
    """

    # ───────────────────────────────────────────────────────────────────────────
    # Abstract Properties (Required Geometric State)
    # ───────────────────────────────────────────────────────────────────────────

    @property
    @abstractmethod
    def means(self) -> torch.Tensor:
        """Gaussian center positions. Shape: [N, 3], activated."""
        ...

    @property
    @abstractmethod
    def scales(self) -> torch.Tensor:
        """Gaussian scales. Shape: [N, 3], activated (exp applied)."""
        ...

    @property
    @abstractmethod
    def quats(self) -> torch.Tensor:
        """Gaussian rotations as quaternions. Shape: [N, 4], normalized."""
        ...

    @property
    @abstractmethod
    def opacities(self) -> torch.Tensor:
        """Gaussian opacities. Shape: [N, 1], activated (sigmoid applied)."""
        ...

    @property
    @abstractmethod
    def sh(self) -> torch.Tensor:
        """Spherical harmonics coefficients (full basis). Shape: [N, D, 3]."""
        ...

    @property
    @abstractmethod
    def dc_rgb(self) -> torch.Tensor:
        """SH DC (zeroth order) color component. Shape: [N, 1, 3]."""
        ...

    @property
    @abstractmethod
    def sh_degree(self) -> int:
        """Maximum spherical harmonics degree (0, 1, 2, or 3)."""
        ...

    @property
    @abstractmethod
    def point_name(self) -> str:
        """Human-readable name for the primitive (e.g. 'Gaussians', 'anchors')."""
        ...

    @property
    @abstractmethod
    def count_label(self) -> str:
        """Short label for progress bar (e.g. 'GS', 'Anchors')."""
        ...

    # ───────────────────────────────────────────────────────────────────────────
    # Abstract Methods (Required Training Interface)
    # ───────────────────────────────────────────────────────────────────────────

    @abstractmethod
    def get_render_params(
        self, cam: dict, sh_cfg, is_training: bool = True, lod: Optional[int] = None
    ) -> "RenderParams":
        """Return rasterization-ready parameters for a single camera view.

        Args:
            cam: Camera dict
            sh_cfg: Config object with disable_sh_rendering flag
            is_training: If True, include training-specific fields
            lod: Optional Level-of-Detail level to render

        Returns:
            RenderParams dataclass with means, colors, opacities, scales, quats, sh_degree
        """
        ...

    @abstractmethod
    def get_params_dict(self) -> Dict[str, nn.Parameter]:
        """Return all learnable parameters as a dict.

        Used by the densification strategy to query/modify Gaussian state.
        Keys must match GSOptimizers field names (means, scales, quats, etc.).
        """
        ...

    @abstractmethod
    def get_optimizers_dict(self, optimizers: "GSOptimizers") -> Dict[str, torch.optim.Optimizer]:
        """Convert GSOptimizers to a dict for strategy interface.

        Args:
            optimizers: GSOptimizers dataclass

        Returns:
            Dict mapping parameter names to their optimizers
        """
        ...

    @abstractmethod
    def update_params_from_dict(self, params: Dict[str, nn.Parameter]) -> None:
        """Update internal parameters after densification/pruning.

        The strategy modifies params in-place (splitting, cloning, pruning).
        This method re-syncs the model's internal state from the updated dict.

        Args:
            params: Updated parameter dict from strategy
        """
        ...

    @abstractmethod
    def create_optimizers(
        self,
        lr_means: float = 0.00016,
        lr_scales: float = 0.005,
        lr_quats: float = 0.001,
        lr_opacities: float = 0.05,
        lr_sh: float = 0.0025,
        lr_semantics: Optional[float] = None,
        means_lr_multiplier: float = 5.0,
    ) -> "GSOptimizers":
        """Create Adam optimizers for all learnable parameters.

        Returns:
            GSOptimizers dataclass with one optimizer per parameter group
        """
        ...

    @abstractmethod
    def save_ply(self, path: str) -> None:
        """Save the model to PLY format.

        Args:
            path: Output file path
        """
        ...

    # ───────────────────────────────────────────────────────────────────────────
    # Concrete Methods (Defaults that subclasses can override)
    # ───────────────────────────────────────────────────────────────────────────

    def create_schedulers(self, optimizers: "GSOptimizers", iterations: int) -> "GS_LR_Schedulers":
        """Create learning rate schedulers for optimizers.

        Default: CosineAnnealingLR on means only. Scaffold-GS MLPs are auto-included
        via GS_LR_Schedulers.create_schedulers's extra-dict handling.

        Args:
            optimizers: GSOptimizers dataclass
            iterations: Total training iterations

        Returns:
            GS_LR_Schedulers dataclass
        """
        # Avoid circular import — import here
        from gs_types import GS_LR_Schedulers

        return GS_LR_Schedulers.create_schedulers(
            optimizers,
            enabled_lrs=GS_LR_Schedulers(means=True),
            step_size=iterations,
            gamma=0.1,
        )

    def compute_lods(
        self, num_levels: int = 1, factor: int = 4, optimizers: Optional["GSOptimizers"] = None
    ) -> None:
        """Compute Level-of-Detail structure if supported.

        Default: no-op (single level). Subclasses with LoD support override.

        Args:
            num_levels: Number of LoD levels
            factor: Reduction factor between levels
            optimizers: Optional optimizers to reorder
        """
        ...
