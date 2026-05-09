"""Base class and capability mixins for all trainable Gaussian Splatting models."""

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Dict, Iterator, Optional, Tuple

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from gs_types import (
        GS_LR_Schedulers,
        GSOptimizers,
        NeuralGaussianOutput,
        RenderParams,
    )

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


class SemanticsMixin(ABC):
    """Mixin for models that support semantic / language feature learning.

    Callers use isinstance(model, SemanticsMixin) for type-safe detection.

    The mixin is deliberately model-agnostic: it declares *what* a semantic
    model must expose (render output, trainable params, feature dim) without
    encoding any assumption about the supervision signal (CLIP, DINO, etc.).
    Model-specific setup (e.g. PCA codebook init for CLIP) goes in
    setup_semantic_training(), which each subclass overrides as needed.
    """

    # ── Required properties ───────────────────────────────────────────────────

    @property
    @abstractmethod
    def semantics_dim(self) -> int:
        """Dimension of the rendered per-Gaussian/anchor semantic feature vector."""
        ...

    @property
    def provider_semantics_dim(self) -> int:
        """Channel count expected from SemanticTargetProvider.get_target().

        Defaults to semantics_dim. Override when the raw provider output
        dimension differs from the rendered one — e.g. SemanticScaffoldModel
        renders 32-dim features but the provider returns 512-dim raw CLIP
        features that are later PCA-compressed in prepare_target().
        """
        return self.semantics_dim

    # ── Required rendering interface ──────────────────────────────────────────

    @abstractmethod
    def render_semantics(
        self,
        cam: dict,
        device: torch.device,
        detach_geometry: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Render the semantic feature map for the given camera.

        Args:
            cam: Camera dict.
            device: Target device.
            detach_geometry: When True (default), geometry tensors are detached
                so only semantic parameters receive gradients.  Set to False
                when joint geometry + semantic training is desired.

        Returns:
            rgb_pred:      [H, W, 3]              RGB rendering.
            semantic_pred: [H, W, semantics_dim]  Rendered semantic feature map.
        """
        ...

    # ── Required optimization interface ──────────────────────────────────────

    @abstractmethod
    def get_semantic_trainable_params(self) -> "list[nn.Parameter]":
        """Return ONLY the parameters that receive gradients during semantic training."""
        ...

    # ── Optional hooks (concrete defaults) ───────────────────────────────────

    def setup_semantic_training(self, semantics_cfg: object, device: torch.device) -> None:
        """Pre-training setup hook called once before the semantic training loop.

        Default: no-op. SemanticScaffoldModel overrides to compute PCA and
        seed the language codebook from CLIP feature principal components.
        """

    def prepare_target(self, raw_target: torch.Tensor, device: torch.device) -> torch.Tensor:
        """Per-step target transformation applied before loss computation.

        Default: identity (raw provider output used as-is). SemanticScaffoldModel
        overrides to project [H, W, clip_dim] → [H, W, lang_feat_dim] via PCA.

        Args:
            raw_target: [H', W', D] raw semantic target from the provider.
            device:     Target device.

        Returns:
            Transformed target tensor, potentially smaller in the channel dim.
        """
        return raw_target

    @property
    def geometry_param_group_names(self) -> frozenset:
        """Names of param groups (from get_finetune_param_groups) that modify geometry.

        The SemanticTrainer uses this to decide whether to activate the full
        Rasterizer + LossComputer render path.  Override in subclasses to declare
        which named groups affect geometry — the base default is the empty set
        (semantics-only fine-tuning).
        """
        return frozenset()

    def get_finetune_param_groups(self) -> "Dict[str, list]":
        """Return named param groups for selective fine-tuning.

        Override in subclasses to expose geometry, MLP heads, appearance, etc.
        SemanticTrainer calls this to resolve ``--finetune-params`` group names to
        actual parameter lists.
        """
        return {}

    def freeze_params_except(self, trainable: "list") -> "Dict[str, bool]":
        """Freeze all parameters except the given list. Returns old requires_grad state."""
        trainable_ids = {id(p) for p in trainable}
        state: "Dict[str, bool]" = {}
        for name, param in self.named_parameters():
            state[name] = bool(param.requires_grad)
            param.requires_grad_(id(param) in trainable_ids)
        return state

    def freeze_non_semantic_params(self) -> "Dict[str, bool]":
        """Freeze all parameters except semantic ones. Returns grad state for later restoration."""
        trainable_ids = {id(p) for p in self.get_semantic_trainable_params()}
        state: "Dict[str, bool]" = {}
        for name, param in self.named_parameters():
            state[name] = bool(param.requires_grad)
            param.requires_grad_(id(param) in trainable_ids)
        return state

    def restore_grad_state(self, state: "Dict[str, bool]") -> None:
        """Restore requires_grad flags previously saved by freeze_params_except."""
        for name, param in self.named_parameters():
            if name in state:
                param.requires_grad_(state[name])

    @property
    def anchor_lang_feat(self) -> "Optional[torch.Tensor]":
        """Per-anchor language feature matrix [N, D]. None if not implemented."""
        return None


# Future mixin stubs for other capabilities (add as needed):
# class LoDAwareMixin(ABC):
#     """Mixin for models with Level-of-Detail support."""
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

    def __init__(self) -> None:
        super().__init__()
        # OPTIMIZATION: Cache for params_dict to avoid repeated dict construction
        self._params_dict_cache: Optional[Dict[str, nn.Parameter]] = None
        self._params_dict_version: int = 0

    def _invalidate_params_cache(self) -> None:
        """Invalidate the params_dict cache when model structure changes."""
        self._params_dict_version += 1
        self._params_dict_cache = None

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
    def quats(self) -> torch.Tensor | None:
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

    def get_params_dict_cached(self) -> Dict[str, nn.Parameter]:
        """Return cached params_dict if available, otherwise compute and cache.

        Subclasses should call this instead of get_params_dict() in training loops
        to avoid repeated dict construction. The cache is automatically invalidated
        when the model structure changes (via _invalidate_params_cache()).
        """
        if self._params_dict_cache is None:
            self._params_dict_cache = self.get_params_dict()
        return self._params_dict_cache

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
        lr_scales: float = 0.007,
        lr_quats: float = 0.002,
        lr_opacities: float = 0.02,
        lr_sh: float = 0.0075,
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

        Default: CosineAnnealingLR on means only.

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

    def set_appearance(self, num_cameras: int) -> None:
        """Optional hook for models with per-camera appearance embeddings.

        Default: no-op for models that do not use appearance embeddings.
        """
        _ = num_cameras

    def iter_extra_optimizers(self) -> Iterator[Tuple[str, torch.optim.Optimizer]]:
        """Iterate model-owned non-geometric optimizers (MLPs, embeddings, etc.)."""
        return iter(())

    def iter_extra_schedulers(self) -> Iterator[Tuple[str, torch.optim.lr_scheduler.LRScheduler]]:
        """Iterate model-owned non-geometric schedulers."""
        return iter(())

    def get_extra_optimizer_states(self) -> Dict[str, dict]:
        """Return checkpoint state for model-owned non-geometric optimizers."""
        return {}

    def load_extra_optimizer_states(self, checkpoint: Dict[str, object]) -> None:
        """Restore model-owned non-geometric optimizer states from checkpoint.

        Default: no-op for models that only use geometric optimizers.
        """
        _ = checkpoint

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
