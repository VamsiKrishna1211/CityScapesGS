from dataclasses import dataclass, asdict
from typing import TypedDict, Any, Dict, Iterator, Optional, Tuple

import torch

# ─────────────────────────────────────────────────────────────────────────────
# Return Type Dataclasses (for type-safe multi-return values)
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class NeuralGaussianOutput:
    """Typed return from ScaffoldModel.generate_neural_gaussians().

    Fields neural_opacity and selection_mask are None in inference mode.
    language_features is None when language feature learning is disabled.
    """
    means: torch.Tensor                         # [M, 3] generated Gaussian positions
    colors: torch.Tensor                        # [M, 3] RGB colors
    opacities: torch.Tensor                     # [M, 1] opacity values
    scales: torch.Tensor                        # [M, 3] scales
    quats: torch.Tensor                         # [M, 4] quaternion rotations
    neural_opacity: Optional[torch.Tensor] = None    # [N*k, 1] full opacity before mask (training only)
    selection_mask: Optional[torch.Tensor] = None    # [N*k] bool mask (training only)
    language_features: Optional[torch.Tensor] = None # [M, lang_feat_dim] compact language features


@dataclass
class RenderParams:
    """Typed return from BaseTrainableModel.get_render_params().

    Replaces the untyped dict currently returned by both GaussianModel and ScaffoldModel.
    Scaffold-GS-specific fields (neural_opacity, selection_mask) are optional.
    language_features is populated when the model has enable_language_features=True.
    """
    means: torch.Tensor                         # [N, 3]
    colors: torch.Tensor                        # [N, D, 3] SH tensor OR [N, 3] RGB
    opacities: torch.Tensor                     # [N, 1]
    scales: torch.Tensor                        # [N, 3]
    quats: torch.Tensor                         # [N, 4]
    sh_degree: Optional[int]                    # SH degree or None for SH-disabled
    neural_opacity: Optional[torch.Tensor] = None    # Scaffold-GS training only
    selection_mask: Optional[torch.Tensor] = None    # Scaffold-GS training only
    language_features: Optional[torch.Tensor] = None # [N, lang_feat_dim] compact language features


# ─────────────────────────────────────────────────────────────────────────────
# Parameter and Optimizer Dataclasses
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class Parameters:
    means: torch.nn.Parameter | tuple[torch.nn.Parameter] | Any  # [N, 3]
    scales: torch.nn.Parameter | tuple[torch.nn.Parameter] | Any  # [N, 3]
    quats: torch.nn.Parameter | tuple[torch.nn.Parameter] | Any  # [N, 4]
    opacities: torch.nn.Parameter | tuple[torch.nn.Parameter] | Any  # [N]
    features_dc: torch.nn.Parameter | tuple[torch.nn.Parameter] | Any  # [N, D]
    features_rest: torch.nn.Parameter | tuple[torch.nn.Parameter] | Any  # [N, D]

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    # def as_dict(self) -> dict[str, torch.nn.Parameter]:
    #     return asdict(self)

@dataclass
class GSOptimizers:
    means: torch.optim.Optimizer
    scales: torch.optim.Optimizer
    quats: torch.optim.Optimizer
    opacities: torch.optim.Optimizer
    features_dc: torch.optim.Optimizer
    features_rest: torch.optim.Optimizer
    features_semantics: torch.optim.Optimizer | None = None
    extra: dict[str, torch.optim.Optimizer] = None

    def __post_init__(self):
        if self.extra is None:
            self.extra = {}

    def __getitem__(self, key):
        if hasattr(self, key):
            return getattr(self, key)
        return self.extra.get(key)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def all_optimizers(self) -> Iterator[Tuple[str, "torch.optim.Optimizer"]]:
        """Iterate over all non-None optimizers (flattened, including extra dict entries).

        Yields:
            (name, optimizer) tuples for all active optimizers
        """
        for name in ("means", "scales", "quats", "opacities", "features_dc", "features_rest"):
            opt = getattr(self, name)
            if opt is not None:
                yield name, opt
        if self.features_semantics is not None:
            yield "features_semantics", self.features_semantics
        for name, opt in self.extra.items():
            if opt is not None:
                yield name, opt

    # def as_dict(self) -> dict[str, torch.optim.Optimizer]:
    #     return asdict(self)


@dataclass
class GS_LR_Schedulers:
    means: torch.optim.lr_scheduler._LRScheduler | bool = True
    scales: torch.optim.lr_scheduler._LRScheduler | bool = False
    quats: torch.optim.lr_scheduler._LRScheduler | bool = False
    opacities: torch.optim.lr_scheduler._LRScheduler | bool = False
    features_dc: torch.optim.lr_scheduler._LRScheduler | bool = False
    features_rest: torch.optim.lr_scheduler._LRScheduler | bool = False
    features_semantics: torch.optim.lr_scheduler._LRScheduler | bool = False
    extra: dict[str, torch.optim.lr_scheduler._LRScheduler] = None

    def __post_init__(self):
        if self.extra is None:
            self.extra = {}

    def __getitem__(self, key):
        if hasattr(self, key):
            return getattr(self, key)
        return self.extra.get(key)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def all_schedulers(self) -> Iterator[Tuple[str, "torch.optim.lr_scheduler._LRScheduler"]]:
        """Iterate over all active schedulers (flattened, including extra dict entries).

        Yields:
            (name, scheduler) tuples for all active schedulers
        """
        for name in ("means", "scales", "quats", "opacities", "features_dc", "features_rest", "features_semantics"):
            sched = getattr(self, name)
            # Check if it's a scheduler (not None, not a bool, and has a step method)
            if sched is not None and not isinstance(sched, bool) and hasattr(sched, "step"):
                yield name, sched
        for name, sched in self.extra.items():
            if sched is not None:
                yield name, sched


    # schedulers = type('GSSchedulers', (), {})()
    # for name, optimizer in optimizers.__dict__.items():
    #     max_lr = optimizer.param_groups[0]['lr']
    #     scheduler = OneCycleLR(
    #         optimizer,
    #         max_lr=max_lr,
    #         total_steps=iterations,
    #         pct_start=0.3,  # 30% of training for warmup
    #         anneal_strategy='cos',  # Cosine annealing
    #         div_factor=25.0,  # initial_lr = max_lr / 25
    #         final_div_factor=1e4,  # final_lr = max_lr / 10000
    #     )
    #     setattr(schedulers, name, scheduler)

    @classmethod
    def create_schedulers(cls, optimizers: GSOptimizers, enabled_lrs, step_size: int, gamma: float):
        schedulers_dict: dict[str, Any] = {}
        extra_schedulers = {}

        # Handle main fields
        for name in ['means', 'scales', 'quats', 'opacities', 'features_dc', 'features_rest', 'features_semantics']:
            enabled = getattr(enabled_lrs, name, False)
            if not enabled:
                schedulers_dict[name] = None
                continue
            optimizer = getattr(optimizers, name)
            if optimizer is None:
                continue
            initial_lr = optimizer.param_groups[0]['lr']
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=step_size,
                eta_min=initial_lr * 1e-4,
            )
            schedulers_dict[name] = scheduler

        # Handle extra fields (MLPs etc)
        if optimizers.extra:
            for name, optimizer in optimizers.extra.items():
                initial_lr = optimizer.param_groups[0]['lr']
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=step_size,
                    eta_min=initial_lr * 1e-4,
                )
                extra_schedulers[name] = scheduler
            
        schedulers_dict['extra'] = extra_schedulers
        return cls(**schedulers_dict)
