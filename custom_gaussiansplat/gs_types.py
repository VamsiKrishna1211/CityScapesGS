from dataclasses import dataclass, asdict
from typing import TypedDict, Any

import torch

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

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    # def as_dict(self) -> dict[str, torch.optim.Optimizer]:
    #     return asdict(self)

@dataclass
class GS_LR_Schedulers:
    means: torch.optim.lr_scheduler._LRScheduler
    scales: torch.optim.lr_scheduler._LRScheduler
    quats: torch.optim.lr_scheduler._LRScheduler
    opacities: torch.optim.lr_scheduler._LRScheduler
    features_dc: torch.optim.lr_scheduler._LRScheduler
    features_rest: torch.optim.lr_scheduler._LRScheduler

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    
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
    def create_schedulers(cls, optimizers: GSOptimizers, step_size: int, gamma: float):
        schedulers_dict = {}
        for name, optimizer in optimizers.__dict__.items():
            initial_lr = optimizer.param_groups[0]['lr']
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=step_size,  # Full cosine period = total training iterations
                eta_min=initial_lr * 1e-4,  # Minimum LR = 0.01% of initial LR
            )
            schedulers_dict[name] = scheduler
            
        return cls(**schedulers_dict)
