"""
Base strategy class for Gaussian densification and pruning.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, Optional
import torch
import torch.nn as nn


@dataclass
class ViewportInfo:
    """Information about the current viewport/camera for gradient normalization."""
    width: int
    height: int
    n_cameras: int = 1  # Number of cameras rendered (for batch processing)


class Strategy(ABC):
    """
    Abstract base class for Gaussian densification strategies.
    
    Strategies control when and how to:
    - Duplicate (clone) Gaussians
    - Split Gaussians
    - Prune Gaussians
    - Reset opacity
    
    This follows the gsplat library architecture for clean separation of concerns.
    """
    
    def __init__(
        self,
        verbose: bool = False,
    ):
        """
        Initialize strategy.
        
        Args:
            verbose: If True, print detailed logs during refinement
        """
        self.verbose = verbose
        self.state: Optional[Dict[str, torch.Tensor]] = None
        self._step = 0
        
    def initialize_state(
        self,
        params: Dict[str, nn.Parameter],
        scene_scale: float,
    ):
        """
        Initialize strategy state.
        
        This is called lazily on first use to ensure correct device placement.
        
        Args:
            params: Dictionary of model parameters (means, scales, quats, etc.)
            scene_scale: Scene extent for normalization
        """
        num_points = params["means"].shape[0]
        device = params["means"].device
        
        self.state = {
            # Accumulated norm of image plane gradients
            "grad2d": torch.zeros(num_points, device=device),
            # Number of times each Gaussian is visible
            "count": torch.zeros(num_points, device=device),
            # Maximum 2D screen-space radii (normalized by resolution)
            "radii": torch.zeros(num_points, device=device),
            # Scene scale for normalization
            "scene_scale": torch.tensor(scene_scale, device=device),
        }
        
    def check_sanity(
        self,
        params: Dict[str, nn.Parameter],
        optimizers: Dict[str, torch.optim.Optimizer],
    ):
        """
        Validate that parameters and optimizers are consistent.
        
        Args:
            params: Model parameters
            optimizers: Optimizers for each parameter group
        
        Raises:
            AssertionError: If parameters and optimizers don't match
        """
        for key, param in params.items():
            if key in optimizers:
                opt = optimizers[key]
                for group in opt.param_groups:
                    for p in group["params"]:
                        if p is param:
                            break
                    else:
                        continue
                    break
                else:
                    raise AssertionError(
                        f"Parameter '{key}' not found in optimizer '{key}'"
                    )
    
    def step_pre_backward(
        self,
        params: Dict[str, nn.Parameter],
        optimizers: Dict[str, torch.optim.Optimizer],
        **kwargs,
    ):
        """
        Callback executed before loss.backward().
        
        Args:
            params: Model parameters
            optimizers: Optimizers
            **kwargs: Additional arguments
        """
        pass
    
    @abstractmethod
    def step_post_backward(
        self,
        params: Dict[str, nn.Parameter],
        optimizers: Dict[str, torch.optim.Optimizer],
        scene_scale: float,
        iteration: int,
        viewport_info: ViewportInfo,
        **kwargs,
    ):
        """
        Callback executed after loss.backward().
        
        This is where the main refinement logic happens:
        - Accumulate gradients
        - Duplicate/split/prune Gaussians
        - Reset opacity
        
        Args:
            params: Model parameters
            optimizers: Optimizers
            scene_scale: Scene extent for normalization
            iteration: Current training iteration
            viewport_info: Camera/viewport information for gradient normalization
            **kwargs: Additional strategy-specific arguments
        """
        pass
    
    def reset_state(self):
        """Reset accumulated statistics."""
        if self.state is not None:
            self.state["grad2d"].zero_()
            self.state["count"].zero_()
            self.state["radii"].zero_()
