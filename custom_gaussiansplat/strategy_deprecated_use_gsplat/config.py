"""
Training configuration management.

Consolidates all training hyperparameters and settings into a single
configuration class with validation and defaults.
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any
import argparse


@dataclass
class TrainingConfig:
    """
    Complete training configuration for Gaussian Splatting.
    
    Organizes hyperparameters into logical sections:
    - Iteration control
    - Learning rates
    - Loss weights
    - Regularization
    - Logging and output
    - Depth supervision
    """
    
    # ==================== Iteration Control ====================
    iterations: int = 7000
    densify_from_iter: int = 500
    densify_until_iter: int = 15000
    densify_interval: int = 100
    opacity_reset_interval: int = 3000
    opacity_reset_value: float = 0.01
    grad_threshold: float = 0.02
    max_screen_size: int = 20
    
    # ==================== Learning Rates ====================
    lr_means: float = 0.00016
    lr_means_final: float = 0.00001
    lr_means_final_iter: int = 15000
    lr_scale_means: float = 0.001
    lr_quats: float = 0.001
    lr_scales: float = 0.005
    lr_opacities: float = 0.05
    lr_sh: float = 0.0025
    
    # ==================== Loss Weights ====================
    l1_weight: float = 0.4
    ssim_weight: float = 0.6
    enable_depth_loss: bool = False
    depth_loss_weight: float = 0.1
    depth_loss_start_iter: int = 500
    
    # ==================== Regularization ====================
    enable_scale_reg: bool = False
    scale_lambda: float = 0.01
    scene_extent: float = 1.0
    enable_opacity_reg: bool = False
    opacity_lambda: float = 0.0001
    
    # ==================== Logging and Output ====================
    output_dir: str = './output'
    enable_tensorboard: bool = False
    tensorboard_image_interval: int = 500
    log_interval: int = 100
    verbosity: int = 1  # 0=QUIET, 1=NORMAL, 2=VERBOSE, 3=DEBUG
    checkpoint_interval: int = 1000
    save_final_checkpoint: bool = True
    
    # ==================== Depth Supervision ====================
    depth_prior_path: Optional[str] = None
    depth_alignment_method: str = 'scale-shift'  # 'scale-shift' or 'direct'
    depth_min_valid_pixels: int = 100
    depth_validation_enabled: bool = True
    
    # ==================== SH Rendering ====================
    use_sh_rendering: bool = True
    max_sh_degree: int = 3
    
    # ==================== Other ====================
    device: str = 'cuda'
    dtype: str = 'float32'
    seed: Optional[int] = None
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        self.validate()
    
    def validate(self) -> None:
        """
        Validate configuration parameters for logical consistency.
        
        Raises:
            ValueError: If any configuration is invalid
        """
        # Iteration sanity checks
        if self.iterations < 100:
            raise ValueError(f"iterations must be >= 100, got {self.iterations}")
        
        if self.densify_from_iter >= self.densify_until_iter:
            raise ValueError(
                f"densify_from_iter ({self.densify_from_iter}) must be < "
                f"densify_until_iter ({self.densify_until_iter})"
            )
        
        # Learning rate sanity checks
        if self.lr_means <= 0:
            raise ValueError(f"lr_means must be positive, got {self.lr_means}")
        
        if self.lr_means_final <= 0:
            raise ValueError(f"lr_means_final must be positive, got {self.lr_means_final}")
        
        if self.lr_means_final_iter > self.iterations:
            raise ValueError(
                f"lr_means_final_iter ({self.lr_means_final_iter}) cannot exceed "
                f"iterations ({self.iterations})"
            )
        
        # Loss weight sanity checks
        if self.l1_weight < 0 or self.ssim_weight < 0:
            raise ValueError("Loss weights must be non-negative")
        
        if abs(self.l1_weight + self.ssim_weight) < 1e-6:
            raise ValueError("At least one of l1_weight or ssim_weight must be non-zero")
        
        if self.enable_depth_loss and self.depth_loss_weight <= 0:
            raise ValueError("depth_loss_weight must be positive when depth loss is enabled")
        
        # Depth loss timing
        if self.enable_depth_loss and self.depth_loss_start_iter >= self.iterations:
            raise ValueError(
                f"depth_loss_start_iter ({self.depth_loss_start_iter}) must be < "
                f"iterations ({self.iterations})"
            )
        
        # Regularization sanity checks
        if self.enable_scale_reg and self.scale_lambda <= 0:
            raise ValueError("scale_lambda must be positive when scale regularization is enabled")
        
        if self.enable_opacity_reg and self.opacity_lambda <= 0:
            raise ValueError("opacity_lambda must be positive when opacity regularization is enabled")
        
        if self.scene_extent <= 0:
            raise ValueError(f"scene_extent must be positive, got {self.scene_extent}")
        
        # Output directory check
        if not self.output_dir:
            raise ValueError("output_dir cannot be empty")
        
        # Depth validation parameters
        if self.depth_min_valid_pixels < 10:
            raise ValueError(
                f"depth_min_valid_pixels must be >= 10 (recommended >= 100), "
                f"got {self.depth_min_valid_pixels}"
            )
        
        # SH degree sanity check
        if not (0 <= self.max_sh_degree <= 3):
            raise ValueError(f"max_sh_degree must be in [0, 3], got {self.max_sh_degree}")
    
    @classmethod
    def from_args(cls, args: argparse.Namespace) -> 'TrainingConfig':
        """
        Create config from argparse arguments.
        
        Args:
            args: Parsed command-line arguments
        
        Returns:
            TrainingConfig instance
        """
        # Extract all fields that exist in both args and TrainingConfig
        config_dict = {}
        for field_name in asdict(cls()).keys():
            if hasattr(args, field_name):
                config_dict[field_name] = getattr(args, field_name)
        
        return cls(**config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)
    
    def __str__(self) -> str:
        """Pretty print configuration."""
        lines = ["=" * 60]
        lines.append("TRAINING CONFIGURATION")
        lines.append("=" * 60)
        
        for key, value in asdict(self).items():
            lines.append(f"  {key:<35} = {value}")
        
        lines.append("=" * 60)
        return "\n".join(lines)
