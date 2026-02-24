"""
Training orchestration utilities.

Manages training loop coordination including logging, densification,
and other per-iteration operations.
"""

import logging
import torch
from typing import Dict, Optional, Any, Tuple
from config import TrainingConfig


def log_iteration_metrics(
    tb_logger,
    render: torch.Tensor,
    gt_image: torch.Tensor,
    metrics_dict: Dict[str, float],
    current_iteration: int,
    config: TrainingConfig,
    logger: logging.Logger,
    model_means_count: int,
    max_radii: Optional[torch.Tensor] = None,
) -> None:
    """
    Log all metrics and images to TensorBoard at specified intervals.
    
    Handles both metrics logging (per log_interval) and image logging 
    (per tensorboard_image_interval).
    
    Args:
        tb_logger: TensorBoard logger instance
        render: [H, W, 3] rendered image for visualization
        gt_image: [H, W, 3] ground truth image for visualization
        metrics_dict: Dictionary of metrics including 'total_loss', 'l1_loss',
                     'ssim_loss', 'lpips_loss', 'psnr', regularization losses, etc.
        current_iteration: Current training iteration
        config: TrainingConfig with logging settings
        logger: Python logger for console output
        model_means_count: Number of Gaussians in model
        max_radii: Optional [N] tensor of max radii per Gaussian
    """
    if not config.enable_tensorboard:
        return
    
    # Log losses and metrics
    if current_iteration % config.log_interval == 0:
        # Prepare losses dict for TensorBoard
        losses_dict = {
            'total_loss': metrics_dict.get('total_loss', 0.0),
            'l1_loss': metrics_dict.get('l1_loss', 0.0),
            'ssim_loss': metrics_dict.get('ssim_loss', 0.0),
            'lpips_loss': metrics_dict.get('lpips_loss', 0.0),
        }
        
        # Add regularization losses if present
        if 'scale_reg_loss' in metrics_dict:
            losses_dict['scale_reg_loss'] = metrics_dict['scale_reg_loss']
        if 'opacity_reg_loss' in metrics_dict:
            losses_dict['opacity_reg_loss'] = metrics_dict['opacity_reg_loss']
        if 'depth_loss' in metrics_dict:
            losses_dict['depth_loss'] = metrics_dict['depth_loss']
        
        tb_logger.log_losses(**losses_dict, step=current_iteration)
        
        # Log quality metrics
        tb_logger.log_quality_metrics(
            psnr=metrics_dict.get('psnr', 0.0),
            ssim_loss=metrics_dict.get('ssim_loss', 0.0),
            lpips=metrics_dict.get('lpips_loss', 0.0),
            step=current_iteration
        )
        
        # Log model stats
        tb_logger.log_model_stats(
            num_gaussians=model_means_count,
            max_radii=max_radii,
            step=current_iteration
        )
        
        # Log system metrics (CPU, RAM, GPU usage, power, etc.)
        tb_logger.log_system_metrics(step=current_iteration)
    
    # Log images periodically (these are expensive to save)
    if current_iteration % config.tensorboard_image_interval == 0:
        # Get rendered depth if available in metrics
        rendered_depth = metrics_dict.get('rendered_depth', torch.zeros_like(render[:, :, 0]))
        gt_depth = metrics_dict.get('gt_depth', torch.zeros_like(render[:, :, 0]))
        alpha = metrics_dict.get('alpha', torch.ones_like(render[:, :, 0]))
        
        tb_logger.log_images(
            rendered=render.detach(),
            ground_truth=gt_image,
            alpha=alpha.detach() if isinstance(alpha, torch.Tensor) else alpha,
            rendered_depth=rendered_depth.detach() if isinstance(rendered_depth, torch.Tensor) else rendered_depth,
            gt_depth=gt_depth.detach() if isinstance(gt_depth, torch.Tensor) else gt_depth,
            step=current_iteration
        )


def handle_densification_step(
    model,
    strategy,
    strategy_state: Dict,
    current_iteration: int,
    meta: Dict,
    logger: logging.Logger,
    config = None,
) -> Tuple[int, int]:
    """
    Orchestrate densification and pruning step.
    
    Gets model parameters, calls strategy's step_post_backward, and updates
    model with potentially changed parameters (due to splits/duplicates/prunes).
    Handles error cases and logging.
    
    Args:
        model: GaussianModel instance
        strategy: Training strategy (e.g., DefaultStrategy)
        strategy_state: State dict for strategy
        current_iteration: Current training iteration
        meta: Metadata from rasterization (means2d, radii, etc.)
        logger: Python logger
        config: TrainingConfig with densification settings (optional)
    
    Returns:
        tuple: (gaussians_before, gaussians_after)
            Counts before and after densification for monitoring
    
    Raises:
        RuntimeError: If no Gaussians remain after densification
    """
    # Count before densification
    gaussians_before = len(model._means)
    
    # Get params and optimizers for strategy
    params = model.get_params_dict()
    optimizers = model._optimizers if hasattr(model, '_optimizers') else {}
    optimizers_dict = model.get_optimizers_dict(optimizers) if optimizers else {}
    
    # Call strategy's step_post_backward (handles densify/prune)
    strategy.step_post_backward(
        params=params,
        optimizers=optimizers_dict,
        state=strategy_state,
        step=current_iteration,
        info=meta,
        packed=True
    )
    
    # Update model with potentially changed parameters
    model.update_params_from_dict(params)
    
    # Count after densification
    gaussians_after = len(model._means)
    
    # Clear CUDA cache after densification/pruning
    torch.cuda.empty_cache()
    
    # Safety check: ensure model still has Gaussians
    if gaussians_after == 0:
        logger.error("[bold red]❌ Error:[/bold red] No Gaussians remaining in model! Training cannot continue.")
        raise RuntimeError(
            f"Model has no Gaussians at iteration {current_iteration}. Training cannot continue. "
            f"This indicates overly aggressive pruning or invalid strategy configuration."
        )
    
    return gaussians_before, gaussians_after


def validate_training_state(
    model,
    current_iteration: int,
    logger: logging.Logger,
    verbosity: int
) -> None:
    """
    Validate training state for consistency.
    
    Checks that model has Gaussians and other critical state is valid.
    Should be called periodically during training.
    
    Args:
        model: GaussianModel instance
        current_iteration: Current training iteration
        logger: Python logger
        verbosity: Verbosity level
    
    Raises:
        RuntimeError: If critical state is invalid
    """
    if len(model._means) == 0:
        logger.error(f"[bold red]❌ Error at iteration {current_iteration}:[/bold red] Model has no Gaussians!")
        raise RuntimeError(
            f"Model has no Gaussians at iteration {current_iteration}. "
            f"This indicates training has diverged or pruning was too aggressive."
        )
    
    if verbosity >= 3:
        logger.debug(
            f"[dim]Training state check passed: "
            f"{len(model._means)} Gaussians, iteration {current_iteration}[/dim]"
        )
