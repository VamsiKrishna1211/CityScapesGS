"""
Helper operations for Gaussian manipulation with optimizer state synchronization.

These functions handle:
- Duplicating Gaussians
- Splitting Gaussians with covariance-based sampling
- Removing Gaussians
- Resetting opacity
- Updating optimizer states

All operations ensure that both parameters and optimizer states remain synchronized.
"""

from typing import Dict, Tuple, Optional
import torch
import torch.nn as nn


def map_param_to_optimizer(
    param: nn.Parameter,
    optimizers: Dict[str, torch.optim.Optimizer],
) -> Tuple[Optional[torch.optim.Optimizer], Optional[int], Optional[int]]:
    """
    Find which optimizer and param_group contains a parameter.
    
    Args:
        param: The parameter to find
        optimizers: Dictionary of optimizers
        
    Returns:
        Tuple of (optimizer, group_idx, param_idx) or (None, None, None)
    """
    for opt in optimizers.values():
        for group_idx, group in enumerate(opt.param_groups):
            for param_idx, p in enumerate(group["params"]):
                if p is param:
                    return opt, group_idx, param_idx
    return None, None, None


def _update_param_with_optimizer(
    param: nn.Parameter,
    optimizer: torch.optim.Optimizer,
    group_idx: int,
    param_idx: int,
    new_param: nn.Parameter,
):
    """
    Replace a parameter in optimizer and transfer/update state.
    
    Args:
        param: Old parameter
        optimizer: Optimizer containing the parameter
        group_idx: Index of param_group
        param_idx: Index of parameter in param_group
        new_param: New parameter to replace with
    """
    # Replace in param_group
    optimizer.param_groups[group_idx]["params"][param_idx] = new_param
    
    # Transfer optimizer state
    if param in optimizer.state:
        old_state = optimizer.state.pop(param)
        new_state = {}
        
        # Transfer state tensors (extend with zeros if new param is larger)
        for key, value in old_state.items():
            if isinstance(value, torch.Tensor):
                # Check if tensor has same shape structure as parameter
                # Optimizer state tensors (exp_avg, exp_avg_sq) should match param shape
                if value.dim() > 0 and value.shape == param.shape:
                    if value.shape[0] < new_param.shape[0]:
                        # Parameter grew (duplicate/split) - pad with zeros
                        padding_size = new_param.shape[0] - value.shape[0]
                        padding_shape = (padding_size,) + value.shape[1:]
                        padding = torch.zeros(padding_shape, dtype=value.dtype, device=value.device)
                        new_state[key] = torch.cat([value, padding], dim=0)
                    elif value.shape[0] > new_param.shape[0]:
                        # Parameter shrunk (prune) - truncate to match keep_mask
                        # This case is handled by passing the keep_mask externally
                        new_state[key] = value
                    else:
                        new_state[key] = value
                else:
                    # Scalar tensor or different shape - keep as is
                    new_state[key] = value
            else:
                # Non-tensor state (e.g., step counter)
                new_state[key] = value
        
        optimizer.state[new_param] = new_state


def normalized_quat_to_rotmat(quat: torch.Tensor) -> torch.Tensor:
    """
    Convert normalized quaternions to rotation matrices.
    
    Args:
        quat: Quaternions [N, 4] in (w, x, y, z) format, assumed normalized
        
    Returns:
        Rotation matrices [N, 3, 3]
    """
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    
    # First row
    R00 = 1 - 2 * (y * y + z * z)
    R01 = 2 * (x * y - w * z)
    R02 = 2 * (x * z + w * y)
    
    # Second row
    R10 = 2 * (x * y + w * z)
    R11 = 1 - 2 * (x * x + z * z)
    R12 = 2 * (y * z - w * x)
    
    # Third row
    R20 = 2 * (x * z - w * y)
    R21 = 2 * (y * z + w * x)
    R22 = 1 - 2 * (x * x + y * y)
    
    # Stack into matrix [N, 3, 3]
    R = torch.stack([
        torch.stack([R00, R01, R02], dim=-1),
        torch.stack([R10, R11, R12], dim=-1),
        torch.stack([R20, R21, R22], dim=-1)
    ], dim=-2)
    
    return R


def duplicate(
    params: Dict[str, nn.Parameter],
    optimizers: Dict[str, torch.optim.Optimizer],
    state: Dict[str, torch.Tensor],
    mask: torch.Tensor,
) -> Dict[str, nn.Parameter]:
    """
    Duplicate Gaussians selected by mask.
    
    Args:
        params: Model parameters
        optimizers: Optimizers for each parameter group
        state: Strategy state (grad2d, count, radii)
        mask: Boolean mask selecting which Gaussians to duplicate
        
    Returns:
        Updated params dictionary
    """
    device = params["means"].device
    n_duplicated = mask.sum().item()
    
    if n_duplicated == 0:
        return params
    
    # Duplicate all parameters
    new_params = {}
    for key, param in params.items():
        new_param = nn.Parameter(
            torch.cat([param, param[mask]], dim=0)
        )
        new_params[key] = new_param
        
        # Update optimizer
        opt, group_idx, param_idx = map_param_to_optimizer(param, optimizers)
        if opt is not None:
            _update_param_with_optimizer(param, opt, group_idx, param_idx, new_param)
    
    # Update strategy state
    state["grad2d"] = torch.cat([state["grad2d"], torch.zeros(n_duplicated, device=device)])
    state["count"] = torch.cat([state["count"], torch.zeros(n_duplicated, device=device)])
    state["radii"] = torch.cat([state["radii"], torch.zeros(n_duplicated, device=device)])
    
    return new_params


def split(
    params: Dict[str, nn.Parameter],
    optimizers: Dict[str, torch.optim.Optimizer],
    state: Dict[str, torch.Tensor],
    mask: torch.Tensor,
    revised_opacity: bool = False,
) -> Dict[str, nn.Parameter]:
    """
    Split Gaussians selected by mask into 2 new Gaussians each.
    
    Uses covariance-based sampling: samples are drawn from the Gaussian's
    3D covariance (rotation + scale) to create two new Gaussians that
    together better represent the original.
    
    Args:
        params: Model parameters
        optimizers: Optimizers for each parameter group
        state: Strategy state
        mask: Boolean mask selecting which Gaussians to split
        revised_opacity: If True, use revised opacity formula
        
    Returns:
        Updated params dictionary
    """
    device = params["means"].device
    n_split = mask.sum().item()
    
    if n_split == 0:
        return params
    
    # Get current values from parameters that need special handling
    selected_means = params["means"][mask]
    selected_scales_log = params["scales"][mask]
    selected_quats = params["quats"][mask]
    selected_opacities_logit = params["opacities"][mask]
    
    # Compute scales from log-space
    selected_scales = torch.exp(selected_scales_log)
    
    # Get rotation matrices
    # Normalize quaternions first
    selected_quats_normalized = torch.nn.functional.normalize(selected_quats, dim=-1)
    rotmats = normalized_quat_to_rotmat(selected_quats_normalized)  # [N, 3, 3]
    
    # Sample 2 new positions per Gaussian using covariance
    # samples = R @ diag(scale) @ randn(2, N, 3)
    samples = torch.einsum(
        "nij,nj,bnj->bni",
        rotmats,
        selected_scales,
        torch.randn(2, n_split, 3, device=device)
    )  # [2, N, 3]
    
    # Create 2 new Gaussians per split
    new_means = torch.cat([
        selected_means + samples[0],
        selected_means - samples[1]
    ], dim=0)
    
    # Scale down by factor of 1.6 (in log space: subtract log(1.6))
    new_scales = torch.cat([
        selected_scales_log - torch.log(torch.tensor(1.6, device=device)),
        selected_scales_log - torch.log(torch.tensor(1.6, device=device))
    ], dim=0)
    
    # Handle opacity
    if revised_opacity:
        # Revised formula: prevents opacity from growing too much
        # opacity_new = 1 - sqrt(1 - sigmoid(opacity_old))
        selected_opacities = torch.sigmoid(selected_opacities_logit)
        new_opacities_val = 1.0 - torch.sqrt(torch.clamp(1.0 - selected_opacities, min=1e-6))
        new_opacities = torch.logit(torch.clamp(new_opacities_val, min=1e-6, max=1-1e-6))
        new_opacities = torch.cat([new_opacities, new_opacities], dim=0)
    else:
        # Standard: just copy opacity
        new_opacities = torch.cat([
            selected_opacities_logit,
            selected_opacities_logit
        ], dim=0)
    
    # Duplicate other attributes
    new_quats = torch.cat([params["quats"][mask], params["quats"][mask]], dim=0)
    new_features_dc = torch.cat([params["features_dc"][mask], params["features_dc"][mask]], dim=0)
    new_features_rest = torch.cat([params["features_rest"][mask], params["features_rest"][mask]], dim=0)
    
    # Remove original Gaussians and add new ones
    keep_mask = ~mask
    new_params = {}
    
    # Construct new parameters
    param_updates = {
        "means": new_means,
        "scales": new_scales,
        "quats": new_quats,
        "opacities": new_opacities,
        "features_dc": new_features_dc,
        "features_rest": new_features_rest,
    }
    
    for key, param in params.items():
        # Keep non-split Gaussians and append new ones
        new_param = nn.Parameter(
            torch.cat([param[keep_mask], param_updates[key]], dim=0)
        )
        new_params[key] = new_param
        
        # Update optimizer state
        opt, group_idx, param_idx = map_param_to_optimizer(param, optimizers)
        if opt is not None:
            # For optimizer state, we need to:
            # 1. Keep state for non-split Gaussians
            # 2. Zero-initialize state for new Gaussians (2 per split)
            if param in opt.state:
                old_state = opt.state.pop(param)
                new_state = {}
                
                for state_key, value in old_state.items():
                    if isinstance(value, torch.Tensor):
                        # Check if tensor has same shape structure as parameter
                        if value.dim() > 0 and value.shape == param.shape:
                            # Keep non-split state and zero-pad for new Gaussians
                            kept_state = value[keep_mask]
                            new_state_size = n_split * 2
                            padding_shape = (new_state_size,) + value.shape[1:]
                            padding = torch.zeros(padding_shape, dtype=value.dtype, device=value.device)
                            new_state[state_key] = torch.cat([kept_state, padding], dim=0)
                        else:
                            # Scalar or different shape - keep as is
                            new_state[state_key] = value
                    else:
                        new_state[state_key] = value
                
                opt.state[new_param] = new_state
            
            # Update param_groups
            opt.param_groups[group_idx]["params"][param_idx] = new_param
    
    # Update strategy state
    state["grad2d"] = torch.cat([
        state["grad2d"][keep_mask],
        torch.zeros(n_split * 2, device=device)
    ])
    state["count"] = torch.cat([
        state["count"][keep_mask],
        torch.zeros(n_split * 2, device=device)
    ])
    state["radii"] = torch.cat([
        state["radii"][keep_mask],
        torch.zeros(n_split * 2, device=device)
    ])
    
    return new_params


def remove(
    params: Dict[str, nn.Parameter],
    optimizers: Dict[str, torch.optim.Optimizer],
    state: Dict[str, torch.Tensor],
    mask: torch.Tensor,
) -> Dict[str, nn.Parameter]:
    """
    Remove Gaussians selected by mask.
    
    Args:
        params: Model parameters
        optimizers: Optimizers for each parameter group
        state: Strategy state
        mask: Boolean mask selecting which Gaussians to REMOVE
        
    Returns:
        Updated params dictionary
    """
    keep_mask = ~mask
    n_removed = mask.sum().item()
    
    if n_removed == 0:
        return params
    
    # Remove from all parameters
    new_params = {}
    for key, param in params.items():
        new_param = nn.Parameter(param[keep_mask])
        new_params[key] = new_param
        
        # Update optimizer state
        opt, group_idx, param_idx = map_param_to_optimizer(param, optimizers)
        if opt is not None:
            if param in opt.state:
                old_state = opt.state.pop(param)
                new_state = {}
                
                for state_key, value in old_state.items():
                    if isinstance(value, torch.Tensor):
                        # Check if tensor has same shape structure as parameter
                        if value.dim() > 0 and value.shape == param.shape:
                            new_state[state_key] = value[keep_mask]
                        else:
                            # Scalar or different shape - keep as is
                            new_state[state_key] = value
                    else:
                        new_state[state_key] = value
                
                opt.state[new_param] = new_state
            
            # Update param_groups
            opt.param_groups[group_idx]["params"][param_idx] = new_param
    
    # Update strategy state
    state["grad2d"] = state["grad2d"][keep_mask]
    state["count"] = state["count"][keep_mask]
    state["radii"] = state["radii"][keep_mask]
    
    return new_params


def reset_opacity(
    params: Dict[str, nn.Parameter],
    optimizers: Dict[str, torch.optim.Optimizer],
    value: float,
    revised: bool = False,
) -> Dict[str, nn.Parameter]:
    """
    Reset opacity of all Gaussians.
    
    Args:
        params: Model parameters
        optimizers: Optimizers
        value: Opacity threshold (e.g., 0.01)
        revised: If True, use revised reset (clamp to 2*value)
        
    Returns:
        Updated params dictionary
    """
    opacities_logit = params["opacities"]
    
    if revised:
        # Clamp opacity to max of 2*value (e.g., 0.01 -> 0.02)
        max_logit = torch.logit(torch.tensor(value * 2.0, device=opacities_logit.device))
        new_opacities = nn.Parameter(
            torch.clamp(opacities_logit, max=max_logit)
        )
    else:
        # Standard reset: set all to value
        new_opacities = nn.Parameter(
            torch.full_like(opacities_logit, torch.logit(torch.tensor(value, device=opacities_logit.device)))
        )
    
    # Update parameter
    params["opacities"] = new_opacities
    
    # Update optimizer and zero out its state for opacities
    opt, group_idx, param_idx = map_param_to_optimizer(opacities_logit, optimizers)
    if opt is not None:
        # Zero out optimizer state (exp_avg, exp_avg_sq)
        if opacities_logit in opt.state:
            old_state = opt.state.pop(opacities_logit)
            new_state = {}
            
            for key, value in old_state.items():
                if isinstance(value, torch.Tensor):
                    new_state[key] = torch.zeros_like(value)
                else:
                    new_state[key] = value
            
            opt.state[new_opacities] = new_state
        
        # Update param_groups
        opt.param_groups[group_idx]["params"][param_idx] = new_opacities
    
    return params
