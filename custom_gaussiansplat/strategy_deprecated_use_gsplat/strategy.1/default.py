"""
Default densification strategy following the original 3D Gaussian Splatting paper.

Implements:
- Adaptive Density Control (ADC): Clone small Gaussians, split large ones
- Gradient-based densification with screen-normalized thresholds
- Periodic opacity reset
- Multi-criteria pruning (opacity, 3D scale, 2D scale)
"""

from typing import Dict, Optional
import torch
import torch.nn as nn

from .base import Strategy, ViewportInfo
from . import ops


class DefaultStrategy(Strategy):
    """
    Default densification strategy from 3D Gaussian Splatting paper.
    
    This strategy:
    1. Accumulates screen-space gradients (normalized by viewport size)
    2. Clones small Gaussians with high gradients
    3. Splits large Gaussians with high gradients (using covariance sampling)
    4. Prunes low-opacity or oversized Gaussians
    5. Periodically resets opacity to prevent saturation
    
    Parameters follow the original paper's recommendations.
    """
    
    def __init__(
        self,
        prune_opa: float = 0.005,
        grow_grad2d: float = 0.0002,
        grow_scale3d: float = 0.01,
        grow_scale2d: float = 0.05,
        prune_scale3d: float = 0.1,
        prune_scale2d: float = 0.15,
        refine_start_iter: int = 500,
        refine_stop_iter: int = 15_000,
        reset_every: int = 3000,
        refine_every: int = 100,
        pause_refine_after_reset: int = 0,
        absgrad: bool = False,
        revised_opacity: bool = False,
        verbose: bool = False,
    ):
        """
        Initialize default strategy.
        
        Args:
            prune_opa: Opacity threshold for pruning (default: 0.005)
            grow_grad2d: Screen-space gradient threshold for densification (default: 0.0002)
            grow_scale3d: 3D scale threshold, as fraction of scene_scale (default: 0.01)
            grow_scale2d: 2D screen-space scale threshold for splitting (default: 0.05)
            prune_scale3d: 3D scale pruning threshold, as fraction of scene_scale (default: 0.1)
            prune_scale2d: 2D screen-space scale pruning threshold (default: 0.15)
            refine_start_iter: Start densification at this iteration (default: 500)
            refine_stop_iter: Stop densification at this iteration (default: 15,000)
            reset_every: Reset opacity every N iterations (default: 3000)
            refine_every: Perform densification every N iterations (default: 100)
            pause_refine_after_reset: Pause refinement for N iterations after reset (default: 0)
            absgrad: Use absolute gradients instead of average (AbsGS mode) (default: False)
            revised_opacity: Use revised opacity formula for splits (default: False)
            verbose: Print detailed logs (default: False)
        """
        super().__init__(verbose=verbose)
        
        # Thresholds
        self.prune_opa = prune_opa
        self.grow_grad2d = grow_grad2d
        self.grow_scale3d = grow_scale3d
        self.grow_scale2d = grow_scale2d
        self.prune_scale3d = prune_scale3d
        self.prune_scale2d = prune_scale2d
        
        # Scheduling
        self.refine_start_iter = refine_start_iter
        self.refine_stop_iter = refine_stop_iter
        self.reset_every = reset_every
        self.refine_every = refine_every
        self.pause_refine_after_reset = pause_refine_after_reset
        
        # Mode flags
        self.absgrad = absgrad
        self.revised_opacity = revised_opacity
        
        # Track last reset iteration
        self._last_reset_iter = 0
    
    def step_post_backward(
        self,
        params: Dict[str, nn.Parameter],
        optimizers: Dict[str, torch.optim.Optimizer],
        scene_scale: float,
        iteration: int,
        viewport_info: ViewportInfo,
        radii: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """
        Main refinement logic executed after backward pass.
        
        Args:
            params: Model parameters dict with keys: means, scales, quats, opacities, features_dc, features_rest
            optimizers: Optimizers dict with same keys
            scene_scale: Scene extent for normalization
            iteration: Current training iteration
            viewport_info: Camera/viewport info for gradient normalization
            radii: Optional maximum 2D screen-space radii (normalized by resolution)
            **kwargs: Additional arguments (e.g., gaussian_ids for frustum culling)
        """
        # Initialize state on first call
        if self.state is None:
            self.initialize_state(params, scene_scale)
        
        # Update scene scale (in case it changed)
        self.state["scene_scale"] = torch.tensor(scene_scale, device=params["means"].device)
        
        # Check if we should perform refinement this iteration
        do_refine = (
            iteration >= self.refine_start_iter
            and iteration < self.refine_stop_iter
            and iteration % self.refine_every == 0
        )
        
        # Check if we're in pause window after reset
        if self.pause_refine_after_reset > 0:
            iters_since_reset = iteration - self._last_reset_iter
            if 0 < iters_since_reset <= self.pause_refine_after_reset:
                do_refine = False
        
        # Accumulate gradients (always, even when not refining)
        self._accumulate_gradients(params, viewport_info, **kwargs)
        
        # Update radii if provided
        if radii is not None:
            # Normalize by resolution (max of width/height)
            max_res = max(viewport_info.width, viewport_info.height)
            self.state["radii"] = torch.maximum(
                self.state["radii"],
                radii / max_res
            )
        
        # Perform refinement if scheduled
        if do_refine:
            self._refine(params, optimizers, iteration)
        
        # Check for opacity reset
        if self.reset_every > 0 and iteration > 0 and iteration % self.reset_every == 0:
            self._reset_opacity(params, optimizers)
            self._last_reset_iter = iteration
            if self.verbose:
                print(f"  🔄 Reset opacity at iteration {iteration}")
    
    def _accumulate_gradients(
        self,
        params: Dict[str, nn.Parameter],
        viewport_info: ViewportInfo,
        gaussian_ids: Optional[torch.Tensor] = None,
    ):
        """
        Accumulate screen-space gradients for visible Gaussians.
        
        Args:
            params: Model parameters
            viewport_info: Camera/viewport info
            gaussian_ids: Optional indices mapping visible to all Gaussians (for frustum culling)
        """
        means = params["means"]
        
        if means.grad is None:
            return
        
        # Get gradients (shape: [N, 3] for all Gaussians or [N_visible, 3] for culled)
        grads = means.grad.detach()
        
        # Take only x, y gradients (screen space)
        grads_2d = grads[..., :2]  # [N, 2]
        
        # Normalize by viewport size and number of cameras
        # This makes the threshold scale-invariant
        grads_2d[..., 0] *= viewport_info.width / 2.0 * viewport_info.n_cameras
        grads_2d[..., 1] *= viewport_info.height / 2.0 * viewport_info.n_cameras
        
        # Compute norm
        grad_norms = torch.norm(grads_2d, dim=-1)  # [N]
        
        # Accumulate
        if gaussian_ids is not None:
            # Frustum culling active: map visible Gaussians to full model
            self.state["grad2d"][gaussian_ids] += grad_norms
            self.state["count"][gaussian_ids] += 1
        else:
            # No culling: direct update
            self.state["grad2d"] += grad_norms
            self.state["count"] += 1
    
    def _refine(
        self,
        params: Dict[str, nn.Parameter],
        optimizers: Dict[str, torch.optim.Optimizer],
        iteration: int,
    ):
        """
        Perform densification and pruning.
        
        Args:
            params: Model parameters
            optimizers: Optimizers
            iteration: Current iteration
        """
        device = params["means"].device
        n_before = params["means"].shape[0]
        
        # Compute average gradients
        if self.absgrad:
            # AbsGS mode: use absolute (unnormalized) gradients
            grads = self.state["grad2d"]
        else:
            # Standard: average over visible frames
            grads = self.state["grad2d"] / self.state["count"].clamp_min(1)
        
        # Get current scales (convert from log-space)
        scales = torch.exp(params["scales"])  # [N, 3]
        max_scales = torch.max(scales, dim=1).values  # [N]
        
        # Get opacities (convert from logit-space)
        opacities = torch.sigmoid(params["opacities"]).squeeze(-1)  # [N]
        
        # === Compute ALL masks BEFORE any operations (to keep sizes consistent) ===
        is_grad_high = grads >= self.grow_grad2d
        is_small = max_scales <= self.grow_scale3d * self.state["scene_scale"]
        is_large = max_scales > self.grow_scale3d * self.state["scene_scale"]
        
        # Compute all masks on original Gaussian count
        mask_duplicate = is_grad_high & is_small
        mask_split = is_grad_high & is_large
        
        # Add 2D scale criterion if enabled
        if self.grow_scale2d > 0:
            is_large_2d = self.state["radii"] > self.grow_scale2d
            mask_split = mask_split | is_large_2d
        
        # === DUPLICATE (Clone) ===
        n_duplicated = mask_duplicate.sum().item()
        if n_duplicated > 0:
            params = ops.duplicate(params, optimizers, self.state, mask_duplicate)
            if self.verbose:
                print(f"  ➕ Duplicated {n_duplicated:,} Gaussians")
        
        # === SPLIT ===
        # IMPORTANT: Don't split newly duplicated Gaussians
        # The duplicate operation adds Gaussians at the end, so exclude them
        if n_duplicated > 0:
            # Extend mask to new size and set new Gaussians to False
            mask_split_extended = torch.zeros(params["means"].shape[0], dtype=torch.bool, device=device)
            mask_split_extended[:len(mask_split)] = mask_split
            mask_split = mask_split_extended
        
        n_split = mask_split.sum().item()
        if n_split > 0:
            params = ops.split(
                params, optimizers, self.state, mask_split,
                revised_opacity=self.revised_opacity
            )
            if self.verbose:
                print(f"  ✂️  Split {n_split:,} Gaussians into {n_split * 2:,}")
        
        # === PRUNE ===
        # Recompute properties on current Gaussian count (after duplicate/split)
        opacities_current = torch.sigmoid(params["opacities"]).squeeze(-1)
        scales_current = torch.exp(params["scales"])
        max_scales_current = torch.max(scales_current, dim=1).values
        
        # Always prune low opacity
        mask_prune = opacities_current < self.prune_opa
        
        # After first reset, also prune oversized Gaussians
        if iteration > self.reset_every:
            # 3D world-space pruning
            is_too_large_3d = max_scales_current > self.prune_scale3d * self.state["scene_scale"]
            mask_prune = mask_prune | is_too_large_3d
            
            # 2D screen-space pruning (if enabled)
            # Note: Often set to high threshold (0.15) to avoid artifacts
            if self.prune_scale2d > 0:
                is_too_large_2d = self.state["radii"] > self.prune_scale2d
                mask_prune = mask_prune | is_too_large_2d
        
        n_pruned = mask_prune.sum().item()
        if n_pruned > 0:
            params = ops.remove(params, optimizers, self.state, mask_prune)
            if self.verbose:
                print(f"  🗑️  Pruned {n_pruned:,} Gaussians")
        
        # Summary
        n_after = params["means"].shape[0]
        net_change = n_after - n_before
        if self.verbose:
            if net_change == 0:
                print(f"  ⏸️  No net change in Gaussian count")
            else:
                print(f"  📊 Net change: {net_change:+,} Gaussians (total: {n_after:,})")
    
    def _reset_opacity(
        self,
        params: Dict[str, nn.Parameter],
        optimizers: Dict[str, torch.optim.Optimizer],
    ):
        """
        Reset opacity to prevent saturation.
        
        Clamps all opacities to max of 2*prune_opa (0.01 by default).
        This follows the original 3DGS implementation.
        
        Args:
            params: Model parameters
            optimizers: Optimizers
        """
        params = ops.reset_opacity(
            params, optimizers, self.prune_opa, revised=True
        )
