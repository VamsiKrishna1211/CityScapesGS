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
        gaussian_ids: Optional[torch.Tensor] = None,
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
            gaussian_ids: Optional indices mapping visible to all Gaussians (for frustum culling)
            radii: Optional maximum 2D screen-space radii for visible Gaussians
            **kwargs: Additional arguments
        """
        # Initialize state on first call
        if self.state is None:
            self.initialize_state(params, scene_scale)
        
        # CHECK: Model may have grown due to densification
        # Reinitialize state if the number of Gaussians changed
        current_model_size = params["means"].shape[0]
        if self.state["grad2d"].shape[0] != current_model_size:
            if self.verbose:
                old_size = self.state["grad2d"].shape[0]
                print(f"  ℹ️  Model size changed {old_size} -> {current_model_size}, reinitializing state")
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
        # gaussian_ids is REQUIRED - it maps visible Gaussians to full model
        if gaussian_ids is None:
            raise RuntimeError(
                "gaussian_ids cannot be None. It must map all visible Gaussians to the full model. "
                "Pass torch.arange(N_visible) if all rendered Gaussians map sequentially."
            )
        self._accumulate_gradients(params, viewport_info, gaussian_ids=gaussian_ids, radii=radii)
        
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
        gaussian_ids: torch.Tensor,
        radii: Optional[torch.Tensor] = None,
    ):
        """
        Accumulate screen-space gradients and radii for visible Gaussians.
        
        Args:
            params: Model parameters
            viewport_info: Camera/viewport info
            gaussian_ids: REQUIRED - indices [N_visible] of visible Gaussians in the full model
            radii: Optional 2D screen-space radii for visible Gaussians [N_visible]
            
        Note:
            - Rasterization operates on ALL Gaussians but only renders visible ones
            - gaussian_ids tells us which Gaussians were visible
            - means.grad is [N_full, 3] but only gaussian_ids positions have non-zero gradients
            - We extract gradients at gaussian_ids positions and accumulate them
        """
        means = params["means"]
        
        if means.grad is None:
            return
        
        # means.grad is [N_full, 3] with gradients for ALL Gaussians (sparse)
        # Extract gradients only for visible Gaussians
        grads = means.grad[gaussian_ids].detach()  # [N_visible, 3]
        
        # Take only x, y gradients (screen space)
        grads_2d = grads[..., :2]  # [N_visible, 2]
        
        # Normalize by viewport size and number of cameras
        # This makes the threshold scale-invariant
        grads_2d[..., 0] *= viewport_info.width / 2.0 * viewport_info.n_cameras
        grads_2d[..., 1] *= viewport_info.height / 2.0 * viewport_info.n_cameras
        
        # Compute norm
        grad_norms = torch.norm(grads_2d, dim=-1)  # [N_visible]
        
        # Accumulate visible gradients at their corresponding indices in the full state
        self.state["grad2d"][gaussian_ids] += grad_norms
        self.state["count"][gaussian_ids] += 1
        
        # Accumulate radii if provided
        if radii is not None:
            max_res = max(viewport_info.width, viewport_info.height)
            # radii is [N_visible], map to full model indices
            self.state["radii"][gaussian_ids] = torch.maximum(
                self.state["radii"][gaussian_ids],
                radii[gaussian_ids] / max_res
            )
    
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
