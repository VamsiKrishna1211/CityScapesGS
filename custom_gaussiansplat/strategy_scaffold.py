"""
Scaffold-GS densification strategy.

Mirrors the interface of gsplat.DefaultStrategy so that train.py can call
step_pre_backward / step_post_backward without any model-type conditionals.

The model owns its densification buffers (opacity_accum, anchor_denom, etc.)
because they need to be checkpointed and resized alongside the anchor tensors.
This strategy acts as an orchestrator that decides *when* to call the model's
stat-update and anchor-adjustment methods.
"""

from typing import Dict, Optional

import torch


class ScaffoldStrategy:
    """Densification strategy for Scaffold-GS.

    Pass the ScaffoldModel instance at construction time so that
    step_post_backward can delegate to the model without needing it in kwargs,
    keeping the call signature identical to gsplat.DefaultStrategy.
    """

    def __init__(
        self,
        model,  # ScaffoldModel — circular import avoided by using Any at type level
        densify_from_iter: int = 500,
        densify_until_iter: int = 15_000,
        densify_interval: int = 100,
        grad_threshold: float = 0.0002,
        prune_opa: float = 0.005,
        verbose: bool = False,
    ):
        self.model = model
        self.densify_from_iter = densify_from_iter
        self.densify_until_iter = densify_until_iter
        self.densify_interval = densify_interval
        self.grad_threshold = grad_threshold
        self.prune_opa = prune_opa
        self.verbose = verbose

    # ------------------------------------------------------------------
    # Interface matching gsplat.DefaultStrategy
    # ------------------------------------------------------------------

    def initialize_state(self, scene_scale: float = 1.0) -> dict:
        """Return an empty state dict.

        ScaffoldModel owns its own densification buffers as registered
        nn.Module buffers, so no external state is needed here.
        """
        return {}

    def check_sanity(
        self,
        params: Dict,
        optimizers: Dict,
    ) -> None:
        """No cross-checks needed for Scaffold-GS."""
        pass

    def step_pre_backward(
        self,
        params: Dict,
        optimizers: Dict,
        state: dict,
        step: int,
        info: dict,
    ) -> None:
        """Retain gradients on means2d so they are available after backward.

        gsplat rasterization outputs means2d as a non-leaf tensor; without
        retain_grad() its .grad is discarded after backward.
        """
        means2d: Optional[torch.Tensor] = info.get("means2d")
        if means2d is not None and means2d.requires_grad:
            means2d.retain_grad()

    def step_post_backward(
        self,
        params: Dict,
        optimizers,          # GSOptimizers or dict — passed through to model
        state: dict,
        step: int,
        info: dict,
        packed: bool = False,
    ) -> None:
        """Update anchor statistics and periodically grow/prune anchors.

        After any anchor adjustment the params dict is refreshed in-place so
        that the caller's subsequent model.update_params_from_dict(params) is
        a clean no-op rather than reverting the densification.
        """
        if not (self.densify_from_iter <= step <= self.densify_until_iter):
            return

        means2d: Optional[torch.Tensor] = info.get("means2d")
        if means2d is not None:
            opacity = info.get("neural_opacity")
            selection_mask = info.get("selection_mask")
            radii = info.get("radii")

            if radii is not None:
                # radii shape: (N, 2) — x and y screen-space radii per Gaussian.
                # Mirrors gsplat's own mask: (radii > 0).all(dim=-1)
                visibility_filter = (radii > 0).all(dim=-1)
                if visibility_filter.dim() > 1:
                    # Defensive: collapse any remaining camera/batch dims (C, N) → (N,)
                    visibility_filter = visibility_filter.any(dim=0)
            else:
                visibility_filter = torch.ones(
                    means2d.shape[0],
                    dtype=torch.bool,
                    device=self.model._anchor.device,
                )

            anchor_visible_mask = torch.ones(
                self.model._anchor.shape[0],
                dtype=torch.bool,
                device=self.model._anchor.device,
            )

            self.model.update_training_stats(
                viewspace_point_tensor=means2d,
                opacity=opacity,
                update_filter=visibility_filter,
                offset_selection_mask=selection_mask,
                anchor_visible_mask=anchor_visible_mask,
            )

        if step > self.densify_from_iter and step % self.densify_interval == 0:
            self.model.adjust_anchor(
                check_interval=self.densify_interval,
                success_threshold=0.8,
                grad_threshold=self.grad_threshold,
                min_opacity=self.prune_opa,
                optimizers=optimizers,
            )
            if self.verbose:
                print(
                    f"[ScaffoldStrategy] step {step}: "
                    f"anchors after adjustment = {self.model._anchor.shape[0]:,}"
                )

            # Refresh params dict so update_params_from_dict sees the new tensors.
            for k, v in self.model.get_params_dict().items():
                params[k] = v
