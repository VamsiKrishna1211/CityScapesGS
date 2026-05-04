"""Rasterizer and image-preparation utilities shared by train.py and train_semantics.py."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

import torch
import torch.nn.functional as F
from gs_types import RenderOutput, RenderParams
from gsplat import rasterization  # type: ignore[import-untyped]

if TYPE_CHECKING:
    from models import BaseTrainableModel


# ---------------------------------------------------------------------------
# Camera helpers
# ---------------------------------------------------------------------------


def _build_viewmat(cam: dict, device: torch.device) -> torch.Tensor:
    """Construct a [1, 4, 4] view matrix from a camera dict."""
    viewmat = torch.eye(4, device=device, dtype=torch.float32)
    viewmat[:3, :3] = cam["R"]
    viewmat[:3, 3] = cam["T"]
    return viewmat.unsqueeze(0)


def _build_intrinsics(cam: dict, device: torch.device) -> torch.Tensor:
    """Construct a [3, 3] intrinsics matrix from a camera dict."""
    return torch.tensor(
        [[cam["fx"], 0.0, cam["cx"]],
         [0.0, cam["fy"], cam["cy"]],
         [0.0, 0.0, 1.0]],
        dtype=torch.float32, device=device,
    )


# ---------------------------------------------------------------------------
# Tensor preparation helpers
# ---------------------------------------------------------------------------


@torch.no_grad()
def _prepare_gt_image(
    gt_image: torch.Tensor,
    device: torch.device,
    target_h: Optional[int] = None,
    target_w: Optional[int] = None,
) -> torch.Tensor:
    """Move gt_image to device, normalize shape, and optionally resize to target HW.

    Returns tensor in [B, H, W, C] layout.
    """
    if not gt_image.is_cuda:
        gt_image = gt_image.to(device)
    if gt_image.dim() == 3:
        gt_image = gt_image.unsqueeze(0)

    if target_h is not None and target_w is not None:
        h, w = gt_image.shape[1], gt_image.shape[2]
        if h != target_h or w != target_w:
            gt_bchw = gt_image.permute(0, 3, 1, 2)
            gt_bchw = F.interpolate(
                gt_bchw,
                size=(target_h, target_w),
                mode="bilinear",
                align_corners=False,
            )
            gt_image = gt_bchw.permute(0, 2, 3, 1)
    return gt_image


@torch.no_grad()
def _prepare_depth_tensor(
    depth_tensor: Optional[torch.Tensor],
    device: torch.device,
    target_h: Optional[int] = None,
    target_w: Optional[int] = None,
) -> Optional[torch.Tensor]:
    """Move depth tensor to device, normalize dims, and optionally resize to target HW.

    Returns depth in [B, H, W] layout when available.
    """
    if depth_tensor is None:
        return None
    if not depth_tensor.is_cuda:
        depth_tensor = depth_tensor.to(device)

    if depth_tensor.dim() == 2:
        depth_tensor = depth_tensor.unsqueeze(0)
    elif depth_tensor.dim() == 4 and depth_tensor.shape[-1] == 1:
        depth_tensor = depth_tensor[..., 0]

    if target_h is not None and target_w is not None and depth_tensor.dim() == 3:
        h, w = depth_tensor.shape[-2], depth_tensor.shape[-1]
        if h != target_h or w != target_w:
            depth_bchw = depth_tensor.unsqueeze(1)
            depth_bchw = F.interpolate(
                depth_bchw,
                size=(target_h, target_w),
                mode="nearest",
            )
            depth_tensor = depth_bchw[:, 0]

    return depth_tensor


# ---------------------------------------------------------------------------
# Rasterizer — encapsulates gsplat rasterization + depth extraction
# ---------------------------------------------------------------------------


class Rasterizer:
    """Encapsulates gsplat rasterization and post-processing.

    Handles the rasterization call, depth map extraction, alpha masking,
    and tensor permutations for loss computation.
    """

    def __init__(
        self,
        model: "BaseTrainableModel",
        sh_cfg: Any,
        packed: bool = False,
        absgrad: bool = False,
    ) -> None:
        self.model = model
        self.sh_cfg = sh_cfg
        self.packed = packed
        self.absgrad = absgrad
        # OPTIMIZATION: Cache view matrices per camera to avoid recomputation
        self._viewmat_cache: dict[int, torch.Tensor] = {}
        self._intrinsics_cache: dict[int, torch.Tensor] = {}

    def render(
        self,
        cam: dict[str, Any],
        gt_image: torch.Tensor,
        device: torch.device,
        lod: Optional[int] = None,
    ) -> RenderOutput:
        """Rasterize the scene for a single camera view.

        Args:
            cam: Camera dict with R, T, fx, fy, cx, cy, width, height.
            gt_image: Ground-truth image [B, H, W, C] (already on device).
            device: Target device.
            lod: Optional LoD level to render.

        Returns:
            RenderOutput with all tensors needed for loss and logging.
        """
        cam_uid = cam["uid"]
        # OPTIMIZATION: Use cached view matrix if available
        if cam_uid not in self._viewmat_cache:
            self._viewmat_cache[cam_uid] = _build_viewmat(cam, device)
            self._intrinsics_cache[cam_uid] = _build_intrinsics(cam, device)
        viewmat = self._viewmat_cache[cam_uid]
        K = self._intrinsics_cache[cam_uid]

        rp: RenderParams = self.model.get_render_params(cam, self.sh_cfg, is_training=True, lod=lod)

        # Background colors - randomly select one from the list
        # backgrounds_list = torch.tensor([
        #     [1.0, 1.0, 1.0],  # White
        #     [0.0, 0.0, 0.0],  # Black
        #     [0.5, 0.5, 0.5],  # Gray
        #     [0.2, 0.2, 0.2],  # Dark gray
        # ], device=device, dtype=rp.means.dtype)

        # Randomly select one background, shape: [3, 1]
        # background = backgrounds_list[torch.randint(0, backgrounds_list.size(0), (1,))]

        render_output, render_alpha, render_meta = rasterization(
            means=rp.means,
            quats=rp.quats,
            scales=rp.scales,
            opacities=rp.opacities.squeeze(-1),
            colors=rp.colors,
            viewmats=viewmat,
            Ks=K[None, ...],
            width=cam["width"],
            height=cam["height"],
            sh_degree=rp.sh_degree,
            packed=self.packed,
            absgrad=self.absgrad,
            render_mode="RGB+ED",
            camera_model="pinhole",
            # backgrounds=background,
        )

        # Forward any model-specific meta fields (e.g. Scaffold-GS neural_opacity).
        if rp.neural_opacity is not None:
            render_meta["neural_opacity"] = rp.neural_opacity
        if rp.selection_mask is not None:
            render_meta["selection_mask"] = rp.selection_mask

        render = render_output[..., 0:3]             # [B, H, W, 3]
        alpha = render_alpha                          # [B, H, W, 1]
        render_depth_raw = render_output[..., 3]      # [B, H, W]

        alpha_2d = alpha[..., 0] if alpha.dim() == 4 else alpha
        depth_map = render_depth_raw # No conversion needed since gsplat outputs depth in linear space (not disparity).

        depth_mask = (
            (alpha_2d > 0.0)
            & torch.isfinite(depth_map)
            & (depth_map > 0)
        )

        render_perm = render.permute(0, 3, 1, 2)     # [B, C, H, W]
        gt_perm = gt_image.permute(0, 3, 1, 2)

        return RenderOutput(
            render=render,
            alpha=alpha,
            depth_map=depth_map,
            depth_mask=depth_mask,
            depth_mask_bchw=depth_mask.unsqueeze(1),
            depth_map_bchw=depth_map.unsqueeze(1),
            render_perm=render_perm,
            gt_perm=gt_perm,
            meta=render_meta,
        )
