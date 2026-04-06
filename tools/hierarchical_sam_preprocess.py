"""
Hierarchical SAM Preprocessing with CLIP Features
====================================================
Combines LangSplat's SAM-based multi-scale segmentation with hierarchical
parent-child mask associations.  Produces:

  - {name}_s.npy   : segmentation maps  (N_levels × H × W)   [LangSplat compat]
  - {name}_f.npy   : CLIP features       (N_total × 512)      [LangSplat compat]
  - {name}_hierarchy.json : tree structure with parent-child relationships

Usage:
    python hierarchical_sam_preprocess.py \\
        --dataset_path /path/to/dataset \\
        --sam_ckpt_path ckpts/sam_vit_h_4b8939.pth \\
        --max_depth 10 \\
        --downscale 1
"""

from __future__ import annotations

import argparse
import json
import os
import random
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Type

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from tqdm import tqdm
from segment_anything import SamAutomaticMaskGenerator, SamPredictor, sam_model_registry

try:
    import open_clip
except ImportError:
    raise ImportError("open_clip is not installed, install it with `pip install open-clip-torch`")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class SegNode:
    """A single segmentation mask node in the hierarchy tree."""
    node_id: int                          # globally unique mask ID
    parent_id: int                        # -1 for root nodes
    depth: int                            # 0 = top-level from initial SAM pass
    label: str                            # e.g. "mask_0", "mask_0_sub_3"
    mask: Optional[np.ndarray] = None     # H×W bool mask in original image space
    bbox: Optional[List[int]] = None      # [x, y, w, h]
    area: int = 0                         # pixel count
    score: float = 0.0                    # SAM quality score (iou * stability)
    clip_feature: Optional[np.ndarray] = None  # 512-d CLIP embedding
    children: List["SegNode"] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Serialisable dict (no mask/feature arrays)."""
        return {
            "node_id": self.node_id,
            "parent_id": self.parent_id,
            "depth": self.depth,
            "label": self.label,
            "bbox": self.bbox,
            "area": self.area,
            "score": round(float(self.score), 4),
            "n_children": len(self.children),
            "children": [c.to_dict() for c in self.children],
        }


# ---------------------------------------------------------------------------
# OpenCLIP wrapper  (same as LangSplat preprocess.py)
# ---------------------------------------------------------------------------

@dataclass
class OpenCLIPNetworkConfig:
    _target: Type = field(default_factory=lambda: OpenCLIPNetwork)
    clip_model_type: str = "ViT-B-16"
    clip_model_pretrained: str = "laion2b_s34b_b88k"
    clip_n_dims: int = 512
    negatives: Tuple[str] = ("object", "things", "stuff", "texture")
    positives: Tuple[str] = ("",)


class OpenCLIPNetwork(nn.Module):
    def __init__(self, config: OpenCLIPNetworkConfig):
        super().__init__()
        self.config = config
        self.process = torchvision.transforms.Compose([
            torchvision.transforms.Resize((224, 224)),
            torchvision.transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711],
            ),
        ])
        model, _, _ = open_clip.create_model_and_transforms(
            self.config.clip_model_type,
            pretrained=self.config.clip_model_pretrained,
            precision="fp16",
        )
        model.eval()
        self.tokenizer = open_clip.get_tokenizer(self.config.clip_model_type)
        self.model = model.to("cuda")
        self.clip_n_dims = self.config.clip_n_dims

    def encode_image(self, input_tensor):
        processed = self.process(input_tensor).half()
        return self.model.encode_image(processed)


# ---------------------------------------------------------------------------
# Utility functions  (adapted from LangSplat preprocess.py)
# ---------------------------------------------------------------------------

def get_seg_img(mask_dict, image_np):
    """Extract the masked region as a cropped image."""
    image = image_np.copy()
    image[mask_dict['segmentation'] == 0] = np.array([0, 0, 0], dtype=np.uint8)
    x, y, w, h = np.int32(mask_dict['bbox'])
    return image[y:y+h, x:x+w, ...]


def pad_and_resize_gpu(seg_img):
    """Pad to square and resize to 224×224 on GPU."""
    t = torch.from_numpy(seg_img).to('cuda').permute(2, 0, 1).float().unsqueeze(0) / 255.0
    h, w = t.shape[2], t.shape[3]
    l = max(h, w)
    padded = torch.zeros(1, 3, l, l, device='cuda')
    if h > w:
        padded[:, :, :, (h - w) // 2:(h - w) // 2 + w] = t
    else:
        padded[:, :, (w - h) // 2:(w - h) // 2 + h, :] = t
    resized = F.interpolate(padded, size=(224, 224), mode='bilinear', align_corners=False)
    return resized.squeeze(0)  # 3×224×224 on GPU


def mask_nms(masks, scores, iou_thr=0.7, score_thr=0.1, inner_thr=0.2, **kwargs):
    """Vectorised mask NMS on GPU (from LangSplat preprocess.py)."""
    device = masks.device
    scores, idx = scores.sort(0, descending=True)
    num_masks = idx.shape[0]
    masks_ord = masks[idx.view(-1), :]
    masks_area = torch.sum(masks_ord, dim=(1, 2), dtype=torch.float)

    HW = masks_ord.shape[1] * masks_ord.shape[2]
    masks_flat = masks_ord.reshape(num_masks, HW).float()
    intersection_matrix = masks_flat @ masks_flat.T
    union_matrix = masks_area.unsqueeze(0) + masks_area.unsqueeze(1) - intersection_matrix
    iou_matrix = intersection_matrix / union_matrix.clamp(min=1e-6)

    ratio_i = intersection_matrix / masks_area.unsqueeze(1).clamp(min=1)
    ratio_j = intersection_matrix / masks_area.unsqueeze(0).clamp(min=1)
    inner_val = 1 - ratio_j * ratio_i

    upper = torch.triu(torch.ones(num_masks, num_masks, dtype=torch.bool, device=device), diagonal=0)
    cond1 = upper & (ratio_i < 0.5) & (ratio_j >= 0.85)
    inner_iou_matrix = torch.where(cond1, inner_val, torch.zeros_like(inner_val))
    cond2 = upper & (ratio_i >= 0.85) & (ratio_j < 0.5)
    inner_iou_matrix = inner_iou_matrix + torch.where(cond2, inner_val, torch.zeros_like(inner_val)).T

    iou_matrix.triu_(diagonal=1)
    iou_max, _ = iou_matrix.max(dim=0)
    inner_iou_matrix_u = torch.triu(inner_iou_matrix, diagonal=1)
    inner_iou_max_u, _ = inner_iou_matrix_u.max(dim=0)
    inner_iou_matrix_l = torch.tril(inner_iou_matrix, diagonal=1)
    inner_iou_max_l, _ = inner_iou_matrix_l.max(dim=0)

    keep = iou_max <= iou_thr
    keep_conf = scores > score_thr
    keep_inner_u = inner_iou_max_u <= 1 - inner_thr
    keep_inner_l = inner_iou_max_l <= 1 - inner_thr

    if keep_conf.sum() == 0:
        index = scores.topk(3).indices
        keep_conf[index, 0] = True
    if keep_inner_u.sum() == 0:
        index = scores.topk(3).indices
        keep_inner_u[index, 0] = True
    if keep_inner_l.sum() == 0:
        index = scores.topk(3).indices
        keep_inner_l[index, 0] = True

    keep *= keep_conf
    keep *= keep_inner_u
    keep *= keep_inner_l
    return idx[keep]


def filter_masks(keep: torch.Tensor, masks_result):
    """Keep only the masks whose indices are in `keep`."""
    keep_set = set(keep.int().cpu().numpy().tolist())
    return [m for i, m in enumerate(masks_result) if i in keep_set]


def masks_update(*args, **kwargs):
    """Apply mask NMS to each scale level independently."""
    masks_new = ()
    for masks_lvl in args:
        if len(masks_lvl) == 0:
            masks_new += (masks_lvl,)
            continue
        seg_pred = torch.from_numpy(
            np.stack([m['segmentation'] for m in masks_lvl], axis=0)
        ).cuda()
        iou_pred = torch.tensor([m['predicted_iou'] for m in masks_lvl], device='cuda')
        stability = torch.tensor([m['stability_score'] for m in masks_lvl], device='cuda')
        scores = stability * iou_pred
        keep_idx = mask_nms(seg_pred, scores, **kwargs)
        masks_lvl = filter_masks(keep_idx, masks_lvl)
        masks_new += (masks_lvl,)
    return masks_new


# ---------------------------------------------------------------------------
# HierarchicalSAMProcessor
# ---------------------------------------------------------------------------

class HierarchicalSAMProcessor:
    """
    Generates SAM masks, builds a hierarchy via containment,
    recursively refines, and extracts CLIP features for each node.
    """

    def __init__(
        self,
        sam_checkpoint: str,
        sam_model_type: str = "vit_h",
        clip_model: Optional[OpenCLIPNetwork] = None,
        max_depth: int = 10,
        min_mask_area: int = 100,
        containment_threshold: float = 0.70,
        refine_points_per_side: int = 32,
        refine_points_per_batch: int = 256,
        refine_min_area: int = 64,
        device: str = "cuda",
    ):
        self.max_depth = max_depth
        self.min_mask_area = min_mask_area
        self.containment_threshold = containment_threshold
        self.refine_points_per_side = refine_points_per_side
        self.refine_points_per_batch = refine_points_per_batch
        self.refine_min_area = refine_min_area
        self.device = device
        self._node_counter = 0

        # Load SAM
        print(f"[SAM] Loading {sam_model_type} from {sam_checkpoint}")
        self.sam = sam_model_registry[sam_model_type](checkpoint=sam_checkpoint).to(device)
        self.sam.eval()

        # Primary mask generator (same settings as LangSplat preprocess.py)
        self.mask_generator = SamAutomaticMaskGenerator(
            model=self.sam,
            points_per_side=32,
            points_per_batch=512,
            pred_iou_thresh=0.7,
            box_nms_thresh=0.7,
            stability_score_thresh=0.85,
            crop_n_layers=1,
            crop_n_points_downscale_factor=1,
            min_mask_region_area=min_mask_area,
        )

        # SAM predictor for box-prompted refinement
        self.predictor = SamPredictor(self.sam)

        # CLIP model
        self.clip_model = clip_model

    def _next_id(self) -> int:
        self._node_counter += 1
        return self._node_counter

    # ----- Phase 1: Generate all SAM masks at multiple scales -----

    def generate_multi_scale_masks(self, image_bgr: np.ndarray):
        """
        Run SAM AutomaticMaskGenerator with crop_n_layers=1 to get
        default / s / m / l scale masks.  Apply NMS filtering.
        Auto-downscales large images to avoid OOM.
        Returns (masks_default, masks_s, masks_m, masks_l) after filtering.
        """
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        # Auto-downscale for SAM if image is too large
        h, w = image_rgb.shape[:2]
        max_side = 1080
        if max(h, w) > max_side:
            scale = max_side / max(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            sam_image = cv2.resize(image_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
            print(f"    Auto-downscaled {w}×{h} → {new_w}×{new_h} for SAM")
        else:
            sam_image = image_rgb
            scale = 1.0

        try:
            masks_default, masks_s, masks_m, masks_l = self.mask_generator.generate(sam_image)
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            if 'out of memory' in str(e).lower():
                torch.cuda.empty_cache()
                # Retry with further downscale
                retry_scale = 0.5
                rw, rh = int(sam_image.shape[1] * retry_scale), int(sam_image.shape[0] * retry_scale)
                sam_image = cv2.resize(sam_image, (rw, rh), interpolation=cv2.INTER_AREA)
                scale *= retry_scale
                print(f"    [OOM] Retrying with further downscale → {rw}×{rh}")
                masks_default, masks_s, masks_m, masks_l = self.mask_generator.generate(sam_image)
            else:
                raise

        # If we downscaled, upscale mask segmentations + bboxes back to original resolution
        if scale < 1.0:
            inv_scale = 1.0 / scale
            for mask_list in [masks_default, masks_s, masks_m, masks_l]:
                for m in mask_list:
                    seg = m['segmentation']
                    # Resize boolean mask back to original image size
                    seg_uint8 = seg.astype(np.uint8) * 255
                    seg_full = cv2.resize(seg_uint8, (w, h), interpolation=cv2.INTER_NEAREST)
                    m['segmentation'] = (seg_full > 127)
                    # Scale bbox
                    bx, by, bw, bh = m['bbox']
                    m['bbox'] = [
                        bx * inv_scale, by * inv_scale,
                        bw * inv_scale, bh * inv_scale,
                    ]
                    m['area'] = int(m['segmentation'].sum())

        masks_default, masks_s, masks_m, masks_l = masks_update(
            masks_default, masks_s, masks_m, masks_l,
            iou_thr=0.8, score_thr=0.7, inner_thr=0.5,
        )
        return masks_default, masks_s, masks_m, masks_l

    # ----- Phase 2: Build hierarchy from containment -----

    def _compute_containment(self, child_mask: np.ndarray, parent_mask: np.ndarray) -> float:
        """Fraction of child pixels that lie within parent."""
        child_area = child_mask.sum()
        if child_area == 0:
            return 0.0
        overlap = np.logical_and(child_mask, parent_mask).sum()
        return float(overlap) / float(child_area)

    def build_hierarchy(
        self,
        all_masks: List[dict],
        image_shape: Tuple[int, int],
    ) -> List[SegNode]:
        """
        Given a list of SAM mask dicts, build a tree based on containment.

        Algorithm:
          1. Sort masks by area (largest first).
          2. For each mask, find the *smallest* existing node whose mask
             contains ≥ containment_threshold of the current mask's pixels.
          3. Attach as child of that node.  If no parent found, it's a root.
        """
        # Sort by area descending
        sorted_masks = sorted(all_masks, key=lambda m: m.get('area', 0), reverse=True)

        nodes: List[SegNode] = []
        node_lookup: Dict[int, SegNode] = {}

        for mask_dict in sorted_masks:
            seg = mask_dict['segmentation'].astype(bool)
            area = int(seg.sum())
            if area < self.min_mask_area:
                continue

            x, y, w, h = mask_dict['bbox']
            bbox = [int(x), int(y), int(w), int(h)]
            iou_score = float(mask_dict.get('predicted_iou', 0.0))
            stability = float(mask_dict.get('stability_score', 0.0))
            score = iou_score * stability

            nid = self._next_id()
            node = SegNode(
                node_id=nid,
                parent_id=-1,
                depth=0,
                label=f"mask_{nid}",
                mask=seg,
                bbox=bbox,
                area=area,
                score=score,
            )

            # Find smallest parent whose mask contains this one
            best_parent: Optional[SegNode] = None
            best_parent_area = float('inf')

            # Walk all existing nodes and check containment
            for candidate in node_lookup.values():
                if candidate.area <= area:
                    continue  # parent must be larger
                containment = self._compute_containment(seg, candidate.mask)
                if containment >= self.containment_threshold:
                    if candidate.area < best_parent_area:
                        best_parent = candidate
                        best_parent_area = candidate.area

            if best_parent is not None:
                node.parent_id = best_parent.node_id
                node.depth = best_parent.depth + 1
                node.label = f"{best_parent.label}_sub_{len(best_parent.children)}"
                best_parent.children.append(node)
            else:
                nodes.append(node)  # root node

            node_lookup[nid] = node

        return nodes

    # ----- Phase 3: Recursive refinement -----

    def _refine_node(
        self,
        node: SegNode,
        image_rgb: np.ndarray,
        current_depth: int,
    ):
        """
        For a node whose area is large enough, crop the image to its bbox,
        re-run SAM with higher density to find finer sub-masks, and attach
        as children.  Recurse.
        """
        if current_depth >= self.max_depth:
            return
        if node.area < self.min_mask_area * 4:
            return  # too small to subdivide

        x, y, w, h = node.bbox
        if w < 16 or h < 16:
            return

        # Crop the image to this node's bbox
        crop = image_rgb[y:y+h, x:x+w].copy()

        # Create a sub-mask generator with adjusted settings for finer detail
        sub_points = max(8, min(32, int(max(w, h) / 32)))
        sub_generator = SamAutomaticMaskGenerator(
            model=self.sam,
            points_per_side=sub_points,
            points_per_batch=min(256, self.refine_points_per_batch),
            pred_iou_thresh=0.75,
            box_nms_thresh=0.7,
            stability_score_thresh=0.88,
            crop_n_layers=0,
            crop_n_points_downscale_factor=1,
            min_mask_region_area=max(self.refine_min_area, 32),
        )

        try:
            # LangSplat SAM returns (default, s, m, l) tuples, not a flat list
            result = sub_generator.generate(crop)
            if isinstance(result, tuple) and len(result) == 4:
                sub_masks = []
                for group in result:
                    if isinstance(group, list):
                        sub_masks.extend(group)
            else:
                sub_masks = result
        except (torch.cuda.OutOfMemoryError, RuntimeError):
            torch.cuda.empty_cache()
            return

        if not sub_masks:
            return

        # Filter sub-masks: must be strictly smaller than parent
        parent_area = node.area
        valid_sub_masks = []
        for sm in sub_masks:
            sm_seg = sm['segmentation'].astype(bool)
            sm_area = int(sm_seg.sum())
            if sm_area < self.refine_min_area:
                continue
            if sm_area >= 0.95 * parent_area:
                continue  # nearly the same as parent, skip

            # Check containment within parent mask (in crop coordinates)
            parent_crop_mask = node.mask[y:y+h, x:x+w]
            overlap = np.logical_and(sm_seg, parent_crop_mask).sum()
            if float(overlap) / max(sm_area, 1) < 0.5:
                continue  # not significantly inside parent

            valid_sub_masks.append(sm)

        if not valid_sub_masks:
            return

        # Apply NMS to valid sub-masks
        if len(valid_sub_masks) > 1:
            seg_pred = torch.from_numpy(
                np.stack([m['segmentation'] for m in valid_sub_masks], axis=0)
            ).cuda()
            iou_pred = torch.tensor(
                [m['predicted_iou'] for m in valid_sub_masks], device='cuda'
            )
            stability = torch.tensor(
                [m['stability_score'] for m in valid_sub_masks], device='cuda'
            )
            scores = stability * iou_pred
            keep_idx = mask_nms(seg_pred, scores, iou_thr=0.7, score_thr=0.5, inner_thr=0.3)
            valid_sub_masks = filter_masks(keep_idx, valid_sub_masks)

        # Check if we already have children (from containment-based hierarchy)
        # Only add sub-masks that don't duplicate existing children
        existing_child_masks = [c.mask for c in node.children if c.mask is not None]

        H_full, W_full = image_rgb.shape[:2]
        new_children = []
        for sm in valid_sub_masks:
            sm_seg_crop = sm['segmentation'].astype(bool)
            sm_area = int(sm_seg_crop.sum())

            # Convert crop-local mask to full-image mask
            full_mask = np.zeros((H_full, W_full), dtype=bool)
            full_mask[y:y+h, x:x+w] = sm_seg_crop

            # Check if this duplicates an existing child (IoU > 0.6)
            is_dup = False
            for existing in existing_child_masks:
                overlap = np.logical_and(full_mask, existing).sum()
                union = np.logical_or(full_mask, existing).sum()
                if union > 0 and float(overlap) / float(union) > 0.6:
                    is_dup = True
                    break
            if is_dup:
                continue

            sx, sy, sw, sh = sm['bbox']
            nid = self._next_id()
            child_node = SegNode(
                node_id=nid,
                parent_id=node.node_id,
                depth=current_depth + 1,
                label=f"{node.label}_ref_{len(new_children)}",
                mask=full_mask,
                bbox=[int(sx) + x, int(sy) + y, int(sw), int(sh)],
                area=sm_area,
                score=float(sm.get('predicted_iou', 0)) * float(sm.get('stability_score', 0)),
            )
            new_children.append(child_node)
            existing_child_masks.append(full_mask)

        node.children.extend(new_children)

        # Recurse into all children (both existing and new)
        for child in node.children:
            self._refine_node(child, image_rgb, current_depth + 1)

    def recursive_refine(self, roots: List[SegNode], image_rgb: np.ndarray):
        """Recursively refine all nodes in the tree."""
        for root in roots:
            self._refine_node(root, image_rgb, root.depth)

    # ----- Phase 4: Extract CLIP features -----

    def _collect_all_nodes(self, roots: List[SegNode]) -> List[SegNode]:
        """Flatten the tree into a list (depth-first)."""
        result = []
        for node in roots:
            result.append(node)
            result.extend(self._collect_all_nodes(node.children))
        return result

    def extract_clip_features(self, image_np: np.ndarray, roots: List[SegNode]):
        """
        Extract CLIP features for every node in the hierarchy.
        image_np should be BGR (matching LangSplat convention).
        """
        if self.clip_model is None:
            return

        all_nodes = self._collect_all_nodes(roots)
        if not all_nodes:
            return

        # Prepare tiles: crop + mask + pad + resize to 224×224
        tiles = []
        valid_nodes = []
        image_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)

        for node in all_nodes:
            if node.mask is None or node.bbox is None:
                continue
            x, y, w, h = node.bbox
            if w == 0 or h == 0:
                continue

            # Extract masked crop
            crop = image_rgb.copy()
            crop[~node.mask] = 0
            seg_crop = crop[y:y+h, x:x+w]
            if seg_crop.shape[0] == 0 or seg_crop.shape[1] == 0:
                continue

            tile = pad_and_resize_gpu(seg_crop)  # 3×224×224 on GPU
            tiles.append(tile)
            valid_nodes.append(node)

        if not tiles:
            return

        # Batch CLIP encoding
        BATCH_SIZE = 64
        all_embeds = []
        for i in range(0, len(tiles), BATCH_SIZE):
            batch = torch.stack(tiles[i:i+BATCH_SIZE], dim=0)
            with torch.no_grad():
                embeds = self.clip_model.encode_image(batch)
            embeds = embeds / embeds.norm(dim=-1, keepdim=True)
            all_embeds.append(embeds.detach().cpu().half())

        all_embeds = torch.cat(all_embeds, dim=0)

        for node, embed in zip(valid_nodes, all_embeds):
            node.clip_feature = embed.numpy()

    # ----- Phase 5: Build LangSplat-compatible outputs -----

    def build_langsplat_outputs(
        self,
        roots: List[SegNode],
        image_shape: Tuple[int, int],
    ) -> Tuple[np.ndarray, np.ndarray, List[dict]]:
        """
        Build outputs compatible with LangSplat's autoencoder training:
          - seg_maps: (N_levels × H × W) int32 arrays with per-pixel mask indices
          - features: (N_total × embed_dim) float16 array
          - hierarchy: list of dicts for JSON serialisation

        We group masks by depth level to create the seg_maps layers.
        Each layer corresponds to one depth level in the hierarchy.
        """
        H, W = image_shape
        all_nodes = self._collect_all_nodes(roots)

        if not all_nodes:
            return np.zeros((1, H, W), dtype=np.int32) - 1, np.zeros((0, 512), dtype=np.float16), []

        # Group nodes by depth
        max_depth_found = max(n.depth for n in all_nodes) if all_nodes else 0
        depth_groups: Dict[int, List[SegNode]] = {}
        for node in all_nodes:
            depth_groups.setdefault(node.depth, []).append(node)

        # Build seg_maps: one layer per depth level
        num_levels = max_depth_found + 1
        seg_maps = np.full((num_levels, H, W), -1, dtype=np.int32)

        # Assign global feature indices and build seg_map layers
        feature_list = []
        global_idx = 0
        node_to_global_idx: Dict[int, int] = {}

        for depth in range(num_levels):
            nodes_at_depth = depth_groups.get(depth, [])
            local_idx = 0
            for node in nodes_at_depth:
                if node.clip_feature is not None and node.mask is not None:
                    node_to_global_idx[node.node_id] = global_idx
                    seg_maps[depth][node.mask] = global_idx
                    feature_list.append(node.clip_feature)
                    global_idx += 1
                    local_idx += 1

        if feature_list:
            features = np.stack(feature_list, axis=0)
        else:
            features = np.zeros((0, 512), dtype=np.float16)

        # Build hierarchy JSON
        hierarchy = [n.to_dict() for n in roots]

        return seg_maps, features, hierarchy

    # ----- Full pipeline -----

    def process_image(
        self,
        image_bgr: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, List[dict]]:
        """
        Full pipeline for one image:
          1. Generate multi-scale SAM masks
          2. Build hierarchy from containment
          3. Recursively refine
          4. Extract CLIP features
          5. Build LangSplat-compatible outputs

        Args:
            image_bgr: BGR image (H × W × 3)

        Returns:
            (seg_maps, features, hierarchy_dicts)
        """
        H, W = image_bgr.shape[:2]
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        # Phase 1: Generate multi-scale masks
        print("    [Phase 1] Generating multi-scale SAM masks...")
        masks_default, masks_s, masks_m, masks_l = self.generate_multi_scale_masks(image_bgr)

        # Pool all masks from all scales
        all_masks = list(masks_default)
        all_masks.extend(masks_s)
        all_masks.extend(masks_m)
        all_masks.extend(masks_l)
        print(f"    Got {len(all_masks)} masks total "
              f"(default={len(masks_default)}, s={len(masks_s)}, "
              f"m={len(masks_m)}, l={len(masks_l)})")

        if not all_masks:
            print("    [WARN] No masks generated!")
            return (
                np.full((1, H, W), -1, dtype=np.int32),
                np.zeros((0, 512), dtype=np.float16),
                [],
            )

        # Phase 2: Build hierarchy from containment
        print("    [Phase 2] Building hierarchy from containment relationships...")
        self._node_counter = 0
        roots = self.build_hierarchy(all_masks, (H, W))
        n_total = len(self._collect_all_nodes(roots))
        print(f"    Built tree: {len(roots)} roots, {n_total} total nodes")

        # Phase 3: Recursive refinement
        print(f"    [Phase 3] Recursive refinement (max_depth={self.max_depth})...")
        self.recursive_refine(roots, image_rgb)
        n_total_after = len(self._collect_all_nodes(roots))
        max_d = max((n.depth for n in self._collect_all_nodes(roots)), default=0)
        print(f"    After refinement: {n_total_after} total nodes, max depth={max_d}")

        # Phase 4: Extract CLIP features
        print("    [Phase 4] Extracting CLIP features...")
        self.extract_clip_features(image_bgr, roots)

        # Phase 5: Build outputs
        print("    [Phase 5] Building outputs...")
        seg_maps, features, hierarchy = self.build_langsplat_outputs(roots, (H, W))
        print(f"    Output: seg_maps shape={seg_maps.shape}, "
              f"features shape={features.shape}, "
              f"hierarchy depth levels={seg_maps.shape[0]}")

        return seg_maps, features, hierarchy


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------

def seed_everything(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True


def load_and_resize(image_path, resolution=-1):
    """Load and optionally downscale an image.
    
    With resolution=-1, auto-caps at 1080p (matching LangSplat behavior).
    With resolution=1, uses native resolution (SAM internally downscales).
    With resolution=N, scales so width=N pixels.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Cannot read image: {image_path}")
    orig_h, orig_w = image.shape[:2]
    if resolution == 1:
        # Native resolution — SAM will auto-downscale internally
        return image
    elif resolution == -1:
        global_down = 1
        if orig_h > 1080:
            global_down = orig_h / 1080
    else:
        global_down = orig_w / resolution
    scale = float(global_down)
    if scale > 1.0:
        new_size = (int(orig_w / scale), int(orig_h / scale))
        image = cv2.resize(image, new_size)
    return image


def save_outputs(save_path, seg_maps, features, hierarchy):
    """Save outputs: _s.npy, _f.npy, _hierarchy.json."""
    np.save(save_path + '_s.npy', seg_maps)
    np.save(save_path + '_f.npy', features)
    with open(save_path + '_hierarchy.json', 'w') as f:
        json.dump(hierarchy, f, indent=2)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    seed_everything(42)

    parser = argparse.ArgumentParser(
        description="Hierarchical SAM Preprocessing with CLIP Features"
    )
    parser.add_argument('--dataset_path', type=str, required=True,
                        help='Path to the dataset directory')
    parser.add_argument('--resolution', type=int, default=-1,
                        help='Target resolution (-1 for auto)')
    parser.add_argument('--downscale', type=int, default=1, choices=[1, 2, 4, 8],
                        help='Downscale factor for image folder selection')
    parser.add_argument('--sam_ckpt_path', type=str, default="ckpts/sam_vit_h_4b8939.pth",
                        help='Path to SAM checkpoint')
    parser.add_argument('--sam_model_type', type=str, default="vit_h",
                        choices=["vit_h", "vit_l", "vit_b"],
                        help='SAM model variant')
    parser.add_argument('--max_depth', type=int, default=10,
                        help='Maximum recursion depth for hierarchy refinement')
    parser.add_argument('--min_mask_area', type=int, default=100,
                        help='Minimum mask area in pixels')
    parser.add_argument('--containment_threshold', type=float, default=0.70,
                        help='Fraction of child pixels in parent to assign parent-child')
    parser.add_argument('--skip_clip', action='store_true',
                        help='Skip CLIP feature extraction (faster, no _f.npy output)')
    parser.add_argument('--skip_refinement', action='store_true',
                        help='Skip recursive refinement (only containment-based hierarchy)')
    args = parser.parse_args()

    torch.set_default_dtype(torch.float32)

    # Select image folder
    if args.downscale == 1:
        img_folder = os.path.join(args.dataset_path, 'images')
    else:
        img_folder = os.path.join(args.dataset_path, f'images_{args.downscale}')
    assert os.path.isdir(img_folder), f"Image folder not found: {img_folder}"
    print(f"[ INFO ] Using image folder: {img_folder}")

    data_list = sorted(os.listdir(img_folder))
    data_list = [f for f in data_list if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'))]
    print(f"[ INFO ] Found {len(data_list)} images")

    # Output folder
    save_folder = os.path.join(args.dataset_path, 'language_features_hierarchical')
    os.makedirs(save_folder, exist_ok=True)

    # Load models
    clip_model = None
    if not args.skip_clip:
        print("[ INFO ] Loading OpenCLIP model...")
        clip_model = OpenCLIPNetwork(OpenCLIPNetworkConfig)

    print("[ INFO ] Loading SAM model...")
    processor = HierarchicalSAMProcessor(
        sam_checkpoint=args.sam_ckpt_path,
        sam_model_type=args.sam_model_type,
        clip_model=clip_model,
        max_depth=args.max_depth if not args.skip_refinement else 0,
        min_mask_area=args.min_mask_area,
        containment_threshold=args.containment_threshold,
    )

    # Filter already-processed images for resume support
    to_process = []
    skipped = 0
    for data_path in data_list:
        save_path = os.path.join(save_folder, data_path.split('.')[0])
        if (os.path.exists(save_path + '_s.npy') and
            os.path.exists(save_path + '_hierarchy.json') and
            (args.skip_clip or os.path.exists(save_path + '_f.npy'))):
            skipped += 1
        else:
            to_process.append(data_path)
    if skipped > 0:
        print(f"[ INFO ] Skipping {skipped} already-processed images")
    print(f"[ INFO ] Processing {len(to_process)} images...")

    # Process images
    for data_path in tqdm(to_process, desc="Processing images"):
        image_path = os.path.join(img_folder, data_path)
        try:
            image_bgr = load_and_resize(image_path, args.resolution)
        except Exception as e:
            print(f"\n  [ERROR] Failed to load {data_path}: {e}")
            continue

        print(f"\n  Processing: {data_path} ({image_bgr.shape[1]}×{image_bgr.shape[0]})")

        try:
            seg_maps, features, hierarchy = processor.process_image(image_bgr)
        except Exception as e:
            print(f"  [ERROR] Failed to process {data_path}: {e}")
            import traceback
            traceback.print_exc()
            continue

        # Save
        save_path = os.path.join(save_folder, data_path.split('.')[0])
        save_outputs(save_path, seg_maps, features, hierarchy)

        # Memory cleanup
        del seg_maps, features, hierarchy
        torch.cuda.empty_cache()

    print(f"\n[ INFO ] Done. Hierarchical features saved to: {save_folder}")


if __name__ == '__main__':
    main()
