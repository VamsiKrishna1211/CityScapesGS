"""Semantic supervision target providers for Gaussian Splatting training.

Providers convert dataset-specific semantic data (npy files, runtime models,
patch features) into a common SemanticTarget that the unified SemanticTrainer
consumes without knowing the source format.

Provider selection guide
------------------------
``"npy"`` Dense per-pixel features [H, W, D] loaded from .npy files.
Use for: pre-extracted LangSplat CLIP maps, any dense feature grid.
``"dino"`` DINOv2 patch features [H//p, W//p, D] saved as *_dino.npy.
scale_factor=1/patch_size tells the trainer to downsample the
rendered semantic map to the patch grid before computing loss.
``"dino_encoded"`` Compressed DINO features from the patch autoencoder.
[He/p, We/p, bottleneck] loaded from {stem}.pt (bottleneck-dim).
``"global_clip"`` Single per-image CLIP embedding [D] saved as *_clip_global.npy.
Rendered map is mean-pooled over HxW before loss.
``"runtime"`` On-the-fly inference from a TorchScript / nn.Module checkpoint.

SemanticTarget.mode controls how the trainer computes the loss:
"dense" -- pixel-wise MSE at full image resolution.
"patch" -- rendered map downsampled by scale_factor, then MSE on patch grid.
"global" -- rendered map mean-pooled [1, 1, D], then MSE vs provider feature.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# SemanticTarget -- typed return value shared by all providers
# ---------------------------------------------------------------------------


@dataclass
class SemanticTarget:
    """Typed output from SemanticTargetProvider.get_target().

    Attributes:
    features: [H', W', D] target tensor on the correct device.
              For ``mode="global"`` this is [1, 1, D].
    mode: Aggregation mode for loss computation:
          ``"dense"`` pixel-wise, ``"patch"`` downsampled grid,
          ``"global"`` mean-pooled scalar.
    scale_factor: Spatial scale relative to rendered image resolution.
                  ``1.0`` -> target is full-resolution.
                  ``< 1.0`` -> DINO patch mode; trainer downsamples rendered map.
    """
    features: torch.Tensor
    mode: str = "dense"  # "dense" | "patch" | "global"
    scale_factor: float = 1.0


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class SemanticTargetProvider(ABC):
    """Interface for semantic supervision target providers.

    Subclasses implement ``_compute_target``. The public ``get_target`` method
    handles caching, ``_to_hwc`` normalisation, and channel validation.
    """

    def __init__(self, semantics_dim: int, cache_enabled: bool = False):
        self.semantics_dim = semantics_dim
        self.cache_enabled = cache_enabled
        self._cache: Dict[str, SemanticTarget] = {}

    def get_target(
        self,
        image_id: str,
        gt_image: torch.Tensor,
        dataset_semantic_tensor: Optional[torch.Tensor],
        device: torch.device,
    ) -> SemanticTarget:
        if self.cache_enabled and image_id in self._cache:
            entry = self._cache[image_id]
            return SemanticTarget(
                features=entry.features.to(device=device),
                mode=entry.mode,
                scale_factor=entry.scale_factor,
            )

        target = self._compute_target(
            image_id=image_id,
            gt_image=gt_image,
            dataset_semantic_tensor=dataset_semantic_tensor,
            device=device,
        )
        features = self._to_hwc(target.features)

        if features.shape[-1] != self.semantics_dim:
            raise RuntimeError(
                f"Semantic channel mismatch for '{image_id}': provider returned "
                f"{features.shape[-1]} channels, expected semantics_dim={self.semantics_dim}"
            )

        result = SemanticTarget(features=features, mode=target.mode, scale_factor=target.scale_factor)

        if self.cache_enabled:
            self._cache[image_id] = SemanticTarget(
                features=features.detach().cpu(),
                mode=target.mode,
                scale_factor=target.scale_factor,
            )

        return result

    @abstractmethod
    def _compute_target(
        self,
        image_id: str,
        gt_image: torch.Tensor,
        dataset_semantic_tensor: Optional[torch.Tensor],
        device: torch.device,
    ) -> SemanticTarget:
        raise NotImplementedError

    @staticmethod
    def _to_hwc(tensor: torch.Tensor) -> torch.Tensor:
        """Normalise arbitrary rank-3/4 tensors to [H, W, C] layout."""
        if tensor.dim() == 4 and tensor.shape[0] == 1:
            tensor = tensor.squeeze(0)
        if tensor.dim() == 2:
            tensor = tensor.unsqueeze(-1)
        # Heuristic: if first dim is small and last dim is large -> C,H,W
        if tensor.dim() == 3 and tensor.shape[0] <= 2048 and tensor.shape[-1] > 2048:
            tensor = tensor.permute(1, 2, 0)
        if tensor.dim() == 3 and tensor.shape[0] <= 32 and tensor.shape[-1] > 32:
            tensor = tensor.permute(1, 2, 0)
        if tensor.dim() != 3:
            raise RuntimeError(
                f"Expected semantic tensor rank 3 after normalisation, got {tuple(tensor.shape)}"
            )
        return tensor.float()


# ---------------------------------------------------------------------------
# NpySemanticProvider -- dense per-pixel [H, W, D] from dataset npy files
# ---------------------------------------------------------------------------


class NpySemanticProvider(SemanticTargetProvider):
    """Uses dataset-loaded semantic tensors from pre-extracted npy files.

    Expects ``[H, W, D]`` tensors already loaded by SemanticColmapDataset.
    ``scale_factor=1.0`` -- no spatial resizing needed at loss time.
    """

    def _compute_target(
        self,
        image_id: str,
        gt_image: torch.Tensor,
        dataset_semantic_tensor: Optional[torch.Tensor],
        device: torch.device,
    ) -> SemanticTarget:
        if dataset_semantic_tensor is None:
            raise RuntimeError(
                f"Semantic tensor missing for '{image_id}'. "
                "Provide npy files or choose --semantic-provider runtime."
            )
        return SemanticTarget(
            features=dataset_semantic_tensor.to(device=device, dtype=torch.float32),
            mode="dense",
            scale_factor=1.0,
        )


# ---------------------------------------------------------------------------
# DINOPatchProvider -- per-patch features from ViT spatial output
# ---------------------------------------------------------------------------


class DINOPatchProvider(SemanticTargetProvider):
    """Loads DINOv2 patch features saved as ``{stem}_dino.npy`` of shape [H//p, W//p, D].

    The rendered semantic map lives at full image resolution; the trainer
    bilinearly downsamples it by ``scale_factor = 1/patch_size`` to the patch
    grid before computing MSE -- preserving DINO's spatial structure rather
    than blurring it by upsampling the other way.

    Args:
    feature_dir: Directory containing ``*_dino.npy`` files.
    patch_size: ViT patch stride in pixels (14 for DINOv2 ViT-14).
    semantics_dim: Feature dimension D (768 for ViT-B/14, 1024 for ViT-L/14).
    cache_enabled: Cache loaded tensors in RAM.
    """

    def __init__(
        self,
        feature_dir: Path,
        patch_size: int = 14,
        semantics_dim: int = 768,
        cache_enabled: bool = False,
    ) -> None:
        super().__init__(semantics_dim=semantics_dim, cache_enabled=cache_enabled)
        self.feature_dir = Path(feature_dir)
        self.patch_size = patch_size

    def _compute_target(
        self,
        image_id: str,
        gt_image: torch.Tensor,
        dataset_semantic_tensor: Optional[torch.Tensor],
        device: torch.device,
    ) -> SemanticTarget:
        stem = Path(image_id).stem
        npy_path = self.feature_dir / f"{stem}_dino.npy"
        if not npy_path.exists():
            raise FileNotFoundError(
                f"DINO feature file not found: {npy_path}. "
                "Extract DINOv2 patch features and save as '<stem>_dino.npy'."
            )
        arr = np.load(str(npy_path)).astype(np.float32)  # [H//p, W//p, D]
        if arr.ndim == 2:
            arr = arr[..., np.newaxis]
        return SemanticTarget(
            features=torch.from_numpy(arr).to(device=device),
            mode="patch",
            scale_factor=1.0 / self.patch_size,
        )


# ---------------------------------------------------------------------------
# DINOEncodedProvider -- compressed DINO features from patch autoencoder
# ---------------------------------------------------------------------------


class DINOEncodedProvider(SemanticTargetProvider):
    """Loads compressed DINO features from the patch autoencoder output.

    Each encoded ``.pt`` file is a ``torch.save`` with keys:
    - ``features``: [N, bottleneck] compressed patch features (L2-normed)
    - ``patch_size``: int -- ViT patch stride
    - ``inference_image_shape``: (H, W) -- full image dims at inference time

    The [N, bottleneck] array is reshaped to a 2D grid using the patch stride:
    ``H_patch = inference_image_shape[0] // patch_size``
    and returned as ``SemanticTarget(mode="patch", scale_factor=1/patch_size)``.

    Args:
    encoded_dir: Directory containing {stem}.pt encoded feature files.
    semantics_dim: Expected bottleneck dimension (validated against the first file).
    cache_enabled: Cache loaded tensors in RAM.
    """

    def __init__(
        self,
        encoded_dir: Path,
        semantics_dim: int,
        cache_enabled: bool = False,
    ) -> None:
        super().__init__(semantics_dim=semantics_dim, cache_enabled=cache_enabled)
        self.encoded_dir = Path(encoded_dir)
        self._bottleneck: Optional[int] = None

    def _load_encoded(self, stem: str, device: torch.device) -> SemanticTarget:
        pt_path = self.encoded_dir / f"{stem}.pt"
        if not pt_path.exists():
            raise FileNotFoundError(
                f"Encoded DINO feature file not found: {pt_path}. "
                "Run train_patch_encoder.py with --encode_only to produce these files."
            )
        data = torch.load(pt_path, map_location="cpu", weights_only=True)
        features = data["features"][1:].float()  # [N, bottleneck]
        patch_size: int = int(data["patch_size"])
        H_img, W_img = data["inference_image_shape"]
        bottleneck = features.shape[-1] # Ignoring the first token (CLS) which is not used for supervision.

        if self._bottleneck is None:
            self._bottleneck = bottleneck
            if bottleneck != self.semantics_dim:
                raise RuntimeError(
                    f"Bottleneck mismatch: provider was constructed with "
                    f"semantics_dim={self.semantics_dim} but encoded file has "
                    f"bottleneck={bottleneck}. "
                    f"Pass --semantics-dim {bottleneck} (and --lang-feat-dim for scaffold models)."
                )

        H_p = H_img // patch_size
        W_p = W_img // patch_size
        expected_N = H_p * W_p
        actual_N = features.shape[0]
        if actual_N != expected_N:
            raise RuntimeError(
                f"Patch count mismatch in {pt_path}: file has {actual_N} patches "
                f"but ({H_img}, {W_img}) / patch_size={patch_size} implies {expected_N}. "
                f"Check that the patch_size matches the encoder."
            )

        grid = features.view(H_p, W_p, bottleneck)  # [H_p, W_p, bottleneck]
        return SemanticTarget(
            features=grid.to(device=device, dtype=torch.float32),
            mode="patch",
            scale_factor=1.0 / patch_size,
        )

    def _compute_target(
        self,
        image_id: str,
        gt_image: torch.Tensor,
        dataset_semantic_tensor: Optional[torch.Tensor],
        device: torch.device,
    ) -> SemanticTarget:
        stem = Path(image_id).stem
        return self._load_encoded(stem, device)

    def get_target(
        self,
        image_id: str,
        gt_image: torch.Tensor,
        dataset_semantic_tensor: Optional[torch.Tensor],
        device: torch.device,
    ) -> SemanticTarget:
        # Override: skip the HWC channel-count check.
        # The base class validates that features shape has channels == semantics_dim
        # at full image resolution. dino_encoded produces [H_p, W_p, bottleneck]
        # where H_p != H_img, so the validator fires incorrectly.
        if self.cache_enabled and image_id in self._cache:
            entry = self._cache[image_id]
            return SemanticTarget(
                features=entry.features.to(device=device),
                mode=entry.mode,
                scale_factor=entry.scale_factor,
            )
        target = self._compute_target(image_id, gt_image, dataset_semantic_tensor, device)
        if self.cache_enabled:
            self._cache[image_id] = SemanticTarget(
                features=target.features.detach().cpu(),
                mode=target.mode,
                scale_factor=target.scale_factor,
            )
        return target


# ---------------------------------------------------------------------------
# GlobalCLIPProvider -- single per-image embedding
# ---------------------------------------------------------------------------


class GlobalCLIPProvider(SemanticTargetProvider):
    """Loads a single global CLIP image embedding [D] per image.

    At loss time the rendered semantic map is mean-pooled over HxW and
    compared with this single vector. File naming: ``{stem}_clip_global.npy``.

    Args:
    feature_dir: Directory containing ``*_clip_global.npy`` files.
    semantics_dim: CLIP dimension (512 for ViT-B/32, 768 for ViT-L/14).
    cache_enabled: Cache loaded tensors in RAM.
    """

    def __init__(
        self,
        feature_dir: Path,
        semantics_dim: int = 512,
        cache_enabled: bool = False,
    ) -> None:
        super().__init__(semantics_dim=semantics_dim, cache_enabled=cache_enabled)
        self.feature_dir = Path(feature_dir)

    def _compute_target(
        self,
        image_id: str,
        gt_image: torch.Tensor,
        dataset_semantic_tensor: Optional[torch.Tensor],
        device: torch.device,
    ) -> SemanticTarget:
        stem = Path(image_id).stem
        npy_path = self.feature_dir / f"{stem}_clip_global.npy"
        if not npy_path.exists():
            raise FileNotFoundError(
                f"Global CLIP feature file not found: {npy_path}. "
                "Save CLIP image embeddings as '<stem>_clip_global.npy'."
            )
        arr = np.load(str(npy_path)).astype(np.float32).reshape(1, 1, -1)  # [1, 1, D]
        return SemanticTarget(
            features=torch.from_numpy(arr).to(device=device),
            mode="global",
            scale_factor=1.0,
        )

    def get_target(
        self,
        image_id: str,
        gt_image: torch.Tensor,
        dataset_semantic_tensor: Optional[torch.Tensor],
        device: torch.device,
    ) -> SemanticTarget:
        # Override to skip the HWC channel check -- [1, 1, D] is intentionally not [H, W, D].
        if self.cache_enabled and image_id in self._cache:
            entry = self._cache[image_id]
            return SemanticTarget(
                features=entry.features.to(device=device),
                mode=entry.mode,
                scale_factor=entry.scale_factor,
            )
        target = self._compute_target(image_id, gt_image, dataset_semantic_tensor, device)
        if self.cache_enabled:
            self._cache[image_id] = SemanticTarget(
                features=target.features.detach().cpu(),
                mode=target.mode,
                scale_factor=target.scale_factor,
            )
        return target


# ---------------------------------------------------------------------------
# RuntimeModelSemanticProvider -- on-the-fly inference from an RGB image
# ---------------------------------------------------------------------------


class RuntimeModelSemanticProvider(SemanticTargetProvider):
    """Runs a TorchScript or nn.Module to infer semantic targets from RGB images."""

    def __init__(self, semantics_dim: int, model_path: Path, cache_enabled: bool = False):
        super().__init__(semantics_dim=semantics_dim, cache_enabled=cache_enabled)
        self.model_path = model_path
        self._model = None

    def _load_model(self, device: torch.device) -> None:
        if self._model is not None:
            return
        if self.model_path.suffix in {".pt", ".pth", ".jit", ".ts"}:
            try:
                self._model = torch.jit.load(str(self.model_path), map_location=device)
            except Exception:
                self._model = torch.load(str(self.model_path), map_location=device, weights_only=False)
        else:
            self._model = torch.load(str(self.model_path), map_location=device, weights_only=False)
        if hasattr(self._model, "to"):
            self._model = self._model.to(device)
        if hasattr(self._model, "eval"):
            self._model.eval()
        if not callable(self._model):
            raise RuntimeError(
                f"Runtime semantic model at '{self.model_path}' is not callable. "
                "Expected TorchScript or callable nn.Module."
            )

    def _compute_target(
        self,
        image_id: str,
        gt_image: torch.Tensor,
        dataset_semantic_tensor: Optional[torch.Tensor],
        device: torch.device,
    ) -> SemanticTarget:
        self._load_model(device=device)
        image = gt_image
        if image.dim() == 4 and image.shape[0] == 1:
            image = image[0]
        if image.dim() != 3:
            raise RuntimeError(
                f"Runtime provider expects [H, W, C] image, got {tuple(image.shape)}"
            )
        image = image.to(device=device, dtype=torch.float32)
        image_bchw = image.permute(2, 0, 1).unsqueeze(0)
        with torch.no_grad():
            pred = self._model(image_bchw)
        if isinstance(pred, (tuple, list)):
            pred = pred[0]
        if not isinstance(pred, torch.Tensor):
            pred = torch.tensor(np.asarray(pred), device=device)
        return SemanticTarget(
            features=pred.to(device=device, dtype=torch.float32),
            mode="dense",
            scale_factor=1.0,
        )


# ---------------------------------------------------------------------------
# PCA utility -- shared by SemanticScaffoldModel.setup_semantic_training()
# ---------------------------------------------------------------------------


def compute_pca_from_clip_dir(
    clip_dir: Path,
    n_components: int,
    clip_dim: int = 512,
    sample_limit: int = 200_000,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute PCA over a directory of CLIP feature npy files.

    Loads ``*_s.npy`` files (shape [H, W, clip_dim]), samples up to
    ``sample_limit`` pixel vectors, and returns the top-``n_components``
    principal directions.

    Args:
    clip_dir: Directory containing ``{stem}_s.npy`` files.
    n_components: Number of components to extract (= codebook_size).
    clip_dim: CLIP feature dimensionality (default 512).
    sample_limit: Max pixel vectors used for covariance estimation.

    Returns:
    pca_components: [n_components, clip_dim] -- for codebook initialisation.
    pca_transform: [clip_dim, n_components] -- projection matrix for target compression.
    """
    log = logging.getLogger("cityscape_gs.semantic_providers")
    npy_files = sorted(clip_dir.glob("*_s.npy"))
    if not npy_files:
        raise FileNotFoundError(f"No '*_s.npy' CLIP feature files found in {clip_dir}")

    all_feats: List[np.ndarray] = []
    collected = 0
    for f in npy_files:
        arr = np.load(f).astype(np.float32)
        if arr.ndim == 3:
            arr = arr.reshape(-1, arr.shape[-1])
        if arr.shape[-1] != clip_dim:
            raise RuntimeError(f"Expected CLIP dim {clip_dim}, got {arr.shape[-1]} in {f.name}")
        remaining = sample_limit - collected
        if arr.shape[0] > remaining:
            idx = np.random.choice(arr.shape[0], remaining, replace=False)
            arr = arr[idx]
        all_feats.append(arr)
        collected += arr.shape[0]
        if collected >= sample_limit:
            break

    features = torch.from_numpy(np.concatenate(all_feats, axis=0))  # [N, clip_dim]
    log.info(
        "Computing PCA over %d CLIP pixel vectors from %d images.",
        features.shape[0],
        len(npy_files),
    )

    mean = features.mean(dim=0, keepdim=True)
    features_c = features - mean
    cov = (features_c.T @ features_c) / max(features_c.shape[0] - 1, 1)
    _, eigenvectors = torch.linalg.eigh(cov)
    top_vecs = eigenvectors[:, -n_components:].flip(dims=[1])  # [clip_dim, n_components]

    pca_components = top_vecs.T.contiguous()  # [n_components, clip_dim]
    pca_transform = top_vecs.contiguous()  # [clip_dim, n_components]
    return pca_components, pca_transform


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def build_semantic_provider(
    provider_name: str,
    semantics_dim: int,
    cache_enabled: bool,
    semantic_model_path: Optional[Path] = None,
    feature_dir: Optional[Path] = None,
    patch_size: int = 14,
) -> SemanticTargetProvider:
    """Build a SemanticTargetProvider by name.

    Args:
    provider_name: ``"npy"`` | ``"dino"`` | ``"dino_encoded"`` | ``"global_clip"`` | ``"runtime"``.
    semantics_dim: Expected channel count from the provider.
    cache_enabled: Cache target tensors in RAM.
    semantic_model_path: Runtime model file (``"runtime"`` only).
    feature_dir: Feature directory (``"dino"`` / ``"dino_encoded"`` / ``"global_clip"``).
    patch_size: ViT patch stride for ``"dino"`` provider.
    """
    name = provider_name.lower()
    if name == "npy":
        return NpySemanticProvider(semantics_dim=semantics_dim, cache_enabled=cache_enabled)

    # NOTE: "dino_encoded" uses the same feature_dir parameter as "dino".
    # The trainer wires dino_encoded_dir -> feature_dir so this works cleanly.
    if name == "dino_encoded":
        if feature_dir is None:
            raise ValueError(
                "feature_dir is required for the dino_encoded provider. "
                "Pass --dino-encoded-dir (wired as --semantics-path in the trainer)."
            )
        return DINOEncodedProvider(
            encoded_dir=feature_dir,
            semantics_dim=semantics_dim,
            cache_enabled=cache_enabled,
        )

    if name == "dino":
        if feature_dir is None:
            raise ValueError("feature_dir is required for the dino provider")
        return DINOPatchProvider(
            feature_dir=feature_dir,
            patch_size=patch_size,
            semantics_dim=semantics_dim,
            cache_enabled=cache_enabled,
        )

    if name == "global_clip":
        if feature_dir is None:
            raise ValueError("feature_dir is required for the global_clip provider")
        return GlobalCLIPProvider(
            feature_dir=feature_dir,
            semantics_dim=semantics_dim,
            cache_enabled=cache_enabled,
        )

    if name == "runtime":
        if semantic_model_path is None:
            raise ValueError("semantic_model_path is required for the runtime provider")
        return RuntimeModelSemanticProvider(
            semantics_dim=semantics_dim,
            model_path=semantic_model_path,
            cache_enabled=cache_enabled,
        )

    raise ValueError(f"Unsupported semantic provider: {provider_name!r}")
