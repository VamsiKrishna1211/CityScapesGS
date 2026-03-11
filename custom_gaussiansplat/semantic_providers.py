from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch


class SemanticTargetProvider(ABC):
    """Interface for semantic supervision target providers."""

    def __init__(self, semantics_dim: int, cache_enabled: bool = False):
        self.semantics_dim = semantics_dim
        self.cache_enabled = cache_enabled
        self._cache: Dict[str, torch.Tensor] = {}

    def get_target(
        self,
        image_id: str,
        gt_image: torch.Tensor,
        dataset_semantic_tensor: Optional[torch.Tensor],
        device: torch.device,
    ) -> torch.Tensor:
        if self.cache_enabled and image_id in self._cache:
            return self._cache[image_id].to(device=device)

        target = self._compute_target(image_id=image_id, gt_image=gt_image, dataset_semantic_tensor=dataset_semantic_tensor, device=device)
        target = self._to_hwc(target)

        if target.shape[-1] != self.semantics_dim:
            raise RuntimeError(
                f"Semantic channel mismatch for '{image_id}': provider returned {target.shape[-1]} channels but semantics_dim={self.semantics_dim}"
            )

        if self.cache_enabled:
            self._cache[image_id] = target.detach().cpu()

        return target

    @abstractmethod
    def _compute_target(
        self,
        image_id: str,
        gt_image: torch.Tensor,
        dataset_semantic_tensor: Optional[torch.Tensor],
        device: torch.device,
    ) -> torch.Tensor:
        raise NotImplementedError

    @staticmethod
    def _to_hwc(tensor: torch.Tensor) -> torch.Tensor:
        if tensor.dim() == 4 and tensor.shape[0] == 1:
            tensor = tensor.squeeze(0)

        if tensor.dim() == 2:
            tensor = tensor.unsqueeze(-1)

        if tensor.dim() == 3 and tensor.shape[0] <= 2048 and tensor.shape[-1] > 2048:
            # Heuristic safety branch for malformed tensors.
            tensor = tensor.permute(1, 2, 0)

        if tensor.dim() == 3 and tensor.shape[0] <= 32 and tensor.shape[-1] > 32:
            # Common case: C,H,W -> H,W,C
            tensor = tensor.permute(1, 2, 0)

        if tensor.dim() != 3:
            raise RuntimeError(f"Expected semantic tensor rank 3 after normalization, got shape {tuple(tensor.shape)}")

        return tensor.float()


class NpySemanticProvider(SemanticTargetProvider):
    """Uses dataset-loaded semantic tensors from npy files."""

    def _compute_target(
        self,
        image_id: str,
        gt_image: torch.Tensor,
        dataset_semantic_tensor: Optional[torch.Tensor],
        device: torch.device,
    ) -> torch.Tensor:
        if dataset_semantic_tensor is None:
            raise RuntimeError(
                f"Semantic tensor missing for '{image_id}'. Provide npy files or choose --semantic-provider runtime."
            )
        return dataset_semantic_tensor.to(device=device, dtype=torch.float32)


class RuntimeModelSemanticProvider(SemanticTargetProvider):
    """Runs a runtime model to infer semantic targets directly from RGB images."""

    def __init__(self, semantics_dim: int, model_path: Path, cache_enabled: bool = False):
        super().__init__(semantics_dim=semantics_dim, cache_enabled=cache_enabled)
        self.model_path = model_path
        self._model = None

    def _load_model(self, device: torch.device):
        if self._model is not None:
            return

        if self.model_path.suffix in {".pt", ".pth", ".jit", ".ts"}:
            try:
                self._model = torch.jit.load(str(self.model_path), map_location=device)
            except Exception:
                loaded = torch.load(str(self.model_path), map_location=device, weights_only=False)
                self._model = loaded
        else:
            loaded = torch.load(str(self.model_path), map_location=device, weights_only=False)
            self._model = loaded

        if hasattr(self._model, "to"):
            self._model = self._model.to(device)
        if hasattr(self._model, "eval"):
            self._model.eval()

        if not callable(self._model):
            raise RuntimeError(
                f"Runtime semantic model at '{self.model_path}' is not callable. Expected torchscript or callable nn.Module."
            )

    def _compute_target(
        self,
        image_id: str,
        gt_image: torch.Tensor,
        dataset_semantic_tensor: Optional[torch.Tensor],
        device: torch.device,
    ) -> torch.Tensor:
        self._load_model(device=device)
        model = self._model
        if model is None or not callable(model):
            raise RuntimeError("Runtime semantic model failed to initialize as a callable model")

        image = gt_image
        if image.dim() == 4 and image.shape[0] == 1:
            image = image[0]

        if image.dim() != 3:
            raise RuntimeError(f"Runtime semantic provider expects [H,W,C] image, got {tuple(image.shape)}")

        image = image.to(device=device, dtype=torch.float32)
        image_bchw = image.permute(2, 0, 1).unsqueeze(0)

        with torch.no_grad():
            pred = model(image_bchw)

        if isinstance(pred, (tuple, list)):
            pred = pred[0]

        if not isinstance(pred, torch.Tensor):
            pred = torch.tensor(np.asarray(pred), device=device)

        return pred.to(device=device, dtype=torch.float32)


def build_semantic_provider(
    provider_name: str,
    semantics_dim: int,
    cache_enabled: bool,
    semantic_model_path: Optional[Path],
) -> SemanticTargetProvider:
    provider_name = provider_name.lower()
    if provider_name == "npy":
        return NpySemanticProvider(semantics_dim=semantics_dim, cache_enabled=cache_enabled)
    if provider_name == "runtime":
        if semantic_model_path is None:
            raise ValueError("semantic_model_path is required for runtime provider")
        return RuntimeModelSemanticProvider(
            semantics_dim=semantics_dim,
            model_path=semantic_model_path,
            cache_enabled=cache_enabled,
        )

    raise ValueError(f"Unsupported semantic provider: {provider_name}")
