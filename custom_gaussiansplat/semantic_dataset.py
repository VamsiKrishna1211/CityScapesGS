from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from dataset import ColmapDataset


class SemanticColmapDataset(Dataset):
    """Semantic dataset that reuses ColmapDataset for camera/image/depth.

    This wrapper keeps semantic supervision fully decoupled from ColmapDataset.
    """

    def __init__(
        self,
        base_dataset: ColmapDataset,
        semantics_path: Optional[Path] = None,
        semantics_resolution: Optional[tuple[int, int]] = None,
    ):
        self.base_dataset = base_dataset
        self.semantics_path = Path(semantics_path) if semantics_path is not None else None
        self.semantics_resolution = semantics_resolution

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int) -> Dict:
        cam, gt_image, depth_tensor = self.base_dataset[idx]
        image_path = Path(cam["image_path"])
        image_id = str(image_path)

        semantic_tensor = None
        if self.semantics_path is not None:
            semantic_tensor = self._load_semantic_tensor(image_path=image_path, image=gt_image)

        return {
            "cam": cam,
            "gt_image": gt_image,
            "depth_tensor": depth_tensor,
            "semantic_tensor": semantic_tensor,
            "image_id": image_id,
        }

    def _load_semantic_tensor(self, image_path: Path, image: torch.Tensor) -> torch.Tensor:
        if self.semantics_path is None:
            raise RuntimeError("semantics_path is not configured for SemanticColmapDataset")
        semantics_file = self.semantics_path / f"{image_path.stem}_s.npy"
        if not semantics_file.exists():
            raise FileNotFoundError(f"Semantic features file not found for {image_path.name}: {semantics_file}")

        semantic_tensor = torch.from_numpy(np.load(semantics_file)).float()

        if self.semantics_resolution is not None:
            target_h, target_w = self.semantics_resolution
            if semantic_tensor.dim() == 3 and semantic_tensor.shape[0] <= 32:
                # [C, H, W] -> resize -> [C, H, W]
                c, h, w = semantic_tensor.shape
                if (h, w) != (target_h, target_w):
                    sem_hwc = semantic_tensor.permute(1, 2, 0).cpu().numpy()
                    sem_hwc = cv2.resize(sem_hwc, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
                    semantic_tensor = torch.from_numpy(sem_hwc).permute(2, 0, 1).float()
            elif semantic_tensor.dim() == 3:
                # [H, W, C]
                h, w, _ = semantic_tensor.shape
                if (h, w) != (target_h, target_w):
                    sem_hwc = semantic_tensor.cpu().numpy()
                    sem_hwc = cv2.resize(sem_hwc, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
                    semantic_tensor = torch.from_numpy(sem_hwc).float()

        # Optional RGB resize parity with semantic tensor can be enabled later.
        _ = image
        return semantic_tensor

    @staticmethod
    def collate_fn(batch):
        return batch[0]
