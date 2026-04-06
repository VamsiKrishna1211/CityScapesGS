from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import imageio.v2 as imageio
import numpy as np


@dataclass(frozen=True)
class ImagePolicy:
    mode: str = "lazy"  # lazy | preload
    max_image_size: int = 128
    cache_size: int = 256


class ThumbnailStore:
    """Load and cache camera thumbnails for viewer overlays.

    In `preload` mode, thumbnails are loaded at construction. In `lazy` mode,
    thumbnails are loaded on first request and kept in an LRU cache.
    """

    def __init__(self, image_paths: list[str], policy: ImagePolicy) -> None:
        self._paths = [Path(p) for p in image_paths]
        self._policy = policy
        self._cache: "OrderedDict[int, np.ndarray]" = OrderedDict()

        if self._policy.mode == "preload":
            for idx in range(len(self._paths)):
                thumb = self._load_thumbnail(idx)
                if thumb is not None:
                    self._cache[idx] = thumb

    def get(self, idx: int) -> Optional[np.ndarray]:
        if idx in self._cache:
            self._cache.move_to_end(idx)
            return self._cache[idx]

        thumb = self._load_thumbnail(idx)
        if thumb is None:
            return None

        self._cache[idx] = thumb
        self._cache.move_to_end(idx)

        if self._policy.mode != "preload":
            while len(self._cache) > max(1, int(self._policy.cache_size)):
                self._cache.popitem(last=False)

        return thumb

    def _load_thumbnail(self, idx: int) -> Optional[np.ndarray]:
        try:
            img = imageio.imread(self._paths[idx])
        except Exception:
            return None

        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)
        elif img.ndim == 3 and img.shape[2] == 4:
            img = img[..., :3]
        elif img.ndim == 3 and img.shape[2] == 1:
            img = np.repeat(img, 3, axis=-1)

        return self._thumbnail(img, max_size=max(16, int(self._policy.max_image_size)))

    @staticmethod
    def _thumbnail(img_np: np.ndarray, max_size: int) -> np.ndarray:
        h, w = img_np.shape[:2]
        longest = max(h, w)
        if longest <= max_size:
            return img_np
        stride = int(np.ceil(longest / max_size))
        return img_np[::stride, ::stride]
