import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import cv2
import imageio.v2 as imageio
import numpy as np
import torch
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from scipy.spatial.transform import Rotation
from torch.utils.data import Dataset

# Module-level logger
logger = logging.getLogger("cityscape_gs.dataset")
console = Console()


@dataclass
class CameraData:
    """Typed camera container while staying dict-like for compatibility."""

    R: torch.Tensor
    T: torch.Tensor
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    image_path: str

    def __getitem__(self, key: str):
        return getattr(self, key)


class BaseReconstructionDataset(Dataset):
    def __init__(
        self,
        image_dir: Union[str, Path],
        device: torch.device = torch.device("cuda"),
        use_dataloader: bool = True,
        train_semantics: bool = False,
        semantics_dim: int = 3,
        semantics_path: Optional[Path] = None,
        semantics_resolution: Optional[Union[tuple[int, int], int]] = None,
        image_scale: int = 2,
        require_depth: bool = True,
    ):
        """
        Base dataset that handles common image/depth/semantics loading behavior.

        Args:
            - image_dir: Directory containing images
            - device: torch.device to load tensors onto (default: 'cuda')
            - use_dataloader: If True, workers keep data on CPU and __getitem__ moves to device
            - train_semantics: If True, also loads semantic features for each image (if available) and includes them in the dataset items.
            - semantics_dim: Dimensionality of semantic features (e.g., 3 for RGB-based semantics, 128 for CLIP-based features).
            - semantics_path: Optional path to semantic features directory (should contain .npy files named after images).
            - semantics_resolution: Optional resolution to which semantic features should be resized (if they are image-based). If None assumes semantic features are already in the correct format and dimension.
            - image_scale: Downscale factor used for training images. If image_dir points to `images`, this resolves to `images_<image_scale>` for scale > 1.
            - require_depth: If True, missing depth directory raises an error
        """
        self.device = device
        self.use_dataloader = use_dataloader
        self._preloaded_images = None
        self._preloaded_depths = None
        self.train_semantics = train_semantics
        self.require_depth = require_depth
        self.semantics_path: Optional[Path] = None
        self.semantics_dim: int = semantics_dim
        self.semantics_resolution: Optional[Union[tuple[int, int], int]] = semantics_resolution
        self.cameras: list[CameraData] = []

        # Set by subclasses.
        self.init_points = torch.empty((0, 3), dtype=torch.float32, device=self.device)
        self.init_colors = torch.empty((0, 3), dtype=torch.float32, device=self.device)
        self.scene_extent = float(1.0)

        if image_scale < 1:
            raise ValueError(f"image_scale must be >= 1, got {image_scale}")
        self.image_scale = int(image_scale)
        image_dir_path = self._resolve_image_dir(Path(image_dir), self.image_scale)
        logger.info(f"[cyan]Using image scale:[/cyan] {self.image_scale} (directory: {image_dir_path})")
        try:
            self.depth_dir = self._resolve_depth_dir(image_dir_path, self.image_scale, require_depth=self.require_depth)
            logger.info(f"[cyan]Using depth scale:[/cyan] {self.image_scale} (directory: {self.depth_dir})")
        except FileNotFoundError:
            self.depth_dir = None

        self.image_dir = image_dir_path

        if train_semantics:
            if semantics_path is None:
                raise ValueError("train_semantics is True but semantics_path is not provided. Please provide a path to the semantic features directory.")
            else:
                self.semantics_path = Path(semantics_path)
                self.semantics_dim = semantics_dim
                if isinstance(semantics_resolution, int) and semantics_resolution > 0:
                    semantics_resolution = (semantics_resolution, semantics_resolution)
                self.semantics_resolution = semantics_resolution
                logger.info(f"[cyan]Semantic features enabled:[/cyan] loading from {semantics_path}, dim={semantics_dim}, resolution={semantics_resolution}")

    @staticmethod
    def _compute_scene_extent(points_xyz: np.ndarray, scene_extent_margin: float) -> float:
        if len(points_xyz) == 0:
            return float(1.0)
        return float(np.linalg.norm(points_xyz.max(axis=0) - points_xyz.min(axis=0)) * scene_extent_margin)

    @staticmethod
    def _prune_point_cloud(
        xyz: np.ndarray,
        rgb: np.ndarray,
        point_cloud_extent_ratio: Optional[float],
    ) -> tuple[np.ndarray, np.ndarray]:
        if point_cloud_extent_ratio is None or len(xyz) == 0:
            return xyz, rgb

        scene_center = xyz.mean(axis=0)
        dists = np.linalg.norm(xyz - scene_center, axis=1)
        max_dist = float(dists.max())
        threshold = float(point_cloud_extent_ratio) * max_dist
        mask = dists <= threshold

        logger.info(
            f"[cyan]Point cloud pruned:[/cyan] kept {int(mask.sum())} / {len(xyz)} "
            f"points within {point_cloud_extent_ratio*100:.1f}% of scene extent "
            f"(threshold={threshold:.3f})"
        )
        return xyz[mask], rgb[mask]

    def _set_init_points_and_colors(
        self,
        xyz: np.ndarray,
        rgb_uint8: np.ndarray,
        scene_extent_margin: float,
        point_cloud_extent_ratio: Optional[float],
    ):
        xyz = np.asarray(xyz, dtype=np.float32)
        rgb_uint8 = np.asarray(rgb_uint8, dtype=np.float32)
        if len(xyz) == 0:
            logger.warning(
                "[yellow]Warning:[/yellow] No initialization points found. "
                "Using a single origin point with neutral color."
            )
            xyz = np.zeros((1, 3), dtype=np.float32)
            rgb_uint8 = np.full((1, 3), 127.5, dtype=np.float32)

        xyz, rgb_uint8 = self._prune_point_cloud(xyz, rgb_uint8, point_cloud_extent_ratio)
        if len(xyz) == 0:
            logger.warning(
                "[yellow]Warning:[/yellow] Point cloud pruning removed all points. "
                "Using a single origin point with neutral color."
            )
            xyz = np.zeros((1, 3), dtype=np.float32)
            rgb_uint8 = np.full((1, 3), 127.5, dtype=np.float32)

        self.scene_extent = self._compute_scene_extent(xyz, scene_extent_margin)

        self.init_points = torch.tensor(xyz, dtype=torch.float32, device=self.device)

        # SH_C0 is the 0th-order SH coefficient: 1/(2*sqrt(pi))
        sh_c0 = 0.28209479177387814
        rgb_normalized = torch.tensor(rgb_uint8, dtype=torch.float32, device=self.device) / 255.0
        self.init_colors = (rgb_normalized - 0.5) / sh_c0

    @staticmethod
    def _resolve_image_dir(image_dir: Path, image_scale: int) -> Path:
        """Resolve the image directory for the requested scale using COLMAP-style folder names."""
        image_dir = image_dir.expanduser().resolve()
        if image_scale == 1:
            if not image_dir.exists():
                raise FileNotFoundError(f"Image directory does not exist: {image_dir}")
            return image_dir

        base_name = image_dir.name
        target_dir = image_dir

        if base_name == "images":
            target_dir = image_dir.parent / f"images_{image_scale}"
        elif base_name.startswith("images_"):
            target_dir = image_dir.parent / f"images_{image_scale}"

        if not target_dir.exists():
            if base_name == "images" or base_name.startswith("images_"):
                raise FileNotFoundError(
                    f"Requested image scale {image_scale}, but directory not found: {target_dir}. "
                    "Expected COLMAP-style scaled folders (e.g., images_2, images_4)."
                )
            raise FileNotFoundError(f"Image directory does not exist: {target_dir}")

        return target_dir

    @staticmethod
    def _resolve_depth_dir(image_dir: Path, image_scale: int, require_depth: bool = True) -> Optional[Path]:
        """Resolve depth directory using the same scale convention as images."""
        scene_root = image_dir.parent
        depth_dir = scene_root / "depths_npy"
        if image_scale > 1:
            depth_dir = scene_root / f"depths_npy_{image_scale}"

        if not depth_dir.exists():
            if not require_depth:
                logger.warning(f"[yellow]Depth directory not found, depth supervision disabled:[/yellow] {depth_dir}")
                return None
            raise FileNotFoundError(
                f"Requested depth scale {image_scale}, but directory not found: {depth_dir}. "
                "Expected depth folders following image scale convention (e.g., depths_npy, depths_npy_2, depths_npy_4)."
            )

        return depth_dir

    def __len__(self):
        return len(self.cameras)

    @staticmethod
    def _image_to_rgb_tensor(img: np.ndarray) -> torch.Tensor:
        """Convert loaded image array to RGB float tensor in [0, 1]."""
        if img.ndim == 2:
            img = np.repeat(img[..., None], 3, axis=-1)
        elif img.ndim == 3 and img.shape[2] == 4:
            img = img[..., :3]
        elif img.ndim == 3 and img.shape[2] == 1:
            img = np.repeat(img, 3, axis=-1)
        return torch.from_numpy(img).float() / 255.0

    def __getitem__(self, idx):
        cam_data = self.cameras[idx]

        img_path = Path(cam_data.image_path)
        if self._preloaded_images is not None:
            img_tensor = self._preloaded_images[idx]
        else:
            img = imageio.imread(img_path)
            img_tensor = self._image_to_rgb_tensor(img)

        if self._preloaded_depths is not None:
            depth_tensor = self._preloaded_depths[idx]
        else:
            depth_tensor = self._load_depth(img_path, cam_data)
        depth_tensor = depth_tensor.to(self.device) if depth_tensor is not None else None

        semantics_output = (None, None)
        if self.train_semantics:
            semantics_output = self._load_semantics(img_path, img_tensor.detach().cpu())
        return cam_data, img_tensor.to(self.device), depth_tensor, semantics_output

    @torch.no_grad()
    def _load_semantics(self, img_path: Path, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Load semantic features for the given image if available."""
        sem_path = self.semantics_path
        if sem_path is None:
            raise RuntimeError("Semantic loading requested but semantics_path is not configured.")

        logger.debug(f"Attempting to load semantic features for {img_path.name} from {sem_path}")
        semantics_file = sem_path / f"{img_path.stem}_s.npy"
        if not semantics_file.exists():
            raise FileNotFoundError(f"Semantic features file not found for {img_path.name} at expected location: {semantics_file}")
        
        try:
            semantics = torch.tensor(np.load(semantics_file))
            semantics_image = np.asarray(image.cpu())
            semantics_image = cv2.resize(semantics_image, semantics.shape[1:], interpolation=cv2.INTER_CUBIC)
            return semantics.float(), torch.tensor(semantics_image)
        except Exception as e:
            raise RuntimeError(f"Failed to load semantic features for {img_path.name} from {semantics_file}: {e}")

    def _load_depth(self, img_path: Path, cam_data: CameraData) -> Optional[torch.Tensor]:
        """Load and validate depth map."""
        depth_dir = self.depth_dir
        if depth_dir is None:
            return None

        depth_path = depth_dir / f"{img_path.stem}.npy"
        if not depth_path.exists():
            return None

        try:
            depth_map = np.load(depth_path)

            if depth_map.shape[:2] != (cam_data.height, cam_data.width):
                return None

            if not np.isfinite(depth_map).all():
                invalid_mask = ~np.isfinite(depth_map)
                if invalid_mask.all():
                    return None
                depth_map[invalid_mask] = np.median(depth_map[~invalid_mask])

            d_min, d_max = depth_map.min(), depth_map.max()
            d_range = d_max - d_min

            if d_range < 1e-8:
                return None

            return torch.from_numpy(depth_map).float()

        except Exception as e:
            logger.debug(f"Depth load failed for {img_path.name}: {e}")
            return None

    def preload_all_data(self):
        """Pre-load all images and depth maps into RAM for faster training.

        Keeps tensors on CPU to avoid VRAM exhaustion. Missing/invalid depth maps are stored as None.
        """
        self._preloaded_images = []
        self._preloaded_depths = []
        depth_available = 0

        total_items = len(self.cameras)
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=console,
            expand=False,
        ) as progress:
            task = progress.add_task("[cyan]Preloading images + depth...", total=total_items)

            for cam_data in self.cameras:
                img_path = Path(cam_data.image_path)
                img = imageio.imread(img_path)
                img_tensor = self._image_to_rgb_tensor(img)
                preload_device = self.device if not self.use_dataloader else torch.device("cpu")
                self._preloaded_images.append(img_tensor.to(preload_device))

                depth_tensor = self._load_depth(img_path, cam_data)
                self._preloaded_depths.append(depth_tensor.to(preload_device) if depth_tensor is not None else None)
                if depth_tensor is not None:
                    depth_available += 1

                progress.update(
                    task,
                    advance=1,
                    description=f"[cyan]Preloading images + depth...[/cyan] [dim](depth: {depth_available}/{total_items})[/dim]",
                )

        logger.info(
            f"[green]✓ Preloaded {len(self.cameras)} images and {depth_available}/{len(self.cameras)} depth maps to CPU RAM[/green]"
        )

    def collate_fn(self, batch):
        """Return single item directly for batch_size=1 training."""
        return batch[0]


class ColmapDataset(BaseReconstructionDataset):
    def __init__(
        self,
        colmap_path,
        image_dir,
        device: torch.device = torch.device("cuda"),
        use_dataloader=True,
        point_cloud_extent_ratio: Optional[float] = None,
        train_semantics=False,
        semantics_dim=3,
        semantics_path: Optional[Path] = None,
        semantics_resolution: Optional[Union[tuple[int, int], int]] = None,
        scene_extent_margin: float = 2.0,
        image_scale: int = 2,
        require_depth: bool = True,
    ):
        super().__init__(
            image_dir=image_dir,
            device=device,
            use_dataloader=use_dataloader,
            train_semantics=train_semantics,
            semantics_dim=semantics_dim,
            semantics_path=semantics_path,
            semantics_resolution=semantics_resolution,
            image_scale=image_scale,
            require_depth=require_depth,
        )
        self.colmap_path = Path(colmap_path).expanduser().resolve()
        self._load_colmap(
            point_cloud_extent_ratio=point_cloud_extent_ratio,
            scene_extent_margin=scene_extent_margin,
        )

    def _load_colmap(self, point_cloud_extent_ratio: Optional[float], scene_extent_margin: float):
        try:
            import pycolmap
        except ImportError as exc:
            raise ImportError("pycolmap is required for ColmapDataset but is not installed.") from exc

        recon = pycolmap.Reconstruction(str(self.colmap_path))

        xyz = []
        rgb = []
        for p in recon.points3D.values():
            xyz.append(p.xyz)
            rgb.append(p.color)

        self._set_init_points_and_colors(
            xyz=np.array(xyz, dtype=np.float32),
            rgb_uint8=np.array(rgb, dtype=np.float32),
            scene_extent_margin=scene_extent_margin,
            point_cloud_extent_ratio=point_cloud_extent_ratio,
        )

        scale_factor = 1.0 / float(self.image_scale)
        for img in recon.images.values():
            cam = recon.cameras[img.camera_id]
            img_path = self.image_dir / img.name
            if not img_path.exists():
                continue

            cam_from_world = img.cam_from_world()
            quat = cam_from_world.rotation.quat
            r = Rotation.from_quat(quat, scalar_first=False).as_matrix()
            r_tensor = torch.tensor(r, dtype=torch.float32)
            t_tensor = torch.tensor(cam_from_world.translation, dtype=torch.float32)

            model_name = cam.model.name
            if model_name == "SIMPLE_PINHOLE":
                fx = fy = cam.params[0]
                cx, cy = cam.params[1], cam.params[2]
            elif model_name == "PINHOLE":
                fx, fy, cx, cy = cam.params[0], cam.params[1], cam.params[2], cam.params[3]
            elif model_name in ["SIMPLE_RADIAL", "RADIAL", "OPENCV"]:
                fx, fy, cx, cy = cam.params[0], cam.params[1], cam.params[2], cam.params[3]
            else:
                fx = cam.params[0] if len(cam.params) > 0 else cam.width
                fy = cam.params[1] if len(cam.params) > 1 else fx
                cx = cam.params[2] if len(cam.params) > 2 else cam.width / 2
                cy = cam.params[3] if len(cam.params) > 3 else cam.height / 2

            self.cameras.append(
                CameraData(
                    R=r_tensor,
                    T=t_tensor,
                    fx=float(fx) * scale_factor,
                    fy=float(fy) * scale_factor,
                    cx=float(cx) * scale_factor,
                    cy=float(cy) * scale_factor,
                    width=max(1, int(round(float(cam.width) * scale_factor))),
                    height=max(1, int(round(float(cam.height) * scale_factor))),
                    image_path=str(img_path),
                )
            )


class InstantNGPDataset(BaseReconstructionDataset):
    def __init__(
        self,
        transforms_path: Union[str, Path],
        image_dir: Optional[Union[str, Path]] = None,
        device: torch.device = torch.device("cuda"),
        use_dataloader=True,
        point_cloud_extent_ratio: Optional[float] = None,
        train_semantics=False,
        semantics_dim=3,
        semantics_path: Optional[Path] = None,
        semantics_resolution: Optional[Union[tuple[int, int], int]] = None,
        scene_extent_margin: float = 2.0,
        image_scale: int = 2,
        require_depth: bool = False,
        point_cloud_path: Optional[Union[str, Path]] = None,
        fallback_init_points: int = 20000,
    ):
        self.transforms_path = self._resolve_transforms_path(transforms_path)
        resolved_image_dir = self._resolve_image_root(image_dir=image_dir, transforms_path=self.transforms_path)
        super().__init__(
            image_dir=resolved_image_dir,
            device=device,
            use_dataloader=use_dataloader,
            train_semantics=train_semantics,
            semantics_dim=semantics_dim,
            semantics_path=semantics_path,
            semantics_resolution=semantics_resolution,
            image_scale=image_scale,
            require_depth=require_depth,
        )
        self._load_instant_ngp(
            point_cloud_extent_ratio=point_cloud_extent_ratio,
            scene_extent_margin=scene_extent_margin,
            point_cloud_path=point_cloud_path,
            fallback_init_points=fallback_init_points,
        )

    @staticmethod
    def _resolve_transforms_path(transforms_path: Union[str, Path]) -> Path:
        path = Path(transforms_path).expanduser().resolve()
        if path.is_dir():
            candidates = [path / "transform.json", path / "transforms.json"]
            for candidate in candidates:
                if candidate.exists():
                    return candidate
            raise FileNotFoundError(
                f"No Instant-NGP transform file found in directory {path}. "
                "Expected one of: transform.json, transforms.json"
            )

        if not path.exists():
            raise FileNotFoundError(f"Instant-NGP transforms file not found: {path}")
        return path

    @staticmethod
    def _resolve_image_root(image_dir: Optional[Union[str, Path]], transforms_path: Path) -> Path:
        if image_dir is not None:
            return Path(image_dir).expanduser().resolve()

        parent = transforms_path.parent
        images_dir = parent / "images"
        if images_dir.exists():
            return images_dir
        return parent

    @staticmethod
    def _resolve_frame_image_path(image_root: Path, transforms_root: Path, file_path_str: str) -> Path:
        candidate = Path(file_path_str)
        if candidate.is_absolute():
            return candidate

        if candidate.suffix == "":
            candidate = candidate.with_suffix(".png")

        for root in [image_root, transforms_root]:
            test_path = (root / candidate).resolve()
            if test_path.exists():
                return test_path

        return (image_root / candidate).resolve()

    @staticmethod
    def _focal_from_fov(width: int, fov_radians: Optional[float]) -> float:
        if fov_radians is None:
            raise ValueError(
                "Instant-NGP intrinsics are missing. Provide fl_x/fl_y (or camera_angle_x/camera_angle_y) "
                "in transforms.json or per-frame metadata."
            )
        return 0.5 * float(width) / np.tan(0.5 * float(fov_radians))

    @staticmethod
    def _sanitize_extrinsics_w2c(extrinsic: np.ndarray) -> np.ndarray:
        """Project the 3x3 block to the closest valid rotation for robust rasterization."""
        rot = extrinsic[:3, :3]
        u, _, vt = np.linalg.svd(rot)
        rot_ortho = u @ vt
        if np.linalg.det(rot_ortho) < 0:
            u[:, -1] *= -1
            rot_ortho = u @ vt
        extrinsic_fixed = extrinsic.copy()
        extrinsic_fixed[:3, :3] = rot_ortho.astype(np.float32)
        return extrinsic_fixed

    @staticmethod
    def _resolve_frame_image_path_from_index(image_root: Path, frame_index: int) -> Optional[Path]:
        # Common names: 0000.png, 0001.png, ... and 1-based frame_index in metadata.
        candidates = [
            image_root / f"{frame_index:04d}.png",
            image_root / f"{frame_index - 1:04d}.png",
            image_root / f"{frame_index:05d}.png",
            image_root / f"{frame_index - 1:05d}.png",
        ]
        for path in candidates:
            if path.exists():
                return path.resolve()
        return None

    def _load_instant_ngp(
        self,
        point_cloud_extent_ratio: Optional[float],
        scene_extent_margin: float,
        point_cloud_path: Optional[Union[str, Path]],
        fallback_init_points: int,
    ):
        with open(self.transforms_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        frames = meta.get("frames", [])
        if len(frames) == 0:
            raise ValueError(f"No frames found in Instant-NGP transforms file: {self.transforms_path}")

        scale_factor = 1.0 / float(self.image_scale)
        camera_centers = []

        for frame in frames:
            file_path = frame.get("file_path")
            frame_index = frame.get("frame_index")
            img_path = None
            if file_path is not None:
                img_path = self._resolve_frame_image_path(self.image_dir, self.transforms_path.parent, file_path)
            elif frame_index is not None:
                img_path = self._resolve_frame_image_path_from_index(self.image_dir, int(frame_index))

            if img_path is None:
                continue
            if not img_path.exists():
                continue

            if "transform_matrix" in frame:
                c2w = np.asarray(frame["transform_matrix"], dtype=np.float32)
                if c2w.shape != (4, 4):
                    raise ValueError(f"Invalid transform_matrix shape for frame {file_path}: {c2w.shape}")
                w2c = np.linalg.inv(c2w)
            elif "rot_mat" in frame:
                w2c = np.asarray(frame["rot_mat"], dtype=np.float32)
                if w2c.shape != (4, 4):
                    frame_name = file_path if file_path is not None else f"frame_index={frame_index}"
                    raise ValueError(f"Invalid rot_mat shape for frame {frame_name}: {w2c.shape}")
                w2c = self._sanitize_extrinsics_w2c(w2c)
                c2w = np.linalg.inv(w2c)
            else:
                continue

            r_tensor = torch.tensor(w2c[:3, :3], dtype=torch.float32)
            t_tensor = torch.tensor(w2c[:3, 3], dtype=torch.float32)
            camera_centers.append(c2w[:3, 3])

            width = int(frame.get("w", meta.get("w", 0)))
            height = int(frame.get("h", meta.get("h", 0)))
            if width <= 0 or height <= 0:
                img_hw = imageio.imread(img_path).shape[:2]
                height, width = int(img_hw[0]), int(img_hw[1])

            fx = frame.get("fl_x", meta.get("fl_x", None))
            fy = frame.get("fl_y", meta.get("fl_y", None))
            cx = frame.get("cx", meta.get("cx", None))
            cy = frame.get("cy", meta.get("cy", None))

            if fx is None:
                fx = self._focal_from_fov(width, frame.get("camera_angle_x", meta.get("camera_angle_x", None)))
            if fy is None:
                if "camera_angle_y" in frame or "camera_angle_y" in meta:
                    fy = self._focal_from_fov(height, frame.get("camera_angle_y", meta.get("camera_angle_y")))
                else:
                    fy = fx
            if cx is None:
                cx = float(width) / 2.0
            if cy is None:
                cy = float(height) / 2.0

            self.cameras.append(
                CameraData(
                    R=r_tensor,
                    T=t_tensor,
                    fx=float(fx) * scale_factor,
                    fy=float(fy) * scale_factor,
                    cx=float(cx) * scale_factor,
                    cy=float(cy) * scale_factor,
                    width=max(1, int(round(float(width) * scale_factor))),
                    height=max(1, int(round(float(height) * scale_factor))),
                    image_path=str(img_path),
                )
            )

        if len(self.cameras) == 0:
            raise RuntimeError(f"No valid frames with existing images were found in {self.transforms_path}")

        xyz, rgb = self._load_or_generate_init_cloud(
            point_cloud_path=point_cloud_path,
            camera_centers=np.asarray(camera_centers, dtype=np.float32),
            fallback_init_points=fallback_init_points,
        )
        self._set_init_points_and_colors(
            xyz=xyz,
            rgb_uint8=rgb,
            scene_extent_margin=scene_extent_margin,
            point_cloud_extent_ratio=point_cloud_extent_ratio,
        )

    def _load_or_generate_init_cloud(
        self,
        point_cloud_path: Optional[Union[str, Path]],
        camera_centers: np.ndarray,
        fallback_init_points: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        if point_cloud_path is not None:
            cloud_path = Path(point_cloud_path).expanduser().resolve()
            if not cloud_path.exists():
                raise FileNotFoundError(f"point_cloud_path does not exist: {cloud_path}")

            xyz, rgb = self._load_point_cloud_file(cloud_path)
            return xyz, rgb

        center = camera_centers.mean(axis=0) if len(camera_centers) > 0 else np.zeros(3, dtype=np.float32)
        if len(camera_centers) > 1:
            radius = np.linalg.norm(camera_centers - center, axis=1).mean()
        else:
            radius = 1.0

        radius = max(float(radius), 0.1)
        xyz = center[None, :] + np.random.normal(0.0, radius * 0.3, size=(int(fallback_init_points), 3)).astype(np.float32)
        rgb = np.full((int(fallback_init_points), 3), 127.5, dtype=np.float32)
        logger.warning(
            "[yellow]Instant-NGP init cloud fallback:[/yellow] generated synthetic initialization "
            f"with {fallback_init_points} points near camera centers."
        )
        return xyz, rgb

    @staticmethod
    def _normalize_point_cloud_arrays(
        xyz: np.ndarray,
        rgb: Optional[np.ndarray] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        xyz = np.asarray(xyz, dtype=np.float32)
        if xyz.ndim != 2 or xyz.shape[1] != 3:
            raise ValueError(f"Expected xyz with shape [N, 3], got {xyz.shape}")

        if rgb is None:
            rgb_out = np.full((xyz.shape[0], 3), 127.5, dtype=np.float32)
        else:
            rgb_arr = np.asarray(rgb, dtype=np.float32)
            if rgb_arr.ndim != 2 or rgb_arr.shape[1] != 3 or rgb_arr.shape[0] != xyz.shape[0]:
                raise ValueError(
                    "Expected rgb with shape [N, 3] matching xyz; "
                    f"got rgb={rgb_arr.shape}, xyz={xyz.shape}"
                )
            rgb_out = np.clip(rgb_arr, 0.0, 255.0)

        return xyz, rgb_out

    @classmethod
    def _load_point_cloud_file(cls, cloud_path: Path) -> tuple[np.ndarray, np.ndarray]:
        suffix = cloud_path.suffix.lower()

        if suffix == ".npy":
            cloud = np.load(cloud_path)
            if cloud.ndim != 2 or cloud.shape[1] not in [3, 6]:
                raise ValueError(
                    "Instant-NGP point cloud npy must have shape [N, 3] (xyz) or [N, 6] (xyzrgb)."
                )
            xyz = cloud[:, :3]
            rgb = cloud[:, 3:6] if cloud.shape[1] == 6 else None
            return cls._normalize_point_cloud_arrays(xyz=xyz, rgb=rgb)

        if suffix in {".pth", ".pt"}:
            payload = torch.load(cloud_path, map_location="cpu")

            if isinstance(payload, dict):
                if "xyz" not in payload:
                    raise ValueError(
                        "Point cloud .pth dict must contain key 'xyz'. Optional key: 'rgb'."
                    )
                xyz_raw = payload["xyz"]
                rgb_raw = payload.get("rgb", None)
            elif isinstance(payload, torch.Tensor) or isinstance(payload, np.ndarray):
                arr = payload.detach().cpu().numpy() if isinstance(payload, torch.Tensor) else np.asarray(payload)
                if arr.ndim != 2 or arr.shape[1] not in [3, 6]:
                    raise ValueError(
                        "Point cloud tensor/array in .pth must have shape [N, 3] or [N, 6]."
                    )
                xyz_raw = arr[:, :3]
                rgb_raw = arr[:, 3:6] if arr.shape[1] == 6 else None
            else:
                raise ValueError(
                    "Unsupported .pth payload type. Use dict {'xyz': [N,3], 'rgb': [N,3]} "
                    "or a tensor/array with shape [N,3] or [N,6]."
                )

            if isinstance(xyz_raw, torch.Tensor):
                xyz_raw = xyz_raw.detach().cpu().numpy()
            if isinstance(rgb_raw, torch.Tensor):
                rgb_raw = rgb_raw.detach().cpu().numpy()

            return cls._normalize_point_cloud_arrays(xyz=xyz_raw, rgb=rgb_raw)

        raise ValueError(
            f"Unsupported point cloud file type '{suffix}' for {cloud_path}. "
            "Supported: .npy, .pth, .pt"
        )


def create_dataset(dataset_type: str, **kwargs) -> BaseReconstructionDataset:
    """Factory for reconstruction datasets."""
    dataset_type_norm = dataset_type.strip().lower().replace("_", "-")
    if dataset_type_norm in {"colmap"}:
        return ColmapDataset(**kwargs)
    if dataset_type_norm in {"instant-ngp", "instantngp", "ngp"}:
        return InstantNGPDataset(**kwargs)
    raise ValueError(f"Unsupported dataset_type '{dataset_type}'. Supported: colmap, instant-ngp")
