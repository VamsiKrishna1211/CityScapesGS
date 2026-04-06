import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence, Union

import cv2
import imageio.v2 as imageio
import numpy as np
import torch
from plyfile import PlyData
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
    uid: int  # Camera index for appearance embeddings (must be unique per camera)
    depth_dir: Optional[str] = None
    depth_scale: float = 1.0

    def __getitem__(self, key: str):
        return getattr(self, key)
    
    @property
    def camera_center(self) -> torch.Tensor:
        """
        Compute camera center (camera position in world coordinates).
        The view matrix transforms world -> camera: [R|T]
        Camera center is at -R^T @ T in world coordinates.
        """
        return -self.R.T @ self.T


class BaseReconstructionDataset(Dataset):
    def __init__(
        self,
        image_dir: Union[str, Path],
        depth_dir: Optional[Union[str, Path]] = None,
        device: torch.device = torch.device("cuda"),
        use_dataloader: bool = True,
        image_scale: int = 2,
        require_depth: bool = True,
    ):
        """
        Base dataset that handles common image/depth loading behavior.

        Args:
            - image_dir: Directory containing images
            - depth_dir: Optional explicit depth directory. If provided, this takes precedence over predefined scale-based depth folder names.
            - device: torch.device to load tensors onto (default: 'cuda')
            - use_dataloader: If True, workers keep data on CPU and __getitem__ moves to device
            - image_scale: Downscale factor used for training images. If image_dir points to `images`, this resolves to `images_<image_scale>` for scale > 1.
            - require_depth: If True, missing depth directory raises an error
        """
        self.device = device
        self.use_dataloader = use_dataloader
        self._preloaded_images = None
        self._preloaded_depths = None
        self.require_depth = require_depth
        self.cameras: list[CameraData] = []

        # Set by subclasses.
        self.init_points = torch.empty((0, 3), dtype=torch.float32, device=self.device)
        self.init_colors = torch.empty((0, 3), dtype=torch.float32, device=self.device)
        self.scene_extent = float(1.0)

        if image_scale < 1:
            raise ValueError(f"image_scale must be >= 1, got {image_scale}")
        self.image_scale = int(image_scale)
        image_dir_path = self._resolve_image_dir(Path(image_dir), self.image_scale)
        depth_dir_path = Path(depth_dir).expanduser().resolve() if depth_dir is not None else None
        logger.info(f"[cyan]Using image scale:[/cyan] {self.image_scale} (directory: {image_dir_path})")
        try:
            self.depth_dir = self._resolve_depth_dir(
                image_dir=image_dir_path,
                image_scale=self.image_scale,
                depth_dir_override=depth_dir_path,
                require_depth=self.require_depth,
            )
            logger.info(f"[cyan]Using depth scale:[/cyan] {self.image_scale} (directory: {self.depth_dir})")
        except FileNotFoundError:
            self.depth_dir = None

        self.image_dir = image_dir_path

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
    def _estimate_camera_forward_z_distance(camera_centers: np.ndarray) -> float:
        """Estimate a sensible forward projection distance from camera trajectory spacing."""
        camera_centers = np.asarray(camera_centers, dtype=np.float32)
        if camera_centers.ndim != 2 or camera_centers.shape[0] < 2:
            return 5.0

        step_dists = np.linalg.norm(camera_centers[1:] - camera_centers[:-1], axis=1)
        valid = step_dists[np.isfinite(step_dists) & (step_dists > 1e-5)]
        if valid.size == 0:
            return 5.0

        z_distance = float(np.median(valid) * 5.0)
        return float(np.clip(z_distance, 1.0, 200.0))

    @classmethod
    def _synthesize_points_from_camera_rays(
        cls,
        cameras: Sequence[CameraData],
        max_points: int,
        z_distance: Optional[float] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Create fallback initialization points along camera forward rays.

        This avoids degeneracy from placing points exactly at camera centers.
        """
        if len(cameras) == 0:
            return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.float32)

        centers: list[np.ndarray] = []
        for cam in cameras:
            r_w2c = cam.R.detach().cpu().numpy().astype(np.float32)
            t_w2c = cam.T.detach().cpu().numpy().astype(np.float32)
            c2w_r = r_w2c.T
            cam_center = -c2w_r @ t_w2c
            centers.append(cam_center)

        centers_np = np.stack(centers, axis=0)
        z = float(z_distance) if z_distance is not None else cls._estimate_camera_forward_z_distance(centers_np)

        ray_offsets = [
            (0.0, 0.0),
            (-0.35, -0.35),
            (0.35, -0.35),
            (-0.35, 0.35),
            (0.35, 0.35),
            (0.0, -0.45),
            (0.0, 0.45),
            (-0.45, 0.0),
            (0.45, 0.0),
        ]

        points: list[np.ndarray] = []
        for cam, cam_center in zip(cameras, centers_np):
            r_w2c = cam.R.detach().cpu().numpy().astype(np.float32)
            c2w_r = r_w2c.T

            right = c2w_r[:, 0]
            down = c2w_r[:, 1]
            forward = c2w_r[:, 2]

            for u, v in ray_offsets:
                direction = forward + u * right + v * down
                norm = float(np.linalg.norm(direction))
                if norm < 1e-8:
                    continue
                direction = direction / norm
                points.append(cam_center + direction * z)

        if len(points) == 0:
            return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.float32)

        xyz = np.asarray(points, dtype=np.float32)
        if max_points > 0 and xyz.shape[0] > max_points:
            rng = np.random.default_rng(0)
            idx = rng.choice(xyz.shape[0], size=max_points, replace=False)
            xyz = xyz[idx]

        rgb = np.full((xyz.shape[0], 3), 127.5, dtype=np.float32)
        logger.info(
            f"[cyan]Fallback init cloud:[/cyan] synthesized {xyz.shape[0]} points along camera rays "
            f"at z={z:.3f}"
        )
        return xyz, rgb

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
    def _resolve_depth_dir(
        image_dir: Path,
        image_scale: int,
        depth_dir_override: Optional[Path] = None,
        require_depth: bool = True,
    ) -> Optional[Path]:
        """Resolve depth directory from explicit override or predefined scale convention."""
        if depth_dir_override is not None:
            if depth_dir_override.exists():
                return depth_dir_override
            if not require_depth:
                logger.warning(
                    f"[yellow]Depth directory override not found, depth supervision disabled:[/yellow] {depth_dir_override}"
                )
                return None
            raise FileNotFoundError(f"Provided depth directory does not exist: {depth_dir_override}")

        scene_root = image_dir.parent
        candidate_dirs = [scene_root / "depths_npy"]
        if image_scale > 1:
            candidate_dirs.insert(0, scene_root / f"depths_npy_{image_scale}")

        for depth_dir in candidate_dirs:
            if depth_dir.exists():
                return depth_dir

        expected_str = ", ".join(str(d) for d in candidate_dirs)
        if not require_depth:
            logger.warning(
                f"[yellow]Depth directory not found, depth supervision disabled:[/yellow] expected one of: {expected_str}"
            )
            return None
        raise FileNotFoundError(
            f"Requested depth scale {image_scale}, but no depth directory found. "
            f"Checked: {expected_str}. "
            "Expected depth folders following image scale convention (e.g., depths_npy, depths_npy_2, depths_npy_4)."
        )

    def __len__(self):
        return len(self.cameras)

    @staticmethod
    def _try_read_exr_depth(depth_path: Path) -> Optional[np.ndarray]:
        """Read EXR depth map via OpenCV first, then imageio as fallback."""
        try:
            exr_depth = cv2.imread(str(depth_path), cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
            if exr_depth is not None:
                if exr_depth.ndim == 3:
                    exr_depth = exr_depth[..., 0]
                return np.asarray(exr_depth, dtype=np.float32)
        except Exception:
            pass

        try:
            exr_depth = imageio.imread(depth_path)
            if exr_depth is not None:
                exr_depth = np.asarray(exr_depth, dtype=np.float32)
                if exr_depth.ndim == 3:
                    exr_depth = exr_depth[..., 0]
                return exr_depth
        except Exception:
            pass

        return None

    @staticmethod
    def _validate_depth_map(depth_map: np.ndarray, cam_data: CameraData) -> Optional[torch.Tensor]:
        """Validate and sanitize depth map before converting to tensor."""
        if depth_map.shape[:2] != (cam_data.height, cam_data.width):
            return None

        if not np.isfinite(depth_map).all():
            invalid_mask = ~np.isfinite(depth_map)
            if invalid_mask.all():
                return None
            depth_map = depth_map.copy()
            depth_map[invalid_mask] = np.median(depth_map[~invalid_mask])

        d_min, d_max = float(depth_map.min()), float(depth_map.max())
        d_range = d_max - d_min
        if d_range < 1e-8:
            return None

        return torch.from_numpy(np.asarray(depth_map, dtype=np.float32)).float()

    @staticmethod
    def _resolve_depth_path(depth_dir: Path, image_stem: str) -> Optional[Path]:
        """Resolve depth file path, preferring .npy then .exr."""
        candidates = [
            depth_dir / f"{image_stem}.npy",
            depth_dir / f"{image_stem}.exr",
            depth_dir / f"{image_stem}.EXR",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return None

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

    def __getitem__(self, idx: int):
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

        return cam_data, img_tensor.to(self.device), depth_tensor

    def _load_depth(self, img_path: Path, cam_data: CameraData) -> Optional[torch.Tensor]:
        """Load and validate depth map."""
        depth_dir = Path(cam_data.depth_dir).expanduser().resolve() if cam_data.depth_dir else self.depth_dir
        if depth_dir is None:
            return None

        depth_path = self._resolve_depth_path(depth_dir, img_path.stem)
        if depth_path is None:
            return None

        try:
            if depth_path.suffix.lower() == ".npy":
                depth_map = np.load(depth_path)
            elif depth_path.suffix.lower() == ".exr":
                depth_map = self._try_read_exr_depth(depth_path)
                if depth_map is None:
                    return None
                # MatrixCity float16 EXR may contain saturated invalid values.
                invalid_exr_mask = depth_map >= 65504.0
                if invalid_exr_mask.any():
                    depth_map = depth_map.astype(np.float32, copy=True)
                    depth_map[invalid_exr_mask] = np.nan
            else:
                return None

            if cam_data.depth_scale != 1.0:
                depth_map = np.asarray(depth_map, dtype=np.float32) * float(cam_data.depth_scale)

            return self._validate_depth_map(depth_map, cam_data)

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
        depth_dir: Optional[Union[str, Path]] = None,
        device: torch.device = torch.device("cuda"),
        use_dataloader=True,
        point_cloud_extent_ratio: Optional[float] = None,
        scene_extent_margin: float = 2.0,
        image_scale: int = 2,
        require_depth: bool = True,
    ):
        super().__init__(
            image_dir=image_dir,
            depth_dir=depth_dir,
            device=device,
            use_dataloader=use_dataloader,
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
                    uid=len(self.cameras),  # Assign sequential camera ID
                )
            )


class InstantNGPDataset(BaseReconstructionDataset):
    def __init__(
        self,
        transforms_path: Union[str, Path],
        image_dir: Optional[Union[str, Path]] = None,
        depth_dir: Optional[Union[str, Path]] = None,
        point_cloud_path: Optional[Union[str, Path]] = None,
        device: torch.device = torch.device("cuda"),
        use_dataloader: bool = True,
        point_cloud_extent_ratio: Optional[float] = None,
        scene_extent_margin: float = 2.0,
        image_scale: int = 1,
        require_depth: bool = True,
    ):
        self.transforms_path = self._resolve_transforms_path(Path(transforms_path))
        resolved_image_dir = (
            Path(image_dir).expanduser().resolve()
            if image_dir is not None
            else self._default_image_dir_from_transforms(self.transforms_path)
        )
        self.point_cloud_path = Path(point_cloud_path).expanduser().resolve() if point_cloud_path else None

        super().__init__(
            image_dir=resolved_image_dir,
            depth_dir=depth_dir,
            device=device,
            use_dataloader=use_dataloader,
            image_scale=image_scale,
            require_depth=require_depth,
        )

        self._load_nerf_transforms(
            transforms_file=self.transforms_path,
            depth_dir_override=self.depth_dir,
            point_cloud_extent_ratio=point_cloud_extent_ratio,
            scene_extent_margin=scene_extent_margin,
        )

    @staticmethod
    def _resolve_transforms_path(transforms_path: Path) -> Path:
        transforms_path = transforms_path.expanduser().resolve()
        if transforms_path.is_file():
            return transforms_path
        if transforms_path.is_dir():
            for name in ("transforms.json", "transform.json", "transforms_train.json"):
                candidate = transforms_path / name
                if candidate.exists():
                    return candidate
        raise FileNotFoundError(
            f"Could not resolve transforms file from: {transforms_path}. "
            "Expected a JSON file or directory containing transforms.json/transform.json/transforms_train.json"
        )

    @staticmethod
    def _default_image_dir_from_transforms(transforms_file: Path) -> Path:
        parent = transforms_file.parent
        candidates = [
            parent / "images",
            parent,
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        raise FileNotFoundError(f"Could not infer image directory from transforms file: {transforms_file}")

    @staticmethod
    def _normalize_rotation_matrix(rot: np.ndarray) -> np.ndarray:
        rot = np.asarray(rot, dtype=np.float64)
        col_norms = np.linalg.norm(rot, axis=0)
        valid = col_norms > 1e-8
        if valid.any():
            scale = float(np.median(col_norms[valid]))
            if scale > 1e-8:
                rot = rot / scale

        u, _, vh = np.linalg.svd(rot)
        rot_ortho = u @ vh
        if np.linalg.det(rot_ortho) < 0:
            u[:, -1] *= -1.0
            rot_ortho = u @ vh
        return rot_ortho.astype(np.float32)

    @classmethod
    def _nerf_c2w_to_w2c(cls, c2w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        c2w = np.asarray(c2w, dtype=np.float64).copy()
        c2w[:3, :3] = cls._normalize_rotation_matrix(c2w[:3, :3])

        # NeRF/Blender camera coordinates -> COLMAP/OpenCV camera coordinates.
        c2w[:3, 1:3] *= -1.0

        w2c = np.linalg.inv(c2w)
        r = w2c[:3, :3].astype(np.float32)
        t = w2c[:3, 3].astype(np.float32)
        return r, t

    @staticmethod
    def _frame_index_to_stem(frame: dict[str, Any]) -> Optional[str]:
        if "frame_index" in frame:
            return f"{int(frame['frame_index']):04d}"
        if "file_path" in frame:
            return Path(str(frame["file_path"])).stem
        return None

    @staticmethod
    def _try_image_candidates(base_paths: Sequence[Path], stem_or_path: str) -> Optional[Path]:
        candidate_paths: list[Path] = []
        suffixes = ("", ".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG")

        as_path = Path(stem_or_path)
        if as_path.is_absolute():
            for sfx in suffixes:
                candidate_paths.append(as_path if sfx == "" else as_path.with_suffix(sfx))
        else:
            for base in base_paths:
                for sfx in suffixes:
                    if sfx == "":
                        candidate_paths.append(base / as_path)
                    else:
                        candidate_paths.append(base / f"{stem_or_path}{sfx}" if as_path.suffix == "" else (base / as_path).with_suffix(sfx))

        for candidate in candidate_paths:
            if candidate.exists() and candidate.is_file():
                return candidate.resolve()
        return None

    @classmethod
    def _resolve_frame_image_path(
        cls,
        frame: dict[str, Any],
        transforms_parent: Path,
        image_root: Path,
    ) -> Optional[Path]:
        base_paths = [
            image_root,
            image_root / "images",
            image_root / "rgb",
            transforms_parent,
            transforms_parent / "images",
            transforms_parent / "rgb",
            image_root.parent,
            image_root.parent.parent,
        ]

        if "file_path" in frame:
            file_path = str(frame["file_path"])
            resolved = cls._try_image_candidates(base_paths, file_path)
            if resolved is not None:
                return resolved

        stem = cls._frame_index_to_stem(frame)
        if stem is None:
            return None
        return cls._try_image_candidates(base_paths, stem)

    @staticmethod
    def _image_size(path: Path) -> tuple[int, int]:
        img = imageio.imread(path)
        h, w = img.shape[:2]
        return int(w), int(h)

    @staticmethod
    def _intrinsics_from_meta(meta: dict[str, Any], width: int, height: int) -> tuple[float, float, float, float]:
        if "fl_x" in meta or "fl_y" in meta:
            fx_raw = meta.get("fl_x", meta.get("fl_y"))
            fy_raw = meta.get("fl_y", meta.get("fl_x"))
            if fx_raw is None or fy_raw is None:
                raise ValueError("Invalid fl_x/fl_y values in transforms metadata")
            fx = float(fx_raw)
            fy = float(fy_raw)
        elif "camera_angle_x" in meta:
            fx = float(0.5 * width / np.tan(0.5 * float(meta["camera_angle_x"])))
            if "camera_angle_y" in meta:
                fy = float(0.5 * height / np.tan(0.5 * float(meta["camera_angle_y"])))
            else:
                fy = fx
        else:
            raise ValueError("Missing intrinsics. Expected fl_x/fl_y or camera_angle_x in transforms metadata.")

        cx = float(meta.get("cx", width * 0.5))
        cy = float(meta.get("cy", height * 0.5))
        return fx, fy, cx, cy

    @staticmethod
    def _load_ply_points(path: Path) -> tuple[np.ndarray, np.ndarray]:
        ply = PlyData.read(path)
        vertex = ply["vertex"]
        xyz = np.stack(
            [
                np.asarray(vertex["x"], dtype=np.float32),
                np.asarray(vertex["y"], dtype=np.float32),
                np.asarray(vertex["z"], dtype=np.float32),
            ],
            axis=1,
        )

        color_fields = ("red", "green", "blue")
        if all(field in vertex.data.dtype.names for field in color_fields):
            rgb = np.stack(
                [
                    np.asarray(vertex["red"], dtype=np.float32),
                    np.asarray(vertex["green"], dtype=np.float32),
                    np.asarray(vertex["blue"], dtype=np.float32),
                ],
                axis=1,
            )
        else:
            rgb = np.full((xyz.shape[0], 3), 127.5, dtype=np.float32)
        return xyz, rgb

    @classmethod
    def _load_point_cloud(cls, point_cloud_path: Optional[Path]) -> Optional[tuple[np.ndarray, np.ndarray]]:
        if point_cloud_path is None or not point_cloud_path.exists():
            return None

        suffix = point_cloud_path.suffix.lower()
        if suffix == ".ply":
            return cls._load_ply_points(point_cloud_path)

        if suffix == ".npy":
            arr = np.load(point_cloud_path)
            arr = np.asarray(arr)
            if arr.ndim != 2 or arr.shape[1] < 3:
                raise ValueError(f"Unsupported .npy point cloud shape: {arr.shape}")
            xyz = arr[:, :3].astype(np.float32)
            if arr.shape[1] >= 6:
                rgb = arr[:, 3:6].astype(np.float32)
                if rgb.max() <= 1.0:
                    rgb *= 255.0
            else:
                rgb = np.full((xyz.shape[0], 3), 127.5, dtype=np.float32)
            return xyz, rgb

        if suffix in {".pt", ".pth"}:
            payload = torch.load(point_cloud_path, map_location="cpu")
            if isinstance(payload, dict) and "xyz" in payload:
                xyz = np.asarray(payload["xyz"], dtype=np.float32)
                if "rgb" in payload:
                    rgb = np.asarray(payload["rgb"], dtype=np.float32)
                    if rgb.max() <= 1.0:
                        rgb *= 255.0
                else:
                    rgb = np.full((xyz.shape[0], 3), 127.5, dtype=np.float32)
                return xyz, rgb
            raise ValueError(f"Unsupported checkpoint point cloud format: {point_cloud_path}")

        raise ValueError(f"Unsupported point cloud format: {point_cloud_path}")

    def _load_nerf_transforms(
        self,
        transforms_file: Path,
        depth_dir_override: Optional[Path],
        point_cloud_extent_ratio: Optional[float],
        scene_extent_margin: float,
    ) -> None:
        with transforms_file.open("r", encoding="utf-8") as f:
            meta = json.load(f)

        frames = meta.get("frames", [])
        if not frames:
            raise ValueError(f"No frames found in transforms file: {transforms_file}")

        first_img = None
        for frame in frames:
            first_img = self._resolve_frame_image_path(frame, transforms_file.parent, self.image_dir)
            if first_img is not None:
                break
        if first_img is None:
            raise FileNotFoundError(f"Could not resolve any image referenced by: {transforms_file}")

        width = int(meta.get("w", 0))
        height = int(meta.get("h", 0))
        if width <= 0 or height <= 0:
            width, height = self._image_size(first_img)

        fx, fy, cx, cy = self._intrinsics_from_meta(meta, width, height)
        scale_factor = 1.0 / float(self.image_scale)

        camera_centers: list[np.ndarray] = []
        for frame in frames:
            img_path = self._resolve_frame_image_path(frame, transforms_file.parent, self.image_dir)
            if img_path is None:
                continue

            if "transform_matrix" in frame:
                c2w = np.asarray(frame["transform_matrix"], dtype=np.float32)
            elif "rot_mat" in frame:
                c2w = np.asarray(frame["rot_mat"], dtype=np.float32)
            else:
                continue

            r, t = self._nerf_c2w_to_w2c(c2w)
            self.cameras.append(
                CameraData(
                    R=torch.tensor(r, dtype=torch.float32),
                    T=torch.tensor(t, dtype=torch.float32),
                    fx=float(fx) * scale_factor,
                    fy=float(fy) * scale_factor,
                    cx=float(cx) * scale_factor,
                    cy=float(cy) * scale_factor,
                    width=max(1, int(round(width * scale_factor))),
                    height=max(1, int(round(height * scale_factor))),
                    image_path=str(img_path),
                    uid=len(self.cameras),  # Assign sequential camera ID
                    depth_dir=str(depth_dir_override) if depth_dir_override is not None else (str(self.depth_dir) if self.depth_dir is not None else None),
                )
            )
            camera_centers.append(np.asarray(c2w[:3, 3], dtype=np.float32))

        if not self.cameras:
            raise RuntimeError(f"No valid camera frames loaded from: {transforms_file}")

        loaded_cloud = self._load_point_cloud(self.point_cloud_path)
        if loaded_cloud is not None:
            xyz, rgb = loaded_cloud
        else:
            fallback_max_points = max(5000, min(200000, len(self.cameras) * 9))
            xyz, rgb = self._synthesize_points_from_camera_rays(
                cameras=self.cameras,
                max_points=fallback_max_points,
            )

        self._set_init_points_and_colors(
            xyz=xyz,
            rgb_uint8=rgb,
            scene_extent_margin=scene_extent_margin,
            point_cloud_extent_ratio=point_cloud_extent_ratio,
        )


class MatrixCityDataset(BaseReconstructionDataset):
    def __init__(
        self,
        matrixcity_paths: Sequence[Union[str, Path]],
        matrixcity_depth_paths: Optional[Sequence[Union[str, Path]]] = None,
        matrixcity_pointcloud_paths: Optional[Sequence[Union[str, Path]]] = None,
        matrixcity_max_init_points: int = 300000,
        device: torch.device = torch.device("cuda"),
        use_dataloader: bool = True,
        point_cloud_extent_ratio: Optional[float] = None,
        scene_extent_margin: float = 2.0,
        image_scale: int = 1,
        require_depth: bool = True,
    ):
        if not matrixcity_paths:
            raise ValueError("matrixcity_paths must contain at least one block path")

        block_paths = [Path(p).expanduser().resolve() for p in matrixcity_paths]
        for block_path in block_paths:
            if not block_path.exists():
                raise FileNotFoundError(f"MatrixCity block path not found: {block_path}")

        self.block_paths = block_paths
        self.matrixcity_max_init_points = int(matrixcity_max_init_points)
        self.matrixcity_pointcloud_paths = [
            Path(p).expanduser().resolve() for p in (matrixcity_pointcloud_paths or [])
        ]

        depth_paths: list[Optional[Path]] = []
        if matrixcity_depth_paths is not None:
            if len(matrixcity_depth_paths) != len(block_paths):
                raise ValueError(
                    "matrixcity_depth_paths must either be omitted or have exactly one entry per matrixcity_paths"
                )
            for p in matrixcity_depth_paths:
                d = Path(p).expanduser().resolve()
                if not d.exists() and require_depth:
                    raise FileNotFoundError(f"MatrixCity depth path not found: {d}")
                depth_paths.append(d if d.exists() else None)
        else:
            for block_path in block_paths:
                depth_paths.append(self._infer_block_depth_path(block_path))

        self._block_depth_paths = depth_paths

        super().__init__(
            image_dir=block_paths[0],
            depth_dir=None,
            device=device,
            use_dataloader=use_dataloader,
            image_scale=image_scale,
            require_depth=False,
        )

        self._load_matrixcity_blocks(
            point_cloud_extent_ratio=point_cloud_extent_ratio,
            scene_extent_margin=scene_extent_margin,
            require_depth=require_depth,
        )

    @staticmethod
    def _infer_block_depth_path(block_path: Path) -> Optional[Path]:
        candidates = [
            block_path / "depth",
            block_path / "depths",
            block_path / "depth_exr",
            block_path.parent / f"{block_path.name}_depth",
            block_path.parent / f"{block_path.name}_depths",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate.resolve()
        return None

    @staticmethod
    def _resolve_matrixcity_transforms(block_path: Path) -> Path:
        for name in ("transforms.json", "transforms_origin.json", "transform.json", "transforms_train.json"):
            candidate = block_path / name
            if candidate.exists():
                return candidate
        raise FileNotFoundError(
            f"No MatrixCity transform file found under: {block_path}. "
            "Expected transforms.json/transforms_origin.json/transform.json/transforms_train.json"
        )

    @classmethod
    def _matrixcity_rotmat_to_c2w(cls, rot_mat: Any) -> np.ndarray:
        c2w = np.asarray(rot_mat, dtype=np.float32).copy()
        if c2w.shape != (4, 4):
            raise ValueError(f"Expected rot_mat shape [4,4], got {c2w.shape}")
        c2w[3, :] = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        c2w[:3, :3] = InstantNGPDataset._normalize_rotation_matrix(c2w[:3, :3])
        return c2w

    def _load_matrixcity_point_clouds(self) -> Optional[tuple[np.ndarray, np.ndarray]]:
        merged_xyz: list[np.ndarray] = []
        merged_rgb: list[np.ndarray] = []
        for pc_path in self.matrixcity_pointcloud_paths:
            if not pc_path.exists():
                logger.warning(f"[yellow]Skipping missing point cloud:[/yellow] {pc_path}")
                continue
            xyz_rgb = InstantNGPDataset._load_point_cloud(pc_path)
            if xyz_rgb is None:
                continue
            xyz, rgb = xyz_rgb
            if xyz.shape[0] == 0:
                continue
            merged_xyz.append(xyz)
            merged_rgb.append(rgb)

        if not merged_xyz:
            return None

        xyz = np.concatenate(merged_xyz, axis=0)
        rgb = np.concatenate(merged_rgb, axis=0)

        if self.matrixcity_max_init_points > 0 and xyz.shape[0] > self.matrixcity_max_init_points:
            rng = np.random.default_rng(0)
            idx = rng.choice(xyz.shape[0], size=self.matrixcity_max_init_points, replace=False)
            xyz = xyz[idx]
            rgb = rgb[idx]

        return xyz, rgb

    def _synthesize_points_from_matrixcity_depth(self, max_points: int) -> tuple[np.ndarray, np.ndarray]:
        """Synthesize initialization points by backprojecting MatrixCity depth into world space."""
        if len(self.cameras) == 0:
            return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.float32)

        max_points = int(max_points)
        if max_points <= 0:
            return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.float32)

        rng = np.random.default_rng(0)
        cam_indices = np.arange(len(self.cameras), dtype=np.int32)
        rng.shuffle(cam_indices)

        # Limit number of contributing cameras for very large datasets.
        max_cameras = min(len(cam_indices), 256)
        selected_cam_indices = cam_indices[:max_cameras]
        samples_per_cam = max(64, max_points // max(1, len(selected_cam_indices)))

        all_xyz: list[np.ndarray] = []
        all_rgb: list[np.ndarray] = []

        for cam_idx in selected_cam_indices:
            cam = self.cameras[int(cam_idx)]
            depth_dir = Path(cam.depth_dir).expanduser().resolve() if cam.depth_dir else None
            if depth_dir is None:
                continue

            img_path = Path(cam.image_path)
            depth_path = self._resolve_depth_path(depth_dir, img_path.stem)
            if depth_path is None:
                continue

            try:
                if depth_path.suffix.lower() == ".npy":
                    depth_map = np.load(depth_path)
                elif depth_path.suffix.lower() == ".exr":
                    depth_map = self._try_read_exr_depth(depth_path)
                    if depth_map is None:
                        continue
                    invalid_exr_mask = depth_map >= 65504.0
                    if invalid_exr_mask.any():
                        depth_map = depth_map.astype(np.float32, copy=True)
                        depth_map[invalid_exr_mask] = np.nan
                else:
                    continue

                if cam.depth_scale != 1.0:
                    depth_map = np.asarray(depth_map, dtype=np.float32) * float(cam.depth_scale)

                if depth_map.shape[:2] != (cam.height, cam.width):
                    continue

                valid = np.isfinite(depth_map) & (depth_map > 1e-4)
                valid_idx = np.flatnonzero(valid.reshape(-1))
                if valid_idx.size == 0:
                    continue

                take = min(samples_per_cam, valid_idx.size)
                chosen = rng.choice(valid_idx, size=take, replace=False)

                ys = chosen // cam.width
                xs = chosen % cam.width
                z = depth_map.reshape(-1)[chosen].astype(np.float32)

                x_cam = ((xs.astype(np.float32) - float(cam.cx)) / float(cam.fx)) * z
                y_cam = ((ys.astype(np.float32) - float(cam.cy)) / float(cam.fy)) * z
                pts_cam = np.stack([x_cam, y_cam, z], axis=1)

                r_w2c = cam.R.detach().cpu().numpy().astype(np.float32)
                t_w2c = cam.T.detach().cpu().numpy().astype(np.float32)
                pts_world = (r_w2c.T @ (pts_cam - t_w2c).T).T
                all_xyz.append(pts_world.astype(np.float32))

                try:
                    img_raw = np.asarray(imageio.imread(img_path))
                    if img_raw.ndim == 2:
                        img_rgb = np.repeat(img_raw[..., None], 3, axis=-1)
                    elif img_raw.ndim == 3 and img_raw.shape[2] == 4:
                        img_rgb = img_raw[..., :3]
                    elif img_raw.ndim == 3 and img_raw.shape[2] == 1:
                        img_rgb = np.repeat(img_raw, 3, axis=-1)
                    else:
                        img_rgb = img_raw
                    colors = np.asarray(img_rgb[ys, xs], dtype=np.float32)
                except Exception:
                    colors = np.full((take, 3), 127.5, dtype=np.float32)

                all_rgb.append(colors)
            except Exception as exc:
                logger.debug(f"Depth point synthesis failed for {img_path.name}: {exc}")
                continue

        if len(all_xyz) == 0:
            logger.warning("[yellow]Depth-based MatrixCity init failed:[/yellow] no valid depth points synthesized")
            return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.float32)

        xyz = np.concatenate(all_xyz, axis=0)
        rgb = np.concatenate(all_rgb, axis=0)

        if xyz.shape[0] > max_points:
            keep = rng.choice(xyz.shape[0], size=max_points, replace=False)
            xyz = xyz[keep]
            rgb = rgb[keep]

        logger.info(
            f"[cyan]Depth-based MatrixCity init cloud:[/cyan] synthesized {xyz.shape[0]} points from depth maps"
        )
        return xyz.astype(np.float32), rgb.astype(np.float32)

    def _load_matrixcity_blocks(
        self,
        point_cloud_extent_ratio: Optional[float],
        scene_extent_margin: float,
        require_depth: bool,
    ) -> None:
        camera_centers: list[np.ndarray] = []
        missing_depth_dirs = 0

        for block_idx, block_path in enumerate(self.block_paths):
            transforms_file = self._resolve_matrixcity_transforms(block_path)
            with transforms_file.open("r", encoding="utf-8") as f:
                meta = json.load(f)

            frames = meta.get("frames", [])
            if not frames:
                logger.warning(f"[yellow]No frames found in block:[/yellow] {block_path}")
                continue

            first_img = None
            for frame in frames:
                first_img = InstantNGPDataset._resolve_frame_image_path(frame, transforms_file.parent, block_path)
                if first_img is not None:
                    break
            if first_img is None:
                logger.warning(f"[yellow]No valid image found for block:[/yellow] {block_path}")
                continue

            width = int(meta.get("w", 0))
            height = int(meta.get("h", 0))
            if width <= 0 or height <= 0:
                width, height = InstantNGPDataset._image_size(first_img)

            fx, fy, cx, cy = InstantNGPDataset._intrinsics_from_meta(meta, width, height)
            scale_factor = 1.0 / float(self.image_scale)

            depth_dir = self._block_depth_paths[block_idx]
            if depth_dir is None:
                missing_depth_dirs += 1

            for frame in frames:
                img_path = InstantNGPDataset._resolve_frame_image_path(frame, transforms_file.parent, block_path)
                if img_path is None:
                    continue

                if "rot_mat" in frame:
                    c2w = self._matrixcity_rotmat_to_c2w(frame["rot_mat"])
                elif "transform_matrix" in frame:
                    c2w = np.asarray(frame["transform_matrix"], dtype=np.float32)
                    c2w[3, :] = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
                    c2w[:3, :3] = InstantNGPDataset._normalize_rotation_matrix(c2w[:3, :3])
                else:
                    continue

                r, t = InstantNGPDataset._nerf_c2w_to_w2c(c2w)
                self.cameras.append(
                    CameraData(
                        R=torch.tensor(r, dtype=torch.float32),
                        T=torch.tensor(t, dtype=torch.float32),
                        fx=float(fx) * scale_factor,
                        fy=float(fy) * scale_factor,
                        cx=float(cx) * scale_factor,
                        cy=float(cy) * scale_factor,
                        width=max(1, int(round(width * scale_factor))),
                        height=max(1, int(round(height * scale_factor))),
                        image_path=str(img_path),
                        uid=len(self.cameras),  # Assign sequential camera ID
                        depth_dir=str(depth_dir) if depth_dir is not None else None,
                        # MatrixCity depth is exported in centimeters.
                        depth_scale=0.01,
                    )
                )
                camera_centers.append(np.asarray(c2w[:3, 3], dtype=np.float32))

        if not self.cameras:
            raise RuntimeError("No valid MatrixCity frames were loaded from the provided block paths")

        if require_depth and missing_depth_dirs > 0:
            raise FileNotFoundError(
                "Depth supervision is enabled, but one or more MatrixCity blocks have no depth directory. "
                "Provide --matrixcity-depth-path once per --matrixcity-path or disable depth loss."
            )

        loaded_cloud = self._load_matrixcity_point_clouds()
        if loaded_cloud is not None:
            xyz, rgb = loaded_cloud
        else:
            xyz, rgb = self._synthesize_points_from_matrixcity_depth(
                max_points=self.matrixcity_max_init_points,
            )
            if xyz.shape[0] == 0:
                xyz, rgb = self._synthesize_points_from_camera_rays(
                    cameras=self.cameras,
                    max_points=self.matrixcity_max_init_points,
                )

        self._set_init_points_and_colors(
            xyz=xyz,
            rgb_uint8=rgb,
            scene_extent_margin=scene_extent_margin,
            point_cloud_extent_ratio=point_cloud_extent_ratio,
        )


def create_dataset(dataset_type: str, **kwargs) -> ColmapDataset | InstantNGPDataset | MatrixCityDataset:
    """Factory for reconstruction datasets."""
    dataset_type_norm = dataset_type.strip().lower().replace("_", "-")
    if dataset_type_norm in {"colmap"}:
        return ColmapDataset(**kwargs)
    if dataset_type_norm in {"instant-ngp", "instantngp", "ngp"}:
        return InstantNGPDataset(**kwargs)
    if dataset_type_norm in {"matrixcity", "matrix-city"}:
        return MatrixCityDataset(**kwargs)
    raise ValueError(f"Unsupported dataset_type '{dataset_type}'. Supported: colmap, instant-ngp, matrixcity")
