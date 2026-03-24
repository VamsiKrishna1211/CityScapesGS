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
from plyfile import PlyData

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
    depth_dir: Optional[str] = None

    def __getitem__(self, key: str):
        return getattr(self, key)


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
            else:
                return None

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
                )
            )


class InstantNGPDataset(BaseReconstructionDataset):
    def __init__(
        self,
        transforms_path: Union[str, Path],
        image_dir: Optional[Union[str, Path]] = None,
        depth_dir: Optional[Union[str, Path]] = None,
        device: torch.device = torch.device("cuda"),
        use_dataloader=True,
        point_cloud_extent_ratio: Optional[float] = None,
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
            depth_dir=depth_dir,
            device=device,
            use_dataloader=use_dataloader,
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
    def _read_image_hw(image_path: Path) -> tuple[int, int]:
        """Read image size once when metadata intrinsics are missing."""
        img = imageio.imread(image_path)
        return int(img.shape[0]), int(img.shape[1])

    @staticmethod
    def _project_to_so3(rotation: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Project a 3x3 matrix to SO(3) and return singular values."""
        u, s, vt = np.linalg.svd(rotation.astype(np.float64))
        rot_ortho = u @ vt
        if np.linalg.det(rot_ortho) < 0:
            u[:, -1] *= -1
            rot_ortho = u @ vt
        return rot_ortho, s

    @staticmethod
    def _rigidify_w2c_preserve_center(extrinsic: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        """Project affine w2c to rigid form while preserving camera center.

        Returns:
            rigid_w2c, camera_center, rotation_det
        """
        if extrinsic.shape != (4, 4):
            raise ValueError(f"Expected 4x4 pose matrix, got {extrinsic.shape}")
        if not np.isfinite(extrinsic).all():
            raise ValueError("Pose matrix contains non-finite values")

        affine_rot = extrinsic[:3, :3].astype(np.float64)
        affine_t = extrinsic[:3, 3].astype(np.float64)

        rot_ortho, _ = InstantNGPDataset._project_to_so3(affine_rot)
        try:
            cam_center = -np.linalg.solve(affine_rot, affine_t)
        except np.linalg.LinAlgError:
            if np.linalg.matrix_rank(affine_rot) < 3:
                raise ValueError("Pose rotation block is singular or near-singular")
            cam_center = -np.linalg.pinv(affine_rot) @ affine_t

        t_rigid = -rot_ortho @ cam_center

        rigid = np.eye(4, dtype=np.float32)
        rigid[:3, :3] = rot_ortho.astype(np.float32)
        rigid[:3, 3] = t_rigid.astype(np.float32)
        return rigid, cam_center.astype(np.float32), float(np.linalg.det(rot_ortho))

    @staticmethod
    def _matrixcity_rotmat_to_rigid_w2c(rot_mat_4x4: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        """Convert MatrixCity-style rot_mat (which is actually c2w with scale) to rigid w2c.

        MatrixCity export conventions encodue uniform scale in the 3x3 block (e.g., 0.01)
        and translation in centimeters. We detect the scale from singular values,
        apply it to translation to convert to meters, and then invert the rigid c2w 
        into w2c.
        """
        c2w_affine = np.asarray(rot_mat_4x4, dtype=np.float64)
        if c2w_affine.shape != (4, 4):
            raise ValueError(f"Invalid rot_mat shape: {c2w_affine.shape}")
        if not np.isfinite(c2w_affine).all():
            raise ValueError("rot_mat contains non-finite values")

        rot_ortho, singular_values = InstantNGPDataset._project_to_so3(c2w_affine[:3, :3])
        uniform_scale = float(np.mean(np.abs(singular_values)))
        if not np.isfinite(uniform_scale) or uniform_scale <= 1e-8:
            raise ValueError("Could not infer a valid uniform scale from rot_mat")

        c2w_rigid = np.eye(4, dtype=np.float32)
        c2w_rigid[:3, :3] = rot_ortho.astype(np.float32)

        t_c2w = c2w_affine[:3, 3]
        if abs(uniform_scale - 1.0) > 1e-6:
            t_c2w = t_c2w * uniform_scale
        c2w_rigid[:3, 3] = t_c2w.astype(np.float32)

        w2c_rigid = np.eye(4, dtype=np.float32)
        R_w2c = c2w_rigid[:3, :3].T
        w2c_rigid[:3, :3] = R_w2c
        w2c_rigid[:3, 3] = -R_w2c @ c2w_rigid[:3, 3]

        cam_center = c2w_rigid[:3, 3]
        return w2c_rigid, cam_center, float(np.linalg.det(rot_ortho))

    @staticmethod
    def _frame_to_w2c_and_center(frame: dict, frame_name: str) -> tuple[np.ndarray, np.ndarray, str, float]:
        """Resolve per-frame pose to rigid w2c and camera center.

        Basic convention used here:
        - If ``rot_mat`` exists, treat it as c2w (Matrix-City compatibility).
        - Else, treat ``transform_matrix`` as c2w (Instant-NGP standard).
        """
        has_rot = "rot_mat" in frame and frame.get("rot_mat") is not None
        has_transform = "transform_matrix" in frame and frame.get("transform_matrix") is not None

        if has_rot:
            try:
                w2c, cam_center, det_r = InstantNGPDataset._matrixcity_rotmat_to_rigid_w2c(frame["rot_mat"])
                return w2c, cam_center, "rot_mat", det_r
            except Exception as exc:
                if has_transform:
                    raise ValueError(
                        f"Frame {frame_name} has both rot_mat and transform_matrix, but rot_mat is invalid. "
                        "MatrixCity-strict precedence refuses fallback to transform_matrix. "
                        f"rot_mat error: {exc}"
                    ) from exc
                raise ValueError(f"Invalid rot_mat for frame {frame_name}: {exc}") from exc

        if has_transform:
            c2w = np.asarray(frame["transform_matrix"], dtype=np.float32)
            if c2w.shape != (4, 4):
                raise ValueError(f"Invalid transform_matrix shape for frame {frame_name}: {c2w.shape}")
            if not np.isfinite(c2w).all():
                raise ValueError(f"Non-finite transform_matrix values for frame {frame_name}")
            try:
                w2c_affine = np.linalg.inv(c2w).astype(np.float32)
            except np.linalg.LinAlgError as exc:
                raise ValueError(f"Singular transform_matrix for frame {frame_name}") from exc

            w2c, cam_center, det_r = InstantNGPDataset._rigidify_w2c_preserve_center(w2c_affine)
            return w2c, cam_center, "transform_matrix", det_r

        raise ValueError(
            f"No extrinsics found for frame {frame_name}. "
            "Expected one of: rot_mat, transform_matrix"
        )

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
        inferred_hw_cache: dict[str, tuple[int, int]] = {}
        pose_source_counts = {"rot_mat": 0, "transform_matrix": 0}
        det_values: list[float] = []

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

            frame_name = file_path if file_path is not None else f"frame_index={frame_index}"
            w2c, cam_center, pose_source, det_r = self._frame_to_w2c_and_center(frame, frame_name)

            r_tensor = torch.tensor(w2c[:3, :3], dtype=torch.float32)
            t_tensor = torch.tensor(w2c[:3, 3], dtype=torch.float32)
            camera_centers.append(cam_center)
            pose_source_counts[pose_source] += 1
            det_values.append(det_r)

            width = int(frame.get("w", meta.get("w", 0)))
            height = int(frame.get("h", meta.get("h", 0)))
            if width <= 0 or height <= 0:
                cache_key = str(img_path)
                if cache_key not in inferred_hw_cache:
                    inferred_hw_cache[cache_key] = self._read_image_hw(img_path)
                height, width = inferred_hw_cache[cache_key]

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

        det_abs_error = np.abs(np.asarray(det_values, dtype=np.float64) - 1.0)
        logger.info(
            "[cyan]Instant-NGP pose summary:[/cyan] "
            f"rot_mat={pose_source_counts['rot_mat']}, "
            f"transform_matrix={pose_source_counts['transform_matrix']}, "
            f"max|det(R)-1|={float(det_abs_error.max()) if len(det_abs_error) > 0 else 0.0:.3e}"
        )

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
        # rgb = np.random.uniform(0.0, 255.0, size=(int(fallback_init_points), 3)).astype(np.float32)
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

        if suffix == ".ply":
            try:
                from plyfile import PlyData
            except ImportError as exc:
                raise ImportError(
                    "Reading .ply point clouds requires 'plyfile'. Install with: pip install plyfile"
                ) from exc

            ply_data = PlyData.read(str(cloud_path))
            if "vertex" not in ply_data:
                raise ValueError(f"PLY file has no 'vertex' element: {cloud_path}")

            vertices = ply_data["vertex"].data
            required_xyz = {"x", "y", "z"}
            available = set(vertices.dtype.names or [])
            if not required_xyz.issubset(available):
                raise ValueError(
                    "PLY vertex element must contain x/y/z properties. "
                    f"Found: {sorted(available)}"
                )

            xyz = np.stack(
                [
                    np.asarray(vertices["x"], dtype=np.float32),
                    np.asarray(vertices["y"], dtype=np.float32),
                    np.asarray(vertices["z"], dtype=np.float32),
                ],
                axis=1,
            )

            rgb = None
            rgb_candidates = [
                ("red", "green", "blue"),
                ("r", "g", "b"),
            ]
            for r_name, g_name, b_name in rgb_candidates:
                if {r_name, g_name, b_name}.issubset(available):
                    rgb = np.stack(
                        [
                            np.asarray(vertices[r_name], dtype=np.float32),
                            np.asarray(vertices[g_name], dtype=np.float32),
                            np.asarray(vertices[b_name], dtype=np.float32),
                        ],
                        axis=1,
                    )
                    # Some pipelines store colors in [0, 1] floats.
                    if rgb.size > 0 and float(np.nanmax(rgb)) <= 1.0:
                        rgb = rgb * 255.0
                    break

            return cls._normalize_point_cloud_arrays(xyz=xyz, rgb=rgb)

        raise ValueError(
            f"Unsupported point cloud file type '{suffix}' for {cloud_path}. "
            "Supported: .npy, .pth, .pt, .ply"
        )


class MatrixCityDataset(InstantNGPDataset):
    """MatrixCity multi-block dataset built on InstantNGP pose/intrinsic parsing.

    Supports aggregating frames from multiple block folders (for example,
    aerial/train/block_1, aerial/train/block_2, ...). Each block must contain
    a transforms file (transform.json or transforms.json) and image folder.
    """

    def __init__(
        self,
        matrixcity_paths: list[Union[str, Path]],
        matrixcity_depth_paths: Optional[list[Union[str, Path]]] = None,
        device: torch.device = torch.device("cuda"),
        use_dataloader=True,
        point_cloud_extent_ratio: Optional[float] = None,
        scene_extent_margin: float = 2.0,
        image_scale: int = 2,
        require_depth: bool = True,
        matrixcity_pointcloud_paths: Optional[list[Union[str, Path]]] = None,
        matrixcity_max_init_points: int = 20000,
        fallback_init_points: int = 20000,
    ):
        if not matrixcity_paths:
            raise ValueError("matrixcity_paths cannot be empty")

        self.transforms_paths = [
            self._resolve_transforms_path(path)
            for path in matrixcity_paths
        ]
        self.scene_roots = [path.parent for path in self.transforms_paths]

        self.matrixcity_depth_paths = None
        if matrixcity_depth_paths is not None:
            if len(matrixcity_depth_paths) != len(self.transforms_paths):
                raise ValueError(
                    "matrixcity_depth_paths must match matrixcity_paths length. "
                    f"Got depths={len(matrixcity_depth_paths)} and blocks={len(self.transforms_paths)}"
                )
            self.matrixcity_depth_paths = [Path(path).expanduser().resolve() for path in matrixcity_depth_paths]

        first_image_root = self._resolve_image_root(image_dir=None, transforms_path=self.transforms_paths[0])

        # MatrixCity uses per-block image/depth roots, so global depth resolution is disabled here.
        BaseReconstructionDataset.__init__(
            self,
            image_dir=first_image_root,
            depth_dir=None,
            device=device,
            use_dataloader=use_dataloader,
            image_scale=image_scale,
            require_depth=False,
        )
        self.require_depth = require_depth
        self.matrixcity_max_init_points = int(matrixcity_max_init_points)
        if self.matrixcity_max_init_points <= 0:
            raise ValueError("matrixcity_max_init_points must be > 0")

        self._load_matrixcity_blocks(
            point_cloud_extent_ratio=point_cloud_extent_ratio,
            scene_extent_margin=scene_extent_margin,
            point_cloud_paths=matrixcity_pointcloud_paths,
            fallback_init_points=fallback_init_points,
        )

    def _load_matrixcity_blocks(
        self,
        point_cloud_extent_ratio: Optional[float],
        scene_extent_margin: float,
        point_cloud_paths: Optional[list[Union[str, Path]]],
        fallback_init_points: int,
    ) -> None:
        scale_factor = 1.0 / float(self.image_scale)
        camera_centers = []
        inferred_hw_cache: dict[str, tuple[int, int]] = {}
        pose_source_counts = {"rot_mat": 0, "transform_matrix": 0}
        det_values: list[float] = []

        total_frames_seen = 0
        total_frames_kept = 0
        depth_dirs_seen = set()

        max_points = self.matrixcity_max_init_points

        for block_idx, transforms_path in enumerate(self.transforms_paths):
            with open(transforms_path, "r", encoding="utf-8") as f:
                meta = json.load(f)

            frames = meta.get("frames", [])
            if len(frames) == 0:
                logger.warning(f"[yellow]No frames in block transforms:[/yellow] {transforms_path}")
                continue

            image_root = self._resolve_image_root(image_dir=None, transforms_path=transforms_path)
            image_dir = self._resolve_image_dir(image_root, self.image_scale)

            depth_override = None
            if self.matrixcity_depth_paths is not None:
                depth_override = self.matrixcity_depth_paths[block_idx]

            try:
                depth_dir = self._resolve_depth_dir(
                    image_dir=image_dir,
                    image_scale=self.image_scale,
                    depth_dir_override=depth_override,
                    require_depth=self.require_depth,
                )
            except FileNotFoundError:
                depth_dir = None

            if depth_dir is not None:
                depth_dirs_seen.add(str(depth_dir))

            logger.debug(f"Number of frames in block {block_idx} ({transforms_path}): {len(frames)}")
            for frame in frames:
                total_frames_seen += 1

                file_path = frame.get("file_path")
                frame_index = frame.get("frame_index")

                img_path = None
                if file_path is not None:
                    img_path = self._resolve_frame_image_path(image_dir, transforms_path.parent, file_path)
                elif frame_index is not None:
                    img_path = self._resolve_frame_image_path_from_index(image_dir, int(frame_index))

                if img_path is None or not img_path.exists():
                    continue

                frame_name = file_path if file_path is not None else f"frame_index={frame_index}"
                w2c, cam_center, pose_source, det_r = self._frame_to_w2c_and_center(frame, frame_name)

                r_tensor = torch.tensor(w2c[:3, :3], dtype=torch.float32)
                t_tensor = torch.tensor(w2c[:3, 3], dtype=torch.float32)
                camera_centers.append(cam_center)
                pose_source_counts[pose_source] += 1
                det_values.append(det_r)

                width = int(frame.get("w", meta.get("w", 0)))
                height = int(frame.get("h", meta.get("h", 0)))
                if width <= 0 or height <= 0:
                    cache_key = str(img_path)
                    if cache_key not in inferred_hw_cache:
                        inferred_hw_cache[cache_key] = self._read_image_hw(img_path)
                    height, width = inferred_hw_cache[cache_key]

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
                        depth_dir=str(depth_dir) if depth_dir is not None else None,
                    )
                )
                total_frames_kept += 1

        # Load and process point clouds
        if point_cloud_paths:
            all_points = []
            all_colors = []

            for cloud_path in point_cloud_paths:
                cloud_path = Path(cloud_path).expanduser().resolve()
                if not cloud_path.exists():
                    logger.warning(f"[yellow]Point cloud file not found:[/yellow] {cloud_path}")
                    continue

                ply = PlyData.read(str(cloud_path))
                v = ply["vertex"].data
                points = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float32)

                color_fields = {"red", "green", "blue"}
                if color_fields.issubset(v.dtype.names):
                    colors = np.stack([v["red"], v["green"], v["blue"]], axis=1).astype(np.float32)
                    if colors.max() > 1.0:
                        colors /= 255.0
                else:
                    colors = np.ones((points.shape[0], 3), dtype=np.float32)  # fallback: white

                if points.shape[0] > max_points:
                    rng = np.random.default_rng()
                    sample_idx = rng.choice(points.shape[0], size=max_points, replace=False)
                    points = points[sample_idx]
                    colors = colors[sample_idx]

                all_points.append(points)
                all_colors.append(colors)

            if all_points:
                combined_points = np.concatenate(all_points, axis=0)
                combined_colors = np.concatenate(all_colors, axis=0)

                if combined_points.shape[0] > max_points:
                    rng = np.random.default_rng()
                    sample_idx = rng.choice(combined_points.shape[0], size=max_points, replace=False)
                    combined_points = combined_points[sample_idx]
                    combined_colors = combined_colors[sample_idx]

                logger.info(
                    "[cyan]Loaded point clouds:[/cyan] "
                    f"total points (after cap)={combined_points.shape[0]}, cap={max_points}"
                )

                self._set_init_points_and_colors(
                    xyz=combined_points,
                    rgb_uint8=(combined_colors * 255).astype(np.uint8),
                    scene_extent_margin=scene_extent_margin,
                    point_cloud_extent_ratio=point_cloud_extent_ratio,
                )

        if len(self.cameras) == 0:
            raise RuntimeError(
                "No valid MatrixCity frames with existing images were found across input blocks. "
                f"Blocks checked: {len(self.transforms_paths)}"
            )

        det_abs_error = np.abs(np.asarray(det_values, dtype=np.float64) - 1.0)
        logger.info(
            "[cyan]MatrixCity pose summary:[/cyan] "
            f"blocks={len(self.transforms_paths)}, "
            f"frames={total_frames_kept}/{total_frames_seen}, "
            f"rot_mat={pose_source_counts['rot_mat']}, "
            f"transform_matrix={pose_source_counts['transform_matrix']}, "
            f"depth_dirs={len(depth_dirs_seen)}, "
            f"max|det(R)-1|={float(det_abs_error.max()) if len(det_abs_error) > 0 else 0.0:.3e}"
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
