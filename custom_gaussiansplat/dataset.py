import torch
import numpy as np
import pycolmap
import os
import logging
import imageio.v2 as imageio
from torch.utils.data import Dataset
from scipy.spatial.transform import Rotation
from pathlib import Path
from typing import Optional
from rich.console import Console
from rich.progress import (
    Progress,
    SpinnerColumn,
    BarColumn,
    TaskProgressColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    TextColumn,
)
# Module-level logger
logger = logging.getLogger("cityscape_gs.dataset")
console = Console()

class ColmapDataset(Dataset):
    def __init__(self, colmap_path, 
                 image_dir, device: torch.device=torch.device('cuda'), 
                 use_dataloader=True, 
                 point_cloud_extent_ratio: Optional[float] = None,
                 scene_extent_margin: float = 2.0,
                 image_scale: int = 2,
                 ):
        """
        Dataset that loads COLMAP reconstructions and corresponding images on demand.
        Args:
            - colmap_path: Path to COLMAP reconstruction (sparse or dense)
            - image_dir: Directory containing the original images (must match COLMAP names)
            - device: torch.device to load tensors onto (default: 'cuda')
            - use_dataloader: If True, __getitem__ will load images on demand. If False, images will be pre-loaded into RAM (useful for small datasets).
            - point_cloud_extent_ratio: If set, prunes the initial point cloud to only include points within this fraction of the scene extent (useful for large outdoor scenes to focus on central areas).
            - image_scale: Downscale factor used for training images. If image_dir points to `images`, this resolves to `images_<image_scale>` for scale > 1.
        """
        # Handle both string and torch.device

        self.device = device
        self.use_dataloader = use_dataloader  # Don't move to CUDA in workers
        self._preloaded_images = None  # Cache for pre-loaded images
        self._preloaded_depths = None  # Cache for pre-loaded depth maps (can contain None entries)

        if image_scale < 1:
            raise ValueError(f"image_scale must be >= 1, got {image_scale}")
        self.image_scale = int(image_scale)
        image_dir_path = self._resolve_image_dir(Path(image_dir), self.image_scale)
        logger.info(f"[cyan]Using image scale:[/cyan] {self.image_scale} (directory: {image_dir_path})")
        self.depth_dir = self._resolve_depth_dir(image_dir_path, self.image_scale)
        logger.info(f"[cyan]Using depth scale:[/cyan] {self.image_scale} (directory: {self.depth_dir})")
        
        # 1. Load Reconstruction
        # pycolmap handles the binary/text parsing automatically
        recon = pycolmap.Reconstruction(colmap_path)
        
        # 2. Extract Point Cloud (for initialization)
        # We need initial means (XYZ) and colors (RGB)
        pts3d = recon.points3D
        xyz = []
        rgb = []
        for p_id, p in pts3d.items():
            xyz.append(p.xyz)
            rgb.append(p.color)
        
        self.init_points = torch.tensor(np.array(xyz), dtype=torch.float32, device=device)
        
        # Convert RGB colors to spherical harmonics DC coefficients
        # SH_C0 is the 0th order SH coefficient: 1/(2*sqrt(pi))
        SH_C0 = 0.28209479177387814
        rgb_normalized = torch.tensor(np.array(rgb), dtype=torch.float32, device=device) / 255.0  # [0, 1]
        # Convert to SH: center around 0.5 and divide by SH constant
        self.init_colors = (rgb_normalized - 0.5) / SH_C0
        
        # Compute scene extent (for densification thresholds)
        if len(xyz) > 0:
            pts_array = np.array(xyz)
            self.scene_extent = float(np.linalg.norm(pts_array.max(axis=0) - pts_array.min(axis=0)) * scene_extent_margin)  # Add some margin
        else:
            logger.warning("[yellow]⚠ Warning:[/yellow] No 3D points found in COLMAP reconstruction. Setting scene extent to 1.0")
            self.scene_extent = float(1.0)

        # Prune point cloud to a fraction of scene extent if requested
        if point_cloud_extent_ratio is not None and len(xyz) > 0:
            pts_array = np.array(xyz)
            scene_center = pts_array.mean(axis=0)
            max_dist = np.linalg.norm(pts_array - scene_center, axis=1).max()
            threshold = point_cloud_extent_ratio * max_dist
            dists = np.linalg.norm(pts_array - scene_center, axis=1)
            mask = dists <= threshold
            n_before = len(xyz)
            self.init_points = self.init_points[mask]
            self.init_colors = self.init_colors[mask]
            logger.info(f"[cyan]Point cloud pruned:[/cyan] kept {mask.sum()} / {n_before} points within {point_cloud_extent_ratio*100:.1f}% of scene extent (threshold={threshold:.3f})")

        # 3. Extract Cameras & Images
        self.cameras = []
        for img_id, img in recon.images.items():
            cam = recon.cameras[img.camera_id]
            
            # Skip if image file doesn't exist
            img_path = str(image_dir_path / img.name)
            if not os.path.exists(img_path): continue

            # Extrinsics: World-to-Camera
            # pycolmap stores rotation as quaternion (qw, qx, qy, qz)
            # Convert quaternion to rotation matrix
            cam_from_world = img.cam_from_world()
            quat = cam_from_world.rotation.quat
            R = Rotation.from_quat(quat, scalar_first=False).as_matrix()
            R = torch.tensor(R, dtype=torch.float32,)
            T = torch.tensor(cam_from_world.translation, dtype=torch.float32,)
            
            # Intrinsics
            # Handle different COLMAP camera models
            # pycolmap uses model.name instead of model_id
            model_name = cam.model.name
            
            if model_name == 'SIMPLE_PINHOLE':  # f, cx, cy
                fx = fy = cam.params[0]
                cx, cy = cam.params[1], cam.params[2]
            elif model_name == 'PINHOLE':  # fx, fy, cx, cy
                fx, fy, cx, cy = cam.params[0], cam.params[1], cam.params[2], cam.params[3]
            elif model_name in ['SIMPLE_RADIAL', 'RADIAL', 'OPENCV']:  # fx, fy, cx, cy, ...
                fx, fy, cx, cy = cam.params[0], cam.params[1], cam.params[2], cam.params[3]
            else:
                # Default: assume first 4 params are focal lengths and principal point
                fx = cam.params[0] if len(cam.params) > 0 else cam.width
                fy = cam.params[1] if len(cam.params) > 1 else fx
                cx = cam.params[2] if len(cam.params) > 2 else cam.width / 2
                cy = cam.params[3] if len(cam.params) > 3 else cam.height / 2
            
            scale_factor = 1.0 / float(self.image_scale)
            self.cameras.append({
                'R': R, 'T': T,
                'fx': float(fx) * scale_factor,
                'fy': float(fy) * scale_factor,
                'cx': float(cx) * scale_factor,
                'cy': float(cy) * scale_factor,
                'width': max(1, int(round(float(cam.width) * scale_factor))),
                'height': max(1, int(round(float(cam.height) * scale_factor))),
                'image_path': img_path
            })

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
    def _resolve_depth_dir(image_dir: Path, image_scale: int) -> Path:
        """Resolve depth directory using the same scale convention as images."""
        scene_root = image_dir.parent
        depth_dir = scene_root / "depths_npy"
        if image_scale > 1:
            depth_dir = scene_root / f"depths_npy_{image_scale}"

        if not depth_dir.exists():
            raise FileNotFoundError(
                f"Requested depth scale {image_scale}, but directory not found: {depth_dir}. "
                "Expected depth folders following image scale convention (e.g., depths_npy, depths_npy_2, depths_npy_4)."
            )

        return depth_dir


    def __len__(self):
        return len(self.cameras)
    
    @staticmethod
    def _qvec_to_rotmat(qvec):
        """
        Convert quaternion to rotation matrix.
        Args:
            qvec: quaternion [qw, qx, qy, qz]
        Returns:
            3x3 rotation matrix
        """
        qw, qx, qy, qz = qvec
        R = np.array([
            [1 - 2*qy**2 - 2*qz**2, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw],
            [2*qx*qy + 2*qz*qw, 1 - 2*qx**2 - 2*qz**2, 2*qy*qz - 2*qx*qw],
            [2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx**2 - 2*qy**2]
        ])
        return R

    def __getitem__(self, idx):
        # Load image on demand to save RAM
        cam_data = self.cameras[idx]
        
        # Use pre-loaded image if available
        img_path = Path(cam_data['image_path'])
        if self._preloaded_images is not None:
            img_tensor = self._preloaded_images[idx]
        else:
            img = imageio.imread(img_path)
            img_tensor = torch.from_numpy(img).float() / 255.0 # [0, 1] range [H, W, C]
        
        # Use pre-loaded depth if available, otherwise load on demand.
        if self._preloaded_depths is not None:
            depth_tensor = self._preloaded_depths[idx]
        else:
            depth_tensor = self._load_depth(img_path, cam_data)
        depth_tensor = depth_tensor.to(self.device) if depth_tensor is not None else None  # Move to device if loaded on demand

        return cam_data, img_tensor.to(self.device), depth_tensor

    def _load_depth(self, img_path: Path, cam_data: dict) -> Optional[torch.Tensor]:
        """Load and normalize depth map with validation."""
        depth_path = self.depth_dir / f"{img_path.stem}.npy"
        if not depth_path.exists():
            return None
            
        try:
            depth_map = np.load(depth_path)
            
            # Fast shape and finite check
            if depth_map.shape[:2] != (cam_data['height'], cam_data['width']):
                return None
            
            # Efficient validation using numpy
            if not np.isfinite(depth_map).all():
                # Only handle NaN/Inf if present (rare but slow to check every pixel)
                invalid_mask = ~np.isfinite(depth_map)
                if invalid_mask.all(): return None
                depth_map[invalid_mask] = np.median(depth_map[~invalid_mask])
            
            # Min-Max normalization for numerical stability
            d_min, d_max = depth_map.min(), depth_map.max()
            d_range = d_max - d_min
            
            if d_range < 1e-8:
                return None
                
            # depth_map = (depth_map - d_min) / (d_range + 1e-8)
            return torch.from_numpy(depth_map).float() # Depth tensor may be None & if returned, is the raw value without normalization (handled in loss function)
                
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
                img_path = Path(cam_data['image_path'])
                img = imageio.imread(img_path)
                # Keep on CPU to avoid VRAM exhaustion
                img_tensor = torch.from_numpy(img).float() / 255.0
                self._preloaded_images.append(img_tensor.to(self.device if not self.use_dataloader else torch.device('cpu')))  # Keep on CPU if using DataLoader to avoid VRAM issues

                depth_tensor = self._load_depth(img_path, cam_data)
                self._preloaded_depths.append(depth_tensor.to(self.device if not self.use_dataloader else torch.device('cpu')) if depth_tensor is not None else None)  # Keep on CPU if using DataLoader
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
        """
        Custom collate function for DataLoader.
        Since each image can have different dimensions and camera parameters,
        we keep batch_size=1 and return the single item directly.
        This ensures numerical stability and proper handling of varying image sizes.
        """
        # batch is a list with single item: [(cam, gt_image)]
        return batch[0]
