import torch
import numpy as np
import pycolmap
import os
import logging
from math import ceil
import imageio.v2 as imageio
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF
from scipy.spatial.transform import Rotation
from pathlib import Path
from typing import Optional, Union, Tuple, cast
from warnings import warn
import cv2
import torch.nn.functional as F
from torch.nn.modules.utils import _pair
# Module-level logger
logger = logging.getLogger("cityscape_gs.dataset")

FullPadType = Tuple[int, int, int, int]
TuplePadType = Union[Tuple[int, int], FullPadType]
PadType = Union[int, TuplePadType]


def create_padding_tuple(padding: PadType, unpadding: bool = False) -> FullPadType:
    """Create argument for padding op."""
    padding = cast(TuplePadType, _pair(padding))

    if len(padding) not in [2, 4]:
        raise AssertionError(
            f"{'Unpadding' if unpadding else 'Padding'} must be either an int, tuple of two ints or tuple of four ints"
        )

    if len(padding) == 2:
        pad_vert = _pair(padding[0])
        pad_horz = _pair(padding[1])
    else:
        pad_vert = padding[:2]
        pad_horz = padding[2:]
    padding = cast(FullPadType, pad_horz + pad_vert)

    return padding


def compute_padding(
    original_size: Union[int, Tuple[int, int]],
    window_size: Union[int, Tuple[int, int]],
    stride: Optional[Union[int, Tuple[int, int]]] = None,
) -> FullPadType:
    """Compute required padding for patch extraction."""
    original_size = cast(Tuple[int, int], _pair(original_size))
    window_size = cast(Tuple[int, int], _pair(window_size))
    if stride is None:
        stride = window_size
    stride = cast(Tuple[int, int], _pair(stride))

    remainder_vertical = (original_size[0] - window_size[0]) % stride[0]
    remainder_horizontal = (original_size[1] - window_size[1]) % stride[1]

    if remainder_vertical != 0:
        vertical_padding = stride[0] - remainder_vertical
    else:
        vertical_padding = 0

    if remainder_horizontal != 0:
        horizontal_padding = stride[1] - remainder_horizontal
    else:
        horizontal_padding = 0

    if vertical_padding % 2 == 0:
        top_padding = bottom_padding = vertical_padding // 2
    else:
        top_padding = vertical_padding // 2
        bottom_padding = ceil(vertical_padding / 2)

    if horizontal_padding % 2 == 0:
        left_padding = right_padding = horizontal_padding // 2
    else:
        left_padding = horizontal_padding // 2
        right_padding = ceil(horizontal_padding / 2)

    return int(top_padding), int(bottom_padding), int(left_padding), int(right_padding)


def _check_patch_fit(original_size: Tuple[int, int], window_size: Tuple[int, int], stride: Tuple[int, int]) -> bool:
    remainder_vertical = (original_size[0] - window_size[0]) % stride[0]
    remainder_horizontal = (original_size[1] - window_size[1]) % stride[1]
    if (remainder_horizontal != 0) or (remainder_vertical != 0):
        return False
    return True


def _extract_tensor_patchesnd(
    input: torch.Tensor, window_sizes: Tuple[int, ...], strides: Tuple[int, ...]
) -> torch.Tensor:
    batch_size, num_channels = input.size()[:2]
    dims = range(2, input.dim())
    for dim, patch_size, stride in zip(dims, window_sizes, strides):
        input = input.unfold(dim, patch_size, stride)
    input = input.permute(0, *dims, 1, *(dim + len(dims) for dim in dims)).contiguous()
    return input.view(batch_size, -1, num_channels, *window_sizes)


def extract_tensor_patches(
    input: torch.Tensor,
    window_size: Union[int, Tuple[int, int]],
    stride: Union[int, Tuple[int, int]] = 1,
    padding: PadType = 0,
    allow_auto_padding: bool = False,
) -> torch.Tensor:
    """Extract patches from tensors and stack them as BxNxCxHxW."""
    if not torch.is_tensor(input):
        raise TypeError(f"Input input type is not a torch.Tensor. Got {type(input)}")

    if len(input.shape) != 4:
        raise ValueError(f"Invalid input shape, we expect BxCxHxW. Got: {input.shape}")

    window_size = cast(Tuple[int, int], _pair(window_size))
    stride = cast(Tuple[int, int], _pair(stride))
    original_size = (input.shape[-2], input.shape[-1])

    if not padding:
        if not _check_patch_fit(original_size, window_size, stride):
            if not allow_auto_padding:
                warn(
                    f"The window will not fit into the image. \nWindow size: {window_size}\nStride: {stride}\n"
                    f"Image size: {original_size}\n"
                    "This means that the final incomplete patches will be dropped. By enabling `allow_auto_padding`, "
                    "the input will be padded to fit the window and stride.",
                    stacklevel=1,
                )
            else:
                padding = compute_padding(original_size=original_size, window_size=window_size, stride=stride)

    if padding:
        padding = create_padding_tuple(padding)
        input = F.pad(input, padding)

    return _extract_tensor_patchesnd(input, window_size, stride)

class ColmapDataset(Dataset):
    def __init__(self, colmap_path, 
                 image_dir, device: torch.device=torch.device('cuda'), 
                 use_dataloader=True, 
                 point_cloud_extent_ratio: Optional[float] = None,
                 train_semantics=False, semantics_dim=3,
                 semantics_path: Optional[Path] = None,
                 semantics_resolution: Optional[Union[tuple[int, int], int]] = None,
                 patch_scale_factor = 8,
                 patch_stride: Optional[Union[int, tuple[int, int]]] = None,
                 patch_padding: PadType = 0,
                 patch_allow_auto_padding: bool = False,
                 ):
        """
        Dataset that loads COLMAP reconstructions and corresponding images on demand.
        Args:
            - colmap_path: Path to COLMAP reconstruction (sparse or dense)
            - image_dir: Directory containing the original images (must match COLMAP names)
            - device: torch.device to load tensors onto (default: 'cuda')
            - use_dataloader: If True, __getitem__ will load images on demand. If False, images will be pre-loaded into RAM (useful for small datasets).
            - point_cloud_extent_ratio: If set, prunes the initial point cloud to only include points within this fraction of the scene extent (useful for large outdoor scenes to focus on central areas).
            - train_semantics: If True, also loads semantic features for each image (if available) and includes them in the dataset items.
            - semantics_dim: Dimensionality of semantic features (e.g., 3 for RGB-based semantics, 128 for CLIP-based features).
            - semantics_path: Optional path to semantic features directory (should contain .npy files named after images).
            - semantics_resolution: Optional resolution to which semantic features should be resized (if they are image-based). If None assumes semantic features are already in the correct format and dimension.
        """
        # Handle both string and torch.device

        self.device = device
        self.use_dataloader = use_dataloader  # Don't move to CUDA in workers
        self._preloaded_images = None  # Cache for pre-loaded images
        self.train_semantics = train_semantics
        self.patch_scale_factor = patch_scale_factor
        self.patch_stride = patch_stride
        self.patch_padding = patch_padding
        self.patch_allow_auto_padding = patch_allow_auto_padding
        
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
            self.scene_extent = float(np.linalg.norm(pts_array.max(axis=0) - pts_array.min(axis=0)) * 5)  # Add some margin
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
            img_path = os.path.join(image_dir, img.name)
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
            
            self.cameras.append({
                'R': R, 'T': T,
                'fx': fx, 'fy': fy, 'cx': cx, 'cy': cy,
                'width': cam.width, 'height': cam.height,
                'image_path': img_path
            })

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
        
        # Load and optimize depth map on demand
        depth_tensor = self._load_depth(img_path, cam_data)

        semantics_output = (None, None)
        if self.train_semantics:
            semantics_output = self._load_semantics(img_path, img_tensor.detach().cpu())
            # semantics_output = (None, None)  # Placeholder for semantics loading (can be implemented similarly to depth loading with validation)

        patch_resolution = self._get_patch_resolution(img_tensor, self.patch_scale_factor)
        patch_stride = self.patch_stride if self.patch_stride is not None else patch_resolution

        patch_input = img_tensor.permute(2, 0, 1).unsqueeze(0)
        patches = extract_tensor_patches(
            patch_input,
            window_size=patch_resolution,
            stride=patch_stride,
            padding=self.patch_padding,
            allow_auto_padding=self.patch_allow_auto_padding,
        ).squeeze(0).permute(0, 2, 3, 1)  # [1, N, C, H, W] -> [N, H, W, C]

        return cam_data, img_tensor, depth_tensor, semantics_output, patches

    def _get_patch_resolution(self, image: torch.Tensor, scale_factor: int) -> tuple[int, int]:
        """Determine patch resolution based on image size and semantics resolution."""

        image_shape = image.shape  # [H, W, C]
        if image_shape[0] % int(scale_factor) != 0 or image_shape[1] % int(scale_factor) != 0:
            logger.warning(f"[yellow]⚠ Warning:[/yellow] Image dimensions {image_shape[:2]} are not perfectly divisible by patch scale factor {scale_factor}. This may lead to issues during training. Consider resizing images or choosing a different scale factor.")
            scale_factor = self._find_closest_divisible_patch_scale_factor(image, scale_factor)

        return (image.shape[0] // int(scale_factor), image.shape[1] // int(scale_factor))

    def _find_closest_divisible_patch_scale_factor(self, image: torch.Tensor, scale_factor: int) -> int:
        """Find the closest patch scale factor that results in integer patch resolution."""
        image_shape = image.shape  # [H, W, C]

        for scale in range(scale_factor, 30):
            if image_shape[0] % scale == 0 and image_shape[1] % scale == 0:
                patch_h = image_shape[0] // scale
                patch_w = image_shape[1] // scale
                logger.info(
                    f"[green]✓ Found divisible patch scale factor:[/green] {scale} "
                    f"for image size {image.shape[:2]} resulting in patch resolution ({patch_h}, {patch_w})"
                )
                return int(scale)

        logger.warning(
            f"[yellow]⚠ Warning:[/yellow] No divisible patch scale factor found for image size "
            f"{image.shape[:2]} with target scale factor {scale_factor}"
        )
        return int(scale_factor)

    @torch.no_grad()
    def _load_semantics(self, img_path: Path, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Load semantic features for the given image if available."""
        logger.debug(f"Attempting to load semantic features for {img_path.name} from {self.semantics_path}")
        semantics_file = self.semantics_path / f"{img_path.stem}_s.npy"
        if not semantics_file.exists():
            raise FileNotFoundError(f"Semantic features file not found for {img_path.name} at expected location: {semantics_file}")
        
        try:
            semantics = torch.tensor(np.load(semantics_file))
            semantics_image = np.asarray(image.cpu())  # Convert to numpy for resizing
            semantics_image = cv2.resize(semantics_image, semantics.shape[1:], interpolation=cv2.INTER_CUBIC)
            # semantics_image = TF.resize(image, semantics.shape[1:], interpolation=TF.InterpolationMode.BICUBIC)  # Resize to match semantics if needed
            return semantics.float(), torch.tensor(semantics_image)
        except Exception as e:
            raise RuntimeError(f"Failed to load semantic features for {img_path.name} from {semantics_file}: {e}")

    def _load_depth(self, img_path: Path, cam_data: dict) -> Optional[torch.Tensor]:
        """Load and normalize depth map with validation."""
        depth_path = img_path.parent.parent / "depths_npy" / f"{img_path.stem}.npy"
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
        
    def preload_all_images(self):
        """Pre-load all images into RAM for faster training.
        
        Keeps images on CPU to save VRAM. They will be moved to GPU during training.
        """
        self._preloaded_images = []
        for cam_data in self.cameras:
            img = imageio.imread(cam_data['image_path'])
            # Keep on CPU to avoid VRAM exhaustion
            img_tensor = torch.from_numpy(img).float() / 255.0
            self._preloaded_images.append(img_tensor)
        logger.info(f"[green]✓ Preloaded {len(self.cameras)} images to CPU RAM[/green]")

    def collate_fn(self, batch):
        """
        Custom collate function for DataLoader.
        Since each image can have different dimensions and camera parameters,
        we keep batch_size=1 and return the single item directly.
        This ensures numerical stability and proper handling of varying image sizes.
        """
        # batch is a list with single item: [(cam, gt_image)]
        return batch[0]
