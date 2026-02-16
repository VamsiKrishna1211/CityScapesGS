import torch
import numpy as np
import pycolmap
import os
import logging
import imageio.v2 as imageio
from torch.utils.data import Dataset
from scipy.spatial.transform import Rotation
from pathlib import Path

# Module-level logger
logger = logging.getLogger("cityscape_gs.dataset")

class ColmapDataset(Dataset):
    def __init__(self, colmap_path, image_dir, device='cuda', use_dataloader=True):
        # Handle both string and torch.device
        if isinstance(device, torch.device):
            device = str(device).replace('device(', '').replace(')', '').replace("'", '')
        self.device = device
        self.use_dataloader = use_dataloader  # Don't move to CUDA in workers
        self._preloaded_images = None  # Cache for pre-loaded images
        
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
            self.scene_extent = np.linalg.norm(pts_array.max(axis=0) - pts_array.min(axis=0)) * 5  # Add some margin
        else:
            logger.warning("[yellow]⚠ Warning:[/yellow] No 3D points found in COLMAP reconstruction. Setting scene extent to 1.0")
            self.scene_extent = 1.0

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
            # When using DataLoader workers, keep on CPU to avoid CUDA fork issues
            if self.use_dataloader:
                img_tensor = torch.from_numpy(img).float() / 255.0
            else:
                img_tensor = torch.from_numpy(img).float() / 255.0
        
        # Load depth map if available (from Depth Anything V2)
        depth_path = img_path.parent.parent / "depths_npy" / f"{img_path.stem}.npy"
        if depth_path.exists():
            try:
                depth_map = np.load(depth_path)
                
                # Validate depth map shape
                if depth_map.shape[:2] != (cam_data['height'], cam_data['width']):
                    logger.warning(
                        f"Depth map shape mismatch for {img_path.name}: "
                        f"expected {(cam_data['height'], cam_data['width'])}, "
                        f"got {depth_map.shape[:2]}. Skipping depth."
                    )
                    depth_tensor = None
                    return cam_data, img_tensor, depth_tensor
                
                # Check for invalid values
                invalid_mask = ~np.isfinite(depth_map)
                invalid_count = invalid_mask.sum()
                
                if invalid_count > 0:
                    if invalid_count == depth_map.size:
                        # All invalid - skip this depth map
                        logger.warning(f"All depth values are NaN/Inf for {img_path.name}. Skipping depth.")
                        depth_tensor = None
                        return cam_data, img_tensor, depth_tensor
                    
                    # Replace invalid values with valid statistics
                    valid_values = depth_map[~invalid_mask]
                    median_value = np.median(valid_values)
                    depth_map[invalid_mask] = median_value
                    
                    if invalid_count > depth_map.size * 0.1:  # More than 10% invalid
                        logger.warning(
                            f"Large number of invalid depth values ({invalid_count}/{depth_map.size}) "
                            f"in {img_path.name}. Replaced with median."
                        )
                
                # Validate depth range
                depth_min = float(depth_map.min())
                depth_max = float(depth_map.max())
                depth_range = depth_max - depth_min
                
                if depth_range < 1e-8:
                    # Constant depth map - probably an error
                    logger.warning(
                        f"Depth map has no variation for {img_path.name} "
                        f"(min={depth_min:.6f}, max={depth_max:.6f}). Skipping depth."
                    )
                    depth_tensor = None
                    return cam_data, img_tensor, depth_tensor
                
                # Check for unreasonable depth values
                if depth_max > 1e6 or depth_min < -1e6:
                    logger.warning(
                        f"Extreme depth values for {img_path.name}: "
                        f"[{depth_min:.2f}, {depth_max:.2f}]. May cause numerical issues."
                    )
                
                # Normalize to [0, 1] range for better stability
                depth_map = (depth_map - depth_min) / (depth_range + 1e-8)
                
                # Final validation after normalization
                if not np.isfinite(depth_map).all():
                    logger.error(f"Depth normalization produced NaN/Inf for {img_path.name}. Skipping depth.")
                    depth_tensor = None
                    return cam_data, img_tensor, depth_tensor
                
                depth_tensor = torch.from_numpy(depth_map).float()
                
            except Exception as e:
                logger.error(f"Failed to load depth map for {img_path.name}: {str(e)}. Skipping depth.")
                depth_tensor = None
        else:
            depth_tensor = None
        
        return cam_data, img_tensor, depth_tensor
    
    def preload_all_images(self):
        """Pre-load all images into RAM for faster training."""
        self._preloaded_images = []
        for cam_data in self.cameras:
            img = imageio.imread(cam_data['image_path'])
            img_tensor = torch.from_numpy(img).float().to(self.device) / 255.0
            self._preloaded_images.append(img_tensor)

    def collate_fn(self, batch):
        """
        Custom collate function for DataLoader.
        Since each image can have different dimensions and camera parameters,
        we keep batch_size=1 and return the single item directly.
        This ensures numerical stability and proper handling of varying image sizes.
        """
        # batch is a list with single item: [(cam, gt_image)]
        return batch[0]
