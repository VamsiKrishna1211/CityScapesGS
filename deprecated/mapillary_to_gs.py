#!/usr/bin/env python3
"""
Mapillary to Gaussian Splatting Converter

Converts Mapillary point cloud data and street view images to formats compatible 
with Gaussian Splatting frameworks like 3D Gaussian Splatting, LongSplat, etc.

Usage:
    python mapillary_to_gs.py --pointcloud path/to/pointcloud.json --images path/to/images --output gs_data
"""

import json
import numpy as np
import open3d as o3d
import os
import cv2
from pathlib import Path
import argparse
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass
from scipy.spatial.transform import Rotation as R
import shutil
from PIL import Image
import glob


@dataclass
class CameraInfo:
    """Camera information for Gaussian Splatting"""
    image_id: str
    image_path: str
    camera_matrix: np.ndarray
    rotation_matrix: np.ndarray
    translation_vector: np.ndarray
    width: int
    height: int
    focal_length: float
    distortion_coeffs: np.ndarray = None


class MapillaryToGaussianSplattingConverter:
    """
    Converts Mapillary point cloud data to Gaussian Splatting format
    """
    
    def __init__(self, output_dir: str = "gaussian_splatting_data"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Create subdirectories
        self.images_dir = self.output_dir / "images"
        self.sparse_dir = self.output_dir / "sparse" / "0"
        self.processed_dir = self.output_dir / "processed"
        
        self.images_dir.mkdir(exist_ok=True)
        self.sparse_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(exist_ok=True)
        
    def load_mapillary_data(self, pointcloud_file: str) -> Dict:
        """Load Mapillary point cloud JSON data"""
        try:
            with open(pointcloud_file, 'r') as f:
                data = json.load(f)
            
            # Handle both single reconstruction and list of reconstructions
            if isinstance(data, list) and len(data) > 0:
                data = data[0]  # Take the first reconstruction
                
            self.logger.info(f"Loaded Mapillary data from {pointcloud_file}")
            return data
        except Exception as e:
            self.logger.error(f"Error loading point cloud data: {e}")
            return {}
    
    def extract_3d_points(self, data: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """Extract 3D points and colors from Mapillary data"""
        points = []
        colors = []
        
        if 'points' in data:
            self.logger.info(f"Found {len(data['points'])} points in reconstruction")
            
            for point_id, point_data in data['points'].items():
                # Extract coordinates
                coords = point_data['coordinates']
                points.append([coords[0], coords[1], coords[2]])
                
                # Extract RGB color (normalize to 0-1 range)
                color = point_data['color']
                colors.append([color[0]/255.0, color[1]/255.0, color[2]/255.0])
        
        points = np.array(points) if points else np.empty((0, 3))
        colors = np.array(colors) if colors else np.empty((0, 3))
        
        self.logger.info(f"Extracted {len(points)} 3D points with colors")
        return points, colors
    
    def extract_camera_data(self, data: Dict) -> Tuple[Dict[str, CameraInfo], Dict[str, dict]]:
        """Extract camera poses and intrinsics from Mapillary data"""
        cameras = {}
        camera_models = {}
        
        # First, extract camera models
        if 'cameras' in data:
            for cam_model_id, cam_model in data['cameras'].items():
                camera_models[cam_model_id] = cam_model
                self.logger.info(f"Camera model: {cam_model_id} - {cam_model}")
        
        # Then extract shots (camera poses)
        if 'shots' in data:
            self.logger.info(f"Found {len(data['shots'])} camera shots")
            
            for shot_id, shot_data in data['shots'].items():
                try:
                    # Get camera model for this shot
                    camera_model_id = shot_data.get('camera', '')
                    camera_model = camera_models.get(camera_model_id, {})
                    
                    # Extract image dimensions
                    width = camera_model.get('width', 1920)
                    height = camera_model.get('height', 1080)
                    
                    # Extract focal length (convert from normalized to pixels)
                    focal_normalized = camera_model.get('focal', 0.85)
                    focal_length = focal_normalized * max(width, height)
                    
                    # Create camera intrinsic matrix
                    fx = fy = focal_length
                    cx = width / 2.0
                    cy = height / 2.0
                    
                    camera_matrix = np.array([
                        [fx, 0, cx],
                        [0, fy, cy],
                        [0, 0, 1]
                    ])
                    
                    # Extract rotation (convert from angle-axis to rotation matrix)
                    rotation_vec = np.array(shot_data['rotation'])
                    rotation_matrix = cv2.Rodrigues(rotation_vec)[0]
                    
                    # Extract translation
                    translation_vector = np.array(shot_data['translation'])
                    
                    # Extract distortion coefficients
                    k1 = camera_model.get('k1', 0.0)
                    k2 = camera_model.get('k2', 0.0)
                    distortion_coeffs = np.array([k1, k2, 0, 0, 0])
                    
                    # Create CameraInfo object
                    cameras[shot_id] = CameraInfo(
                        image_id=shot_id,
                        image_path="",  # Will be set later when copying images
                        camera_matrix=camera_matrix,
                        rotation_matrix=rotation_matrix,
                        translation_vector=translation_vector,
                        width=width,
                        height=height,
                        focal_length=focal_length,
                        distortion_coeffs=distortion_coeffs
                    )
                    
                except Exception as e:
                    self.logger.warning(f"Error processing camera {shot_id}: {e}")
                    continue
        
        self.logger.info(f"Extracted {len(cameras)} camera poses")
        return cameras, camera_models
    
    def copy_and_match_images(self, cameras: Dict[str, CameraInfo], images_dir: str):
        """Copy and match images with camera poses"""
        if not os.path.exists(images_dir):
            self.logger.error(f"Images directory not found: {images_dir}")
            return
        
        # Find all image files
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
        image_files = []
        for ext in image_extensions:
            image_files.extend(glob.glob(os.path.join(images_dir, '**', ext), recursive=True))
        
        self.logger.info(f"Found {len(image_files)} image files")
        
        # Match images with camera poses
        matched_cameras = {}
        copied_images = 0
        
        for camera_id, camera_info in cameras.items():
            # Try to find matching image file
            # Look for files that contain the camera_id in their name
            matching_files = [f for f in image_files if camera_id in os.path.basename(f)]
            
            if not matching_files:
                # Try alternative matching strategies
                matching_files = [f for f in image_files if camera_id.replace('_', '') in os.path.basename(f).replace('_', '')]
            
            if matching_files:
                source_image = matching_files[0]
                # Copy image to our images directory
                dest_image = self.images_dir / f"{camera_id}.jpg"
                
                try:
                    # Copy and potentially resize image
                    img = Image.open(source_image)
                    
                    # Resize if needed to match camera intrinsics
                    if img.size != (camera_info.width, camera_info.height):
                        self.logger.info(f"Resizing image {camera_id} from {img.size} to {camera_info.width}x{camera_info.height}")
                        img = img.resize((camera_info.width, camera_info.height), Image.Resampling.LANCZOS)
                    
                    # Save as JPEG
                    img.convert('RGB').save(dest_image, 'JPEG', quality=95)
                    
                    # Update camera info with image path
                    camera_info.image_path = str(dest_image)
                    matched_cameras[camera_id] = camera_info
                    copied_images += 1
                    
                except Exception as e:
                    self.logger.error(f"Error processing image {source_image}: {e}")
            else:
                self.logger.warning(f"No matching image found for camera {camera_id}")
        
        self.logger.info(f"Successfully matched and copied {copied_images} images")
        return matched_cameras
    
    def save_colmap_format(self, cameras: Dict[str, CameraInfo], points: np.ndarray, colors: np.ndarray):
        """Save data in COLMAP format for Gaussian Splatting"""
        
        # 1. Save cameras.txt
        cameras_file = self.sparse_dir / "cameras.txt"
        with open(cameras_file, 'w') as f:
            f.write("# Camera list with one line of data per camera:\n")
            f.write("# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
            
            camera_id = 1
            for i, (cam_key, cam_info) in enumerate(cameras.items()):
                fx = cam_info.camera_matrix[0, 0]
                fy = cam_info.camera_matrix[1, 1]
                cx = cam_info.camera_matrix[0, 2]
                cy = cam_info.camera_matrix[1, 2]
                
                # Use PINHOLE model
                f.write(f"{camera_id} PINHOLE {cam_info.width} {cam_info.height} {fx} {fy} {cx} {cy}\n")
                camera_id += 1
        
        # 2. Save images.txt
        images_file = self.sparse_dir / "images.txt"
        with open(images_file, 'w') as f:
            f.write("# Image list with two lines of data per image:\n")
            f.write("# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
            f.write("# POINTS2D[] as (X, Y, POINT3D_ID)\n")
            
            image_id = 1
            camera_id = 1
            
            for cam_key, cam_info in cameras.items():
                # Convert rotation matrix to quaternion
                r = R.from_matrix(cam_info.rotation_matrix)
                quat = r.as_quat()  # Returns [x, y, z, w]
                qw, qx, qy, qz = quat[3], quat[0], quat[1], quat[2]  # COLMAP uses [w, x, y, z]
                
                # Get translation
                tx, ty, tz = cam_info.translation_vector
                
                # Image filename
                image_name = f"{cam_info.image_id}.jpg"
                
                f.write(f"{image_id} {qw} {qx} {qy} {qz} {tx} {ty} {tz} {camera_id} {image_name}\n")
                f.write("\n")  # Empty line for POINTS2D (we don't have 2D-3D correspondences)
                
                image_id += 1
                camera_id += 1
        
        # 3. Save points3D.txt
        points3d_file = self.sparse_dir / "points3D.txt"
        with open(points3d_file, 'w') as f:
            f.write("# 3D point list with one line of data per point:\n")
            f.write("# POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[]\n")
            
            for i, (point, color) in enumerate(zip(points, colors)):
                # Convert color back to 0-255 range
                r, g, b = int(color[0] * 255), int(color[1] * 255), int(color[2] * 255)
                x, y, z = point
                f.write(f"{i+1} {x} {y} {z} {r} {g} {b} 0.0\n")
        
        self.logger.info(f"Saved COLMAP format data to {self.sparse_dir}")
    
    def save_nerf_format(self, cameras: Dict[str, CameraInfo], points: np.ndarray):
        """Save data in NeRF transforms.json format"""
        
        # Calculate scene bounds
        if len(points) > 0:
            bounds_min = np.min(points, axis=0)
            bounds_max = np.max(points, axis=0)
            center = (bounds_min + bounds_max) / 2
            scale = np.max(bounds_max - bounds_min)
        else:
            center = np.zeros(3)
            scale = 1.0
        
        transforms = {
            "camera_angle_x": 0.8,  # Will be calculated per camera
            "frames": []
        }
        
        for cam_key, cam_info in cameras.items():
            # Calculate camera angle
            fx = cam_info.camera_matrix[0, 0]
            camera_angle_x = 2 * np.arctan(cam_info.width / (2 * fx))
            
            # Create transformation matrix
            # Convert from OpenCV to OpenGL coordinate system
            R = cam_info.rotation_matrix
            t = cam_info.translation_vector
            
            # Create 4x4 transformation matrix
            transform_matrix = np.eye(4)
            transform_matrix[:3, :3] = R
            transform_matrix[:3, 3] = t
            
            # Convert to NeRF coordinate system (flip Y and Z)
            transform_matrix[1, :] *= -1
            transform_matrix[2, :] *= -1
            
            frame = {
                "file_path": f"./images/{cam_info.image_id}.jpg",
                "transform_matrix": transform_matrix.tolist(),
                "camera_angle_x": camera_angle_x
            }
            
            transforms["frames"].append(frame)
        
        # Use average camera angle
        if transforms["frames"]:
            avg_angle = np.mean([frame["camera_angle_x"] for frame in transforms["frames"]])
            transforms["camera_angle_x"] = avg_angle
        
        # Save transforms.json
        transforms_file = self.output_dir / "transforms.json"
        with open(transforms_file, 'w') as f:
            json.dump(transforms, f, indent=2)
        
        self.logger.info(f"Saved NeRF format data to {transforms_file}")
    
    def save_point_cloud(self, points: np.ndarray, colors: np.ndarray):
        """Save point cloud in PLY format"""
        if len(points) == 0:
            self.logger.warning("No points to save")
            return
        
        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        if len(colors) > 0:
            pcd.colors = o3d.utility.Vector3dVector(colors)
        
        # Save PLY file
        ply_path = self.output_dir / "input_pointcloud.ply"
        o3d.io.write_point_cloud(str(ply_path), pcd)
        
        # Also save in Gaussian Splatting initialization format
        gs_ply_path = self.output_dir / "points3d.ply"
        o3d.io.write_point_cloud(str(gs_ply_path), pcd)
        
        self.logger.info(f"Saved point cloud to {ply_path} and {gs_ply_path}")
    
    def create_dataset_info(self, cameras: Dict[str, CameraInfo], points: np.ndarray):
        """Create dataset information file"""
        info = {
            "dataset_type": "mapillary",
            "num_cameras": len(cameras),
            "num_points": len(points),
            "image_resolution": f"{list(cameras.values())[0].width}x{list(cameras.values())[0].height}" if cameras else "unknown",
            "coordinate_system": "OpenCV (right-handed, Y down, Z forward)",
            "notes": [
                "Converted from Mapillary SfM reconstruction",
                "Point cloud represents street-level 3D structure",
                "Camera poses are in world coordinates",
                "Images are perspective street-view captures"
            ],
            "usage_instructions": {
                "gaussian_splatting": {
                    "command": "python train.py -s /path/to/this/dataset -m /path/to/output",
                    "requirements": "COLMAP format data in sparse/ directory"
                },
                "longsplat": {
                    "command": "python train.py --source_path /path/to/this/dataset",
                    "requirements": "Input point cloud as points3d.ply"
                },
                "nerf": {
                    "command": "python run_nerf.py --config configs/mapillary.txt --datadir /path/to/this/dataset",
                    "requirements": "transforms.json file"
                }
            }
        }
        
        info_file = self.output_dir / "dataset_info.json"
        with open(info_file, 'w') as f:
            json.dump(info, f, indent=2)
        
        self.logger.info(f"Saved dataset info to {info_file}")
    
    def convert(self, pointcloud_file: str, images_dir: str):
        """Main conversion function"""
        self.logger.info("Starting Mapillary to Gaussian Splatting conversion...")
        
        # 1. Load Mapillary data
        data = self.load_mapillary_data(pointcloud_file)
        if not data:
            self.logger.error("Failed to load Mapillary data")
            return False
        
        # 2. Extract 3D points and colors
        points, colors = self.extract_3d_points(data)
        
        # 3. Extract camera data
        cameras, camera_models = self.extract_camera_data(data)
        
        # 4. Copy and match images
        if images_dir:
            cameras = self.copy_and_match_images(cameras, images_dir)
        
        if not cameras:
            self.logger.error("No cameras with matching images found")
            return False
        
        # 5. Save in multiple formats
        self.save_colmap_format(cameras, points, colors)
        self.save_nerf_format(cameras, points)
        self.save_point_cloud(points, colors)
        self.create_dataset_info(cameras, points)
        
        # 6. Create usage instructions
        self.create_usage_instructions()
        
        self.logger.info(f"Conversion completed! Output saved to {self.output_dir}")
        return True
    
    def create_usage_instructions(self):
        """Create detailed usage instructions for different frameworks"""
        instructions = """
# Mapillary to Gaussian Splatting - Usage Instructions

## Dataset Structure
Your converted dataset has the following structure:

```
{output_dir}/
├── images/                    # Street view images
│   ├── {camera_id}.jpg       # Individual camera images
├── sparse/0/                 # COLMAP format (for 3D Gaussian Splatting)
│   ├── cameras.txt           # Camera intrinsics
│   ├── images.txt            # Camera poses
│   └── points3D.txt          # 3D points
├── transforms.json           # NeRF format
├── input_pointcloud.ply      # Original point cloud
├── points3d.ply             # Point cloud for initialization
└── dataset_info.json        # Dataset metadata

## Usage with Different Frameworks

### 1. 3D Gaussian Splatting (Original)
```bash
# Clone the repository
git clone https://github.com/graphdeco-inria/gaussian-splatting.git
cd gaussian-splatting

# Train on your Mapillary data
python train.py -s {output_dir} -m output_models/mapillary_scene

# Render the trained model
python render.py -m output_models/mapillary_scene
```

### 2. LongSplat (for street scenes)
```bash
# Clone LongSplat (better for street-level scenes)
git clone https://github.com/NVlabs/LongSplat.git
cd LongSplat

# Train on your data
python train.py --source_path {output_dir} --model_path output/mapillary_scene

# Render with camera path
python render.py --model_path output/mapillary_scene --skip_train --skip_test
```

### 3. NeRF (InstantNGP)
```bash
# Use with instant-ngp
git clone https://github.com/NVlabs/instant-ngp.git
cd instant-ngp

# Train NeRF
python scripts/run.py --scene {output_dir} --mode nerf
```

### 4. Nerfstudio
```bash
# Install nerfstudio
pip install nerfstudio

# Process your data
ns-process-data images --data {output_dir}/images --output-dir {output_dir}/nerfstudio

# Train with nerfstudio
ns-train nerfacto --data {output_dir}/nerfstudio
```

## Tips for Street-Level Scenes

1. **Use LongSplat**: Better for street scenes with long trajectories
2. **Initialize with point cloud**: Use the provided points3d.ply for better initialization
3. **Adjust camera poses**: Street scenes may need pose refinement
4. **Resolution considerations**: High-res images may need downsampling for memory

## Troubleshooting

1. **Memory issues**: Reduce image resolution or use fewer images
2. **Poor reconstruction**: Check camera poses in dataset_info.json
3. **Missing textures**: Ensure good image coverage of the scene
4. **Coordinate system**: All data is in OpenCV coordinate system (Y down, Z forward)

## Quality Tips

- Use panoramic images when available (better coverage)
- Ensure temporal coherence in image sequence
- Check for dynamic objects in the scene
- Consider masking moving objects (cars, people)
"""

        readme_file = self.output_dir / "README.md"
        with open(readme_file, 'w') as f:
            f.write(instructions.format(output_dir=str(self.output_dir)))
        
        self.logger.info(f"Created usage instructions: {readme_file}")


def main():
    parser = argparse.ArgumentParser(description='Convert Mapillary data to Gaussian Splatting format')
    parser.add_argument('--pointcloud', required=True, help='Path to Mapillary point cloud JSON file')
    parser.add_argument('--images', required=True, help='Path to directory containing street view images')
    parser.add_argument('--output', default='gaussian_splatting_data', help='Output directory')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create converter
    converter = MapillaryToGaussianSplattingConverter(args.output)
    
    # Run conversion
    success = converter.convert(args.pointcloud, args.images)
    
    if success:
        print(f"\n✅ Conversion completed successfully!")
        print(f"📁 Output directory: {args.output}")
        print(f"📖 See README.md in output directory for usage instructions")
        return 0
    else:
        print(f"\n❌ Conversion failed. Check logs for details.")
        return 1


if __name__ == "__main__":
    exit(main())
