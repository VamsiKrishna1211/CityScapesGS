#!/usr/bin/env python3
"""
Batch Mapillary to Gaussian Splatting Converter

Processes multiple Mapillary point cloud files and combines them into a single
Gaussian Splatting dataset. Useful for city-wide reconstructions.

Usage:
    python batch_mapillary_to_gs.py --data_dir city_data/manhattan --output combined_manhattan_gs
"""

import os
import json
import numpy as np
import argparse
from pathlib import Path
import logging
from typing import List, Dict, Tuple
import glob
from mapillary_to_gs import MapillaryToGaussianSplattingConverter
import shutil


class BatchMapillaryConverter:
    """Batch converter for multiple Mapillary point cloud files"""
    
    def __init__(self, output_dir: str = "combined_gs_data"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Initialize combined data
        self.all_points = []
        self.all_colors = []
        self.all_cameras = {}
        self.camera_id_counter = 1
        self.point_id_offset = 0
        
        # Create converter
        self.converter = MapillaryToGaussianSplattingConverter(str(self.output_dir))
    
    def find_pointcloud_files(self, data_dir: str) -> List[Tuple[str, str]]:
        """Find all point cloud files and their corresponding image directories"""
        pointcloud_files = []
        
        # Find all pointcloud JSON files
        json_files = glob.glob(os.path.join(data_dir, "**", "*_pointcloud.json"), recursive=True)
        
        for json_file in json_files:
            # Find corresponding images directory
            chunk_dir = os.path.dirname(os.path.dirname(json_file))  # Go up from pointclouds/ to chunk_dir/
            images_dir = os.path.join(chunk_dir, "images")
            
            if os.path.exists(images_dir):
                pointcloud_files.append((json_file, images_dir))
                self.logger.info(f"Found: {json_file} with images in {images_dir}")
            else:
                self.logger.warning(f"No images directory found for {json_file}")
        
        self.logger.info(f"Found {len(pointcloud_files)} point cloud files with images")
        return pointcloud_files
    
    def combine_reconstructions(self, pointcloud_files: List[Tuple[str, str]]) -> bool:
        """Combine multiple reconstructions into a single dataset"""
        
        for i, (pointcloud_file, images_dir) in enumerate(pointcloud_files):
            self.logger.info(f"Processing reconstruction {i+1}/{len(pointcloud_files)}: {pointcloud_file}")
            
            try:
                # Load individual reconstruction
                data = self.converter.load_mapillary_data(pointcloud_file)
                if not data:
                    self.logger.warning(f"Failed to load {pointcloud_file}")
                    continue
                
                # Extract points and colors
                points, colors = self.converter.extract_3d_points(data)
                if len(points) > 0:
                    self.all_points.append(points)
                    self.all_colors.append(colors)
                    self.logger.info(f"Added {len(points)} points from {pointcloud_file}")
                
                # Extract camera data
                cameras, _ = self.converter.extract_camera_data(data)
                
                # Copy and match images for this reconstruction
                if cameras:
                    matched_cameras = self.converter.copy_and_match_images(cameras, images_dir)
                    
                    # Add cameras with unique IDs
                    for cam_key, cam_info in matched_cameras.items():
                        unique_cam_id = f"chunk_{i}_{cam_key}"
                        cam_info.image_id = unique_cam_id
                        # Update image path to use unique name
                        old_path = Path(cam_info.image_path)
                        new_path = self.converter.images_dir / f"{unique_cam_id}.jpg"
                        if old_path.exists():
                            shutil.move(str(old_path), str(new_path))
                            cam_info.image_path = str(new_path)
                        
                        self.all_cameras[unique_cam_id] = cam_info
                        self.camera_id_counter += 1
                    
                    self.logger.info(f"Added {len(matched_cameras)} cameras from {pointcloud_file}")
                
            except Exception as e:
                self.logger.error(f"Error processing {pointcloud_file}: {e}")
                continue
        
        # Combine all points
        if self.all_points:
            combined_points = np.vstack(self.all_points)
            combined_colors = np.vstack(self.all_colors)
            self.logger.info(f"Combined total: {len(combined_points)} points, {len(self.all_cameras)} cameras")
            return combined_points, combined_colors
        else:
            self.logger.error("No points found in any reconstruction")
            return None, None
    
    def align_reconstructions(self, points: np.ndarray) -> np.ndarray:
        """Align and clean up the combined reconstruction"""
        if len(points) == 0:
            return points
        
        # Remove outliers (points too far from median)
        center = np.median(points, axis=0)
        distances = np.linalg.norm(points - center, axis=1)
        distance_threshold = np.percentile(distances, 95)  # Keep 95% of points
        
        mask = distances <= distance_threshold
        filtered_points = points[mask]
        
        self.logger.info(f"Filtered {len(points) - len(filtered_points)} outlier points")
        self.logger.info(f"Final point cloud: {len(filtered_points)} points")
        
        return filtered_points, mask
    
    def save_combined_dataset(self, points: np.ndarray, colors: np.ndarray):
        """Save the combined dataset in all formats"""
        
        # Save COLMAP format
        self.converter.save_colmap_format(self.all_cameras, points, colors)
        
        # Save NeRF format
        self.converter.save_nerf_format(self.all_cameras, points)
        
        # Save point cloud
        self.converter.save_point_cloud(points, colors)
        
        # Create dataset info
        self.converter.create_dataset_info(self.all_cameras, points)
        
        # Create usage instructions
        self.converter.create_usage_instructions()
        
        # Create batch-specific summary
        self.create_batch_summary()
    
    def create_batch_summary(self):
        """Create summary of the batch conversion"""
        summary = {
            "conversion_type": "batch_mapillary_to_gaussian_splatting",
            "total_reconstructions_processed": len(self.all_points),
            "total_cameras": len(self.all_cameras),
            "total_points": sum(len(pts) for pts in self.all_points),
            "scene_type": "street_level_multi_chunk",
            "recommended_frameworks": [
                "LongSplat (best for street scenes)",
                "3D Gaussian Splatting",
                "NeRF with street scene modifications"
            ],
            "notes": [
                "This dataset combines multiple Mapillary reconstructions",
                "Camera poses are in a shared coordinate system",
                "Point cloud represents street-level structure from multiple viewpoints",
                "Consider using LongSplat for better handling of street scene trajectories"
            ]
        }
        
        summary_file = self.output_dir / "batch_conversion_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        self.logger.info(f"Created batch summary: {summary_file}")
    
    def convert_batch(self, data_dir: str) -> bool:
        """Main batch conversion function"""
        self.logger.info(f"Starting batch conversion of Mapillary data from {data_dir}")
        
        # Find all point cloud files
        pointcloud_files = self.find_pointcloud_files(data_dir)
        if not pointcloud_files:
            self.logger.error(f"No point cloud files found in {data_dir}")
            return False
        
        # Combine all reconstructions
        combined_points, combined_colors = self.combine_reconstructions(pointcloud_files)
        if combined_points is None:
            self.logger.error("Failed to combine reconstructions")
            return False
        
        # Align and filter the reconstruction
        filtered_points, point_mask = self.align_reconstructions(combined_points)
        filtered_colors = combined_colors[point_mask] if len(combined_colors) > 0 else combined_colors
        
        # Save the combined dataset
        self.save_combined_dataset(filtered_points, filtered_colors)
        
        self.logger.info(f"Batch conversion completed! Output saved to {self.output_dir}")
        return True


def main():
    parser = argparse.ArgumentParser(description='Batch convert Mapillary city data to Gaussian Splatting format')
    parser.add_argument('--data_dir', required=True, help='Directory containing Mapillary city data (e.g., city_data/manhattan)')
    parser.add_argument('--output', default='combined_gs_data', help='Output directory for combined dataset')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create batch converter
    batch_converter = BatchMapillaryConverter(args.output)
    
    # Run batch conversion
    success = batch_converter.convert_batch(args.data_dir)
    
    if success:
        print(f"\n✅ Batch conversion completed successfully!")
        print(f"📁 Combined dataset: {args.output}")
        print(f"📖 See README.md in output directory for usage instructions")
        print(f"\n🚀 Quick start with LongSplat (recommended for street scenes):")
        print(f"   git clone https://github.com/NVlabs/LongSplat.git")
        print(f"   cd LongSplat")
        print(f"   python train.py --source_path {os.path.abspath(args.output)} --model_path output/street_scene")
        return 0
    else:
        print(f"\n❌ Batch conversion failed. Check logs for details.")
        return 1


if __name__ == "__main__":
    exit(main())
