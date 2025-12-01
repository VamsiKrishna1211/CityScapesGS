#!/usr/bin/env python3
"""
Mapillary to OpenSfM Converter
Converts Mapillary downloader output to OpenSfM data structure format
"""

import json
import shutil
import yaml
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
import numpy as np
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MapillaryToOpenSfMConverter:
    """Converts Mapillary data structure to OpenSfM format"""
    
    def __init__(self, mapillary_dir: Path, opensfm_dir: Path):
        self.mapillary_dir = Path(mapillary_dir)
        self.opensfm_dir = Path(opensfm_dir)
        
        # Mapillary structure
        self.mapillary_images_dir = self.mapillary_dir / "images"
        self.mapillary_exif_dir = self.mapillary_dir / "exif"
        self.mapillary_cameras_dir = self.mapillary_dir / "cameras"
        self.mapillary_poses_dir = self.mapillary_dir / "poses"
        self.mapillary_sequences_dir = self.mapillary_dir / "sequences"
        self.mapillary_pointclouds_dir = self.mapillary_dir / "pointclouds"
        self.mapillary_metadata_dir = self.mapillary_dir / "metadata"
        
        # OpenSfM structure
        self.opensfm_images_dir = self.opensfm_dir / "images"
        self.opensfm_exif_dir = self.opensfm_dir / "exif"
        self.opensfm_features_dir = self.opensfm_dir / "features"
        self.opensfm_matches_dir = self.opensfm_dir / "matches"
        self.opensfm_undistorted_dir = self.opensfm_dir / "undistorted"
        self.opensfm_stats_dir = self.opensfm_dir / "stats"
        
        # Create OpenSfM directory structure
        self._create_opensfm_structure()
        
        # Data containers
        self.camera_models = {}
        self.shots = {}
        self.points = {}
        self.image_list = []
        
    def _create_opensfm_structure(self):
        """Create OpenSfM directory structure"""
        directories = [
            self.opensfm_dir,
            self.opensfm_images_dir,
            self.opensfm_exif_dir,
            self.opensfm_features_dir,
            self.opensfm_matches_dir,
            self.opensfm_undistorted_dir,
            self.opensfm_undistorted_dir / "images",
            self.opensfm_undistorted_dir / "depthmaps",
            self.opensfm_stats_dir
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            
        logger.info(f"Created OpenSfM directory structure in {self.opensfm_dir}")
    
    def _load_mapillary_camera_models(self) -> Dict:
        """Load camera models from Mapillary format"""
        camera_models_file = self.mapillary_dir / "camera_models.json"
        if camera_models_file.exists():
            with open(camera_models_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _load_mapillary_intrinsics(self) -> Dict:
        """Load all camera intrinsics from Mapillary format"""
        intrinsics_file = self.mapillary_cameras_dir / "all_camera_intrinsics.json"
        if intrinsics_file.exists():
            with open(intrinsics_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _load_mapillary_poses(self) -> Dict:
        """Load all camera poses from Mapillary format"""
        poses_file = self.mapillary_poses_dir / "all_camera_poses.json"
        if poses_file.exists():
            with open(poses_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _load_mapillary_sequences(self) -> Dict:
        """Load sequence data from Mapillary format"""
        sequences_file = self.mapillary_sequences_dir / "sequences_summary.json"
        if sequences_file.exists():
            with open(sequences_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _convert_camera_model(self, mapillary_camera: Dict, camera_id: str) -> Dict:
        """Convert Mapillary camera model to OpenSfM format"""
        opensfm_camera = {
            "projection_type": mapillary_camera.get("projection_type", "perspective"),
            "width": mapillary_camera.get("width", 1920),
            "height": mapillary_camera.get("height", 1080),
        }
        
        # Handle different camera parameter formats
        if "focal" in mapillary_camera:
            opensfm_camera["focal"] = mapillary_camera["focal"]
        elif "focal_ratio" in mapillary_camera:
            # Convert focal_ratio to focal (normalized by image width)
            opensfm_camera["focal"] = mapillary_camera["focal_ratio"]
        else:
            # Default focal length if not available (common value for smartphones)
            opensfm_camera["focal"] = 0.85
        
        # Distortion parameters (following OpenSfM perspective model)
        opensfm_camera["k1"] = mapillary_camera.get("k1", 0.0)
        opensfm_camera["k2"] = mapillary_camera.get("k2", 0.0)
        
        # Additional distortion parameters if available
        if "p1" in mapillary_camera:
            opensfm_camera["p1"] = mapillary_camera["p1"]
        if "p2" in mapillary_camera:
            opensfm_camera["p2"] = mapillary_camera["p2"]
        if "k3" in mapillary_camera:
            opensfm_camera["k3"] = mapillary_camera["k3"]
        
        # Handle fisheye cameras
        if opensfm_camera["projection_type"] in ["fisheye", "fisheye_opencv"]:
            # Fisheye cameras typically have lower focal values
            if opensfm_camera["focal"] > 0.7:
                opensfm_camera["focal"] = 0.5
        
        # Handle equirectangular/spherical cameras
        elif opensfm_camera["projection_type"] in ["spherical", "equirectangular"]:
            # Remove focal and distortion for spherical projections
            opensfm_camera.pop("focal", None)
            opensfm_camera.pop("k1", None)
            opensfm_camera.pop("k2", None)
            opensfm_camera.pop("p1", None)
            opensfm_camera.pop("p2", None)
            opensfm_camera.pop("k3", None)
        
        return opensfm_camera
    
    def _convert_shot(self, image_filename: str, mapillary_pose: Dict, 
                     mapillary_exif: Dict, camera_id: str) -> Dict:
        """Convert Mapillary pose/exif to OpenSfM shot format"""
        shot = {
            "camera": camera_id,
            "rotation": [0.0, 0.0, 0.0],  # Default rotation
            "translation": [0.0, 0.0, 0.0],  # Default translation
        }
        
        # GPS information
        gps_computed = mapillary_pose.get("gps_computed")
        gps_original = mapillary_pose.get("gps_original")
        
        # Use computed GPS if available, otherwise original
        gps_data = gps_computed or gps_original
        if gps_data:
            shot["gps_position"] = [
                gps_data.get("longitude", 0.0),
                gps_data.get("latitude", 0.0),
                gps_data.get("altitude", 0.0)
            ]
            shot["gps_dop"] = gps_data.get("dop", 5.0)
        
        # Capture time
        capture_time = mapillary_pose.get("capture_time")
        if capture_time:
            if isinstance(capture_time, (int, float)):
                shot["capture_time"] = capture_time / 1000.0 if capture_time > 1e10 else capture_time
            else:
                # Try to parse datetime string
                try:
                    dt = datetime.fromisoformat(capture_time.replace('Z', '+00:00'))
                    shot["capture_time"] = dt.timestamp()
                except:
                    shot["capture_time"] = 0.0
        
        # Orientation from EXIF
        shot["orientation"] = mapillary_exif.get("orientation", 1)
        
        # Convert compass angle to rotation if available
        compass_angle = (mapillary_pose.get("computed_compass_angle") or 
                        mapillary_pose.get("compass_angle"))
        if compass_angle is not None:
            # Convert compass angle to rotation around Z-axis (yaw)
            # OpenSfM uses angle-axis representation
            yaw_radians = np.radians(compass_angle)
            shot["rotation"] = [0.0, 0.0, yaw_radians]
        
        return shot
    
    def _copy_images(self):
        """Copy images from Mapillary to OpenSfM structure"""
        logger.info("Copying images...")
        
        copied_count = 0
        
        # Copy regular street view images
        if self.mapillary_images_dir.exists():
            for image_file in self.mapillary_images_dir.glob("*"):
                if image_file.is_file() and image_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    dest_file = self.opensfm_images_dir / image_file.name
                    shutil.copy2(image_file, dest_file)
                    self.image_list.append(image_file.name)
                    copied_count += 1
        
        # Copy panoramic images if they exist
        pano_dir = self.mapillary_images_dir / "panoramic"
        if pano_dir.exists():
            for image_file in pano_dir.glob("*"):
                if image_file.is_file() and image_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    dest_file = self.opensfm_images_dir / image_file.name
                    shutil.copy2(image_file, dest_file)
                    self.image_list.append(image_file.name)
                    copied_count += 1
        
        logger.info(f"Copied {copied_count} images")
    
    def _copy_exif_data(self, mapillary_poses: Dict, mapillary_intrinsics: Dict, mapillary_cameras: Dict):
        """Convert and enhance EXIF data for OpenSfM format"""
        logger.info("Processing and converting EXIF data...")
        
        if not self.mapillary_exif_dir.exists():
            logger.warning("No EXIF data found in Mapillary structure")
            return
        
        processed_count = 0
        
        for exif_file in self.mapillary_exif_dir.glob("*.json"):
            try:
                # Load existing EXIF data
                with open(exif_file, 'r') as f:
                    exif_data = json.load(f)
                
                # Get image filename from EXIF file
                image_filename = exif_file.stem + ".jpg"  # Assume .jpg extension
                base_name = exif_file.stem
                
                # Find corresponding data in other Mapillary sources
                pose_data = mapillary_poses.get(image_filename)
                if not pose_data:
                    # Try with different extensions
                    for ext in ['.jpg', '.jpeg', '.png']:
                        test_name = base_name + ext
                        if test_name in mapillary_poses:
                            pose_data = mapillary_poses[test_name]
                            image_filename = test_name
                            break
                
                intrinsics_data = mapillary_intrinsics.get(image_filename)
                if not intrinsics_data:
                    # Try with different extensions
                    for ext in ['.jpg', '.jpeg', '.png']:
                        test_name = base_name + ext
                        if test_name in mapillary_intrinsics:
                            intrinsics_data = mapillary_intrinsics[test_name]
                            break
                
                # Create enhanced EXIF data in OpenSfM format
                enhanced_exif = self._create_enhanced_exif(exif_data, pose_data, intrinsics_data, mapillary_cameras)
                
                # Save enhanced EXIF data
                dest_file = self.opensfm_exif_dir / exif_file.name
                with open(dest_file, 'w') as f:
                    json.dump(enhanced_exif, f, indent=2)
                
                processed_count += 1
                
            except Exception as e:
                logger.warning(f"Failed to process EXIF file {exif_file.name}: {e}")
                # Copy original file as fallback
                dest_file = self.opensfm_exif_dir / exif_file.name
                shutil.copy2(exif_file, dest_file)
        
        logger.info(f"Processed {processed_count} EXIF files")
        
        # Create EXIF files for images that don't have them but have other data
        self._create_missing_exif_files(mapillary_poses, mapillary_intrinsics, mapillary_cameras)
    
    def _create_missing_exif_files(self, mapillary_poses: Dict, mapillary_intrinsics: Dict, mapillary_cameras: Dict):
        """Create EXIF files for images that don't have them but have pose/intrinsics data"""
        logger.info("Creating EXIF files for images missing them...")
        
        created_count = 0
        
        # Get all images that have pose or intrinsics data but no EXIF file
        all_data_images = set(mapillary_poses.keys()) | set(mapillary_intrinsics.keys())
        
        # Get existing EXIF files (try different extensions)
        existing_exif_bases = set()
        for exif_file in self.opensfm_exif_dir.glob("*.json"):
            existing_exif_bases.add(exif_file.stem)
        
        for image_filename in all_data_images:
            base_name = Path(image_filename).stem
            exif_filename = base_name + ".json"
            exif_path = self.opensfm_exif_dir / exif_filename
            
            # Skip if EXIF file already exists
            if base_name in existing_exif_bases:
                continue
            
            # Get available data
            pose_data = mapillary_poses.get(image_filename)
            intrinsics_data = mapillary_intrinsics.get(image_filename)
            
            if pose_data or intrinsics_data:
                # Create minimal EXIF data structure
                minimal_exif = {
                    "make": "unknown",
                    "model": "unknown",
                    "width": 1920,
                    "height": 1080,
                    "projection_type": "perspective",
                    "focal_ratio": 0.0,
                    "orientation": 1,
                    "capture_time": 0.0,
                    "gps": {},
                    "camera": "v2 unknown unknown 1920 1080 perspective 0.0"
                }
                
                # Enhance with available data
                enhanced_exif = self._create_enhanced_exif(minimal_exif, pose_data, intrinsics_data, mapillary_cameras)
                
                # Save the created EXIF file
                with open(exif_path, 'w') as f:
                    json.dump(enhanced_exif, f, indent=2)
                
                created_count += 1
        
        if created_count > 0:
            logger.info(f"Created {created_count} missing EXIF files")
    
    def _create_enhanced_exif(self, original_exif: Dict, pose_data: Optional[Dict], 
                             intrinsics_data: Optional[Dict], camera_models: Dict) -> Dict:
        """Create enhanced EXIF data by merging original EXIF with Mapillary data"""
        
        # Start with original EXIF data
        enhanced_exif = original_exif.copy()
        
        # Fill missing basic camera information
        if not enhanced_exif.get("make") or enhanced_exif.get("make") == "unknown":
            enhanced_exif["make"] = "unknown"
        
        if not enhanced_exif.get("model") or enhanced_exif.get("model") == "unknown":
            enhanced_exif["model"] = "unknown"
        
        # Fill image dimensions from intrinsics or pose data
        if intrinsics_data:
            enhanced_exif["width"] = intrinsics_data.get("width", enhanced_exif.get("width", 1920))
            enhanced_exif["height"] = intrinsics_data.get("height", enhanced_exif.get("height", 1080))
        elif pose_data:
            enhanced_exif["width"] = pose_data.get("width", enhanced_exif.get("width", 1920))
            enhanced_exif["height"] = pose_data.get("height", enhanced_exif.get("height", 1080))
        
        # Fill projection type
        if not enhanced_exif.get("projection_type"):
            if intrinsics_data:
                enhanced_exif["projection_type"] = intrinsics_data.get("projection_type", "perspective")
            else:
                enhanced_exif["projection_type"] = "perspective"
        
        # Fill focal ratio from intrinsics or camera model
        if enhanced_exif.get("focal_ratio", 0.0) == 0.0:
            if intrinsics_data:
                focal_ratio = intrinsics_data.get("focal_ratio")
                if focal_ratio is not None:
                    enhanced_exif["focal_ratio"] = focal_ratio
                else:
                    # Try to get from camera model
                    camera_id = intrinsics_data.get("camera_id")
                    if camera_id and camera_id in camera_models:
                        camera_model = camera_models[camera_id]
                        enhanced_exif["focal_ratio"] = camera_model.get("focal", 0.85)
                    else:
                        enhanced_exif["focal_ratio"] = 0.85
            else:
                enhanced_exif["focal_ratio"] = 0.85
        
        # Fill GPS data from pose information
        if not enhanced_exif.get("gps") or not enhanced_exif["gps"]:
            enhanced_exif["gps"] = {}
            
            if pose_data:
                # Try computed GPS first, then original
                gps_computed = pose_data.get("gps_computed")
                gps_original = pose_data.get("gps_original")
                gps_source = gps_computed or gps_original
                
                if gps_source:
                    enhanced_exif["gps"]["latitude"] = gps_source.get("latitude", 0.0)
                    enhanced_exif["gps"]["longitude"] = gps_source.get("longitude", 0.0)
                    enhanced_exif["gps"]["altitude"] = gps_source.get("altitude", 0.0)
                    enhanced_exif["gps"]["dop"] = gps_source.get("dop", 5.0)
        
        # Fill capture time from pose data
        if enhanced_exif.get("capture_time", 0.0) == 0.0 and pose_data:
            capture_time = pose_data.get("capture_time")
            if capture_time:
                if isinstance(capture_time, (int, float)):
                    # Convert milliseconds to seconds if needed
                    enhanced_exif["capture_time"] = capture_time / 1000.0 if capture_time > 1e10 else capture_time
                else:
                    # Try to parse datetime string
                    try:
                        dt = datetime.fromisoformat(capture_time.replace('Z', '+00:00'))
                        enhanced_exif["capture_time"] = dt.timestamp()
                    except:
                        enhanced_exif["capture_time"] = 0.0
        
        # Ensure orientation is set
        if not enhanced_exif.get("orientation"):
            enhanced_exif["orientation"] = 1
        
        # Set camera ID based on intrinsics or create one
        if not enhanced_exif.get("camera"):
            if intrinsics_data and intrinsics_data.get("camera_id"):
                enhanced_exif["camera"] = intrinsics_data["camera_id"]
            else:
                # Create camera ID from available data
                make = enhanced_exif.get("make", "unknown")
                model = enhanced_exif.get("model", "unknown")
                width = enhanced_exif.get("width", 1920)
                height = enhanced_exif.get("height", 1080)
                projection = enhanced_exif.get("projection_type", "perspective")
                focal_ratio = enhanced_exif.get("focal_ratio", 0.0)
                
                enhanced_exif["camera"] = f"v2 {make} {model} {width} {height} {projection} {focal_ratio}"
        
        return enhanced_exif
    
    def _convert_point_cloud(self):
        """Convert point cloud data to OpenSfM reconstruction format"""
        logger.info("Converting point cloud data...")
        
        # Check for combined reconstruction
        combined_reconstruction = self.mapillary_pointclouds_dir / "combined_reconstruction.json"
        if combined_reconstruction.exists():
            try:
                with open(combined_reconstruction, 'r') as f:
                    reconstruction_data = json.load(f)
                
                # Extract points if they exist in the reconstruction
                if isinstance(reconstruction_data, dict) and "points" in reconstruction_data:
                    self.points = reconstruction_data["points"]
                elif isinstance(reconstruction_data, list) and len(reconstruction_data) > 0:
                    # Multiple reconstructions, take the first one
                    if "points" in reconstruction_data[0]:
                        self.points = reconstruction_data[0]["points"]
                
                logger.info(f"Loaded {len(self.points)} 3D points from reconstruction")
            except Exception as e:
                logger.warning(f"Failed to load reconstruction data: {e}")
        
        # Also check for PLY files and convert them if no reconstruction exists
        if not self.points:
            for ply_file in self.mapillary_pointclouds_dir.glob("*.ply"):
                logger.info(f"Found PLY file: {ply_file.name}")
                # Note: Converting PLY to OpenSfM points format would require
                # additional PLY parsing which could be added if needed
    
    def _create_camera_models_json(self):
        """Create OpenSfM camera_models.json file according to OpenSfM documentation"""
        logger.info("Creating camera_models.json...")
        
        camera_models_file = self.opensfm_dir / "camera_models.json"
        with open(camera_models_file, 'w') as f:
            json.dump(self.camera_models, f, indent=2)
        
        logger.info(f"Created camera_models.json with {len(self.camera_models)} camera models")
    
    def _create_camera_models_overrides_template(self):
        """Create an example camera_models_overrides.json file for user reference"""
        overrides_file = self.opensfm_dir / "camera_models_overrides.json.example"
        
        # Create example override configurations
        example_overrides = {
            "# Comment: Override all cameras with perspective model": {
                "all": {
                    "projection_type": "perspective",
                    "width": 1920,
                    "height": 1080,
                    "focal": 0.9,
                    "k1": 0.0,
                    "k2": 0.0
                }
            },
            "# Comment: Example fisheye override": {
                "all": {
                    "projection_type": "fisheye",
                    "width": 1920,
                    "height": 1080,
                    "focal": 0.5,
                    "k1": 0.0,
                    "k2": 0.0
                }
            },
            "# Comment: Example equirectangular/360 override": {
                "all": {
                    "projection_type": "equirectangular",
                    "width": 2000,
                    "height": 1000
                }
            }
        }
        
        with open(overrides_file, 'w') as f:
            json.dump(example_overrides, f, indent=2)
        
        logger.info("Created camera_models_overrides.json.example template")
    
    def _create_reconstruction_json(self):
        """Create OpenSfM reconstruction.json file"""
        logger.info("Creating reconstruction.json...")
        
        reconstruction = {
            "cameras": self.camera_models,
            "shots": self.shots,
            "points": self.points
        }
        
        # Save as a list containing one reconstruction
        reconstruction_data = [reconstruction]
        
        reconstruction_file = self.opensfm_dir / "reconstruction.json"
        with open(reconstruction_file, 'w') as f:
            json.dump(reconstruction_data, f, indent=2)
        
        logger.info(f"Created reconstruction.json with {len(self.camera_models)} cameras, "
                   f"{len(self.shots)} shots, and {len(self.points)} points")
    
    def _create_config_yaml(self):
        """Create OpenSfM config.yaml optimized for Mapillary data"""
        logger.info("Creating config.yaml...")
        
        # Load existing config if it exists in Mapillary data
        existing_config_file = self.mapillary_dir / "config.yaml"
        if existing_config_file.exists():
            with open(existing_config_file, 'r') as f:
                config = yaml.safe_load(f)
        else:
            # Default config optimized for Mapillary data
            config = {
                # Feature detection
                "feature_type": "sift",
                "feature_root": True,
                "feature_min_frames": 4000,
                "feature_process_size": 2048,
                "feature_use_adaptive_suppression": True,
                
                # Matching
                "matching_gps_neighbors": 8,
                "matching_time_neighbors": 1,
                "matching_order_neighbors": 50,
                "matching_bow_neighbors": 0,
                "matching_lowes_ratio": 0.8,
                "matching_gps_distance": 150,
                "matching_time_prior_sigma": 1.0,
                "matching_use_words": True,
                
                # Reconstruction
                "retrieval_use_words": True,
                "align_method": "auto",
                "align_orientation_prior": "horizontal",
                "triangulation_type": "ROBUST",
                "resection_type": "ROBUST",
                
                # Bundle adjustment
                "bundle_use_gps": True,
                "bundle_use_gcp": True,
                "bundle_gps_dop_threshold": 25,
                "bundle_gps_max_error": 5.0,
                "optimize_camera_parameters": True,
                
                # Output
                "undistorted_image_format": "jpg",
                
                # Mapillary-specific optimizations
                "depthmap_method": "PATCH_MATCH",
                "depthmap_resolution": 640,
            }
        
        config_file = self.opensfm_dir / "config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        logger.info("Created config.yaml")
    
    def _create_reference_lla(self):
        """Create reference.lla file from GPS data"""
        logger.info("Creating reference.lla...")
        
        # Load reference from Mapillary data if it exists
        reference_file = self.mapillary_dir / "reference.json"
        if reference_file.exists():
            with open(reference_file, 'r') as f:
                reference_data = json.load(f)
            
            lat = reference_data.get("latitude", 0.0)
            lon = reference_data.get("longitude", 0.0)
            alt = reference_data.get("altitude", 0.0)
        else:
            # Calculate from available GPS data
            lat, lon, alt = self._calculate_reference_point()
        
        reference_lla_file = self.opensfm_dir / "reference.lla"
        with open(reference_lla_file, 'w') as f:
            f.write(f"{lat} {lon} {alt}\n")
        
        logger.info(f"Created reference.lla: {lat:.6f}, {lon:.6f}, {alt:.2f}")
    
    def _calculate_reference_point(self) -> tuple:
        """Calculate reference point from GPS data in shots"""
        if not self.shots:
            return 0.0, 0.0, 0.0
        
        latitudes = []
        longitudes = []
        altitudes = []
        
        for shot in self.shots.values():
            gps_pos = shot.get("gps_position")
            if gps_pos and len(gps_pos) >= 3:
                longitudes.append(gps_pos[0])
                latitudes.append(gps_pos[1])
                altitudes.append(gps_pos[2])
        
        if latitudes:
            avg_lat = sum(latitudes) / len(latitudes)
            avg_lon = sum(longitudes) / len(longitudes)
            avg_alt = sum(altitudes) / len(altitudes)
            return avg_lat, avg_lon, avg_alt
        
        return 0.0, 0.0, 0.0
    
    def _create_image_list(self):
        """Create image_list.txt file"""
        if self.image_list:
            image_list_file = self.opensfm_dir / "image_list.txt"
            with open(image_list_file, 'w') as f:
                for image_name in sorted(self.image_list):
                    f.write(f"{self.opensfm_images_dir.name}/{image_name}\n")
            
            logger.info(f"Created image_list.txt with {len(self.image_list)} images")
    
    def convert(self):
        """Main conversion process"""
        logger.info(f"Converting Mapillary data from {self.mapillary_dir} to OpenSfM format in {self.opensfm_dir}")
        
        # Verify input directory structure
        if not self.mapillary_dir.exists():
            raise FileNotFoundError(f"Mapillary directory not found: {self.mapillary_dir}")
        
        # Load Mapillary data
        logger.info("Loading Mapillary data...")
        mapillary_cameras = self._load_mapillary_camera_models()
        mapillary_intrinsics = self._load_mapillary_intrinsics()
        mapillary_poses = self._load_mapillary_poses()
        mapillary_sequences = self._load_mapillary_sequences()
        
        # Convert camera models
        logger.info("Converting camera models...")
        for camera_id, camera_data in mapillary_cameras.items():
            self.camera_models[camera_id] = self._convert_camera_model(camera_data, camera_id)
        
        # Copy images
        self._copy_images()
        
        # Copy and enhance EXIF data (pass loaded data to avoid reloading)
        self._copy_exif_data(mapillary_poses, mapillary_intrinsics, mapillary_cameras)
        
        # Convert shots (camera poses)
        logger.info("Converting camera poses to shots...")
        for image_filename in self.image_list:
            # Get corresponding pose and EXIF data
            base_name = Path(image_filename).stem
            
            # Find matching pose data
            pose_data = mapillary_poses.get(image_filename)
            if not pose_data:
                # Try with different extensions
                for ext in ['.jpg', '.jpeg', '.png']:
                    test_name = base_name + ext
                    if test_name in mapillary_poses:
                        pose_data = mapillary_poses[test_name]
                        break
            
            # Find matching EXIF data
            exif_data = {}
            exif_file = self.mapillary_exif_dir / f"{base_name}.json"
            if exif_file.exists():
                try:
                    with open(exif_file, 'r') as f:
                        exif_data = json.load(f)
                except Exception as e:
                    logger.warning(f"Failed to load EXIF for {image_filename}: {e}")
            
            # Find matching intrinsics to get camera ID
            intrinsics_data = mapillary_intrinsics.get(image_filename)
            if not intrinsics_data:
                # Try with different extensions
                for ext in ['.jpg', '.jpeg', '.png']:
                    test_name = base_name + ext
                    if test_name in mapillary_intrinsics:
                        intrinsics_data = mapillary_intrinsics[test_name]
                        break
            
            camera_id = intrinsics_data.get("camera_id") if intrinsics_data else "default_camera"
            
            # Ensure camera exists
            if camera_id not in self.camera_models:
                # Create default camera if not found
                self.camera_models[camera_id] = {
                    "projection_type": "perspective",
                    "width": 1920,
                    "height": 1080,
                    "focal": 0.85,
                    "k1": 0.0,
                    "k2": 0.0
                }
            
            # Convert to shot
            if pose_data:
                shot_id = base_name
                self.shots[shot_id] = self._convert_shot(image_filename, pose_data, exif_data, camera_id)
        
        # Convert point cloud data
        self._convert_point_cloud()
        
        # Create OpenSfM files
        self._create_camera_models_json()
        self._create_camera_models_overrides_template()
        self._create_reconstruction_json()
        self._create_config_yaml()
        self._create_reference_lla()
        self._create_image_list()
        
        # Create empty files for OpenSfM pipeline
        (self.opensfm_dir / "tracks.csv").touch()
        
        logger.info("✅ Conversion completed successfully!")
        logger.info(f"📁 OpenSfM data structure created in: {self.opensfm_dir}")
        logger.info(f"📊 Summary:")
        logger.info(f"   - Images: {len(self.image_list)}")
        logger.info(f"   - Cameras: {len(self.camera_models)}")
        logger.info(f"   - Shots: {len(self.shots)}")
        logger.info(f"   - 3D Points: {len(self.points)}")
        logger.info("📋 Generated OpenSfM files:")
        logger.info(f"   - camera_models.json: Camera parameters and projection models")
        logger.info(f"   - camera_models_overrides.json.example: Template for overriding camera parameters")
        logger.info(f"   - reconstruction.json: Complete reconstruction with cameras, shots, and points")
        logger.info(f"   - config.yaml: OpenSfM configuration optimized for Mapillary data")
        logger.info(f"   - reference.lla: Geographic reference point")
        logger.info(f"   - image_list.txt: List of images to process")
        logger.info(f"   - exif/: Enhanced EXIF data with missing fields filled from Mapillary data")


def main():
    parser = argparse.ArgumentParser(
        description="Convert Mapillary downloader output to OpenSfM data structure",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert Mapillary data to OpenSfM format
  python mapillary_to_opensfm_converter.py --input ./data/mapillary --output ./data/opensfm

  # Convert with custom paths
  python mapillary_to_opensfm_converter.py \\
    --input /path/to/mapillary/data \\
    --output /path/to/opensfm/project
        """
    )
    
    parser.add_argument(
        '--input', '-i',
        required=True,
        type=Path,
        help='Input directory containing Mapillary data structure'
    )
    
    parser.add_argument(
        '--output', '-o',
        required=True,
        type=Path,
        help='Output directory for OpenSfM data structure'
    )
    
    parser.add_argument(
        '--force',
        action='store_true',
        help='Overwrite output directory if it exists'
    )
    
    args = parser.parse_args()
    
    # Validate input directory
    if not args.input.exists():
        logger.error(f"Input directory does not exist: {args.input}")
        return 1
    
    # Check if output directory exists
    if args.output.exists() and not args.force:
        response = input(f"Output directory {args.output} exists. Overwrite? (y/N): ")
        if response.lower() != 'y':
            logger.info("Conversion cancelled")
            return 0
    
    try:
        # Create converter and run conversion
        converter = MapillaryToOpenSfMConverter(args.input, args.output)
        converter.convert()
        
        logger.info("🎉 Conversion completed successfully!")
        logger.info(f"You can now run OpenSfM commands in: {args.output}")
        logger.info("")
        logger.info("Next steps:")
        logger.info("  cd " + str(args.output))
        logger.info("  # Optional: Customize camera parameters")
        logger.info("  # cp camera_models_overrides.json.example camera_models_overrides.json")
        logger.info("  # nano camera_models_overrides.json  # Edit as needed")
        logger.info("  # opensfm extract_metadata .  # Rerun if you modified overrides")
        logger.info("")
        logger.info("  # Standard OpenSfM pipeline:")
        logger.info("  opensfm extract_metadata .")
        logger.info("  opensfm detect_features .")
        logger.info("  opensfm match_features .")
        logger.info("  opensfm create_tracks .")
        logger.info("  opensfm reconstruct .")
        logger.info("")
        logger.info("  # Optional: Dense reconstruction")
        logger.info("  opensfm undistort .")
        logger.info("  opensfm compute_depthmaps .")
        
        return 0
        
    except Exception as e:
        logger.error(f"Conversion failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
