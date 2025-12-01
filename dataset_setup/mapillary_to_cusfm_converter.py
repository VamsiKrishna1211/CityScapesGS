#!/usr/bin/env python3

"""
Mapillary to PyCuSFM Converter

This script converts Mapillary data (OpenSfM format) to PyCuSFM's frames_meta.json format.

Mapillary Structure:
- cameras/ : Individual camera intrinsics files (per image)
- poses/ : Individual pose files with GPS and rotation data
- sequences/ : Temporal organization of images
- images/ : Street view images
- config.yaml : Processing configuration
- reference.json : GPS reference frame

PyCuSFM Structure:
- frames_meta.json : Unified metadata with keyframes_metadata array
- images/ : Organized images (can be symlinked)

Author: Generated for 3D Reconstruction Pipeline
Date: October 2025
"""

import json
import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from datetime import datetime
import logging
import math
import argparse

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MapillaryToCuSFMConverter:
    """Convert Mapillary data structure to PyCuSFM format"""
    
    def __init__(self, mapillary_data_dir: Path, output_dir: Path):
        self.mapillary_dir = Path(mapillary_data_dir)
        self.output_dir = Path(output_dir)
        
        # Mapillary structure
        self.images_dir = self.mapillary_dir / "images"
        self.cameras_dir = self.mapillary_dir / "cameras"
        self.poses_dir = self.mapillary_dir / "poses"
        self.sequences_dir = self.mapillary_dir / "sequences"
        self.reference_file = self.mapillary_dir / "reference.json"
        
        # Legacy Mapillary structure (fallback)
        self.exif_dir = self.mapillary_dir / "exif"
        self.camera_models_file = self.mapillary_dir / "camera_models.json"
        self.pointclouds_dir = self.mapillary_dir / "pointclouds"
        
        # Output structure
        self.output_images_dir = self.output_dir / "images"
        self.frames_meta_file = self.output_dir / "frames_meta.json"
        
        # Validate input
        if not self.mapillary_dir.exists():
            raise ValueError(f"Mapillary directory not found: {self.mapillary_dir}")
        if not self.images_dir.exists():
            raise ValueError(f"Images directory not found: {self.images_dir}")
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.output_images_dir.mkdir(exist_ok=True)
        
        logger.info(f"Initialized converter:")
        logger.info(f"  Input:  {self.mapillary_dir}")
        logger.info(f"  Output: {self.output_dir}")
    
    def convert(self):
        """Main conversion method"""
        logger.info("Starting Mapillary to PyCuSFM conversion...")
        
        try:
            # Determine data format and convert accordingly
            if self.cameras_dir.exists() and self.poses_dir.exists():
                # Enhanced Mapillary format with cameras/ and poses/
                logger.info("Detected enhanced Mapillary format (cameras/ and poses/)")
                camera_intrinsics = self._load_camera_intrinsics()
                pose_data = self._load_pose_data()
                
                # Check if sequences directory exists for sequence-based conversion
                if self.sequences_dir.exists():
                    logger.info("Found sequences/ directory - creating sequence-based structure")
                    self._create_sequence_based_structure(camera_intrinsics, pose_data)
                else:
                    logger.info("No sequences/ directory - grouping by camera type")
                    # Group data by camera type and create separate sequences
                    self._create_camera_sequences_enhanced(camera_intrinsics, pose_data)
            elif self.camera_models_file.exists():
                # Legacy OpenSfM format
                logger.info("Detected legacy OpenSfM format")
                camera_models = self._load_camera_models()
                if (self.mapillary_dir / "reconstruction.json").exists():
                    # SfM reconstruction available
                    sfm_data = self._load_sfm_data()
                    exif_data = self._load_exif_data()
                    self._create_frames_meta_from_sfm(sfm_data, camera_models, exif_data)
                else:
                    # EXIF-only conversion
                    exif_data = self._load_exif_data()
                    self._create_frames_meta_from_exif(camera_models, exif_data)
            else:
                raise ValueError("No supported Mapillary format detected")
            
            logger.info("✅ Conversion completed successfully!")
            logger.info(f"   Output: {self.output_dir}")
            
        except Exception as e:
            logger.error(f"❌ Conversion failed: {e}")
            import traceback
            logger.error(f"Stack trace:\n{traceback.format_exc()}")
            raise
    
    def _load_camera_intrinsics(self) -> Dict:
        """Load camera intrinsics from enhanced format cameras/ directory"""
        intrinsics_data = {}
        
        if not self.cameras_dir.exists():
            logger.warning("cameras directory not found")
            return intrinsics_data
        
        logger.info(f"Loading camera intrinsics from {self.cameras_dir}")
        for intrinsics_file in self.cameras_dir.glob("*_intrinsics.json"):
            # Extract image ID by removing "_intrinsics.json" suffix
            image_id = intrinsics_file.stem.replace("_intrinsics", "")
            try:
                with open(intrinsics_file, 'r') as f:
                    intrinsics_data[image_id] = json.load(f)
            except Exception as e:
                logger.error(f"Error loading intrinsics file {intrinsics_file}: {e}")
        
        logger.info(f"Loaded intrinsics for {len(intrinsics_data)} images")
        return intrinsics_data
    
    def _load_pose_data(self) -> Dict:
        """Load pose data from enhanced format poses/ directory"""
        pose_data = {}
        
        if not self.poses_dir.exists():
            logger.warning("poses directory not found")
            return pose_data
        
        logger.info(f"Loading pose data from {self.poses_dir}")
        for pose_file in self.poses_dir.glob("*_pose.json"):
            # Extract image ID by removing "_pose.json" suffix
            image_id = pose_file.stem.replace("_pose", "")
            try:
                with open(pose_file, 'r') as f:
                    pose_data[image_id] = json.load(f)
            except Exception as e:
                logger.error(f"Error loading pose file {pose_file}: {e}")
        
        logger.info(f"Loaded pose data for {len(pose_data)} images")
        return pose_data
    
    def _load_camera_models(self) -> Dict:
        """Load camera models from Mapillary format"""
        if not self.camera_models_file.exists():
            logger.warning("camera_models.json not found")
            return {}
        
        try:
            with open(self.camera_models_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading camera models: {e}")
            return {}
    
    def _load_sfm_data(self) -> Dict:
        """Load SfM reconstruction data"""
        sfm_file = self.mapillary_dir / "reconstruction.json"
        if not sfm_file.exists():
            return {}
        
        try:
            with open(sfm_file, 'r') as f:
                data = json.load(f)
                return data[0] if isinstance(data, list) and len(data) > 0 else data
        except Exception as e:
            logger.error(f"Error loading SfM data: {e}")
            return {}
    
    def _load_exif_data(self) -> Dict:
        """Load EXIF data"""
        exif_data = {}
        
        if self.exif_dir.exists():
            for exif_file in self.exif_dir.glob("*.json"):
                image_id = exif_file.stem
                try:
                    with open(exif_file, 'r') as f:
                        exif_data[image_id] = json.load(f)
                except Exception as e:
                    logger.error(f"Error loading EXIF file {exif_file}: {e}")
        
        return exif_data
    
    def _copy_images_enhanced(self):
        """Copy images for enhanced format"""
        if not self.images_dir.exists():
            logger.error("images directory not found")
            return
        
        # Create camera subdirectories based on camera models found in intrinsics
        camera_dirs = set()
        
        # Load camera intrinsics to determine camera types
        for intrinsics_file in self.cameras_dir.glob("*_intrinsics.json"):
            try:
                with open(intrinsics_file, 'r') as f:
                    data = json.load(f)
                camera_name = self._get_camera_name_from_intrinsics(data)
                camera_dirs.add(camera_name)
            except Exception as e:
                logger.error(f"Error reading {intrinsics_file}: {e}")
        
        # Default camera if none found
        if not camera_dirs:
            camera_dirs.add("camera_0")
        
        # Create camera subdirectories
        for camera_dir in camera_dirs:
            (self.output_images_dir / camera_dir).mkdir(exist_ok=True)
        
        # Copy images
        copied_count = 0
        for image_file in self.images_dir.glob("*.jpg"):
            image_id = image_file.stem
            
            # Get camera name for this image
            camera_name = self._get_camera_name_for_image_enhanced(image_id)
            
            # Copy image
            dest_path = self.output_images_dir / camera_name / f"{image_id}.jpg"
            shutil.copy2(image_file, dest_path)
            copied_count += 1
        
        logger.info(f"Copied {copied_count} images to camera subdirectories")
    
    def _get_camera_name_from_intrinsics(self, intrinsics_data: Dict) -> str:
        """Extract camera name from intrinsics data"""
        # Try to get camera make/model for naming
        make = intrinsics_data.get("make", "")
        model = intrinsics_data.get("model", "")
        
        # Get focal length and principal point to help distinguish cameras
        camera_params = intrinsics_data.get("camera_parameters", [])
        focal_length = camera_params[0] if len(camera_params) > 0 else 0
        
        # Create a more unique camera identifier
        if make and model:
            base_name = f"{make}_{model}".replace(" ", "_").lower()
        elif model:
            base_name = f"{model}".replace(" ", "_").lower()
        else:
            base_name = "camera"
        
        # Add focal length to make it more unique (rounded to avoid floating point precision issues)
        if focal_length > 0:
            focal_str = f"_f{int(focal_length)}"
            return f"{base_name}{focal_str}"
        else:
            return f"{base_name}_0"
    
    def _get_camera_name_for_image_enhanced(self, image_id: str) -> str:
        """Get camera name for an image in enhanced format"""
        # Try to load intrinsics for this image
        intrinsics_file = self.cameras_dir / f"{image_id}_intrinsics.json"
        if intrinsics_file.exists():
            try:
                with open(intrinsics_file, 'r') as f:
                    data = json.load(f)
                return self._get_camera_name_from_intrinsics(data)
            except:
                pass
        
        return "camera_0"  # Default fallback
    
    def _copy_images(self):
        """Copy images from Mapillary structure to PyCuSFM structure"""
        if not self.images_dir.exists():
            logger.error("images directory not found")
            return
        
        # Create camera subdirectories
        camera_dirs = set()
        
        # First pass: identify unique cameras and collect all images
        all_image_files = list(self.images_dir.glob("*.jpg"))
        
        # Sort images by filename to ensure consistent ordering
        all_image_files.sort(key=lambda x: x.stem)
        
        for image_file in all_image_files:
            camera_name = self._get_camera_name_for_image(image_file.stem)
            camera_dirs.add(camera_name)
        
        # Create camera subdirectories
        for camera_dir in camera_dirs:
            (self.output_images_dir / camera_dir).mkdir(exist_ok=True)
        
        # Second pass: copy images with sequence indices
        for idx, image_file in enumerate(all_image_files):
            image_id = image_file.stem
            camera_name = self._get_camera_name_for_image(image_id)
            new_filename = f"{idx:06d}_{image_id}.jpg"
            dest_path = self.output_images_dir / camera_name / new_filename
            shutil.copy2(image_file, dest_path)
    
    def _get_camera_name_for_image(self, image_id: str) -> str:
        """Get camera name for an image"""
        return "camera_0"  # Simplified for now
    
    def _copy_images_with_order(self, ordered_image_ids: List[str]):
        """Copy images in a specific order with sequence index prefixes"""
        if not self.images_dir.exists():
            logger.error("images directory not found")
            return
        
        # Create camera subdirectories
        camera_dirs = set()
        
        # Determine unique cameras
        for image_id in ordered_image_ids:
            camera_name = self._get_camera_name_for_image(image_id)
            camera_dirs.add(camera_name)
        
        # Create camera subdirectories
        for camera_dir in camera_dirs:
            (self.output_images_dir / camera_dir).mkdir(exist_ok=True)
        
        # Copy images with sequence indices
        copied_count = 0
        for idx, image_id in enumerate(ordered_image_ids):
            src_image = self.images_dir / f"{image_id}.jpg"
            if src_image.exists():
                camera_name = self._get_camera_name_for_image(image_id)
                new_filename = f"{idx:06d}_{image_id}.jpg"
                dest_path = self.output_images_dir / camera_name / new_filename
                shutil.copy2(src_image, dest_path)
                copied_count += 1
        
        logger.info(f"Copied {copied_count} images with sequence indices")
    
    def _create_camera_sequences_enhanced(self, camera_intrinsics: Dict, pose_data: Dict):
        """Create separate sequences for each camera type (similar to KITTI structure)"""
        logger.info("Creating separate camera sequences from enhanced Mapillary data")
        
        # Group images by camera type
        camera_groups = {}
        common_images = set(camera_intrinsics.keys()) & set(pose_data.keys())
        
        for image_id in common_images:
            intrinsics = camera_intrinsics[image_id]
            camera_id = intrinsics.get("camera_id", "unknown")
            
            if camera_id not in camera_groups:
                camera_groups[camera_id] = []
            camera_groups[camera_id].append(image_id)
        
        logger.info(f"Found {len(camera_groups)} different camera types:")
        for camera_id, images in camera_groups.items():
            logger.info(f"  {camera_id}: {len(images)} images")
        
        # Create a sequence directory for each camera type
        sequence_id = 0
        for camera_id, image_ids in camera_groups.items():
            # Skip cameras with too few images
            if len(image_ids) < 3:
                logger.warning(f"Skipping camera {camera_id} with only {len(image_ids)} images")
                continue
                
            # Create sequence directory (like KITTI: 00, 01, 02, etc.)
            sequence_dir = self.output_dir / f"{sequence_id:02d}"
            sequence_dir.mkdir(parents=True, exist_ok=True)
            
            # Create camera subdirectory
            camera_dir = sequence_dir / "camera_0"  # Use single camera per sequence
            camera_dir.mkdir(exist_ok=True)
            
            logger.info(f"Creating sequence {sequence_id:02d} for camera {camera_id} ({len(image_ids)} images)")
            
            # Copy images for this camera
            self._copy_images_for_camera(image_ids, camera_dir)
            
            # Create frames_meta.json for this camera sequence
            self._create_frames_meta_for_camera_sequence(
                camera_id, image_ids, camera_intrinsics, pose_data, sequence_dir
            )
            
            sequence_id += 1
        
        logger.info(f"Created {sequence_id} camera sequences")
    
    def _copy_images_for_camera(self, image_ids: List[str], camera_dir: Path):
        """Copy images for a specific camera to its sequence directory"""
        copied_count = 0
        valid_image_idx = 0
        for image_id in image_ids:
            src_image = self.images_dir / f"{image_id}.jpg"
            if src_image.exists():
                # Add sequence index as prefix: 000000_originalname.jpg, 000001_originalname.jpg, etc.
                dst_image = camera_dir / f"{valid_image_idx:06d}_{image_id}.jpg"
                if not dst_image.exists():
                    # Use symlink if possible, copy if not
                    try:
                        # dst_image.symlink_to(src_image.resolve())
                        shutil.copy2(src_image, dst_image)
                    except OSError:
                        # Fallback to copying if symlink fails
                        shutil.copy2(src_image, dst_image)
                copied_count += 1
                valid_image_idx += 1
        
        logger.info(f"  Copied {copied_count} images to {camera_dir}")
    
    def _create_frames_meta_for_camera_sequence(self, camera_id: str, image_ids: List[str], 
                                              camera_intrinsics: Dict, pose_data: Dict, 
                                              sequence_dir: Path):
        """Create frames_meta.json for a single camera sequence"""
        keyframes_metadata = []
        camera_params_map = {}
        camera_name_to_id = {}
        
        # Sort images by timestamp
        image_ids_with_timestamps = []
        for image_id in image_ids:
            if image_id in pose_data:
                timestamp = self._get_timestamp_from_pose_data(pose_data[image_id])
                image_ids_with_timestamps.append((timestamp, image_id))
        
        image_ids_with_timestamps.sort()
        sorted_image_ids = [img_id for _, img_id in image_ids_with_timestamps]
        
        # Find reference GPS coordinate (use first valid GPS point)
        ref_lat, ref_lon, ref_alt = self._find_reference_gps_point(
            {img_id: pose_data[img_id] for img_id in sorted_image_ids if img_id in pose_data}
        )
        
        frame_id = 0
        sample_id = 0
        
        # Process each image in chronological order
        valid_image_idx = 0  # Track index for valid images only
        for image_id in sorted_image_ids:
            if image_id not in camera_intrinsics or image_id not in pose_data:
                continue
                
            intrinsics = camera_intrinsics[image_id]
            pose = pose_data[image_id]
            
            # Get timestamp for this specific image
            timestamp_us = self._get_timestamp_from_pose_data(pose)
            
            # Create camera parameters (should be same for all images in this sequence)
            camera_name = "camera_0"  # Single camera per sequence
            camera_param_id = self._get_or_create_camera_params_from_enhanced_intrinsics(
                camera_name, intrinsics, camera_params_map, camera_name_to_id
            )
            
            # Convert Mapillary pose to PyCuSFM format using reference point
            cusfm_pose = self._convert_mapillary_pose_to_cusfm(pose, ref_lat, ref_lon, ref_alt)
            
            # Create image filename (relative to sequence directory) with sequence index prefix
            image_name = f"camera_0/{valid_image_idx:06d}_{image_id}.jpg"
            
            # Create keyframe metadata
            keyframe = {
                "id": str(frame_id),
                "camera_params_id": str(camera_param_id),
                "timestamp_microseconds": str(timestamp_us),
                "image_name": image_name,
                "camera_to_world": cusfm_pose,
                "synced_sample_id": str(sample_id)
            }
            
            keyframes_metadata.append(keyframe)
            frame_id += 1
            sample_id += 1
            valid_image_idx += 1
        
        # Check if any keyframes were successfully processed
        if not keyframes_metadata:
            logger.warning(f"No valid keyframes found for camera sequence {camera_id}, skipping frames_meta.json creation")
            return
        
        # Create frames_meta structure for this camera sequence
        frames_meta = {
            "keyframes_metadata": keyframes_metadata,
            "initial_pose_type": "EGO_MOTION",  # Use EGO_MOTION for sequential data
            "camera_params_id_to_session_name": {
                str(param_id): "0" for param_id in camera_params_map.keys()
            },
            "camera_params_id_to_camera_params": {
                str(param_id): params for param_id, params in camera_params_map.items()
            }
        }
        
        # Save frames_meta.json for this sequence
        frames_meta_file = sequence_dir / "frames_meta.json"
        with open(frames_meta_file, 'w') as f:
            json.dump(frames_meta, f, indent=2)
        
        logger.info(f"  ✅ Created {frames_meta_file} with {len(keyframes_metadata)} keyframes")
        logger.info(f"     Camera type: {camera_id}")
        if keyframes_metadata:
            logger.info(f"     Time range: {min(int(k['timestamp_microseconds']) for k in keyframes_metadata)} to {max(int(k['timestamp_microseconds']) for k in keyframes_metadata)} microseconds")
    
    def _create_sequence_based_structure(self, camera_intrinsics: Dict, pose_data: Dict):
        """Create sequence-based structure similar to KITTI dataset format
        
        This method creates numbered sequence directories (0000, 0001, etc.) for each
        Mapillary sequence found in the sequences/ directory. Each sequence contains:
        - camera_0/ directory with sequentially numbered images (000000.jpg, 000001.jpg, etc.)
        - frames_meta.json with camera intrinsics, poses, and timestamps
        
        Images within each sequence maintain their temporal order from the sequence JSON.
        """
        logger.info("Creating sequence-based structure from Mapillary sequences")
        
        # Load sequence data
        sequence_data = self._load_sequence_data()
        
        if not sequence_data:
            logger.warning("No sequence data found, falling back to camera-based grouping")
            self._create_camera_sequences_enhanced(camera_intrinsics, pose_data)
            return
        
        logger.info(f"Found {len(sequence_data)} sequences")
        
        # Create numbered sequence directories (0000, 0001, 0002, etc.)
        sequence_idx = 0
        for sequence_id, sequence_info in sequence_data.items():
            sequence_dir = self.output_dir / f"{sequence_idx:04d}_{sequence_id}"
            sequence_dir.mkdir(exist_ok=True)
            
            camera_dir = sequence_dir / "camera_0"
            camera_dir.mkdir(exist_ok=True)
            
            # Get image IDs for this sequence (already in temporal order from Mapillary)
            image_ids = [img["image_id"] for img in sequence_info["images"]]
            
            # Filter images that have both intrinsics and pose data
            valid_image_ids = [
                img_id for img_id in image_ids 
                if img_id in camera_intrinsics and img_id in pose_data
            ]
            
            if not valid_image_ids:
                logger.warning(f"Sequence {sequence_id} has no valid images with both intrinsics and pose data, skipping")
                continue
            
            logger.info(f"Creating sequence {sequence_idx:04d} for Mapillary sequence {sequence_id} ({len(valid_image_ids)} valid images)")
            
            # Copy images for this sequence
            self._copy_images_for_sequence(valid_image_ids, camera_dir)
            
            # Create frames_meta.json for this sequence
            self._create_frames_meta_for_sequence(
                sequence_id, valid_image_ids, camera_intrinsics, pose_data, sequence_dir
            )
            
            sequence_idx += 1
        
        logger.info(f"Created {sequence_idx} sequence directories")
    
    def _load_sequence_data(self) -> Dict:
        """Load sequence data from sequences/ directory"""
        sequence_data = {}
        
        if not self.sequences_dir.exists():
            logger.warning(f"Sequences directory does not exist: {self.sequences_dir}")
            return sequence_data
        
        # Load sequences_summary.json if it exists (faster)
        summary_file = self.sequences_dir / "sequences_summary.json"
        if summary_file.exists():
            logger.info(f"Loading sequence data from {summary_file}")
            try:
                with open(summary_file, 'r') as f:
                    sequence_data = json.load(f)
                logger.info(f"Loaded {len(sequence_data)} sequences from summary file")
                return sequence_data
            except Exception as e:
                logger.warning(f"Failed to load sequences summary: {e}, falling back to individual files")
        
        # Load individual sequence files
        logger.info(f"Loading individual sequence files from {self.sequences_dir}")
        for sequence_file in self.sequences_dir.glob("sequence_*.json"):
            if sequence_file.name == "sequences_summary.json":
                continue
            
            try:
                with open(sequence_file, 'r') as f:
                    seq_data = json.load(f)
                    sequence_id = seq_data.get("sequence_id")
                    if sequence_id:
                        sequence_data[sequence_id] = seq_data
            except Exception as e:
                logger.warning(f"Failed to load sequence file {sequence_file}: {e}")
        
        logger.info(f"Loaded {len(sequence_data)} sequences from individual files")
        return sequence_data
    
    def _copy_images_for_sequence(self, image_ids: List[str], camera_dir: Path):
        """Copy images for a specific sequence to its camera directory"""
        copied_count = 0
        valid_image_idx = 0
        
        for image_id in image_ids:
            src_image = self.images_dir / f"{image_id}.jpg"
            if src_image.exists():
                # Add sequence index as prefix: 000000.jpg, 000001.jpg, etc.
                dst_image = camera_dir / f"{valid_image_idx:06d}.jpg"
                if not dst_image.exists():
                    try:
                        # Use copy instead of symlink for better compatibility
                        shutil.copy2(src_image, dst_image)
                        copied_count += 1
                        valid_image_idx += 1
                    except Exception as e:
                        logger.warning(f"Failed to copy {src_image} to {dst_image}: {e}")
                else:
                    valid_image_idx += 1
            else:
                logger.warning(f"Source image not found: {src_image}")
        
        logger.info(f"  Copied {copied_count} images to {camera_dir}")
    
    def _create_frames_meta_for_sequence(self, sequence_id: str, image_ids: List[str], 
                                       camera_intrinsics: Dict, pose_data: Dict, 
                                       sequence_dir: Path):
        """Create frames_meta.json for a single sequence
        
        Processes images from a Mapillary sequence and creates PyCuSFM metadata.
        Images are sorted by timestamp to ensure chronological order, which should
        match the order in the sequence JSON file.
        """
        keyframes_metadata = []
        camera_params_map = {}
        camera_name_to_id = {}
        
        # Sort images by timestamp from pose data to ensure chronological order
        # (This should match the order in the sequence JSON, but we re-sort to be certain)
        image_ids_with_timestamps = []
        for image_id in image_ids:
            if image_id in pose_data:
                timestamp = self._get_timestamp_from_pose_data(pose_data[image_id])
                image_ids_with_timestamps.append((timestamp, image_id))
        
        image_ids_with_timestamps.sort()
        sorted_image_ids = [img_id for _, img_id in image_ids_with_timestamps]
        
        # Find reference GPS coordinate (use first valid GPS point)
        ref_lat, ref_lon, ref_alt = self._find_reference_gps_point(
            {img_id: pose_data[img_id] for img_id in sorted_image_ids if img_id in pose_data}
        )
        
        frame_id = 0
        sample_id = 0
        
        # Process each image in chronological order
        valid_image_idx = 0  # Track index for valid images only
        for image_id in sorted_image_ids:
            if image_id not in camera_intrinsics or image_id not in pose_data:
                logger.warning(f"Skipping image {image_id}: missing intrinsics or pose data")
                continue
            
            intrinsics = camera_intrinsics[image_id]
            pose = pose_data[image_id]
            
            # Check for None values
            if intrinsics is None:
                logger.warning(f"Skipping image {image_id}: intrinsics data is None")
                continue
            if pose is None:
                logger.warning(f"Skipping image {image_id}: pose data is None")
                continue
            
            try:
                # Get or create camera parameters
                camera_name = self._get_camera_name_from_intrinsics(intrinsics)
                camera_params_id = self._get_or_create_camera_params_from_enhanced_intrinsics(
                    camera_name, intrinsics, camera_params_map, camera_name_to_id
                )
                
                # Convert pose to CuSFM format
                cusfm_pose = self._convert_mapillary_pose_to_cusfm(pose, ref_lat, ref_lon, ref_alt)
                
                # Get timestamp
                timestamp_microseconds = self._get_timestamp_from_pose_data(pose)
                
                # Create keyframe metadata
                keyframe = {
                    "id": str(frame_id),
                    "camera_params_id": str(camera_params_id),
                    "timestamp_microseconds": str(timestamp_microseconds),
                    "image_name": f"camera_0/{valid_image_idx:06d}.jpg",
                    "camera_to_world": cusfm_pose,
                    "synced_sample_id": str(sample_id)
                }
                
                keyframes_metadata.append(keyframe)
                frame_id += 1
                sample_id += 1
                valid_image_idx += 1
                
            except Exception as e:
                logger.error(f"Error processing image {image_id}: {e}")
                logger.error(f"  Intrinsics type: {type(intrinsics)}")
                logger.error(f"  Pose type: {type(pose)}")
                if intrinsics is not None:
                    logger.error(f"  Intrinsics keys: {list(intrinsics.keys()) if isinstance(intrinsics, dict) else 'Not a dict'}")
                if pose is not None:
                    logger.error(f"  Pose keys: {list(pose.keys()) if isinstance(pose, dict) else 'Not a dict'}")
                continue
        
        # Check if any keyframes were successfully processed
        if not keyframes_metadata:
            logger.warning(f"No valid keyframes found for sequence {sequence_id}, skipping frames_meta.json creation")
            return
        
        # Create frames_meta structure for this sequence
        frames_meta = {
            "keyframes_metadata": keyframes_metadata,
            "initial_pose_type": "EGO_MOTION",  # Standard for vehicle/mobile data
            "camera_params_id_to_session_name": {
                str(param_id): "0" for param_id in camera_params_map.keys()
            },
            "camera_params_id_to_camera_params": {
                str(param_id): params for param_id, params in camera_params_map.items()
            }
        }
        
        # Save frames_meta.json for this sequence
        frames_meta_file = sequence_dir / "frames_meta.json"
        with open(frames_meta_file, 'w') as f:
            json.dump(frames_meta, f, indent=2)
        
        logger.info(f"  ✅ Created {frames_meta_file} with {len(keyframes_metadata)} keyframes")
        logger.info(f"     Mapillary sequence: {sequence_id}")
        if keyframes_metadata:
            logger.info(f"     Time range: {min(int(k['timestamp_microseconds']) for k in keyframes_metadata)} to {max(int(k['timestamp_microseconds']) for k in keyframes_metadata)} microseconds")
    
    def _create_frames_meta_from_enhanced_data(self, camera_intrinsics: Dict, pose_data: Dict):
        """Create frames_meta.json from enhanced Mapillary data with cameras/ and poses/"""
        logger.info("Creating frames_meta.json from enhanced Mapillary data")
        
        keyframes_metadata = []
        camera_params_map = {}
        camera_name_to_id = {}
        
        # Process each image that has both intrinsics and pose data
        common_images = set(camera_intrinsics.keys()) & set(pose_data.keys())
        logger.info(f"Found {len(common_images)} images with both intrinsics and pose data")
        
        if not common_images:
            logger.error("No images found with both camera intrinsics and pose data")
            return
        
        # Find reference GPS coordinate (use first valid GPS point)
        ref_lat, ref_lon, ref_alt = self._find_reference_gps_point(pose_data)
        logger.info(f"Using reference GPS point: lat={ref_lat:.6f}, lon={ref_lon:.6f}, alt={ref_alt:.2f}")
        
        # Analyze data structure for diagnostics
        self._analyze_input_data(camera_intrinsics, pose_data)
        
        # For Mapillary data, typically each image is captured at a different time
        # So we'll treat each image as a separate sample (no multi-camera synchronization)
        # This is different from stereo camera setups like KITTI
        
        frame_id = 0
        sample_id = 0
        
        # Sort common images for consistent ordering
        sorted_common_images = sorted(common_images)
        
        # Process each image individually (monocular sequence)
        for idx, image_id in enumerate(sorted_common_images):
            intrinsics = camera_intrinsics[image_id]
            pose = pose_data[image_id]
            
            # Get timestamp for this specific image
            timestamp_us = self._get_timestamp_from_pose_data(pose)
            
            # Get camera name and create camera parameters
            camera_name = self._get_camera_name_from_intrinsics(intrinsics)
            camera_param_id = self._get_or_create_camera_params_from_enhanced_intrinsics(
                camera_name, intrinsics, camera_params_map, camera_name_to_id
            )
            
            # Convert Mapillary pose to PyCuSFM format using reference point
            cusfm_pose = self._convert_mapillary_pose_to_cusfm(pose, ref_lat, ref_lon, ref_alt)
            
            # Create image filename with sequence index prefix
            image_name = f"{camera_name}/{idx:06d}_{image_id}.jpg"
            
            # Create keyframe metadata - each image gets its own sample_id for monocular sequence
            keyframe = {
                "id": str(frame_id),
                "camera_params_id": str(camera_param_id),
                "timestamp_microseconds": str(timestamp_us),
                "image_name": image_name,
                "camera_to_world": cusfm_pose,
                "synced_sample_id": str(sample_id)  # Each image gets unique sample_id
            }
            
            keyframes_metadata.append(keyframe)
            frame_id += 1
            sample_id += 1
        
        # Sort by timestamp
        keyframes_metadata.sort(key=lambda x: int(x["timestamp_microseconds"]))
        
        # Create frames_meta structure
        frames_meta = {
            "keyframes_metadata": keyframes_metadata,
            "initial_pose_type": "EGO_MOTION",  # Use EGO_MOTION for sequential data
            "camera_params_id_to_session_name": {
                str(param_id): "0" for param_id in camera_params_map.keys()
            },
            "camera_params_id_to_camera_params": {
                str(param_id): params for param_id, params in camera_params_map.items()
            },
            # "reference_latlngalt": {
            #     "latitude": ref_lat,
            #     "longitude": ref_lon,
            #     "altitude": ref_alt
            # }
        }
        
        # Save frames_meta.json
        with open(self.frames_meta_file, 'w') as f:
            json.dump(frames_meta, f, indent=2)
        
        logger.info(f"✅ Created frames_meta.json with {len(keyframes_metadata)} keyframes from enhanced data")
        logger.info(f"   📄 Output file: {self.frames_meta_file}")
        logger.info(f"   🎥 Cameras used: {len(camera_params_map)}")
        if keyframes_metadata:
            logger.info(f"   ⏰ Time range: {min(int(k['timestamp_microseconds']) for k in keyframes_metadata)} to {max(int(k['timestamp_microseconds']) for k in keyframes_metadata)} microseconds")
    
    def _analyze_input_data(self, camera_intrinsics: Dict, pose_data: Dict):
        """Analyze input data structure for debugging"""
        logger.info("=== Input Data Analysis ===")
        
        # Analyze camera types
        camera_types = {}
        focal_lengths = set()
        
        for image_id, intrinsics in camera_intrinsics.items():
            camera_name = self._get_camera_name_from_intrinsics(intrinsics)
            if camera_name not in camera_types:
                camera_types[camera_name] = 0
            camera_types[camera_name] += 1
            
            # Track focal lengths
            camera_params = intrinsics.get("camera_parameters", [])
            if len(camera_params) > 0:
                focal_lengths.add(int(camera_params[0]))
        
        logger.info(f"Camera types found: {dict(camera_types)}")
        logger.info(f"Focal lengths found: {sorted(focal_lengths)}")
        
        # Analyze timestamps
        timestamps = set()
        for image_id, pose in pose_data.items():
            timestamp_us = self._get_timestamp_from_pose_data(pose)
            timestamps.add(timestamp_us)
        
        logger.info(f"Total unique timestamps: {len(timestamps)}")
        logger.info(f"Total images with intrinsics: {len(camera_intrinsics)}")
        logger.info(f"Total images with poses: {len(pose_data)}")
        logger.info("=== End Analysis ===")
    
    def _find_reference_gps_point(self, pose_data: Dict) -> Tuple[float, float, float]:
        """Find a reference GPS point from pose data (typically the first valid point)"""
        
        for pose in pose_data.values():
            if pose is None:
                continue
                
            # Try to get geometry from various possible keys
            geometry = pose.get("computed_geometry") or pose.get("geometry") or {}
            
            if geometry is not None and geometry.get("type") == "Point" and "coordinates" in geometry:
                lon, lat = geometry["coordinates"][:2]
                altitude = pose.get("computed_altitude", pose.get("altitude", 0.0))
                return lat, lon, altitude
        
        # Fallback to origin if no GPS data found
        logger.warning("No GPS coordinates found in pose data, using origin as reference")
        return 0.0, 0.0, 0.0
    
    def _get_or_create_camera_params_from_enhanced_intrinsics(self, camera_name: str, 
                                                           intrinsics: Dict, 
                                                           camera_params_map: Dict, 
                                                           camera_name_to_id: Dict) -> int:
        """Create camera parameters from enhanced intrinsics data"""
        
        if camera_name in camera_name_to_id:
            return camera_name_to_id[camera_name]
        
        camera_id = len(camera_params_map)
        camera_name_to_id[camera_name] = camera_id
        
        # Extract camera parameters from intrinsics
        width = intrinsics.get("width", 1920)
        height = intrinsics.get("height", 1080)
        
        # Get focal length in pixels
        focal_length_pixels = intrinsics.get("focal_length_pixels", None)
        if focal_length_pixels is None:
            # Fallback to computing from normalized focal length
            focal_length_normalized = intrinsics.get("focal_length", 0.73)
            focal_length_pixels = focal_length_normalized * max(width, height)
        
        # Principal point is typically at image center for Mapillary data
        fx = fy = focal_length_pixels
        cx, cy = width / 2.0, height / 2.0
        
        # Ensure principal point is valid (>= 0)
        cx = max(0.0, cx)
        cy = max(0.0, cy)
        
        # Camera matrix in PyCuSFM format (3x3 intrinsic matrix)
        camera_matrix = [
            fx, 0, cx,
            0, fy, cy,
            0, 0, 1
        ]
        
        # Extract distortion coefficients from distortion_coefficients object
        distortion_data = intrinsics.get("distortion_coefficients", {})
        distortion_coeffs = [
            distortion_data.get("k1", 0.0),
            distortion_data.get("k2", 0.0),
            distortion_data.get("p1", 0.0),
            distortion_data.get("p2", 0.0),
            distortion_data.get("k3", 0.0)
        ]
        
        # Create camera parameters structure
        camera_params = {
            "sensor_meta_data": {
                "sensor_id": camera_id,
                "sensor_type": "CAMERA",
                "sensor_name": camera_name,
                "frequency": 30,  # Default frequency
                "sensor_to_vehicle_transform": {
                    "axis_angle": {"x": 0, "y": 0, "z": 0, "angle_degrees": 0},
                    "translation": {"x": 0, "y": 0, "z": 0}
                }
            },
            "calibration_parameters": {
                "image_width": width,
                "image_height": height,
                "camera_matrix": {"data": camera_matrix},
                "distortion_coefficients": {"data": distortion_coeffs}
            }
        }
        
        camera_params_map[camera_id] = camera_params
        return camera_id
    
    def _convert_mapillary_pose_to_cusfm(self, pose_data: Dict, ref_lat: float = None, ref_lon: float = None, ref_alt: float = None) -> Dict:
        """Convert Mapillary pose data to PyCuSFM camera_to_world format"""
        
        # Extract GPS coordinates
        geometry = pose_data.get("computed_geometry") or pose_data.get("geometry") or {}
        if geometry is not None and geometry.get("type") == "Point" and "coordinates" in geometry:
            lon, lat = geometry["coordinates"][:2]
            
            # Get altitude
            altitude = pose_data.get("computed_altitude", pose_data.get("altitude", 0.0))
            
            # Convert GPS to local ENU coordinates using reference point
            if ref_lat is not None and ref_lon is not None:
                x, y, z = self._gps_to_local_enu(lat, lon, altitude, ref_lat, ref_lon, ref_alt)
            else:
                # Use first point as reference
                x, y, z = self._gps_to_local_enu(lat, lon, altitude)
        else:
            x = y = z = 0.0
        
        # Extract rotation from compass angle and computed rotation
        compass_angle = pose_data.get("computed_compass_angle", pose_data.get("compass_angle", 0.0))
        computed_rotation = pose_data.get("computed_rotation", [0, 0, 0])
        
        # Convert rotation to axis-angle representation
        if computed_rotation and len(computed_rotation) == 3 and any(abs(r) > 0.001 for r in computed_rotation):
            # Use computed rotation if available and non-zero
            axis_angle_rad = np.array(computed_rotation)
            angle_rad = np.linalg.norm(axis_angle_rad)
            angle_deg = np.degrees(angle_rad)
            
            if angle_rad > 0.001:  # Avoid division by very small numbers
                axis = axis_angle_rad / angle_rad
            else:
                axis = np.array([0, 0, 1])
                angle_deg = 0.0
        else:
            # Use compass angle for Z-axis rotation
            # Convert compass bearing (degrees from North) to camera rotation
            # Mapillary compass angles: 0 = North, 90 = East, 180 = South, 270 = West
            axis = np.array([0, 0, 1])
            angle_deg = compass_angle
        
        return {
            "axis_angle": {
                "x": float(axis[0]),
                "y": float(axis[1]),
                "z": float(axis[2]),
                "angle_degrees": float(angle_deg)
            },
            "translation": {
                "x": float(x),
                "y": float(y),
                "z": float(z)
            }
        }
    
    def _gps_to_local_enu(self, lat: float, lon: float, alt: float, 
                         ref_lat: Optional[float] = None, ref_lon: Optional[float] = None, ref_alt: Optional[float] = None) -> Tuple[float, float, float]:
        """
        Convert GPS coordinates to local East-North-Up (ENU) coordinates
        
        Args:
            lat: Latitude in degrees
            lon: Longitude in degrees  
            alt: Altitude in meters
            ref_lat: Reference latitude (if None, uses current lat as reference)
            ref_lon: Reference longitude (if None, uses current lon as reference)
            ref_alt: Reference altitude (if None, uses 0)
            
        Returns:
            Tuple of (East, North, Up) coordinates in meters
        """
        if ref_lat is None:
            ref_lat = lat
        if ref_lon is None:
            ref_lon = lon
        if ref_alt is None:
            ref_alt = 0.0
            
        # Earth radius in meters
        R = 6378137.0
        
        # Convert to radians
        lat_rad = math.radians(lat)
        lon_rad = math.radians(lon)
        ref_lat_rad = math.radians(ref_lat)
        ref_lon_rad = math.radians(ref_lon)
        
        # Compute differences
        dlat = lat_rad - ref_lat_rad
        dlon = lon_rad - ref_lon_rad
        
        # Compute local coordinates
        # East: positive towards East
        east = R * dlon * math.cos(ref_lat_rad)
        
        # North: positive towards North
        north = R * dlat
        
        # Up: altitude difference
        up = alt - ref_alt
        
        return east, north, up
    
    def _get_timestamp_from_pose_data(self, pose_data: Dict) -> int:
        """Extract timestamp from pose data and convert to microseconds"""
        
        # Try capture_time first (milliseconds)
        capture_time = pose_data.get("capture_time")
        if capture_time:
            # Convert milliseconds to microseconds
            return int(capture_time) * 1000
        
        # Fallback to current time if no timestamp available
        return int(datetime.now().timestamp() * 1_000_000)
    
    def _create_frames_meta_from_sfm(self, sfm_data: Dict, camera_models: Dict, exif_data: Dict):
        """Create frames_meta.json from SfM reconstruction data"""
        logger.info("Creating frames_meta.json from SfM data")
        
        keyframes_metadata = []
        camera_params_map = {}
        camera_name_to_id = {}
        
        frame_id = 0
        sample_id = 0
        
        # Process each shot individually (monocular sequence)
        shots_by_timestamp = []
        for shot_id, shot_data in sfm_data.get("shots", {}).items():
            timestamp_us = self._get_timestamp_from_shot(shot_data, exif_data.get(shot_id, {}))
            shots_by_timestamp.append((timestamp_us, shot_id, shot_data))
        
        # Sort by timestamp
        shots_by_timestamp.sort(key=lambda x: x[0])
        
        logger.info(f"Processing {len(shots_by_timestamp)} shots from SfM data")
        
        for idx, (timestamp_us, shot_id, shot_data) in enumerate(shots_by_timestamp):
            # Get camera info
            camera_key = shot_data.get("camera", "")
            camera_name = self._get_camera_name_for_shot(camera_key, sfm_data.get("cameras", {}))
            
            # Create camera parameters if not exists
            camera_param_id = self._get_or_create_camera_params_from_sfm(
                camera_name, camera_key, sfm_data.get("cameras", {}), camera_models, 
                camera_params_map, camera_name_to_id, exif_data.get(shot_id, {})
            )
            
            # Convert SfM pose to PyCuSFM format
            pose = self._convert_sfm_pose_to_cusfm(shot_data)
            
            # Create image filename with sequence index prefix
            image_name = f"{camera_name}/{idx:06d}_{shot_id}.jpg"
            
            # Create keyframe metadata - each shot gets unique sample_id
            keyframe = {
                "id": str(frame_id),
                "camera_params_id": str(camera_param_id),
                "timestamp_microseconds": str(timestamp_us),
                "image_name": image_name,
                "camera_to_world": pose,
                "synced_sample_id": str(sample_id)  # Each shot gets unique sample_id
            }
            
            keyframes_metadata.append(keyframe)
            frame_id += 1
            sample_id += 1
        
        # Sort by timestamp
        keyframes_metadata.sort(key=lambda x: int(x["timestamp_microseconds"]))
        
        # Create frames_meta structure
        frames_meta = {
            "keyframes_metadata": keyframes_metadata,
            "initial_pose_type": "EGO_MOTION",  # Use EGO_MOTION for SfM data
            "camera_params_id_to_session_name": {
                str(param_id): "0" for param_id in camera_params_map.keys()
            },
            "camera_params_id_to_camera_params": {
                str(param_id): params for param_id, params in camera_params_map.items()
            }
        }
        
        # Add reference coordinates if available
        self._add_reference_coordinates(frames_meta, exif_data)
        
        # Copy images in the same order as the metadata
        self._copy_images_with_order([shot_id for _, shot_id, _ in shots_by_timestamp])
        
        # Save frames_meta.json
        with open(self.frames_meta_file, 'w') as f:
            json.dump(frames_meta, f, indent=2)
        
        logger.info(f"✅ Created frames_meta.json with {len(keyframes_metadata)} keyframes from SfM data")
        logger.info(f"   📄 Output file: {self.frames_meta_file}")
        logger.info(f"   🎥 Cameras used: {len(camera_params_map)}")
    
    def _create_frames_meta_from_exif(self, camera_models: Dict, exif_data: Dict):
        """Create frames_meta.json from EXIF data when no SfM data is available"""
        logger.info("Creating frames_meta.json from EXIF data")
        
        keyframes_metadata = []
        camera_params_map = {}
        camera_name_to_id = {}
        
        # Collect and sort images by timestamp for monocular sequence processing
        images_by_timestamp = []
        
        for image_file in self.images_dir.glob("*.jpg"):
            image_id = image_file.stem
            exif = exif_data.get(image_id, {})
            timestamp_us = self._get_timestamp_microseconds(exif)
            images_by_timestamp.append((timestamp_us, image_id, exif))
        
        # Sort by timestamp
        images_by_timestamp.sort(key=lambda x: x[0])
        
        logger.info(f"Processing {len(images_by_timestamp)} images from EXIF data")
        
        frame_id = 0
        sample_id = 0
        
        # Process each image individually (monocular sequence)
        for idx, (timestamp_us, image_id, exif) in enumerate(images_by_timestamp):
            camera_name = self._get_camera_name_for_image(image_id)
            
            # Create camera parameters
            camera_param_id = self._get_or_create_camera_params_from_exif(
                camera_name, camera_models, camera_params_map, camera_name_to_id, exif
            )
            
            # Convert GPS to pose
            pose = self._gps_to_pose(exif.get("gps"))
            
            # Create keyframe metadata - each image gets unique sample_id
            keyframe = {
                "id": str(frame_id),
                "camera_params_id": str(camera_param_id),
                "timestamp_microseconds": str(timestamp_us),
                "image_name": f"{camera_name}/{idx:06d}_{image_id}.jpg",
                "camera_to_world": pose,
                "synced_sample_id": str(sample_id)  # Each image gets unique sample_id
            }
            
            keyframes_metadata.append(keyframe)
            frame_id += 1
            sample_id += 1
        
        # Create frames_meta structure
        frames_meta = {
            "keyframes_metadata": keyframes_metadata,
            "initial_pose_type": "EGO_MOTION",  # Use EGO_MOTION for sequential data
            "camera_params_id_to_session_name": {
                str(param_id): "0" for param_id in camera_params_map.keys()
            },
            "camera_params_id_to_camera_params": {
                str(param_id): params for param_id, params in camera_params_map.items()
            }
        }
        
        # Add reference coordinates
        self._add_reference_coordinates(frames_meta, exif_data)
        
        # Copy images in the same order as the metadata
        self._copy_images_with_order([image_id for _, image_id, _ in images_by_timestamp])
        
        # Save frames_meta.json
        with open(self.frames_meta_file, 'w') as f:
            json.dump(frames_meta, f, indent=2)
        
        logger.info(f"✅ Created frames_meta.json with {len(keyframes_metadata)} keyframes from EXIF data")
        logger.info(f"   📄 Output file: {self.frames_meta_file}")
        logger.info(f"   🎥 Cameras used: {len(camera_params_map)}")
    
    def _get_camera_name_for_shot(self, camera_key: str, cameras: Dict) -> str:
        """Get camera name from SfM camera key"""
        if camera_key and camera_key in cameras:
            # Use first part of camera key as name
            return camera_key.split()[0] if ' ' in camera_key else "camera_0"
        return "camera_0"
    
    def _convert_sfm_pose_to_cusfm(self, shot_data: Dict) -> Dict:
        """Convert SfM pose to PyCuSFM format"""
        # Get rotation and translation from SfM shot
        rotation = shot_data.get("rotation", [0, 0, 0])
        translation = shot_data.get("translation", [0, 0, 0])
        
        # Convert axis-angle rotation to degrees
        if len(rotation) == 3:
            # Calculate angle in degrees
            angle_rad = np.linalg.norm(rotation)
            angle_deg = np.degrees(angle_rad)
            
            # Normalize axis
            if angle_rad > 0:
                axis = np.array(rotation) / angle_rad
            else:
                axis = [0, 0, 1]  # Default axis
            
            return {
                "axis_angle": {
                    "x": float(axis[0]),
                    "y": float(axis[1]),
                    "z": float(axis[2]),
                    "angle_degrees": float(angle_deg)
                },
                "translation": {
                    "x": float(translation[0]),
                    "y": float(translation[1]),
                    "z": float(translation[2])
                }
            }
        
        # Fallback to identity pose
        return {
            "axis_angle": {"x": 0, "y": 0, "z": 0, "angle_degrees": 0},
            "translation": {"x": 0, "y": 0, "z": 0}
        }
    
    def _get_timestamp_from_shot(self, shot_data: Dict, exif: Dict) -> int:
        """Get timestamp from shot data or EXIF"""
        # Try capture_time from shot data first
        if "capture_time" in shot_data:
            return int(shot_data["capture_time"] * 1_000_000)
        
        # Fall back to EXIF timestamp
        return self._get_timestamp_microseconds(exif)
    
    def _get_or_create_camera_params_from_sfm(self, camera_name: str, camera_key: str, 
                                            cameras: Dict, camera_models: Dict,
                                            camera_params_map: Dict, camera_name_to_id: Dict, 
                                            exif: Dict) -> int:
        """Create camera parameters from SfM camera data"""
        if camera_name not in camera_name_to_id:
            camera_name_to_id[camera_name] = len(camera_name_to_id)
        
        camera_param_id = camera_name_to_id[camera_name]
        
        if camera_param_id not in camera_params_map:
            # Get camera data from SfM
            camera_data = cameras.get(camera_key, {})
            
            # Extract camera parameters
            width = camera_data.get("width", exif.get("width", 1920))
            height = camera_data.get("height", exif.get("height", 1080))
            
            # Determine projection type and parameters
            projection_type = camera_data.get("projection_type", "perspective")
            
            if projection_type == "perspective":
                # For perspective cameras, extract focal length
                focal = 0.85  # Default
                if "focal" in camera_data:
                    focal = camera_data["focal"]
                
                # Convert relative focal to pixels if needed
                if focal <= 1.0:
                    focal_pixels = focal * max(width, height)
                else:
                    focal_pixels = focal
                
                # Principal point
                cx = width / 2
                cy = height / 2
                
                # Distortion parameters
                k1 = camera_data.get("k1", 0.0)
                k2 = camera_data.get("k2", 0.0)
                
                # Camera matrix (3x3 intrinsic matrix)
                camera_matrix = [
                    focal_pixels, 0, cx,
                    0, focal_pixels, cy,
                    0, 0, 1
                ]
                distortion_coeffs = [k1, k2, 0, 0, 0]
                
            elif projection_type in ["equirectangular", "spherical"]:
                # For spherical/equirectangular cameras
                fx = width / (2 * np.pi)
                fy = height / np.pi
                cx = width / 2
                cy = height / 2
                
                camera_matrix = [
                    fx, 0, cx,
                    0, fy, cy,
                    0, 0, 1
                ]
                distortion_coeffs = [0, 0, 0, 0, 0]
            else:
                # Default perspective projection
                focal_pixels = 0.85 * max(width, height)
                cx = width / 2
                cy = height / 2
                
                camera_matrix = [
                    focal_pixels, 0, cx,
                    0, focal_pixels, cy,
                    0, 0, 1
                ]
                distortion_coeffs = [0, 0, 0, 0, 0]
            
            # Create camera parameters
            camera_params = {
                "sensor_meta_data": {
                    "sensor_id": camera_param_id,
                    "sensor_type": "CAMERA",
                    "sensor_name": camera_name,
                    "frequency": 30,
                    "sensor_to_vehicle_transform": {
                        "axis_angle": {"x": 0, "y": 0, "z": 0, "angle_degrees": 0},
                        "translation": {"x": 0, "y": 0, "z": 0}
                    }
                },
                "calibration_parameters": {
                    "image_width": int(width),
                    "image_height": int(height),
                    "camera_matrix": {"data": camera_matrix},
                    "distortion_coefficients": {"data": distortion_coeffs}
                }
            }
            
            camera_params_map[camera_param_id] = camera_params
        
        return camera_param_id
    
    def _get_or_create_camera_params_from_exif(self, camera_name: str, camera_models: Dict,
                                             camera_params_map: Dict, camera_name_to_id: Dict,
                                             exif: Dict) -> int:
        """Create camera parameters from EXIF data"""
        if camera_name not in camera_name_to_id:
            camera_name_to_id[camera_name] = len(camera_name_to_id)
        
        camera_param_id = camera_name_to_id[camera_name]
        
        if camera_param_id not in camera_params_map:
            # Get camera model info
            camera_id = exif.get("camera", camera_name)
            camera_model = camera_models.get(camera_id, {})
            
            # Extract camera parameters
            width = exif.get("width", camera_model.get("width", 1920))
            height = exif.get("height", camera_model.get("height", 1080))
            focal = camera_model.get("focal", 0.85)
            
            # Convert relative focal to pixel focal
            if focal <= 1.0:
                focal_pixels = focal * max(width, height)
            else:
                focal_pixels = focal
            
            # Principal point
            cx = width / 2
            cy = height / 2
            
            # Distortion coefficients
            k1 = camera_model.get("k1", 0.0)
            k2 = camera_model.get("k2", 0.0)
            
            # Camera matrix (3x3 intrinsic matrix)
            camera_matrix = [
                focal_pixels, 0, cx,
                0, focal_pixels, cy,
                0, 0, 1
            ]
            
            # Create camera parameters
            camera_params = {
                "sensor_meta_data": {
                    "sensor_id": camera_param_id,
                    "sensor_type": "CAMERA",
                    "sensor_name": camera_name,
                    "frequency": 30,
                    "sensor_to_vehicle_transform": {
                        "axis_angle": {"x": 0, "y": 0, "z": 0, "angle_degrees": 0},
                        "translation": {"x": 0, "y": 0, "z": 0}
                    }
                },
                "calibration_parameters": {
                    "image_width": int(width),
                    "image_height": int(height),
                    "camera_matrix": {
                        "data": camera_matrix
                    },
                    "distortion_coefficients": {
                        "data": [k1, k2, 0, 0, 0]
                    }
                }
            }
            
            camera_params_map[camera_param_id] = camera_params
        
        return camera_param_id
    
    def _get_timestamp_microseconds(self, exif: Dict) -> int:
        """Extract timestamp and convert to microseconds"""
        timestamp = exif.get("capture_time") or exif.get("captured_at") or 0
        
        if isinstance(timestamp, str):
            try:
                dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                timestamp = dt.timestamp()
            except:
                timestamp = datetime.now().timestamp()
        
        return int(timestamp * 1_000_000)
    
    def _gps_to_pose(self, gps_data: Optional[Dict]) -> Dict:
        """Convert GPS coordinates to camera pose"""
        if not gps_data:
            return {
                "axis_angle": {"x": 0, "y": 0, "z": 0, "angle_degrees": 0},
                "translation": {"x": 0, "y": 0, "z": 0}
            }
        
        lat = gps_data.get("latitude", 0)
        lon = gps_data.get("longitude", 0)
        alt = gps_data.get("altitude", 0)
        
        # Simple local coordinate conversion (should use reference point)
        x = (lon - 0) * 111000 * np.cos(np.radians(lat))
        y = (lat - 0) * 111000
        z = alt
        
        return {
            "axis_angle": {"x": 0, "y": 0, "z": 0, "angle_degrees": 0},
            "translation": {"x": x, "y": y, "z": z}
        }
    
    def _add_reference_coordinates(self, frames_meta: Dict, exif_data: Dict):
        """Add reference GPS coordinates to frames_meta"""
        if exif_data:
            first_exif = next(iter(exif_data.values()), {})
            gps_data = first_exif.get("gps")
            if gps_data:
                frames_meta["reference_latlngalt"] = {
                    "latitude": gps_data.get("latitude", 0),
                    "longitude": gps_data.get("longitude", 0),
                    "altitude": gps_data.get("altitude", 0)
                }


def main():
    parser = argparse.ArgumentParser(
        description="Convert Mapillary data to PyCuSFM format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s data/mapillary/msp_downtown_2 output/pycusfm_data
  %(prog)s --input data/mapillary --output output/cusfm -v
        """
    )
    
    parser.add_argument('input_dir', nargs='?', help='Input Mapillary data directory')
    parser.add_argument('output_dir', nargs='?', help='Output directory for PyCuSFM format')
    parser.add_argument('--input', '-i', dest='input_alt', help='Input directory (alternative)')
    parser.add_argument('--output', '-o', dest='output_alt', help='Output directory (alternative)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose logging')
    
    args = parser.parse_args()
    
    # Handle input/output arguments
    input_dir = args.input_dir or args.input_alt
    output_dir = args.output_dir or args.output_alt
    
    if not input_dir or not output_dir:
        parser.error("Both input and output directories are required")
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        converter = MapillaryToCuSFMConverter(input_dir, output_dir)
        converter.convert()
    except Exception as e:
        logger.error(f"❌ Conversion failed: {e}", exc_info=args.verbose)
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
