#!/usr/bin/env python3#!/usr/bin/env python3#!/usr/bin/env python3

"""

Mapillary to PyCuSFM Converter""""""



This script converts Mapillary data (OpenSfM format) to PyCuSFM's frames_meta.json format.Mapillary to PyCuSFM ConverterMapillary to PyCuSFM Converter



Mapillary Structure:

- cameras/ : Individual camera intrinsics files (per image)

- poses/ : Individual pose files with GPS and rotation dataThis script converts Mapillary data (OpenSfM format) to PyCuSFM's frames_meta.json format.This script converts Mapillary data (OpenSfM format) to PyCuSFM's frames_meta.json format.

- sequences/ : Temporal organization of images

- images/ : Street view images

- config.yaml : Processing configuration

- reference.json : GPS reference frameMapillary Structure:Mapillary Structure:



PyCuSFM Structure:- cameras/ : Individual camera intrinsics files (per image)- cameras/ : Individual camera intrinsics files

- frames_meta.json : Unified metadata with keyframes_metadata array

- images/ : Organized images (can be symlinked)- poses/ : Individual pose files with GPS and rotation data- poses/ : Individual pose files with GPS and rotation data



Author: Generated for 3D Reconstruction Pipeline- sequences/ : Temporal organization of images- sequences/ : Temporal organization of images

Date: October 2025

"""- images/ : Street view images- images/ : Street view images



import json- config.yaml : Processing configuration- config.yaml : Processing configuration

import os

import shutil- reference.json : GPS reference frame- reference.json : GPS reference frame

from pathlib import Path

from typing import Dict, List, Optional, Tuple

import numpy as np

from datetime import datetimePyCuSFM Structure:PyCuSFM Structure:

import logging

import math- frames_meta.json : Unified metadata with keyframes_metadata array- frames_meta.json : Unified metadata with keyframes_metadata array

import argparse

- images/ : Organized images (can be symlinked)"""

# Configure logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)

Author: Generated for 3D Reconstruction Pipelineimport json



class MapillaryToCuSFMConverter:Date: October 2025import os

    """Convert Mapillary data structure to PyCuSFM format"""

    """import shutil

    def __init__(self, mapillary_data_dir: Path, output_dir: Path):

        self.mapillary_dir = Path(mapillary_data_dir)from pathlib import Path

        self.output_dir = Path(output_dir)

        import jsonfrom typing import Dict, List, Optional, Tuple

        # Mapillary structure

        self.images_dir = self.mapillary_dir / "images"import osimport numpy as np

        self.cameras_dir = self.mapillary_dir / "cameras"

        self.poses_dir = self.mapillary_dir / "poses"import numpy as npfrom datetime import datetime

        self.sequences_dir = self.mapillary_dir / "sequences"

        self.reference_file = self.mapillary_dir / "reference.json"import yamlimport logging

        

        # Output structurefrom pathlib import Pathimport math

        self.output_images_dir = self.output_dir / "images"

        self.frames_meta_file = self.output_dir / "frames_meta.json"from typing import Dict, List, Any, Tuple, Optional

        

        # Validate inputimport argparse# Configure logging

        if not self.mapillary_dir.exists():

            raise ValueError(f"Mapillary directory not found: {self.mapillary_dir}")from datetime import datetimelogging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

        if not self.images_dir.exists():

            raise ValueError(f"Images directory not found: {self.images_dir}")import logginglogger = logging.getLogger(__name__)

        

        # Create output directoryimport shutil

        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.output_images_dir.mkdir(exist_ok=True)

        

        logger.info(f"Initialized converter:")# Set up loggingclass MapillaryToCuSFMConverter:

        logger.info(f"  Input:  {self.mapillary_dir}")

        logger.info(f"  Output: {self.output_dir}")logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')    """Convert Mapillary data structure to PyCuSFM format"""

    

    def convert(self):logger = logging.getLogger(__name__)    

        """Main conversion method"""

        logger.info("Starting Mapillary to PyCuSFM conversion...")    def __init__(self, mapillary_data_dir: Path, output_dir: Path):

        

        # Load camera intrinsics        self.mapillary_dir = Path(mapillary_data_dir)

        logger.info("Loading camera intrinsics...")

        camera_intrinsics = self._load_camera_intrinsics()class MapillaryToCuSFMConverter:        self.output_dir = Path(output_dir)

        

        # Load pose data    """        

        logger.info("Loading pose data...")

        pose_data = self._load_pose_data()    Converter for Mapillary OpenSfM format to PyCuSFM frames_meta.json format.        # Enhanced Mapillary structure (new format)

        

        # Create frames_meta.json            self.images_dir = self.mapillary_dir / "images"

        self._create_frames_meta_from_enhanced_data(camera_intrinsics, pose_data)

            This converter handles:        self.cameras_dir = self.mapillary_dir / "cameras"

        # Copy images

        logger.info("Organizing images...")    - Multiple camera types and sequences        self.poses_dir = self.mapillary_dir / "poses"

        self._copy_images_enhanced()

            - GPS coordinate conversion to local ENU coordinates        self.sequences_dir = self.mapillary_dir / "sequences"

        logger.info("✅ Conversion completed successfully!")

        logger.info(f"   Output: {self.output_dir}")    - Camera intrinsics with distortion        

    

    def _load_camera_intrinsics(self) -> Dict:    - Pose data with rotation and translation        # Legacy Mapillary structure (fallback)

        """Load camera intrinsics from enhanced format cameras/ directory"""

        intrinsics_data = {}    - Temporal sequencing of images        self.exif_dir = self.mapillary_dir / "exif"

        

        if not self.cameras_dir.exists():    """        self.camera_models_file = self.mapillary_dir / "camera_models.json"

            logger.warning("cameras directory not found")

            return intrinsics_data            self.pointclouds_dir = self.mapillary_dir / "pointclouds"

        

        logger.info(f"Loading camera intrinsics from {self.cameras_dir}")    def __init__(self, input_dir: str, output_dir: str):        

        for intrinsics_file in self.cameras_dir.glob("*.json"):

            image_id = intrinsics_file.stem        """        # Output structure for PyCuSFM

            try:

                with open(intrinsics_file, 'r') as f:        Initialize converter.        self.output_images_dir = self.output_dir / "images"

                    intrinsics_data[image_id] = json.load(f)

            except Exception as e:                self.frames_meta_file = self.output_dir / "frames_meta.json"

                logger.error(f"Error loading intrinsics file {intrinsics_file}: {e}")

                Args:        

        logger.info(f"Loaded intrinsics for {len(intrinsics_data)} images")

        return intrinsics_data            input_dir: Path to Mapillary data directory        # Create output directories

    

    def _load_pose_data(self) -> Dict:            output_dir: Path to output directory for PyCuSFM format        self.output_dir.mkdir(parents=True, exist_ok=True)

        """Load pose data from enhanced format poses/ directory"""

        pose_data = {}        """        self.output_images_dir.mkdir(parents=True, exist_ok=True)

        

        if not self.poses_dir.exists():        self.input_dir = Path(input_dir)    

            logger.warning("poses directory not found")

            return pose_data        self.output_dir = Path(output_dir)    def convert(self):

        

        logger.info(f"Loading pose data from {self.poses_dir}")                """Main conversion function"""

        for pose_file in self.poses_dir.glob("*.json"):

            image_id = pose_file.stem.replace("_pose", "")        # Validate input directory        print("Converting Mapillary data to PyCuSFM format...")

            try:

                with open(pose_file, 'r') as f:        required_dirs = ['cameras', 'poses', 'images']        

                    pose_data[image_id] = json.load(f)

            except Exception as e:        for dir_name in required_dirs:        # Check which format we're dealing with

                logger.error(f"Error loading pose file {pose_file}: {e}")

                    dir_path = self.input_dir / dir_name        if self.cameras_dir.exists() and self.poses_dir.exists():

        logger.info(f"Loaded poses for {len(pose_data)} images")

        return pose_data            if not dir_path.exists():            # Enhanced format

    

    def _copy_images_enhanced(self):                raise ValueError(f"Required directory '{dir_name}' not found in {input_dir}")            print("✅ Detected enhanced Mapillary format with cameras/ and poses/ directories")

        """Copy images for enhanced format"""

        if not self.images_dir.exists():                    self._convert_enhanced_format()

            logger.error("images directory not found")

            return        # Create output directory        else:

        

        # Create camera subdirectories based on camera models found in intrinsics        self.output_dir.mkdir(parents=True, exist_ok=True)            # Legacy format

        camera_dirs = set()

                            print("⚠️  Using legacy conversion method")

        # Load camera intrinsics to determine camera types

        for intrinsics_file in self.cameras_dir.glob("*.json"):        logger.info(f"Initialized converter:")            self._convert_legacy_format()

            try:

                with open(intrinsics_file, 'r') as f:        logger.info(f"  Input:  {self.input_dir}")        

                    data = json.load(f)

                camera_name = self._get_camera_name_from_intrinsics(data)        logger.info(f"  Output: {self.output_dir}")        print(f"✅ Conversion completed! PyCuSFM data saved to: {self.output_dir}")

                camera_dirs.add(camera_name)

            except Exception as e:        

                logger.error(f"Error reading {intrinsics_file}: {e}")

            def convert(self):    def _convert_enhanced_format(self):

        # Default camera if none found

        if not camera_dirs:        """Main conversion function."""        """Convert enhanced Mapillary format with cameras/, poses/, sequences/ directories"""

            camera_dirs.add("camera_0")

                logger.info("=" * 60)        

        # Create camera subdirectories

        for camera_dir in camera_dirs:        logger.info("Starting Mapillary to PyCuSFM conversion")        # Load enhanced metadata

            (self.output_images_dir / camera_dir).mkdir(exist_ok=True)

                logger.info("=" * 60)        camera_intrinsics = self._load_camera_intrinsics()

        # Copy images

        copied_count = 0                pose_data = self._load_pose_data()

        for image_file in self.images_dir.glob("*.jpg"):

            image_id = image_file.stem        # Load all data        

            

            # Get camera name for this image        logger.info("\n📥 Loading Mapillary data...")        # Copy images to output directory

            camera_name = self._get_camera_name_for_image_enhanced(image_id)

                    reference = self.load_reference_frame()        self._copy_images_enhanced()

            # Copy image

            dest_path = self.output_images_dir / camera_name / f"{image_id}.jpg"        cameras = self.load_camera_intrinsics()        

            shutil.copy2(image_file, dest_path)

            copied_count += 1        poses = self.load_poses()        # Create frames_meta.json from enhanced data

        

        logger.info(f"Copied {copied_count} images to camera subdirectories")        sequences = self.load_sequences()        self._create_frames_meta_from_enhanced_data(camera_intrinsics, pose_data)

    

    def _get_camera_name_from_intrinsics(self, intrinsics_data: Dict) -> str:            

        """Extract camera name from intrinsics data"""

        # Try to get camera make/model for naming        # Convert to PyCuSFM format    def _convert_legacy_format(self):

        make = intrinsics_data.get("make", "")

        model = intrinsics_data.get("model", "")        logger.info("\n🔄 Converting to PyCuSFM format...")        """Convert legacy Mapillary format"""

        

        # Get focal length and principal point to help distinguish cameras        frames_meta = self.convert_to_cusfm_format(cameras, poses, sequences, reference)        # Load SfM reconstruction data

        camera_params = intrinsics_data.get("camera_parameters", [])

        focal_length = camera_params[0] if len(camera_params) > 0 else 0                sfm_data = self._load_sfm_reconstruction_data()

        

        # Create a more unique camera identifier        # Save output        

        if make and model:

            base_name = f"{make}_{model}".replace(" ", "_").lower()        logger.info("\n💾 Saving output files...")        # Load camera models

        elif model:

            base_name = f"{model}".replace(" ", "_").lower()        self.save_output(frames_meta)        camera_models = self._load_camera_models()

        else:

            base_name = "camera"                

        

        # Add focal length to make it more unique (rounded to avoid floating point precision issues)        logger.info("\n" + "=" * 60)        # Load EXIF data

        if focal_length > 0:

            focal_str = f"_f{int(focal_length)}"        logger.info("✅ Conversion completed successfully!")        exif_data = self._load_exif_data()

            return f"{base_name}{focal_str}"

        else:        logger.info("=" * 60)        

            return f"{base_name}_0"

            logger.info(f"Output directory: {self.output_dir}")        # Copy images to output directory

    def _get_camera_name_for_image_enhanced(self, image_id: str) -> str:

        """Get camera name for an image in enhanced format"""        logger.info(f"Total keyframes: {len(frames_meta['keyframes_metadata'])}")        self._copy_images()

        # Try to load intrinsics for this image

        intrinsics_file = self.cameras_dir / f"{image_id}.json"        logger.info(f"Unique cameras: {len(set(kf['camera_params_id'] for kf in frames_meta['keyframes_metadata']))}")        

        if intrinsics_file.exists():

            try:            # Create frames_meta.json using SfM data if available, otherwise EXIF

                with open(intrinsics_file, 'r') as f:

                    data = json.load(f)    def load_reference_frame(self) -> Dict[str, Any]:        if sfm_data:

                return self._get_camera_name_from_intrinsics(data)

            except:        """Load GPS reference frame."""            self._create_frames_meta_from_sfm(sfm_data, camera_models, exif_data)

                pass

                ref_file = self.input_dir / 'reference.json'        else:

        return "camera_0"  # Default fallback

            if ref_file.exists():            self._create_frames_meta_from_exif(camera_models, exif_data)

    def _create_frames_meta_from_enhanced_data(self, camera_intrinsics: Dict, pose_data: Dict):

        """Create frames_meta.json from enhanced Mapillary data with cameras/ and poses/"""            with open(ref_file, 'r') as f:    

        logger.info("Creating frames_meta.json from enhanced Mapillary data")

                        reference = json.load(f)    def _load_sfm_reconstruction_data(self) -> Optional[Dict]:

        keyframes_metadata = []

        camera_params_map = {}                logger.info(f"  ✓ Loaded reference frame: lat={reference.get('latitude', 0):.6f}, lon={reference.get('longitude', 0):.6f}")        """Load SfM reconstruction data from Mapillary point clouds"""

        camera_name_to_id = {}

                        return reference        sfm_data = {"cameras": {}, "shots": {}, "points": {}}

        # Process each image that has both intrinsics and pose data

        common_images = set(camera_intrinsics.keys()) & set(pose_data.keys())        logger.info("  ⚠ No reference frame found, will use first GPS point")        

        logger.info(f"Found {len(common_images)} images with both intrinsics and pose data")

                return {}        if not self.pointclouds_dir.exists():

        if not common_images:

            logger.error("No images found with both camera intrinsics and pose data")                logger.warning("No pointclouds directory found")

            return

            def load_camera_intrinsics(self) -> Dict[str, Dict[str, Any]]:            return None

        # Find reference GPS coordinate (use first valid GPS point)

        ref_lat, ref_lon, ref_alt = self._find_reference_gps_point(pose_data)        """Load camera intrinsics from cameras/ directory."""        

        logger.info(f"Using reference GPS point: lat={ref_lat:.6f}, lon={ref_lon:.6f}, alt={ref_alt:.2f}")

                cameras = {}        # Look for SfM cluster files

        # Analyze data structure for diagnostics

        self._analyze_input_data(camera_intrinsics, pose_data)        cameras_dir = self.input_dir / 'cameras'        cluster_files = list(self.pointclouds_dir.glob("sfm_cluster_*.json"))

        

        # For Mapillary data, typically each image is captured at a different time                if not cluster_files:

        # So we'll treat each image as a separate sample (no multi-camera synchronization)

        # This is different from stereo camera setups like KITTI        camera_files = list(cameras_dir.glob('*.json'))            # Try combined reconstruction file

        

        frame_id = 0        for camera_file in camera_files:            combined_file = self.pointclouds_dir / "combined_reconstruction.json"

        sample_id = 0

                    image_id = camera_file.stem            if combined_file.exists():

        # Process each image individually (monocular sequence)

        for image_id in sorted(common_images):            with open(camera_file, 'r') as f:                cluster_files = [combined_file]

            intrinsics = camera_intrinsics[image_id]

            pose = pose_data[image_id]                cameras[image_id] = json.load(f)        

            

            # Get timestamp for this specific image                if not cluster_files:

            timestamp_us = self._get_timestamp_from_pose_data(pose)

                    logger.info(f"  ✓ Loaded {len(cameras)} camera intrinsics")            logger.warning("No SfM reconstruction data found")

            # Get camera name and create camera parameters

            camera_name = self._get_camera_name_from_intrinsics(intrinsics)        return cameras            return None

            camera_param_id = self._get_or_create_camera_params_from_enhanced_intrinsics(

                camera_name, intrinsics, camera_params_map, camera_name_to_id            

            )

                def load_poses(self) -> Dict[str, Dict[str, Any]]:        logger.info(f"Loading SfM data from {len(cluster_files)} files")

            # Convert Mapillary pose to PyCuSFM format using reference point

            cusfm_pose = self._convert_mapillary_pose_to_cusfm(pose, ref_lat, ref_lon, ref_alt)        """Load pose data from poses/ directory."""        

            

            # Create image filename        poses = {}        for cluster_file in cluster_files:

            image_name = f"{camera_name}/{image_id}.jpg"

                    poses_dir = self.input_dir / 'poses'            try:

            # Create keyframe metadata - each image gets its own sample_id for monocular sequence

            keyframe = {                        with open(cluster_file, 'r') as f:

                "id": str(frame_id),

                "camera_params_id": str(camera_param_id),        pose_files = list(poses_dir.glob('*_pose.json'))                    data = json.load(f)

                "timestamp_microseconds": str(timestamp_us),

                "image_name": image_name,        for pose_file in pose_files:                

                "camera_to_world": cusfm_pose,

                "synced_sample_id": str(sample_id)  # Each image gets unique sample_id            image_id = pose_file.stem.replace('_pose', '')                # Handle different data structures

            }

                        with open(pose_file, 'r') as f:                if isinstance(data, list):

            keyframes_metadata.append(keyframe)

            frame_id += 1                poses[image_id] = json.load(f)                    # List of reconstructions

            sample_id += 1

                                    for reconstruction in data:

        # Sort by timestamp

        keyframes_metadata.sort(key=lambda x: int(x["timestamp_microseconds"]))        logger.info(f"  ✓ Loaded {len(poses)} poses")                        if isinstance(reconstruction, dict):

        

        # Create frames_meta structure        return poses                            self._merge_reconstruction_data(sfm_data, reconstruction)

        frames_meta = {

            "keyframes_metadata": keyframes_metadata,                    elif isinstance(data, dict):

            "initial_pose_type": "EGO_MOTION",  # Use EGO_MOTION for sequential data

            "camera_params_id_to_session_name": {    def load_sequences(self) -> Dict[str, Dict[str, Any]]:                    # Single reconstruction

                str(param_id): "0" for param_id in camera_params_map.keys()

            },        """Load sequence information."""                    if "cameras" in data or "shots" in data or "points" in data:

            "camera_params_id_to_camera_params": {

                str(param_id): params for param_id, params in camera_params_map.items()        sequences = {}                        self._merge_reconstruction_data(sfm_data, data)

            },

            "reference_latlngalt": {        sequences_dir = self.input_dir / 'sequences'                    else:

                "latitude": ref_lat,

                "longitude": ref_lon,                                # Might be a list of individual reconstructions

                "altitude": ref_alt

            }        if not sequences_dir.exists():                        for key, value in data.items():

        }

                    logger.info("  ⚠ No sequences directory found")                            if isinstance(value, dict) and ("cameras" in value or "shots" in value):

        # Save frames_meta.json

        with open(self.frames_meta_file, 'w') as f:            return sequences                                self._merge_reconstruction_data(sfm_data, value)

            json.dump(frames_meta, f, indent=2)

                                

        logger.info(f"✅ Created frames_meta.json with {len(keyframes_metadata)} keyframes from enhanced data")

        logger.info(f"   📄 Output file: {self.frames_meta_file}")        seq_files = list(sequences_dir.glob('*.json'))            except Exception as e:

        logger.info(f"   🎥 Cameras used: {len(camera_params_map)}")

            for seq_file in seq_files:                logger.error(f"Error loading {cluster_file}: {e}")

    def _analyze_input_data(self, camera_intrinsics: Dict, pose_data: Dict):

        """Analyze input data structure for debugging"""            seq_id = seq_file.stem                continue

        logger.info("=== Input Data Analysis ===")

                    with open(seq_file, 'r') as f:        

        # Analyze camera types

        camera_types = {}                sequences[seq_id] = json.load(f)        if not sfm_data["shots"]:

        focal_lengths = set()

                            logger.warning("No shots (camera poses) found in SfM data")

        for image_id, intrinsics in camera_intrinsics.items():

            camera_name = self._get_camera_name_from_intrinsics(intrinsics)        logger.info(f"  ✓ Loaded {len(sequences)} sequences")            return None

            if camera_name not in camera_types:

                camera_types[camera_name] = 0        return sequences        

            

            camera_types[camera_name] += 1            logger.info(f"Loaded {len(sfm_data['cameras'])} cameras, {len(sfm_data['shots'])} shots, {len(sfm_data['points'])} points")

            

            camera_params = intrinsics.get("camera_parameters", [])    def convert_to_cusfm_format(self, cameras: Dict, poses: Dict,         return sfm_data

            if camera_params:

                focal_lengths.add(int(camera_params[0]))                                sequences: Dict, reference: Dict) -> Dict[str, Any]:    

        

        logger.info(f"Camera types found: {dict(camera_types)}")        """    def _merge_reconstruction_data(self, target: Dict, source: Dict):

        logger.info(f"Focal lengths found: {sorted(focal_lengths)}")

                Convert Mapillary data to PyCuSFM format.        """Merge reconstruction data from source into target"""

        # Analyze timestamps

        timestamps = set()                for key in ["cameras", "shots", "points"]:

        for image_id, pose in pose_data.items():

            timestamp_us = self._get_timestamp_from_pose_data(pose)        Args:            if key in source and isinstance(source[key], dict):

            timestamps.add(timestamp_us)

                    cameras: Camera intrinsics data                target[key].update(source[key])

        logger.info(f"Total unique timestamps: {len(timestamps)}")

        logger.info(f"Total images with intrinsics: {len(camera_intrinsics)}")            poses: Pose data    

        logger.info(f"Total images with poses: {len(pose_data)}")

        logger.info("=== End Analysis ===")            sequences: Sequence information    def _load_camera_models(self) -> Dict:

    

    def _find_reference_gps_point(self, pose_data: Dict) -> Tuple[float, float, float]:            reference: GPS reference frame        """Load camera models from Mapillary format"""

        """Find a reference GPS point from pose data (typically the first valid point)"""

                            if not self.camera_models_file.exists():

        for image_id, pose in pose_data.items():

            geometry = pose.get("computed_geometry", pose.get("geometry", {}))        Returns:            logger.warning("camera_models.json not found, using default camera parameters")

            if geometry.get("type") == "Point" and "coordinates" in geometry:

                lon, lat = geometry["coordinates"][:2]            Complete PyCuSFM frames_meta structure            return {}

                altitude = pose.get("computed_altitude", pose.get("altitude", 0.0))

                return lat, lon, altitude        """        

        

        # Fallback to origin        # Find images with both camera and pose data        with open(self.camera_models_file, 'r') as f:

        logger.warning("No GPS coordinates found in pose data, using origin as reference")

        return 0.0, 0.0, 0.0        valid_images = set(cameras.keys()) & set(poses.keys())            return json.load(f)

    

    def _get_or_create_camera_params_from_enhanced_intrinsics(self, camera_name: str,         logger.info(f"  Found {len(valid_images)} images with complete data")    

                                                           intrinsics: Dict, 

                                                           camera_params_map: Dict,             def _load_exif_data(self) -> Dict:

                                                           camera_name_to_id: Dict) -> int:

        """Create camera parameters from enhanced intrinsics data"""        if not valid_images:        """Load all EXIF data from individual JSON files"""

        

        # Check if we already have params for this camera            raise ValueError("No images found with both camera intrinsics and pose data!")        exif_data = {}

        if camera_name in camera_name_to_id:

            return camera_name_to_id[camera_name]                

        

        # Create new camera parameters        # Determine reference GPS point        if not self.exif_dir.exists():

        camera_id = len(camera_params_map)

        camera_name_to_id[camera_name] = camera_id        ref_lat, ref_lon, ref_alt = self.get_reference_point(reference, poses)            logger.warning("exif directory not found")

        

        # Extract dimensions        logger.info(f"  Reference GPS: ({ref_lat:.6f}, {ref_lon:.6f}, {ref_alt:.2f}m)")            return exif_data

        width = intrinsics.get("width", 1920)

        height = intrinsics.get("height", 1080)                

        

        # Extract camera parameters [fx, fy, cx, cy, k1, k2, p1, p2, k3]        # Create camera parameters map        for exif_file in self.exif_dir.glob("*.json"):

        camera_params = intrinsics.get("camera_parameters", [])

        if len(camera_params) >= 2:        camera_params_map = {}            image_id = exif_file.stem

            # Mapillary format: [focal_x, focal_y, c_x, c_y, k1, k2, p1, p2, k3]

            fx, fy = camera_params[0], camera_params[1]        camera_type_to_id = {}            try:

            cx = camera_params[2] if len(camera_params) > 2 else width / 2

            cy = camera_params[3] if len(camera_params) > 3 else height / 2                        with open(exif_file, 'r') as f:

        else:

            # Fallback to default perspective camera        # Create keyframes                    exif_data[image_id] = json.load(f)

            focal_length = intrinsics.get("focal_length", 28.0)  # mm

            fx = fy = width * focal_length / 36.0  # Assuming 35mm equivalent        keyframes = []            except Exception as e:

            cx, cy = width / 2, height / 2

                frame_id = 0                logger.error(f"Error loading EXIF file {exif_file}: {e}")

        # Create 3x4 projection matrix

        projection_matrix = [fx, 0, cx, 0, 0, fy, cy, 0, 0, 0, 1, 0]                

        

        # Extract distortion coefficients        # Get image list with timestamps        return exif_data

        distortion_coeffs = [

            intrinsics.get("k1", 0.0),        image_list = []    

            intrinsics.get("k2", 0.0),

            intrinsics.get("p1", 0.0),        for image_id in valid_images:    def _load_camera_intrinsics(self) -> Dict:

            intrinsics.get("p2", 0.0),

            intrinsics.get("k3", 0.0)            timestamp = self.get_timestamp(poses[image_id], sequences, image_id)        """Load camera intrinsics from enhanced format cameras/ directory"""

        ]

                    image_list.append((timestamp, image_id))        intrinsics_data = {}

        # Create camera params structure

        camera_params = {                

            "sensor_meta_data": {

                "sensor_id": camera_id,        # Sort by timestamp        if not self.cameras_dir.exists():

                "sensor_type": "CAMERA",

                "sensor_name": camera_name,        image_list.sort()            logger.warning("cameras directory not found")

                "frequency": 30,

                "sensor_to_vehicle_transform": {        logger.info(f"  Processing {len(image_list)} images in temporal order...")            return intrinsics_data

                    "axis_angle": {"x": 0, "y": 0, "z": 0, "angle_degrees": 0},

                    "translation": {"x": 0, "y": 0, "z": 0}                

                }

            },        for sample_id, (timestamp, image_id) in enumerate(image_list):        logger.info(f"Loading camera intrinsics from {self.cameras_dir}")

            "calibration_parameters": {

                "image_width": width,            camera_data = cameras[image_id]        for intrinsics_file in self.cameras_dir.glob("*_intrinsics.json"):

                "image_height": height,

                "projection_matrix": {"data": projection_matrix},            pose_data = poses[image_id]            image_id = intrinsics_file.stem.replace("_intrinsics", "")

                "distortion_coefficients": {"data": distortion_coeffs}

            }                        try:

        }

                    # Get or create camera parameters                with open(intrinsics_file, 'r') as f:

        camera_params_map[camera_id] = camera_params

        return camera_id            camera_type = self.get_camera_type(camera_data)                    intrinsics_data[image_id] = json.load(f)

    

    def _convert_mapillary_pose_to_cusfm(self, pose_data: Dict, ref_lat: float, ref_lon: float, ref_alt: float) -> Dict:            if camera_type not in camera_type_to_id:            except Exception as e:

        """Convert Mapillary pose data to PyCuSFM camera_to_world format"""

                        cam_id = len(camera_params_map)                logger.error(f"Error loading intrinsics file {intrinsics_file}: {e}")

        # Extract GPS coordinates

        geometry = pose_data.get("computed_geometry", pose_data.get("geometry", {}))                camera_type_to_id[camera_type] = cam_id        

        if geometry.get("type") == "Point" and "coordinates" in geometry:

            lon, lat = geometry["coordinates"][:2]                camera_params_map[cam_id] = self.create_camera_params(        logger.info(f"Loaded intrinsics for {len(intrinsics_data)} images")

            # Handle altitude - check multiple possible fields

            altitude = pose_data.get("computed_altitude", pose_data.get("altitude", 0.0))                    cam_id, camera_type, camera_data        return intrinsics_data

            

            # Convert GPS to local ENU coordinates                )    

            if ref_lat is not None and ref_lon is not None:

                x, y, z = self._gps_to_local_enu(lat, lon, altitude, ref_lat, ref_lon, ref_alt)                def _load_pose_data(self) -> Dict:

            else:

                # No reference point, use GPS directly            camera_param_id = camera_type_to_id[camera_type]        """Load pose data from enhanced format poses/ directory"""

                x, y, z = self._gps_to_local_enu(lat, lon, altitude)

        else:                    pose_data = {}

            x = y = z = 0.0

                    # Convert pose        

        # Extract rotation

        compass_angle = pose_data.get("computed_compass_angle", pose_data.get("compass_angle", 0.0))            pose = self.convert_pose(pose_data, ref_lat, ref_lon, ref_alt)        if not self.poses_dir.exists():

        computed_rotation = pose_data.get("computed_rotation", [0, 0, 0])

                                logger.warning("poses directory not found")

        # Convert rotation to axis-angle representation

        if computed_rotation and len(computed_rotation) == 3 and any(abs(r) > 0.001 for r in computed_rotation):            # Create keyframe            return pose_data

            # Already in axis-angle format

            axis_angle_rad = np.array(computed_rotation)            keyframe = {        

            angle_rad = np.linalg.norm(axis_angle_rad)

            angle_deg = np.degrees(angle_rad)                "id": str(frame_id),        logger.info(f"Loading pose data from {self.poses_dir}")

            

            if angle_rad > 0.001:  # Avoid division by zero                "camera_params_id": str(camera_param_id),        for pose_file in self.poses_dir.glob("*_pose.json"):

                axis = axis_angle_rad / angle_rad

            else:                "timestamp_microseconds": str(timestamp),            image_id = pose_file.stem.replace("_pose", "")

                axis = np.array([0, 0, 1])

                angle_deg = 0.0                "image_name": f"camera_{camera_param_id}/{image_id}.jpg",            try:

        else:

            # Use compass angle (typically in degrees, pointing direction)                "camera_to_world": pose,                with open(pose_file, 'r') as f:

            axis = np.array([0, 0, 1])  # Rotation around vertical axis

            angle_deg = compass_angle                "synced_sample_id": str(sample_id)                    pose_data[image_id] = json.load(f)

        

        return {            }            except Exception as e:

            "axis_angle": {

                "x": float(axis[0]),                            logger.error(f"Error loading pose file {pose_file}: {e}")

                "y": float(axis[1]),

                "z": float(axis[2]),            keyframes.append(keyframe)        

                "angle_degrees": float(angle_deg)

            },            frame_id += 1        logger.info(f"Loaded poses for {len(pose_data)} images")

            "translation": {

                "x": float(x),                return pose_data

                "y": float(y),

                "z": float(z)        logger.info(f"  ✓ Created {len(keyframes)} keyframes with {len(camera_params_map)} unique cameras")    

            }

        }            def _copy_images_enhanced(self):

    

    def _gps_to_local_enu(self, lat: float, lon: float, alt: float,         # Create frames_meta structure        """Copy images for enhanced format"""

                         ref_lat: float = 0.0, ref_lon: float = 0.0, ref_alt: float = 0.0) -> Tuple[float, float, float]:

        """        frames_meta = {        if not self.images_dir.exists():

        Convert GPS coordinates to local East-North-Up (ENU) coordinates

                    "keyframes_metadata": keyframes,            logger.error("images directory not found")

        Args:

            lat: Latitude in degrees            "initial_pose_type": "EGO_MOTION",            return

            lon: Longitude in degrees  

            alt: Altitude in meters            "camera_params_id_to_session_name": {        

            ref_lat: Reference latitude (defaults to 0)

            ref_lon: Reference longitude (defaults to 0)                str(cam_id): "0" for cam_id in camera_params_map.keys()        # Create camera subdirectories based on camera models found in intrinsics

            ref_alt: Reference altitude (defaults to 0)

                        },        camera_dirs = set()

        Returns:

            Tuple of (East, North, Up) coordinates in meters            "camera_params_id_to_camera_params": {        

        """

        # Earth radius in meters (WGS84)                str(cam_id): params for cam_id, params in camera_params_map.items()        # Load camera intrinsics to determine camera types

        R = 6378137.0

                    },        for intrinsics_file in self.cameras_dir.glob("*_intrinsics.json"):

        # Convert to radians

        lat_rad = math.radians(lat)            "reference_latlngalt": {            try:

        lon_rad = math.radians(lon)

        ref_lat_rad = math.radians(ref_lat)                "latitude": ref_lat,                with open(intrinsics_file, 'r') as f:

        ref_lon_rad = math.radians(ref_lon)

                        "longitude": ref_lon,                    data = json.load(f)

        # Calculate differences

        dlat = lat_rad - ref_lat_rad                "altitude": ref_alt                camera_name = self._get_camera_name_from_intrinsics(data)

        dlon = lon_rad - ref_lon_rad

                    },                camera_dirs.add(camera_name)

        # Convert to ENU

        # East: longitude difference scaled by Earth radius and latitude            "metadata": {            except Exception as e:

        east = R * dlon * math.cos(ref_lat_rad)

                        "source": "Mapillary",                logger.error(f"Error reading {intrinsics_file}: {e}")

        # North: latitude difference scaled by Earth radius

        north = R * dlat                "conversion_date": datetime.now().isoformat(),        

        

        # Up: altitude difference                "total_images": len(keyframes),        # Default camera if none found

        up = alt - ref_alt

                        "unique_cameras": len(camera_params_map)        if not camera_dirs:

        return east, north, up

                }            camera_dirs.add("camera_0")

    def _get_timestamp_from_pose_data(self, pose_data: Dict) -> int:

        """Extract timestamp from pose data and convert to microseconds"""        }        

        # Try to get capture_time (typically in milliseconds)

        capture_time = pose_data.get("capture_time")                # Create camera subdirectories

        if capture_time:

            # Convert milliseconds to microseconds        return frames_meta        for camera_dir in camera_dirs:

            return int(capture_time) * 1000

                        (self.output_images_dir / camera_dir).mkdir(exist_ok=True)

        # Fallback to current time

        return int(datetime.now().timestamp() * 1_000_000)    def get_reference_point(self, reference: Dict, poses: Dict) -> Tuple[float, float, float]:        



        """Get reference GPS point."""        # Copy images

def main():

    parser = argparse.ArgumentParser(        if reference and 'latitude' in reference and 'longitude' in reference:        copied_count = 0

        description='Convert Mapillary data to PyCuSFM format',

        formatter_class=argparse.RawDescriptionHelpFormatter,            return (        for image_file in self.images_dir.glob("*.jpg"):

        epilog="""

Examples:                reference['latitude'],            image_id = image_file.stem

  %(prog)s data/msp_downtown_2 output/pycusfm_data

  %(prog)s --input data/mapillary --output output/cusfm                reference['longitude'],            

        """

    )                reference.get('altitude', 0.0)            # Get camera name for this image

    

    parser.add_argument('input_dir', nargs='?', help='Input Mapillary data directory')            )            camera_name = self._get_camera_name_for_image_enhanced(image_id)

    parser.add_argument('output_dir', nargs='?', help='Output directory for PyCuSFM format')

    parser.add_argument('--input', '-i', dest='input_alt', help='Input directory (alternative)')                    

    parser.add_argument('--output', '-o', dest='output_alt', help='Output directory (alternative)')

    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose logging')        # Use first pose as reference            # Copy image

    

    args = parser.parse_args()        for pose in poses.values():            dest_path = self.output_images_dir / camera_name / f"{image_id}.jpg"

    

    # Get input/output from either positional or named arguments            geom = pose.get('computed_geometry', {})            shutil.copy2(image_file, dest_path)

    input_dir = args.input_dir or args.input_alt

    output_dir = args.output_dir or args.output_alt            if geom.get('type') == 'Point' and 'coordinates' in geom:            copied_count += 1

    

    if not input_dir or not output_dir:                lon, lat = geom['coordinates'][:2]        

        parser.error("Both input and output directories are required")

                    alt = geom['coordinates'][2] if len(geom['coordinates']) > 2 else 0.0        logger.info(f"Copied {copied_count} images to camera subdirectories")

    if args.verbose:

        logging.getLogger().setLevel(logging.DEBUG)                return lat, lon, alt    

    

    try:            def _get_camera_name_from_intrinsics(self, intrinsics_data: Dict) -> str:

        converter = MapillaryToCuSFMConverter(input_dir, output_dir)

        converter.convert()        return 0.0, 0.0, 0.0        """Extract camera name from intrinsics data"""

        return 0

    except Exception as e:            # Try to get camera make/model for naming

        logger.error(f"❌ Conversion failed: {e}", exc_info=args.verbose)

        return 1    def get_camera_type(self, camera_data: Dict) -> str:        make = intrinsics_data.get("make", "")



        """Get unique camera type identifier."""        model = intrinsics_data.get("model", "")

if __name__ == "__main__":

    exit(main())        projection = camera_data.get('projection_type', 'perspective')        


        width = camera_data.get('width', 0)        # Get focal length and principal point to help distinguish cameras

        height = camera_data.get('height', 0)        camera_params = intrinsics_data.get("camera_parameters", [])

        focal = camera_data.get('focal', camera_data.get('focal_x', 0.0))        focal_length = camera_params[0] if len(camera_params) > 0 else 0

                

        # Create unique identifier        # Create a more unique camera identifier

        return f"{projection}_{width}x{height}_f{int(focal * 1000)}"        if make and model:

                base_name = f"{make}_{model}".replace(" ", "_").lower()

    def create_camera_params(self, cam_id: int, camera_type: str, camera_data: Dict) -> Dict:        elif model:

        """Create PyCuSFM camera parameters."""            base_name = f"{model}".replace(" ", "_").lower()

        width = camera_data.get('width', 1920)        else:

        height = camera_data.get('height', 1080)            base_name = "camera"

                

        # Extract focal length        # Add focal length to make it more unique (rounded to avoid floating point precision issues)

        focal_x = camera_data.get('focal_x', camera_data.get('focal', 0.85))        if focal_length > 0:

        focal_y = camera_data.get('focal_y', focal_x)            focal_str = f"_f{int(focal_length)}"

                    return f"{base_name}{focal_str}"

        # Convert normalized to pixels if needed        else:

        if abs(focal_x) <= 1.0:            return f"{base_name}_0"

            focal_x *= max(width, height)    

            focal_y *= max(width, height)    def _get_camera_name_for_image_enhanced(self, image_id: str) -> str:

                """Get camera name for an image in enhanced format"""

        # Principal point        # Try to load intrinsics for this image

        c_x = camera_data.get('c_x', 0.0)        intrinsics_file = self.cameras_dir / f"{image_id}_intrinsics.json"

        c_y = camera_data.get('c_y', 0.0)        if intrinsics_file.exists():

                    try:

        if abs(c_x) <= 1.0:                with open(intrinsics_file, 'r') as f:

            c_x *= width                    data = json.load(f)

            c_y *= height                return self._get_camera_name_from_intrinsics(data)

        else:            except:

            c_x = width / 2                pass

            c_y = height / 2        

                return "camera_0"  # Default fallback

        # Distortion coefficients    

        k1 = camera_data.get('k1', 0.0)    def _copy_images(self):

        k2 = camera_data.get('k2', 0.0)        """Copy images from Mapillary structure to PyCuSFM structure"""

        p1 = camera_data.get('p1', 0.0)        if not self.images_dir.exists():

        p2 = camera_data.get('p2', 0.0)            logger.error("images directory not found")

        k3 = camera_data.get('k3', 0.0)            return

                

        # Create projection matrix (3x4)        # Create camera subdirectories

        projection_matrix = [        camera_dirs = set()

            focal_x, 0.0, c_x, 0.0,        

            0.0, focal_y, c_y, 0.0,        # First pass: identify unique cameras

            0.0, 0.0, 1.0, 0.0        for image_file in self.images_dir.glob("*.jpg"):

        ]            camera_name = self._get_camera_name_for_image(image_file.stem)

                    camera_dirs.add(camera_name)

        return {        

            "sensor_meta_data": {        # Create camera subdirectories

                "sensor_id": cam_id,        for camera_dir in camera_dirs:

                "sensor_type": "CAMERA",            (self.output_images_dir / camera_dir).mkdir(exist_ok=True)

                "sensor_name": f"camera_{cam_id}",        

                "frequency": 30,        # Second pass: copy images

                "sensor_to_vehicle_transform": {        for image_file in self.images_dir.glob("*.jpg"):

                    "axis_angle": {"x": 0, "y": 0, "z": 0, "angle_degrees": 0},            image_id = image_file.stem

                    "translation": {"x": 0, "y": 0, "z": 0}            camera_name = self._get_camera_name_for_image(image_id)

                }            new_filename = f"{image_id}.jpg"

            },            dest_path = self.output_images_dir / camera_name / new_filename

            "calibration_parameters": {            shutil.copy2(image_file, dest_path)

                "image_width": int(width),    

                "image_height": int(height),    def _get_camera_name_for_image(self, image_id: str) -> str:

                "projection_matrix": {"data": projection_matrix},        """Get camera name for an image"""

                "distortion_coefficients": {"data": [k1, k2, p1, p2, k3]}        return "camera_0"  # Simplified for now

            }    

        }    def _create_frames_meta_from_enhanced_data(self, camera_intrinsics: Dict, pose_data: Dict):

            """Create frames_meta.json from enhanced Mapillary data with cameras/ and poses/"""

    def get_timestamp(self, pose_data: Dict, sequences: Dict, image_id: str) -> int:        logger.info("Creating frames_meta.json from enhanced Mapillary data")

        """Get timestamp in microseconds."""        

        # Try pose data first        keyframes_metadata = []

        if 'capture_time' in pose_data:        camera_params_map = {}

            return int(pose_data['capture_time'] * 1000)  # Convert ms to us        camera_name_to_id = {}

                

        # Try sequences        # Process each image that has both intrinsics and pose data

        for seq_data in sequences.values():        common_images = set(camera_intrinsics.keys()) & set(pose_data.keys())

            for img in seq_data.get('images', []):        logger.info(f"Found {len(common_images)} images with both intrinsics and pose data")

                if str(img.get('image_id', '')) == str(image_id):        

                    captured_at = img.get('captured_at', 0)        if not common_images:

                    return int(captured_at * 1_000_000) if captured_at else 0            logger.error("No images found with both camera intrinsics and pose data")

                    return

        return 0        

            # Find reference GPS coordinate (use first valid GPS point)

    def convert_pose(self, pose_data: Dict, ref_lat: float,         ref_lat, ref_lon, ref_alt = self._find_reference_gps_point(pose_data)

                    ref_lon: float, ref_alt: float) -> Dict:        logger.info(f"Using reference GPS point: lat={ref_lat:.6f}, lon={ref_lon:.6f}, alt={ref_alt:.2f}")

        """Convert Mapillary pose to PyCuSFM format."""        

        # Extract GPS coordinates        # Analyze data structure for diagnostics

        geom = pose_data.get('computed_geometry', {})        self._analyze_input_data(camera_intrinsics, pose_data)

        if geom.get('type') == 'Point' and 'coordinates' in geom:        

            lon, lat = geom['coordinates'][:2]        # For Mapillary data, typically each image is captured at a different time

            alt = geom['coordinates'][2] if len(geom['coordinates']) > 2 else 0.0        # So we'll treat each image as a separate sample (no multi-camera synchronization)

        else:        # This is different from stereo camera setups like KITTI

            return self.identity_pose()        

                frame_id = 0

        # Convert to local ENU coordinates        sample_id = 0

        x, y, z = self.gps_to_enu(lat, lon, alt, ref_lat, ref_lon, ref_alt)        

                # Process each image individually (monocular sequence)

        # Extract rotation        for image_id in sorted(common_images):

        rotation = pose_data.get('computed_rotation', [])            intrinsics = camera_intrinsics[image_id]

        compass_angle = pose_data.get('compass_angle', 0.0)            pose = pose_data[image_id]

                    

        if rotation and len(rotation) == 3:            # Get timestamp for this specific image

            # Axis-angle rotation            timestamp_us = self._get_timestamp_from_pose_data(pose)

            angle_rad = np.linalg.norm(rotation)            

            angle_deg = np.degrees(angle_rad)            # Get camera name and create camera parameters

                        camera_name = self._get_camera_name_from_intrinsics(intrinsics)

            if angle_rad > 1e-6:            camera_param_id = self._get_or_create_camera_params_from_enhanced_intrinsics(

                axis = np.array(rotation) / angle_rad                camera_name, intrinsics, camera_params_map, camera_name_to_id

                axis_x, axis_y, axis_z = axis            )

            else:            

                axis_x, axis_y, axis_z = 0, 0, 1            # Convert Mapillary pose to PyCuSFM format using reference point

                angle_deg = compass_angle            cusfm_pose = self._convert_mapillary_pose_to_cusfm(pose, ref_lat, ref_lon, ref_alt)

        else:            

            # Use compass angle            # Create image filename

            axis_x, axis_y, axis_z = 0, 0, 1            image_name = f"{camera_name}/{image_id}.jpg"

            angle_deg = compass_angle            

                    # Create keyframe metadata - each image gets its own sample_id for monocular sequence

        return {            keyframe = {

            "axis_angle": {                "id": str(frame_id),

                "x": float(axis_x),                "camera_params_id": str(camera_param_id),

                "y": float(axis_y),                "timestamp_microseconds": str(timestamp_us),

                "z": float(axis_z),                "image_name": image_name,

                "angle_degrees": float(angle_deg)                # "camera_to_world": cusfm_pose,

            },                "synced_sample_id": str(sample_id)  # Each image gets unique sample_id

            "translation": {            }

                "x": float(x),            

                "y": float(y),            keyframes_metadata.append(keyframe)

                "z": float(z)            frame_id += 1

            }            sample_id += 1

        }        

            # Sort by timestamp

    def identity_pose(self) -> Dict:        keyframes_metadata.sort(key=lambda x: int(x["timestamp_microseconds"]))

        """Return identity pose."""        

        return {        # Create frames_meta structure

            "axis_angle": {"x": 0, "y": 0, "z": 0, "angle_degrees": 0},        frames_meta = {

            "translation": {"x": 0, "y": 0, "z": 0}            "keyframes_metadata": keyframes_metadata,

        }            "initial_pose_type": "EGO_MOTION",  # Use EGO_MOTION for sequential data

                "camera_params_id_to_session_name": {

    def gps_to_enu(self, lat: float, lon: float, alt: float,                str(param_id): "0" for param_id in camera_params_map.keys()

                   ref_lat: float, ref_lon: float, ref_alt: float) -> Tuple[float, float, float]:            },

        """            "camera_params_id_to_camera_params": {

        Convert GPS to East-North-Up coordinates.                str(param_id): params for param_id, params in camera_params_map.items()

                    },

        Args:            "reference_latlngalt": {

            lat, lon, alt: Target GPS coordinates                "latitude": ref_lat,

            ref_lat, ref_lon, ref_alt: Reference GPS coordinates                "longitude": ref_lon,

                            "altitude": ref_alt

        Returns:            }

            (east, north, up) in meters        }

        """        

        # Earth radius        # Save frames_meta.json

        R = 6378137.0        with open(self.frames_meta_file, 'w') as f:

                    json.dump(frames_meta, f, indent=2)

        # Differences in radians        

        dlat = np.radians(lat - ref_lat)        logger.info(f"✅ Created frames_meta.json with {len(keyframes_metadata)} keyframes from enhanced data")

        dlon = np.radians(lon - ref_lon)        logger.info(f"   📄 Output file: {self.frames_meta_file}")

                logger.info(f"   🎥 Cameras used: {len(camera_params_map)}")

        # Convert to meters        logger.info(f"   ⏰ Time range: {min(int(k['timestamp_microseconds']) for k in keyframes_metadata)} to {max(int(k['timestamp_microseconds']) for k in keyframes_metadata)} microseconds")

        north = dlat * R    

        east = dlon * R * np.cos(np.radians(ref_lat))    

        up = alt - ref_alt    def _analyze_input_data(self, camera_intrinsics: Dict, pose_data: Dict):

                """Analyze input data structure for debugging"""

        return east, north, up        logger.info("=== Input Data Analysis ===")

            

    def save_output(self, frames_meta: Dict):        # Analyze camera types

        """Save output files."""        camera_types = {}

        # Save frames_meta.json        focal_lengths = set()

        output_file = self.output_dir / 'frames_meta.json'        

        with open(output_file, 'w') as f:        for image_id, intrinsics in camera_intrinsics.items():

            json.dump(frames_meta, f, indent=2)            camera_name = self._get_camera_name_from_intrinsics(intrinsics)

        logger.info(f"  ✓ Saved frames_meta.json ({output_file.stat().st_size / 1024:.1f} KB)")            if camera_name not in camera_types:

                        camera_types[camera_name] = 0

        # Create image directories            camera_types[camera_name] += 1

        images_output = self.output_dir / 'images'            

        images_output.mkdir(exist_ok=True)            # Track focal lengths

                    camera_params = intrinsics.get("camera_parameters", [])

        # Get unique camera IDs            if len(camera_params) > 0:

        camera_ids = set(kf['camera_params_id'] for kf in frames_meta['keyframes_metadata'])                focal_lengths.add(int(camera_params[0]))

        for cam_id in camera_ids:        

            (images_output / f'camera_{cam_id}').mkdir(exist_ok=True)        logger.info(f"Camera types found: {dict(camera_types)}")

                logger.info(f"Focal lengths found: {sorted(focal_lengths)}")

        # Create symbolic links to images        

        images_input = self.input_dir / 'images'        # Analyze timestamps

        linked_count = 0        timestamps = set()

                for image_id, pose in pose_data.items():

        for keyframe in frames_meta['keyframes_metadata']:            timestamp_us = self._get_timestamp_from_pose_data(pose)

            image_name = keyframe['image_name']            timestamps.add(timestamp_us)

            image_id = Path(image_name).stem        

                    logger.info(f"Total unique timestamps: {len(timestamps)}")

            src = images_input / f'{image_id}.jpg'        logger.info(f"Total images with intrinsics: {len(camera_intrinsics)}")

            dst = images_output / image_name        logger.info(f"Total images with poses: {len(pose_data)}")

                    logger.info("=== End Analysis ===")

            if src.exists() and not dst.exists():    

                try:    

                    dst.symlink_to(src.absolute())    def _find_reference_gps_point(self, pose_data: Dict) -> Tuple[float, float, float]:

                    linked_count += 1        """Find a reference GPS point from pose data (typically the first valid point)"""

                except OSError:        

                    # Fallback to copy if symlinks not supported        for pose in pose_data.values():

                    shutil.copy2(src, dst)            geometry = pose.get("computed_geometry", pose.get("geometry", {}))

                    linked_count += 1            if geometry.get("type") == "Point" and "coordinates" in geometry:

                        lon, lat = geometry["coordinates"][:2]

        logger.info(f"  ✓ Linked {linked_count} images")                altitude = pose.get("computed_altitude", pose.get("altitude", 0.0))

                        return lat, lon, altitude

        # Create README        

        readme = self.output_dir / 'README.md'        # Fallback to origin if no GPS data found

        with open(readme, 'w') as f:        logger.warning("No GPS coordinates found in pose data, using origin as reference")

            f.write(f"""# Mapillary to PyCuSFM Conversion        return 0.0, 0.0, 0.0

    

## Conversion Info    def _get_or_create_camera_params_from_enhanced_intrinsics(self, camera_name: str, 

- **Source**: {self.input_dir}                                                           intrinsics: Dict, 

- **Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}                                                           camera_params_map: Dict, 

- **Total Keyframes**: {len(frames_meta['keyframes_metadata'])}                                                           camera_name_to_id: Dict) -> int:

- **Unique Cameras**: {len(camera_ids)}        """Create camera parameters from enhanced intrinsics data"""

        

## Output Structure        if camera_name in camera_name_to_id:

```            return camera_name_to_id[camera_name]

{self.output_dir.name}/        

├── frames_meta.json     # PyCuSFM metadata        camera_id = len(camera_params_map)

├── images/              # Organized images        camera_name_to_id[camera_name] = camera_id

│   ├── camera_0/        

│   ├── camera_1/        # Extract camera parameters from intrinsics

│   └── ...        width = intrinsics.get("width", 1920)

└── README.md        height = intrinsics.get("height", 1080)

```        

        # Get camera matrix from camera_parameters array

## Reference Frame        camera_params = intrinsics.get("camera_parameters", [])

- **Latitude**: {frames_meta['reference_latlngalt']['latitude']:.6f}°        if len(camera_params) >= 2:

- **Longitude**: {frames_meta['reference_latlngalt']['longitude']:.6f}°            # Mapillary typically stores [fx, fy, cx, cy]

- **Altitude**: {frames_meta['reference_latlngalt']['altitude']:.2f} m            fx, fy = camera_params[0], camera_params[1]

            cx = camera_params[2] if len(camera_params) > 2 else width / 2

## Usage            cy = camera_params[3] if len(camera_params) > 3 else height / 2

Use this data with PyCuSFM for Structure from Motion reconstruction.        else:

""")            # Default values

        logger.info(f"  ✓ Created README.md")            focal_length = intrinsics.get("focal_length", 28.0)  # mm

            fx = fy = width * focal_length / 36.0  # Approximate conversion

            cx, cy = width / 2, height / 2

def main():        

    parser = argparse.ArgumentParser(        # Projection matrix in PyCuSFM format (3x4)

        description='Convert Mapillary data to PyCuSFM format',        projection_matrix = [fx, 0, cx, 0, 0, fy, cy, 0, 0, 0, 1, 0]

        formatter_class=argparse.RawDescriptionHelpFormatter,        

        epilog="""        # Extract distortion coefficients

Examples:        distortion_coeffs = [

  %(prog)s data/mapillary/msp_downtown_2 output/pycusfm_data            intrinsics.get("k1", 0.0),

  %(prog)s --input data/mapillary --output output/cusfm -v            intrinsics.get("k2", 0.0),

        """            intrinsics.get("p1", 0.0),

    )            intrinsics.get("p2", 0.0),

                intrinsics.get("k3", 0.0)

    parser.add_argument('input_dir', nargs='?', help='Input Mapillary data directory')        ]

    parser.add_argument('output_dir', nargs='?', help='Output directory for PyCuSFM format')        

    parser.add_argument('--input', '-i', dest='input_alt', help='Input directory (alternative)')        # Create camera parameters structure

    parser.add_argument('--output', '-o', dest='output_alt', help='Output directory (alternative)')        camera_params = {

    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose logging')            "sensor_meta_data": {

                    "sensor_id": camera_id,

    args = parser.parse_args()                "sensor_type": "CAMERA",

                    "sensor_name": camera_name,

    # Handle input/output arguments                "frequency": 30,  # Default frequency

    input_dir = args.input_dir or args.input_alt                "sensor_to_vehicle_transform": {

    output_dir = args.output_dir or args.output_alt                    "axis_angle": {"x": 0, "y": 0, "z": 0, "angle_degrees": 0},

                        "translation": {"x": 0, "y": 0, "z": 0}

    if not input_dir or not output_dir:                }

        parser.error("Both input and output directories are required")            },

                "calibration_parameters": {

    if args.verbose:                "image_width": width,

        logging.getLogger().setLevel(logging.DEBUG)                "image_height": height,

                    "projection_matrix": {"data": projection_matrix},

    try:                "distortion_coefficients": {"data": distortion_coeffs}

        converter = MapillaryToCuSFMConverter(input_dir, output_dir)            }

        converter.convert()        }

    except Exception as e:        

        logger.error(f"❌ Conversion failed: {e}", exc_info=args.verbose)        camera_params_map[camera_id] = camera_params

        return 1        return camera_id

        

    return 0    def _convert_mapillary_pose_to_cusfm(self, pose_data: Dict, ref_lat: float = None, ref_lon: float = None, ref_alt: float = None) -> Dict:

        """Convert Mapillary pose data to PyCuSFM camera_to_world format"""

        

if __name__ == "__main__":        # Extract GPS coordinates

    exit(main())        geometry = pose_data.get("computed_geometry", pose_data.get("geometry", {}))

        if geometry.get("type") == "Point" and "coordinates" in geometry:
            lon, lat = geometry["coordinates"][:2]
            
            # Get altitude
            altitude = pose_data.get("computed_altitude", pose_data.get("altitude", 0.0))
            
            # Convert GPS to local ENU coordinates using reference point
            if ref_lat is not None and ref_lon is not None:
                x, y, z = self._gps_to_local_enu(lat, lon, altitude, ref_lat, ref_lon)
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
                         ref_lat: Optional[float] = None, ref_lon: Optional[float] = None) -> Tuple[float, float, float]:
        """
        Convert GPS coordinates to local East-North-Up (ENU) coordinates
        
        Args:
            lat: Latitude in degrees
            lon: Longitude in degrees  
            alt: Altitude in meters
            ref_lat: Reference latitude (if None, uses current lat as reference)
            ref_lon: Reference longitude (if None, uses current lon as reference)
            
        Returns:
            Tuple of (East, North, Up) coordinates in meters
        """
        if ref_lat is None:
            ref_lat = lat
        if ref_lon is None:
            ref_lon = lon
            
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
        
        # Up: altitude difference (assuming same reference altitude)
        up = alt
        
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
        for shot_id, shot_data in sfm_data["shots"].items():
            timestamp_us = self._get_timestamp_from_shot(shot_data, exif_data.get(shot_id, {}))
            shots_by_timestamp.append((timestamp_us, shot_id, shot_data))
        
        # Sort by timestamp
        shots_by_timestamp.sort(key=lambda x: x[0])
        
        logger.info(f"Processing {len(shots_by_timestamp)} shots from SfM data")
        
        for timestamp_us, shot_id, shot_data in shots_by_timestamp:
            # Get camera info
            camera_key = shot_data.get("camera", "")
            camera_name = self._get_camera_name_for_shot(camera_key, sfm_data["cameras"])
            
            # Create camera parameters if not exists
            camera_param_id = self._get_or_create_camera_params_from_sfm(
                camera_name, camera_key, sfm_data["cameras"], camera_models, 
                camera_params_map, camera_name_to_id, exif_data.get(shot_id, {})
            )
            
            # Convert SfM pose to PyCuSFM format
            pose = self._convert_sfm_pose_to_cusfm(shot_data)
            
            # Create image filename
            image_name = f"{camera_name}/{shot_id}.jpg"
            
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
        for timestamp_us, image_id, exif in images_by_timestamp:
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
                "image_name": f"{camera_name}/{image_id}.jpg",
                "camera_to_world": pose,
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
            }
        }
        
        # Add reference coordinates
        self._add_reference_coordinates(frames_meta, exif_data)
        
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
                    "x": float(axis[0]) * angle_deg,
                    "y": float(axis[1]) * angle_deg,
                    "z": float(axis[2]) * angle_deg,
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
                
                # Distortion parameters
                k1 = camera_data.get("k1", 0.0)
                k2 = camera_data.get("k2", 0.0)
                
                projection_matrix = [
                    focal_pixels, 0, width / 2, 0,
                    0, focal_pixels, height / 2, 0,
                    0, 0, 1, 0
                ]
                distortion_coeffs = [k1, k2, 0, 0, 0]
                
            elif projection_type in ["equirectangular", "spherical"]:
                # For spherical/equirectangular cameras
                projection_matrix = [
                    width / (2 * np.pi), 0, width / 2, 0,
                    0, height / np.pi, height / 2, 0,
                    0, 0, 1, 0
                ]
                distortion_coeffs = [0, 0, 0, 0, 0]
            else:
                # Default perspective projection
                focal_pixels = 0.85 * max(width, height)
                projection_matrix = [
                    focal_pixels, 0, width / 2, 0,
                    0, focal_pixels, height / 2, 0,
                    0, 0, 1, 0
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
                    "projection_matrix": {"data": projection_matrix},
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
            
            # Distortion coefficients
            k1 = camera_model.get("k1", 0.0)
            k2 = camera_model.get("k2", 0.0)
            
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
                    "projection_matrix": {
                        "data": [
                            focal_pixels, 0, width / 2, 0,
                            0, focal_pixels, height / 2, 0,
                            0, 0, 1, 0
                        ]
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
        
        # Simple local coordinate conversion
        x = (lon - lon) * 111000 * np.cos(np.radians(lat))  # Should use reference point
        y = (lat - lat) * 111000  # Should use reference point
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
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert Mapillary data to PyCuSFM format")
    parser.add_argument("--mapillary-dir", required=True, type=Path,
                       help="Path to Mapillary data directory")
    parser.add_argument("--output-dir", required=True, type=Path,
                       help="Output directory for PyCuSFM format")
    parser.add_argument("--disable-rolling-shutter", action="store_true",
                       help="Disable rolling shutter correction (recommended for Mapillary data)")
    
    args = parser.parse_args()
    
    converter = MapillaryToCuSFMConverter(args.mapillary_dir, args.output_dir)
    converter.convert()
    
    if args.disable_rolling_shutter:
        print("\n⚠️  Note: To disable rolling shutter correction in PyCuSFM, run with:")
        print("  --use_rolling_shutter_correction=False --do_rolling_shutter_correction=False")


if __name__ == "__main__":
    main()
