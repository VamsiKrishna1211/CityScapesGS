#!/usr/bin/env python3
"""
Mapillary Data Downloader - A modular downloader for Mapillary data
Supports downloading street view images, aerial view, point clouds, detections, and map features
Follows OpenSfM data structure for seamless integration with 3D reconstruction pipelines
"""

import os
import sys
import json
import time
import base64
import argparse
import requests
import zlib
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import logging
from datetime import datetime
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
import cv2
from PIL import Image
import yaml
import threading

# Progress bar imports
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    logging.warning("tqdm not installed. Progress bars will be disabled. Install with: pip install tqdm")

# Optional import for detection geometry decoding
try:
    import mapbox_vector_tile
    HAS_MAPBOX_VECTOR_TILE = True
except ImportError:
    HAS_MAPBOX_VECTOR_TILE = False
    logging.warning("mapbox_vector_tile not installed. Detection geometry decoding will be limited.")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ProgressTracker:
    """Progress tracking with nested progress bars and download speed calculation"""
    
    def __init__(self, use_progress_bars: bool = True):
        self.use_progress_bars = use_progress_bars and HAS_TQDM
        self.parent_pbar = None
        self.child_pbar = None
        self.download_lock = threading.Lock()
        self.total_bytes = 0
        self.start_time = time.time()
        
    def create_parent_progress(self, total_chunks: int, description: str = "Processing chunks"):
        """Create parent progress bar for chunks"""
        if self.use_progress_bars and total_chunks > 1:
            self.parent_pbar = tqdm(
                total=total_chunks,
                desc=description,
                position=0,
                leave=True,
                unit="chunk",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} chunks [{elapsed}<{remaining}]"
            )
        
    def create_child_progress(self, total_items: int, description: str = "Downloading"):
        """Create child progress bar for individual items"""
        if self.use_progress_bars:
            position = 1 if self.parent_pbar else 0
            self.child_pbar = tqdm(
                total=total_items,
                desc=description,
                position=position,
                leave=False,
                unit="item",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
            )
    
    def create_download_progress(self, total_size: int, description: str = "Downloading"):
        """Create progress bar for download with size tracking"""
        if self.use_progress_bars:
            position = 2 if self.parent_pbar else (1 if self.child_pbar else 0)
            return tqdm(
                total=total_size,
                desc=description,
                position=position,
                leave=False,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
            )
        return None
    
    def update_parent(self, amount: int = 1):
        """Update parent progress bar"""
        if self.parent_pbar:
            self.parent_pbar.update(amount)
    
    def update_child(self, amount: int = 1):
        """Update child progress bar"""
        if self.child_pbar:
            self.child_pbar.update(amount)
    
    def add_download_bytes(self, bytes_downloaded: int):
        """Add downloaded bytes for speed calculation"""
        with self.download_lock:
            self.total_bytes += bytes_downloaded
    
    def get_download_speed(self) -> float:
        """Get current download speed in MB/s"""
        with self.download_lock:
            elapsed = time.time() - self.start_time
            if elapsed > 0:
                return (self.total_bytes / (1024 * 1024)) / elapsed
            return 0.0
    
    def close_child(self):
        """Close child progress bar"""
        if self.child_pbar:
            self.child_pbar.close()
            self.child_pbar = None
    
    def close_parent(self):
        """Close parent progress bar"""
        if self.parent_pbar:
            self.parent_pbar.close()
            self.parent_pbar = None
    
    def close_all(self):
        """Close all progress bars"""
        self.close_child()
        self.close_parent()
    
    def set_child_postfix(self, **kwargs):
        """Set postfix for child progress bar"""
        if self.child_pbar:
            self.child_pbar.set_postfix(**kwargs)
    
    def set_parent_postfix(self, **kwargs):
        """Set postfix for parent progress bar"""
        if self.parent_pbar:
            self.parent_pbar.set_postfix(**kwargs)


class BoundingBox:
    """Helper class for bounding box operations"""
    
    def __init__(self, left: float, bottom: float, right: float, top: float):
        self.left = left
        self.bottom = bottom
        self.right = right
        self.top = top
    
    def to_tuple(self) -> Tuple[float, float, float, float]:
        return (self.left, self.bottom, self.right, self.top)
    
    def to_string(self) -> str:
        return f"{self.left},{self.bottom},{self.right},{self.top}"
    
    def width(self) -> float:
        return self.right - self.left
    
    def height(self) -> float:
        return self.top - self.bottom
    
    def area(self) -> float:
        return self.width() * self.height()
    
    def is_valid(self) -> bool:
        return (self.left < self.right and 
                self.bottom < self.top and
                -180 <= self.left <= 180 and
                -180 <= self.right <= 180 and
                -90 <= self.bottom <= 90 and
                -90 <= self.top <= 90)
    
    def split_into_chunks(self, max_chunk_size: float = 0.01, overlap: float = 0.001) -> List['BoundingBox']:
        """Split large bounding box into smaller chunks for API compliance"""
        chunks = []
        
        width = self.width()
        height = self.height()
        
        # If the bbox is already small enough, return it as is
        if width <= max_chunk_size and height <= max_chunk_size:
            return [self]
        
        # Calculate number of chunks needed in each direction
        x_chunks = max(1, int(math.ceil(width / max_chunk_size)))
        y_chunks = max(1, int(math.ceil(height / max_chunk_size)))
        
        # Calculate actual chunk sizes
        x_chunk_size = width / x_chunks
        y_chunk_size = height / y_chunks
        
        for i in range(x_chunks):
            for j in range(y_chunks):
                # Calculate chunk boundaries
                chunk_left = self.left + i * x_chunk_size
                chunk_right = min(self.right, self.left + (i + 1) * x_chunk_size)
                chunk_bottom = self.bottom + j * y_chunk_size
                chunk_top = min(self.top, self.bottom + (j + 1) * y_chunk_size)
                
                # Add overlap to avoid gaps (except at boundaries)
                if i > 0:
                    chunk_left -= overlap
                if i < x_chunks - 1:
                    chunk_right += overlap
                if j > 0:
                    chunk_bottom -= overlap
                if j < y_chunks - 1:
                    chunk_top += overlap
                
                # Ensure we don't exceed original boundaries
                chunk_left = max(self.left, chunk_left)
                chunk_right = min(self.right, chunk_right)
                chunk_bottom = max(self.bottom, chunk_bottom)
                chunk_top = min(self.top, chunk_top)
                
                chunk = BoundingBox(chunk_left, chunk_bottom, chunk_right, chunk_top)
                if chunk.is_valid():
                    chunks.append(chunk)
        
        return chunks


@dataclass
class MapillaryConfig:
    """Configuration for Mapillary downloader"""
    access_token: str
    bbox: Optional[Tuple[float, float, float, float]] = None  # left, bottom, right, top
    creator_username: Optional[str] = None
    organization_id: Optional[int] = None
    start_captured_at: Optional[str] = None
    end_captured_at: Optional[str] = None
    limit: int = 2000
    max_workers: int = 4
    retry_count: int = 3
    delay_between_requests: float = 0.1
    chunk_size: float = 0.01  # Maximum chunk size in degrees
    overlap: float = 0.001   # Overlap between chunks to avoid gaps

class MapillaryAPI:
    """Mapillary API client for downloading various data types"""
    
    ENTITY_BASE_URL = "https://graph.mapillary.com"
    TILES_BASE_URL = "https://tiles.mapillary.com"
    
    def __init__(self, config: MapillaryConfig):
        self.config = config
        self.session = requests.Session()
        self.session.headers.update({
            'Authorization': f'OAuth {config.access_token}',
            'User-Agent': 'MapillaryDownloader/1.0'
        })
    
    def _make_request(self, url: str, params: Optional[Dict] = None) -> requests.Response:
        """Make HTTP request with retry logic"""
        for attempt in range(self.config.retry_count):
            try:
                time.sleep(self.config.delay_between_requests)
                response = self.session.get(url, params=params, timeout=30)
                response.raise_for_status()
                return response
            except requests.exceptions.RequestException as e:
                logger.warning(f"Request failed (attempt {attempt + 1}/{self.config.retry_count}): {e}")
                if attempt == self.config.retry_count - 1:
                    raise
                time.sleep(2 ** attempt)  # Exponential backoff
    
    def search_images(self, fields: List[str], additional_params: Optional[Dict] = None) -> List[Dict]:
        """Search for images with specified fields"""
        url = f"{self.ENTITY_BASE_URL}/images"
        params = {
            'access_token': self.config.access_token,
            'fields': ','.join(fields),
            'limit': self.config.limit
        }
        
        # Add bounding box if specified
        if self.config.bbox:
            params['bbox'] = ','.join(map(str, self.config.bbox))
        
        # Add other filters
        if self.config.creator_username:
            params['creator_username'] = self.config.creator_username
        if self.config.organization_id:
            params['organization_id'] = self.config.organization_id
        if self.config.start_captured_at:
            params['start_captured_at'] = self.config.start_captured_at
        if self.config.end_captured_at:
            params['end_captured_at'] = self.config.end_captured_at
        
        if additional_params:
            params.update(additional_params)
        
        all_images = []
        
        while True:
            response = self._make_request(url, params)
            data = response.json()
            
            images = data.get('data', [])
            all_images.extend(images)
            
            logger.info(f"Retrieved {len(images)} images, total: {len(all_images)}")
            
            # Check for pagination
            paging = data.get('paging', {})
            next_url = paging.get('next')
            if not next_url or len(images) == 0:
                break
            
            # Extract cursor from next URL
            if 'after=' in next_url:
                cursor = next_url.split('after=')[1].split('&')[0]
                params['after'] = cursor
            else:
                break
        
        return all_images
    
    def get_image_details(self, image_id: str, fields: List[str]) -> Dict:
        """Get detailed information about a specific image"""
        url = f"{self.ENTITY_BASE_URL}/{image_id}"
        params = {
            'access_token': self.config.access_token,
            'fields': ','.join(fields)
        }
        
        response = self._make_request(url, params)
        return response.json()
    
    def search_map_features(self, fields: List[str], additional_params: Optional[Dict] = None) -> List[Dict]:
        """Search for map features"""
        url = f"{self.ENTITY_BASE_URL}/map_features"
        params = {
            'access_token': self.config.access_token,
            'fields': ','.join(fields),
            'limit': self.config.limit
        }
        
        if self.config.bbox:
            params['bbox'] = ','.join(map(str, self.config.bbox))
        
        if additional_params:
            params.update(additional_params)
        
        response = self._make_request(url, params)
        data = response.json()
        return data.get('data', [])
    
    def get_detections(self, image_id: str, fields: List[str]) -> List[Dict]:
        """Get detections for a specific image"""
        url = f"{self.ENTITY_BASE_URL}/{image_id}/detections"
        params = {
            'access_token': self.config.access_token,
            'fields': ','.join(fields)
        }
        
        response = self._make_request(url, params)
        data = response.json()
        return data.get('data', [])
    
    def download_sfm_cluster(self, sfm_cluster_url: str) -> List[Dict]:
        """Download and decode SfM cluster data (point cloud)"""
        try:
            response = self._make_request(sfm_cluster_url)
            
            # Decompress the data
            decompressed_data = zlib.decompress(response.content)
            
            # Parse JSON objects (one per line)
            points = []
            for line in decompressed_data.decode('utf-8').strip().split('\n'):
                if line.strip():
                    points.append(json.loads(line))
            
            return points
        except Exception as e:
            logger.error(f"Failed to download SfM cluster: {e}")
            return []
    
    def download_image(self, image_url: str, output_path: Path, progress_tracker: Optional[ProgressTracker] = None) -> bool:
        """Download an image from URL with progress tracking"""
        try:
            # Get file size first
            head_response = self.session.head(image_url, timeout=10)
            total_size = int(head_response.headers.get('content-length', 0))
            
            response = self._make_request(image_url)
            
            # Create download progress bar if we have size info
            download_pbar = None
            if progress_tracker and total_size > 0:
                download_pbar = progress_tracker.create_download_progress(
                    total_size, 
                    f"Downloading {output_path.name}"
                )
            
            downloaded = 0
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        
                        if progress_tracker:
                            progress_tracker.add_download_bytes(len(chunk))
                        
                        if download_pbar:
                            download_pbar.update(len(chunk))
            
            if download_pbar:
                download_pbar.close()
            
            return True
        except Exception as e:
            logger.error(f"Failed to download image {image_url}: {e}")
            return False

class OpenSfMStructure:
    """Manages OpenSfM-compatible data structure with comprehensive camera metadata"""
    
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.images_dir = self.output_dir / "images"
        self.exif_dir = self.output_dir / "exif"
        self.metadata_dir = self.output_dir / "metadata"
        self.pointclouds_dir = self.output_dir / "pointclouds"
        self.detections_dir = self.output_dir / "detections"
        self.features_dir = self.output_dir / "map_features"
        self.cameras_dir = self.output_dir / "cameras"
        self.poses_dir = self.output_dir / "poses"
        self.sequences_dir = self.output_dir / "sequences"
        
        # Create directories
        for dir_path in [self.images_dir, self.exif_dir, self.metadata_dir, 
                        self.pointclouds_dir, self.detections_dir, self.features_dir,
                        self.cameras_dir, self.poses_dir, self.sequences_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        self.camera_models = {}
        self.image_metadata = {}
        self.camera_intrinsics = {}
        self.camera_poses = {}
        self.sequence_data = {}
        self.gcp_data = {"points": []}
    
    def add_image_metadata(self, image_data: Dict, image_filename: str):
        """Add comprehensive image metadata in OpenSfM format with detailed camera info"""
        # Extract all available metadata
        metadata = {
            "width": image_data.get("width"),
            "height": image_data.get("height"),
            "capture_time": image_data.get("captured_at", 0) / 1000.0 if image_data.get("captured_at") else 0,
            "gps": self._extract_gps_from_geometry(image_data.get("computed_geometry") or image_data.get("geometry")),
            "compass_angle": image_data.get("computed_compass_angle") or image_data.get("compass_angle", 0),
            "camera": self._get_camera_id(image_data),
            "orientation": image_data.get("exif_orientation", 1),
            "sequence_id": image_data.get("sequence"),
            "creator": image_data.get("creator"),
            "is_panoramic": image_data.get("is_pano", False),
            "make": image_data.get("make"),
            "model": image_data.get("model"),
            "camera_type": image_data.get("camera_type"),
            "atomic_scale": image_data.get("atomic_scale"),
            "merge_cc": image_data.get("merge_cc"),
            "computed_rotation": image_data.get("computed_rotation"),
            "altitude": image_data.get("altitude"),
            "computed_altitude": image_data.get("computed_altitude"),
        }
        
        # Store comprehensive EXIF data
        exif_data = {
            **metadata,
            "raw_mapillary_data": {
                "camera_parameters": image_data.get("camera_parameters", []),
                "thumb_urls": {
                    "thumb_256_url": image_data.get("thumb_256_url"),
                    "thumb_1024_url": image_data.get("thumb_1024_url"),
                    "thumb_2048_url": image_data.get("thumb_2048_url"),
                    "thumb_original_url": image_data.get("thumb_original_url"),
                },
                "sfm_cluster": image_data.get("sfm_cluster"),
                "mesh": image_data.get("mesh"),
                "original_geometry": image_data.get("geometry"),
                "computed_geometry": image_data.get("computed_geometry"),
            }
        }
        
        exif_path = self.exif_dir / f"{Path(image_filename).stem}.json"
        with open(exif_path, 'w') as f:
            json.dump(exif_data, f, indent=2)
        
        self.image_metadata[image_filename] = metadata
        
        # Add camera model and intrinsics if not exists
        camera_id = metadata["camera"]
        if camera_id not in self.camera_models:
            self._add_camera_model(image_data, camera_id)
        
        # Store camera intrinsics and pose data
        self._store_camera_intrinsics(image_data, image_filename)
        self._store_camera_pose(image_data, image_filename)
        
        # Store sequence information
        self._store_sequence_data(image_data)
    
    def _extract_gps_from_geometry(self, geometry: Optional[Dict]) -> Optional[Dict]:
        """Extract GPS coordinates from GeoJSON geometry"""
        if not geometry or geometry.get("type") != "Point":
            return None
        
        coordinates = geometry.get("coordinates", [])
        if len(coordinates) >= 2:
            return {
                "latitude": coordinates[1],
                "longitude": coordinates[0],
                "altitude": coordinates[2] if len(coordinates) > 2 else 0.0,
                "dop": 5.0  # Default dilution of precision
            }
        return None
    
    def _get_camera_id(self, image_data: Dict) -> str:
        """Generate camera ID from image data"""
        make = image_data.get("make", "unknown")
        model = image_data.get("model", "unknown")
        width = image_data.get("width", 0)
        height = image_data.get("height", 0)
        return f"{make}_{model}_{width}x{height}"
    
    def _add_camera_model(self, image_data: Dict, camera_id: str):
        """Add comprehensive camera model to camera_models.json"""
        camera_params = image_data.get("camera_parameters", [])
        
        # Determine projection type
        projection_type = "perspective"  # Default
        if image_data.get("is_pano"):
            projection_type = "equirectangular"
        elif image_data.get("camera_type") == "fisheye":
            projection_type = "fisheye"
        elif image_data.get("camera_type"):
            projection_type = image_data.get("camera_type")
        
        camera_model = {
            "projection_type": projection_type,
            "width": image_data.get("width", 1920),
            "height": image_data.get("height", 1080),
            "make": image_data.get("make", "unknown"),
            "model": image_data.get("model", "unknown"),
            "camera_type": image_data.get("camera_type", "perspective"),
        }
        
        # Handle camera parameters based on projection type
        if camera_params and len(camera_params) >= 1:
            camera_model["focal"] = camera_params[0]
            if len(camera_params) >= 2:
                camera_model["k1"] = camera_params[1] if len(camera_params) > 1 else 0.0
            if len(camera_params) >= 3:
                camera_model["k2"] = camera_params[2] if len(camera_params) > 2 else 0.0
            if len(camera_params) >= 4:
                camera_model["p1"] = camera_params[3] if len(camera_params) > 3 else 0.0
            if len(camera_params) >= 5:
                camera_model["p2"] = camera_params[4] if len(camera_params) > 4 else 0.0
            if len(camera_params) >= 6:
                camera_model["k3"] = camera_params[5] if len(camera_params) > 5 else 0.0
        else:
            # Use default focal length
            camera_model["focal"] = 0.85
            camera_model["k1"] = 0.0
            camera_model["k2"] = 0.0
            camera_model["p1"] = 0.0
            camera_model["p2"] = 0.0
            camera_model["k3"] = 0.0
        
        # Store the complete parameter list for reference
        camera_model["camera_parameters_raw"] = camera_params
        camera_model["parameter_count"] = len(camera_params)
        
        self.camera_models[camera_id] = camera_model
    
    def _store_camera_intrinsics(self, image_data: Dict, image_filename: str):
        """Store detailed camera intrinsic parameters"""
        camera_id = self._get_camera_id(image_data)
        camera_params = image_data.get("camera_parameters", [])
        
        intrinsics = {
            "camera_id": camera_id,
            "image_id": image_data.get("id"),
            "width": image_data.get("width"),
            "height": image_data.get("height"),
            "camera_parameters": camera_params,
            "camera_type": image_data.get("camera_type"),
            "projection_type": "perspective",
            "make": image_data.get("make"),
            "model": image_data.get("model"),
            "is_panoramic": image_data.get("is_pano", False),
        }
        
        # Determine projection type
        if image_data.get("is_pano"):
            intrinsics["projection_type"] = "equirectangular"
        elif image_data.get("camera_type") == "fisheye":
            intrinsics["projection_type"] = "fisheye"
        elif image_data.get("camera_type"):
            intrinsics["projection_type"] = image_data.get("camera_type")
        
        # Parse camera parameters into standard format
        if camera_params:
            if len(camera_params) >= 1:
                intrinsics["focal_length"] = camera_params[0]
                # Convert to focal length in pixels if normalized
                if intrinsics["focal_length"] <= 1.0:
                    # Assume normalized, convert to pixels
                    intrinsics["focal_length_pixels"] = intrinsics["focal_length"] * max(
                        image_data.get("width", 1920), image_data.get("height", 1080)
                    )
                else:
                    intrinsics["focal_length_pixels"] = intrinsics["focal_length"]
            
            # Distortion coefficients
            intrinsics["distortion_coefficients"] = {
                "k1": camera_params[1] if len(camera_params) > 1 else 0.0,
                "k2": camera_params[2] if len(camera_params) > 2 else 0.0,
                "p1": camera_params[3] if len(camera_params) > 3 else 0.0,
                "p2": camera_params[4] if len(camera_params) > 4 else 0.0,
                "k3": camera_params[5] if len(camera_params) > 5 else 0.0,
            }
        
        # Save individual camera intrinsics file
        intrinsics_path = self.cameras_dir / f"{Path(image_filename).stem}_intrinsics.json"
        with open(intrinsics_path, 'w') as f:
            json.dump(intrinsics, f, indent=2)
        
        # Store in memory for later aggregation
        self.camera_intrinsics[image_filename] = intrinsics
    
    def _store_camera_pose(self, image_data: Dict, image_filename: str):
        """Store camera pose information (extrinsic parameters)"""
        pose_data = {
            "image_id": image_data.get("id"),
            "image_filename": image_filename,
            "capture_time": image_data.get("captured_at"),
            "compass_angle": image_data.get("compass_angle"),
            "computed_compass_angle": image_data.get("computed_compass_angle"),
            "altitude": image_data.get("altitude"),
            "computed_altitude": image_data.get("computed_altitude"),
            "atomic_scale": image_data.get("atomic_scale"),
            "computed_rotation": image_data.get("computed_rotation"),
            "exif_orientation": image_data.get("exif_orientation"),
            "geometry": image_data.get("geometry"),
            "computed_geometry": image_data.get("computed_geometry"),
            "merge_cc": image_data.get("merge_cc"),  # Connected component ID for SfM
        }
        
        # Extract GPS coordinates in various formats
        gps_original = self._extract_gps_from_geometry(image_data.get("geometry"))
        gps_computed = self._extract_gps_from_geometry(image_data.get("computed_geometry"))
        
        pose_data["gps_original"] = gps_original
        pose_data["gps_computed"] = gps_computed
        
        # Calculate pose accuracy metrics if both original and computed exist
        if gps_original and gps_computed:
            pose_data["gps_drift"] = {
                "lat_diff": gps_computed["latitude"] - gps_original["latitude"],
                "lon_diff": gps_computed["longitude"] - gps_original["longitude"],
                "alt_diff": gps_computed.get("altitude", 0) - gps_original.get("altitude", 0),
            }
        
        # Save individual pose file
        pose_path = self.poses_dir / f"{Path(image_filename).stem}_pose.json"
        with open(pose_path, 'w') as f:
            json.dump(pose_data, f, indent=2)
        
        # Store in memory for later aggregation
        self.camera_poses[image_filename] = pose_data
    
    def _store_sequence_data(self, image_data: Dict):
        """Store sequence information for trajectory reconstruction"""
        sequence_id = image_data.get("sequence")
        if not sequence_id:
            return
        
        if sequence_id not in self.sequence_data:
            self.sequence_data[sequence_id] = {
                "sequence_id": sequence_id,
                "images": [],
                "creator": image_data.get("creator"),
                "first_captured_at": None,
                "last_captured_at": None,
                "total_images": 0,
                "is_panoramic_sequence": image_data.get("is_pano", False),
            }
        
        # Add image to sequence
        image_info = {
            "image_id": image_data.get("id"),
            "captured_at": image_data.get("captured_at"),
            "geometry": image_data.get("computed_geometry") or image_data.get("geometry"),
            "compass_angle": image_data.get("computed_compass_angle") or image_data.get("compass_angle"),
        }
        
        self.sequence_data[sequence_id]["images"].append(image_info)
        self.sequence_data[sequence_id]["total_images"] += 1
        
        # Update sequence timing
        captured_at = image_data.get("captured_at")
        if captured_at:
            if (self.sequence_data[sequence_id]["first_captured_at"] is None or 
                captured_at < self.sequence_data[sequence_id]["first_captured_at"]):
                self.sequence_data[sequence_id]["first_captured_at"] = captured_at
            
            if (self.sequence_data[sequence_id]["last_captured_at"] is None or 
                captured_at > self.sequence_data[sequence_id]["last_captured_at"]):
                self.sequence_data[sequence_id]["last_captured_at"] = captured_at
    
    def save_point_cloud(self, points: List[Dict], filename: str):
        """Save point cloud data in PLY format"""
        if not points:
            return
        
        output_path = self.pointclouds_dir / filename
        
        # Convert to numpy arrays
        coordinates = []
        colors = []
        
        for point in points:
            if "coordinates" in point and "color" in point:
                coordinates.append(point["coordinates"])
                colors.append(point["color"])
        
        if coordinates:
            coordinates = np.array(coordinates)
            colors = np.array(colors)
            
            # Save as PLY
            self._save_ply(output_path, coordinates, colors)
    
    def _save_ply(self, filepath: Path, points: np.ndarray, colors: Optional[np.ndarray] = None):
        """Save points as PLY file"""
        with open(filepath, 'w') as f:
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {len(points)}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            
            if colors is not None:
                f.write("property uchar red\n")
                f.write("property uchar green\n")
                f.write("property uchar blue\n")
            
            f.write("end_header\n")
            
            for i, point in enumerate(points):
                if colors is not None:
                    f.write(f"{point[0]} {point[1]} {point[2]} {int(colors[i][0])} {int(colors[i][1])} {int(colors[i][2])}\n")
                else:
                    f.write(f"{point[0]} {point[1]} {point[2]}\n")
    
    def save_detections(self, image_id: str, detections: List[Dict]):
        """Save detection data"""
        if not detections:
            return
        
        output_path = self.detections_dir / f"{image_id}_detections.json"
        
        processed_detections = []
        for detection in detections:
            processed_detection = {
                "id": detection.get("id"),
                "value": detection.get("value"),
                "geometry": detection.get("geometry"),
                "decoded_geometry": self._decode_detection_geometry(detection.get("geometry"))
            }
            processed_detections.append(processed_detection)
        
        with open(output_path, 'w') as f:
            json.dump(processed_detections, f, indent=2)
    
    def _decode_detection_geometry(self, geometry_b64: Optional[str]) -> Optional[Dict]:
        """Decode base64 encoded detection geometry"""
        if not geometry_b64:
            return None
        
        try:
            decoded_data = base64.decodebytes(geometry_b64.encode('utf-8'))
            
            if HAS_MAPBOX_VECTOR_TILE:
                detection_geometry = mapbox_vector_tile.decode(decoded_data)
                return detection_geometry
            else:
                # Return basic info without full decoding
                return {"decoded": True, "length": len(decoded_data), "note": "Install mapbox-vector-tile for full geometry decoding"}
        except Exception as e:
            logger.warning(f"Failed to decode detection geometry: {e}")
            return None
    
    def save_map_features(self, features: List[Dict]):
        """Save map features data"""
        if not features:
            return
        
        output_path = self.features_dir / "map_features.json"
        with open(output_path, 'w') as f:
            json.dump(features, f, indent=2)
        
        # Also save as GeoJSON for visualization
        geojson = {
            "type": "FeatureCollection",
            "features": []
        }
        
        for feature in features:
            geojson_feature = {
                "type": "Feature",
                "properties": {
                    "id": feature.get("id"),
                    "object_value": feature.get("object_value"),
                    "object_type": feature.get("object_type"),
                    "first_seen_at": feature.get("first_seen_at"),
                    "last_seen_at": feature.get("last_seen_at")
                },
                "geometry": feature.get("geometry")
            }
            geojson["features"].append(geojson_feature)
        
        geojson_path = self.features_dir / "map_features.geojson"
        with open(geojson_path, 'w') as f:
            json.dump(geojson, f, indent=2)
    
    def save_metadata_files(self):
        """Save comprehensive metadata files for OpenSfM and 3D reconstruction"""
        # Save camera models
        camera_models_path = self.output_dir / "camera_models.json"
        with open(camera_models_path, 'w') as f:
            json.dump(self.camera_models, f, indent=2)
        
        # Save aggregated camera intrinsics
        all_intrinsics_path = self.cameras_dir / "all_camera_intrinsics.json"
        with open(all_intrinsics_path, 'w') as f:
            json.dump(self.camera_intrinsics, f, indent=2)
        
        # Save aggregated camera poses
        all_poses_path = self.poses_dir / "all_camera_poses.json"
        with open(all_poses_path, 'w') as f:
            json.dump(self.camera_poses, f, indent=2)
        
        # Save sequence data with trajectory information
        for seq_id, seq_data in self.sequence_data.items():
            # Sort images by capture time for proper trajectory
            seq_data["images"].sort(key=lambda x: x.get("captured_at", 0))
            
            sequence_path = self.sequences_dir / f"sequence_{seq_id}.json"
            with open(sequence_path, 'w') as f:
                json.dump(seq_data, f, indent=2)
        
        # Save all sequences summary
        sequences_summary_path = self.sequences_dir / "sequences_summary.json"
        with open(sequences_summary_path, 'w') as f:
            json.dump(self.sequence_data, f, indent=2)
        
        # Save comprehensive camera calibration summary
        calibration_summary = {
            "total_cameras": len(self.camera_models),
            "camera_models": list(self.camera_models.keys()),
            "projection_types": list(set(model.get("projection_type", "perspective") 
                                       for model in self.camera_models.values())),
            "manufacturers": list(set(model.get("make", "unknown") 
                                    for model in self.camera_models.values())),
            "camera_count_by_type": {},
            "total_images": len(self.image_metadata),
            "total_sequences": len(self.sequence_data),
        }
        
        # Count cameras by type
        for model in self.camera_models.values():
            proj_type = model.get("projection_type", "perspective")
            calibration_summary["camera_count_by_type"][proj_type] = \
                calibration_summary["camera_count_by_type"].get(proj_type, 0) + 1
        
        calibration_path = self.cameras_dir / "calibration_summary.json"
        with open(calibration_path, 'w') as f:
            json.dump(calibration_summary, f, indent=2)
        
        # Save config.yaml with OpenSfM defaults optimized for Mapillary data
        config_data = {
            "feature_type": "sift",
            "feature_root": True,
            "feature_min_frames": 4000,
            "feature_process_size": 2048,
            "feature_use_adaptive_suppression": False,
            "matching_gps_neighbors": 8,
            "matching_time_neighbors": 1,
            "matching_order_neighbors": 0,
            "matching_bow_neighbors": 0,
            "matching_lowes_ratio": 0.8,
            "matching_gps_distance": 150,
            "matching_time_prior_sigma": 1.0,
            "retrieval_use_words": True,
            "align_method": "auto",
            "align_orientation_prior": "horizontal",
            "triangulation_type": "ROBUST",
            "resection_type": "ROBUST",
            "bundle_use_gps": True,
            "bundle_use_gcp": True,
            "optimize_camera_parameters": True,
            "undistorted_image_format": "jpg",
            # Mapillary-specific optimizations
            "bundle_use_gps": True,
            "bundle_gps_dop_threshold": 25,
            "bundle_gps_max_error": 5.0,
            "matching_use_words": True,
            "matching_order_neighbors": 50,
            "feature_use_adaptive_suppression": True,
        }
        
        config_path = self.output_dir / "config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config_data, f, default_flow_style=False)
        
        # Save reference.json with WGS84 reference
        reference_data = {
            "latitude": 0.0,
            "longitude": 0.0,
            "altitude": 0.0
        }
        
        # If we have GPS data, use the first image's location as reference
        if self.image_metadata:
            first_metadata = next(iter(self.image_metadata.values()))
            gps = first_metadata.get("gps")
            if gps:
                reference_data.update({
                    "latitude": gps["latitude"],
                    "longitude": gps["longitude"],
                    "altitude": gps["altitude"]
                })
        
        reference_path = self.output_dir / "reference.json"
        with open(reference_path, 'w') as f:
            json.dump(reference_data, f, indent=2)
        
        # Save dataset statistics
        stats = self._calculate_dataset_statistics()
        stats_path = self.metadata_dir / "dataset_statistics.json"
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
    
    def _calculate_dataset_statistics(self) -> Dict:
        """Calculate comprehensive dataset statistics"""
        stats = {
            "total_images": len(self.image_metadata),
            "total_cameras": len(self.camera_models),
            "total_sequences": len(self.sequence_data),
            "image_types": {"street_view": 0, "panoramic": 0},
            "camera_manufacturers": {},
            "capture_time_range": {"earliest": None, "latest": None},
            "spatial_extent": {"min_lat": None, "max_lat": None, "min_lon": None, "max_lon": None},
            "resolution_statistics": {"min_width": None, "max_width": None, "min_height": None, "max_height": None},
        }
        
        for image_meta in self.image_metadata.values():
            # Count image types
            if image_meta.get("is_panoramic"):
                stats["image_types"]["panoramic"] += 1
            else:
                stats["image_types"]["street_view"] += 1
            
            # Track manufacturers
            make = image_meta.get("make", "unknown")
            stats["camera_manufacturers"][make] = stats["camera_manufacturers"].get(make, 0) + 1
            
            # Track capture time range
            capture_time = image_meta.get("capture_time", 0)
            if capture_time > 0:
                if stats["capture_time_range"]["earliest"] is None or capture_time < stats["capture_time_range"]["earliest"]:
                    stats["capture_time_range"]["earliest"] = capture_time
                if stats["capture_time_range"]["latest"] is None or capture_time > stats["capture_time_range"]["latest"]:
                    stats["capture_time_range"]["latest"] = capture_time
            
            # Track spatial extent
            gps = image_meta.get("gps")
            if gps:
                lat, lon = gps["latitude"], gps["longitude"]
                if stats["spatial_extent"]["min_lat"] is None or lat < stats["spatial_extent"]["min_lat"]:
                    stats["spatial_extent"]["min_lat"] = lat
                if stats["spatial_extent"]["max_lat"] is None or lat > stats["spatial_extent"]["max_lat"]:
                    stats["spatial_extent"]["max_lat"] = lat
                if stats["spatial_extent"]["min_lon"] is None or lon < stats["spatial_extent"]["min_lon"]:
                    stats["spatial_extent"]["min_lon"] = lon
                if stats["spatial_extent"]["max_lon"] is None or lon > stats["spatial_extent"]["max_lon"]:
                    stats["spatial_extent"]["max_lon"] = lon
            
            # Track resolution statistics
            width, height = image_meta.get("width", 0), image_meta.get("height", 0)
            if width > 0:
                if stats["resolution_statistics"]["min_width"] is None or width < stats["resolution_statistics"]["min_width"]:
                    stats["resolution_statistics"]["min_width"] = width
                if stats["resolution_statistics"]["max_width"] is None or width > stats["resolution_statistics"]["max_width"]:
                    stats["resolution_statistics"]["max_width"] = width
            if height > 0:
                if stats["resolution_statistics"]["min_height"] is None or height < stats["resolution_statistics"]["min_height"]:
                    stats["resolution_statistics"]["min_height"] = height
                if stats["resolution_statistics"]["max_height"] is None or height > stats["resolution_statistics"]["max_height"]:
                    stats["resolution_statistics"]["max_height"] = height
        
        return stats

class MapillaryDownloader:
    """Main downloader class that orchestrates the download process"""
    
    def __init__(self, config: MapillaryConfig, output_dir: Path, show_progress: bool = True):
        self.config = config
        self.api = MapillaryAPI(config)
        self.structure = OpenSfMStructure(output_dir)
        self.downloaded_images = set()
        self.progress_tracker = ProgressTracker(show_progress)
        
        # Handle chunking for large bounding boxes
        self.chunks = self._prepare_chunks()
    
    def _prepare_chunks(self) -> List[BoundingBox]:
        """Prepare chunks for the bounding box if it's too large"""
        if not self.config.bbox:
            return []
        
        bbox = BoundingBox(*self.config.bbox)
        if not bbox.is_valid():
            raise ValueError(f"Invalid bounding box: {self.config.bbox}")
        
        chunks = bbox.split_into_chunks(self.config.chunk_size, self.config.overlap)
        
        if len(chunks) > 1:
            logger.info(f"Large bounding box detected. Splitting into {len(chunks)} chunks for API compliance.")
            logger.info(f"Original bbox: {bbox.to_string()}")
            logger.info(f"Chunk size: {self.config.chunk_size} degrees")
        
        return chunks
    
    def _process_with_chunks(self, download_func, *args, **kwargs):
        """Process downloads with chunking support"""
        if not self.chunks:
            # No chunking needed, use original bbox
            return download_func(*args, **kwargs)
        
        # Create parent progress bar for chunks
        self.progress_tracker.create_parent_progress(
            len(self.chunks), 
            "Processing chunks"
        )
        
        logger.info(f"Processing {len(self.chunks)} chunks...")
        all_results = []
        
        for i, chunk in enumerate(self.chunks, 1):
            logger.info(f"Processing chunk {i}/{len(self.chunks)}: {chunk.to_string()}")
            
            # Update parent progress bar postfix with current chunk info
            self.progress_tracker.set_parent_postfix(
                chunk=f"{i}/{len(self.chunks)}",
                speed=f"{self.progress_tracker.get_download_speed():.1f} MB/s"
            )
            
            # Temporarily update the config bbox for this chunk
            original_bbox = self.config.bbox
            self.config.bbox = chunk.to_tuple()
            
            try:
                # Call the download function for this chunk
                result = download_func(*args, **kwargs)
                if result:
                    all_results.extend(result if isinstance(result, list) else [result])
            except Exception as e:
                logger.warning(f"Error processing chunk {i}: {e}")
                continue
            finally:
                # Restore original bbox
                self.config.bbox = original_bbox
                # Update parent progress
                self.progress_tracker.update_parent()
        
        # Close parent progress bar
        self.progress_tracker.close_parent()
        
        return all_results
    
    def download_street_view_images(self, include_detections: bool = False):
        """Download street view images with metadata"""
        logger.info("Downloading street view images...")
        
        if self.chunks:
            # Use chunking approach
            return self._process_with_chunks(self._download_street_view_chunk, include_detections)
        else:
            # Direct download without chunking
            return self._download_street_view_chunk(include_detections)
    
    def _download_street_view_chunk(self, include_detections: bool = False):
        """Download street view images for a single chunk or the entire bbox"""
        # Define comprehensive fields to retrieve all available camera and metadata
        fields = [
            # Basic image info
            "id", "captured_at", "width", "height", "sequence", "is_pano",
            
            # Camera information
            "make", "model", "camera_parameters", "camera_type", "exif_orientation",
            
            # Geometry and pose information
            "geometry", "computed_geometry", "compass_angle", "computed_compass_angle",
            "altitude", "computed_altitude", "computed_rotation", "atomic_scale",
            
            # Image URLs
            "thumb_256_url", "thumb_1024_url", "thumb_2048_url", "thumb_original_url",
            
            # SfM and reconstruction data
            "sfm_cluster.id", "sfm_cluster.url", "mesh.id", "mesh.url", "merge_cc",
            
            # Creator and organization info
            "creator.id", "creator.username",
        ]
        
        if include_detections:
            fields.extend([
                "detections.id", "detections.value", "detections.geometry", "detections.created_at"
            ])
        
        # Search for images
        images = self.api.search_images(fields, {"is_pano": False})  # Street view only
        
        if not images:
            return []
        
        logger.info(f"Found {len(images)} street view images in current area")
        
        # Filter out already downloaded images
        new_images = []
        for image in images:
            image_id = image["id"]
            if image_id not in self.downloaded_images:
                image_filename = f"{image_id}.jpg"
                output_path = self.structure.images_dir / image_filename
                if not output_path.exists():
                    new_images.append(image)
                else:
                    self.downloaded_images.add(image_id)
        
        if not new_images:
            return []
        
        # Create child progress bar
        self.progress_tracker.create_child_progress(
            len(new_images), 
            "Downloading street view images"
        )
        
        # Download images with thread pool
        downloaded_images = []
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            download_tasks = []
            
            for image in new_images:
                image_id = image["id"]
                
                # Use highest quality available
                image_url = image.get("thumb_original_url") or image.get("thumb_2048_url")
                if not image_url:
                    logger.warning(f"No image URL found for {image_id}")
                    self.progress_tracker.update_child()
                    continue
                
                image_filename = f"{image_id}.jpg"
                output_path = self.structure.images_dir / image_filename
                
                # Submit download task
                task = executor.submit(
                    self._download_single_image, 
                    image, image_url, output_path, image_filename, 
                    include_detections, self.progress_tracker
                )
                download_tasks.append((task, image_id))
            
            # Process completed downloads
            for task, image_id in download_tasks:
                try:
                    if task.result():
                        downloaded_images.append(image_id)
                        self.downloaded_images.add(image_id)
                    
                    # Update progress and speed info
                    self.progress_tracker.update_child()
                    self.progress_tracker.set_child_postfix(
                        speed=f"{self.progress_tracker.get_download_speed():.1f} MB/s"
                    )
                    
                except Exception as e:
                    logger.error(f"Failed to download image {image_id}: {e}")
                    self.progress_tracker.update_child()
        
        # Close child progress bar
        self.progress_tracker.close_child()
        
        return downloaded_images
    
    def download_panoramic_images(self, include_detections: bool = False):
        """Download panoramic/360-degree images"""
        logger.info("Downloading panoramic images...")
        
        if self.chunks:
            # Use chunking approach
            return self._process_with_chunks(self._download_panoramic_chunk, include_detections)
        else:
            # Direct download without chunking
            return self._download_panoramic_chunk(include_detections)
    
    def _download_panoramic_chunk(self, include_detections: bool = False):
        """Download panoramic images for a single chunk or the entire bbox"""
        # Define comprehensive fields for panoramic images
        fields = [
            # Basic image info
            "id", "captured_at", "width", "height", "sequence", "is_pano",
            
            # Camera information
            "make", "model", "camera_parameters", "camera_type", "exif_orientation",
            
            # Geometry and pose information
            "geometry", "computed_geometry", "compass_angle", "computed_compass_angle",
            "altitude", "computed_altitude", "computed_rotation", "atomic_scale",
            
            # Image URLs
            "thumb_256_url", "thumb_1024_url", "thumb_2048_url", "thumb_original_url",
            
            # SfM and reconstruction data
            "sfm_cluster.id", "sfm_cluster.url", "mesh.id", "mesh.url", "merge_cc",
            
            # Creator and organization info
            "creator.id", "creator.username",
        ]
        
        if include_detections:
            fields.extend([
                "detections.id", "detections.value", "detections.geometry", "detections.created_at"
            ])
        
        # Search for panoramic images
        images = self.api.search_images(fields, {"is_pano": True})
        
        if not images:
            return []
        
        logger.info(f"Found {len(images)} panoramic images in current area")
        
        # Create separate directory for panoramic images
        pano_dir = self.structure.images_dir / "panoramic"
        pano_dir.mkdir(exist_ok=True)
        
        # Filter out already downloaded images
        new_images = []
        for image in images:
            image_id = image["id"]
            if image_id not in self.downloaded_images:
                image_filename = f"pano_{image_id}.jpg"
                output_path = pano_dir / image_filename
                if not output_path.exists():
                    new_images.append(image)
                else:
                    self.downloaded_images.add(image_id)
        
        if not new_images:
            return []
        
        # Create child progress bar
        self.progress_tracker.create_child_progress(
            len(new_images), 
            "Downloading panoramic images"
        )
        
        downloaded_images = []
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            download_tasks = []
            
            for image in new_images:
                image_id = image["id"]
                image_url = image.get("thumb_original_url") or image.get("thumb_2048_url")
                
                if not image_url:
                    self.progress_tracker.update_child()
                    continue
                
                image_filename = f"pano_{image_id}.jpg"
                output_path = pano_dir / image_filename
                
                task = executor.submit(
                    self._download_single_image, 
                    image, image_url, output_path, image_filename, 
                    include_detections, self.progress_tracker
                )
                download_tasks.append((task, image_id))
            
            # Process completed downloads
            for task, image_id in download_tasks:
                try:
                    if task.result():
                        downloaded_images.append(image_id)
                        self.downloaded_images.add(image_id)
                    
                    # Update progress and speed info
                    self.progress_tracker.update_child()
                    self.progress_tracker.set_child_postfix(
                        speed=f"{self.progress_tracker.get_download_speed():.1f} MB/s"
                    )
                    
                except Exception as e:
                    logger.error(f"Failed to download panoramic image {image_id}: {e}")
                    self.progress_tracker.update_child()
        
        # Close child progress bar
        self.progress_tracker.close_child()
        
        return downloaded_images
    
    def download_point_clouds(self):
        """Download SfM point cloud data"""
        logger.info("Downloading point cloud data...")
        
        if self.chunks:
            # Use chunking approach
            return self._process_with_chunks(self._download_point_clouds_chunk)
        else:
            # Direct download without chunking
            return self._download_point_clouds_chunk()
    
    def _download_point_clouds_chunk(self):
        """Download point clouds for a single chunk or the entire bbox"""
        # Define fields for SfM cluster data
        fields = [
            "id", "computed_geometry", "geometry", "captured_at", "sequence",
            "sfm_cluster.id", "sfm_cluster.url", "mesh.id", "mesh.url", 
            "merge_cc", "atomic_scale", "camera_parameters", "make", "model"
        ]
        
        images = self.api.search_images(fields)
        
        if not images:
            return []
        
        # Collect unique clusters
        unique_clusters = {}
        for image in images:
            sfm_cluster = image.get("sfm_cluster")
            if sfm_cluster:
                cluster_id = sfm_cluster.get("id")
                cluster_url = sfm_cluster.get("url")
                if cluster_url and cluster_id:
                    unique_clusters[cluster_id] = cluster_url
        
        if not unique_clusters:
            return []
        
        # Create child progress bar for point cloud clusters
        self.progress_tracker.create_child_progress(
            len(unique_clusters), 
            "Downloading point cloud clusters"
        )
        
        point_cloud_data = []
        downloaded_clusters = []
        
        for cluster_id, cluster_url in unique_clusters.items():
            logger.info(f"Downloading SfM cluster {cluster_id}")
            
            try:
                points = self.api.download_sfm_cluster(cluster_url)
                if points:
                    point_cloud_data.extend(points)
                    downloaded_clusters.append(cluster_id)
                    
                    # Save individual cluster
                    cluster_filename = f"sfm_cluster_{cluster_id}.json"
                    cluster_path = self.structure.pointclouds_dir / cluster_filename
                    with open(cluster_path, 'w') as f:
                        json.dump(points, f, indent=2)
                        
            except Exception as e:
                logger.error(f"Failed to download cluster {cluster_id}: {e}")
            
            # Update progress
            self.progress_tracker.update_child()
            self.progress_tracker.set_child_postfix(
                clusters=f"{len(downloaded_clusters)}/{len(unique_clusters)}"
            )
        
        # Close child progress bar
        self.progress_tracker.close_child()
        
        # Save combined point cloud for this chunk/area
        if point_cloud_data:
            # For chunks, save with chunk identifier, otherwise save normally
            if self.chunks and len(self.chunks) > 1:
                bbox_str = f"{self.config.bbox[0]:.6f}_{self.config.bbox[1]:.6f}_{self.config.bbox[2]:.6f}_{self.config.bbox[3]:.6f}"
                chunk_filename = f"pointcloud_chunk_{bbox_str}.json"
                chunk_path = self.structure.pointclouds_dir / chunk_filename
                with open(chunk_path, 'w') as f:
                    json.dump(point_cloud_data, f, indent=2)
            else:
                # Save as combined when no chunking
                self.structure.save_point_cloud(point_cloud_data, "combined_pointcloud.ply")
                combined_path = self.structure.pointclouds_dir / "combined_pointcloud.json"
                with open(combined_path, 'w') as f:
                    json.dump(point_cloud_data, f, indent=2)
        
        logger.info(f"Downloaded {len(downloaded_clusters)} point cloud clusters in current area")
        return downloaded_clusters
    
    def download_map_features(self, feature_types: Optional[List[str]] = None):
        """Download map features (traffic signs, points of interest, etc.)"""
        logger.info("Downloading map features...")
        
        fields = [
            "id", "geometry", "object_value", "object_type", "aligned_direction",
            "first_seen_at", "last_seen_at", "images"
        ]
        
        additional_params = {}
        if feature_types:
            additional_params["object_values"] = ",".join(feature_types)
        
        features = self.api.search_map_features(fields, additional_params)
        
        logger.info(f"Found {len(features)} map features")
        
        self.structure.save_map_features(features)
        
        return features
    
    def download_detections(self, image_ids: Optional[List[str]] = None):
        """Download detection data for specific images"""
        logger.info("Downloading detection data...")
        
        if not image_ids:
            # Get image IDs from already processed images
            image_ids = [Path(f).stem for f in self.downloaded_images]
        
        fields = ["id", "value", "geometry", "image"]
        
        all_detections = {}
        
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            detection_tasks = {
                executor.submit(self.api.get_detections, image_id, fields): image_id 
                for image_id in image_ids
            }
            
            for task in as_completed(detection_tasks):
                image_id = detection_tasks[task]
                try:
                    detections = task.result()
                    if detections:
                        all_detections[image_id] = detections
                        self.structure.save_detections(image_id, detections)
                except Exception as e:
                    logger.error(f"Failed to download detections for {image_id}: {e}")
        
        logger.info(f"Downloaded detections for {len(all_detections)} images")
        return all_detections
    
    def _download_single_image(self, image_data: Dict, image_url: str, output_path: Path, 
                             image_filename: str, include_detections: bool, 
                             progress_tracker: Optional[ProgressTracker] = None):
        """Download a single image and its metadata"""
        try:
            # Download image
            if self.api.download_image(image_url, output_path, progress_tracker):
                # Add metadata
                self.structure.add_image_metadata(image_data, image_filename)
                self.downloaded_images.add(image_filename)
                
                # Download detections if requested
                if include_detections:
                    image_id = image_data["id"]
                    detections = self.api.get_detections(image_id, ["id", "value", "geometry"])
                    if detections:
                        self.structure.save_detections(image_id, detections)
                
                logger.debug(f"Downloaded {image_filename}")
                return True
            else:
                logger.warning(f"Failed to download {image_filename}")
                return False
                
        except Exception as e:
            logger.error(f"Error downloading {image_filename}: {e}")
            return False
    
    def _merge_chunked_point_clouds(self):
        """Merge all chunked point cloud files into a single combined file"""
        if not self.chunks or len(self.chunks) <= 1:
            return
        
        logger.info("Merging chunked point cloud data...")
        
        all_reconstructions = []
        chunk_files = []
        
        # Find all chunk files
        for chunk_file in self.structure.pointclouds_dir.glob("pointcloud_chunk_*.json"):
            chunk_files.append(chunk_file)
            try:
                with open(chunk_file, 'r') as f:
                    chunk_data = json.load(f)
                    all_reconstructions.extend(chunk_data)
            except Exception as e:
                logger.error(f"Failed to read chunk file {chunk_file}: {e}")
        
        if all_reconstructions:
            # For SfM reconstruction data, we need to merge differently
            # Each reconstruction contains cameras, shots, and points
            merged_reconstruction = {
                "cameras": {},
                "shots": {},
                "points": {}
            }
            
            # Merge all reconstructions
            for reconstruction in all_reconstructions:
                if isinstance(reconstruction, dict):
                    # Merge cameras
                    if "cameras" in reconstruction:
                        merged_reconstruction["cameras"].update(reconstruction["cameras"])
                    
                    # Merge shots (image poses)
                    if "shots" in reconstruction:
                        merged_reconstruction["shots"].update(reconstruction["shots"])
                    
                    # Merge points
                    if "points" in reconstruction:
                        merged_reconstruction["points"].update(reconstruction["points"])
                elif isinstance(reconstruction, list):
                    # Handle case where reconstruction is a list of reconstructions
                    for sub_recon in reconstruction:
                        if isinstance(sub_recon, dict):
                            if "cameras" in sub_recon:
                                merged_reconstruction["cameras"].update(sub_recon["cameras"])
                            if "shots" in sub_recon:
                                merged_reconstruction["shots"].update(sub_recon["shots"])
                            if "points" in sub_recon:
                                merged_reconstruction["points"].update(sub_recon["points"])
            
            # Save final combined reconstruction
            combined_path = self.structure.pointclouds_dir / "combined_reconstruction.json"
            with open(combined_path, 'w') as f:
                json.dump([merged_reconstruction], f, indent=2)
            
            # Also try to extract actual 3D points for PLY export if they exist
            if merged_reconstruction["points"]:
                try:
                    point_coordinates = []
                    point_colors = []
                    
                    for point_id, point_data in merged_reconstruction["points"].items():
                        if isinstance(point_data, dict):
                            # Extract coordinates
                            if "coordinates" in point_data:
                                coords = point_data["coordinates"]
                                if len(coords) >= 3:
                                    point_coordinates.append(coords[:3])
                            
                            # Extract color if available
                            if "color" in point_data:
                                color = point_data["color"]
                                if len(color) >= 3:
                                    point_colors.append(color[:3])
                            elif len(point_coordinates) > len(point_colors):
                                # Default color if none provided
                                point_colors.append([128, 128, 128])
                    
                    if point_coordinates:
                        # Create point cloud data in expected format
                        point_cloud_data = []
                        for i, coords in enumerate(point_coordinates):
                            point_dict = {
                                "coordinates": coords,
                                "color": point_colors[i] if i < len(point_colors) else [128, 128, 128]
                            }
                            point_cloud_data.append(point_dict)
                        
                        self.structure.save_point_cloud(point_cloud_data, "combined_pointcloud.ply")
                        
                        combined_points_path = self.structure.pointclouds_dir / "combined_pointcloud.json"
                        with open(combined_points_path, 'w') as f:
                            json.dump(point_cloud_data, f, indent=2)
                        
                        logger.info(f"Merged {len(point_coordinates)} 3D points from {len(chunk_files)} chunks")
                    else:
                        logger.info(f"Merged SfM reconstruction data from {len(chunk_files)} chunks (no 3D points found)")
                except Exception as e:
                    logger.warning(f"Could not extract 3D points for PLY export: {e}")
                    logger.info(f"Merged SfM reconstruction data from {len(chunk_files)} chunks")
            else:
                logger.info(f"Merged SfM reconstruction data from {len(chunk_files)} chunks (no 3D points in reconstruction)")
            
            # Optionally remove chunk files to save space
            # for chunk_file in chunk_files:
            #     chunk_file.unlink()
    
    def finalize(self):
        """Finalize the download process and save metadata files"""
        logger.info("Finalizing download and saving metadata...")
        
        # Merge chunked point clouds if needed
        self._merge_chunked_point_clouds()
        
        self.structure.save_metadata_files()
        
        # Close all progress bars
        self.progress_tracker.close_all()
        
        # Print final statistics
        total_mb = self.progress_tracker.total_bytes / (1024 * 1024)
        avg_speed = self.progress_tracker.get_download_speed()
        
        logger.info("Download completed successfully!")
        if total_mb > 0:
            logger.info(f"Total downloaded: {total_mb:.1f} MB at average speed: {avg_speed:.1f} MB/s")

def parse_bbox(bbox_str: str) -> Tuple[float, float, float, float]:
    """Parse bounding box string to tuple"""
    try:
        coords = [float(x.strip()) for x in bbox_str.split(',')]
        if len(coords) != 4:
            raise ValueError("Bounding box must have 4 coordinates")
        return tuple(coords)
    except Exception as e:
        raise argparse.ArgumentTypeError(f"Invalid bounding box format: {e}")

def main():
    parser = argparse.ArgumentParser(
        description="Mapillary Data Downloader - Download street view, aerial, point clouds, and detections",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download street view and point clouds (default)
  python mapillary_downloader.py --access-token YOUR_TOKEN --output-dir ./data/mapillary

  # Download specific data types
  python mapillary_downloader.py --access-token YOUR_TOKEN --output-dir ./data/mapillary \\
    --street-view --aerial-view --point-clouds --detections --map-features

  # Download with geographic filtering
  python mapillary_downloader.py --access-token YOUR_TOKEN --output-dir ./data/mapillary \\
    --bbox "13.400,52.519,13.401,52.520" --street-view --point-clouds

  # Download for specific user
  python mapillary_downloader.py --access-token YOUR_TOKEN --output-dir ./data/mapillary \\
    --creator-username "your_username" --street-view
        """
    )
    
    # Required arguments
    parser.add_argument(
        '--access-token', 
        required=True,
        help='Mapillary API access token'
    )
    parser.add_argument(
        '--output-dir', 
        required=True,
        type=Path,
        help='Output directory for downloaded data (OpenSfM structure)'
    )
    
    # Data type flags
    parser.add_argument(
        '--street-view',
        action='store_true',
        help='Download street view images'
    )
    parser.add_argument(
        '--aerial-view',
        action='store_true',
        help='Download aerial/panoramic images'
    )
    parser.add_argument(
        '--point-clouds',
        action='store_true',
        help='Download SfM point cloud data'
    )
    parser.add_argument(
        '--detections',
        action='store_true',
        help='Download object detection data'
    )
    parser.add_argument(
        '--map-features',
        action='store_true',
        help='Download map features (traffic signs, POIs)'
    )
    
    # Filtering options
    parser.add_argument(
        '--bbox',
        type=parse_bbox,
        help='Bounding box: "left,bottom,right,top" (longitude,latitude format)'
    )
    parser.add_argument(
        '--creator-username',
        help='Filter by username who uploaded the images'
    )
    parser.add_argument(
        '--organization-id',
        type=int,
        help='Filter by organization ID'
    )
    parser.add_argument(
        '--start-date',
        help='Start date for captured images (ISO format: 2022-08-16T16:42:46Z)'
    )
    parser.add_argument(
        '--end-date',
        help='End date for captured images (ISO format: 2022-08-16T16:42:46Z)'
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=2000,
        help='Maximum number of items to download per request (default: 2000)'
    )
    
    # Performance options
    parser.add_argument(
        '--max-workers',
        type=int,
        default=4,
        help='Maximum number of concurrent download threads (default: 4)'
    )
    parser.add_argument(
        '--retry-count',
        type=int,
        default=3,
        help='Number of retries for failed requests (default: 3)'
    )
    parser.add_argument(
        '--delay',
        type=float,
        default=0.1,
        help='Delay between requests in seconds (default: 0.1)'
    )
    parser.add_argument(
        '--chunk-size',
        type=float,
        default=0.01,
        help='Maximum chunk size in degrees for large bounding boxes (default: 0.01)'
    )
    parser.add_argument(
        '--overlap',
        type=float,
        default=0.001,
        help='Overlap between chunks in degrees to avoid gaps (default: 0.001)'
    )
    parser.add_argument(
        '--no-progress',
        action='store_true',
        help='Disable progress bars (useful for logging/automation)'
    )
    
    # Feature filtering
    parser.add_argument(
        '--feature-types',
        nargs='+',
        help='Specific map feature types to download (e.g., object--sign--* object--support--utility-pole)'
    )
    
    args = parser.parse_args()
    
    # If no data types specified, download street-view and point-clouds by default
    if not any([args.street_view, args.aerial_view, args.point_clouds, args.detections, args.map_features]):
        args.street_view = True
        args.point_clouds = True
        logger.info("No data types specified, downloading street view and point clouds by default")
    
    # Create configuration
    config = MapillaryConfig(
        access_token=args.access_token,
        bbox=args.bbox,
        creator_username=args.creator_username,
        organization_id=args.organization_id,
        start_captured_at=args.start_date,
        end_captured_at=args.end_date,
        limit=args.limit,
        max_workers=args.max_workers,
        retry_count=args.retry_count,
        delay_between_requests=args.delay,
        chunk_size=args.chunk_size,
        overlap=args.overlap
    )
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize downloader
    downloader = MapillaryDownloader(config, args.output_dir, show_progress=not args.no_progress)
    
    try:
        # Download requested data types
        if args.street_view:
            downloader.download_street_view_images(include_detections=args.detections)
        
        if args.aerial_view:
            downloader.download_panoramic_images(include_detections=args.detections)
        
        if args.point_clouds:
            downloader.download_point_clouds()
        
        if args.map_features:
            downloader.download_map_features(args.feature_types)
        
        if args.detections and not (args.street_view or args.aerial_view):
            # Download detections separately if not already downloaded with images
            downloader.download_detections()
        
        # Finalize
        downloader.finalize()
        
        logger.info(f"✅ Download completed successfully!")
        logger.info(f"📁 Data saved to: {args.output_dir}")
        logger.info(f"📋 Structure follows OpenSfM format for 3D reconstruction")
        
    except KeyboardInterrupt:
        logger.info("Download interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Download failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
