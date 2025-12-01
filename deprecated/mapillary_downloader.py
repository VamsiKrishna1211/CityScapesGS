#!/usr/bin/env python3
"""
Mapillary Data Downloader

A modular Python script for downloading street view images and point cloud data
from the Mapillary API based on a bounding box.

Usage:
    python mapillary_downloader.py --bbox "west,south,east,north" --token "your_access_token"
    python mapillary_downloader.py --bbox "-80.134,25.773,-80.126,25.789" --token "MLY|xxx"
"""

import argparse
import asyncio
import aiohttp
import aiofiles
import json
import logging
import os
import sys
import time
import zlib
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
import mercantile
from tqdm.asyncio import tqdm
import base64


class MapillaryAuthenticator:
    """Handles authentication and token management for Mapillary API."""
    
    def __init__(self, access_token: str):
        """
        Initialize authenticator with access token.
        
        Args:
            access_token: Mapillary access token
        """
        self.access_token = access_token
        self.token_usage_count = 0
        self.last_request_time = 0
        self.rate_limit_delay = 0.1  # Minimum delay between requests
        
    def get_headers(self) -> Dict[str, str]:
        """Get authorization headers for API requests."""
        return {
            'Authorization': f'OAuth {self.access_token}',
            'Content-Type': 'application/json'
        }
    
    def get_token_param(self) -> str:
        """Get access token for URL parameter."""
        return self.access_token
    
    async def rate_limit_wait(self):
        """Implement rate limiting to avoid hitting API limits."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.rate_limit_delay:
            await asyncio.sleep(self.rate_limit_delay - time_since_last)
        
        self.last_request_time = time.time()
        self.token_usage_count += 1


class BoundingBoxHandler:
    """Handles bounding box operations and tile generation."""
    
    def __init__(self, bbox: List[float], zoom_level: int = 14):
        """
        Initialize bounding box handler.
        
        Args:
            bbox: Bounding box as [west, south, east, north]
            zoom_level: Zoom level for tile generation (default: 14)
        """
        self.west, self.south, self.east, self.north = bbox
        self.zoom_level = zoom_level
        self.validate_bbox()
    
    def validate_bbox(self):
        """Validate bounding box constraints."""
        # Check that the bbox area is not too large (Mapillary limit: 0.01 degrees square)
        area = (self.east - self.west) * (self.north - self.south)
        if area > 0.01:
            raise ValueError(f"Bounding box area ({area:.6f}) exceeds Mapillary limit of 0.01 square degrees")
        
        # Check coordinate validity
        if not (-180 <= self.west < self.east <= 180):
            raise ValueError("Invalid longitude coordinates")
        if not (-90 <= self.south < self.north <= 90):
            raise ValueError("Invalid latitude coordinates")
    
    def get_tiles(self) -> List[mercantile.Tile]:
        """Generate tiles that intersect with the bounding box."""
        return list(mercantile.tiles(self.west, self.south, self.east, self.north, self.zoom_level))
    
    def is_point_in_bbox(self, lon: float, lat: float) -> bool:
        """Check if a point is within the bounding box."""
        return self.west <= lon <= self.east and self.south <= lat <= self.north
    
    def to_string(self) -> str:
        """Convert bounding box to string format for API calls."""
        return f"{self.west},{self.south},{self.east},{self.north}"


class VectorTileProcessor:
    """Processes vector tiles from Mapillary API."""
    
    def __init__(self, authenticator: MapillaryAuthenticator):
        """
        Initialize vector tile processor.
        
        Args:
            authenticator: MapillaryAuthenticator instance
        """
        self.authenticator = authenticator
        self.tile_coverage = 'mly1_public'
        self.tile_layer = 'image'
        
    async def fetch_tile_data(self, session: aiohttp.ClientSession, tile: mercantile.Tile) -> Dict[str, Any]:
        """
        Fetch vector tile data from Mapillary API.
        
        Args:
            session: aiohttp session
            tile: Mercantile tile object
            
        Returns:
            GeoJSON-like dictionary with features
        """
        await self.authenticator.rate_limit_wait()
        
        url = (f'https://tiles.mapillary.com/maps/vtp/{self.tile_coverage}/2/'
               f'{tile.z}/{tile.x}/{tile.y}?access_token={self.authenticator.get_token_param()}')
        
        try:
            async with session.get(url) as response:
                if response.status == 200:
                    content = await response.read()
                    return await self._process_vector_tile(content, tile.x, tile.y, tile.z)
                elif response.status == 429:
                    logging.warning("Rate limit hit, waiting...")
                    await asyncio.sleep(60)
                    return await self.fetch_tile_data(session, tile)
                else:
                    logging.error(f"Failed to fetch tile {tile.x}/{tile.y}/{tile.z}: {response.status}")
                    return {"features": []}
        except Exception as e:
            logging.error(f"Error fetching tile {tile.x}/{tile.y}/{tile.z}: {e}")
            return {"features": []}
    
    async def _process_vector_tile(self, content: bytes, x: int, y: int, z: int) -> Dict[str, Any]:
        """Process vector tile content to GeoJSON."""
        try:
            # This would require the vt2geojson library
            # For now, we'll implement a basic parser or skip this step
            # In a real implementation, you'd use: from vt2geojson.tools import vt_bytes_to_geojson
            # return vt_bytes_to_geojson(content, x, y, z, layer=self.tile_layer)
            
            # Placeholder implementation - in reality you'd parse the MVT
            logging.warning("Vector tile processing not fully implemented - requires vt2geojson library")
            return {"features": []}
        except Exception as e:
            logging.error(f"Error processing vector tile: {e}")
            return {"features": []}


class ImageDownloader:
    """Handles downloading of Mapillary images with support for multiple types and resolutions."""
    
    def __init__(self, authenticator: MapillaryAuthenticator, output_dir: str, 
                 download_panoramic: bool = True, download_standard: bool = True,
                 resolutions: List[str] = None):
        """
        Initialize image downloader.
        
        Args:
            authenticator: MapillaryAuthenticator instance
            output_dir: Output directory for images
            download_panoramic: Whether to download panoramic (360°) images
            download_standard: Whether to download standard perspective images
            resolutions: List of resolutions to download ['thumb_256_url', 'thumb_1024_url', 'thumb_2048_url', 'thumb_original_url']
        """
        self.authenticator = authenticator
        self.output_dir = output_dir
        self.download_panoramic = download_panoramic
        self.download_standard = download_standard
        self.resolutions = resolutions or ['thumb_2048_url', 'thumb_1024_url']
        
        # Create directories for different image types
        self.images_dir = os.path.join(output_dir, "images")
        self.panoramic_dir = os.path.join(self.images_dir, "panoramic")
        self.standard_dir = os.path.join(self.images_dir, "standard")
        
        os.makedirs(self.images_dir, exist_ok=True)
        os.makedirs(self.panoramic_dir, exist_ok=True)
        os.makedirs(self.standard_dir, exist_ok=True)
        
    async def search_images(self, session: aiohttp.ClientSession, bbox: BoundingBoxHandler, 
                          limit: int = 2000, is_pano: Optional[bool] = None) -> List[Dict[str, Any]]:
        """
        Search for images within bounding box with enhanced filtering.
        
        Args:
            session: aiohttp session
            bbox: BoundingBoxHandler instance
            limit: Maximum number of images to return
            is_pano: Filter for panoramic images (True), standard images (False), or all (None)
            
        Returns:
            List of image metadata dictionaries
        """
        await self.authenticator.rate_limit_wait()
        
        url = f"https://graph.mapillary.com/images"
        
        # Enhanced fields including panoramic info, camera type, and multiple resolution URLs
        fields = [
            'id', 'computed_geometry', 'geometry', 'captured_at', 'creator', 'compass_angle',
            'is_pano', 'camera_type', 'sequence', 'width', 'height', 'make', 'model',
            'thumb_256_url', 'thumb_1024_url', 'thumb_2048_url', 'thumb_original_url',
            'computed_compass_angle', 'computed_altitude', 'altitude'
        ]
        
        params = {
            'bbox': bbox.to_string(),
            'limit': limit,
            'fields': ','.join(fields)
        }
        
        # Add panoramic filter if specified
        if is_pano is not None:
            params['is_pano'] = 'true' if is_pano else 'false'
        
        headers = self.authenticator.get_headers()
        
        try:
            async with session.get(url, params=params, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get('data', [])
                elif response.status == 429:
                    logging.warning("Rate limit hit while searching images")
                    await asyncio.sleep(60)
                    return await self.search_images(session, bbox, limit, is_pano)
                else:
                    logging.error(f"Failed to search images: {response.status}")
                    return []
        except Exception as e:
            logging.error(f"Error searching images: {e}")
            return []
    
    async def download_image(self, session: aiohttp.ClientSession, image_data: Dict[str, Any]) -> bool:
        """
        Download a single image in multiple resolutions and categorize by type.
        
        Args:
            session: aiohttp session
            image_data: Image metadata dictionary
            
        Returns:
            True if at least one resolution was downloaded successfully, False otherwise
        """
        image_id = image_data.get('id')
        is_pano = image_data.get('is_pano', False)
        camera_type = image_data.get('camera_type', 'unknown')
        
        if not image_id:
            logging.warning(f"Missing image ID for image: {image_data}")
            return False
        
        # Skip if image type doesn't match download preferences
        if is_pano and not self.download_panoramic:
            return True  # Skip but don't count as failure
        if not is_pano and not self.download_standard:
            return True  # Skip but don't count as failure
        
        # Determine target directory based on image type
        if is_pano:
            target_dir = self.panoramic_dir
            subdir = os.path.join(target_dir, camera_type)
        else:
            target_dir = self.standard_dir
            subdir = os.path.join(target_dir, camera_type)
        
        os.makedirs(subdir, exist_ok=True)
        
        # Download multiple resolutions
        download_success = False
        
        for resolution in self.resolutions:
            image_url = image_data.get(resolution)
            if not image_url:
                continue
                
            # Extract resolution from URL field name (e.g., thumb_2048_url -> 2048)
            res_name = resolution.replace('thumb_', '').replace('_url', '')
            if res_name == 'original':
                res_name = 'orig'
            
            filename = f"{image_id}_{res_name}.jpg"
            filepath = os.path.join(subdir, filename)
            
            # Skip if already downloaded
            if os.path.exists(filepath):
                download_success = True
                continue
            
            try:
                async with session.get(image_url) as response:
                    if response.status == 200:
                        async with aiofiles.open(filepath, 'wb') as f:
                            async for chunk in response.content.iter_chunked(8192):
                                await f.write(chunk)
                        download_success = True
                        logging.debug(f"Downloaded {filename}")
                    else:
                        logging.warning(f"Failed to download {filename}: {response.status}")
            except Exception as e:
                logging.error(f"Error downloading {filename}: {e}")
        
        # Save enhanced metadata
        if download_success:
            metadata_path = os.path.join(subdir, f"{image_id}_metadata.json")
            enhanced_metadata = {
                **image_data,
                'download_info': {
                    'downloaded_at': datetime.now().isoformat(),
                    'image_type': 'panoramic' if is_pano else 'standard',
                    'camera_type': camera_type,
                    'resolutions_downloaded': [res for res in self.resolutions if image_data.get(res)],
                    'directory': subdir
                }
            }
            
            async with aiofiles.open(metadata_path, 'w') as f:
                await f.write(json.dumps(enhanced_metadata, indent=2))
        
        return download_success
    
    async def download_all_image_types(self, session: aiohttp.ClientSession, bbox: BoundingBoxHandler, 
                                     limit: int = 2000) -> Dict[str, int]:
        """
        Download both panoramic and standard images from a bounding box.
        
        Args:
            session: aiohttp session
            bbox: BoundingBoxHandler instance
            limit: Maximum number of images to download
            
        Returns:
            Dictionary with download statistics by type
        """
        stats = {
            'panoramic_downloaded': 0,
            'standard_downloaded': 0,
            'total_found': 0,
            'errors': 0
        }
        
        # Download panoramic images if enabled
        if self.download_panoramic:
            logging.info("Searching for panoramic images...")
            pano_images = await self.search_images(session, bbox, limit, is_pano=True)
            stats['total_found'] += len(pano_images)
            
            for image_data in pano_images:
                try:
                    if await self.download_image(session, image_data):
                        stats['panoramic_downloaded'] += 1
                    else:
                        stats['errors'] += 1
                except Exception as e:
                    logging.error(f"Error processing panoramic image: {e}")
                    stats['errors'] += 1
        
        # Download standard images if enabled
        if self.download_standard:
            logging.info("Searching for standard images...")
            standard_images = await self.search_images(session, bbox, limit, is_pano=False)
            stats['total_found'] += len(standard_images)
            
            for image_data in standard_images:
                try:
                    if await self.download_image(session, image_data):
                        stats['standard_downloaded'] += 1
                    else:
                        stats['errors'] += 1
                except Exception as e:
                    logging.error(f"Error processing standard image: {e}")
                    stats['errors'] += 1
        
        return stats


class PointCloudDownloader:
    """Handles downloading of point cloud data from Mapillary."""
    
    def __init__(self, authenticator: MapillaryAuthenticator, output_dir: str):
        """
        Initialize point cloud downloader.
        
        Args:
            authenticator: MapillaryAuthenticator instance
            output_dir: Output directory for point clouds
        """
        self.authenticator = authenticator
        self.output_dir = output_dir
        self.pointclouds_dir = os.path.join(output_dir, "pointclouds")
        os.makedirs(self.pointclouds_dir, exist_ok=True)
    
    async def get_image_pointcloud(self, session: aiohttp.ClientSession, image_id: str) -> Optional[str]:
        """
        Get point cloud data for a specific image.
        
        Args:
            session: aiohttp session
            image_id: Mapillary image ID
            
        Returns:
            Point cloud data URL or None
        """
        await self.authenticator.rate_limit_wait()
        
        url = f"https://graph.mapillary.com/{image_id}"
        params = {'fields': 'sfm_cluster'}
        headers = self.authenticator.get_headers()
        
        try:
            async with session.get(url, params=params, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    sfm_cluster = data.get('sfm_cluster')
                    if sfm_cluster:
                        return sfm_cluster.get('url')
                elif response.status == 429:
                    logging.warning("Rate limit hit while getting point cloud")
                    await asyncio.sleep(60)
                    return await self.get_image_pointcloud(session, image_id)
                else:
                    logging.warning(f"No point cloud data for image {image_id}: {response.status}")
                    
        except Exception as e:
            logging.error(f"Error getting point cloud for image {image_id}: {e}")
        
        return None
    
    async def download_pointcloud(self, session: aiohttp.ClientSession, image_id: str, 
                                pointcloud_url: str) -> bool:
        """
        Download and decompress point cloud data.
        
        Args:
            session: aiohttp session
            image_id: Mapillary image ID
            pointcloud_url: URL to point cloud data
            
        Returns:
            True if successful, False otherwise
        """
        filename = f"{image_id}_pointcloud.json"
        filepath = os.path.join(self.pointclouds_dir, filename)
        
        # Skip if already downloaded
        if os.path.exists(filepath):
            return True
        
        try:
            async with session.get(pointcloud_url) as response:
                if response.status == 200:
                    compressed_data = await response.read()
                    
                    # Decompress the data
                    try:
                        decompressed_data = zlib.decompress(compressed_data)
                        
                        # Save decompressed JSON data
                        async with aiofiles.open(filepath, 'wb') as f:
                            await f.write(decompressed_data)
                        
                        return True
                    except Exception as decompress_error:
                        logging.error(f"Failed to decompress point cloud data for {image_id}: {decompress_error}")
                        return False
                else:
                    logging.error(f"Failed to download point cloud {image_id}: {response.status}")
                    return False
        except Exception as e:
            logging.error(f"Error downloading point cloud {image_id}: {e}")
            return False


class MapillaryDataDownloader:
    """Main class orchestrating the download of Mapillary data with enhanced image type support."""
    
    def __init__(self, access_token: str, bbox: List[float], output_dir: str = "mapillary_data",
                 download_panoramic: bool = True, download_standard: bool = True,
                 resolutions: List[str] = None):
        """
        Initialize the main downloader.
        
        Args:
            access_token: Mapillary access token
            bbox: Bounding box as [west, south, east, north]
            output_dir: Output directory for downloaded data
            download_panoramic: Whether to download panoramic (360°) images
            download_standard: Whether to download standard perspective images
            resolutions: List of resolutions to download
        """
        self.authenticator = MapillaryAuthenticator(access_token)
        self.bbox_handler = BoundingBoxHandler(bbox)
        self.output_dir = output_dir
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize downloaders with enhanced settings
        self.image_downloader = ImageDownloader(
            self.authenticator, output_dir, 
            download_panoramic, download_standard, resolutions
        )
        self.pointcloud_downloader = PointCloudDownloader(self.authenticator, output_dir)
        self.vector_tile_processor = VectorTileProcessor(self.authenticator)
        
        # Setup logging
        self._setup_logging()
    
    def _setup_logging(self):
        """Setup logging configuration."""
        log_file = os.path.join(self.output_dir, "download.log")
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
    
    async def download_all(self, download_images: bool = True, download_pointclouds: bool = True,
                          max_images: int = 2000) -> Dict[str, int]:
        """
        Download all data for the specified bounding box with enhanced image type support.
        
        Args:
            download_images: Whether to download images
            download_pointclouds: Whether to download point clouds
            max_images: Maximum number of images to download
            
        Returns:
            Dictionary with download statistics
        """
        stats = {
            'panoramic_downloaded': 0,
            'standard_downloaded': 0,
            'total_images_downloaded': 0,
            'pointclouds_downloaded': 0,
            'tiles_processed': 0,
            'errors': 0
        }
        
        logging.info(f"Starting enhanced download for bounding box: {self.bbox_handler.to_string()}")
        logging.info(f"Image types: Panoramic={self.image_downloader.download_panoramic}, Standard={self.image_downloader.download_standard}")
        logging.info(f"Resolutions: {self.image_downloader.resolutions}")
        
        connector = aiohttp.TCPConnector(limit=50)
        timeout = aiohttp.ClientTimeout(total=300)
        
        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            if download_images:
                # Use the enhanced image downloader
                logging.info("Starting enhanced image download...")
                image_stats = await self.image_downloader.download_all_image_types(
                    session, self.bbox_handler, max_images)
                
                # Update stats
                stats['panoramic_downloaded'] = image_stats['panoramic_downloaded']
                stats['standard_downloaded'] = image_stats['standard_downloaded']
                stats['total_images_downloaded'] = image_stats['panoramic_downloaded'] + image_stats['standard_downloaded']
                stats['errors'] += image_stats['errors']
                
                logging.info(f"Image download completed: {stats['panoramic_downloaded']} panoramic, {stats['standard_downloaded']} standard")
                
                # Download point clouds if requested
                if download_pointclouds and stats['total_images_downloaded'] > 0:
                    logging.info("Downloading point clouds for images...")
                    
                    # Get all image IDs from both panoramic and standard downloads
                    await self._download_pointclouds_for_images(session, stats)
                    
            else:
                logging.info("Image download skipped")
        
        # Save enhanced download summary
        summary = {
            'bbox': self.bbox_handler.to_string(),
            'download_time': datetime.now().isoformat(),
            'statistics': stats,
            'settings': {
                'download_images': download_images,
                'download_pointclouds': download_pointclouds,
                'download_panoramic': self.image_downloader.download_panoramic,
                'download_standard': self.image_downloader.download_standard,
                'resolutions': self.image_downloader.resolutions,
                'max_images': max_images
            },
            'directory_structure': {
                'images': {
                    'panoramic': os.path.join(self.output_dir, 'images', 'panoramic'),
                    'standard': os.path.join(self.output_dir, 'images', 'standard')
                },
                'pointclouds': os.path.join(self.output_dir, 'pointclouds')
            }
        }
        
        summary_path = os.path.join(self.output_dir, "download_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logging.info(f"Enhanced download completed. Stats: {stats}")
        return stats
    
    async def _download_pointclouds_for_images(self, session: aiohttp.ClientSession, stats: Dict[str, int]):
        """Download point clouds for all downloaded images."""
        image_dirs = [
            os.path.join(self.output_dir, 'images', 'panoramic'),
            os.path.join(self.output_dir, 'images', 'standard')
        ]
        
        for image_dir in image_dirs:
            if not os.path.exists(image_dir):
                continue
                
            # Walk through all subdirectories
            for root, dirs, files in os.walk(image_dir):
                for file in files:
                    if file.endswith('_metadata.json'):
                        try:
                            metadata_path = os.path.join(root, file)
                            with open(metadata_path, 'r') as f:
                                metadata = json.load(f)
                            
                            image_id = metadata.get('id')
                            if image_id:
                                pointcloud_url = await self.pointcloud_downloader.get_image_pointcloud(
                                    session, image_id)
                                
                                if pointcloud_url:
                                    pc_success = await self.pointcloud_downloader.download_pointcloud(
                                        session, image_id, pointcloud_url)
                                    if pc_success:
                                        stats['pointclouds_downloaded'] += 1
                        except Exception as e:
                            logging.error(f"Error processing point cloud for {file}: {e}")
                            stats['errors'] += 1


def parse_bbox(bbox_str: str) -> List[float]:
    """Parse bounding box string to list of floats."""
    try:
        coords = [float(x.strip()) for x in bbox_str.split(',')]
        if len(coords) != 4:
            raise ValueError("Bounding box must have exactly 4 coordinates")
        return coords
    except ValueError as e:
        raise ValueError(f"Invalid bounding box format: {e}")


def main():
    """Main function to run the enhanced downloader."""
    parser = argparse.ArgumentParser(
        description='Download Mapillary street view images and point cloud data with enhanced type support',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --bbox "-80.134,25.773,-80.126,25.789" --token "MLY|your_token"
  %(prog)s --bbox "2.294,48.857,2.296,48.859" --token "MLY|your_token" --output "paris_data"
  %(prog)s --bbox "2.294,48.857,2.296,48.859" --token "MLY|your_token" --no-pointclouds
  %(prog)s --bbox "2.294,48.857,2.296,48.859" --token "MLY|your_token" --panoramic-only
  %(prog)s --bbox "2.294,48.857,2.296,48.859" --token "MLY|your_token" --all-resolutions
        """
    )
    
    parser.add_argument('--bbox', '--bounding-box', required=True,
                       help='Bounding box as "west,south,east,north"')
    parser.add_argument('--token', '--access-token', required=True,
                       help='Mapillary access token')
    parser.add_argument('--output', '-o', default='mapillary_data',
                       help='Output directory (default: mapillary_data)')
    parser.add_argument('--max-images', type=int, default=2000,
                       help='Maximum number of images to download (default: 2000)')
    parser.add_argument('--no-images', action='store_true',
                       help='Skip image download')
    parser.add_argument('--no-pointclouds', action='store_true',
                       help='Skip point cloud download')
    
    # Enhanced image type options
    parser.add_argument('--no-panoramic', action='store_true',
                       help='Skip panoramic (360°) images')
    parser.add_argument('--no-standard', action='store_true',
                       help='Skip standard perspective images')
    parser.add_argument('--panoramic-only', action='store_true',
                       help='Download only panoramic images')
    parser.add_argument('--standard-only', action='store_true',
                       help='Download only standard images')
    
    # Resolution options
    parser.add_argument('--all-resolutions', action='store_true',
                       help='Download all available resolutions (256, 1024, 2048, original)')
    parser.add_argument('--high-res-only', action='store_true',
                       help='Download only 2048px and original resolution')
    parser.add_argument('--low-res-only', action='store_true',
                       help='Download only 256px and 1024px resolution')
    
    args = parser.parse_args()
    
    try:
        # Parse bounding box
        bbox = parse_bbox(args.bbox)
        
        # Determine image type preferences
        download_panoramic = True
        download_standard = True
        
        if args.panoramic_only:
            download_standard = False
        elif args.standard_only:
            download_panoramic = False
        elif args.no_panoramic:
            download_panoramic = False
        elif args.no_standard:
            download_standard = False
        
        # Determine resolution preferences
        if args.all_resolutions:
            resolutions = ['thumb_256_url', 'thumb_1024_url', 'thumb_2048_url', 'thumb_original_url']
        elif args.high_res_only:
            resolutions = ['thumb_2048_url', 'thumb_original_url']
        elif args.low_res_only:
            resolutions = ['thumb_256_url', 'thumb_1024_url']
        else:
            resolutions = ['thumb_2048_url', 'thumb_1024_url']  # Default
        
        # Create enhanced downloader
        downloader = MapillaryDataDownloader(
            access_token=args.token,
            bbox=bbox,
            output_dir=args.output,
            download_panoramic=download_panoramic,
            download_standard=download_standard,
            resolutions=resolutions
        )
        
        # Run download
        stats = asyncio.run(downloader.download_all(
            download_images=not args.no_images,
            download_pointclouds=not args.no_pointclouds,
            max_images=args.max_images
        ))
        
        print(f"\nEnhanced download completed successfully!")
        print(f"Panoramic images downloaded: {stats['panoramic_downloaded']}")
        print(f"Standard images downloaded: {stats['standard_downloaded']}")
        print(f"Total images downloaded: {stats['total_images_downloaded']}")
        print(f"Point clouds downloaded: {stats['pointclouds_downloaded']}")
        print(f"Errors: {stats['errors']}")
        print(f"Output directory: {args.output}")
        print(f"\nImage organization:")
        print(f"  - Panoramic images: {args.output}/images/panoramic/")
        print(f"  - Standard images: {args.output}/images/standard/")
        print(f"  - Point clouds: {args.output}/pointclouds/")
        
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nDownload interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)
        
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nDownload interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
