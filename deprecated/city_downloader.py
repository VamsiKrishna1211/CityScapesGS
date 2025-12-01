#!/usr/bin/env python3
"""
City-wide Mapillary Data Downloader

Downloads Mapillary data for entire cities by splitting them into smaller chunks
that respect the API's bounding box size limits.
"""

import asyncio
import argparse
import json
import math
import os
import sys
from typing import List, Tuple, Dict, Any
from datetime import datetime
import logging

from mapillary_downloader import MapillaryDataDownloader, BoundingBoxHandler


class CityGridDownloader:
    """Handles downloading entire cities by splitting them into grids."""
    
    def __init__(self, access_token: str, output_dir: str = "city_data"):
        """
        Initialize city downloader.
        
        Args:
            access_token: Mapillary access token
            output_dir: Base output directory
        """
        self.access_token = access_token
        self.output_dir = output_dir
        self.max_chunk_area = 0.001  # Conservative chunk size (square degrees)
        self.completed_chunks = set()
        self.failed_chunks = []
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Setup logging
        self._setup_logging()
        
    def _setup_logging(self):
        """Setup logging for city-wide download."""
        log_file = os.path.join(self.output_dir, "city_download.log")
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)

    def calculate_grid_size(self, bbox: List[float]) -> Tuple[int, int]:
        """
        Calculate optimal grid dimensions for a city bounding box.
        
        Args:
            bbox: City bounding box [west, south, east, north]
            
        Returns:
            Tuple of (cols, rows) for the grid
        """
        west, south, east, north = bbox
        
        # Calculate total area
        width = east - west
        height = north - south
        total_area = width * height
        
        # Calculate number of chunks needed
        chunks_needed = math.ceil(total_area / self.max_chunk_area)
        
        # Calculate grid dimensions (try to keep chunks roughly square)
        aspect_ratio = width / height
        rows = math.ceil(math.sqrt(chunks_needed / aspect_ratio))
        cols = math.ceil(chunks_needed / rows)
        
        self.logger.info(f"City area: {total_area:.6f} sq degrees")
        self.logger.info(f"Chunks needed: {chunks_needed}")
        self.logger.info(f"Grid size: {cols} x {rows}")
        
        return cols, rows

    def generate_grid_chunks(self, bbox: List[float]) -> List[Dict[str, Any]]:
        """
        Generate grid chunks for a city bounding box.
        
        Args:
            bbox: City bounding box [west, south, east, north]
            
        Returns:
            List of chunk dictionaries with bbox and metadata
        """
        west, south, east, north = bbox
        cols, rows = self.calculate_grid_size(bbox)
        
        chunks = []
        
        # Calculate chunk dimensions
        chunk_width = (east - west) / cols
        chunk_height = (north - south) / rows
        
        for row in range(rows):
            for col in range(cols):
                # Calculate chunk boundaries
                chunk_west = west + col * chunk_width
                chunk_east = west + (col + 1) * chunk_width
                chunk_south = south + row * chunk_height
                chunk_north = south + (row + 1) * chunk_height
                
                chunk_bbox = [chunk_west, chunk_south, chunk_east, chunk_north]
                chunk_area = chunk_width * chunk_height
                
                chunk = {
                    'id': f"chunk_{row}_{col}",
                    'bbox': chunk_bbox,
                    'area': chunk_area,
                    'row': row,
                    'col': col,
                    'grid_size': (cols, rows)
                }
                
                chunks.append(chunk)
        
        self.logger.info(f"Generated {len(chunks)} chunks")
        return chunks

    def save_grid_plan(self, chunks: List[Dict[str, Any]], city_name: str):
        """Save the grid plan for reference and resumability."""
        plan = {
            'city_name': city_name,
            'created_at': datetime.now().isoformat(),
            'total_chunks': len(chunks),
            'max_chunk_area': self.max_chunk_area,
            'chunks': chunks
        }
        
        plan_file = os.path.join(self.output_dir, f"{city_name}_grid_plan.json")
        with open(plan_file, 'w') as f:
            json.dump(plan, f, indent=2)
        
        self.logger.info(f"Grid plan saved to {plan_file}")

    def load_progress(self, city_name: str) -> List[str]:
        """Load completed chunks from previous runs."""
        progress_file = os.path.join(self.output_dir, f"{city_name}_progress.txt")
        completed = []
        
        if os.path.exists(progress_file):
            try:
                with open(progress_file, 'r') as f:
                    completed = [line.strip() for line in f if line.strip()]
                self.logger.info(f"Resumed: {len(completed)} chunks already completed")
            except Exception as e:
                self.logger.error(f"Error loading progress: {e}")
        
        return completed

    def save_progress(self, chunk_id: str, city_name: str):
        """Save progress after completing a chunk."""
        progress_file = os.path.join(self.output_dir, f"{city_name}_progress.txt")
        with open(progress_file, 'a') as f:
            f.write(f"{chunk_id}\n")

    async def download_chunk(self, chunk: Dict[str, Any], city_name: str, 
                           download_images: bool = True, download_pointclouds: bool = True,
                           max_images_per_chunk: int = 2000) -> Dict[str, Any]:
        """
        Download data for a single chunk.
        
        Args:
            chunk: Chunk dictionary with bbox and metadata
            city_name: Name of the city
            download_images: Whether to download images
            download_pointclouds: Whether to download point clouds
            max_images_per_chunk: Max images per chunk
            
        Returns:
            Download statistics for the chunk
        """
        chunk_id = chunk['id']
        chunk_bbox = chunk['bbox']
        
        # Create chunk-specific output directory
        chunk_dir = os.path.join(self.output_dir, city_name, chunk_id)
        
        self.logger.info(f"Starting download for {chunk_id}")
        self.logger.info(f"Chunk bbox: {chunk_bbox}")
        self.logger.info(f"Chunk area: {chunk['area']:.8f} sq degrees")
        
        try:
            # Create downloader for this chunk
            downloader = MapillaryDataDownloader(
                access_token=self.access_token,
                bbox=chunk_bbox,
                output_dir=chunk_dir
            )
            
            # Download the chunk
            stats = await downloader.download_all(
                download_images=download_images,
                download_pointclouds=download_pointclouds,
                max_images=max_images_per_chunk
            )
            
            # Add chunk metadata to stats
            stats['chunk_id'] = chunk_id
            stats['chunk_bbox'] = chunk_bbox
            stats['chunk_area'] = chunk['area']
            
            # Save progress
            self.save_progress(chunk_id, city_name)
            
            self.logger.info(f"Completed {chunk_id}: {stats['images_downloaded']} images, "
                           f"{stats['pointclouds_downloaded']} point clouds")
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Failed to download {chunk_id}: {e}")
            self.failed_chunks.append(chunk_id)
            return {
                'chunk_id': chunk_id,
                'error': str(e),
                'images_downloaded': 0,
                'pointclouds_downloaded': 0,
                'errors': 1
            }

    async def download_city(self, bbox: List[float], city_name: str,
                          download_images: bool = True, download_pointclouds: bool = True,
                          max_images_per_chunk: int = 2000, 
                          max_concurrent_chunks: int = 1) -> Dict[str, Any]:
        """
        Download an entire city by processing chunks.
        
        Args:
            bbox: City bounding box [west, south, east, north]
            city_name: Name of the city for organization
            download_images: Whether to download images
            download_pointclouds: Whether to download point clouds
            max_images_per_chunk: Max images per chunk
            max_concurrent_chunks: Number of chunks to process concurrently
            
        Returns:
            Overall download statistics
        """
        self.logger.info(f"Starting city download: {city_name}")
        self.logger.info(f"City bbox: {bbox}")
        
        # Generate grid chunks
        chunks = self.generate_grid_chunks(bbox)
        
        # Save grid plan
        self.save_grid_plan(chunks, city_name)
        
        # Load previous progress
        completed_chunk_ids = self.load_progress(city_name)
        self.completed_chunks = set(completed_chunk_ids)
        
        # Filter out completed chunks
        remaining_chunks = [c for c in chunks if c['id'] not in self.completed_chunks]
        
        self.logger.info(f"Chunks to process: {len(remaining_chunks)} of {len(chunks)}")
        
        # Process chunks with limited concurrency
        all_stats = []
        
        # Process chunks in batches to respect rate limits
        for i in range(0, len(remaining_chunks), max_concurrent_chunks):
            batch = remaining_chunks[i:i + max_concurrent_chunks]
            
            self.logger.info(f"Processing batch {i//max_concurrent_chunks + 1} "
                           f"({len(batch)} chunks)")
            
            # Process batch
            if max_concurrent_chunks == 1:
                # Sequential processing (recommended for rate limiting)
                for chunk in batch:
                    stats = await self.download_chunk(
                        chunk, city_name, download_images, 
                        download_pointclouds, max_images_per_chunk
                    )
                    all_stats.append(stats)
            else:
                # Concurrent processing (use with caution)
                tasks = [
                    self.download_chunk(
                        chunk, city_name, download_images,
                        download_pointclouds, max_images_per_chunk
                    )
                    for chunk in batch
                ]
                batch_stats = await asyncio.gather(*tasks)
                all_stats.extend(batch_stats)
        
        # Calculate overall statistics
        overall_stats = {
            'city_name': city_name,
            'total_chunks': len(chunks),
            'completed_chunks': len(chunks) - len(remaining_chunks) + len(all_stats),
            'failed_chunks': len(self.failed_chunks),
            'total_images': sum(s.get('images_downloaded', 0) for s in all_stats),
            'total_pointclouds': sum(s.get('pointclouds_downloaded', 0) for s in all_stats),
            'total_errors': sum(s.get('errors', 0) for s in all_stats),
            'city_bbox': bbox,
            'download_time': datetime.now().isoformat(),
            'chunk_stats': all_stats
        }
        
        # Save final summary
        summary_file = os.path.join(self.output_dir, f"{city_name}_final_summary.json")
        with open(summary_file, 'w') as f:
            json.dump(overall_stats, f, indent=2)
        
        self.logger.info(f"City download completed!")
        self.logger.info(f"Total images: {overall_stats['total_images']}")
        self.logger.info(f"Total point clouds: {overall_stats['total_pointclouds']}")
        self.logger.info(f"Failed chunks: {len(self.failed_chunks)}")
        
        return overall_stats


# Predefined city bounding boxes (examples)
CITY_BBOXES = {
    'manhattan': [-74.0479, 40.7000, -73.9441, 40.7831],  # Manhattan, NYC
    'central_paris': [2.2241, 48.8155, 2.4697, 48.9022],  # Central Paris
    'central_london': [-0.2817, 51.4673, -0.0076, 51.5673],  # Central London
    'downtown_miami': [-80.2634, 25.7376, -80.1408, 25.8032],  # Downtown Miami
    'downtown_sf': [-122.5174, 37.7549, -122.3816, 37.8049],  # Downtown San Francisco
    'downtown_tokyo': [139.6503, 35.6295, 139.7731, 35.7295],  # Central Tokyo
    'singapore_central': [103.7930, 1.2630, 103.8730, 1.3430],  # Central Singapore
}


def main():
    """Main function for city-wide downloading."""
    parser = argparse.ArgumentParser(
        description='Download Mapillary data for entire cities',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Available predefined cities:
{chr(10).join(f"  {name}: {bbox}" for name, bbox in CITY_BBOXES.items())}

Examples:
  %(prog)s --city manhattan --token "MLY|your_token"
  %(prog)s --bbox "-74.05,40.70,-73.94,40.78" --name "manhattan" --token "MLY|your_token"
  %(prog)s --city central_paris --token "MLY|your_token" --no-pointclouds
        """
    )
    
    # City selection
    city_group = parser.add_mutually_exclusive_group(required=True)
    city_group.add_argument('--city', choices=list(CITY_BBOXES.keys()),
                           help='Predefined city to download')
    city_group.add_argument('--bbox', help='Custom bounding box as "west,south,east,north"')
    
    # Required arguments
    parser.add_argument('--token', required=True, help='Mapillary access token')
    
    # Optional arguments
    parser.add_argument('--name', help='City name (required if using --bbox)')
    parser.add_argument('--output', '-o', default='city_data',
                       help='Output directory (default: city_data)')
    parser.add_argument('--max-images-per-chunk', type=int, default=2000,
                       help='Max images per chunk (default: 2000)')
    parser.add_argument('--max-concurrent-chunks', type=int, default=1,
                       help='Max concurrent chunks (default: 1, recommended)')
    parser.add_argument('--chunk-size', type=float, default=0.001,
                       help='Max chunk area in square degrees (default: 0.001)')
    parser.add_argument('--no-images', action='store_true',
                       help='Skip image download')
    parser.add_argument('--no-pointclouds', action='store_true',
                       help='Skip point cloud download')
    
    args = parser.parse_args()
    
    try:
        # Determine city name and bbox
        if args.city:
            city_name = args.city
            bbox = CITY_BBOXES[args.city]
        else:
            if not args.name:
                print("Error: --name is required when using --bbox")
                sys.exit(1)
            city_name = args.name
            bbox = [float(x.strip()) for x in args.bbox.split(',')]
            if len(bbox) != 4:
                print("Error: bbox must have exactly 4 coordinates")
                sys.exit(1)
        
        print(f"City: {city_name}")
        print(f"Bounding box: {bbox}")
        
        # Calculate estimated chunks
        west, south, east, north = bbox
        area = (east - west) * (north - south)
        estimated_chunks = math.ceil(area / args.chunk_size)
        
        print(f"Estimated chunks: {estimated_chunks}")
        print(f"Estimated area: {area:.6f} square degrees")
        
        if estimated_chunks > 1000:
            response = input(f"This will create {estimated_chunks} chunks. Continue? (y/N): ")
            if response.lower() != 'y':
                print("Cancelled.")
                sys.exit(0)
        
        # Create downloader
        downloader = CityGridDownloader(
            access_token=args.token,
            output_dir=args.output
        )
        downloader.max_chunk_area = args.chunk_size
        
        # Start download
        print(f"\nStarting download for {city_name}...")
        stats = asyncio.run(downloader.download_city(
            bbox=bbox,
            city_name=city_name,
            download_images=not args.no_images,
            download_pointclouds=not args.no_pointclouds,
            max_images_per_chunk=args.max_images_per_chunk,
            max_concurrent_chunks=args.max_concurrent_chunks
        ))
        
        print(f"\n🎉 Download completed!")
        print(f"📊 Final Statistics:")
        print(f"   Total chunks: {stats['total_chunks']}")
        print(f"   Completed chunks: {stats['completed_chunks']}")
        print(f"   Failed chunks: {stats['failed_chunks']}")
        print(f"   Total images: {stats['total_images']}")
        print(f"   Total point clouds: {stats['total_pointclouds']}")
        print(f"   Output directory: {args.output}")
        
    except KeyboardInterrupt:
        print("\nDownload interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
