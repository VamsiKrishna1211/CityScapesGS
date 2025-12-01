#!/usr/bin/env python3
"""
City Grid Calculator

Calculate how many chunks and what grid size would be needed for different cities.
"""

import argparse
import math
from typing import Dict, List, Tuple


# Major city bounding boxes (approximate urban areas)
CITY_BBOXES = {
    # North America
    'manhattan': [-74.0479, 40.7000, -73.9441, 40.7831],
    'brooklyn': [-74.0421, 40.5751, -73.8331, 40.7394],
    'los_angeles_downtown': [-118.3281, 34.0094, -118.1955, 34.0902],
    'chicago_downtown': [-87.7501, 41.8349, -87.5569, 41.9278],
    'san_francisco': [-122.5174, 37.7549, -122.3816, 37.8049],
    'miami_downtown': [-80.2634, 25.7376, -80.1408, 25.8032],
    'washington_dc': [-77.1190, 38.8016, -76.9094, 38.9955],
    'boston_downtown': [-71.0992, 42.3193, -71.0208, 42.3707],
    
    # Europe
    'central_paris': [2.2241, 48.8155, 2.4697, 48.9022],
    'central_london': [-0.2817, 51.4673, -0.0076, 51.5673],
    'central_berlin': [13.2884, 52.4426, 13.5009, 52.5887],
    'central_rome': [12.4203, 41.8527, 12.5664, 41.9378],
    'central_madrid': [-3.7492, 40.3516, -3.6395, 40.4739],
    'amsterdam_center': [4.8319, 52.3308, 4.9417, 52.3924],
    'vienna_center': [16.2970, 48.1671, 16.4306, 48.2530],
    
    # Asia
    'tokyo_central': [139.6503, 35.6295, 139.7731, 35.7295],
    'singapore_central': [103.7930, 1.2630, 103.8730, 1.3430],
    'hong_kong_central': [114.1277, 22.2439, 114.2288, 22.3193],
    'seoul_gangnam': [126.9536, 37.4449, 127.0896, 37.5664],
    'mumbai_south': [72.7757, 18.8760, 72.8777, 19.0176],
    'bangkok_central': [100.4607, 13.6519, 100.5984, 13.8229],
    
    # Example larger areas (will need many chunks)
    'manhattan_extended': [-74.0591, 40.6892, -73.9326, 40.7967],
    'greater_paris': [2.1587, 48.7693, 2.5284, 48.9441],
    'london_zone1': [-0.3218, 51.4346, 0.0505, 51.5673],
}


def calculate_grid_info(bbox: List[float], max_chunk_area: float = 0.001) -> Dict:
    """Calculate grid information for a bounding box."""
    west, south, east, north = bbox
    
    # Calculate dimensions
    width = east - west
    height = north - south
    total_area = width * height
    
    # Calculate chunks needed
    chunks_needed = math.ceil(total_area / max_chunk_area)
    
    # Calculate grid dimensions
    aspect_ratio = width / height
    rows = math.ceil(math.sqrt(chunks_needed / aspect_ratio))
    cols = math.ceil(chunks_needed / rows)
    
    # Calculate actual chunk dimensions
    chunk_width = width / cols
    chunk_height = height / rows
    actual_chunk_area = chunk_width * chunk_height
    
    return {
        'bbox': bbox,
        'width_degrees': width,
        'height_degrees': height,
        'total_area': total_area,
        'chunks_needed': chunks_needed,
        'grid_cols': cols,
        'grid_rows': rows,
        'chunk_width': chunk_width,
        'chunk_height': chunk_height,
        'actual_chunk_area': actual_chunk_area,
        'width_km': width * 111.32,  # Approximate km per degree at equator
        'height_km': height * 111.32,
        'total_area_km2': total_area * (111.32 ** 2),
    }


def print_city_analysis(city_name: str, info: Dict):
    """Print analysis for a city."""
    print(f"\n📍 {city_name.upper()}")
    print(f"   📐 Dimensions: {info['width_degrees']:.4f} × {info['height_degrees']:.4f} degrees")
    print(f"   📏 Approximate size: {info['width_km']:.1f} × {info['height_km']:.1f} km")
    print(f"   📊 Total area: {info['total_area']:.6f} sq degrees ({info['total_area_km2']:.1f} km²)")
    print(f"   🧩 Chunks needed: {info['chunks_needed']}")
    print(f"   📱 Grid size: {info['grid_cols']} × {info['grid_rows']}")
    print(f"   🔍 Chunk size: {info['actual_chunk_area']:.8f} sq degrees")
    
    # Time estimates
    print(f"   ⏱️  Estimated time:")
    print(f"      - At 1 chunk/min: {info['chunks_needed']} minutes ({info['chunks_needed']/60:.1f} hours)")
    print(f"      - At 10 chunks/hour: {info['chunks_needed']/10:.1f} hours")


def analyze_all_cities(chunk_size: float = 0.001):
    """Analyze all predefined cities."""
    print(f"🌍 CITY ANALYSIS (Max chunk size: {chunk_size} sq degrees)")
    print("=" * 70)
    
    # Group cities by size
    small_cities = []
    medium_cities = []
    large_cities = []
    
    for city_name, bbox in CITY_BBOXES.items():
        info = calculate_grid_info(bbox, chunk_size)
        
        if info['chunks_needed'] <= 10:
            small_cities.append((city_name, info))
        elif info['chunks_needed'] <= 100:
            medium_cities.append((city_name, info))
        else:
            large_cities.append((city_name, info))
    
    # Print by category
    if small_cities:
        print(f"\n🟢 SMALL AREAS (≤ 10 chunks):")
        for city_name, info in sorted(small_cities, key=lambda x: x[1]['chunks_needed']):
            print_city_analysis(city_name, info)
    
    if medium_cities:
        print(f"\n🟡 MEDIUM AREAS (11-100 chunks):")
        for city_name, info in sorted(medium_cities, key=lambda x: x[1]['chunks_needed']):
            print_city_analysis(city_name, info)
    
    if large_cities:
        print(f"\n🔴 LARGE AREAS (100+ chunks):")
        for city_name, info in sorted(large_cities, key=lambda x: x[1]['chunks_needed']):
            print_city_analysis(city_name, info)
    
    print(f"\n💡 RECOMMENDATIONS:")
    print(f"   • Start with small areas (≤ 10 chunks) for testing")
    print(f"   • Medium areas are good for comprehensive coverage")
    print(f"   • Large areas may take many hours/days to complete")
    print(f"   • Consider using smaller chunk sizes for rate limiting")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description='Analyze grid requirements for city downloads',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                                    # Analyze all cities
  %(prog)s --city manhattan                   # Analyze specific city
  %(prog)s --bbox "-74.05,40.70,-73.94,40.78" # Analyze custom area
  %(prog)s --chunk-size 0.0005               # Use smaller chunks
        """
    )
    
    parser.add_argument('--city', choices=list(CITY_BBOXES.keys()),
                       help='Analyze specific city')
    parser.add_argument('--bbox', help='Custom bounding box as "west,south,east,north"')
    parser.add_argument('--chunk-size', type=float, default=0.001,
                       help='Maximum chunk area in square degrees (default: 0.001)')
    parser.add_argument('--list-cities', action='store_true',
                       help='List all available cities')
    
    args = parser.parse_args()
    
    if args.list_cities:
        print("Available cities:")
        for city_name, bbox in sorted(CITY_BBOXES.items()):
            info = calculate_grid_info(bbox, args.chunk_size)
            print(f"  {city_name:20} {bbox} ({info['chunks_needed']:3d} chunks)")
        return
    
    if args.city:
        # Analyze specific city
        bbox = CITY_BBOXES[args.city]
        info = calculate_grid_info(bbox, args.chunk_size)
        print_city_analysis(args.city, info)
        
        print(f"\n🚀 COMMAND TO RUN:")
        print(f"python3 city_downloader.py --city {args.city} --token \"YOUR_TOKEN\"")
        
    elif args.bbox:
        # Analyze custom bbox
        try:
            bbox = [float(x.strip()) for x in args.bbox.split(',')]
            if len(bbox) != 4:
                raise ValueError("Must have 4 coordinates")
            
            info = calculate_grid_info(bbox, args.chunk_size)
            print_city_analysis("Custom Area", info)
            
            print(f"\n🚀 COMMAND TO RUN:")
            print(f'python3 city_downloader.py --bbox "{args.bbox}" --name "custom_area" --token "YOUR_TOKEN"')
            
        except ValueError as e:
            print(f"Error: Invalid bounding box: {e}")
            return
    else:
        # Analyze all cities
        analyze_all_cities(args.chunk_size)


if __name__ == "__main__":
    main()
