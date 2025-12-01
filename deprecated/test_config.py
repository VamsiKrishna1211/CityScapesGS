#!/usr/bin/env python3
"""
Test script for Mapillary downloader to validate configuration before running full download.
"""

import asyncio
import aiohttp
import argparse
import sys
from mapillary_downloader import MapillaryAuthenticator, BoundingBoxHandler


async def test_token(token: str) -> bool:
    """Test if the provided token is valid."""
    print("Testing access token...")
    
    authenticator = MapillaryAuthenticator(token)
    headers = authenticator.get_headers()
    
    # Test with a simple API call
    test_url = "https://graph.mapillary.com/images"
    params = {
        'bbox': '-80.134,25.773,-80.126,25.789',  # Small test area
        'limit': 1,
        'fields': 'id'
    }
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(test_url, params=params, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    print(f"✓ Token is valid! Found {len(data.get('data', []))} test images.")
                    return True
                elif response.status == 401:
                    print("✗ Token is invalid or expired.")
                    return False
                elif response.status == 403:
                    print("✗ Token doesn't have required permissions.")
                    return False
                else:
                    print(f"✗ Unexpected response: {response.status}")
                    return False
    except Exception as e:
        print(f"✗ Error testing token: {e}")
        return False


def test_bbox(bbox_str: str) -> bool:
    """Test if the provided bounding box is valid."""
    print(f"Testing bounding box: {bbox_str}")
    
    try:
        # Parse coordinates
        coords = [float(x.strip()) for x in bbox_str.split(',')]
        if len(coords) != 4:
            print("✗ Bounding box must have exactly 4 coordinates")
            return False
        
        # Create handler to validate
        bbox_handler = BoundingBoxHandler(coords)
        
        area = (bbox_handler.east - bbox_handler.west) * (bbox_handler.north - bbox_handler.south)
        
        print(f"✓ Bounding box is valid!")
        print(f"  - West: {bbox_handler.west}")
        print(f"  - South: {bbox_handler.south}")
        print(f"  - East: {bbox_handler.east}")
        print(f"  - North: {bbox_handler.north}")
        print(f"  - Area: {area:.8f} square degrees")
        
        if area > 0.01:
            print(f"⚠ Warning: Area ({area:.6f}) exceeds Mapillary limit of 0.01 square degrees")
            return False
        elif area > 0.001:
            print(f"⚠ Warning: Large area ({area:.6f}). Consider using smaller bounding box.")
            
        return True
        
    except ValueError as e:
        print(f"✗ Invalid bounding box: {e}")
        return False
    except Exception as e:
        print(f"✗ Error validating bounding box: {e}")
        return False


async def estimate_images(token: str, bbox_str: str) -> int:
    """Estimate number of images in the bounding box."""
    print("Estimating number of images...")
    
    try:
        coords = [float(x.strip()) for x in bbox_str.split(',')]
        bbox_handler = BoundingBoxHandler(coords)
        authenticator = MapillaryAuthenticator(token)
        
        url = "https://graph.mapillary.com/images"
        params = {
            'bbox': bbox_handler.to_string(),
            'limit': 10,  # Small sample
            'fields': 'id'
        }
        headers = authenticator.get_headers()
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    sample_count = len(data.get('data', []))
                    
                    if sample_count == 0:
                        print("✗ No images found in this bounding box")
                        return 0
                    elif sample_count < 10:
                        print(f"✓ Found {sample_count} images (total)")
                        return sample_count
                    else:
                        print(f"✓ Found 10+ images (sampled). Actual count may be much higher.")
                        print("  Run the full downloader to get exact count.")
                        return sample_count
                else:
                    print(f"✗ Failed to estimate images: HTTP {response.status}")
                    return -1
                    
    except Exception as e:
        print(f"✗ Error estimating images: {e}")
        return -1


async def main():
    """Main test function."""
    parser = argparse.ArgumentParser(description='Test Mapillary downloader configuration')
    parser.add_argument('--bbox', required=True, help='Bounding box as "west,south,east,north"')
    parser.add_argument('--token', required=True, help='Mapillary access token')
    
    args = parser.parse_args()
    
    print("Mapillary Downloader Configuration Test")
    print("=" * 40)
    
    # Test bounding box
    bbox_valid = test_bbox(args.bbox)
    print()
    
    # Test token
    token_valid = await test_token(args.token)
    print()
    
    # If both are valid, estimate images
    if bbox_valid and token_valid:
        await estimate_images(args.token, args.bbox)
        print()
        print("✓ Configuration looks good! You can proceed with the download.")
        print(f"Command to run:")
        print(f'python mapillary_downloader.py --bbox "{args.bbox}" --token "{args.token}"')
    else:
        print("✗ Configuration has issues. Please fix them before running the downloader.")
        sys.exit(1)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)
