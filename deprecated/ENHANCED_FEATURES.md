# Enhanced Mapillary Data Downloader - Image Type Support

## 🚨 Important Note About Aerial Images

**Mapillary does NOT provide aerial/satellite images.** Mapillary is specifically designed for **street-level imagery** captured by cameras on vehicles, bicycles, or carried by pedestrians. 

However, the enhanced downloader now supports:

## 🌍 Enhanced Image Types

### 1. **Panoramic Images (360°)**
- **What**: 360-degree street-level panoramic images
- **Coverage**: Much wider field of view than standard images
- **Camera Types**: Equirectangular (spherical) projection
- **Use Case**: Better for understanding street layout and surroundings

### 2. **Standard Perspective Images**
- **What**: Regular street-level photos with normal field of view
- **Camera Types**: Perspective, fisheye projections
- **Use Case**: Detailed view of specific street features

### 3. **Multiple Resolutions**
- **256px**: Thumbnail for quick preview
- **1024px**: Medium resolution for analysis
- **2048px**: High resolution for detailed work
- **Original**: Full resolution as uploaded (when available)

## 🎯 Enhanced Usage Examples

### Download Both Image Types (Default)
```bash
python3 mapillary_downloader.py --bbox "-80.134,25.773,-80.126,25.789" --token "MLY|your_token"
```

### Download Only Panoramic Images (Best Coverage)
```bash
python3 mapillary_downloader.py --bbox "-80.134,25.773,-80.126,25.789" --token "MLY|your_token" --panoramic-only
```

### Download Only Standard Images
```bash
python3 mapillary_downloader.py --bbox "-80.134,25.773,-80.126,25.789" --token "MLY|your_token" --standard-only
```

### Download All Available Resolutions
```bash
python3 mapillary_downloader.py --bbox "-80.134,25.773,-80.126,25.789" --token "MLY|your_token" --all-resolutions
```

### Download Only High Resolution Images
```bash
python3 mapillary_downloader.py --bbox "-80.134,25.773,-80.126,25.789" --token "MLY|your_token" --high-res-only
```

## 📁 Enhanced Output Structure

```
output_directory/
├── images/
│   ├── panoramic/              # 360° panoramic images
│   │   ├── perspective/        # By camera type
│   │   ├── fisheye/
│   │   └── equirectangular/
│   │       ├── {image_id}_2048.jpg     # Different resolutions
│   │       ├── {image_id}_1024.jpg
│   │       ├── {image_id}_256.jpg
│   │       ├── {image_id}_orig.jpg
│   │       └── {image_id}_metadata.json # Enhanced metadata
│   └── standard/               # Standard perspective images
│       ├── perspective/
│       ├── fisheye/
│       └── equirectangular/
├── pointclouds/               # 3D point cloud data
│   └── {image_id}_pointcloud.json
├── download.log              # Detailed log
└── download_summary.json     # Enhanced summary with type statistics
```

## 🔧 New Command Line Options

### Image Type Control
- `--panoramic-only`: Download only 360° panoramic images
- `--standard-only`: Download only standard perspective images  
- `--no-panoramic`: Skip panoramic images
- `--no-standard`: Skip standard images

### Resolution Control
- `--all-resolutions`: Download all available resolutions (256, 1024, 2048, original)
- `--high-res-only`: Download only 2048px and original resolution
- `--low-res-only`: Download only 256px and 1024px resolution

## 🌆 For Aerial-Like Coverage

While Mapillary doesn't provide aerial images, for the best street-level coverage of a city:

1. **Use panoramic images**: `--panoramic-only`
2. **Use high resolution**: `--high-res-only` 
3. **Use city downloader**: For large areas, use `city_downloader.py`

### Example for Maximum Coverage
```bash
# Download panoramic images in high resolution for best coverage
python3 mapillary_downloader.py \
  --bbox "-80.134,25.773,-80.126,25.789" \
  --token "MLY|your_token" \
  --panoramic-only \
  --high-res-only \
  --output "miami_panoramic"
```

## 📊 Enhanced Metadata

Each image now includes enhanced metadata:

```json
{
  "id": "image_id",
  "is_pano": true,
  "camera_type": "equirectangular",
  "width": 4096,
  "height": 2048,
  "computed_geometry": {...},
  "captured_at": 1634567890000,
  "creator": {...},
  "download_info": {
    "downloaded_at": "2025-10-25T17:30:00Z",
    "image_type": "panoramic",
    "camera_type": "equirectangular",
    "resolutions_downloaded": ["thumb_2048_url", "thumb_original_url"],
    "directory": "/path/to/images/panoramic/equirectangular"
  }
}
```

## 🗺️ Alternative for Aerial Images

If you need aerial/satellite imagery, consider these alternatives:

1. **Google Earth Engine**: Satellite imagery API
2. **Mapbox Satellite**: Satellite tiles API  
3. **Bing Maps**: Aerial imagery API
4. **OpenStreetMap**: With aerial tile layers
5. **NASA Earthdata**: Free satellite imagery

The Mapillary data is excellent for street-level 3D reconstruction and understanding urban environments from ground perspective, but it's not designed for aerial views.
