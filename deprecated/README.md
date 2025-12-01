# Mapillary Data Downloader

A modular Python script for downloading street view images and point cloud data from the Mapillary API. Supports both small area downloads and entire city downloads through intelligent grid-based chunking.

## Features

### Core Features
- **Modular Design**: Clean class-based architecture with separate components for:
  - Authentication management
  - Bounding box handling
  - Image downloading
  - Point cloud downloading
  - Vector tile processing

- **Async Operations**: High-performance asynchronous downloading with proper rate limiting
- **Progress Tracking**: Real-time progress bars using tqdm
- **Error Handling**: Robust error handling with retries and logging
- **Resumable Downloads**: Skip already downloaded content
- **Metadata Storage**: Save image metadata alongside downloaded images

### City-Wide Downloading
- **Grid-Based Chunking**: Automatically splits large areas into API-compliant chunks
- **Resumable City Downloads**: Continue interrupted city downloads
- **Predefined Cities**: Built-in bounding boxes for major world cities
- **Grid Analysis**: Preview chunk requirements before downloading
- **Progress Tracking**: City-level and chunk-level progress monitoring

## Installation

1. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

```bash
python mapillary_downloader.py --bbox "west,south,east,north" --token "your_access_token"
```

### Examples

**Download data for a small area in Miami:**
```bash
python mapillary_downloader.py --bbox "-80.134,25.773,-80.126,25.789" --token "MLY|your_token_here"
```

**Download data for a small area in Paris:**
```bash
python mapillary_downloader.py --bbox "2.294,48.857,2.296,48.859" --token "MLY|your_token_here" --output "paris_data"
```

**Download only images (skip point clouds):**
```bash
python mapillary_downloader.py --bbox "2.294,48.857,2.296,48.859" --token "MLY|your_token_here" --no-pointclouds
```

**Limit the number of images:**
```bash
python mapillary_downloader.py --bbox "2.294,48.857,2.296,48.859" --token "MLY|your_token_here" --max-images 500
```

### City-Wide Downloads

For downloading entire cities (which exceed the 0.01 square degree API limit), use the city downloader:

**Analyze a city first:**
```bash
python city_analyzer.py --city manhattan  # See chunk requirements
python city_analyzer.py                   # Analyze all predefined cities
```

**Download a predefined city:**
```bash
python city_downloader.py --city manhattan --token "MLY|your_token_here"
python city_downloader.py --city central_paris --token "MLY|your_token_here" --no-pointclouds
```

**Download a custom large area:**
```bash
python city_downloader.py --bbox "-74.05,40.70,-73.94,40.78" --name "custom_manhattan" --token "MLY|your_token_here"
```

**Available predefined cities:**
- `manhattan`, `brooklyn`, `los_angeles_downtown`, `chicago_downtown`, `san_francisco`
- `central_paris`, `central_london`, `central_berlin`, `central_rome`, `amsterdam_center`
- `tokyo_central`, `singapore_central`, `hong_kong_central`, `seoul_gangnam`
- And many more (see `city_analyzer.py --list-cities`)

### Command Line Arguments

#### Small Area Downloader (`mapillary_downloader.py`)
- `--bbox` or `--bounding-box`: **Required**. Bounding box coordinates as "west,south,east,north"
- `--token` or `--access-token`: **Required**. Your Mapillary access token
- `--output` or `-o`: Output directory (default: "mapillary_data")
- `--max-images`: Maximum number of images to download (default: 2000)
- `--no-images`: Skip image download
- `--no-pointclouds`: Skip point cloud download

#### City Downloader (`city_downloader.py`)
- `--city`: Predefined city name (e.g., `manhattan`, `central_paris`)
- `--bbox` and `--name`: Custom area bounding box and name
- `--token`: **Required**. Your Mapillary access token
- `--output`: Output directory (default: "city_data")
- `--chunk-size`: Max chunk area in square degrees (default: 0.001)
- `--max-images-per-chunk`: Max images per chunk (default: 2000)
- `--max-concurrent-chunks`: Concurrent chunk processing (default: 1)

## Getting a Mapillary Access Token

1. Go to [Mapillary Developer Dashboard](https://mapillary.com/dashboard/developers)
2. Create a new application
3. Copy your Client Token (starts with "MLY|")

## API Limits and Constraints

### Bounding Box Constraints
- The bounding box area must be smaller than **0.01 square degrees** (Mapillary API limitation)
- Coordinates must be in the format: `west,south,east,north`
- Example: `"-80.134,25.773,-80.126,25.789"` (approximately 1km²)

### Rate Limits
- **Entity API**: 60,000 requests per minute per app
- **Search API**: 10,000 requests per minute per app  
- **Tiles API**: 50,000 requests per day

## Output Structure

The script creates the following directory structure:

```
output_directory/
├── images/                    # Downloaded street view images
│   ├── {image_id}.jpg        # Image files
│   └── {image_id}_metadata.json  # Image metadata
├── pointclouds/              # Point cloud data (if available)
│   └── {image_id}_pointcloud.json
├── download.log              # Download log
└── download_summary.json     # Summary of download session
```

## Code Architecture

### Classes

1. **MapillaryAuthenticator**: Handles API authentication and rate limiting
2. **BoundingBoxHandler**: Manages bounding box operations and validation
3. **ImageDownloader**: Downloads street view images and metadata
4. **PointCloudDownloader**: Downloads point cloud data when available
5. **VectorTileProcessor**: Processes vector tiles (placeholder implementation)
6. **MapillaryDataDownloader**: Main orchestrator class

### Key Features

- **Rate Limiting**: Automatic rate limiting to respect API limits
- **Error Recovery**: Retry logic for failed requests
- **Progress Tracking**: Visual progress bars for downloads
- **Metadata Storage**: JSON metadata saved with each image
- **Logging**: Comprehensive logging to file and console

## Troubleshooting

### Common Issues

1. **"Bounding box area exceeds limit"**
   - Make your bounding box smaller (< 0.01 square degrees)
   - Try splitting large areas into smaller chunks

2. **"Rate limit exceeded"**
   - The script automatically handles rate limiting
   - Consider using fewer concurrent requests

3. **"No images found"**
   - Check if your bounding box covers an area with Mapillary coverage
   - Verify your bounding box coordinates are correct

4. **Import errors**
   - Make sure all dependencies are installed: `pip install -r requirements.txt`

### Logging

Check the `download.log` file in your output directory for detailed error information.

## Dependencies

- `aiohttp`: Async HTTP client
- `aiofiles`: Async file operations
- `mercantile`: Web mercator tile utilities
- `tqdm`: Progress bars

Optional:
- `vt2geojson`: Vector tile processing (not implemented in current version)

## License

This script is provided as-is for educational and development purposes. Please respect Mapillary's API terms of service and rate limits.

## Contributing

Feel free to submit issues and enhancement requests!
