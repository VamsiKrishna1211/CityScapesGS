# CityScapeGS - 3D Urban Reconstruction Pipeline

A comprehensive end-to-end pipeline for downloading street-level imagery, preprocessing sequences, and preparing data for 3D reconstruction using Gaussian Splatting, NeRF, and traditional SfM workflows.

## 🎯 Pipeline Overview

This project provides a complete workflow from data acquisition to 3D-ready datasets:

1. **Data Acquisition** → Download Mapillary street-view imagery with rich metadata
2. **Data Conversion** → Transform to reconstruction-compatible formats (PyCuSFM, OpenSfM, COLMAP)
3. **Depth Estimation** → Generate depth maps using state-of-the-art models
4. **Object Removal** → Clean sequences by removing dynamic objects (cars, people)
5. **Motion Filtering** → Select high-quality frames based on camera motion
6. **3D Reconstruction** → Ready for Gaussian Splatting, NeRF, or SfM pipelines

---

## 📦 Components

<details>
<summary><h3>🗺️ Mapillary Data Downloader</h3></summary>

### Overview
Comprehensive downloader for Mapillary street-level imagery and metadata, specifically designed for 3D reconstruction pipelines.

### Features
- **Multi-Modal Data Collection**: Street view images, panoramic images, point clouds, object detections, and map features
- **Rich Camera Metadata**: Detailed intrinsics, extrinsics, and calibration parameters
- **Geographic Filtering**: Bounding box, time range, creator-based filtering
- **Chunked Downloads**: Automatic splitting of large areas into API-compliant chunks
- **Multi-threaded**: Concurrent downloads with progress tracking
- **Robust Error Handling**: Retry logic and graceful failure recovery

### Quick Start

```bash
# Basic download - street view images with metadata
python deprecated/mapillary_downloader.py \
    --access-token YOUR_MAPILLARY_TOKEN \
    --output-dir ./data/downtown \
    --bbox "13.400,52.519,13.401,52.520" \
    --street-view

# Comprehensive download - all data types
python deprecated/mapillary_downloader.py \
    --access-token YOUR_MAPILLARY_TOKEN \
    --output-dir ./data/comprehensive_dataset \
    --bbox "13.400,52.519,13.401,52.520" \
    --street-view --aerial-view --point-clouds --detections --map-features \
    --max-workers 8
```

### Output Structure

```
```

### Configuration Options

| Option | Description | Default |
|--------|-------------|---------|
| `--access-token` | Mapillary API access token | Required |
| `--output-dir` | Output directory path | Required |
| `--bbox` | Geographic bounding box (left,bottom,right,top) | None |
| `--street-view` | Download street view images | False |
| `--aerial-view` | Download panoramic images | False |
| `--point-clouds` | Download SfM point clouds | False |
| `--detections` | Download object detections | False |
| `--map-features` | Download map features | False |
| `--max-workers` | Concurrent download threads | 4 |
| `--chunk-size` | Max chunk size (degrees) | 0.01 |

### Getting API Token
1. Visit [Mapillary Developer Dashboard](https://mapillary.com/dashboard/developers)
2. Create a new application
3. Copy your Client Token
4. Use with `--access-token` parameter

</details>

---

<details>
<summary><h3>🔄 Dataset Format Converter</h3></summary>

### Overview
Transform Mapillary data to PyCuSFM-compatible format with GPS coordinate transformation and proper camera parameter extraction.

### Features
- ✅ **Enhanced Format Support**: Processes `cameras/`, `poses/`, and `sequences/` directories
- 🌍 **GPS → Local ENU Transformation**: Converts GPS to local East-North-Up coordinates
- 📷 **Camera Parameter Extraction**: Detailed calibration from intrinsics
- 🧭 **Rotation Conversion**: Compass angles to axis-angle representation
- ⏱️ **Timestamp Processing**: Microsecond precision for PyCuSFM
- 📁 **Camera Organization**: Groups images by camera type automatically

### Input Format (Enhanced Mapillary)

```
data/mapillary_dataset/
├── images/                          # Image files
├── cameras/                         # Per-image intrinsics
│   ├── {image_id}_intrinsics.json
│   └── ...
├── poses/                           # Per-image poses
│   ├── {image_id}_pose.json
│   └── ...
└── sequences/                       # Trajectory data
    ├── sequence_{id}_trajectory.json
    └── ...
```

### Output Format (PyCuSFM)

```
output/cusfm_data/
├── frames_meta.json                 # KeyframesMetadataCollection format
└── images/                          # Organized by camera type
    ├── apple_iphone/
    │   ├── {image_id}.jpg
    │   └── ...
    └── camera_0/
        └── ...
```

### Usage

```python
from dataset_setup.mapillary_to_cusfm_converter import MapillaryToCuSFMConverter

# Convert enhanced Mapillary format to PyCuSFM
converter = MapillaryToCuSFMConverter(
    mapillary_data_dir="data/mapillary_downtown",
    output_dir="output/cusfm_data"
)
converter.convert()
```

### Command Line

```bash
# Run converter
cd dataset_setup
python mapillary_to_cusfm_converter.py \
    --input-dir ../data/mapillary_downtown \
    --output-dir ../output/cusfm_data
```

### Key Transformations

#### GPS Coordinate Transformation
- Converts WGS84 (lat, lon, alt) to local ENU coordinates
- Uses first valid GPS point as reference origin
- Maintains proper coordinate system for SLAM algorithms

#### Camera Parameters
- Extracts focal length, principal point, distortion coefficients
- Handles multiple camera models and manufacturers
- Creates camera matrix in standard 3x3 format

#### Rotation Handling
- Processes Mapillary compass angles (degrees from North)
- Converts to axis-angle representation
- Maintains camera-to-world transformation

For detailed documentation, see [`dataset_setup/CONVERTER_README.md`](./dataset_setup/CONVERTER_README.md)

</details>

---

<details>
<summary><h3>📏 Batch Depth Prediction</h3></summary>

### Overview
High-performance batch depth map prediction using state-of-the-art depth estimation models from Hugging Face, supporting both Depth Anything V2 and V3.

### Features
- 🎯 **Multiple Model Support**: Depth Anything V2/V3, DPT-Large, and other HF models
- ⚡ **Batch Processing**: Efficient DataLoader-based batching
- 💾 **Dual Output**: Raw depth maps (`.npy`) and visualization (`.png`)
- 🔧 **Memory Optimized**: Automatic CUDA cache management
- 📊 **Progress Tracking**: Real-time processing statistics

### Supported Models

#### Depth Anything V2 (Default)
- `depth-anything/Depth-Anything-V2-Small-hf` (Fast)
- `depth-anything/Depth-Anything-V2-Base-hf` (Balanced)
- `depth-anything/Depth-Anything-V2-Large-hf` (Accurate)

#### Depth Anything V3 (Requires Installation)
- `depth-anything/DA3NESTED-GIANT-SMALL`
- `depth-anything/DA3NESTED-GIANT-BASE`
- `depth-anything/DA3NESTED-GIANT-LARGE`

#### Other Models
- `LiheYoung/depth-anything-large-hf`
- `Intel/dpt-large`

### Usage

```bash
# Basic usage with Depth Anything V2 Small
python batch_depth_prediction.py \
    --input-folder data/boston/images \
    --output-npy-folder data/boston/depth_npy \
    --output-png-folder data/boston/depth_png

# Use larger model for better quality
python batch_depth_prediction.py \
    --input-folder data/boston/images \
    --output-npy-folder data/boston/depth_npy \
    --output-png-folder data/boston/depth_png \
    --model depth-anything/Depth-Anything-V2-Large-hf \
    --batch-size 4

# Use Depth Anything V3 (more accurate)
python batch_depth_prediction.py \
    --input-folder data/boston/images \
    --output-npy-folder data/boston/depth_npy \
    --output-png-folder data/boston/depth_png \
    --model depth-anything/DA3NESTED-GIANT-BASE \
    --use-da3 \
    --batch-size 2
```

### Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--input-folder` | Input image directory | Required |
| `--output-npy-folder` | Output directory for .npy depth maps | Required |
| `--output-png-folder` | Output directory for .png visualizations | Required |
| `--model` | Hugging Face model name | `Depth-Anything-V2-Small-hf` |
| `--use-da3` | Enable Depth Anything V3 | False |
| `--device` | Device (cuda/cpu) | `cuda` |
| `--batch-size` | Batch size for processing | 1 |
| `--num-workers` | DataLoader workers | 4 |

### Output Format

```
output_npy/
├── {image_name}_depth.npy          # Raw depth values (float32)
└── ...

output_png/
├── {image_name}_depth.png          # Normalized visualization
└── ...
```

### Performance Tips
- **GPU Memory**: Reduce batch size if OOM errors occur
- **Speed**: Use Small model for faster processing
- **Quality**: Use Large/V3 models for better accuracy
- **I/O**: Increase `--num-workers` for faster loading

</details>

---

<details>
<summary><h3>🎭 Intelligent Object Removal with Optical Flow</h3></summary>

### Overview
Advanced object removal pipeline that uses optical flow motion detection to intelligently identify and remove dynamic objects (cars, people, vehicles) from street-view sequences, preparing clean data for 3D reconstruction.

### Features
- 🎯 **Optical Flow Refinement**: Uses RAFT to distinguish moving vs static objects
- 🤖 **RT-DETR Detection**: Fast and accurate object detection
- ✂️ **SAM2 Segmentation**: Precise segmentation masks
- 🔍 **RANSAC Motion Filtering**: Separates camera motion from object motion
- 💾 **Mask Saving**: Optional mask preservation for analysis
- ⚡ **Memory Optimized**: Efficient sequential processing

### Target Objects
Removes: cars, trucks, buses, bicycles, people, motorcycles, trains, boats, animals (cats, dogs, horses, etc.), and common accessories (backpacks, umbrellas, suitcases)

### Workflow
1. **Object Detection** → RT-DETR identifies target objects
2. **Optical Flow** → RAFT computes motion between frames
3. **Motion Mask** → RANSAC separates background/foreground motion
4. **Segmentation** → SAM2 creates precise masks
5. **Refinement** → Masks refined using motion information
6. **Removal** → Objects removed (zeroed out)

### Usage

```bash
# Basic usage - remove objects with optical flow
python batch_object_removal_with_flow.py \
    --base-path data/boston_cusfm_sequence

# Disable optical flow refinement
python batch_object_removal_with_flow.py \
    --base-path data/boston_cusfm_sequence \
    --no-flow

# Save masks for analysis
python batch_object_removal_with_flow.py \
    --base-path data/boston_cusfm_sequence \
    --save-masks

# Save combined masks (single file per image)
python batch_object_removal_with_flow.py \
    --base-path data/boston_cusfm_sequence \
    --save-masks \
    --combine-masks

# Custom output paths
python batch_object_removal_with_flow.py \
    --base-path data/boston_cusfm_sequence \
    --filtered-output data/boston_clean \
    --masks-output data/boston_masks
```

### Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--base-path` | Input image directory | Required |
| `--filtered-output` | Output directory for cleaned images | `../filtered` |
| `--masks-output` | Output directory for masks | `../masks` |
| `--no-flow` | Disable optical flow refinement | False |
| `--save-masks` | Save segmentation masks | False |
| `--combine-masks` | Combine all masks per image | False |

### Output Structure

```
filtered/                            # Cleaned images
├── {image_name}.jpg                # Objects removed
└── ...

masks/                              # Optional mask output
├── {image_name}/                   # Individual masks
│   ├── mask_000.png
│   ├── mask_001.png
│   └── ...
└── {image_name}_combined.png      # Combined mask (if --combine-masks)
```

### Technical Details

#### Optical Flow Motion Detection
- Uses RAFT Small model for flow estimation
- RANSAC-based homography to separate camera motion
- Threshold-based motion mask generation (default: 20px flow error)

#### Object Detection & Segmentation
- RT-DETR V2 for real-time detection (confidence > 0.25)
- SAM2 Hiera-Large for precise segmentation
- Batch processing for memory efficiency

### Performance Characteristics
- **Speed**: ~2-5 seconds per image (GPU)
- **Memory**: ~8GB GPU VRAM recommended
- **Quality**: High-precision object removal with minimal artifacts

</details>

---

<details>
<summary><h3>🎬 Motion-Based Sequence Filtering</h3></summary>

### Overview
Intelligent frame selection tool that analyzes optical flow between consecutive frames to filter out static or low-motion frames, ensuring high-quality image sequences for 3D reconstruction.

### Features
- 🌊 **Optical Flow Analysis**: RAFT-based motion detection
- 🎭 **Mask-Aware Filtering**: Exclude segmented objects from motion calculation
- 📊 **Motion Statistics**: Detailed per-frame motion analysis
- 🔗 **Auxiliary Data Sync**: Automatically filter corresponding depth maps, masks, etc.
- 💾 **Batch Processing**: Memory-efficient GPU processing
- 📈 **Configurable Threshold**: Adjustable motion sensitivity

### Why Filter by Motion?
- **Reduces Redundancy**: Removes near-duplicate frames
- **Improves Reconstruction**: SfM works best with sufficient baseline
- **Saves Computation**: Fewer frames = faster reconstruction
- **Better Convergence**: Eliminates degenerate configurations

### Usage

```bash
# Basic usage - filter sequence with default threshold (5.0)
python filter_sequence_by_motion.py \
    data/boston_cusfm_sequence \
    --output-dir data/boston_filtered

# Custom motion threshold (higher = more selective)
python filter_sequence_by_motion.py \
    data/boston_cusfm_sequence \
    --output-dir data/boston_filtered \
    --motion-threshold 10.0

# Use masks to exclude objects from motion calculation
python filter_sequence_by_motion.py \
    data/boston_cusfm_sequence \
    --output-dir data/boston_filtered \
    --mask-dir data/boston_masks

# Sync auxiliary data (depth maps, masks, etc.)
python filter_sequence_by_motion.py \
    data/boston_cusfm_sequence \
    --output-dir data/boston_filtered \
    --auxiliary-dir data/boston_depth \
    --auxiliary-dir data/boston_masks

# Analyze only (no copying)
python filter_sequence_by_motion.py \
    data/boston_cusfm_sequence \
    --no-copy \
    --motion-threshold 5.0
```

### Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `sequence_dir` | Input image directory | Required |
| `--output-dir` | Output directory for filtered images | `{input}_filtered` |
| `--motion-threshold` | Mean flow threshold (pixels) | 5.0 |
| `--mask-dir` | Directory with masks to exclude from flow | None |
| `--auxiliary-dir` | Auxiliary data to filter (repeatable) | None |
| `--no-copy` | Analysis only, don't copy files | False |

### Motion Threshold Guidelines
- **2.0-3.0**: Very aggressive, maximum frame reduction
- **5.0**: Balanced (default), good for most scenarios
- **10.0**: Conservative, keeps more frames
- **15.0+**: Minimal filtering, only removes truly static frames

### Output

```
output_dir/
├── 000000.jpg                      # Renumbered sequence
├── 000001.jpg
├── ...
└── motion_statistics.txt          # Detailed motion report

auxiliary_output/
├── 000000_depth.npy               # Synced auxiliary data
├── 000001_depth.npy
└── ...
```

### Motion Statistics File

```
Motion Threshold: 5.0
Mask Directory: data/boston_masks
Original images: 500
Selected images: 287

============================================================
Detailed Motion Statistics:
============================================================
000000.jpg -> 000001.jpg: 8.4532 (kept)
000001.jpg -> 000002.jpg: 2.1456 (dropped)
000002.jpg -> 000003.jpg: 12.7891 (kept)
...

============================================================
Selected Images:
============================================================
000000: 000000.jpg (original: 0)
000001: 000001.jpg (original: 1)
000002: 000003.jpg (original: 3)
...
```

### Technical Details

#### Optical Flow Computation
- Model: RAFT Small (compiled for speed)
- Resolution: 1080×1920 (divisible by 8 for RAFT)
- Flow magnitude: Mean of √(u² + v²) across image

#### Mask Integration
- Masks loaded and resized to flow resolution
- Motion calculated only in valid (non-masked) regions
- Prevents object motion from triggering frame selection

### Performance
- **Speed**: ~1-2 seconds per frame pair (GPU)
- **Memory**: ~6GB GPU VRAM
- **Typical Reduction**: 40-60% fewer frames

</details>

---

## 🚀 Complete Workflow Example

Here's an end-to-end example processing a Boston downtown dataset:

```bash
# Step 1: Download Mapillary data
python deprecated/mapillary_downloader.py \
    --access-token YOUR_TOKEN \
    --output-dir data/boston \
    --bbox "-71.0589,42.3601,-71.0489,42.3701" \
    --street-view --detections

# Step 2: Convert to PyCuSFM format
cd dataset_setup
python mapillary_to_cusfm_converter.py \
    --input-dir ../data/boston \
    --output-dir ../data/boston_cusfm
cd ..

# Step 3: Remove dynamic objects
python batch_object_removal_with_flow.py \
    --base-path data/boston_cusfm/images/camera_0 \
    --filtered-output data/boston_clean \
    --save-masks --combine-masks

# Step 4: Filter by motion
python filter_sequence_by_motion.py \
    data/boston_clean \
    --output-dir data/boston_final \
    --motion-threshold 5.0

# Step 5: Generate depth maps
python batch_depth_prediction.py \
    --input-folder data/boston_final \
    --output-npy-folder data/boston_final_depth_npy \
    --output-png-folder data/boston_final_depth_png \
    --model depth-anything/Depth-Anything-V2-Base-hf \
    --batch-size 4

# Step 6: Ready for 3D reconstruction!
# Now use with Gaussian Splatting, NeRF, or COLMAP
```

---

## 📋 Installation

```bash
# Clone repository
git clone <repository-url>
cd CityScapeGS

# Install base dependencies
pip install -r requirements.txt

# Install third-party dependencies (optional)
# - SAM2: cd thrid_party/sam2 && pip install -e .
# - Depth Anything V3: cd thrid_party/Depth-Anything-3 && pip install -e .
```

### Core Dependencies
```
torch>=2.0.0
torchvision>=0.15.0
transformers>=4.30.0
opencv-python>=4.6.0
numpy>=1.21.0
Pillow>=9.0.0
tqdm>=4.64.0
requests>=2.28.0
PyYAML>=6.0
```

---

## 🎯 Integration with 3D Reconstruction

### Gaussian Splatting
```bash
# Convert data to Gaussian Splatting format
# Use camera parameters from frames_meta.json
# Use depth maps for initialization
```

### NeRF / Instant-NGP
```bash
# Camera poses from frames_meta.json
# Compatible with NeRF-style workflows
```

### OpenSfM
```bash
cd data/boston
opensfm extract_metadata .
opensfm detect_features .
opensfm match_features .
opensfm create_tracks .
opensfm reconstruct .
```

### COLMAP
```bash
# Use camera intrinsics and poses
# Import into COLMAP database format
```

---

## 📊 API Rate Limits (Mapillary)

- **Entity API**: 60,000 requests/minute per app
- **Search API**: 10,000 requests/minute per app  
- **Tiles API**: 50,000 requests/day per app

**Optimization Tips**:
- Use automatic chunking for large areas
- Adjust `--max-workers` based on rate limits
- Use date ranges and filters to reduce requests

---

## 🤝 Contributing

Contributions welcome! Please check the issues tab or submit pull requests.

---

## 📄 License

[Add your license information here]

---

## 🙏 Acknowledgments

- [Mapillary](https://mapillary.com) for street-level imagery API
- [OpenSfM](https://opensfm.org) for structure-from-motion framework
- [Depth Anything](https://github.com/LiheYoung/Depth-Anything) for depth estimation
- [SAM2](https://github.com/facebookresearch/segment-anything-2) for segmentation
- [RAFT](https://github.com/princeton-vl/RAFT) for optical flow
- [3D Gaussian Splatting](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/) for novel view synthesis

---

## 📚 Related Projects

- [OpenSfM](https://github.com/mapillary/OpenSfM) - Structure from Motion library
- [COLMAP](https://colmap.github.io/) - General-purpose SfM and MVS pipeline
- [Mapillary Tools](https://github.com/mapillary/mapillary_tools) - Official upload tools
- [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2) - Depth estimation
- [SAM2](https://github.com/facebookresearch/segment-anything-2) - Segment Anything

