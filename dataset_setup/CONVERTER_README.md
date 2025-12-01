# Enhanced Mapillary to PyCuSFM Converter

This converter transforms Mapillary downloaded data into PyCuSFM-compatible format, supporting both the enhanced metadata structure and legacy formats.

## Features

### 🆕 Enhanced Format Support
- **Camera Intrinsics**: Processes individual `{image_id}_intrinsics.json` files from `cameras/` directory
- **Pose Data**: Handles `{image_id}_pose.json` files from `poses/` directory with GPS coordinates and compass angles
- **Sequence Information**: Utilizes trajectory data from `sequences/` directory
- **Coordinate Transformation**: Converts GPS coordinates to local ENU (East-North-Up) coordinate system
- **Reference Point**: Automatically establishes common reference GPS coordinates for consistent local mapping

### 🔧 Legacy Format Support
- **EXIF Data**: Processes individual EXIF JSON files
- **SfM Reconstruction**: Handles OpenSfM reconstruction data from pointclouds
- **Camera Models**: Uses camera_models.json for camera parameters

## Input Data Structure

### Enhanced Format (Preferred)
```
data/msp_downtown_2/
├── images/                          # Image files
│   ├── 190796012763763.jpg
│   ├── 190918552850497.jpg
│   └── ...
├── cameras/                         # Camera intrinsics (per image)
│   ├── 190796012763763_intrinsics.json
│   ├── 190918552850497_intrinsics.json
│   └── ...
├── poses/                           # Camera poses (per image)
│   ├── 190796012763763_pose.json
│   ├── 190918552850497_pose.json
│   └── ...
├── sequences/                       # Sequence trajectories
│   ├── sequence_123_trajectory.json
│   └── ...
└── metadata.json                    # Overall dataset metadata
```

### Legacy Format
```
data/legacy_dataset/
├── images/
├── exif/
├── camera_models.json
├── pointclouds/
└── ...
```

## Output Format (PyCuSFM Compatible)

```
output/cusfm_data/
├── frames_meta.json                 # Main metadata file (KeyframesMetadataCollection format)
└── images/                          # Organized by camera
    ├── apple_iphone/                # Camera-specific subdirectories
    │   ├── 190796012763763.jpg
    │   └── ...
    └── camera_0/                    # Default camera directory
        └── ...
```

## frames_meta.json Structure

The converter generates a `frames_meta.json` file following the PyCuSFM `KeyframesMetadataCollection` format:

```json
{
  "keyframes_metadata": [
    {
      "id": "0",
      "camera_params_id": "0",
      "timestamp_microseconds": "1444087414220000",
      "image_name": "apple_iphone/190796012763763.jpg",
      "camera_to_world": {
        "axis_angle": {
          "x": 0.0,
          "y": 0.0,
          "z": 1.0,
          "angle_degrees": 333.8976
        },
        "translation": {
          "x": 1245.23,
          "y": 2341.67,
          "z": 257.95
        }
      },
      "synced_sample_id": "0"
    }
  ],
  "initial_pose_type": "GPS_IMU",
  "camera_params_id_to_session_name": {
    "0": "0"
  },
  "camera_params_id_to_camera_params": {
    "0": {
      "sensor_meta_data": {
        "sensor_id": 0,
        "sensor_type": "CAMERA",
        "sensor_name": "apple_iphone",
        "frequency": 30,
        "sensor_to_vehicle_transform": {
          "axis_angle": {"x": 0, "y": 0, "z": 0, "angle_degrees": 0},
          "translation": {"x": 0, "y": 0, "z": 0}
        }
      },
      "calibration_parameters": {
        "image_width": 4032,
        "image_height": 3024,
        "camera_matrix": {
          "data": [2698.0, 0, 2016, 0, 2698.0, 1512, 0, 0, 1]
        },
        "distortion_coefficients": {
          "data": [-0.173, 0.027, 0, 0, 0]
        }
      }
    }
  },
  "reference_latlngalt": {
    "latitude": 44.977865,
    "longitude": -93.275115,
    "altitude": 257.95
  }
}
```

## Key Features Explained

### 📍 GPS Coordinate Transformation
- Converts GPS (lat, lon, alt) to local ENU coordinates
- Uses first valid GPS point as reference for consistent local mapping
- Maintains proper coordinate system for SLAM algorithms

### 🧭 Rotation Handling
- Processes Mapillary compass angles (degrees from North)
- Utilizes computed rotation vectors when available
- Converts to axis-angle representation required by PyCuSFM

### 📷 Camera Parameter Extraction
- Extracts focal length, principal point, and distortion coefficients
- Handles different camera models and makes
- Groups images by camera type for multi-camera scenarios

### ⏱️ Timestamp Processing
- Converts Mapillary capture timestamps to microseconds
- Maintains temporal ordering for sequential processing
- Supports synchronization across multiple cameras

## Usage

### Basic Usage
```python
from mapillary_to_cusfm_converter import MapillaryToCuSFMConverter

# Convert enhanced format
converter = MapillaryToCuSFMConverter(
    mapillary_data_dir="data/msp_downtown_2",
    output_dir="output/cusfm_data"
)
converter.convert()
```

### Command Line Testing
```bash
python test_converter.py
```

## Camera Parameter Details

### Intrinsic Parameters
- **Camera Matrix**: 3x3 matrix with focal lengths and principal point
- **Distortion Coefficients**: Brown-Conrady model [k1, k2, p1, p2, k3]
- **Image Dimensions**: Width and height in pixels

### Extrinsic Parameters
- **Translation**: 3D position in local ENU coordinates (meters)
- **Rotation**: Axis-angle representation with angle in degrees
- **Reference Frame**: Camera-to-world transformation

## Coordinate Systems

### Input (Mapillary)
- **GPS**: WGS84 latitude, longitude, altitude
- **Compass**: Degrees from North (0° = North, 90° = East)
- **Computed Rotation**: Axis-angle in radians

### Output (PyCuSFM)
- **Local ENU**: East-North-Up coordinate system in meters
- **Axis-Angle**: Rotation representation with angle in degrees
- **Camera-to-World**: Transformation from camera frame to world frame

## Error Handling

The converter includes robust error handling for:
- Missing intrinsics or pose files
- Invalid GPS coordinates
- Malformed JSON data
- Missing image files
- Camera parameter extraction failures

## Integration with 3D Reconstruction

The generated `frames_meta.json` can be directly used with:
- **PyCuSFM**: For structure-from-motion and SLAM
- **OpenSfM**: With minor format adaptations
- **COLMAP**: After coordinate system transformation
- **Other SLAM frameworks**: Supporting the KeyframesMetadataCollection format

## Example Workflow

1. **Download Mapillary Data**: Using the enhanced downloader
2. **Convert Format**: Run the converter to generate PyCuSFM format
3. **Process with PyCuSFM**: Use for 3D reconstruction
4. **Visualize Results**: View reconstructed point clouds and camera trajectories

This enhanced converter bridges the gap between Mapillary's rich metadata and modern SLAM frameworks, enabling seamless integration of street-view imagery into 3D reconstruction pipelines.
