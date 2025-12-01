# Using Your Manhattan Mapillary Data with Gaussian Splatting

## 🎯 Quick Start Guide

You have Mapillary point cloud data in `/home/vamsik1211/Data/Projects/3D-Reconstructions/CityScapeGS/city_data/manhattan/` and want to use it with Gaussian Splatting frameworks.

## 📋 Prerequisites

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Make sure you have the street view images** that correspond to your point cloud data.

## 🚀 Option 1: Convert Single Point Cloud (Quick Test)

For testing with a single point cloud file:

```bash
# Convert a single point cloud file to Gaussian Splatting format
python mapillary_to_gs.py \
  --pointcloud city_data/manhattan/chunk_0_1/pointclouds/106788411500279_pointcloud.json \
  --images city_data/manhattan/chunk_0_1/images \
  --output test_gs_data
```

## 🏙️ Option 2: Convert Entire Manhattan Dataset (Recommended)

For the complete Manhattan reconstruction:

```bash
# Convert all Manhattan chunks to a combined Gaussian Splatting dataset
python batch_mapillary_to_gs.py \
  --data_dir city_data/manhattan \
  --output manhattan_gs_dataset
```

This will:
- Find all point cloud files in all chunks
- Combine them into a single coherent dataset
- Match images with camera poses
- Create all necessary format files

## 📁 Output Structure

After conversion, you'll get:

```
manhattan_gs_dataset/
├── images/                    # All street view images (renamed for consistency)
├── sparse/0/                 # COLMAP format for 3D Gaussian Splatting
│   ├── cameras.txt           # Camera intrinsics
│   ├── images.txt            # Camera poses  
│   └── points3D.txt          # 3D points with colors
├── transforms.json           # NeRF format
├── input_pointcloud.ply      # Original combined point cloud
├── points3d.ply             # For Gaussian Splatting initialization
├── README.md                 # Detailed usage instructions
└── dataset_info.json        # Dataset metadata
```

## 🎮 Using with Different Frameworks

### 1. LongSplat (Best for Street Scenes)

```bash
# Clone LongSplat (optimized for long street trajectories)
git clone https://github.com/NVlabs/LongSplat.git
cd LongSplat

# Install dependencies
pip install -r requirements.txt

# Train on your Manhattan data
python train.py \
  --source_path /path/to/manhattan_gs_dataset \
  --model_path output/manhattan_longsplat \
  --resolution 2

# Render the result
python render.py \
  --model_path output/manhattan_longsplat \
  --skip_train --skip_test
```

### 2. 3D Gaussian Splatting (Original)

```bash
# Clone original Gaussian Splatting
git clone https://github.com/graphdeco-inria/gaussian-splatting.git
cd gaussian-splatting

# Train on Manhattan data
python train.py \
  -s /path/to/manhattan_gs_dataset \
  -m output/manhattan_3dgs

# Render views
python render.py -m output/manhattan_3dgs
```

### 3. Nerfstudio (Alternative)

```bash
# Install nerfstudio
pip install nerfstudio

# Train with nerfstudio
ns-train nerfacto \
  --data /path/to/manhattan_gs_dataset
```

## 🔧 Troubleshooting Your Data

### Check Point Cloud Structure
```bash
# Examine your point cloud file structure
python -c "
import json
with open('city_data/manhattan/chunk_0_1/pointclouds/106788411500279_pointcloud.json', 'r') as f:
    data = json.load(f)
    print('Data keys:', list(data[0].keys()) if isinstance(data, list) else list(data.keys()))
    if 'points' in (data[0] if isinstance(data, list) else data):
        points = data[0]['points'] if isinstance(data, list) else data['points']
        print(f'Number of 3D points: {len(points)}')
    if 'shots' in (data[0] if isinstance(data, list) else data):
        shots = data[0]['shots'] if isinstance(data, list) else data['shots']
        print(f'Number of camera shots: {len(shots)}')
"
```

### Verify Images Directory
```bash
# Check if you have corresponding images
ls -la city_data/manhattan/chunk_0_1/images/
```

### Test Single Conversion
```bash
# Test with verbose output
python mapillary_to_gs.py \
  --pointcloud city_data/manhattan/chunk_0_1/pointclouds/106788411500279_pointcloud.json \
  --images city_data/manhattan/chunk_0_1/images \
  --output test_single \
  --verbose
```

## 📊 Expected Results

- **Point Cloud**: Typically 1000-10000 3D points per chunk
- **Images**: Street view images matching camera poses
- **Cameras**: Perspective cameras with known intrinsics
- **Scene Type**: Street-level urban environment

## 💡 Tips for Best Results

1. **Use LongSplat** for street scenes (handles long trajectories better)
2. **Combine multiple chunks** for better coverage
3. **Check image quality** - remove blurry or low-quality images
4. **Consider downsampling** high-resolution images to 1920x1080 for faster training
5. **Use panoramic images** if available (better coverage)

## 🐛 Common Issues

1. **No matching images**: Check that image filenames match camera IDs
2. **Memory issues**: Reduce image resolution or use fewer images
3. **Poor reconstruction**: Ensure good camera pose quality
4. **Coordinate system**: All data is in OpenCV format (Y down, Z forward)

## 🚀 Next Steps

1. Run the batch converter on your Manhattan data
2. Start with LongSplat for best street scene results
3. Experiment with different training parameters
4. Consider combining with other datasets for larger scenes

```bash
# Complete pipeline for your data
python batch_mapillary_to_gs.py --data_dir city_data/manhattan --output manhattan_gs_dataset
cd /path/to/LongSplat
python train.py --source_path /path/to/manhattan_gs_dataset --model_path output/manhattan_scene
```
