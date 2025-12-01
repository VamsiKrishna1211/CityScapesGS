import json
import numpy as np
from PIL import Image
import viser
import os
from pathlib import Path

SEQUENCE_PATH = "/home/vamsik1211/Data/Projects/3D-Reconstructions/CityScapeGS/data/boston_cusfm_sequence/0000_zUKZjC0qOqVkYWxD-1n6EA"
camera_info = json.loads(open("/home/vamsik1211/Data/Projects/3D-Reconstructions/CityScapeGS/data/boston/cameras/105972988238547_intrinsics.json").read())

# Downsampling parameters
DOWNSAMPLE_FACTOR = 4  # Downsample images by this factor (4x4 = 16x fewer points per frame)
FRAME_SKIP = 2  # Process every Nth frame (2 = process every other frame)
POINT_SKIP = 4  # Keep every Nth point after unprojection (further reduction)

# Load frames metadata with camera poses
frames_meta = json.load(open(os.path.join(SEQUENCE_PATH, "frames_meta.json")))

# Get camera intrinsics
focal_length = camera_info["focal_length_pixels"]
width = camera_info["width"]
height = camera_info["height"]
cx = width / 2
cy = height / 2

def axis_angle_to_rotation_matrix(axis_angle_dict):
    """Convert axis-angle representation to rotation matrix."""
    x = axis_angle_dict['x']
    y = axis_angle_dict['y']
    z = axis_angle_dict['z']
    angle_deg = axis_angle_dict['angle_degrees']
    
    # Convert to radians
    angle = np.deg2rad(angle_deg)
    
    # Normalize axis
    axis = np.array([x, y, z])
    axis = axis / np.linalg.norm(axis)
    
    # Rodrigues' rotation formula
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    
    R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * np.dot(K, K)
    return R

def unproject_depth_to_points(depth, focal_length, cx, cy):
    """Unproject depth map to 3D points in camera space."""
    h, w = depth.shape[:2]
    i, j = np.meshgrid(np.arange(w), np.arange(h), indexing='xy')
    
    z = depth.astype(np.float32)
    x = (i - cx) * z / focal_length
    y = (j - cy) * z / focal_length
    
    points = np.stack([x, y, z], axis=-1).reshape(-1, 3)
    return points

# Visualize with viser
server = viser.ViserServer()

all_points = []
all_colors = []

# Process each frame
print(f"Processing {len(frames_meta['keyframes_metadata'])} frames (skipping every {FRAME_SKIP})...")
for idx, frame in enumerate(frames_meta['keyframes_metadata']):
    # Skip frames for memory efficiency
    if idx % FRAME_SKIP != 0:
        continue
    
    # Load image and depth
    image_path = os.path.join(SEQUENCE_PATH, frame['image_name'])
    depth_filename = frame['image_name'].replace('camera_0/', 'depth/').replace('.jpg', '.npy')
    depth_path = os.path.join(SEQUENCE_PATH, depth_filename)
    
    if not os.path.exists(image_path) or not os.path.exists(depth_path):
        print(f"Skipping frame {idx}: missing files")
        continue
    
    # Load RGB and depth
    color = np.array(Image.open(image_path))
    depth = np.load(depth_path)
    
    # Downsample images for memory efficiency
    if DOWNSAMPLE_FACTOR > 1:
        new_h = color.shape[0] // DOWNSAMPLE_FACTOR
        new_w = color.shape[1] // DOWNSAMPLE_FACTOR
        color = np.array(Image.fromarray(color).resize((new_w, new_h), Image.BILINEAR))
        depth = np.array(Image.fromarray(depth).resize((new_w, new_h), Image.NEAREST))
    
    # Unproject to camera space (with adjusted focal length and principal point for downsampled image)
    focal_adjusted = focal_length / DOWNSAMPLE_FACTOR
    cx_adjusted = cx / DOWNSAMPLE_FACTOR
    cy_adjusted = cy / DOWNSAMPLE_FACTOR
    points_cam = unproject_depth_to_points(depth, focal_adjusted, cx_adjusted, cy_adjusted)
    colors = color.reshape(-1, 3) / 255.0
    
    # Filter valid points
    valid_mask = (depth.reshape(-1) > 0)
    points_cam = points_cam[valid_mask]
    colors = colors[valid_mask]
    
    # Further downsample points by keeping every Nth point
    if POINT_SKIP > 1 and len(points_cam) > 0:
        indices = np.arange(0, len(points_cam), POINT_SKIP)
        points_cam = points_cam[indices]
        colors = colors[indices]
    
    # Get camera-to-world transform
    R = axis_angle_to_rotation_matrix(frame['camera_to_world']['axis_angle'])
    t = np.array([
        frame['camera_to_world']['translation']['x'],
        frame['camera_to_world']['translation']['y'],
        frame['camera_to_world']['translation']['z']
    ])
    
    # Apply coordinate transform (camera space: right, down, forward -> world space)
    # OpenCV/COLMAP convention: flip Y and Z
    points_cam[:, 1] = -points_cam[:, 1]
    points_cam[:, 2] = -points_cam[:, 2]
    
    # Transform to world space
    points_world = (R @ points_cam.T).T + t
    
    all_points.append(points_world)
    all_colors.append(colors)
    
    if (idx + 1) % 10 == 0:
        print(f"Processed {idx + 1} frames...")

# Combine all points
all_points = np.vstack(all_points)
all_colors = np.vstack(all_colors)

print(f"Total points: {len(all_points)}")

# Add to scene
server.scene.add_point_cloud(
    name="full_sequence_pointcloud",
    points=all_points,
    colors=all_colors,
    point_size=0.001
)

# Add camera frustums to visualize camera positions
for idx, frame in enumerate(frames_meta['keyframes_metadata']):
    R = axis_angle_to_rotation_matrix(frame['camera_to_world']['axis_angle'])
    t = np.array([
        frame['camera_to_world']['translation']['x'],
        frame['camera_to_world']['translation']['y'],
        frame['camera_to_world']['translation']['z']
    ])
    
    # Create 4x4 transformation matrix
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    
    # Add camera frustum (show every 5th camera to avoid clutter)
    if idx % 5 == 0:
        server.scene.add_camera_frustum(
            name=f"camera_{idx:06d}",
            fov=2 * np.arctan(width / (2 * focal_length)),
            aspect=width / height,
            scale=0.1,
            wxyz=viser.transforms.SO3.from_matrix(R).wxyz,
            position=t,
            color=(255, 0, 0)
        )

print(f"Point cloud created with {len(all_points)} points from {len(frames_meta['keyframes_metadata'])} frames")
print(f"Server running at http://localhost:8080")
print("Press Ctrl+C to stop the server")

# Keep the server alive
import time
while True:
    time.sleep(1.0)
