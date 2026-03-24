from plyfile import PlyData
import viser
import numpy as np
import time

# Set your .ply path here
PLY_PATH = "/home/exouser/CityScapesGS/data/Matrix-City/small_city/small_city_pointcloud/point_cloud/aerial/Block_A.ply"

ply = PlyData.read(PLY_PATH)

# Read vertex positions
v = ply["vertex"].data
points = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float32)

# Read vertex colors if present
color_fields = {"red", "green", "blue"}
if color_fields.issubset(v.dtype.names):
    colors = np.stack([v["red"], v["green"], v["blue"]], axis=1).astype(np.float32)
    if colors.max() > 1.0:
        colors /= 255.0
else:
    colors = np.ones((points.shape[0], 3), dtype=np.float32)  # fallback: white

MAX_POINTS = 2000000
print(f"Original point count: {points.shape[0]}")
if points.shape[0] > MAX_POINTS:
    rng = np.random.default_rng()
    sample_idx = rng.choice(points.shape[0], size=MAX_POINTS, replace=False)
    points = points[sample_idx]
    colors = colors[sample_idx]

# Visualize in viser
server = viser.ViserServer()
server.scene.add_point_cloud(
    name="ply_pointcloud",
    points=points,
    colors=colors,
    point_size=0.001,
)

print(f"Loaded {points.shape[0]} points from: {PLY_PATH}")
print("Viewer: http://localhost:8080")

try:
    while True:
        time.sleep(1.0)
except KeyboardInterrupt:
    print("Shutting down viewer.")
