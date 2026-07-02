import numpy as np
import open3d as o3d

filename = "src/bricklaying/assets/flat_brick_large"

# Load mesh
mesh = o3d.io.read_triangle_mesh(filename + ".stl")

# Sample points
pcd = mesh.sample_points_uniformly(20000)

# Visualize
origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)
o3d.visualization.draw_geometries([pcd, origin])

# Shift and scale points
pts = np.asarray(pcd.points)
pts = pts - np.mean(pts, axis=0)
pts = pts / 1000  # millimeters to meters

np.save(filename, pts)