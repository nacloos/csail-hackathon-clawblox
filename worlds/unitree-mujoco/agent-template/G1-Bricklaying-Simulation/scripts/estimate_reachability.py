"""Estimate reachable workspace volumes for Unitree G1 arms.

Uses the reduced robot model from `g1_bricklaying.modules.arm_ik` to sample
joint configurations, compute end-effector positions, and visualize reachable space.

To-do's:
- doesn't exclude samples "too close" (is that necessary?)
"""
import argparse
from pathlib import Path
import time
import numpy as np
import matplotlib.pyplot as plt

import trimesh
import cvxpy as cp
import pinocchio as pin
import pinocchio.visualize as visualize
import meshcat.geometry as meshgeom

from bricklaying.robot import DualArmIK


COLORS = {
    'left_points': 0xFF6B6B,    # Soft red
    'right_points': 0x4ECDC4,   # Soft teal
    'ellipsoid': 0x95E1D3,      # Light mint
    'hull': 0xF38181,           # Soft coral
    'ground': 0x2C2C2C,         # Dark gray
    'sphere': 0xF6D55C,         # Bright yellow
}


def fit_ellipsoid_in_hull(hull):
    normals = hull.face_normals
    face_vertices = hull.vertices[hull.faces[:, 0]]

    A = normals
    b = np.einsum('ij,ij->i', normals, face_vertices)

    # Assume ellipsoid is axis-aligned
    q = cp.Variable(3, pos=True)
    z = cp.Variable()

    constraints = []
    for i in range(len(A)):
        ai = A[i]
        bi = b[i]
        constraints.append(cp.norm(cp.multiply(q, ai)) + ai[2] * z <= bi)

    prob = cp.Problem(cp.Maximize(cp.sum(cp.log(q))), constraints)
    prob.solve(solver=cp.SCS)

    radii = q.value
    c = np.zeros(3)
    c[2] = z.value

    print(f"Fitted ellipsoid center: {c}, radii: {radii}")

    sphere = trimesh.creation.icosphere(subdivisions=3)
    sphere.apply_scale(radii)

    T = np.eye(4)
    T[:3, 3] = c
    sphere.apply_transform(T)

    return sphere
    

def sample_reach(armik, n_samples):
    model = armik.urdf_model.reduced_robot.model
    data = armik.urdf_model.reduced_robot.data
    collision_model = armik.urdf_model.reduced_robot.collision_model
    collision_data = armik.urdf_model.reduced_robot.collision_data
    visual_model = armik.urdf_model.reduced_robot.visual_model
    visual_data = armik.urdf_model.reduced_robot.visual_data

    pts_L = []
    pts_R = []
    counter = 0
    while len(pts_L) < n_samples:
        if counter % (n_samples // 10) == 0:
            print(f"Sampled {counter} (valid: {len(pts_L)})")
        counter += 1

        # Random joint config
        q = pin.randomConfiguration(model)
        q[0] = np.random.uniform(-np.pi/3, np.pi/3)

        # Forward kinematics
        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacements(model, data)
        pin.updateGeometryPlacements(model, data, collision_model, collision_data, q)
        pin.updateGeometryPlacements(model, data, visual_model, visual_data, q)

        # Collision checking
        collision = pin.computeCollisions(collision_model, collision_data, False)
        if collision:
            continue
            
        pL = data.oMf[armik.left_ee_id].translation.copy()
        pR = data.oMf[armik.right_ee_id].translation.copy()
        pts_L.append(pL)
        pts_R.append(pR)

    return np.array(pts_L), np.array(pts_R)


def add_thick_triad(viz, name, transform, scale=0.1, radius=0.005):
    """Add a thick coordinate frame at the given transform."""
    cylinder = meshgeom.Cylinder(height=scale, radius=radius)
    
    # X-axis (red)
    T_x = transform.copy()
    T_x[:3, :3] = T_x[:3, :3] @ pin.rpy.rpyToMatrix(0, 0, np.pi / 2)
    T_x[:3, 3] = transform[:3, 3] + transform[:3, :3] @ np.array([scale/2, 0, 0])
    viz.viewer[f"{name}/x"].set_object(cylinder, meshgeom.MeshLambertMaterial(color=0xff0000))
    viz.viewer[f"{name}/x"].set_transform(T_x)
    
    # Y-axis (green)
    T_y = transform.copy()
    T_y[:3, 3] = transform[:3, 3] + transform[:3, :3] @ np.array([0, scale/2, 0])
    viz.viewer[f"{name}/y"].set_object(cylinder, meshgeom.MeshLambertMaterial(color=0x00ff00))
    viz.viewer[f"{name}/y"].set_transform(T_y)
    
    # Z-axis (blue)
    T_z = transform.copy()
    T_z[:3, :3] = T_z[:3, :3] @ pin.rpy.rpyToMatrix(np.pi / 2, 0, 0)
    T_z[:3, 3] = transform[:3, 3] + transform[:3, :3] @ np.array([0, 0, scale/2])
    viz.viewer[f"{name}/z"].set_object(cylinder, meshgeom.MeshLambertMaterial(color=0x0000ff))
    viz.viewer[f"{name}/z"].set_transform(T_z)


def visualize_workspace(armik, pts_L, pts_R):
    """Visualizes the robot and the volume in Meshcat."""
    model = armik.urdf_model.reduced_robot.model
    data = armik.urdf_model.reduced_robot.data
    collision_model = armik.urdf_model.reduced_robot.collision_model
    collision_data = armik.urdf_model.reduced_robot.collision_data
    visual_model = armik.urdf_model.reduced_robot.visual_model
    visual_data = armik.urdf_model.reduced_robot.visual_data

    # Start meshcat
    viz = visualize.MeshcatVisualizer(model, collision_model, visual_model)
    viz.initViewer(open=True)
    viz.loadViewerModel("g1")

    # Display the robot in a neutral pose
    q_neutral = pin.neutral(model)
    pin.forwardKinematics(model, data, q_neutral)
    pin.updateFramePlacements(model, data)
    pin.updateGeometryPlacements(model, data, collision_model, collision_data, q_neutral)
    pin.updateGeometryPlacements(model, data, visual_model, visual_data, q_neutral)
    viz.display(q_neutral)
    viz.displayVisuals(True)

    # End effector frames
    for frame_name in ['left_ee', 'right_ee']:
        f_id = model.getFrameId(frame_name)
        add_thick_triad(viz, frame_name, data.oMf[f_id].homogeneous)
        # sphere = meshgeom.Sphere(radius=0.02)
        # viz.viewer[f"{frame_name}/center"].set_object(sphere, meshgeom.MeshLambertMaterial(color=COLORS["sphere"]))
        # viz.viewer[f"{frame_name}/center"].set_transform(data.oMf[f_id].homogeneous)

    # Point clouds
    red_L = np.tile(np.array([[1.0], [0.0], [0.0]]), (1, pts_L.shape[0])) 
    blue_R = np.tile(np.array([[0.0], [0.0], [1.0]]), (1, pts_R.shape[0])) 
    point_cloud_L = meshgeom.PointCloud(pts_L.T, red_L, size=0.01)
    point_cloud_R = meshgeom.PointCloud(pts_R.T, blue_R, size=0.01)
    viz.viewer["workspace/points_L"].set_object(point_cloud_L)
    viz.viewer["workspace/points_R"].set_object(point_cloud_R)

    # Convex hull
    pts = np.vstack([pts_L, pts_R])
    hull = trimesh.points.PointCloud(pts).convex_hull
    viz.viewer["hull"].set_object(
        meshgeom.TriangularMeshGeometry(hull.vertices, hull.faces),
        meshgeom.MeshLambertMaterial(color=COLORS["hull"], opacity=0.5)
    )

    # Maximal inscribed ellipsoid
    ell = fit_ellipsoid_in_hull(hull)
    viz.viewer["ellipsoid"].set_object(
        meshgeom.TriangularMeshGeometry(ell.vertices, ell.faces),
        meshgeom.MeshLambertMaterial(color=COLORS["ellipsoid"], opacity=0.5)
    )

    # Rm defaults
    viz.viewer["/Background"].set_property("top_color", [1.0, 1.0, 1.0])
    viz.viewer["/Background"].set_property("bottom_color", [0.9, 0.9, 0.9])
    viz.viewer["/Grid"].set_property("visible", False)
    viz.viewer["/Axes"].set_property("visible", False)
    
    print("Point cloud loaded. Check your browser.")
    while True:
        time.sleep(1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--samples', type=int, default=10000)
    p.add_argument('--out', type=str, default='outputs/reachability')
    args = p.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print('Building reduced robot model...')
    armik = DualArmIK()

    print(f'Sampling {args.samples} valid configurations...')
    pts_L, pts_R = sample_reach(armik, n_samples=args.samples)

    # meshcat visualization (optional)
    visualize_workspace(armik, pts_L, pts_R)
    print('Launched Meshcat visualizer. Close it to finish.')


if __name__ == '__main__':
    main()
