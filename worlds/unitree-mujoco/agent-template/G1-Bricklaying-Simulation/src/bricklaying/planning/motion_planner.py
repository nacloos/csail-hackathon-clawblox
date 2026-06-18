"""
Cartesian space motion planning for Unitree G1 robot.
Given start/goal SE(3) poses, produces smooth Cartesian trajectories.
No knowledge of joints, IK, or hardware - pure geometry.
"""
from __future__ import annotations

import numpy as np
import pinocchio as pin
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation, RotationSpline
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class CartesianWaypoint:
    """
    A single waypoint in Cartesian space.
    Represents desired EE state at a given time.
    """
    left_pose: np.ndarray    # 4x4 homogeneous transform
    right_pose: np.ndarray   # 4x4 homogeneous transform
    timestamp: float         # Time from trajectory start (seconds)


class CartesianTrajectory:
    """
    A planned Cartesian trajectory.
    Stores waypoints and provides query interface.
    """
    
    def __init__(self, waypoints: List[CartesianWaypoint]):
        self.waypoints = waypoints

        # As arrays, for quick access
        self.time = np.array([wp.timestamp for wp in waypoints])
        self.left_poses = np.array([wp.left_pose for wp in waypoints])
        self.right_poses = np.array([wp.right_pose for wp in waypoints])
    
    @property
    def duration(self) -> float:
        """Total duration of trajectory (seconds)"""
        return self.waypoints[-1].timestamp if self.waypoints else 0.0
    
    @property
    def n_waypoints(self) -> int:
        return len(self.waypoints)
    
    def sample(self, t: float) -> CartesianWaypoint:
        """
        Sample trajectory at time t via linear interpolation between waypoints.
        
        Args:
            t: Time from trajectory start (seconds), clamped to [0, duration]
        """
        t = np.clip(t, 0.0, self.duration)
        
        # Find the segment where t0 <= t <= t1
        idx = np.searchsorted(self.time, t, side='right') - 1
        idx = np.clip(idx, 0, len(self.waypoints) - 2)
        
        t0 = self.waypoints[idx].timestamp
        t1 = self.waypoints[idx + 1].timestamp
        alpha = (t - t0) / (t1 - t0) if t1 > t0 else 0.0
        
        return CartesianWaypoint(
            left_pose=_interpolate_pose(
                self.waypoints[idx].left_pose,
                self.waypoints[idx + 1].left_pose,
                alpha
            ),
            right_pose=_interpolate_pose(
                self.waypoints[idx].right_pose,
                self.waypoints[idx + 1].right_pose,
                alpha
            ),
            timestamp=t
        )

    def sample_velocity(self, t: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sample Cartesian velocity at time t via finite difference between
        neighbouring waypoints.

        Returns:
            left_vel:  (6,) spatial velocity [vx,vy,vz, wx,wy,wz] in world frame
            right_vel: (6,) spatial velocity
        """
        t = np.clip(t, 0.0, self.duration)
        idx = np.searchsorted(self.time, t, side='right') - 1
        idx = np.clip(idx, 0, len(self.waypoints) - 2)

        dt = self.time[idx + 1] - self.time[idx]
        if dt < 1e-6:
            return np.zeros(6), np.zeros(6)

        left_vel  = _pose_velocity(self.waypoints[idx].left_pose,
                                self.waypoints[idx + 1].left_pose, dt)
        right_vel = _pose_velocity(self.waypoints[idx].right_pose,
                                self.waypoints[idx + 1].right_pose, dt)
        return left_vel, right_vel

    def reverse(self) -> CartesianTrajectory:
        """Returns a new CartesianTrajectory with reverse order waypoints."""
        reverse_waypoints = [
            CartesianWaypoint(lp, rp, t) for lp, rp, t in zip(
                self.left_poses[::-1], self.right_poses[::-1], self.time
            )
        ]
        return CartesianTrajectory(reverse_waypoints)


def _pose_velocity(pose0: np.ndarray, pose1: np.ndarray, dt: float) -> np.ndarray:
    """
    Compute spatial velocity between two 4x4 poses over dt seconds.
    Returns (6,) vector [vx,vy,vz, wx,wy,wz] in world frame.
    """
    v_lin = (pose1[:3, 3] - pose0[:3, 3]) / dt

    # Rotational velocity via matrix log: omega = log(R0.T @ R1) / dt
    R0, R1 = pose0[:3, :3], pose1[:3, :3]
    dR = R0.T @ R1
    omega_body = pin.log3(dR) / dt
    omega_world = R0 @ omega_body  # rotate to world frame

    return np.concatenate([v_lin, omega_world])


def _interpolate_pose(
    pose_start: np.ndarray,
    pose_goal: np.ndarray,
    alpha: float
) -> np.ndarray:
    """
    Interpolate between two SE(3) poses.
    Linear interpolation for position, SLERP for rotation.
    
    Args:
        pose_start: Start pose (4x4)
        pose_goal: Goal pose (4x4)
        alpha: Interpolation parameter [0, 1]
    
    Returns:
        Interpolated pose (4x4)
    """
    M_start = pin.SE3(pose_start[:3, :3], pose_start[:3, 3])
    M_goal = pin.SE3(pose_goal[:3, :3], pose_goal[:3, 3])
    return pin.SE3.Interpolate(M_start, M_goal, alpha).homogeneous


def _minimum_jerk_profile(t: float, T: float) -> Tuple[float, float, float]:
    """
    Minimum jerk trajectory profile.
    
    Minimizes integral of squared jerk, giving smooth human-like motion
    with zero velocity and acceleration at endpoints.
    
    Args:
        t: Current time
        T: Total duration
    
    Returns:
        (s, ds, dds): Position, velocity, acceleration scalings in [0, 1]
    """
    tau = np.clip(t / T, 0.0, 1.0)
    
    s   = 10*tau**3 - 15*tau**4 + 6*tau**5
    ds  = (30*tau**2 - 60*tau**3 + 30*tau**4) / T
    dds = (60*tau - 180*tau**2 + 120*tau**3) / T**2
    
    return s, ds, dds


class MotionPlanner:
    """
    Cartesian space motion planner.
    
    For point-to-point motion: minimum jerk profile.
    For multi-waypoint motion: Bezier path + minimum jerk timing.
    
    Separates path (shape) from timing (speed):
        - Bezier curve: defines the shape of the path through space
        - Minimum jerk: defines how fast we traverse that path,
                        guaranteeing zero velocity/acceleration at endpoints
    """
    
    def __init__(self, waypoints_per_second: float = 50.0):
        self.waypoints_per_second = waypoints_per_second
    
    def plan_through_waypoints(
        self,
        left_waypoints: List[np.ndarray],
        right_waypoints: List[np.ndarray],
        duration: float,
    ) -> CartesianTrajectory:
        """
        Plan smooth trajectory through multiple Cartesian waypoints
        using cubic splines for both position and orientation.
        Overlays a "minimum jerk" (abuse of notation) time profile.

        Positions: C2 cubic spline in R3
        Rotations: C2 RotationSpline in SO(3)

        Parameterized directly over time ∈ [0, duration].
        """
        assert len(left_waypoints) == len(right_waypoints), \
            "Left and right waypoint lists must be same length"
        assert len(left_waypoints) >= 2, \
            "Need at least 2 waypoints"

        n_ctrl = len(left_waypoints)

        # Time stamps for control waypoints (uniformly spaced)
        control_times = np.linspace(0.0, duration, n_ctrl)

        # Extract positions
        left_positions = np.array([wp[:3, 3] for wp in left_waypoints])
        right_positions = np.array([wp[:3, 3] for wp in right_waypoints])

        # Build cubic splines for position (natural boundary conditions)
        left_pos_spline = CubicSpline(control_times, left_positions, axis=0, bc_type="natural")
        right_pos_spline = CubicSpline(control_times, right_positions, axis=0, bc_type="natural")

        # Extract rotations
        left_rotations = Rotation.from_matrix([wp[:3, :3] for wp in left_waypoints])
        right_rotations = Rotation.from_matrix([wp[:3, :3] for wp in right_waypoints])

        # Build SO(3) cubic splines
        left_rot_spline = RotationSpline(control_times, left_rotations)
        right_rot_spline = RotationSpline(control_times, right_rotations)

        # Number of trajectory samples
        n_samples = max(2, int(duration * self.waypoints_per_second))

        sample_times = np.linspace(0.0, duration, n_samples)

        waypoints = []

        for t in sample_times:
            s, _, _ = _minimum_jerk_profile(t, duration)
            s = s * duration

            # Sample position splines
            left_pos = left_pos_spline(s)
            right_pos = right_pos_spline(s)

            # Sample rotation splines
            left_R = left_rot_spline(s).as_matrix()
            right_R = right_rot_spline(s).as_matrix()

            # Assemble poses
            left_pose = np.eye(4)
            left_pose[:3, :3] = left_R
            left_pose[:3, 3] = left_pos

            right_pose = np.eye(4)
            right_pose[:3, :3] = right_R
            right_pose[:3, 3] = right_pos

            waypoints.append(
                CartesianWaypoint(
                    left_pose=left_pose,
                    right_pose=right_pose,
                    timestamp=t,
                )
            )

        return CartesianTrajectory(waypoints)


# ===== Test =====

if __name__ == '__main__':
    import time
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from pinocchio.visualize import MeshcatVisualizer

    from bricklaying.robot import G1URDFModel, DualArmIK
    from .constants import R_LEFT_NOMINAL_PICK, R_RIGHT_NOMINAL_PICK

    print("=" * 60)
    print("Testing Motion Planner + IK (3 free_joints variants)")
    print("=" * 60)

    # [1] Load model + IK
    print("\n[1] Loading robot model...")
    urdf_model = G1URDFModel(reduced=True)
    ik_solver  = DualArmIK()
    print(f"    ✓ {ik_solver.model.nq} DOF reduced model")

    # [2] MeshCat (optional)
    print("\n[2] Loading meshcat...")
    viz = MeshcatVisualizer(
        urdf_model.reduced_robot.model,
        urdf_model.reduced_robot.collision_model,
        urdf_model.reduced_robot.visual_model,
    )
    try:
        viz.initViewer(open=True)
        viz.loadViewerModel()
        print("    ✓ MeshCat: http://127.0.0.1:7000/static/")
    except Exception:
        viz = None
        print("    MeshCat unavailable, skipping visualization.")

    # [3] Plan trajectory
    print("\n[3] Planning trajectory...")
    planner = MotionPlanner()
    q_start = pin.neutral(ik_solver.model)   # 15-DOF neutral config

    left_start  = urdf_model.get_frame_transform(q_start, "left_ee",  use_reduced=True)
    right_start = urdf_model.get_frame_transform(q_start, "right_ee", use_reduced=True)

    def _pose(pos, Rot=np.eye(3)):
        T = np.eye(4); T[:3, 3] = pos; T[:3, :3] = Rot; return T

    left_waypoints = [
        left_start,
        _pose([0.15,  0.4, 0.15]),
        _pose([0.45,  0.3, 0.15], R_LEFT_NOMINAL_PICK),
    ]
    right_waypoints = [
        right_start,
        _pose([0.15, -0.4, 0.15]),
        _pose([0.45, -0.3, 0.15], R_RIGHT_NOMINAL_PICK),
    ]
    duration = 5.0
    traj = planner.plan_through_waypoints(left_waypoints, right_waypoints, duration)
    print(f"    ✓ {traj.n_waypoints} waypoints over {traj.duration:.2f}s")

    # [4] IK + animate for each free_joints variant
    dt = 1.0 / planner.waypoints_per_second
    times = np.array([wp.timestamp for wp in traj.waypoints])

    variants = [
        ("Left + Right  (waist locked)", ik_solver.both_arms_q_idx),
        ("Left  + Waist (right locked)", ik_solver.left_q_idx + ik_solver.waist_q_idx),
        ("Right + Waist (left  locked)", ik_solver.right_q_idx + ik_solver.waist_q_idx),
    ]
    colors = ['tab:blue', 'tab:green', 'tab:red']

    results = {}

    for label, free_joints in variants:
        print(f"\n[4] {label}")
        free_set   = set(free_joints)
        left_free  = any(i in free_set for i in ik_solver.left_q_idx)
        right_free = any(i in free_set for i in ik_solver.right_q_idx)
        waist_free = any(i in free_set for i in ik_solver.waist_q_idx)

        q_traj       = []
        ik_errors_l  = []
        ik_errors_r  = []
        waist_angles = []

        q_curr  = q_start.copy()
        dq_curr = np.zeros(ik_solver.model.nv)

        for wp in traj.waypoints:
            q_sol, dq_sol, _ = ik_solver.solve(
                wp.left_pose, wp.right_pose,
                q_current=q_curr, dq_current=dq_curr, dt=dt,
                free_joints=free_joints,
            )
            l_pose, r_pose = urdf_model.get_frame_transform(
                q_sol, ["left_ee", "right_ee"], use_reduced=True
            )
            if left_free:
                ik_errors_l.append(np.linalg.norm(wp.left_pose[:3, 3]  - l_pose[:3, 3]))
            if right_free:
                ik_errors_r.append(np.linalg.norm(wp.right_pose[:3, 3] - r_pose[:3, 3]))
            waist_angles.append(float(q_sol[ik_solver.waist_q_idx[0]]))
            q_traj.append(q_sol)
            q_curr, dq_curr = q_sol, dq_sol

        results[label] = dict(
            q_traj=q_traj,
            ik_errors_l=ik_errors_l,
            ik_errors_r=ik_errors_r,
            waist_angles=waist_angles,
            left_free=left_free,
            right_free=right_free,
            waist_free=waist_free,
        )

        if left_free:
            print(f"    Left  IK: mean {np.mean(ik_errors_l)*1000:.2f}mm  "
                  f"max {np.max(ik_errors_l)*1000:.2f}mm")
        if right_free:
            print(f"    Right IK: mean {np.mean(ik_errors_r)*1000:.2f}mm  "
                  f"max {np.max(ik_errors_r)*1000:.2f}mm")
        w = np.degrees(waist_angles)
        print(f"    Waist ({'free' if waist_free else 'locked'}): "
              f"[{w.min():.1f}°, {w.max():.1f}°]")

        if viz:
            print(f"    Animating...")
            viz.display(q_traj[0])
            time.sleep(1.0)
            for q in q_traj:
                start = time.time()
                viz.display(q)
                time.sleep(max(0, dt - (time.time() - start)))
            time.sleep(1.0)

    # [5] Comparison plot
    print("\n[5] Plotting comparison...")

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    fig.suptitle('IK free_joints Variant Comparison', fontsize=13)
    ax_l, ax_r, ax_w = axes

    for (label, _), color in zip(variants, colors):
        r = results[label]
        if r['left_free']:
            ax_l.plot(times, np.array(r['ik_errors_l']) * 1000,
                      color=color, linewidth=1.5, label=label)
        if r['right_free']:
            ax_r.plot(times, np.array(r['ik_errors_r']) * 1000,
                      color=color, linewidth=1.5, label=label)
        ax_w.plot(times, np.degrees(r['waist_angles']),
                  color=color, linewidth=1.5, label=label)

    ax_l.set_ylabel('Left EE IK error (mm)')
    ax_l.legend(fontsize=8); ax_l.grid(True)

    ax_r.set_ylabel('Right EE IK error (mm)')
    ax_r.legend(fontsize=8); ax_r.grid(True)

    ax_w.set_ylabel('Waist angle (°)')
    ax_w.set_xlabel('Time (s)')
    ax_w.axhline(0, color='k', linestyle=':', linewidth=0.8)
    ax_w.legend(fontsize=8); ax_w.grid(True)

    plt.tight_layout()
    plt.savefig('motion_planner_analysis.png', dpi=150)
    print("    ✓ Saved motion_planner_analysis.png")

    print("\n" + "=" * 60)
    print("Motion planner tests complete!")
    print("=" * 60)
