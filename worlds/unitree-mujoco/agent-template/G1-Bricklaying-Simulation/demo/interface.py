
from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation as R
from typing import Optional

from bricklaying.robot import DDSInterface, G1URDFModel, DualArmIK, TrajectoryController
from bricklaying.robot import T_CAMERA_TO_REALSENSE
from bricklaying.planning import MotionPlanner, EllipsoidRegion, R_RIGHT_NOMINAL_PICK
from bricklaying.perception import (RealSenseCamera, SimRealSenseCamera, BrickPoseEstimator, BrickPose, ArucoLocalizer, TablePoseResult,)
from bricklaying.segmentation import FastSAMSegmentor

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Twist


class G1TwistCmdNode(Node):
    def __init__(self, network_interface: str = "lo"):
        if not rclpy.ok():
            rclpy.init()
        super().__init__('g1_twist_cmd_node')
        self.publisher = self.create_publisher(Twist, 'cmd_vel', 10)

    def publish_twist(self, x, theta):
        msg = Twist()
        msg.linear.x = float(x)
        msg.angular.z = float(theta)
        self.publisher.publish(msg)



def _pose(pos, rot=np.eye(3)):
    T = np.eye(4)
    T[:3, 3] = pos
    T[:3, :3] = rot
    return T


# ── Constants ─────────────────────────────────────────────────────────────────

# Safe home pose for right EE
T_RIGHT_INIT = _pose([0.1, -0.3, 0.15])

# Brick -> grasp offsets
HAND_PICK_OFFSET    = np.array([0.02, 0., 0.07])
HAND_PREPICK_OFFSET = np.array([0.02, 0., 0.25])

# Brick pose scoring: penalty per metre of distance from pelvis
POSE_DISTANCE_WEIGHT = 0.6

# Curved wall placement targets in table frame: [x, y, z, roll_deg, pitch_deg, yaw_deg]
CURVE_WALL = [
    np.array([ 0.09,  0.34,  0.05,  0.0,  0.0,  25.0]),
    #np.array([ 0.17,  0.12,  0.05,  0.0,  0.0,  10.0]),
    #np.array([ 0.17, -0.12,  0.05,  0.0,  0.0, -10.0]),
    #np.array([ 0.09, -0.34,  0.05,  0.0,  0.0, -25.0]),
    #np.array([ 0.09,  0.34,  0.11,  0.0,  0.0,  25.0]),
    #np.array([ 0.17,  0.12,  0.11,  0.0,  0.0,  10.0]),
    #np.array([ 0.17, -0.12,  0.11,  0.0,  0.0, -10.0]),
    #np.array([ 0.09, -0.34,  0.11,  0.0,  0.0, -25.0]),
    #np.array([ 0.09,  0.34,  0.17,  0.0,  0.0,  25.0]),
    #np.array([ 0.17,  0.12,  0.17,  0.0,  0.0,  10.0]),
    #np.array([ 0.17, -0.12,  0.17,  0.0,  0.0, -10.0]),
    #np.array([ 0.09, -0.34,  0.17,  0.0,  0.0, -25.0]),
]

PYRAMID = [
    #np.array([ 0.09,  0.22,  0.05,  0.0,  0.0,  0.0]),
    #np.array([ 0.09,  0.00,  0.05,  0.0,  0.0,  0.0]),
    #np.array([ 0.09, -0.22,  0.05,  0.0,  0.0,  0.0]),
    #np.array([ 0.09,  0.11,  0.11,  0.0,  0.0,  0.0]),
    #np.array([ 0.09, -0.11,  0.11,  0.0,  0.0,  0.0]),
    #np.array([ 0.09,  0.00,  0.17,  0.0,  0.0,  0.0]),

  ]

WALL = [
    #np.array([ 0.09,  0.36,  0.05,  0.0,  0.0,  0.0]),
    #np.array([ 0.09,  0.12,  0.05,  0.0,  0.0,  0.0]),
    #np.array([ 0.09, -0.12,  0.05,  0.0,  0.0,  0.0]),
    #np.array([ 0.09, -0.36,  0.05,  0.0,  0.0,  0.0]),
    #np.array([ 0.09,  0.36,  0.11,  0.0,  0.0,  0.0]),
    #np.array([ 0.09,  0.12,  0.11,  0.0,  0.0,  0.0]),
    #np.array([ 0.09, -0.12,  0.11,  0.0,  0.0,  0.0]),
    #np.array([ 0.09, -0.36,  0.11,  0.0,  0.0,  0.0]),

  ]

# ── Helpers ──────────────────────────────────────────────────────────────────


def _brick_to_table_pose(row: np.ndarray) -> np.ndarray:
    """Convert a structure row [x, y, z, roll_deg, pitch_deg, yaw_deg] to a 4x4 table-frame pose."""
    return _pose(row[:3], R.from_euler('xyz', row[3:6], degrees=True).as_matrix())


def _brick_to_grasp_pose(T_brick: np.ndarray, offset: np.ndarray = HAND_PICK_OFFSET) -> np.ndarray:
    """
    Compute the right-hand EE pose for grasping a brick (or placing at a target).

    Extracts the canonical yaw from T_brick (folded into (-90°, 90°] to handle
    brick symmetry), builds the grasp rotation Rz(yaw) @ R_RIGHT_NOMINAL_PICK,
    and positions the hand by applying offset in the yaw-rotated frame.

    Pass HAND_PREPICK_OFFSET for the pre-approach pose, or the default
    HAND_PICK_OFFSET for the final grasp pose.
    """
    R_mat = T_brick[:3, :3]
    yaw, _, _ = R.from_matrix(R_mat).as_euler('zyx')
    if R_mat[2, 2] < 0:
        yaw = -yaw
    yaw = yaw % np.pi
    if yaw > np.pi / 2:
        yaw -= np.pi

    R_yaw = R.from_euler('z', yaw).as_matrix()
    return _pose(T_brick[:3, 3] + R_yaw @ offset, R_yaw @ R_RIGHT_NOMINAL_PICK)


def _save_trajectory(path: Path, traj, T_brick: np.ndarray, stats=None):
    """Save a CartesianTrajectory + target brick pose + execution stats to a .npz file."""
    data = dict(
        time=traj.time,
        left_poses=traj.left_poses,
        right_poses=traj.right_poses,
        T_brick=T_brick,
    )
    if stats is not None:
        data.update(
            # Per-step error series
            left_ik_pos_errors=stats.left_ik_pos_errors,
            right_ik_pos_errors=stats.right_ik_pos_errors,
            left_ik_rot_errors=stats.left_ik_rot_errors,
            right_ik_rot_errors=stats.right_ik_rot_errors,
            left_track_pos_errors=stats.left_track_pos_errors,
            right_track_pos_errors=stats.right_track_pos_errors,
            left_track_rot_errors=stats.left_track_rot_errors,
            right_track_rot_errors=stats.right_track_rot_errors,
            track_q_errors=stats.track_q_errors,
            loop_times=stats.loop_times,
            # Raw measured time series
            time_meas=stats.time_meas,
            left_pose_meas=stats.left_pose_meas,
            right_pose_meas=stats.right_pose_meas,
            left_pose_ik=stats.left_pose_ik,
            right_pose_ik=stats.right_pose_ik,
            q_meas=stats.q_meas,
            q_ik=stats.q_ik,
            # Summary scalars
            mean_ik_pos_error=stats.mean_ik_pos_error,
            max_ik_pos_error=stats.max_ik_pos_error,
            mean_ik_rot_error=stats.mean_ik_rot_error,
            max_ik_rot_error=stats.max_ik_rot_error,
            mean_track_pos_error=stats.mean_track_pos_error,
            max_track_pos_error=stats.max_track_pos_error,
            mean_track_rot_error=stats.mean_track_rot_error,
            max_track_rot_error=stats.max_track_rot_error,
            mean_track_q_error=stats.mean_track_q_error,
            max_track_q_error=stats.max_track_q_error,
            mean_loop_time=stats.mean_loop_time,
            max_loop_time=stats.max_loop_time,
        )
    np.savez(path, **data)


# ── Demo Class ───────────────────────────────────────────────────────

class PickPlace:
    """
    Orchestrates right-arm-only pick-and-place with ArUco table localization.
    """

    def __init__(self, network_interface: str = "docker0"):
        self.network_interface = network_interface
        self._build()
        self.q_start, _ = self.dds.get_upper_body_state()

    def _build(self):
        print("Building dds interface...")
        self.dds        = DDSInterface(self.network_interface)
        time.sleep(2)
        
        print("Building urdf model...")
        self.urdf_model = G1URDFModel(reduced=True)
        time.sleep(2)
        
        print("Building ik solver...")
        self.ik_solver  = DualArmIK()
        time.sleep(4)
        
        print("Building trajectory controller...")
        self.controller = TrajectoryController(dds=self.dds, urdf_model=self.urdf_model, ik_solver=self.ik_solver,)
        self.segmentor  = FastSAMSegmentor()
        #[FLAG]: reinstate camera
        self.camera     = SimRealSenseCamera()
        self.estimator  = BrickPoseEstimator(
            segmentor=self.segmentor,
            urdf_model=self.urdf_model,
            #[FLAG]: reistante camera
            camera=self.camera,
        )
        self.planner    = MotionPlanner()
        self.reach      = EllipsoidRegion()
        #[FLAG]: Reinstate camera
        self.localizer  = ArucoLocalizer(intrinsics=self.camera.intrinsics)

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------

    def _right_arm_free_joints(self) -> list:
        """Right arm + waist; left arm locked."""
        return self.ik_solver.right_q_idx + self.ik_solver.waist_q_idx

    def _current_ee(self):
        """Return (left_ee, right_ee) transforms at the current measured state."""
        q, _ = self.dds.get_upper_body_state()
        return self.urdf_model.get_frame_transform(q, ["left_ee", "right_ee"], use_reduced=True)

    # -----------------------------------------------------------------------
    # Perception
    # -----------------------------------------------------------------------

    def estimate_brick(self) -> list[BrickPose]:
        return self.estimator.estimate()

    def select_best_pose(self, poses: list[BrickPose]) -> Optional[BrickPose]:
        """Highest ICP fitness, penalized by distance."""
        if not poses:
            return None
        return max(poses, key=lambda p: p.icp_fitness - POSE_DISTANCE_WEIGHT * np.linalg.norm(p.position))

    def is_pickable(self, brick_pose: np.ndarray) -> tuple[bool, float]:
        """EllipsoidRegion reachability check. Returns (pickable, signed_distance)."""
        if brick_pose is None:
            return False, float('inf')
        pick_pose = _brick_to_grasp_pose(brick_pose, HAND_PICK_OFFSET)
        sd = self.reach.signed_distance(pick_pose[:3, 3])
        return sd <= 0.0, sd

    def localize_table(self) -> Optional[tuple[np.ndarray, TablePoseResult]]:
        """Detect ArUco markers and estimate T_pelvis_to_table."""
        color, _ = self.camera.get_frames()
        detections = self.localizer.detect(color)
        result = self.localizer.estimate_table_pose(detections)

        if result is None:
            return None

        # Get camera frame
        q_meas, _ = self.dds.get_upper_body_state()
        T_pelvis_to_camera = self.urdf_model.get_frame_transform(
            q_meas, "d435_link", use_reduced=True
        )

        T_pelvis_to_table = (
            T_pelvis_to_camera
            @ T_CAMERA_TO_REALSENSE
            @ result.T_camera_to_table
        )
        return T_pelvis_to_table, result

    # -----------------------------------------------------------------------
    # Trajectory planning
    # -----------------------------------------------------------------------

    def plan_init_trajectory(self, duration: float = 10.0):
        """Move right arm to its nominal ready pose; left arm held fixed."""
        left_start, right_start = self._current_ee()
        return self.planner.plan_through_waypoints(
            [left_start,  left_start],
            [right_start, T_RIGHT_INIT],
            duration=duration,
        )

    def plan_brick_trajectory(self, brick_pose: np.ndarray, duration: float = 25.0):
        """Approach/deposit a brick with the right arm."""
        left_start, right_start = self._current_ee()
        return self.planner.plan_through_waypoints(
            [left_start, left_start, left_start],
            [right_start,
             _brick_to_grasp_pose(brick_pose, HAND_PREPICK_OFFSET),
             _brick_to_grasp_pose(brick_pose, HAND_PICK_OFFSET)],
            duration=duration,
        )

    # -----------------------------------------------------------------------
    # Execution
    # -----------------------------------------------------------------------

    def execute_trajectory(self, traj, free_joints=None):
        """Execute trajectory; returns (success, stats)."""
        if free_joints is None:
            free_joints = self._right_arm_free_joints()
        success = self.controller.execute(traj, free_joints=free_joints)
        return success, self.controller.get_stats()

    def open_hand(self):
        """Open right hand; left hand stays closed (left arm disabled)."""
        self.dds.set_hand_mode(left="close", right="open")

    def grasp_hand(self):
        """Grasp with right hand; left hand stays closed (left arm disabled)."""
        self.dds.set_hand_mode(left="close", right="grasp")

    def close_hand(self):
        """Close with right hand; left hand stays closed (left arm disabled)."""
        self.dds.set_hand_mode(left="close", right="close")

    def center_joints(self, duration: float = 2.0) -> bool:
        return self.controller.execute_joint_interpolation(self.q_start, duration=duration)

    def center_waist(self, duration: float = 1.0) -> bool:
        """Interpolate waist yaw to 0 while holding all other joints fixed."""
        q_current, _ = self.dds.get_upper_body_state()
        q_target = q_current.copy()
        q_target[0] = 0.0
        return self.controller.execute_joint_interpolation(q_target, duration=duration)

    def safe_shutdown(self):
        """
        Error-recovery shutdown:
          1. Plan and execute a trajectory back to the safe (init) pose.
          2. Close hands.
          3. Joint-interpolate back to home configuration.
          4. Shut down DDS.
        Step 1 is best-effort — if it fails the remaining steps still run.
        """
        print("\nSafe shutdown: returning to safe pose...")
        self.dds.set_hand_mode("close", "close")
        time.sleep(1.0)
        try:
            traj = self.plan_init_trajectory()
            free_joints = self.ik_solver.right_q_idx + self.ik_solver.waist_q_idx
            self.controller.execute(traj, free_joints=free_joints)
        except Exception as e:
            print(f"  Safe-pose trajectory failed ({e}) — skipping.")
        print("Safe shutdown: centering joints...")
        self.center_joints(duration=3.0)
        print("Safe shutdown complete.")
        self.shutdown()

    def shutdown(self):
        self.camera.stop()
        self.dds.shutdown()


class G1_Bricklaying_Interface:
    
    def init(self):
         print("\nInitializing PickPlace demo...")
         log_dir = Path(__file__).parents[1] / "logs" / datetime.now().strftime("%Y%m%d_%H%M%S")
         log_dir.mkdir(parents=True, exist_ok=True)
         print(f"Logging to {log_dir}")

         self.demo = PickPlace()
         self.twist_publisher_node = G1TwistCmdNode(self.demo.network_interface)

         # For init/return: waist locked (arm-only); for pick/place/lift: waist free
         self.init_free_joints = self.demo.ik_solver.right_q_idx
         self.pick_free_joints = self.demo.ik_solver.right_q_idx + self.demo.ik_solver.waist_q_idx

         input("\nPress Enter to move right arm to init pose...")

         traj_init = self.demo.plan_init_trajectory()
         print(f"Planned: {traj_init.n_waypoints} waypoints, {traj_init.duration:.2f}s")

         success, stats = self.demo.execute_trajectory(traj_init, free_joints=self.init_free_joints)
         if not success:
            print("Init trajectory failed — aborting.")
            self.demo.safe_shutdown()
         print("Right arm at init pose.")

         print("\nOpening right hand...")
         self.demo.open_hand()
         time.sleep(1.0)
         return
        
    def pick(self):
        self.demo.center_waist()
        poses = self.demo.estimate_brick()
        print(f"  Poses: {len(poses)}")

        # Right arm only — discard bricks on the left side (y > 0)
        n_before = len(poses)
        poses = [p for p in poses if p.position[1] <= 0]
        if len(poses) < n_before:
            print(f"  Filtered out {n_before - len(poses)} left-side brick(s) (y > 0) — right arm only.")
        if not poses:
            print("  No right-side bricks found — reposition and try again.")
            return

        pose = self.demo.select_best_pose(poses)
        print(f"  Best pose: pos={np.round(pose.position, 3)}, "
              f"fitness={pose.icp_fitness:.3f}, rmse={pose.icp_rmse:.4f}, "
              f"dist={np.linalg.norm(pose.position):.2f}m")

        pickable, sd = self.demo.is_pickable(pose.transform)
        print(f"  Reach signed distance: {sd:.4f}  ({'reachable' if pickable else 'NOT reachable'})")
        if not pickable:
            print("  Brick not in reachable workspace — reposition and try again.")
            return
        T_pelvis_to_brick = pose.transform.copy()
        traj_pick = self.demo.plan_brick_trajectory(T_pelvis_to_brick)
        print(f"  Pick trajectory: {traj_pick.n_waypoints} waypoints, {traj_pick.duration:.2f}s")

        input("Press Enter to execute pick...")
        success, stats = self.demo.execute_trajectory(traj_pick, free_joints=self.pick_free_joints)
        #_save_trajectory(log_dir / f"brick_{brick_idx:02d}_pick.npz", traj_pick, T_pelvis_to_brick, stats)
        if not success:
            print("  Pick trajectory failed — aborting.")
            self.demo.safe_shutdown()
            return

        print("  Grasping hand...")
        self.demo.grasp_hand()
        time.sleep(0.5)

        print("  Returning arm to safe pose...")
        traj_pick_return = traj_pick.reverse()
        success, stats = self.demo.execute_trajectory(traj_pick_return, free_joints=self.pick_free_joints)
        #_save_trajectory(log_dir / f"brick_{brick_idx:02d}_pick_return.npz", traj_pick_return, T_pelvis_to_brick, stats)
        if not success:
            print("  Return pick trajectory failed — aborting.")
            self.demo.safe_shutdown()
            quit()

        print("  Centering waist...")
        self.demo.center_waist()
        print("  Brick in hand.")
        
        return
    
    def place(self, position_arr):
        T_pelvis_to_brick = None

        self.demo.center_waist()
        loc = self.demo.localize_table()
        if loc is None:
            print("  No ArUco markers visible — reposition and try again.")
            return

        T_pelvis_to_table, loc_result = loc
        print(f"  Localized: {loc_result.n_markers_used} markers used, "
              f"reprojection error = {loc_result.reprojection_error:.2f} px")

        T_pelvis_to_brick = T_pelvis_to_table @ _brick_to_table_pose(position_arr)

        pickable, sd = self.demo.is_pickable(T_pelvis_to_brick)
        print(f"  Reach signed distance: {sd:.4f}  ({'reachable' if pickable else 'NOT reachable'})")
        if not pickable:
           print("  Place target out of reach — reposition robot and try again.")
           return
           

        # ── PLACE ────────────────────────────────────────────────────
        input(f"\nPress Enter to place brick at table "
          f"{np.round(position_arr, 3)}...")

        traj_place = self.demo.plan_brick_trajectory(T_pelvis_to_brick)
        print(f"  Place trajectory: {traj_place.n_waypoints} waypoints, {traj_place.duration:.2f}s")

        success, stats = self.demo.execute_trajectory(traj_place, free_joints=self.pick_free_joints)
        #_save_trajectory(log_dir / f"brick_{brick_idx:02d}_place.npz", traj_place, T_pelvis_to_brick, stats)
        
        if not success:
            print("  Place trajectory failed — aborting.")
            self.demo.safe_shutdown()
            quit()

        print("  Opening hand...")
        self.demo.open_hand()
        time.sleep(0.5)

        print("  Returning arm to safe pose...")
        traj_place_return = traj_place.reverse()
        success, stats = self.demo.execute_trajectory(traj_place_return, free_joints=self.pick_free_joints)
        #_save_trajectory(log_dir / f"brick_{brick_idx:02d}_place_return.npz", traj_place_return, T_pelvis_to_brick, stats)
        if not success:
            print("  Return from place failed — aborting.")
            self.demo.safe_shutdown()
            quit

        # Per loop reset -- make sure we're at init location
        print("Re-setting right arm back to init pose...")
        traj_init = self.demo.plan_init_trajectory()
        print(f"Planned: {traj_init.n_waypoints} waypoints, {traj_init.duration:.2f}s")

        success, _ = self.demo.execute_trajectory(traj_init, free_joints=self.init_free_joints)
        if not success:
            print("Init trajectory failed — aborting.")
            self.demo.safe_shutdown()
            quit()
        print("Right arm at init pose.")
        return
    
    def _publish_motion_pulse(self, x: float, theta: float, duration: float = 0.5):
        self.twist_publisher_node.publish_twist(x, theta)
        time.sleep(duration)
        self.twist_publisher_node.publish_twist(0, 0)

    def step_forward(self):
        self._publish_motion_pulse(5, 0)

    def step_backward(self):
        self._publish_motion_pulse(-5, 0)

    def turn_left(self):
        self._publish_motion_pulse(0, 5)

    def turn_right(self):
        self._publish_motion_pulse(0, -5)

if __name__ == "__main__":

    interface = G1_Bricklaying_Interface()
    interface.init()
    
    while True:
        key = input("[p]ick [P]lace [w]fwd [s]back [q]left [e]right: ")
        if key == 'p':
            interface.pick()
                 
        elif key == 'P':
            #feel free to pass any position...
            hardcoded_place_position = np.array([ 0.09,  0.0,  0.05,  0.0,  0.0,  25.0])
            interface.place(hardcoded_place_position)
           
        elif key == 'w':
            interface.step_forward()
        elif key == 's':
            interface.step_backward()
        elif key == 'q':
            interface.turn_left()
        elif key == 'e':
            interface.turn_right()
        else:
            print("Invalid key..")
        
