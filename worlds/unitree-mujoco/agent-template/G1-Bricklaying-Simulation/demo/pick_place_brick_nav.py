"""
Right-arm-only pick-and-place demo using ArUco table localization.
Left arm is locked throughout (hardware fault).

Flow for each brick:
  1. Detect brick pose with camera (segmentation + ICP).
  2. Pick with right arm.
  -- interlude: remote control navigation
  3. Localize table with ArUco markers.
  4. Open-loop place at hardcoded target.
  -- interlude: remote control navigation
  5. Repeat

"""
from __future__ import annotations

import os
import time
import numpy as np
from scipy.spatial.transform import Rotation as R
from typing import Optional

from bricklaying.robot import DDSInterface, NavDDSInterface, G1URDFModel, DualArmIK, TrajectoryController
from bricklaying.robot import T_CAMERA_TO_REALSENSE
from bricklaying.planning import MotionPlanner, EllipsoidRegion, R_RIGHT_NOMINAL_PICK
from bricklaying.perception import (
    RealSenseCamera, BrickPoseEstimator, BrickPose,
    ArucoLocalizer, TablePoseResult,
)
from bricklaying.segmentation import FastSAMSegmentor


def _pose(pos, rot=np.eye(3)):
    T = np.eye(4)
    T[:3, 3] = pos
    T[:3, :3] = rot
    return T


# Safe home pose for right EE throughout operation
T_RIGHT_INIT = _pose([0.15, -0.4, 0.15], np.eye(3))

# Pick/place offsets, wrt brick frame
HAND_PICK_OFFSET    = np.array([-0.025, 0., 0.08])
HAND_PREPICK_OFFSET = np.array([-0.025, 0., 0.20])

# Priority for pose estimation
POSE_DISTANCE_WEIGHT = 0.3  # score penalty per meter of distance from pelvis

# Pre-defined structure
PLACE_X = 0.09  # distance from table front edge [m]
PLACE_Z = 0.04  # brick center height above table surface [m]
BRICK_H = 0.06  # brick height [m]

PLACEMENT_SEQUENCE = [
    # # First layer
    np.array([PLACE_X,  0.375, PLACE_Z]),
    np.array([PLACE_X,  0.125, PLACE_Z]),
    np.array([PLACE_X, -0.125, PLACE_Z]),
    np.array([PLACE_X, -0.375, PLACE_Z]),
    # Second layer
    #np.array([PLACE_X,  0.25, PLACE_Z + BRICK_H]),
    #np.array([PLACE_X,  0.00, PLACE_Z + BRICK_H]),
    #np.array([PLACE_X, -0.25, PLACE_Z + BRICK_H])
]


def canonical_grasp_yaw(T: np.ndarray) -> float:
    """Extract and fold brick yaw into (-90°, 90°] from a 4x4 transform."""
    R_mat = T[:3, :3]
    yaw, _, _ = R.from_matrix(R_mat).as_euler('zyx')
    if R_mat[2, 2] < 0:
        yaw = -yaw
    yaw = yaw % np.pi
    if yaw > np.pi / 2:
        yaw -= np.pi
    return yaw


# !! ONLY for right hand !!
def build_grasp_rotation(yaw: float) -> np.ndarray:
    """Rz(yaw) @ R_RIGHT_NOMINAL_PICK."""
    return R.from_euler('z', yaw).as_matrix() @ R_RIGHT_NOMINAL_PICK


class PickPlaceBrick:
    """
    Orchestrates right-arm-only pick-and-place with ArUco table localization.
    """

    def __init__(self, network_interface: str = "eth0"):
        self.network_interface = network_interface
        self._build()
        self.q_start, _ = self.dds.get_upper_body_state()

    def _build(self):
        self.dds        = DDSInterface(self.network_interface)
        self.nav        = NavDDSInterface("wlan0", 1)
        self.urdf_model = G1URDFModel(reduced=True)
        self.ik_solver  = DualArmIK()
        self.controller = TrajectoryController(
            dds=self.dds,
            urdf_model=self.urdf_model,
            ik_solver=self.ik_solver,
        )
        self.segmentor  = FastSAMSegmentor()
        self.camera     = RealSenseCamera()
        self.estimator  = BrickPoseEstimator(
            segmentor=self.segmentor,
            urdf_model=self.urdf_model,
            camera=self.camera,
        )
        self.planner    = MotionPlanner()
        self.reach      = EllipsoidRegion()
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

    def is_pickable(self, pose: np.ndarray) -> tuple[bool, float]:
        """EllipsoidRegion reachability check. Returns (pickable, signed_distance)."""
        if pose is None:
            return False, float('inf')
        sd = self.reach.signed_distance(pose[:3, 3])
        return sd <= 0.0, sd
        
    def nav_goto(self, x, y, z, w):
        self.nav.send_nav_goto(float(),float(),float(),float())
    
    def nav_get_pos(self):
        return self.nav.get_nav_pos()
        
    def is_nav_healthy(self):
        return self.nav.is_state_healthy()

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

    def plan_init_trajectory(self, duration: float = 3.0):
        """Move right arm to its nominal ready pose; left arm held fixed."""
        left_start, right_start = self._current_ee()
        return self.planner.plan_through_waypoints(
            [left_start,  left_start],
            [right_start, T_RIGHT_INIT],
            duration=duration,
        )

    def plan_pick_trajectory(self, brick_pose: np.ndarray, duration: float = 6.0):
        """Approach and grasp a brick with the right arm."""
        left_start, right_start = self._current_ee()

        grasp_yaw = canonical_grasp_yaw(brick_pose)
        R_grasp   = build_grasp_rotation(grasp_yaw)
        R_yaw     = R.from_euler('z', grasp_yaw).as_matrix()

        p_brick  = brick_pose[:3, 3]
        prepick  = _pose(p_brick + R_yaw @ HAND_PREPICK_OFFSET, R_grasp)
        pick     = _pose(p_brick + R_yaw @ HAND_PICK_OFFSET,    R_grasp)

        return self.planner.plan_through_waypoints(
            [left_start, left_start, left_start],
            [right_start, prepick, pick],
            duration=duration,
        )

    def plan_place_trajectory(self, place_pose: np.ndarray, duration: float = 6.0):
        """Approach and deposit a brick at the given table-frame pose."""
        left_start, right_start = self._current_ee()

        grasp_yaw = canonical_grasp_yaw(place_pose)
        R_grasp   = build_grasp_rotation(grasp_yaw)
        R_yaw     = R.from_euler('z', grasp_yaw).as_matrix()

        p_place   = place_pose[:3, 3]
        preplace  = _pose(p_place + R_yaw @ HAND_PREPICK_OFFSET, R_grasp)
        place     = _pose(p_place + R_yaw @ HAND_PICK_OFFSET,    R_grasp)

        return self.planner.plan_through_waypoints(
            [left_start, left_start, left_start],
            [right_start, preplace, place],
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

    def center_joints(self, duration: float = 3.0) -> bool:
        return self.controller.execute_joint_interpolation(self.q_start, duration=duration)

    def center_waist(self, duration: float = 1.5) -> bool:
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
        time.sleep(0.5)
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
        self.nav.shutdown()


# ---------------------------------------------------------------------------
# Demo entry point
# ---------------------------------------------------------------------------

def main():
    global world_pick_location
    global world_place_location
    # ------------------------------------------------------------------
    # Initialize
    # ------------------------------------------------------------------
    print("\nInitialising PickPlaceBrick demo...")
    demo = PickPlaceBrick()
    
    # For init/return: waist locked (arm-only); for pick/place/lift: waist free
    init_free_joints = demo.ik_solver.right_q_idx
    pick_free_joints = demo.ik_solver.right_q_idx + demo.ik_solver.waist_q_idx
    
    # Flush nav
    flag = 'c'
    while flag != 'x':
        print("cur pos: " + str(demo.nav_get_pos()) + " health: " + str(demo.is_nav_healthy()))
        time.sleep(0.5)
        flag = input("Enter 'x' to confirm getting correct positon data")
    
    while flag != 'a':
        input("Move to place table, then press Enter to set waypoint...")
        world_place_location = demo.nav_get_pos()
        flag = input("Waypoint set to " + str(world_place_location) + " enter 'a' to confirm")
    
    while flag != 'b':
        input("Move to pick table, then press Enter to set waypoint...")
        world_pick_location = demo.nav_get_pos()
        flag = input("Waypoint set to " + str(world_pick_location) + " enter 'b' to confirm")


    input("\nPress Enter to move right arm to init pose...")
    traj_init = demo.plan_init_trajectory()
    print(f"Planned: {traj_init.n_waypoints} waypoints, {traj_init.duration:.2f}s")
    success, _ = demo.execute_trajectory(traj_init, free_joints=init_free_joints)
    if not success:
        print("Init trajectory failed — aborting.")
        demo.safe_shutdown(); return
    print("Right arm at init pose.")

    print("\nOpening right hand...")
    demo.open_hand()
    time.sleep(1.0)

    # ------------------------------------------------------------------
    # Brick loop
    # ------------------------------------------------------------------
    n_bricks = len(PLACEMENT_SEQUENCE)
    for brick_idx, place_pos_table in enumerate(PLACEMENT_SEQUENCE):
        print(f"\n{'='*60}")
        print(f"Brick {brick_idx + 1}/{n_bricks}  |  "
              f"table target = {np.round(place_pos_table, 3)}")
        print('='*60)

        # ── PICK (retry loop) ─────────────────────────────────────────
        # Operator can reposition robot between attempts; 'q' skips this brick.
        T_pelvis_to_brick = None
        while True:
            cmd = input("\nPress Enter to detect brick, or 'q' to skip this brick: ")
            if cmd.strip().lower() == 'q':
                print("  Skipping brick.")
                break

            print("  Centering waist...")
            demo.center_waist()
            t2 = time.time()
            poses = demo.estimate_brick()
            t_est = time.time() - t2
            print(f"  Poses: {len(poses)}")

            # Right arm only — discard bricks on the left side (y > 0)
            n_before = len(poses)
            poses = [p for p in poses if p.position[1] <= 0]
            if len(poses) < n_before:
                print(f"  Filtered out {n_before - len(poses)} left-side brick(s) (y > 0) — right arm only.")
            if not poses:
                print("  No right-side bricks found — reposition and try again.")
                continue

            pose = demo.select_best_pose(poses)
            print(f"  Best pose: pos={np.round(pose.position, 3)}, "
                  f"fitness={pose.icp_fitness:.3f}, rmse={pose.icp_rmse:.4f}, "
                  f"dist={np.linalg.norm(pose.position):.2f}m")

            T_check = pose.transform.copy()
            T_check[:3, 3] += R.from_euler('z', canonical_grasp_yaw(T_check)).as_matrix() @ HAND_PICK_OFFSET
            pickable, sd = demo.is_pickable(T_check)
            print(f"  Reach signed distance: {sd:.4f}  ({'reachable' if pickable else 'NOT reachable'})")
            if not pickable:
                print("  Brick not in reachable workspace — reposition and try again.")
                continue

            T_pelvis_to_brick = pose.transform.copy()
            break

        if T_pelvis_to_brick is None:
            continue  # user skipped this brick

        traj_pick = demo.plan_pick_trajectory(T_pelvis_to_brick)
        print(f"  Pick trajectory: {traj_pick.n_waypoints} waypoints, {traj_pick.duration:.2f}s")

        input("Press Enter to execute pick...")
        success, pick_stats = demo.execute_trajectory(traj_pick, free_joints=pick_free_joints)
        if not success:
            print("  Pick trajectory failed — aborting.")
            demo.safe_shutdown()
            return
        if pick_stats:
            traj_duration = pick_stats.time_meas[-1] - pick_stats.time_meas[0]
            print(f"[TRAJ EXECUTION TIME] {traj_duration}")

        print("  Grasping hand...")
        demo.grasp_hand()
        time.sleep(1.0)

        print("  Returning arm to safe pose...")
        success, _ = demo.execute_trajectory(traj_pick.reverse(), free_joints=pick_free_joints)
        if not success:
            print("  Return pick trajectory failed — aborting.")
            demo.safe_shutdown()
            return

        print("  Centering waist...")
        demo.center_waist()
        print("  Brick in hand.")
        
        # -- Navigate to place table -- #
        print("World place location: " + str(world_place_location))
        input("\nPress Enter to navigate to placement table:  ")
        demo.nav_goto(*world_place_location)

        # ── LOCALIZE (retry loop) ─────────────────────────────────────
        # Operator may reposition the robot between attempts.
        # 'q' drops the brick and triggers safe shutdown.
        T_pelvis_to_place = None
        while True:
            cmd = input("\nPress Enter to localize table, or 'q' to drop brick and abort: ")
            if cmd.strip().lower() == 'q':
                print("  Dropping brick...")
                demo.open_hand()
                time.sleep(1.0)
                try: 
                    times = np.array(pick_stats.time_meas)
                    times = times - times[0]

                    q_ik   = np.array(pick_stats.q_ik)
                    q_meas = np.array(pick_stats.q_meas)

                    left_pose_des, right_pose_des = [], []
                    for t in times:
                        wp = traj_pick.sample(t)
                        left_pose_des.append(wp.left_pose)
                        right_pose_des.append(wp.right_pose)

                    left_pose_des  = np.array(left_pose_des)
                    right_pose_des = np.array(right_pose_des)
                    left_pose_ik    = np.array(pick_stats.left_pose_ik)
                    right_pose_ik   = np.array(pick_stats.right_pose_ik)
                    left_pose_meas  = np.array(pick_stats.left_pose_meas)
                    right_pose_meas = np.array(pick_stats.right_pose_meas)
                    np.savez_compressed("traj_stats.npz", times=times,
                                        q_ik=q_ik,q_meas=q_meas, left_pose_des=left_pose_des, right_pose_des=right_pose_des,
                                        left_pose_ik=left_pose_ik, right_pose_ik=right_pose_ik, left_pose_meas=left_pose_meas,
                                        right_pose_meas=right_pose_meas)
                    print(f"Data saved to traj_stats.npz")
                except:
                    print("Error saving data...")

                demo.safe_shutdown()
                return

            demo.center_waist()
            loc = demo.localize_table()
            if loc is None:
                print("  No ArUco markers visible — reposition and try again.")
                continue

            T_pelvis_to_table, loc_result = loc
            print(f"  Localized: {loc_result.n_markers_used} markers used, "
                  f"reprojection error = {loc_result.reprojection_error:.2f} px")

            T_pelvis_to_place = T_pelvis_to_table @ _pose(place_pos_table, np.eye(3))

            T_check = T_pelvis_to_place.copy()
            offset_ee = R.from_euler('z', canonical_grasp_yaw(T_check)).as_matrix() @ HAND_PICK_OFFSET
            T_check[:3, 3] += offset_ee

            pickable, sd = demo.is_pickable(T_check)
            print(f"  Reach signed distance: {sd:.4f}  ({'reachable' if pickable else 'NOT reachable'})")
            if pickable:
                break
            print("  Place target out of reach — reposition robot and try again.")

        # ── PLACE ────────────────────────────────────────────────────
        input(f"\nPress Enter to place brick {brick_idx + 1} at table "
              f"{np.round(place_pos_table, 3)}...")

        traj_place = demo.plan_place_trajectory(T_pelvis_to_place)
        print(f"  Place trajectory: {traj_place.n_waypoints} waypoints, {traj_place.duration:.2f}s")

        input("Press Enter to execute place...")
        success, _ = demo.execute_trajectory(traj_place, free_joints=pick_free_joints)
        if not success:
            print("  Place trajectory failed — aborting.")
            demo.safe_shutdown()
            return

        print("  Opening hand...")
        demo.open_hand()
        time.sleep(1.0)

        print("  Returning arm to safe pose...")
        success, _ = demo.execute_trajectory(traj_place.reverse(), free_joints=pick_free_joints)
        if not success:
            print("  Return from place failed — aborting.")
            demo.safe_shutdown()
            return
        # -- Navigate to pick table -- #
        input("\nPress Enter to navigate to pick table:  ")
        demo.nav_goto(*world_pick_location)


    # ------------------------------------------------------------------
    # Finish
    # ------------------------------------------------------------------
    print(f"\nAll {n_bricks} bricks placed!")

    input("\nPress Enter to return to rest and shut down...")

    print("  Returning to home pose...")
    success, _ = demo.execute_trajectory(traj_init.reverse(), free_joints=pick_free_joints)
    if not success:
        print("  Return to home failed — aborting.")
        demo.safe_shutdown(); return
    demo.close_hand()
    time.sleep(1.0)
    demo.center_joints(duration=3.0)
    demo.shutdown()


if __name__ == "__main__":
    main()

