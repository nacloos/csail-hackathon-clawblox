"""
Simple demonstration for picking brick.
"""
from __future__ import annotations

import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

from bricklaying.robot import DDSInterface, G1URDFModel, DualArmIK, TrajectoryController
from bricklaying.planning import MotionPlanner, EllipsoidRegion, R_LEFT_NOMINAL_PICK, R_RIGHT_NOMINAL_PICK
from bricklaying.perception import RealSenseCamera, BrickPoseEstimator, BrickPose
from bricklaying.segmentation import FastSAMSegmentor


def _pose(pos, R):
    T = np.eye(4)
    T[:3, 3] = pos
    T[:3, :3] = R
    return T


# Various keypoint poses and offsets for the brick placing task (tunable)
T_LEFT_INIT = _pose([0.15, 0.4, 0.15], np.eye(3))
T_RIGHT_INIT = _pose([0.15, -0.4, 0.15], np.eye(3))

HAND_PICK_OFFSET = np.array([-0.05, 0., 0.075])
HAND_PREPICK_OFFSET = np.array([-0.05, 0., 0.11])


def _extract_yaw(T: np.ndarray) -> float:
    """Extract yaw (rotation about coord Z) from a transform."""
    yaw, _, _ = R.from_matrix(T[:3, :3]).as_euler('zyx')
    return float(yaw)


def _canonical_grasp_yaw(yaw: float) -> float:
    """
    Fold brick yaw into (-90°, 90°] to exploit the brick's 180° symmetry
    and avoid wrist-inverted configurations.
    """
    yaw = yaw % np.pi           # fold into [0°, 180°)
    if yaw > np.pi / 2:
        yaw -= np.pi            # shift to (-90°, 0°]
    return yaw


def _build_grasp_rotation(yaw: float, arm: str) -> np.ndarray:
    """
    Build the grasp rotation matrix for a given canonical brick yaw and arm.

    Composes:
      1. Rz(yaw)  — align hand to brick's long axis
      2. Rx(±45°) — tilt wrist down for a top-down approach
    """
    R_yaw = R.from_euler('z', yaw).as_matrix()
    R_wrist = R_LEFT_NOMINAL_PICK if arm == "left" else R_RIGHT_NOMINAL_PICK
    return R_yaw @ R_wrist


class PickBrick:
    def __init__(self, network_interface: str = "eth0"):
        self.network_interface = network_interface
        self._build()

        # Store initial joint config, for later return
        self.q_start, _ = self.dds.get_arm_state()

    def _build(self):
        # Initialize all necessary submodules
        self.dds = DDSInterface(self.network_interface)
        self.camera = RealSenseCamera()
        self.urdf_model = G1URDFModel(reduced=True)
        self.ik_solver = DualArmIK()
        self.controller = TrajectoryController(
            dds=self.dds,
            urdf_model=self.urdf_model,
            ik_solver=self.ik_solver,    
        )
        self.segmentor = FastSAMSegmentor()
        self.estimator = BrickPoseEstimator(
            segmentor=self.segmentor,
            urdf_model=self.urdf_model,
            camera=self.camera,
        )
        self.planner = MotionPlanner()
        self.reach = EllipsoidRegion()

    def detect_and_estimate(self, save_dir: str = "outputs", timeout: float = 5.0) -> tuple[list[BrickPose], dict]:
        """Capture one frame, run segmentation and pose estimation.

        Returns:
            poses: list of BrickPose
            diagnostics: dict with timing and images saved
        """
        t0 = time.time()
        color, depth = self.camera.get_frames()
        t_cam = time.time() - t0
fest
        # segmentation
        t1 = time.time()
        detections = self.segmentor.segment(color)
        t_seg = time.time() - t1

        # visualize detections
        try:
            from bricklaying.segmentation import visualize_detections
            vis = visualize_detections(color, detections)
            plt.imsave(f"{save_dir}/segmentation_overlay.png", vis)
        except Exception:
            vis = color

        # run pose estimator
        t2 = time.time()
        poses = self.estimator.estimate_from_frame(
            color, depth, self.camera.intrinsics, detections=detections
        )
        t_est = time.time() - t2

        diag = {
            "camera_time_s": t_cam,
            "segmentation_time_s": t_seg,
            "estimation_time_s": t_est,
            "n_detections": len(detections),
            "n_poses": len(poses),
        }

        # save depth and rgb for debugging
        try:
            import cv2
            cv2.imwrite(f"{save_dir}/rgb.png", cv2.cvtColor(color, cv2.COLOR_RGB2BGR))
            # depth is uint16
            cv2.imwrite(f"{save_dir}/depth.png", depth)
        except Exception:
            pass

        return poses, diag

    def select_best_pose(self, poses: list[BrickPose]) -> BrickPose:
        """Choose best pose based on ICP fitness and RMSE."""
        if not poses:
            return None
        # prefer highest fitness, break ties by lower rmse
        poses_sorted = sorted(poses, key=lambda p: (-p.icp_fitness, p.icp_rmse))
        return poses_sorted[0]
    
    def is_pickable(self, pose: np.ndarray) -> tuple[bool, float]:
        """Check reachability using EllipsoidRegion. Returns (pickable, signed_distance)."""
        if pose is None:
            return False, float('inf')
        sd = self.reach.signed_distance(pose[:3, 3])
        return sd <= 0.0, sd
    
    def determine_arm(self, pose: np.ndarray) -> str:
        """Choose left arm if brick on left side, right arm if brick on right side."""
        return "left" if pose[1, 3] > 0 else "right"

    def plan_initial_trajectory(self, duration: float = 3.0):
        """Plan trajectory from current arm position to the nominal init pose."""
        q_current, _ = self.dds.get_arm_state()
        left_start  = self.urdf_model.get_frame_transform(q_current, "left_ee",  use_reduced=True)
        right_start = self.urdf_model.get_frame_transform(q_current, "right_ee", use_reduced=True)
        traj = self.planner.plan_through_waypoints(
            [left_start, T_LEFT_INIT], [right_start, T_RIGHT_INIT], duration=duration
        )
        return traj
    
    def plan_pick_trajectory(self, pose: np.ndarray, arm: str, duration: float = 6.0):
        """Plan a simple approach and grasp trajectory.

        Assumes brick pose is expressed in the same frame used by the kinematics.
        """
        # Get current joint configuration
        q_current, _ = self.dds.get_arm_state()

        # Current end-effector poses
        left_start = self.urdf_model.get_frame_transform(q_current, "left_ee", use_reduced=True)
        right_start = self.urdf_model.get_frame_transform(q_current, "right_ee", use_reduced=True)

        # Extract yaw from estimated brick pose and snap to valid grasp angle
        raw_yaw = _extract_yaw(pose)
        grasp_yaw = _canonical_grasp_yaw(raw_yaw)

        # Build grasp and approach poses
        R_grasp = _build_grasp_rotation(grasp_yaw, arm)

        prepick = _pose(pose[:3, 3] + HAND_PREPICK_OFFSET, R_grasp)
        pick = _pose(pose[:3, 3] + HAND_PICK_OFFSET, R_grasp)

        if arm == "left":
            left_waypoints  = [left_start, prepick, pick]
            right_waypoints = [right_start] * 3
        else:
            right_waypoints = [right_start, prepick, pick]
            left_waypoints  = [left_start] * 3

        traj = self.planner.plan_through_waypoints(left_waypoints, right_waypoints, duration=duration)
        return traj
    
    def plan_up_trajectory(self, arm: str, duration: float = 1.0):
        # Get current joint configuration
        q_current, _ = self.dds.get_arm_state()

        # Current end-effector poses
        left_start = self.urdf_model.get_frame_transform(q_current, "left_ee", use_reduced=True)
        right_start = self.urdf_model.get_frame_transform(q_current, "right_ee", use_reduced=True)

        z_offset = 0.1
        left_up = left_start.copy()
        left_up[2, 3] += z_offset
        right_up = right_start.copy()
        right_up[2, 3] += z_offset

        if arm == "left":
            left_waypoints  = [left_start, left_up]
            right_waypoints = [right_start] * 2
        else:
            right_waypoints = [right_start, right_up]
            left_waypoints  = [left_start] * 2

        traj = self.planner.plan_through_waypoints(left_waypoints, right_waypoints, duration=duration)
        return traj

    def actuate_hand(self, left: bool, right: bool):
        """Write open (False) / close (True) commands to hands."""
        self.dds.set_hand_binary(left, right)

    def execute_trajectory(self, traj):
        """Execute trajectory and return boolean success and stats."""
        success = self.controller.execute(traj)
        stats   = self.controller.get_stats()
        return success, stats
    
    def center_joints(self, duration: float = 2.0) -> bool:
        self.controller.execute_joint_interpolation(self.q_start, duration=duration)
    
    def shutdown(self):
        self.dds.shutdown()
    

def main():
    # ---------------------------------------------
    # Init
    # ---------------------------------------------
    # Step 1: Move to an initializattion pose
    print("\nInitializing...")
    demo = PickBrick()

    input("Press Enter to plan and execute an initial trajectory...")

    traj_init = demo.plan_initial_trajectory()
    print(f"Planned trajectory: {traj_init.n_waypoints} waypoints over {traj_init.duration:.2f}s")

    print("Executing initial trajectory...")
    success, stats = demo.execute_trajectory(traj_init)
    if not success:
        print("Trajectory failed — aborting")
        return
    print("Trajectory executed.")

    print("Actuating hand to open...")
    demo.actuate_hand(left=False, right=False)
    time.sleep(1.0)
    print("Hands opened.")


    # MULTIPLE TRIALS 

    while True:
    # ---------------------------------------------
    # Brick estimation
    # ---------------------------------------------
    # Step 2: Identify and estimate brick. Transform to robot frame. Check reachability.
        max_detect_attempts = 3
        pose = None

        for attempt in range(1, max_detect_attempts + 1):
            input(f"Press Enter to detect brick (attempt {attempt}/{max_detect_attempts})...")

            poses, diag = demo.detect_and_estimate(save_dir="outputs")
            print(f"Detections: {diag['n_detections']}, poses: {diag['n_poses']}")
            print(f"Camera: {diag['camera_time_s']:.3f}s | Seg: {diag['segmentation_time_s']}s | Est: {diag['estimation_time_s']}s")

            candidate = demo.select_best_pose(poses)
            if candidate is None:
                print("No pose estimated — retrying.")
                continue

            T_check = candidate.transform.copy()
            T_check[:3, 3] += HAND_PICK_OFFSET
            pickable, sd = demo.is_pickable(T_check)
            print(f"Reach signed distance: {sd:.4f}")

            if not pickable:
                print("Brick not reachable — retrying.")
                continue

            pose = candidate
            print(f"Best pose: pos={pose.position}, fitness={pose.icp_fitness:.3f}, rmse={pose.icp_rmse:.4f}")
            break

        if pose is None:
            print(f"Failed to find a reachable brick after {max_detect_attempts} attempts — skipping trial.")
            return
        print(f"Best pose: pos={pose.position}, fitness={pose.icp_fitness:.3f}, \
            rmse={pose.icp_rmse:.4f}, confidence={pose.detection.confidence:.4f}")

        T_check = pose.transform.copy()
        T_check[:3, 3] += HAND_PICK_OFFSET

        pickable, sd = demo.is_pickable(T_check)
        print(f"Reach signed distance: {sd:.4f} (<=0 means inside reach)")
        if not pickable:
            print("Brick not in reachable workspace. Aborting demo.")
            return

        # ---------------------------------------------
        # Picking trajectory
        # ---------------------------------------------
        # Step 3: Plan trajectory, execute, and grasp
        input("Press Enter to plan and execute pick trajectory...")

        T = pose.transform.copy()
        arm = demo.determine_arm(T)
        traj = demo.plan_pick_trajectory(T, arm)
        print(f"Planned trajectory: {traj.n_waypoints} waypoints over {traj.duration:.2f}s")

        print("Executing approach and pick...")
        success, stats = demo.execute_trajectory(traj)
        if not success:
            print("Trajectory failed — aborting")
            return
        print("Trajectory executed.")
        
        print("Actuating hand to close...")
        demo.actuate_hand(left=True, right=True)
        time.sleep(1.0)
        print("Pick complete.")

        # ---------------------------------------------
        # Intermediate trajectory
        # ---------------------------------------------
        # Step 4: Raise and lower arm (to demonstrate grasp), then open hand
        input("Press Enter to plan and execute a raising trajectory...")

        up_traj = demo.plan_up_trajectory(arm)
        print(f"Planned trajectory: {up_traj.n_waypoints} waypoints over {up_traj.duration:.2f}s")

        print("Executing up trajectory...")
        success, stats = demo.execute_trajectory(up_traj)
        if not success:
            print("Trajectory failed — aborting")
            return
        print("Trajectory executed.")

        print("Executing down trajectory...")
        success, stats = demo.execute_trajectory(up_traj.reverse())
        if not success:
            print("Trajectory failed — aborting")
            return
        print("Trajectory executed.")

        print("Actuating hand to open...")
        demo.actuate_hand(left=False, right=False)
        time.sleep(1.0)
        print("Hands opened.")

        # ---------------------------------------------
        # Return and shut down
        # ---------------------------------------------
        # Step 5: Return to home and center the joints
        input("Press Enter to plan and execute return trajectory....")
        
        print("Executing return trajectory...")
        success, stats = demo.execute_trajectory(traj.reverse())
        if not success:
            print("Return failed — aborting")
            return
        print("Trajectory executed.")

        # Re-plan from current state to avoid joint step errors
        traj_return = demo.plan_initial_trajectory()
        print("Executing return to init...")
        success, stats = demo.execute_trajectory(traj_return)
        if not success:
            print("Recovery to init failed.")
            return
        print("Replan to init pos successful")
        again = input("Run another trial? [y/N]: ").strip().lower()
        if again != 'y':
            break

    print("Actuating hand to close...")
    demo.actuate_hand(left=True, right=True)
    time.sleep(1.0)
    print("Hands closed.")

    print("Executing pre-shutdown trajectory...")
    success, stats = demo.execute_trajectory(traj_init.reverse())
    if not success:
        print("Return failed — aborting")
        return
    print("Trajectory executed.")

    print("Centering joints...")
    demo.center_joints(duration=1.0)
    print("Joints centered.")

    print("Demo finished.")
    demo.shutdown()
    return


if __name__ == "__main__":
    main()
