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
from bricklaying.perception import SimRealSenseCamera, RealSenseCamera, BrickPoseEstimator, BrickPose
from bricklaying.segmentation import FastSAMSegmentor


def _pose(pos, rot=np.eye(3)):
    T = np.eye(4)
    T[:3, 3] = pos
    T[:3, :3] = rot
    return T


# Various keypoint poses and offsets for the brick placing task (tunable)
T_LEFT_INIT = _pose([0.15, 0.4, 0.15], np.eye(3))
T_RIGHT_INIT = _pose([0.15, -0.4, 0.15], np.eye(3))

HAND_PICK_OFFSET = np.array([0.02, 0., 0.075])
HAND_PREPICK_OFFSET = np.array([0.02, 0., 0.125])

# Pose selection: fitness penalised by distance so a nearby brick beats a
# marginally better-fit brick that is farther away.
POSE_DISTANCE_WEIGHT = 0.5  # score penalty per metre of distance from pelvis


def canonical_grasp_yaw(T: np.ndarray) -> float:
    """
    Fold brick yaw into (-90°, 90°] to exploit the brick's 180° symmetry
    and avoid wrist-inverted configurations.
    """
    # Regular yaw computation
    R_mat = T[:3, :3]
    yaw, _, _ = R.from_matrix(R_mat).as_euler('zyx')

    # Check vertical symmetry
    if R_mat[2, 2] < 0:
        yaw = -yaw

    # Check angular symmetry
    yaw = yaw % np.pi           # fold into [0°, 180°)
    if yaw > np.pi / 2:
        yaw -= np.pi            # shift to (-90°, 0°]

    return yaw


def build_grasp_rotation(yaw: float, arm: str) -> np.ndarray:
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
    def __init__(self, network_interface: str = "docker0"):
        self.network_interface = network_interface
        self._build()

        # Store initial joint config, for later return
        self.q_start, _ = self.dds.get_upper_body_state()

    def _build(self):
        # Initialize all necessary submodules
        self.dds = DDSInterface(self.network_interface)
        self.camera = SimRealSenseCamera()
        time.sleep(2)
        self.urdf_model = G1URDFModel(reduced=True)
        time.sleep(2)
        self.ik_solver = DualArmIK()
        time.sleep(2)
        self.controller = TrajectoryController(
            dds=self.dds,
            urdf_model=self.urdf_model,
            ik_solver=self.ik_solver,    
        )
        time.sleep(2)
        self.segmentor = FastSAMSegmentor()
        self.estimator = BrickPoseEstimator(
            segmentor=self.segmentor,
            urdf_model=self.urdf_model,
            camera=self.camera,
        )
        self.planner = MotionPlanner()
        self.reach = EllipsoidRegion()

    def detect_and_estimate(self, save_dir: str = "outputs") -> tuple[list[BrickPose], dict]:
        """Capture one frame, run segmentation and pose estimation.

        Returns:
            poses: list of BrickPose
            diagnostics: dict with timing and images saved
        """
        self.camera.flush()  # discard buffered frames to get a current image
        t0 = time.time()
        color, depth = self.camera.get_frames()
        t_cam = time.time() - t0

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
        """Select best pose: ICP fitness penalised by distance from the robot."""
        if not poses:
            return None
        return max(poses, key=lambda p: p.icp_fitness - POSE_DISTANCE_WEIGHT * np.linalg.norm(p.position))
    
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
        q_current, _ = self.dds.get_upper_body_state()
        left_start  = self.urdf_model.get_frame_transform(q_current, "left_ee",  use_reduced=True)
        right_start = self.urdf_model.get_frame_transform(q_current, "right_ee", use_reduced=True)
        traj = self.planner.plan_through_waypoints(
            [left_start, T_LEFT_INIT], [right_start, T_RIGHT_INIT], duration=duration
        )
        return traj
    
    def plan_pick_trajectory(self, pose: np.ndarray, arm: str, duration: float = 30.0):
        """Plan a simple approach and grasp trajectory.

        Assumes brick pose is expressed in the same frame used by the kinematics.
        """
        # Get current joint configuration
        q_current, _ = self.dds.get_upper_body_state()

        # Current end-effector poses
        left_start = self.urdf_model.get_frame_transform(q_current, "left_ee", use_reduced=True)
        right_start = self.urdf_model.get_frame_transform(q_current, "right_ee", use_reduced=True)

        # Extract yaw from estimated brick pose and snap to valid grasp angle
        grasp_yaw = canonical_grasp_yaw(pose)

        # Build grasp and approach poses
        R_grasp = build_grasp_rotation(grasp_yaw, arm)

        # Rotate the EE offsets into the brick's horizontal frame so the
        # approach direction follows the brick's long axis (yaw-only rotation;
        # z-component of the offset is preserved since R_yaw is around z).
        R_yaw = R.from_euler('z', grasp_yaw).as_matrix()
        prepick = _pose(pose[:3, 3] + R_yaw @ HAND_PREPICK_OFFSET, R_grasp)
        pick    = _pose(pose[:3, 3] + R_yaw @ HAND_PICK_OFFSET,    R_grasp)

        if arm == "left":
            left_waypoints  = [left_start, prepick, pick]
            right_waypoints = [right_start.copy() for _ in range(3)]
        else:
            right_waypoints = [right_start, prepick, pick]
            left_waypoints  = [left_start.copy() for _ in range(3)]

        traj = self.planner.plan_through_waypoints(left_waypoints, right_waypoints, duration=duration)
        return traj
    
    def plan_up_trajectory(self, arm: str, duration: float = 15.0):
        # Get current joint configuration
        q_current, _ = self.dds.get_upper_body_state()

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

    def actuate_hand(self, left: str, right: str):
        """Command hands by mode: 'open', 'grasp', or 'close'."""
        self.dds.set_hand_mode(left, right)

    def arm_free_joints(self, arm: str) -> list:
        """q-indices for the active arm + waist; locks the other arm during IK."""
        arm_idx = (self.ik_solver.left_q_idx if arm == "left"
                   else self.ik_solver.right_q_idx)
        return arm_idx + self.ik_solver.waist_q_idx

    def execute_trajectory(self, traj, free_joints=None):
        """Execute trajectory and return boolean success and stats."""
        success = self.controller.execute(traj, free_joints=free_joints)
        stats   = self.controller.get_stats()
        return success, stats
    
    def center_joints(self, duration: float = 3.0) -> bool:
        return self.controller.execute_joint_interpolation(self.q_start, duration=duration)
    
    def safe_shutdown(self, arm: str = "right"):
        """
        Error-recovery shutdown:
          1. Plan and execute a trajectory back to the safe (init) pose.
          2. Close hands.
          3. Joint-interpolate back to home configuration.
          4. Shut down DDS.
        Step 1 is best-effort — if it fails the remaining steps still run.
        """
        print("\nSafe shutdown: returning to safe pose...")
        free_joints = self.arm_free_joints(arm)
        try:
            traj = self.plan_initial_trajectory()
            self.controller.execute(traj, free_joints=free_joints)
        except Exception as e:
            print(f"  Safe-pose trajectory failed ({e}) — skipping.")
        self.dds.set_hand_mode("close", "close")
        time.sleep(0.5)
        print("Safe shutdown: centering joints...")
        self.center_joints(duration=3.0)
        print("Safe shutdown complete.")
        self.shutdown()

    def center_waist(self, duration: float = 1.5) -> bool:
        """Interpolate waist yaw to 0 while holding all other joints fixed."""
        q_current, _ = self.dds.get_upper_body_state()
        q_target = q_current.copy()
        q_target[0] = 0.0
        return self.controller.execute_joint_interpolation(q_target, duration=duration)

    def shutdown(self):
        self.dds.shutdown()


def main():
    import os
    save_dir = "outputs"
    os.makedirs(save_dir, exist_ok=True)

    # ---------------------------------------------
    # Init
    # ---------------------------------------------
    print("\nInitializing...")
    demo = PickBrick()

    # Left arm disabled (hardware fault) — right arm only.
    arm = "right"
    init_free_joints = demo.ik_solver.right_q_idx  # no waist for init
    free_joints = demo.ik_solver.right_q_idx + demo.ik_solver.waist_q_idx  # waist free for pick

    input("Press Enter to plan and execute an initial trajectory...")

    traj_init = demo.plan_initial_trajectory()
    print(f"Planned trajectory: {traj_init.n_waypoints} waypoints over {traj_init.duration:.2f}s")

    print("Executing initial trajectory...")
    success, stats = demo.execute_trajectory(traj_init, free_joints=init_free_joints)
    if not success:
        print("Trajectory failed — aborting")
        demo.safe_shutdown(arm)
        return
    print("Trajectory executed.")

    print("Actuating hand to open...")
    demo.actuate_hand(left="close", right="open")
    time.sleep(1.0)
    print("Hands opened.")

    # ---------------------------------------------
    # Brick detection (retry loop)
    # ---------------------------------------------
    # Step 2: Detect, filter to right side, check reachability.
    # Operator can reposition the robot between attempts; 'q' aborts.
    pose = None
    while True:
        cmd = input("\nPress Enter to detect brick, or 'q' to abort: ")
        if cmd.strip().lower() == 'q':
            demo.safe_shutdown(arm)
            return

        print("Centering waist...")
        demo.center_waist()
        poses, diag = demo.detect_and_estimate(save_dir=save_dir)
        print(f"Detections: {diag['n_detections']}, poses: {diag['n_poses']}  "
              f"({diag['segmentation_time_s']:.2f}s seg, {diag['estimation_time_s']:.2f}s est)")

        # Right arm only — discard bricks on the left side (y > 0)
        n_before = len(poses)
        poses = [p for p in poses if p.position[1] <= 0]
        if len(poses) < n_before:
            print(f"Filtered out {n_before - len(poses)} left-side brick(s) (y > 0) — right arm only.")
        if not poses:
            print("No right-side bricks found — reposition and try again.")
            continue

        pose = demo.select_best_pose(poses)
        print(f"Best pose: pos={np.round(pose.position, 3)}, fitness={pose.icp_fitness:.3f}, "
              f"rmse={pose.icp_rmse:.4f}, dist={np.linalg.norm(pose.position):.2f}m")

        T_check = pose.transform.copy()
        T_check[:3, 3] += R.from_euler('z', canonical_grasp_yaw(T_check)).as_matrix() @ HAND_PICK_OFFSET
        pickable, sd = demo.is_pickable(T_check)
        print(f"Reach signed distance: {sd:.4f}  ({'reachable' if pickable else 'NOT reachable'})")
        if not pickable:
            print("Brick not in reachable workspace — reposition and try again.")
            continue

        break

    # ---------------------------------------------
    # Picking trajectory
    # ---------------------------------------------
    # Step 3: Plan trajectory, execute, and grasp
    input("\nPress Enter to plan and execute pick trajectory...")

    T = pose.transform.copy()
    traj = demo.plan_pick_trajectory(T, arm)
    print(f"Planned trajectory: {traj.n_waypoints} waypoints over {traj.duration:.2f}s")

    print("Executing approach and pick...")
    success, stats = demo.execute_trajectory(traj, free_joints=free_joints)
    if not success:
        print("Trajectory failed — aborting")
        demo.safe_shutdown(arm)
        return
    print("Trajectory executed.")

    print("Actuating hand to close...")
    demo.actuate_hand(left="close", right="grasp")
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
    success, stats = demo.execute_trajectory(up_traj, free_joints=free_joints)
    if not success:
        print("Trajectory failed — aborting")
        demo.safe_shutdown(arm)
        return
    print("Trajectory executed.")

    print("Executing down trajectory...")
    success, stats = demo.execute_trajectory(up_traj.reverse(), free_joints=free_joints)
    if not success:
        print("Trajectory failed — aborting")
        demo.safe_shutdown(arm)
        return
    print("Trajectory executed.")

    print("Actuating hand to open...")
    demo.actuate_hand(left="close", right="open")
    time.sleep(1.0)
    print("Hands opened.")

    # ---------------------------------------------
    # Return and shut down
    # ---------------------------------------------
    # Step 5: Return to home and center the joints
    input("Press Enter to plan and execute return trajectory....")

    print("Executing return trajectory...")
    success, stats = demo.execute_trajectory(traj.reverse(), free_joints=free_joints)
    if not success:
        print("Return failed — aborting")
        demo.safe_shutdown(arm)
        return
    print("Trajectory executed.")

    print("Actuating hand to close...")
    demo.actuate_hand(left="close", right="close")
    time.sleep(1.0)
    print("Hands closed.")

    print("Executing pre-shutdown trajectory...")
    success, stats = demo.execute_trajectory(traj_init.reverse(), free_joints=free_joints)
    if not success:
        print("Return failed — aborting")
        demo.safe_shutdown(arm)
        return
    print("Trajectory executed.")

    print("Centering joints...")
    demo.center_joints(duration=1.0)
    print("Joints centered.")

    print("Demo finished.")
    demo.shutdown()


if __name__ == "__main__":
    main()
