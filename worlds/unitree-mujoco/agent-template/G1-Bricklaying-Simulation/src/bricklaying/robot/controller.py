"""
Trajectory execution controller for Unitree G1 robot.
IK is run offline via compute_joint_trajectory(); the real-time loop only
does array lookup + one FK call on q_meas for Cartesian safety checks.

To-do's:
    - check_collisions is broken (set to false always for now)
    - Tests and evals
    - Cleanup joint command to motion control handoff / error handling / shutdown
    - Debug pose estimation (definitely wrong here)
        Ex: Best pose: pos=[  -0.089963   -0.033053     0.54372], fitness=0.718, rmse=0.0115, confidence=0.9766
"""
import time
import numpy as np
import pinocchio as pin
from typing import Optional
from dataclasses import dataclass
from enum import Enum, auto

from bricklaying.planning import CartesianTrajectory
from .dds_interface import DDSInterface
from .urdf_model import G1URDFModel
from .kinematics import DualArmIK

#import pdb
#pdb.set_trace()


def _rotation_angle(R: np.ndarray) -> float:
    """Geodesic angle [rad] from rotation residual R = R_achieved @ R_desired.T."""
    return float(np.arccos(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)))


@dataclass
class JointTrajectory:
    """Precomputed joint-space trajectory produced by offline IK."""
    timestamps:          np.ndarray  # (N,)      elapsed time [s]
    q:                   np.ndarray  # (N, nq)   joint positions
    dq:                  np.ndarray  # (N, nv)   joint velocities
    tau_ff:              np.ndarray  # (N, nv)   feedforward torques
    left_poses:          np.ndarray  # (N, 4, 4) IK-achieved EE poses
    right_poses:         np.ndarray  # (N, 4, 4) IK-achieved EE poses
    left_targets:        np.ndarray  # (N, 4, 4) original Cartesian targets
    right_targets:       np.ndarray  # (N, 4, 4) original Cartesian targets
    left_ik_pos_errors:  np.ndarray  # (N,) [m]
    right_ik_pos_errors: np.ndarray  # (N,) [m]
    left_ik_rot_errors:  np.ndarray  # (N,) [rad]
    right_ik_rot_errors: np.ndarray  # (N,) [rad]

    @property
    def duration(self) -> float:
        return float(self.timestamps[-1])

    @property
    def n_waypoints(self) -> int:
        return len(self.timestamps)


def compute_joint_trajectory(
    ik_solver:  DualArmIK,
    urdf_model: G1URDFModel,
    traj:       CartesianTrajectory,
    q_start:    np.ndarray,
    dt:         float,
    max_dq_per_step:  float = 0.1,
    max_ik_pos_error: float = 0.02,
    max_ik_rot_error: float = 0.2,
    free_joints: Optional[list] = None,
) -> JointTrajectory:
    """
    Run IK offline over all trajectory waypoints to produce a joint-space trajectory.

    Args:
        free_joints: q-vector indices to optimize (None → both arms, waist locked).
                     Locked arms have their EE targets replaced by FK at q_start.

    Warns (but does not abort) if per-step joint discontinuities or IK errors
    exceed the provided thresholds.
    """
    if free_joints is None:
        free_joints = ik_solver.both_arms_q_idx
    free_set = set(free_joints)

    use_fixed_left  = not any(i in free_set for i in ik_solver.left_q_idx)
    use_fixed_right = not any(i in free_set for i in ik_solver.right_q_idx)

    n          = int(np.ceil(traj.duration / dt)) + 1
    timestamps = np.minimum(np.arange(n) * dt, traj.duration)

    q_list, dq_list, tau_list   = [], [], []
    left_poses,  right_poses    = [], []
    left_targets, right_targets = [], []
    l_pos_err, r_pos_err        = [], []
    l_rot_err, r_rot_err        = [], []

    q_prev  = q_start.copy()
    dq_prev = np.zeros(len(q_start))

    for t in timestamps:
        wp = traj.sample(t)

        # Locked arms: IK cost is zeroed so the target value doesn't matter
        q, dq, tau = ik_solver.solve(
            wp.left_pose, wp.right_pose,
            q_current=q_prev, dq_current=dq_prev, dt=dt,
            free_joints=free_joints,
        )

        step_dq = float(np.max(np.abs(q - q_prev)))
        if step_dq > max_dq_per_step:
            print(f"  [t={t:.3f}s] Warning: joint step {np.degrees(step_dq):.1f}deg "
                  f"> {np.degrees(max_dq_per_step):.1f}deg limit")

        l_pose, r_pose = urdf_model.get_frame_transform(
            q, ["left_ee", "right_ee"], use_reduced=True
        )

        # Locked arms are fixed in joint space, not Cartesian space
        lpe = 0.0 if use_fixed_left  else float(np.linalg.norm(l_pose[:3, 3] - wp.left_pose[:3, 3]))
        rpe = 0.0 if use_fixed_right else float(np.linalg.norm(r_pose[:3, 3] - wp.right_pose[:3, 3]))
        lre = 0.0 if use_fixed_left  else _rotation_angle(l_pose[:3, :3] @ wp.left_pose[:3, :3].T)
        rre = 0.0 if use_fixed_right else _rotation_angle(r_pose[:3, :3] @ wp.right_pose[:3, :3].T)

        if max(lpe, rpe) > max_ik_pos_error:
            print(f"  [t={t:.3f}s] Warning: IK pos error {max(lpe, rpe)*1000:.1f}mm "
                  f"> {max_ik_pos_error*1000:.1f}mm limit")
        if max(lre, rre) > max_ik_rot_error:
            print(f"  [t={t:.3f}s] Warning: IK rot error {np.degrees(max(lre, rre)):.1f}deg "
                  f"> {np.degrees(max_ik_rot_error):.1f}deg limit")

        q_list.append(q);   dq_list.append(dq);   tau_list.append(tau)
        left_poses.append(l_pose);   right_poses.append(r_pose)
        left_targets.append(l_pose  if use_fixed_left  else wp.left_pose)
        right_targets.append(r_pose if use_fixed_right else wp.right_pose)
        l_pos_err.append(lpe); r_pos_err.append(rpe)
        l_rot_err.append(lre); r_rot_err.append(rre)
        q_prev, dq_prev = q, dq

    return JointTrajectory(
        timestamps=timestamps,
        q=np.array(q_list),    dq=np.array(dq_list),   tau_ff=np.array(tau_list),
        left_poses=np.array(left_poses),   right_poses=np.array(right_poses),
        left_targets=np.array(left_targets), right_targets=np.array(right_targets),
        left_ik_pos_errors=np.array(l_pos_err),  right_ik_pos_errors=np.array(r_pos_err),
        left_ik_rot_errors=np.array(l_rot_err),  right_ik_rot_errors=np.array(r_rot_err),
    )


class ControllerState(Enum):
    IDLE      = auto()
    EXECUTING = auto()
    ERROR     = auto()


@dataclass
class ExecutionStats:
    """Statistics from a single trajectory execution."""
    duration: float

    # Per-step per-arm position errors [m]
    left_ik_pos_errors:     list
    right_ik_pos_errors:    list
    left_track_pos_errors:  list
    right_track_pos_errors: list

    # Per-step per-arm rotation errors [rad]
    left_ik_rot_errors:     list
    right_ik_rot_errors:    list
    left_track_rot_errors:  list
    right_track_rot_errors: list

    # Per-step joint tracking RMSE [rad]
    track_q_errors: list

    # Summary scalars — max(left, right) per step for pos/rot, scalar for joints
    mean_ik_pos_error:    float
    max_ik_pos_error:     float
    mean_ik_rot_error:    float
    max_ik_rot_error:     float
    mean_track_pos_error: float
    max_track_pos_error:  float
    mean_track_rot_error: float
    max_track_rot_error:  float
    mean_track_q_error:   float
    max_track_q_error:    float

    # Loop timing
    loop_times:     list   # inter-step durations [s]
    mean_loop_time: float
    max_loop_time:  float

    # Raw time series
    time_meas:       list
    left_pose_meas:  list
    right_pose_meas: list
    left_pose_ik:    list
    right_pose_ik:   list
    q_meas:          list
    q_ik:            list


class TrajectoryController:
    """
    Executes Cartesian trajectories by calling IK and commanding hardware.

    Responsibilities:
    - Precompute joint space trajectory
    - Run control loop at fixed frequency
    - Send joint commands to DDS interface
    - Track per-arm position and rotation errors
    - Abort on threshold violations
    """

    def __init__(self,
                 dds: DDSInterface,
                 urdf_model: Optional[G1URDFModel] = None,
                 ik_solver: Optional[DualArmIK] = None,
                 control_rate_hz: float = 100.0,
                 check_collisions: bool = False):
        self.dds          = dds
        self.control_rate = control_rate_hz
        self.dt           = 1.0 / control_rate_hz

        if urdf_model is None:
            print("Controller: Loading URDF model...")
            urdf_model = G1URDFModel(reduced=True)
        self.urdf_model = urdf_model

        if ik_solver is None:
            print("Controller: Initializing IK solver...")
            ik_solver = DualArmIK()
        self.ik_solver = ik_solver

        self.state               = ControllerState.IDLE
        self.current_joint_traj  = None
        self.start_time          = None
        self.check_collisions    = check_collisions

        # Per-arm error lists (reset on each execute() call)
        self.left_track_pos_errors:  list = []
        self.right_track_pos_errors: list = []
        self.left_track_rot_errors:  list = []
        self.right_track_rot_errors: list = []
        self.track_q_errors:         list = []

        # Time series
        self.time_meas:       list = []
        self.q_meas:          list = []
        self.left_pose_meas:  list = []
        self.right_pose_meas: list = []

        # IK safety thresholds (passed to compute_joint_trajectory)
        self.max_dq_per_step  = 0.1   # joint continuity [rad] (~5.8 deg/step)
        self.max_ik_pos_error = 0.02  # position [m]
        self.max_ik_rot_error = 0.2   # rotation [rad] (~11.6 deg)

        # Tracking safety thresholds
        self.max_track_q_error   = 0.2   # rotation [rad] (~11.6 deg)
        self.max_track_pos_error = 0.05  # position [m]
        self.max_track_rot_error = 0.3   # rotation [rad] (~17.4 deg)

        print(f"Controller: Ready at {control_rate_hz} Hz")

    # ===== Offline Precomputation =====

    def _precompute(self, traj: CartesianTrajectory, free_joints: Optional[list] = None) -> JointTrajectory:
        """Run IK offline from current hardware state."""
        q_start, _ = self.dds.get_upper_body_state()

        print(f"Controller: Precomputing joint trajectory ({traj.n_waypoints} waypoints)...")
        jt = compute_joint_trajectory(self.ik_solver, self.urdf_model, traj, q_start, self.dt,
                                      self.max_dq_per_step, self.max_ik_pos_error, self.max_ik_rot_error,
                                      free_joints=free_joints)

        l_mean = np.mean(jt.left_ik_pos_errors)  * 1000
        r_mean = np.mean(jt.right_ik_pos_errors) * 1000
        print(f"Controller: Precompute done — IK pos err L={l_mean:.2f}mm  R={r_mean:.2f}mm")

        return jt

    # ===== Control Methods =====

    def _step(self):
        if self.state != ControllerState.EXECUTING:
            return

        elapsed  = time.time() - self.start_time
        jt       = self.current_joint_traj

        if elapsed >= jt.duration:
            print("Controller: Trajectory complete")
            self.state = ControllerState.IDLE

        idx       = min(int(elapsed / self.dt), jt.n_waypoints - 1)
        q_target  = jt.q[idx]
        dq_target = jt.dq[idx]
        tau_ff    = jt.tau_ff[idx]
        T_des_l   = jt.left_poses[idx]
        T_des_r   = jt.right_poses[idx]

        # Get current state
        q_meas, _ = self.dds.get_upper_body_state()
        left_pose_meas, right_pose_meas = self.urdf_model.get_frame_transform(
            q_meas, ["left_ee", "right_ee"], use_reduced=True
        )

        # TODO: how long does this call take? can we fit in control loop?
        if self.check_collisions:
            if self.urdf_model.check_self_collision(q_target):
                print("Controller: Configuration in collision, aborting!")
                self.state = ControllerState.ERROR
                self.stop()
                return

        # Tracking errors
        q_track     = float(np.sqrt(np.mean((q_target - q_meas) ** 2)))
        l_pos_track = float(np.linalg.norm(left_pose_meas[:3, 3]  - T_des_l[:3, 3]))
        r_pos_track = float(np.linalg.norm(right_pose_meas[:3, 3] - T_des_r[:3, 3]))
        l_rot_track = _rotation_angle(left_pose_meas[:3, :3]  @ T_des_l[:3, :3].T)
        r_rot_track = _rotation_angle(right_pose_meas[:3, :3] @ T_des_r[:3, :3].T)

        # Safeguard against distal commands and poor tracking
        if q_track > self.max_track_q_error:
            print(f"Controller: Tracking joint error {np.degrees(q_track):.1f}deg exceeds limit, aborting!")
            self.state = ControllerState.ERROR
            self.stop()
            return
        if max(l_pos_track, r_pos_track) > self.max_track_pos_error:
            print(f"Controller: Tracking position error {max(l_pos_track, r_pos_track)*1000:.1f}mm exceeds limit, aborting!")
            self.state = ControllerState.ERROR
            self.stop()
            return
        if max(l_rot_track, r_rot_track) > self.max_track_rot_error:
            print(f"Controller: Tracking rotation error {np.degrees(max(l_rot_track, r_rot_track)):.1f}deg exceeds limit, aborting!")
            self.state = ControllerState.ERROR
            self.stop()
            return

        # Record statistics
        self.left_track_pos_errors.append(l_pos_track)
        self.right_track_pos_errors.append(r_pos_track)
        self.left_track_rot_errors.append(l_rot_track)
        self.right_track_rot_errors.append(r_rot_track)
        self.track_q_errors.append(q_track)
        self.time_meas.append(elapsed)
        self.q_meas.append(q_meas)
        self.left_pose_meas.append(left_pose_meas)
        self.right_pose_meas.append(right_pose_meas)

        self.dds.set_upper_body_target(q_target, dq_target, tau_ff)

    # ===== Execution Interface =====

    def execute_joint_interpolation(self, q_target: np.ndarray, duration: float) -> bool:
        """Interpolate in joint space from current position to q_target over duration."""
        if self.state == ControllerState.EXECUTING:
            print("Controller: Already executing!")
            return False

        # Starting conditions
        q_start, _ = self.dds.get_upper_body_state()
        if q_start is None:
            print("Controller: No state available!")
            return False

        # Constant joint velocity
        dq_cmd = (q_target - q_start) / duration

        print(f"Controller: Joint interpolation over {duration:.2f}s")
        self.state  = ControllerState.EXECUTING
        start_time  = time.time()

        # Control loop
        while self.state == ControllerState.EXECUTING:
            loop_start = time.time()
            elapsed    = time.time() - start_time
            if elapsed >= duration:
                print("Controller: Joint interpolation complete")
                dq_cmd = np.zeros_like(q_target)
                self.state = ControllerState.IDLE

            # Interpolate joint command
            alpha  = np.clip(elapsed / duration, 0.0, 1.0)
            q_cmd  = (1 - alpha) * q_start + alpha * q_target

            self.urdf_model.update_forward_kinematics(q_cmd, use_reduced=True)

            # Check collisions
            if self.check_collisions and self.urdf_model.check_self_collision(q_cmd):
                print("Controller: Configuration in collision, aborting!")
                self.state = ControllerState.ERROR
                self.stop()
                break

            # Feedforward torque
            tau_ff = pin.computeGeneralizedGravity(
                self.urdf_model.reduced_robot.model,
                self.urdf_model.reduced_robot.data,
                q_cmd,
            )

            # Send command
            self.dds.set_upper_body_target(q_cmd, dq_cmd, tau_ff)

            # Wait control dt
            loop_elapsed = time.time() - loop_start
            if loop_elapsed < self.dt:
            #[FLAG]: Extend sleep time
                time.sleep(self.dt - loop_elapsed)
                #time.sleep(0.1)

        return self.state == ControllerState.IDLE

    def execute(self, trajectory: CartesianTrajectory, free_joints: Optional[list] = None) -> bool:
        """
        Execute a Cartesian trajectory, first converting to joint space.

        Args:
            trajectory:  Cartesian trajectory to execute.
            free_joints: q-vector indices to optimize during IK (None → both arms, waist locked).

        Returns True iff completed successfully.
        """
        if self.state == ControllerState.EXECUTING:
            print("Controller: Already executing a trajectory!")
            return False

        if trajectory.n_waypoints == 0:
            print("Controller: Empty trajectory!")
            return False
        
        if free_joints is None:
            free_joints = self.ik_solver.both_arms_q_idx
        free_set = set(free_joints)
        use_fixed_left  = not any(i in free_set for i in self.ik_solver.left_q_idx)
        use_fixed_right = not any(i in free_set for i in self.ik_solver.right_q_idx)

        # Check that current EE poses match the trajectory's start poses (for free arms only)
        q_now, _ = self.dds.get_upper_body_state()
        left_now, right_now = self.urdf_model.get_frame_transform(
            q_now, ["left_ee", "right_ee"], use_reduced=True
        )

        wp0 = trajectory.sample(0.0)
        lpe = 0.0 if use_fixed_left  else float(np.linalg.norm(left_now[:3, 3]  - wp0.left_pose[:3, 3]))
        rpe = 0.0 if use_fixed_right else float(np.linalg.norm(right_now[:3, 3] - wp0.right_pose[:3, 3]))
        lre = 0.0 if use_fixed_left  else _rotation_angle(left_now[:3, :3]  @ wp0.left_pose[:3, :3].T)
        rre = 0.0 if use_fixed_right else _rotation_angle(right_now[:3, :3] @ wp0.right_pose[:3, :3].T)

        if max(lpe, rpe) > self.max_track_pos_error:
            print(f"Controller: Start position mismatch — "
                  f"L={lpe*1000:.1f}mm R={rpe*1000:.1f}mm "
                  f"(limit {self.max_track_pos_error*1000:.1f}mm). Aborting.")
            return False
        if max(lre, rre) > self.max_track_rot_error:
            print(f"Controller: Start rotation mismatch — "
                  f"L={np.degrees(lre):.1f}deg R={np.degrees(rre):.1f}deg "
                  f"(limit {np.degrees(self.max_track_rot_error):.1f}deg). Aborting.")
            return False

        # Offline IK
        jt = self._precompute(trajectory, free_joints=free_joints)

        # Reset globals
        self.current_joint_traj = jt
        self.start_time         = time.time()
        self.state              = ControllerState.EXECUTING
        self.left_track_pos_errors  = []
        self.right_track_pos_errors = []
        self.left_track_rot_errors  = []
        self.right_track_rot_errors = []
        self.track_q_errors  = []
        self.time_meas       = []
        self.q_meas          = []
        self.left_pose_meas  = []
        self.right_pose_meas = []

        print(f"Controller: Executing trajectory "
              f"({trajectory.n_waypoints} waypoints, {trajectory.duration:.2f}s)")

        while self.state == ControllerState.EXECUTING:
            loop_start = time.time()
            self._step()
            elapsed = time.time() - loop_start
            if elapsed < self.dt:
                time.sleep(self.dt - elapsed)
            else:
                print(f"Warning: Control loop overran by {(elapsed - self.dt)*1000:.1f}ms")

        if self.state == ControllerState.IDLE:
            self._print_stats()
            return True
        else:
            print("Controller: Execution failed!")
            return False

    def stop(self):
        """Hold current position. Does not modify controller state — caller is responsible."""
        print("Controller: Stopping...")
        q_current, _ = self.dds.get_upper_body_state()

        self.urdf_model.update_forward_kinematics(q_current, use_reduced=True)
        tau_ff = pin.computeGeneralizedGravity(
            self.urdf_model.reduced_robot.model,
            self.urdf_model.reduced_robot.data,
            q_current,
        )

        self.dds.set_upper_body_target(q_current, np.zeros_like(q_current), tau_ff)

    # ===== Statistics =====

    def get_stats(self) -> Optional[ExecutionStats]:
        """Return execution statistics for the last completed trajectory."""
        if not self.left_track_pos_errors:
            return None

        jt       = self.current_joint_traj
        duration = jt.duration if jt is not None else 0.0

        # Map recorded wall-time steps back to precomputed trajectory indices
        n       = jt.n_waypoints
        indices = [min(int(t / self.dt), n - 1) for t in self.time_meas]

        left_ik_pos_errors  = [jt.left_ik_pos_errors[i]  for i in indices]
        right_ik_pos_errors = [jt.right_ik_pos_errors[i] for i in indices]
        left_ik_rot_errors  = [jt.left_ik_rot_errors[i]  for i in indices]
        right_ik_rot_errors = [jt.right_ik_rot_errors[i] for i in indices]
        q_ik         = [jt.q[i]           for i in indices]
        left_pose_ik = [jt.left_poses[i]  for i in indices]
        right_pose_ik= [jt.right_poses[i] for i in indices]

        ik_pos_max    = np.maximum(left_ik_pos_errors,             right_ik_pos_errors)
        ik_rot_max    = np.maximum(left_ik_rot_errors,             right_ik_rot_errors)
        track_pos_max = np.maximum(self.left_track_pos_errors,     self.right_track_pos_errors)
        track_rot_max = np.maximum(self.left_track_rot_errors,     self.right_track_rot_errors)
        track_q       = self.track_q_errors.copy()

        loop_times = (
            np.diff(self.time_meas).tolist()
            if len(self.time_meas) > 1
            else [self.dt]
        )

        return ExecutionStats(
            duration               = duration,
            left_ik_pos_errors     = left_ik_pos_errors,
            right_ik_pos_errors    = right_ik_pos_errors,
            left_ik_rot_errors     = left_ik_rot_errors,
            right_ik_rot_errors    = right_ik_rot_errors,
            left_track_pos_errors  = self.left_track_pos_errors.copy(),
            right_track_pos_errors = self.right_track_pos_errors.copy(),
            left_track_rot_errors  = self.left_track_rot_errors.copy(),
            right_track_rot_errors = self.right_track_rot_errors.copy(),
            track_q_errors         = track_q,
            mean_ik_pos_error      = float(np.mean(ik_pos_max)),
            max_ik_pos_error       = float(np.max(ik_pos_max)),
            mean_ik_rot_error      = float(np.mean(ik_rot_max)),
            max_ik_rot_error       = float(np.max(ik_rot_max)),
            mean_track_pos_error   = float(np.mean(track_pos_max)),
            max_track_pos_error    = float(np.max(track_pos_max)),
            mean_track_rot_error   = float(np.mean(track_rot_max)),
            max_track_rot_error    = float(np.max(track_rot_max)),
            mean_track_q_error     = float(np.mean(track_q)),
            max_track_q_error      = float(np.max(track_q)),
            loop_times             = loop_times,
            mean_loop_time         = float(np.mean(loop_times)),
            max_loop_time          = float(np.max(loop_times)),
            time_meas              = self.time_meas.copy(),
            left_pose_meas         = self.left_pose_meas.copy(),
            right_pose_meas        = self.right_pose_meas.copy(),
            left_pose_ik           = left_pose_ik,
            right_pose_ik          = right_pose_ik,
            q_meas                 = self.q_meas.copy(),
            q_ik                   = q_ik,
        )

    def _print_stats(self):
        s = self.get_stats()
        if s is None:
            return

        def mm(a):  return f"mean {np.mean(a)*1000:.2f}mm  max {np.max(a)*1000:.2f}mm"
        def deg(a): return f"mean {np.degrees(np.mean(a)):.2f}deg  max {np.degrees(np.max(a)):.2f}deg"

        print("\nController Statistics:")
        print(f"  Duration         : {s.duration:.2f}s")
        print(f"  Left  IK  pos    : {mm(s.left_ik_pos_errors)}")
        print(f"  Right IK  pos    : {mm(s.right_ik_pos_errors)}")
        print(f"  Left  IK  rot    : {deg(s.left_ik_rot_errors)}")
        print(f"  Right IK  rot    : {deg(s.right_ik_rot_errors)}")
        print(f"  Left  track pos  : {mm(s.left_track_pos_errors)}")
        print(f"  Right track pos  : {mm(s.right_track_pos_errors)}")
        print(f"  Left  track rot  : {deg(s.left_track_rot_errors)}")
        print(f"  Right track rot  : {deg(s.right_track_rot_errors)}")
        print(f"  IK  pos   (worst): mean {s.mean_ik_pos_error*1000:.2f}mm  max {s.max_ik_pos_error*1000:.2f}mm")
        print(f"  IK  rot   (worst): mean {np.degrees(s.mean_ik_rot_error):.2f}deg  max {np.degrees(s.max_ik_rot_error):.2f}deg")
        print(f"  Track pos (worst): mean {s.mean_track_pos_error*1000:.2f}mm  max {s.max_track_pos_error*1000:.2f}mm")
        print(f"  Track rot (worst): mean {np.degrees(s.mean_track_rot_error):.2f}deg  max {np.degrees(s.max_track_rot_error):.2f}deg")
        print(f"  Track q   (RMSE) : mean {np.degrees(s.mean_track_q_error):.2f}deg  max {np.degrees(s.max_track_q_error):.2f}deg")
        print(f"  Loop budget      : {self.dt*1000:.1f}ms")
        print(f"  Loop time        : mean {s.mean_loop_time*1000:.2f}ms  max {s.max_loop_time*1000:.2f}ms")


# ===== Visualization =====

def joint_tracking_figure(t, q_ik, q_meas):
    fig, axes = plt.subplots(2, 7, figsize=(22, 8), sharex=True)
    fig.suptitle('Joint Angles: IK Commanded vs Measured', fontsize=14)
    for j in range(7):
        for row, offset, color, label in [(0, 1, 'b', 'Left'), (1, 8, 'r', 'Right')]:
            ax = axes[row, j]
            q_des_j  = np.degrees(q_ik[:, offset + j])
            q_meas_j = np.degrees(q_meas[:, offset + j])
            ax.plot(t, q_des_j,  f'{color}--', linewidth=1.5, label='IK cmd')
            ax.plot(t, q_meas_j, f'{color}-',  linewidth=1.0, label='Measured')
            ax.set_title(f'{label} J{j+1}', fontsize=9)
            ax.set_ylabel('Angle (°)', fontsize=8)
            ax.grid(True, alpha=0.4)
            if j == 0:
                ax.legend(fontsize=7)

    plt.tight_layout()
    return fig


def position_tracking_figure(t, pose_des, pose_ik, pose_meas, title_prefix, color):
    fig, axes = plt.subplots(1, 3, figsize=(15, 8), sharex=True)
    fig.suptitle(f'{title_prefix} Position Tracking: Desired vs IK vs Measured', fontsize=14)

    for i, label in enumerate(['X', 'Y', 'Z']):
        ax = axes[i]
        ax.plot(t, pose_des[:, i, 3]  * 1000, color=color, linestyle='--', label='Desired',  linewidth=1.5)
        ax.plot(t, pose_ik[:, i, 3]   * 1000, color=color, linestyle='-.', label='IK',       linewidth=1.0)
        ax.plot(t, pose_meas[:, i, 3] * 1000, color=color, linestyle='-',  label='Measured', linewidth=1.0)
        ax.fill_between(t, pose_des[:, i, 3] * 1000, pose_meas[:, i, 3] * 1000, 
                        alpha=0.15, color=color, label='Des-Meas error')
        rms_ik   = np.sqrt(np.mean((pose_ik[:, i, 3]   - pose_des[:, i, 3])**2)) * 1000
        rms_meas = np.sqrt(np.mean((pose_meas[:, i, 3] - pose_des[:, i, 3])**2)) * 1000
        ax.set_title(f'{label}\nIK RMS: {rms_ik:.2f}mm  |  Meas RMS: {rms_meas:.2f}mm', fontsize=9)
        ax.set_ylabel('Position (mm)')
        ax.set_xlabel('Time (s)')
        ax.legend(fontsize=8)
        ax.grid(True)

    plt.tight_layout()
    return fig


def rotation_tracking_figure(t, pose_des, pose_ik, pose_meas, title_prefix, color):
    from scipy.spatial.transform import Rotation

    def to_euler(poses):
        Rs = poses[:, :3, :3]
        return Rotation.from_matrix(Rs).as_euler('xyz', degrees=True)

    euler_des  = to_euler(pose_des)
    euler_ik   = to_euler(pose_ik)
    euler_meas = to_euler(pose_meas)

    angle_labels = ['Roll (X)', 'Pitch (Y)', 'Yaw (Z)']

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharex=True)
    fig.suptitle(f'{title_prefix} Rotation Tracking: Desired vs IK vs Measured', fontsize=14)

    for i, label in enumerate(angle_labels):
        ax = axes[i]
        ax.plot(t, euler_des[:, i],  color=color, linestyle='--', linewidth=1.5, label='Desired')
        ax.plot(t, euler_ik[:, i],   color=color, linestyle='-.',  linewidth=1.0, label='IK')
        ax.plot(t, euler_meas[:, i], color=color, linestyle='-',   linewidth=1.0, label='Measured')
        ax.fill_between(t, euler_des[:, i], euler_meas[:, i],
                        alpha=0.15, color=color, label='Des→Meas error')
        rms_ik   = np.sqrt(np.mean((euler_ik[:, i]   - euler_des[:, i])**2))
        rms_meas = np.sqrt(np.mean((euler_meas[:, i] - euler_des[:, i])**2))
        ax.set_title(f'{label}\nIK RMS: {rms_ik:.2f}°  |  Meas RMS: {rms_meas:.2f}°', fontsize=9)
        ax.set_ylabel('Angle (°)')
        ax.set_xlabel('Time (s)')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.4)

    plt.tight_layout()
    return fig


# ===== Test =====

if __name__ == '__main__':
    import sys
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    from bricklaying.planning import MotionPlanner, R_LEFT_NOMINAL_PICK, R_RIGHT_NOMINAL_PICK

    print("=" * 60)
    print("Testing Trajectory Controller")
    print("=" * 60)

    # ---------------------------------------------
    # Initialize system
    # ---------------------------------------------
    print("\n[1] Initializing...")

    NETWORK_INTERFACE = sys.argv[1] if len(sys.argv) > 1 else "eth0"
    input("Press enter to instantiate dds interface...")
    print(f"Testing DDSInterface on network interface: {NETWORK_INTERFACE}")
    try:
        dds = DDSInterface(NETWORK_INTERFACE)
    except TimeoutError as e:
        print(f"Connection failed: {e}")
        sys.exit(1)
    input("Press enter to instantiate urdf...")
    urdf_model = G1URDFModel(reduced=True)
    input("Press enter to instantiate ik_solver...")
    ik_solver = DualArmIK()
    input("Press enter to instantiate planner...")
    planner = MotionPlanner()
    input("Press enter to instantiate controller...")
    controller = TrajectoryController(dds, urdf_model, ik_solver, control_rate_hz=100.0)

    # ---------------------------------------------
    # Close hands
    # ---------------------------------------------
    
    input("Press enter to open hands...")
    print("\n[2] Opening hands...")
    dds.set_hand_mode("open", "open")
    time.sleep(2.0)

    
    input("Press enter to close hands...")
    print("\n[2] Closing hands...")
    dds.set_hand_mode("close", "close")
    time.sleep(2.0)
    
    
    # ---------------------------------------------
    # Plan a trajectory
    # ---------------------------------------------
    print("\n[3] Planning trajectory...")

    q_start, _ = dds.get_upper_body_state()
    left_start  = urdf_model.get_frame_transform(q_start, "left_ee",  use_reduced=True)
    right_start = urdf_model.get_frame_transform(q_start, "right_ee", use_reduced=True)

    def pose(pos, R):
        T = np.eye(4)
        T[:3, 3] = pos
        T[:3, :3] = R
        return T
    
    left_poses = [
        left_start,
        pose([0.10,  0.4,  0.15], np.eye(3)),
        pose([0.35,  0.3,  0.15], R_LEFT_NOMINAL_PICK),
    ]
    right_poses = [
        right_start,
        pose([0.10, -0.4,  0.15], np.eye(3)),
        pose([0.35, -0.3,  0.15], R_RIGHT_NOMINAL_PICK),
    ]

    traj = planner.plan_through_waypoints(left_poses, right_poses, duration=6.0)
    print(f"    ✓ {traj.n_waypoints} waypoints over {traj.duration:.2f}s")

    #[FLAG]: Assume this is where left is disabled?
    free_joints = ik_solver.right_q_idx + ik_solver.waist_q_idx + ik_solver.left_q_idx

    # ---------------------------------------------
    # Execute trajectory
    # ---------------------------------------------
    input("Press enter to execute trajectory...")
    print("\n[4] Executing trajectory...")

    success = controller.execute(traj, free_joints=free_joints)
    stats   = controller.get_stats() #if success else None
    if success:
        print("    ✓ Execution successful!")
    else:
        print("    ✗ Execution failed!")
    time.sleep(1.0)

    # ---------------------------------------------
    # Return trajectory
    # ---------------------------------------------
    print("\n[5] Returning to start position...")

    success = controller.execute(traj.reverse(), free_joints=free_joints)
    if success:
        print("    ✓ Execution successful!")
    else:
        print("    ✗ Execution failed!")
    time.sleep(1.0)

    # ---------------------------------------------
    # Center joints
    # ---------------------------------------------
    print("\n[6] Center joints to homer's position...")

    success = controller.execute_joint_interpolation(q_start, duration=1.0)
    if success:
        print("    ✓ Execution successful!")
    else:
        print("    ✗ Execution failed!")

    # ---------------------------------------------
    # Shutdown
    # ---------------------------------------------
    print("\n[7] Test complete. Shutting down...")
    dds.shutdown()

    # ---------------------------------------------
    # Generate plots
    # ---------------------------------------------
    if True:
        print("\n[8] Generating plots...")
        times = np.array(stats.time_meas)
        times = times - times[0]

        q_ik   = np.array(stats.q_ik)
        q_meas = np.array(stats.q_meas)

        left_pose_des, right_pose_des = [], []
        for t in times:
            wp = traj.sample(t)
            left_pose_des.append(wp.left_pose)
            right_pose_des.append(wp.right_pose)

        left_pose_des  = np.array(left_pose_des)
        right_pose_des = np.array(right_pose_des)
        left_pose_ik    = np.array(stats.left_pose_ik)
        right_pose_ik   = np.array(stats.right_pose_ik)
        left_pose_meas  = np.array(stats.left_pose_meas)
        right_pose_meas = np.array(stats.right_pose_meas)

        # ── Figure 1: Joint angles ──
        fig1 = joint_tracking_figure(times, q_ik, q_meas)
        fig1.savefig('trajectory_joints.png', dpi=150)
        print("    ✓ trajectory_joints.png")

        # ── Figure 2: Cartesian positions ──
        fig2_left = position_tracking_figure(
            times, left_pose_des, left_pose_ik, left_pose_meas, 'Left EE', 'blue')
        fig2_left.savefig('trajectory_position_left.png', dpi=150)
        print("    ✓ trajectory_position_left.png")

        fig2_right = position_tracking_figure(
            times, right_pose_des, right_pose_ik, right_pose_meas, 'Right EE', 'red')
        fig2_right.savefig('trajectory_position_right.png', dpi=150)
        print("    ✓ trajectory_position_right.png")

        # ── Figure 3: Rotation tracking ──
        fig3_left = rotation_tracking_figure(
            times, left_pose_des, left_pose_ik, left_pose_meas, 'Left EE', 'blue')
        fig3_left.savefig('trajectory_rotation_left.png', dpi=150)
        print("    ✓ trajectory_rotation_left.png")

        fig3_right = rotation_tracking_figure(
            times, right_pose_des, right_pose_ik, right_pose_meas, 'Right EE', 'red')
        fig3_right.savefig('trajectory_rotation_right.png', dpi=150)
        print("    ✓ trajectory_rotation_right.png")
