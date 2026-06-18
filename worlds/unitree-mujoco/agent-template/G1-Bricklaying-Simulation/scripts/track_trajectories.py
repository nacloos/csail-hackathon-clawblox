"""
Evaluates bimanual arm tracking accuracy over a set of trajectory variations.

Each variation applies randomly sampled positional deltas to waypoints 1-3
and a yaw rotation delta to the final end-effector pose.  Reports IK and
tracking error per trajectory and aggregate results, and saves a dataset of
desired/IK/measured trajectories for offline analysis.

Usage:
    python scripts/track_trajectories.py --dry-run --animate
    python scripts/track_trajectories.py --network-interface eth0 --n-variants 20

To-do:
    - Include rotational misalignments
    - Debug/run on hardware
"""
from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import numpy.random as random
import matplotlib.pyplot as plt

from bricklaying.robot import (
    DDSInterface, ExecutionStats,
    compute_joint_trajectory,
    TrajectoryController, G1URDFModel, DualArmIK,
)
from bricklaying.planning import (
    MotionPlanner, CartesianTrajectory,
    R_LEFT_NOMINAL_PICK, R_RIGHT_NOMINAL_PICK,
    R_LEFT_PALM_IN, R_RIGHT_PALM_IN, Q_UPPER_BODY_REST,
)


# ===========================================================================
# RNG — fix seed for reproducibility
# ===========================================================================

SEED = 42
rng = random.default_rng(seed=SEED)

POSITION_SCALE = 0.04       # Uniform half-range for xyz waypoint perturbations [m]
YAW_SCALE      = np.pi / 6  # Uniform half-range for final-pose yaw perturbation [rad]


# ===========================================================================
# Nominal trajectory
# ===========================================================================

def make_nominal_poses() -> tuple[np.ndarray, np.ndarray]:
    """
    Define the nominal (unperturbed) dual-arm Cartesian poses for waypoints 1-3.
    Waypoint 0 (current robot pose) is prepended at execution time.

    Returns:
        left_poses:  (3, 4, 4) array of SE(3) matrices
        right_poses: (3, 4, 4) array of SE(3) matrices
    """
    def pose(pos, R):
        T = np.eye(4)
        T[:3, 3] = pos
        T[:3, :3] = R
        return T

    left_poses = [
        pose([0.10,  0.4,  0.15], R_LEFT_PALM_IN),        # intermediate
        pose([0.25,  0.25, 0.25], R_LEFT_NOMINAL_PICK),   # pre-pick
        pose([0.3,   0.25, 0.15], R_LEFT_NOMINAL_PICK),   # pick
    ]
    right_poses = [
        pose([0.10, -0.4,  0.15], R_RIGHT_PALM_IN),
        pose([0.25, -0.25, 0.25], R_RIGHT_NOMINAL_PICK),
        pose([0.3,  -0.25, 0.15], R_RIGHT_NOMINAL_PICK),
    ]
    return np.array(left_poses), np.array(right_poses)


# ===========================================================================
# Variation sampling
# ===========================================================================

def _Rz(yaw: float) -> np.ndarray:
    """3x3 rotation matrix about Z axis."""
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])


@dataclass
class TrajectoryVariant:
    """A sampled trajectory variant with its perturbation parameters stored for reporting."""
    index: int
    left_pos_deltas: np.ndarray     # (3, 3) — one xyz delta per waypoint 1-3
    right_pos_deltas: np.ndarray    # (3, 3) — one xyz delta per waypoint 1-3
    left_yaw_delta: float           # applied to rotation of final waypoints
    right_yaw_delta: float
    left_poses: np.ndarray          # (3, 4, 4) perturbed poses
    right_poses: np.ndarray         # (3, 4, 4) perturbed poses


def sample_variant(
    index: int,
    nominal_left: np.ndarray,
    nominal_right: np.ndarray,
) -> TrajectoryVariant:
    """Sample one trajectory variant by drawing independent Gaussian perturbations."""
    left_pos_deltas  = rng.uniform(-POSITION_SCALE, POSITION_SCALE, (3, 3))
    right_pos_deltas = rng.uniform(-POSITION_SCALE, POSITION_SCALE, (3, 3))
    left_yaw_delta   = float(rng.uniform(-YAW_SCALE, YAW_SCALE))
    right_yaw_delta  = float(rng.uniform(-YAW_SCALE, YAW_SCALE))

    left_poses  = nominal_left.copy()
    right_poses = nominal_right.copy()

    for i in range(3):
        left_poses[i, :3, 3]  += left_pos_deltas[i]
        right_poses[i, :3, 3] += right_pos_deltas[i]

    RL, RR = _Rz(left_yaw_delta), _Rz(right_yaw_delta)
    left_poses[-2, :3, :3]  = RL @ left_poses[-2, :3, :3]
    left_poses[-1, :3, :3]  = RL @ left_poses[-1, :3, :3]
    right_poses[-2, :3, :3] = RR @ right_poses[-2, :3, :3]
    right_poses[-1, :3, :3] = RR @ right_poses[-1, :3, :3]

    return TrajectoryVariant(
        index=index,
        left_pos_deltas=left_pos_deltas,
        right_pos_deltas=right_pos_deltas,
        left_yaw_delta=left_yaw_delta,
        right_yaw_delta=right_yaw_delta,
        left_poses=left_poses,
        right_poses=right_poses,
    )


# ===========================================================================
# Result dataclass
# ===========================================================================

@dataclass
class VariantResult:
    index: int
    variant: TrajectoryVariant
    trajectory: CartesianTrajectory
    success: bool
    mean_ik_pos_mm: float      = 0.0
    max_ik_pos_mm: float       = 0.0
    mean_ik_rot_deg: float     = 0.0
    max_ik_rot_deg: float      = 0.0
    mean_track_pos_mm: float   = 0.0
    max_track_pos_mm: float    = 0.0
    mean_track_rot_deg: float  = 0.0
    max_track_rot_deg: float   = 0.0
    mean_track_q_deg: float    = 0.0
    max_track_q_deg: float     = 0.0
    mean_loop_ms: float        = 0.0
    max_loop_ms: float         = 0.0
    failure_reason: str        = ""
    stats: Optional[ExecutionStats] = None   # full time-series data (hardware only)


# ===========================================================================
# Reporting
# ===========================================================================

def _sep(char: str = "-", width: int = 80) -> str:
    return char * width


def print_variant_result(r: VariantResult) -> None:
    v = r.variant
    status = "PASS" if r.success else f"FAIL ({r.failure_reason})"

    print(f"\n[{r.index+1:02d}] {status}")
    print(f"     Pos deltas (m)       Left                        Right")
    for i in range(3):
        dl, dr = v.left_pos_deltas[i], v.right_pos_deltas[i]
        print(f"       Pose {i+1}:  [{dl[0]:+.3f}, {dl[1]:+.3f}, {dl[2]:+.3f}]"
              f"    [{dr[0]:+.3f}, {dr[1]:+.3f}, {dr[2]:+.3f}]")
    print(f"     Yaw delta  :  left={np.degrees(v.left_yaw_delta):+.1f}deg   "
          f"right={np.degrees(v.right_yaw_delta):+.1f}deg")

    if r.success:
        print(f"     IK  pos  — mean: {r.mean_ik_pos_mm:6.2f}mm   max: {r.max_ik_pos_mm:6.2f}mm")
        print(f"     IK  rot  — mean: {r.mean_ik_rot_deg:6.2f}deg  max: {r.max_ik_rot_deg:6.2f}deg")
        if r.mean_track_pos_mm > 0:
            print(f"     Trk pos  — mean: {r.mean_track_pos_mm:6.2f}mm   max: {r.max_track_pos_mm:6.2f}mm")
            print(f"     Trk rot  — mean: {r.mean_track_rot_deg:6.2f}deg  max: {r.max_track_rot_deg:6.2f}deg")
            print(f"     Trk jnt  — mean: {r.mean_track_q_deg:6.2f}deg  max: {r.max_track_q_deg:6.2f}deg")
        if r.mean_loop_ms > 0:
            print(f"     Loop     — mean: {r.mean_loop_ms:6.2f}ms   max: {r.max_loop_ms:6.2f}ms")


def print_aggregate_report(results: list[VariantResult]) -> None:
    passed = [r for r in results if r.success]
    failed = [r for r in results if not r.success]

    print("\n" + _sep("="))
    print("AGGREGATE RESULTS")
    print(_sep("="))
    print(f"  Seed           : {SEED}")
    print(f"  Position std   : {POSITION_SCALE*100:.2f}cm")
    print(f"  Yaw scale      : {np.degrees(YAW_SCALE):.1f} deg")
    print(f"  Total variants : {len(results)}")
    print(f"  Passed         : {len(passed)}")
    print(f"  Failed         : {len(failed)}")

    if passed:
        def _agg(attr):
            vals = [getattr(r, attr) for r in passed if getattr(r, attr) > 0]
            return (float(np.mean(vals)), float(np.max(vals))) if vals else (0.0, 0.0)

        mean_ik_pos, _  = _agg('mean_ik_pos_mm')
        mean_ik_rot, _  = _agg('mean_ik_rot_deg')
        mean_trk_pos, _ = _agg('mean_track_pos_mm')
        mean_trk_rot, _ = _agg('mean_track_rot_deg')
        mean_trk_q, _   = _agg('mean_track_q_deg')
        _, max_ik_pos   = _agg('max_ik_pos_mm')
        _, max_ik_rot   = _agg('max_ik_rot_deg')
        _, max_trk_pos  = _agg('max_track_pos_mm')
        _, max_trk_rot  = _agg('max_track_rot_deg')
        _, max_trk_q    = _agg('max_track_q_deg')

        print(f"\n  IK errors (passing variants):")
        print(f"    Position — mean of means: {mean_ik_pos:.2f}mm    max of maxes: {max_ik_pos:.2f}mm")
        print(f"    Rotation — mean of means: {mean_ik_rot:.2f}deg   max of maxes: {max_ik_rot:.2f}deg")

        hw_passed = [r for r in passed if r.mean_track_pos_mm > 0]
        if hw_passed:
            print(f"\n  Tracking errors (passing variants):")
            print(f"    Position — mean of means: {mean_trk_pos:.2f}mm    max of maxes: {max_trk_pos:.2f}mm")
            print(f"    Rotation — mean of means: {mean_trk_rot:.2f}deg   max of maxes: {max_trk_rot:.2f}deg")
            print(f"    Joints   — mean of means: {mean_trk_q:.2f}deg   max of maxes: {max_trk_q:.2f}deg")

            best  = min(hw_passed, key=lambda r: r.mean_track_pos_mm)
            worst = max(hw_passed, key=lambda r: r.mean_track_pos_mm)
            print(f"\n  Best tracking  : variant {best.index+1} (mean pos={best.mean_track_pos_mm:.2f}mm)")
            print(f"  Worst tracking : variant {worst.index+1} (mean pos={worst.mean_track_pos_mm:.2f}mm)")

        loop_results = [r for r in passed if r.mean_loop_ms > 0]
        if loop_results:
            mean_loop = np.mean([r.mean_loop_ms for r in loop_results])
            max_loop  = np.max( [r.max_loop_ms  for r in loop_results])
            print(f"\n  Loop time (passing variants):")
            print(f"    Mean of means : {mean_loop:.2f}ms")
            print(f"    Max of maxes  : {max_loop:.2f}ms")

    if failed:
        print(f"\n  Failed variants:")
        for r in failed:
            print(f"    - [{r.index+1:02d}] {r.failure_reason}")

    print(_sep("=") + "\n")


# ===========================================================================
# Dataset saving
# ===========================================================================

def save_dataset(results: list[VariantResult], output_path: str) -> None:
    """
    Save per-variant tracking data to a compressed numpy archive.

    For each successful hardware variant, saves arrays keyed as v{index:02d}_*:
        time                          (N,)     elapsed time [s]
        left/right_pose_des           (N,4,4)  desired EE pose (planned trajectory)
        left/right_pose_ik            (N,4,4)  IK solution EE pose
        left/right_pose_meas          (N,4,4)  measured EE pose
        q_ik / q_meas                 (N,14)   joint angles [rad]
        left/right_ik_pos_errors      (N,)     IK position error per arm [m]
        left/right_ik_rot_errors      (N,)     IK rotation error per arm [rad]
        left/right_track_pos_errors   (N,)     tracking position error per arm [m]
        left/right_track_rot_errors   (N,)     tracking rotation error per arm [rad]
        track_q_errors                (N,)     joint tracking RMSE across all 14 joints [rad]

    Plus top-level metadata: seed, position_scale, yaw_scale, n_variants,
    successful_variants, and per-variant perturbation parameters.
    """
    hw = [r for r in results if r.success and r.stats is not None]
    if not hw:
        print("No hardware results to save.")
        return

    data: dict = {
        'seed':                SEED,
        'position_scale':      POSITION_SCALE,
        'yaw_scale':           YAW_SCALE,
        'n_variants':          len(results),
        'successful_variants': np.array([r.index for r in hw]),
    }

    for r in hw:
        s   = r.stats
        t   = np.array(s.time_meas)
        pfx = f"v{r.index:02d}"

        # Sample desired trajectory at controller's measured timestamps
        des = [r.trajectory.sample(ts) for ts in t]
        left_des  = np.array([wp.left_pose  for wp in des])
        right_des = np.array([wp.right_pose for wp in des])

        data.update({
            f"{pfx}_time":                    t,
            f"{pfx}_left_pose_des":           left_des,
            f"{pfx}_right_pose_des":          right_des,
            f"{pfx}_left_pose_ik":            np.array(s.left_pose_ik),
            f"{pfx}_right_pose_ik":           np.array(s.right_pose_ik),
            f"{pfx}_left_pose_meas":          np.array(s.left_pose_meas),
            f"{pfx}_right_pose_meas":         np.array(s.right_pose_meas),
            f"{pfx}_q_ik":                    np.array(s.q_ik),
            f"{pfx}_q_meas":                  np.array(s.q_meas),
            f"{pfx}_left_ik_pos_errors":      np.array(s.left_ik_pos_errors),
            f"{pfx}_right_ik_pos_errors":     np.array(s.right_ik_pos_errors),
            f"{pfx}_left_ik_rot_errors":      np.array(s.left_ik_rot_errors),
            f"{pfx}_right_ik_rot_errors":     np.array(s.right_ik_rot_errors),
            f"{pfx}_left_track_pos_errors":   np.array(s.left_track_pos_errors),
            f"{pfx}_right_track_pos_errors":  np.array(s.right_track_pos_errors),
            f"{pfx}_left_track_rot_errors":   np.array(s.left_track_rot_errors),
            f"{pfx}_right_track_rot_errors":  np.array(s.right_track_rot_errors),
            f"{pfx}_track_q_errors":          np.array(s.track_q_errors),
            f"{pfx}_left_pos_deltas":         r.variant.left_pos_deltas,
            f"{pfx}_right_pos_deltas":        r.variant.right_pos_deltas,
            f"{pfx}_left_yaw_delta":          r.variant.left_yaw_delta,
            f"{pfx}_right_yaw_delta":         r.variant.right_yaw_delta,
        })

    np.savez_compressed(output_path, **data)
    print(f"Dataset saved → {output_path}  ({len(hw)}/{len(results)} variants)")


# ===========================================================================
# Dry-run (IK only, no hardware)
# ===========================================================================

def run_dry(variants: list[TrajectoryVariant], animate: bool = False) -> list[VariantResult]:
    print("\n[DRY RUN] Loading URDF and IK solver...")
    urdf_model = G1URDFModel(reduced=True)
    ik_solver  = DualArmIK()
    planner    = MotionPlanner()

    if animate:
        from pinocchio.visualize import MeshcatVisualizer
        viz = MeshcatVisualizer(
            urdf_model.reduced_robot.model,
            urdf_model.reduced_robot.collision_model,
            urdf_model.reduced_robot.visual_model,
        )
        try:
            viz.initViewer(open=True)
            viz.loadViewerModel()
            print("    MeshCat viewer: http://127.0.0.1:7000/static/")
        except ImportError:
            print("    meshcat not installed — skipping animation.")
            animate = False

    q_current = np.asarray(Q_UPPER_BODY_REST)
    duration  = 6.0
    dt        = 1.0 / 100.0

    left_pose_init, right_pose_init = urdf_model.get_frame_transform(
        q_current, ["left_ee", "right_ee"], use_reduced=True
    )

    results = []

    for variant in variants:
        print(f"\n[{variant.index+1:02d}/{len(variants)}] Evaluating variant...")

        left_waypoints  = [left_pose_init]  + list(variant.left_poses)
        right_waypoints = [right_pose_init] + list(variant.right_poses)

        traj = planner.plan_through_waypoints(left_waypoints, right_waypoints, duration)
        jt   = compute_joint_trajectory(ik_solver, urdf_model, traj, q_current, dt)

        pos_ik_max = np.maximum(jt.left_ik_pos_errors, jt.right_ik_pos_errors)
        rot_ik_max = np.maximum(jt.left_ik_rot_errors, jt.right_ik_rot_errors)
        results.append(VariantResult(
            index=variant.index,
            variant=variant,
            trajectory=traj,
            success=True,
            mean_ik_pos_mm=float(np.mean(pos_ik_max))  * 1000,
            max_ik_pos_mm=float(np.max(pos_ik_max))    * 1000,
            mean_ik_rot_deg=float(np.mean(rot_ik_max)) * 180 / np.pi,
            max_ik_rot_deg=float(np.max(rot_ik_max))   * 180 / np.pi,
        ))
        print_variant_result(results[-1])

        if animate:
            for q in jt.q:
                start = time.time()
                viz.display(q)
                delay = traj.duration / len(jt.q)
                wait = max(0, delay - (time.time() - start))
                time.sleep(wait)

    return results


# ===========================================================================
# Hardware execution
# ===========================================================================

def run_hardware(
    variants: list[TrajectoryVariant],
    network_interface: str,
) -> list[VariantResult]:

    print(f"\nConnecting to robot on '{network_interface}'...")
    dds        = DDSInterface(network_interface)
    urdf_model = G1URDFModel(reduced=True)
    ik_solver  = DualArmIK()
    controller = TrajectoryController(
        dds=dds,
        urdf_model=urdf_model,
        ik_solver=ik_solver,
        control_rate_hz=100.0,
    )
    planner  = MotionPlanner()
    duration = 6.0
    results  = []

    q_start, _ = dds.get_upper_body_state()

    for variant in variants:
        print(f"\n[{variant.index+1:02d}/{len(variants)}] Evaluating variant...")

        q_home, _ = dds.get_upper_body_state()
        left_pose_init, right_pose_init = urdf_model.get_frame_transform(
            q_home, ["left_ee", "right_ee"], use_reduced=True
        )

        left_waypoints  = [left_pose_init]  + list(variant.left_poses)
        right_waypoints = [right_pose_init] + list(variant.right_poses)

        traj    = planner.plan_through_waypoints(left_waypoints, right_waypoints, duration)
        success = controller.execute(traj)
        stats   = controller.get_stats()

        if success and stats is not None:
            results.append(VariantResult(
                index=variant.index,
                variant=variant,
                trajectory=traj,
                success=True,
                mean_ik_pos_mm=stats.mean_ik_pos_error          * 1000,
                max_ik_pos_mm=stats.max_ik_pos_error            * 1000,
                mean_ik_rot_deg=stats.mean_ik_rot_error         * 180 / np.pi,
                max_ik_rot_deg=stats.max_ik_rot_error           * 180 / np.pi,
                mean_track_pos_mm=stats.mean_track_pos_error    * 1000,
                max_track_pos_mm=stats.max_track_pos_error      * 1000,
                mean_track_rot_deg=stats.mean_track_rot_error   * 180 / np.pi,
                max_track_rot_deg=stats.max_track_rot_error     * 180 / np.pi,
                mean_track_q_deg=stats.mean_track_q_error       * 180 / np.pi,
                max_track_q_deg=stats.max_track_q_error         * 180 / np.pi,
                mean_loop_ms=stats.mean_loop_time               * 1000,
                max_loop_ms=stats.max_loop_time                 * 1000,
                stats=stats,
            ))
        else:
            results.append(VariantResult(
                index=variant.index,
                variant=variant,
                trajectory=traj,
                success=False,
                failure_reason="controller execution failed",
            ))

        print_variant_result(results[-1])

        print("\n  Returning to home pose...")
        controller.execute(traj.reverse())
        controller.execute_joint_interpolation(q_start, duration=1.0)
        time.sleep(1.0)

    dds.shutdown()
    return results


# ===========================================================================
# Visualization
# ===========================================================================

def plot_trajectory_variants(results: list[VariantResult]) -> None:
    """3D plot of all planned trajectory variants."""
    fig = plt.figure(figsize=(10, 8), facecolor='white')
    ax  = fig.add_subplot(111, projection='3d', facecolor='white')
    ax.set_title("Bimanual Trajectory Variants", fontsize=16, pad=20)

    for r in results:
        lp = r.trajectory.left_poses[:, :3, 3]
        rp = r.trajectory.right_poses[:, :3, 3]
        kw = dict(linewidth=2, alpha=0.6)
        ax.plot(lp[:,0], lp[:,1], lp[:,2], color='blue', **kw,
                label='Left arm'  if r.index == 0 else "")
        ax.plot(rp[:,0], rp[:,1], rp[:,2], color='red',  **kw,
                label='Right arm' if r.index == 0 else "")

    ax.set_xlabel('X [m]', fontsize=12, labelpad=10)
    ax.set_ylabel('Y [m]', fontsize=12, labelpad=10)
    ax.set_zlabel('Z [m]', fontsize=12, labelpad=10)
    ax.view_init(elev=30, azim=-60)
    ax.legend(fontsize=12)

    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_pane_color((1., 1., 1., 1.))
        axis._axinfo["grid"].update(color="gray", linestyle="-", linewidth=0.5, alpha=0.3)

    all_pts = np.concatenate([
        np.vstack([r.trajectory.left_poses[:, :3, 3]  for r in results]),
        np.vstack([r.trajectory.right_poses[:, :3, 3] for r in results]),
    ])
    lo, hi = all_pts.min(0) - 0.05, all_pts.max(0) + 0.05
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_zlim(lo[2], hi[2])
    plt.tight_layout()


def plot_tracking_errors(results: list[VariantResult]) -> None:
    """Per-arm position and rotation errors over time, with mean ± 1σ envelope."""
    hw = [r for r in results if r.success and r.stats is not None]
    if not hw:
        return

    # Each entry: (title, [(color, attr, label), ...], scale, ylabel)
    panels = [
        ("IK Position Error",
         [('royalblue', 'left_ik_pos_errors', 'Left'), ('tomato', 'right_ik_pos_errors', 'Right')],
         1000, "mm"),
        ("IK Rotation Error",
         [('royalblue', 'left_ik_rot_errors', 'Left'), ('tomato', 'right_ik_rot_errors', 'Right')],
         180 / np.pi, "deg"),
        ("Tracking Position",
         [('royalblue', 'left_track_pos_errors', 'Left'), ('tomato', 'right_track_pos_errors', 'Right')],
         1000, "mm"),
        ("Tracking Rotation",
         [('royalblue', 'left_track_rot_errors', 'Left'), ('tomato', 'right_track_rot_errors', 'Right')],
         180 / np.pi, "deg"),
        ("Tracking Joint RMSE",
         [('mediumpurple', 'track_q_errors', 'All joints')],
         180 / np.pi, "deg"),
    ]

    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    axes[2, 1].set_visible(False)   # 5 panels in a 3×2 grid
    fig.suptitle("Tracking Error over Time", fontsize=14)
    t_grid = np.linspace(0, min(np.array(r.stats.time_meas)[-1] for r in hw), 300)

    for ax, (title, series, scale, ylabel) in zip(axes.flat, panels):
        for color, attr, label in series:
            times_list = [np.array(r.stats.time_meas) for r in hw]
            arrs       = [np.array(getattr(r.stats, attr)) * scale for r in hw]

            for t, a in zip(times_list, arrs):
                ax.plot(t, a, alpha=0.2, linewidth=0.7, color=color)

            interp = np.vstack([np.interp(t_grid, t, a) for t, a in zip(times_list, arrs)])
            mean, std = interp.mean(0), interp.std(0)
            ax.plot(t_grid, mean, color=color, linewidth=2, label=label)
            ax.fill_between(t_grid, mean - std, mean + std, alpha=0.2, color=color)

        ax.set_title(title)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel(f'Error ({ylabel})')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.4)

    plt.tight_layout()


# ===========================================================================
# Entry point
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(description="Evaluate bimanual arm trajectory tracking")
    parser.add_argument("--network-interface", "-n", default="eth0",
                        help="Network interface for DDS (default: eth0)")
    parser.add_argument("--dry-run", action="store_true",
                        help="IK-only evaluation without hardware")
    parser.add_argument("--n-variants", type=int, default=10,
                        help="Number of variants to evaluate (default: 10)")
    parser.add_argument("--animate", action="store_true",
                        help="Animate in MeshCat (dry-run only)")
    parser.add_argument("--output", "-o", default="tracking_results.npz",
                        help="Output path for tracking dataset (default: tracking_results.npz)")
    args = parser.parse_args()

    nominal_left, nominal_right = make_nominal_poses()
    variants = [sample_variant(i, nominal_left, nominal_right) for i in range(args.n_variants)]

    print(_sep("="))
    print("BIMANUAL ARM TRAJECTORY EVALUATION")
    print(_sep("="))
    print(f"  Seed           : {SEED}")
    print(f"  Variants       : {args.n_variants}")
    print(f"  Position std   : {POSITION_SCALE*100:.2f}cm")
    print(f"  Yaw scale      : {np.degrees(YAW_SCALE):.1f} deg")
    print(f"  Mode           : {'DRY RUN (IK only)' if args.dry_run else 'HARDWARE'}")
    print(_sep("="))

    if args.dry_run:
        results = run_dry(variants, args.animate)
    else:
        results = run_hardware(variants, args.network_interface)

    print_aggregate_report(results)

    if not args.dry_run:
        save_dataset(results, args.output)

    plot_trajectory_variants(results)
    plot_tracking_errors(results)
    plt.show()


if __name__ == "__main__":
    main()
