import builtins
import json
import os
import subprocess
import sys
import threading
import time
import traceback
import urllib.request
from pathlib import Path

import numpy as np
import pinocchio as pin
from scipy.spatial.transform import Rotation as R, Slerp

builtins.input = lambda *a, **k: ""
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("ROS_LOCALHOST_ONLY", "1")
os.environ.setdefault("RMW_IMPLEMENTATION", "rmw_fastrtps_cpp")

ROOT = Path("/storage/nacloos/projects/csail-hackathon-clawblox/worlds/unitree-mujoco/G1-Bricklaying-Simulation")
os.chdir(ROOT)
sys.path.insert(0, str(ROOT / "demo"))
sys.path.insert(0, str(ROOT / "src"))

from interface import (
    PickPlace,
    G1TwistCmdNode,
    T_RIGHT_INIT,
    _brick_to_grasp_pose,
    _brick_to_table_pose,
    HAND_PICK_OFFSET,
    CURVE_WALL,
)
from bricklaying.perception.sim_realsense import SIM_INTRINSICS
from bricklaying.perception.realsense import deproject_pixels_to_points
from bricklaying.planning.motion_planner import CartesianTrajectory, CartesianWaypoint
from bricklaying.robot.controller import compute_joint_trajectory
from bricklaying.robot.joint_config import G1JointConfiguration, G1JointGroup, HAND_OPEN_LEFT, HAND_CLOSED_LEFT

FRAME_NPZ = Path("/tmp/codex_latest_camera_frame.npz")
LOG = Path("/tmp/codex_servo_pick_attempt.jsonl")
LOG.write_text("")
RIGHT_KP = float(os.environ.get("RIGHT_KP", "25.0"))
RIGHT_KD = float(os.environ.get("RIGHT_KD", "1.0"))
RIGHT_SHOULDER_PITCH_KP = float(os.environ.get("RIGHT_SHOULDER_PITCH_KP", str(RIGHT_KP)))
RIGHT_SHOULDER_PITCH_KD = float(os.environ.get("RIGHT_SHOULDER_PITCH_KD", str(RIGHT_KD)))
RIGHT_ELBOW_KP = float(os.environ.get("RIGHT_ELBOW_KP", str(RIGHT_KP)))
RIGHT_ELBOW_KD = float(os.environ.get("RIGHT_ELBOW_KD", str(RIGHT_KD)))
MAX_CMD_STEP = float(os.environ.get("MAX_CMD_STEP", "0.0005"))
INIT_MAX_CMD_STEP = float(os.environ.get("INIT_MAX_CMD_STEP", str(MAX_CMD_STEP)))
TRACK_TOL = float(os.environ.get("TRACK_TOL", "0.18"))
INIT_TRACK_TOL = float(os.environ.get("INIT_TRACK_TOL", str(TRACK_TOL)))
PICK_TRACK_TOL = float(os.environ.get("PICK_TRACK_TOL", str(TRACK_TOL)))
UPPER_DQ_ABORT = float(os.environ.get("UPPER_DQ_ABORT", "6.0"))
UPPER_DQ_RECOVER = float(os.environ.get("UPPER_DQ_RECOVER", "1.0"))
SERVO_MAX_RECOVERIES = int(os.environ.get("SERVO_MAX_RECOVERIES", "8"))
PICK_DT = float(os.environ.get("PICK_DT", "0.20"))
IK_MAX_DQ_PER_STEP = float(os.environ.get("IK_MAX_DQ_PER_STEP", "0.25"))
IK_ABORT_STEP = float(os.environ.get("IK_ABORT_STEP", "0.20"))
PICK_FREE_MODE = os.environ.get("PICK_FREE_MODE", "right")
PLANNER_WAYPOINTS_PER_SECOND = float(os.environ.get("PLANNER_WAYPOINTS_PER_SECOND", "5.0"))
CLOSE_FOR_PREPICK = os.environ.get("CLOSE_FOR_PREPICK", "0") == "1"
GRASP_HAND_MODE = os.environ.get("GRASP_HAND_MODE", "grasp")
PREPICK_HIGH_Z = float(os.environ.get("PREPICK_HIGH_Z", "0.35"))
PREPICK_LIFT_Z = float(os.environ.get("PREPICK_LIFT_Z", "0.0"))
PREPICK_LIFT_ATTEMPTS = int(os.environ.get("PREPICK_LIFT_ATTEMPTS", "2"))
PREPICK_ROUTE_MODE = os.environ.get("PREPICK_ROUTE_MODE", "spline")
PREPICK_ROUTE_CLEARANCE = float(os.environ.get("PREPICK_ROUTE_CLEARANCE", "0.16"))
PREPICK_ROUTE_WPS = float(os.environ.get("PREPICK_ROUTE_WPS", "8.0"))
PREPICK_STAGE_ATTEMPTS = int(os.environ.get("PREPICK_STAGE_ATTEMPTS", "1"))
STOP_AFTER_PICK_DESCEND = os.environ.get("STOP_AFTER_PICK_DESCEND", "0") == "1"
MIN_CARRY_DISP = float(os.environ.get("MIN_CARRY_DISP", "0.08"))
MIN_CARRY_Z_DELTA = float(os.environ.get("MIN_CARRY_Z_DELTA", "0.06"))
NAV_TARGET_SD = float(os.environ.get("NAV_TARGET_SD", "-0.03"))
NAV_TARGET_X = float(os.environ.get("NAV_TARGET_X", "0.39"))
NAV_REQUIRE_PICKABLE = os.environ.get("NAV_REQUIRE_PICKABLE", "0") == "1"
NAV_NOPOSE_X = float(os.environ.get("NAV_NOPOSE_X", "0.12"))
NAV_NOPOSE_DURATION = float(os.environ.get("NAV_NOPOSE_DURATION", "0.5"))
PREPICK_ATTEMPTS = int(os.environ.get("PREPICK_ATTEMPTS", "1"))
RESEED_HOLD_DURATION = float(os.environ.get("RESEED_HOLD_DURATION", "1.0"))
LEFT_HAND_MODE = os.environ.get("LEFT_HAND_MODE", "close")
INITIAL_RIGHT_HAND_MODE = os.environ.get("INITIAL_RIGHT_HAND_MODE", "open")
JOINT_FINAL_LABEL_PREFIXES = tuple(
    x.strip() for x in os.environ.get("JOINT_FINAL_LABEL_PREFIXES", "").split(",") if x.strip()
)
PREINIT_RECOVERY = os.environ.get("PREINIT_RECOVERY", "0") == "1"
PREINIT_RIGHT_SHOULDER_ROLL = float(os.environ.get("PREINIT_RIGHT_SHOULDER_ROLL", "-0.35"))
PREINIT_MAX_CMD_STEP = float(os.environ.get("PREINIT_MAX_CMD_STEP", "0.00025"))
PREINIT_TRACK_TOL = float(os.environ.get("PREINIT_TRACK_TOL", str(INIT_TRACK_TOL)))
PREINIT_LEFT_PARK = os.environ.get("PREINIT_LEFT_PARK", "1") == "1"
PREINIT_LEFT_PARK_Q = np.array([
    float(x) for x in os.environ.get("PREINIT_LEFT_PARK_Q", "0.15,0.75,0.0,1.1,0.0,-0.2,0.0").split(",")
])
INITIAL_HAND_SETTLE_TIMEOUT = float(os.environ.get("INITIAL_HAND_SETTLE_TIMEOUT", "12.0"))
ALL_UPPER_KP = os.environ.get("ALL_UPPER_KP")
ALL_UPPER_KD = os.environ.get("ALL_UPPER_KD")
RUN_SCALED_INIT = os.environ.get("RUN_SCALED_INIT", "0") == "1"
SCALED_INIT_DURATION = float(os.environ.get("SCALED_INIT_DURATION", "30.0"))
SCALED_INIT_EXEC_SCALE = float(os.environ.get("SCALED_INIT_EXEC_SCALE", "4.0"))
SCALED_INIT_MAX_QVEL = float(os.environ.get("SCALED_INIT_MAX_QVEL", "8.0"))
SCALED_INIT_FINAL_ERR = float(os.environ.get("SCALED_INIT_FINAL_ERR", "0.15"))
USE_SCALED_JOINT_EXECUTE = os.environ.get("USE_SCALED_JOINT_EXECUTE", "0") == "1"
SCALED_JOINT_EXEC_SCALE = float(os.environ.get("SCALED_JOINT_EXEC_SCALE", str(SCALED_INIT_EXEC_SCALE)))
SCALED_JOINT_MAX_QVEL = float(os.environ.get("SCALED_JOINT_MAX_QVEL", str(SCALED_INIT_MAX_QVEL)))
SCALED_JOINT_FINAL_ERR = float(os.environ.get("SCALED_JOINT_FINAL_ERR", str(SCALED_INIT_FINAL_ERR)))
SCALED_TAU_MODE = os.environ.get("SCALED_TAU_MODE", "rnea_scaled")
SCALED_INIT_TAU_MODE = os.environ.get("SCALED_INIT_TAU_MODE", SCALED_TAU_MODE)
SCALED_PICK_TAU_MODE = os.environ.get("SCALED_PICK_TAU_MODE", SCALED_TAU_MODE)
POST_GRASP_LIFT_Z = float(os.environ.get("POST_GRASP_LIFT_Z", "0.18"))
POST_GRASP_LIFT_DURATION = float(os.environ.get("POST_GRASP_LIFT_DURATION", "22.0"))
PLACE_DURATION = float(os.environ.get("PLACE_DURATION", "30.0"))
PLACE_RETURN_DURATION = float(os.environ.get("PLACE_RETURN_DURATION", "25.0"))
PLACE_TARGET_INDEX = int(os.environ.get("PLACE_TARGET_INDEX", "0"))
PLACE_TARGET_ROW = os.environ.get("PLACE_TARGET_ROW")
SKIP_PICK_RETURN_BEFORE_PLACE = os.environ.get("SKIP_PICK_RETURN_BEFORE_PLACE", "1") == "1"
SKIP_PLACE_RETURN_AFTER_RELEASE = os.environ.get("SKIP_PLACE_RETURN_AFTER_RELEASE", "1") == "1"
SUCCESS_TARGET_DIST = float(os.environ.get("SUCCESS_TARGET_DIST", "0.18"))
PLACE_REQUIRE_PICKABLE = os.environ.get("PLACE_REQUIRE_PICKABLE", "0") == "1"
EXTERNAL_HOLD_HZ = float(os.environ.get("EXTERNAL_HOLD_HZ", "100"))
EXTERNAL_HOLD_WARMUP = float(os.environ.get("EXTERNAL_HOLD_WARMUP", "2.5"))
SETTLE_CONSECUTIVE_SAMPLES = int(os.environ.get("SETTLE_CONSECUTIVE_SAMPLES", "4"))
NAV_HOLD_HZ = float(os.environ.get("NAV_HOLD_HZ", "100"))
RIGHT_INIT_POSE = os.environ.get("RIGHT_INIT_POSE")
POST_NAV_DEMO_INIT = os.environ.get("POST_NAV_DEMO_INIT", "0") == "1"

QVEL_ROBOT_NAMES = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "left_hand_thumb_0_joint",
    "left_hand_thumb_1_joint",
    "left_hand_thumb_2_joint",
    "left_hand_middle_0_joint",
    "left_hand_middle_1_joint",
    "left_hand_index_0_joint",
    "left_hand_index_1_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
    "right_hand_thumb_0_joint",
    "right_hand_thumb_1_joint",
    "right_hand_thumb_2_joint",
    "right_hand_middle_0_joint",
    "right_hand_middle_1_joint",
    "right_hand_index_0_joint",
    "right_hand_index_1_joint",
]


def _jsonable(v):
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, (np.floating, np.integer, np.bool_)):
        return v.item()
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, dict):
        return {k: _jsonable(x) for k, x in v.items()}
    return v


def jlog(event, **kw):
    rec = {"t_wall": time.time(), "event": event}
    rec.update({k: _jsonable(v) for k, v in kw.items()})
    with LOG.open("a") as f:
        f.write(json.dumps(rec) + "\n")
    print("[LOG]", event, {k: rec[k] for k in rec if k not in ("t_wall", "event")}, flush=True)


def apply_pick_gains(dds):
    if (
        RIGHT_SHOULDER_PITCH_KP == RIGHT_KP
        and RIGHT_SHOULDER_PITCH_KD == RIGHT_KD
        and RIGHT_ELBOW_KP == RIGHT_KP
        and RIGHT_ELBOW_KD == RIGHT_KD
    ):
        jlog("pick_gains_skipped", reason="pick gains match base gains")
        return True
    q, _ = dds.get_upper_body_state()
    dds.upper_body_gains["kp"][8] = RIGHT_SHOULDER_PITCH_KP
    dds.upper_body_gains["kd"][8] = RIGHT_SHOULDER_PITCH_KD
    dds.upper_body_gains["kp"][11] = RIGHT_ELBOW_KP
    dds.upper_body_gains["kd"][11] = RIGHT_ELBOW_KD
    z = np.zeros_like(q)
    for _ in range(100):
        dds.set_upper_body_target(q, z, z)
        time.sleep(0.01)
    qvel = maxqvel()
    udq = upper_max_dq(dds)
    jlog("pick_gains_applied",
         right_shoulder_pitch_kp=RIGHT_SHOULDER_PITCH_KP,
         right_shoulder_pitch_kd=RIGHT_SHOULDER_PITCH_KD,
         right_elbow_kp=RIGHT_ELBOW_KP,
         right_elbow_kd=RIGHT_ELBOW_KD,
         qvel=qvel,
         upper_max_dq=udq)
    return udq <= 0.5 and qvel <= 1.0


def set_hands(demo, label, right):
    demo.dds.set_hand_mode(left=LEFT_HAND_MODE, right=right)
    jlog("hand_mode", label=label, left=LEFT_HAND_MODE, right=right, qvel=maxqvel(), contacts=contact_snapshot())


def set_gravity_hold(demo, q, label=None):
    tau = pin.computeGeneralizedGravity(
        demo.urdf_model.reduced_robot.model,
        demo.urdf_model.reduced_robot.data,
        q,
    )
    demo.dds.set_upper_body_target(q, np.zeros_like(q), tau)
    if label:
        jlog("main_hold_synced", label=label, qvel=maxqvel(), upper_max_dq=upper_max_dq(demo.dds))


def set_initial_hands(demo):
    if INITIAL_RIGHT_HAND_MODE == "none":
        jlog("initial_hand_unchanged", qvel=maxqvel(), contacts=contact_snapshot())
        return True
    if INITIAL_RIGHT_HAND_MODE == "current":
        _, right_q = demo.dds.get_hand_state()
        if right_q is None:
            jlog("initial_hand_current_missing")
            return False
        left_q = HAND_OPEN_LEFT if LEFT_HAND_MODE == "open" else HAND_CLOSED_LEFT
        demo.dds.set_hand_target(left_q, right_q)
        jlog("initial_hand_left_mode_right_current", left=LEFT_HAND_MODE, right_q=right_q, qvel=maxqvel(), contacts=contact_snapshot())
        return True
    set_hands(demo, "initial_hand", INITIAL_RIGHT_HAND_MODE)
    return True


class FileCamera:
    @property
    def intrinsics(self):
        return SIM_INTRINSICS

    def get_frames(self):
        data = np.load(FRAME_NPZ)
        return data["color"], data["depth"]

    def get_point_cloud(self, color, depth, mask=None):
        return deproject_pixels_to_points(depth, color, self.intrinsics, mask)

    def stop(self):
        pass


def observe():
    with urllib.request.urlopen("http://127.0.0.1:18140/observe", timeout=2.0) as r:
        d = json.loads(r.read().decode("utf-8"))
    return d, d["state"]


def maxqvel():
    _, s = observe()
    return max(abs(float(v)) for v in s["qvel"])


def qvel_info():
    _, s = observe()
    qvel = np.array(s["qvel"], dtype=float)
    idx = int(np.argmax(np.abs(qvel)))
    top = sorted(
        [(float(abs(v)), int(i), float(v), qvel_name(int(i))) for i, v in enumerate(qvel)],
        reverse=True,
    )[:8]
    return {
        "qvel": float(abs(qvel[idx])),
        "qvel_idx": idx,
        "qvel_name": qvel_name(idx),
        "qvel_value": float(qvel[idx]),
        "qvel_top": top,
        "qvel_vector": qvel,
    }


def qvel_name(idx):
    if idx < len(QVEL_ROBOT_NAMES):
        return QVEL_ROBOT_NAMES[idx]
    j = idx - len(QVEL_ROBOT_NAMES)
    brick = j // 6
    comp = j % 6
    return f"brick_{brick}_free_dof_{comp}"


def configured_right_init_pose():
    if not RIGHT_INIT_POSE:
        return T_RIGHT_INIT
    vals = [float(x) for x in RIGHT_INIT_POSE.split(",")]
    if len(vals) != 3:
        raise ValueError("RIGHT_INIT_POSE must be x,y,z")
    pose = np.eye(4)
    pose[:3, 3] = np.array(vals, dtype=float)
    return pose


def upper_max_dq(dds):
    _, dq = dds.get_upper_body_state()
    return float(np.max(np.abs(dq)))


def contact_snapshot():
    d, _ = observe()
    return d.get("contacts", [])


def has_hand_table_contact():
    hits = []
    for c in contact_snapshot():
        names = {
            str(c.get("geom1_name")),
            str(c.get("geom2_name")),
            str(c.get("body1_name")),
            str(c.get("body2_name")),
        }
        body_names = {str(c.get("body1_name")), str(c.get("body2_name"))}
        if "table_collider" in names and any(n.startswith(("right_hand", "left_hand")) for n in body_names):
            hits.append(c)
    return hits


def has_right_hand_table_contact():
    return has_hand_table_contact()


def allow_table_contact_for_label(label):
    return label.startswith("post_grasp_lift")


def brick_states():
    _, s = observe()
    q = s["qpos"]
    return [q[a:a + 7] for a in [43, 50, 57, 64, 71]]


def wait_settled(label, limit=0.05, timeout=20):
    t0 = time.time()
    last = None
    consecutive = 0
    while time.time() - t0 < timeout:
        d, s = observe()
        qvel = np.array(s["qvel"], dtype=float)
        idx = int(np.argmax(np.abs(qvel)))
        mq = float(abs(qvel[idx]))
        last = (d["time"], mq, idx, qvel_name(idx), s.get("mocap_pos"))
        if d.get("ready") and d.get("time", 0) > 1 and mq < limit:
            consecutive += 1
        else:
            consecutive = 0
        if consecutive >= SETTLE_CONSECUTIVE_SAMPLES:
            jlog("settled", label=label, sim_time=d["time"], max_qvel=mq,
                 qvel_idx=idx, qvel_name=qvel_name(idx), qvel_value=float(qvel[idx]),
                 samples=consecutive, mocap=s.get("mocap_pos"))
            return True
        time.sleep(0.2)
    jlog("settle_timeout", label=label, last=last)
    return False


def capture_external_frame(label, timeout=20):
    res = subprocess.run(
        [sys.executable, "/tmp/codex_capture_frame.py", str(FRAME_NPZ), str(timeout)],
        cwd=str(ROOT),
        text=True,
        capture_output=True,
        timeout=timeout + 10,
        env=os.environ.copy(),
    )
    jlog("external_camera_capture", label=label, returncode=res.returncode,
         stdout=res.stdout[-400:], stderr=res.stderr[-400:])
    return res.returncode == 0 and FRAME_NPZ.exists()


def estimate_best(demo, label):
    if not capture_external_frame(label):
        return None, False, float("inf")
    jlog("estimate_subprocess_start", label=label, frame=str(FRAME_NPZ))
    try:
        res = subprocess.run(
            [sys.executable, "/tmp/codex_estimate_frame.py", str(FRAME_NPZ)],
            cwd=str(ROOT),
            text=True,
            capture_output=True,
            timeout=120,
            env=os.environ.copy(),
        )
    except subprocess.TimeoutExpired as e:
        jlog("estimate_timeout", label=label, timeout=e.timeout,
             stdout=(e.stdout or "")[-800:], stderr=(e.stderr or "")[-800:])
        return None, False, float("inf")
    stdout = res.stdout.strip().splitlines()
    payload = None
    for line in reversed(stdout):
        try:
            payload = json.loads(line)
            break
        except Exception:
            pass
    if res.returncode != 0 or payload is None:
        jlog("estimate_subprocess_failed", label=label, returncode=res.returncode,
             stdout=res.stdout[-800:], stderr=res.stderr[-800:])
        return None, False, float("inf")
    best = payload.get("best")
    if best is None:
        jlog("estimate", label=label, n=payload.get("n", 0), candidates=payload.get("candidates", []), best_pos=None,
             stderr=res.stderr[-400:])
        return None, False, float("inf")
    class PoseObj:
        pass
    pose = PoseObj()
    pose.position = np.array(best["pos"], dtype=float)
    pose.transform = np.array(best["transform"], dtype=float)
    pose.icp_fitness = best["fitness"]
    pose.icp_rmse = best["rmse"]
    pickable = bool(best["pickable"])
    sd = float(best["sd"])
    jlog("estimate", label=label, n=payload.get("n", 0), candidates=payload.get("candidates", []),
         best_pos=pose.position, best_sd=sd, pickable=pickable, stderr=res.stderr[-400:])
    return pose, pickable, sd


def publish_pulse(node, x, theta, duration, hz=20):
    stop_at = time.time() + duration
    while time.time() < stop_at:
        node.publish_twist(x, theta)
        time.sleep(1.0 / hz)
    for _ in range(10):
        node.publish_twist(0, 0)
        time.sleep(0.05)


def pulse_and_check(node, label, x, theta, duration, brick0, dds=None):
    info = qvel_info()
    jlog("pulse_start", label=label, x=x, theta=theta, duration=duration, bricks=brick_states(),
         upper_max_dq=None if dds is None else upper_max_dq(dds), **info)
    publish_pulse(node, x, theta, duration)
    time.sleep(0.5)
    wait_settled("after_" + label, limit=0.10, timeout=8)
    now = brick_states()
    disp = [float(np.linalg.norm(np.array(now[i][:3]) - np.array(brick0[i][:3]))) for i in range(5)]
    d, s = observe()
    info = qvel_info()
    jlog("pulse_end", label=label, sim_time=d["time"], mocap=s.get("mocap_pos"),
         brick_disp=disp, bricks=now, upper_max_dq=None if dds is None else upper_max_dq(dds), **info)
    return max(disp), info["qvel"]


def confirm_settled_after_nav(label, qv, limit=0.25, dds=None):
    if qv <= limit:
        return qv, True
    jlog("nav_qvel_recheck_start", label=label, input_qvel=qv,
         upper_max_dq=None if dds is None else upper_max_dq(dds), **qvel_info())
    if not wait_settled(label + "_recheck", limit=0.10, timeout=8):
        info = qvel_info()
        qv2 = info["qvel"]
        jlog("nav_qvel_recheck_failed", label=label,
             upper_max_dq=None if dds is None else upper_max_dq(dds), **info)
        return qv2, False
    time.sleep(0.5)
    info = qvel_info()
    qv2 = info["qvel"]
    ok = qv2 <= limit
    jlog("nav_qvel_recheck_done", label=label, ok=ok,
         upper_max_dq=None if dds is None else upper_max_dq(dds), **info)
    return qv2, ok


def servo_joint_path(dds, q_path, label, max_cmd_step=None, cmd_hz=50, track_tol=None, stall_timeout=4.0, urdf_model=None):
    if max_cmd_step is None:
        max_cmd_step = MAX_CMD_STEP
    if track_tol is None:
        track_tol = TRACK_TOL
    q_cmd, _ = dds.get_upper_body_state()
    q_cmd = q_cmd.copy()
    last_log = 0.0
    wait_started = None
    max_seen_qvel = 0.0
    max_seen_err = 0.0
    recoveries = 0
    def hold_current(reason, wi, upper_dq_value, qvel, duration=1.0):
        nonlocal q_cmd, wait_started, recoveries
        q_hold, _ = dds.get_upper_body_state()
        q_hold = q_hold.copy()
        if urdf_model is None:
            tau_cmd = np.zeros_like(q_hold)
        else:
            tau_cmd = pin.computeGeneralizedGravity(
                urdf_model.reduced_robot.model,
                urdf_model.reduced_robot.data,
                q_hold,
            )
        jlog("servo_velocity_recovery_start", label=label, reason=reason, wi=wi,
             recovery=recoveries + 1, qvel=qvel, upper_max_dq=upper_dq_value, q_hold=q_hold)
        for _ in range(int(duration * cmd_hz)):
            dds.set_upper_body_target(q_hold, np.zeros_like(q_hold), tau_cmd)
            time.sleep(1.0 / cmd_hz)
        q_cmd = q_hold
        wait_started = None
        recoveries += 1
        jlog("servo_velocity_recovery_done", label=label, wi=wi,
             recovery=recoveries, qvel=maxqvel(), upper_max_dq=upper_max_dq(dds))

    for wi, q_goal in enumerate(q_path):
        while True:
            q_meas, _ = dds.get_upper_body_state()
            qerr = q_cmd - q_meas
            err_idx = int(np.argmax(np.abs(qerr)))
            cmd_err = float(np.max(np.abs(qerr)))
            if cmd_err > track_tol:
                if urdf_model is None:
                    tau_cmd = np.zeros_like(q_cmd)
                else:
                    tau_cmd = pin.computeGeneralizedGravity(
                        urdf_model.reduced_robot.model,
                        urdf_model.reduced_robot.data,
                        q_cmd,
                    )
                dds.set_upper_body_target(q_cmd, np.zeros_like(q_cmd), tau_cmd)
                now = time.time()
                if wait_started is None:
                    wait_started = now
                if now - last_log > 1.0:
                    qvel = maxqvel()
                    _, dq_meas = dds.get_upper_body_state()
                    upper_max_dq = float(np.max(np.abs(dq_meas)))
                    table_hits = has_right_hand_table_contact()
                    max_seen_qvel = max(max_seen_qvel, qvel)
                    max_seen_err = max(max_seen_err, cmd_err)
                    jlog(
                        "servo_wait",
                        label=label,
                        wi=wi,
                        n=len(q_path),
                        qvel=qvel,
                        upper_max_dq=upper_max_dq,
                        cmd_track=cmd_err,
                        err_idx=err_idx,
                        err_value=float(qerr[err_idx]),
                        q_cmd=q_cmd,
                        q_meas=q_meas,
                        contacts=contact_snapshot(),
                        table_hits=table_hits,
                        wait=now - wait_started,
                    )
                    if table_hits:
                        if allow_table_contact_for_label(label):
                            jlog("servo_allow_table_contact", label=label, wi=wi, hits=table_hits)
                        else:
                            jlog("servo_abort_table_contact", label=label, wi=wi, hits=table_hits)
                            return False, {"status": "table_contact", "max_qvel": max_seen_qvel, "max_cmd_err": max_seen_err}
                    if upper_max_dq > UPPER_DQ_ABORT:
                        jlog("servo_abort_upper_dq", label=label, wi=wi, upper_max_dq=upper_max_dq, **qvel_info())
                        return False, {"status": "unsafe", "max_qvel": max_seen_qvel, "max_cmd_err": max_seen_err}
                    if upper_max_dq > UPPER_DQ_RECOVER and recoveries < SERVO_MAX_RECOVERIES:
                        hold_current("wait_upper_dq", wi, upper_max_dq, qvel)
                        continue
                    if now - wait_started > stall_timeout and upper_max_dq < 0.02 and qvel < 0.5:
                        jlog("servo_stalled", label=label, wi=wi, qvel=qvel, upper_max_dq=upper_max_dq, cmd_track=cmd_err)
                        return False, {"status": "stalled", "wi": wi, "max_qvel": max_seen_qvel, "max_cmd_err": max_seen_err}
                    last_log = now
                time.sleep(1.0 / cmd_hz)
                continue
            wait_started = None
            diff = q_goal - q_cmd
            if float(np.max(np.abs(diff))) <= max_cmd_step:
                q_cmd = q_goal.copy()
                if urdf_model is None:
                    tau_cmd = np.zeros_like(q_cmd)
                else:
                    tau_cmd = pin.computeGeneralizedGravity(
                        urdf_model.reduced_robot.model,
                        urdf_model.reduced_robot.data,
                        q_cmd,
                    )
                dds.set_upper_body_target(q_cmd, np.zeros_like(q_cmd), tau_cmd)
                break
            q_cmd = q_cmd + np.clip(diff, -max_cmd_step, max_cmd_step)
            if urdf_model is None:
                tau_cmd = np.zeros_like(q_cmd)
            else:
                tau_cmd = pin.computeGeneralizedGravity(
                    urdf_model.reduced_robot.model,
                    urdf_model.reduced_robot.data,
                    q_cmd,
                )
            dds.set_upper_body_target(q_cmd, np.zeros_like(q_cmd), tau_cmd)
            now = time.time()
            if now - last_log > 1.0:
                qvel = maxqvel()
                q_meas2, dq_meas = dds.get_upper_body_state()
                qerr_vec = q_cmd - q_meas2
                err_idx = int(np.argmax(np.abs(qerr_vec)))
                qerr = float(np.max(np.abs(qerr_vec)))
                upper_max_dq = float(np.max(np.abs(dq_meas)))
                max_seen_qvel = max(max_seen_qvel, qvel)
                max_seen_err = max(max_seen_err, qerr)
                extra = {}
                if qerr > 0.14 or upper_max_dq > 1.0:
                    table_hits = has_right_hand_table_contact()
                    extra = {
                        "err_idx": err_idx,
                        "err_value": float(qerr_vec[err_idx]),
                        "q_cmd": q_cmd,
                        "q_meas": q_meas2,
                        "contacts": contact_snapshot(),
                        "table_hits": table_hits,
                    }
                    if table_hits:
                        if allow_table_contact_for_label(label):
                            jlog("servo_allow_table_contact", label=label, wi=wi, hits=table_hits)
                        else:
                            jlog("servo_abort_table_contact", label=label, wi=wi, hits=table_hits)
                            return False, {"status": "table_contact", "max_qvel": max_seen_qvel, "max_cmd_err": max_seen_err}
                jlog("servo_progress", label=label, wi=wi, n=len(q_path), qvel=qvel, upper_max_dq=upper_max_dq, cmd_track=qerr, dqnorm=float(np.linalg.norm(dq_meas)), **extra)
                if upper_max_dq > UPPER_DQ_ABORT:
                    jlog("servo_abort_upper_dq", label=label, wi=wi, upper_max_dq=upper_max_dq, **qvel_info())
                    return False, {"status": "unsafe", "max_qvel": max_seen_qvel, "max_cmd_err": max_seen_err}
                if upper_max_dq > UPPER_DQ_RECOVER and recoveries < SERVO_MAX_RECOVERIES:
                    hold_current("progress_upper_dq", wi, upper_max_dq, qvel)
                last_log = now
            time.sleep(1.0 / cmd_hz)
    for _ in range(100):
        if urdf_model is None:
            tau_cmd = np.zeros_like(q_cmd)
        else:
            tau_cmd = pin.computeGeneralizedGravity(
                urdf_model.reduced_robot.model,
                urdf_model.reduced_robot.data,
                q_cmd,
            )
        dds.set_upper_body_target(q_cmd, np.zeros_like(q_cmd), tau_cmd)
        time.sleep(1.0 / cmd_hz)
    qvel = maxqvel()
    q_meas, _ = dds.get_upper_body_state()
    final_err = float(np.max(np.abs(q_cmd - q_meas)))
    jlog("servo_done", label=label, qvel=qvel, final_cmd_err=final_err, max_qvel=max(max_seen_qvel, qvel), max_cmd_err=max(max_seen_err, final_err))
    ok = qvel < 1.0 and final_err < track_tol
    return ok, {"status": "done" if ok else "final_lag", "max_qvel": max(max_seen_qvel, qvel), "max_cmd_err": max(max_seen_err, final_err)}


def preinit_recovery(demo):
    q0, _ = demo.dds.get_upper_body_state()
    q_target = q0.copy()
    q_target[9] = PREINIT_RIGHT_SHOULDER_ROLL
    max_delta = float(np.max(np.abs(q_target - q0)))
    n = max(1, int(np.ceil(max_delta / PREINIT_MAX_CMD_STEP)))
    q_path = np.linspace(q0, q_target, n + 1)
    jlog(
        "preinit_recovery_start",
        q0=q0,
        q_target=q_target,
        n=len(q_path),
        max_delta=max_delta,
        contacts=contact_snapshot(),
        qvel=maxqvel(),
    )
    ok, info = servo_joint_path(
        demo.dds,
        q_path,
        "preinit_right_shoulder_roll",
        max_cmd_step=PREINIT_MAX_CMD_STEP,
        cmd_hz=100,
        track_tol=PREINIT_TRACK_TOL,
        stall_timeout=6.0,
        urdf_model=demo.urdf_model,
    )
    qf, _ = demo.dds.get_upper_body_state()
    jlog(
        "preinit_recovery_done",
        ok=ok,
        info=info,
        q_final=qf,
        qvel=maxqvel(),
        contacts=contact_snapshot(),
    )
    return ok


def preinit_left_park(demo):
    if len(PREINIT_LEFT_PARK_Q) != 7:
        jlog("preinit_left_park_bad_config", q=PREINIT_LEFT_PARK_Q)
        return False
    q0, _ = demo.dds.get_upper_body_state()
    q_target = q0.copy()
    q_target[1:8] = PREINIT_LEFT_PARK_Q
    max_delta = float(np.max(np.abs(q_target - q0)))
    n = max(1, int(np.ceil(max_delta / PREINIT_MAX_CMD_STEP)))
    q_path = np.linspace(q0, q_target, n + 1)
    jlog(
        "preinit_left_park_start",
        q0=q0,
        q_target=q_target,
        n=len(q_path),
        max_delta=max_delta,
        contacts=contact_snapshot(),
        qvel=maxqvel(),
    )
    _, right_hand = demo.dds.get_hand_state()
    if right_hand is None:
        right_hand = HAND_OPEN_RIGHT
    for _ in range(100):
        demo.dds.set_hand_target(HAND_OPEN_LEFT, right_hand)
        time.sleep(0.01)
    ok, info = servo_joint_path(
        demo.dds,
        q_path,
        "preinit_left_park",
        max_cmd_step=PREINIT_MAX_CMD_STEP,
        cmd_hz=100,
        track_tol=PREINIT_TRACK_TOL,
        stall_timeout=6.0,
        urdf_model=demo.urdf_model,
    )
    qf, _ = demo.dds.get_upper_body_state()
    jlog(
        "preinit_left_park_done",
        ok=ok,
        info=info,
        q_final=qf,
        qvel=maxqvel(),
        contacts=contact_snapshot(),
    )
    return ok


def execute_joint_traj_scaled(demo, jt, label, scale):
    zstats = {"max_abs_qerr": 0.0, "max_cmd_err": 0.0, "max_qvel": 0.0}
    start = time.time()
    last_log = 0.0
    jlog("scaled_joint_execute_start", label=label, n=jt.n_waypoints, duration=jt.duration, scale=scale)
    while True:
        elapsed = (time.time() - start) / scale
        done = elapsed >= jt.duration
        idx = (
            jt.n_waypoints - 1
            if done
            else min(max(int(np.searchsorted(jt.timestamps, elapsed, side="right") - 1), 0), jt.n_waypoints - 1)
        )
        q_target = jt.q[idx]
        dq_target = jt.dq[idx] / scale
        tau_mode = SCALED_INIT_TAU_MODE if label == "scaled_init" else SCALED_PICK_TAU_MODE
        if tau_mode == "zero":
            tau_ff = np.zeros_like(dq_target)
        elif tau_mode == "gravity":
            tau_ff = pin.computeGeneralizedGravity(
                demo.urdf_model.reduced_robot.model,
                demo.urdf_model.reduced_robot.data,
                q_target,
            )
        elif tau_mode == "rnea_scaled":
            tau_ff = pin.rnea(
                demo.urdf_model.reduced_robot.model,
                demo.urdf_model.reduced_robot.data,
                q_target,
                dq_target,
                np.zeros_like(dq_target),
            )
        else:
            tau_ff = jt.tau_ff[idx]
        q_meas, _ = demo.dds.get_upper_body_state()
        qerr_vec = q_target - q_meas
        err_idx = int(np.argmax(np.abs(qerr_vec)))
        abs_qerr = float(abs(qerr_vec[err_idx]))
        zstats["max_abs_qerr"] = max(zstats["max_abs_qerr"], abs_qerr)
        zstats["max_cmd_err"] = max(zstats["max_cmd_err"], abs_qerr)
        demo.dds.set_upper_body_target(q_target, dq_target, tau_ff)
        now = time.time()
        if now - last_log > 2.0 or done:
            qvinfo = qvel_info()
            qvel = qvinfo["qvel"]
            contacts = contact_snapshot()
            zstats["max_qvel"] = max(zstats["max_qvel"], qvel)
            extra = {}
            if qvel > 1.0 or abs_qerr > 0.08:
                extra = {
                    "q_target": q_target,
                    "q_meas": q_meas,
                    "qerr_vec": qerr_vec,
                    "dq_target": dq_target,
                    "tau_ff": tau_ff,
                }
            jlog("scaled_joint_execute_progress", label=label, idx=idx, n=jt.n_waypoints,
                 elapsed=elapsed, qvel=qvel, abs_qerr=abs_qerr, err_idx=err_idx,
                 err_value=float(qerr_vec[err_idx]), tau_mode=tau_mode, contacts=contacts[:8],
                 qvel_idx=qvinfo["qvel_idx"], qvel_value=qvinfo["qvel_value"],
                 qvel_vector=qvinfo["qvel_vector"], **extra)
            table_hits = has_right_hand_table_contact()
            if table_hits:
                if allow_table_contact_for_label(label):
                    jlog("scaled_joint_execute_allow_table_contact", label=label, idx=idx, hits=table_hits, abs_qerr=abs_qerr, stats=zstats, **qvinfo)
                else:
                    jlog("scaled_joint_execute_abort_table_contact", label=label, idx=idx, hits=table_hits, abs_qerr=abs_qerr, stats=zstats, **qvinfo)
                    return False, {"status": "table_contact", **zstats}
            if qvel > SCALED_JOINT_MAX_QVEL:
                jlog("scaled_joint_execute_abort", label=label, idx=idx, abs_qerr=abs_qerr, stats=zstats, **qvinfo)
                return False, {"status": "unsafe", **zstats}
            last_log = now
        if done:
            break
        time.sleep(demo.controller.dt)
    q_final = jt.q[-1]
    for _ in range(200):
        tau_hold = pin.computeGeneralizedGravity(
            demo.urdf_model.reduced_robot.model,
            demo.urdf_model.reduced_robot.data,
            q_final,
        )
        demo.dds.set_upper_body_target(q_final, np.zeros_like(q_final), tau_hold)
        time.sleep(demo.controller.dt)
    qvel = maxqvel()
    q_meas, _ = demo.dds.get_upper_body_state()
    final_abs_err = float(np.max(np.abs(q_final - q_meas)))
    final_table_hits = has_right_hand_table_contact()
    zstats["max_qvel"] = max(zstats["max_qvel"], qvel)
    zstats["final_abs_qerr"] = final_abs_err
    ok = qvel < 1.0 and final_abs_err < SCALED_JOINT_FINAL_ERR and not final_table_hits
    status = "done" if ok else ("table_contact" if final_table_hits else "final_lag")
    jlog("scaled_joint_execute_done", label=label, ok=ok, qvel=qvel, final_abs_qerr=final_abs_err, contacts=contact_snapshot(), table_hits=final_table_hits, stats=zstats)
    return ok, {"status": status, **zstats}


def scaled_init(demo):
    traj = plan_init(demo, duration=SCALED_INIT_DURATION)
    q_hold, _ = demo.dds.get_upper_body_state()
    set_gravity_hold(demo, q_hold, "scaled_init_before_external_hold")
    ext = ExternalHoldPublisher(q_hold, hz=EXTERNAL_HOLD_HZ)
    ext.__enter__()
    try:
        jt = compute_joint_trajectory(
            demo.ik_solver, demo.urdf_model, traj, q_hold, dt=demo.controller.dt,
            max_dq_per_step=IK_MAX_DQ_PER_STEP, max_ik_pos_error=0.08, max_ik_rot_error=1.2,
            free_joints=demo.ik_solver.right_q_idx,
        )
        set_gravity_hold(demo, q_hold, "scaled_init_before_external_stop")
    finally:
        ext.__exit__(*sys.exc_info())
    set_gravity_hold(demo, q_hold, "scaled_init_after_external_hold")
    wait_settled("after_scaled_init_precompute", limit=0.10, timeout=30)
    return execute_joint_traj_scaled(demo, jt, "scaled_init", SCALED_INIT_EXEC_SCALE)


class HoldPublisher:
    def __init__(self, dds, q, urdf_model=None, hz=50):
        self.dds = dds
        self.q = q.copy()
        self.urdf_model = urdf_model
        self.hz = hz
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        z = np.zeros_like(self.q)
        while not self._stop.is_set():
            if self.urdf_model is None:
                tau = z
            else:
                tau = pin.computeGeneralizedGravity(
                    self.urdf_model.reduced_robot.model,
                    self.urdf_model.reduced_robot.data,
                    self.q,
                )
            self.dds.set_upper_body_target(self.q, z, tau)
            time.sleep(1.0 / self.hz)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self._stop.set()
        self._thread.join(timeout=1.0)


class ExternalHoldPublisher:
    def __init__(self, q, hz=50):
        self.q = q.copy()
        self.hz = hz
        self.proc = None
        self.stdout = None
        self.stderr = None

    def __enter__(self):
        stamp = str(int(time.time() * 1000))
        out_path = Path(f"/tmp/codex_upper_hold_{stamp}.out")
        err_path = Path(f"/tmp/codex_upper_hold_{stamp}.err")
        self.stdout = out_path.open("w")
        self.stderr = err_path.open("w")
        qvel_before = maxqvel()
        self.proc = subprocess.Popen(
            [sys.executable, "/tmp/codex_upper_hold.py", json.dumps(self.q.tolist()), str(self.hz)],
            cwd=str(ROOT),
            env=os.environ.copy(),
            stdout=self.stdout,
            stderr=self.stderr,
        )
        time.sleep(EXTERNAL_HOLD_WARMUP)
        alive = self.proc.poll() is None
        qvel_after = maxqvel()
        jlog("external_hold_started", pid=self.proc.pid, alive=alive, qvel_before=qvel_before,
             qvel_after=qvel_after, stdout=str(out_path), stderr=str(err_path))
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.proc is not None and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=2.0)
        rc = None if self.proc is None else self.proc.returncode
        if self.stdout is not None:
            self.stdout.close()
        if self.stderr is not None:
            self.stderr.close()
        jlog("external_hold_stopped", returncode=rc, qvel=maxqvel())


def servo_execute(demo, traj, free_joints, label, dt=0.20, track_tol=None, max_cmd_step=None):
    q0, _ = demo.dds.get_upper_body_state()
    q0_safe = G1JointConfiguration.clamp_positions(q0, G1JointGroup.UPPER_BODY)
    jlog("servo_precompute_start", label=label, q0=q0, qvel=maxqvel(), q0_clamp_delta=q0_safe - q0, free_joints=free_joints, contacts=contact_snapshot())
    set_gravity_hold(demo, q0, f"{label}_before_external_hold")
    ext = ExternalHoldPublisher(q0, hz=EXTERNAL_HOLD_HZ)
    ext.__enter__()
    try:
        jt = compute_joint_trajectory(
            demo.ik_solver, demo.urdf_model, traj, q0, dt=dt,
            max_dq_per_step=IK_MAX_DQ_PER_STEP, max_ik_pos_error=0.08, max_ik_rot_error=1.2,
            free_joints=free_joints,
        )
        set_gravity_hold(demo, q0, f"{label}_before_external_stop")
    finally:
        ext.__exit__(*sys.exc_info())
    set_gravity_hold(demo, q0, f"{label}_after_external_hold")
    qv_after_precompute = maxqvel()
    upper_dq_after_precompute = upper_max_dq(demo.dds)
    q_after, _ = demo.dds.get_upper_body_state()
    jlog("servo_precompute_done", label=label, qvel=qv_after_precompute, upper_max_dq=upper_dq_after_precompute, q_delta=q_after - q0, contacts=contact_snapshot())
    if upper_dq_after_precompute > 0.5:
        jlog("servo_abort_precompute_motion", label=label, qvel=qv_after_precompute, upper_max_dq=upper_dq_after_precompute)
        return False, {"status": "precompute_motion", "max_qvel": qv_after_precompute, "max_cmd_err": 0.0}
    if len(jt.q) > 0:
        if len(jt.q) > 1:
            path_steps = np.diff(jt.q, axis=0)
            abs_steps = np.abs(path_steps)
            max_step_flat = int(np.argmax(abs_steps))
            max_step_wi, max_step_idx = np.unravel_index(max_step_flat, abs_steps.shape)
            max_path_step = float(abs_steps[max_step_wi, max_step_idx])
        else:
            path_steps = np.zeros((0, len(q0)))
            max_step_wi = 0
            max_step_idx = 0
            max_path_step = 0.0
        jlog(
            "servo_path_diagnostics",
            label=label,
            q0_to_path0=jt.q[0] - q0,
            max_q0_to_path0=float(np.max(np.abs(jt.q[0] - q0))),
            path0_to_path1=(jt.q[1] - jt.q[0]) if len(jt.q) > 1 else np.zeros_like(q0),
            max_path0_to_path1=float(np.max(np.abs(jt.q[1] - jt.q[0]))) if len(jt.q) > 1 else 0.0,
            max_path_step=max_path_step,
            max_path_step_wi=int(max_step_wi),
            max_path_step_idx=int(max_step_idx),
            max_path_step_delta=path_steps[max_step_wi] if len(jt.q) > 1 else np.zeros_like(q0),
            max_path_step_q_before=jt.q[max_step_wi] if len(jt.q) > 1 else jt.q[0],
            max_path_step_q_after=jt.q[max_step_wi + 1] if len(jt.q) > 1 else jt.q[0],
            ik_abort_step=IK_ABORT_STEP,
        )
        if max_path_step > IK_ABORT_STEP:
            jlog(
                "servo_abort_ik_path_jump",
                label=label,
                max_path_step=max_path_step,
                max_path_step_wi=int(max_step_wi),
                max_path_step_idx=int(max_step_idx),
                ik_abort_step=IK_ABORT_STEP,
                qvel=qv_after_precompute,
                upper_max_dq=upper_dq_after_precompute,
            )
            return False, {"status": "ik_path_jump", "max_qvel": qv_after_precompute, "max_cmd_err": 0.0}
        if USE_SCALED_JOINT_EXECUTE:
            jt.q[0] = q0.copy()
            jlog(
                "servo_scaled_joint_execute_selected",
                label=label,
                scale=SCALED_JOINT_EXEC_SCALE,
                n=jt.n_waypoints,
                duration=jt.duration,
            )
            return execute_joint_traj_scaled(demo, jt, label, SCALED_JOINT_EXEC_SCALE)
        jt.q[0] = q0.copy()
    q_path = jt.q
    if any(label.startswith(prefix) for prefix in JOINT_FINAL_LABEL_PREFIXES) and len(jt.q) > 1:
        q_path = np.linspace(q0, jt.q[-1], len(jt.q))
        jlog(
            "servo_joint_final_path",
            label=label,
            n=len(q_path),
            final_delta=jt.q[-1] - q0,
            max_final_delta=float(np.max(np.abs(jt.q[-1] - q0))),
        )
    return servo_joint_path(
        demo.dds,
        q_path,
        label,
        track_tol=track_tol,
        max_cmd_step=max_cmd_step,
        urdf_model=demo.urdf_model,
    )


def hold_measured(demo, label, duration=1.0, hz=50):
    q, _ = demo.dds.get_upper_body_state()
    q = q.copy()
    tau = pin.computeGeneralizedGravity(
        demo.urdf_model.reduced_robot.model,
        demo.urdf_model.reduced_robot.data,
        q,
    )
    for _ in range(int(duration * hz)):
        demo.dds.set_upper_body_target(q, np.zeros_like(q), tau)
        time.sleep(1.0 / hz)
    jlog("hold_measured", label=label, qvel=maxqvel(), q=q)


def servo_execute_replanned(demo, plan_fn, free_joints, label, attempts=4, dt=0.20, track_tol=None, max_cmd_step=None):
    last_info = None
    for attempt in range(attempts):
        q_hold, _ = demo.dds.get_upper_body_state()
        jlog("servo_attempt_plan_start", label=label, attempt=attempt, q=q_hold, qvel=maxqvel(), upper_max_dq=upper_max_dq(demo.dds), contacts=contact_snapshot())
        set_gravity_hold(demo, q_hold, f"{label}_plan_before_external_hold_{attempt}")
        with ExternalHoldPublisher(q_hold, hz=EXTERNAL_HOLD_HZ):
            traj = plan_fn()
        set_gravity_hold(demo, q_hold, f"{label}_plan_after_external_hold_{attempt}")
        q_after_plan, _ = demo.dds.get_upper_body_state()
        table_hits = has_right_hand_table_contact()
        start_qvel = maxqvel()
        start_upper_dq = upper_max_dq(demo.dds)
        jlog("servo_attempt_start", label=label, attempt=attempt, n=traj.n_waypoints, duration=traj.duration, qvel=start_qvel, upper_max_dq=start_upper_dq, q_delta=q_after_plan - q_hold, table_hits=table_hits)
        if table_hits:
            if allow_table_contact_for_label(label):
                jlog("servo_allow_table_contact_before_execute", label=label, attempt=attempt, hits=table_hits)
            else:
                jlog("servo_abort_table_contact_before_execute", label=label, attempt=attempt, hits=table_hits)
                return False, {"attempts": attempt + 1, "status": "table_contact_before_execute", "max_qvel": maxqvel(), "max_cmd_err": 0.0}
        if start_upper_dq > 0.5:
            jlog("servo_attempt_start_motion_reseed", label=label, attempt=attempt, qvel=start_qvel, upper_max_dq=start_upper_dq)
            hold_measured(demo, f"{label}_start_reseed_{attempt}", duration=max(1.5, RESEED_HOLD_DURATION))
            if attempt + 1 < attempts:
                continue
            return False, {"attempts": attempt + 1, "status": "attempt_start_motion", "max_qvel": start_qvel, "max_cmd_err": 0.0}
        ok, info = servo_execute(
            demo,
            traj,
            free_joints,
            f"{label}_a{attempt}",
            dt=dt,
            track_tol=track_tol,
            max_cmd_step=max_cmd_step,
        )
        jlog("servo_attempt_done", label=label, attempt=attempt, ok=ok, info=info, qvel=maxqvel())
        last_info = info
        if ok:
            return True, {"attempts": attempt + 1, **info}
        if info.get("status") not in ("stalled", "precompute_motion", "unsafe"):
            return False, {"attempts": attempt + 1, **info}
        if info.get("status") == "unsafe":
            jlog("servo_attempt_unsafe_reseed", label=label, attempt=attempt, info=info, qvel=maxqvel(), contacts=contact_snapshot())
        hold_measured(demo, f"{label}_reseed_{attempt}", duration=RESEED_HOLD_DURATION)
    return False, {"attempts": attempts, **(last_info or {})}


def plan_init(demo, duration=20.0):
    left_start, right_start = demo._current_ee()
    target = configured_right_init_pose()
    jlog("plan_init_target", target=target)
    return demo.planner.plan_through_waypoints([left_start, left_start], [right_start, target], duration=duration)


def plan_demo_init(demo, duration=20.0):
    left_start, right_start = demo._current_ee()
    jlog("plan_demo_init_target", target=T_RIGHT_INIT)
    return demo.planner.plan_through_waypoints([left_start, left_start], [right_start, T_RIGHT_INIT], duration=duration)


def plan_right_to_brick_offset(demo, brick_pose, offset, duration=30.0):
    left_start, right_start = demo._current_ee()
    target = _brick_to_grasp_pose(brick_pose, np.array(offset, dtype=float))
    return demo.planner.plan_through_waypoints([left_start, left_start], [right_start, target], duration=duration)


def plan_right_to_pose(demo, target, duration=15.0):
    left_start, right_start = demo._current_ee()
    return demo.planner.plan_through_waypoints([left_start, left_start], [right_start, target], duration=duration)


def _make_segmented_traj(left_pose, right_waypoints, duration, wps=None):
    n = max(2, int(duration * (wps or PREPICK_ROUTE_WPS)))
    seg_count = len(right_waypoints) - 1
    per_seg = max(2, int(np.ceil(n / seg_count)))
    waypoints = []
    t = 0.0
    dt = duration / max(1, seg_count * (per_seg - 1))
    for si in range(seg_count):
        r0 = right_waypoints[si]
        r1 = right_waypoints[si + 1]
        for j in range(per_seg):
            if si > 0 and j == 0:
                continue
            a = j / (per_seg - 1)
            pos = (1.0 - a) * r0[:3, 3] + a * r1[:3, 3]
            rr = R.from_matrix([r0[:3, :3], r1[:3, :3]])
            slerp = Slerp([0.0, 1.0], rr)
            pose = np.eye(4)
            pose[:3, :3] = slerp([a]).as_matrix()[0]
            pose[:3, 3] = pos
            waypoints.append(CartesianWaypoint(left_pose.copy(), pose, min(t, duration)))
            t += dt
    if waypoints:
        waypoints[-1].timestamp = duration
    return CartesianTrajectory(waypoints)


def plan_right_to_brick_offset_segmented(demo, brick_pose, offset, duration=30.0):
    left_start, right_start = demo._current_ee()
    target = _brick_to_grasp_pose(brick_pose, np.array(offset, dtype=float))
    safe_z = max(float(right_start[2, 3]), float(target[2, 3])) + PREPICK_ROUTE_CLEARANCE
    up = right_start.copy()
    up[2, 3] = safe_z
    over_current_rot = right_start.copy()
    over_current_rot[:3, 3] = target[:3, 3]
    over_current_rot[2, 3] = safe_z
    over_target_rot = target.copy()
    over_target_rot[2, 3] = safe_z
    jlog("segmented_prepick_target", mode=PREPICK_ROUTE_MODE, start=right_start, target=target, safe_z=safe_z,
         clearance=PREPICK_ROUTE_CLEARANCE)
    return _make_segmented_traj(left_start, [right_start, up, over_current_rot, over_target_rot, target], duration)


def execute_staged_prepick(demo, brick_pose, offset, free_joints):
    target = _brick_to_grasp_pose(brick_pose, np.array(offset, dtype=float))
    _, right_start = demo._current_ee()
    safe_z = max(float(right_start[2, 3]), float(target[2, 3])) + PREPICK_ROUTE_CLEARANCE
    stages = []

    up = right_start.copy()
    up[2, 3] = safe_z
    stages.append(("prepick_stage_up", up, 12.0))

    over_current_rot = up.copy()
    over_current_rot[:3, 3] = target[:3, 3]
    over_current_rot[2, 3] = safe_z
    stages.append(("prepick_stage_over", over_current_rot, 18.0))

    over_target_rot = target.copy()
    over_target_rot[2, 3] = safe_z
    stages.append(("prepick_stage_rotate", over_target_rot, 14.0))

    stages.append(("prepick_stage_settle", target, 16.0))
    jlog("staged_prepick_target", target=target, safe_z=safe_z, clearance=PREPICK_ROUTE_CLEARANCE,
         attempts=PREPICK_STAGE_ATTEMPTS)

    last_info = {}
    for label, target_pose, duration in stages:
        ok, info = servo_execute_replanned(
            demo,
            lambda pose=target_pose, dur=duration: plan_right_to_pose(demo, pose, duration=dur),
            free_joints,
            label,
            attempts=PREPICK_STAGE_ATTEMPTS,
            dt=PICK_DT,
            track_tol=PICK_TRACK_TOL,
        )
        jlog("staged_prepick_stage_done", label=label, ok=ok, info=info, qvel=maxqvel(),
             upper_max_dq=upper_max_dq(demo.dds), contacts=contact_snapshot())
        last_info = {"stage": label, **info}
        if not ok:
            return False, last_info
        if not wait_settled(label + "_settled", limit=0.10, timeout=12):
            return False, {"stage": label, "status": "not_settled", "qvel": maxqvel(),
                           "upper_max_dq": upper_max_dq(demo.dds)}
    return True, {"status": "done", "stages": len(stages), **last_info}


def plan_right_lift_current(demo, lift_z, duration=20.0):
    left_start, right_start = demo._current_ee()
    target = right_start.copy()
    target[2, 3] += lift_z
    jlog("right_lift_target", lift_z=lift_z, start=right_start, target=target)
    return demo.planner.plan_through_waypoints([left_start, left_start], [right_start, target], duration=duration)


def plan_place_to_target(demo, T_pelvis_to_brick, duration=30.0):
    return demo.plan_brick_trajectory(T_pelvis_to_brick, duration=duration)


def mujoco_freejoint_pose(q):
    T = np.eye(4)
    q = np.array(q, dtype=float)
    T[:3, 3] = q[:3]
    T[:3, :3] = R.from_quat([q[4], q[5], q[6], q[3]]).as_matrix()
    return T


def selected_place_row():
    if PLACE_TARGET_ROW:
        vals = [float(x) for x in PLACE_TARGET_ROW.split(",")]
        if len(vals) != 6:
            raise ValueError(f"PLACE_TARGET_ROW must have 6 comma-separated values, got {PLACE_TARGET_ROW!r}")
        return np.array(vals, dtype=float)
    return CURVE_WALL[PLACE_TARGET_INDEX]


def main():
    jlog("start")
    wait_settled("initial")
    b0 = brick_states()
    jlog("brick_initial", bricks=b0)
    demo = None
    try:
        demo = PickPlace(network_interface="lo")
        if ALL_UPPER_KP is not None:
            kp = float(ALL_UPPER_KP)
            kd = float(ALL_UPPER_KD if ALL_UPPER_KD is not None else RIGHT_KD)
            demo.dds.upper_body_gains["kp"][:] = kp
            demo.dds.upper_body_gains["kd"][:] = kd
            jlog("all_upper_gains", kp=kp, kd=kd)
        demo.dds.upper_body_gains["kp"][8:15] = RIGHT_KP
        demo.dds.upper_body_gains["kd"][8:15] = RIGHT_KD
        jlog("right_arm_gains", kp=RIGHT_KP, kd=RIGHT_KD,
             pick_right_shoulder_pitch_kp=RIGHT_SHOULDER_PITCH_KP,
             pick_right_shoulder_pitch_kd=RIGHT_SHOULDER_PITCH_KD,
             pick_right_elbow_kp=RIGHT_ELBOW_KP,
             pick_right_elbow_kd=RIGHT_ELBOW_KD,
             max_cmd_step=MAX_CMD_STEP, track_tol=TRACK_TOL,
             pick_track_tol=PICK_TRACK_TOL,
             init_track_tol=INIT_TRACK_TOL,
             upper_dq_abort=UPPER_DQ_ABORT,
             upper_dq_recover=UPPER_DQ_RECOVER,
             servo_max_recoveries=SERVO_MAX_RECOVERIES,
             prepick_high_z=PREPICK_HIGH_Z,
             prepick_lift_z=PREPICK_LIFT_Z,
             prepick_lift_attempts=PREPICK_LIFT_ATTEMPTS,
             prepick_route_mode=PREPICK_ROUTE_MODE,
             prepick_route_clearance=PREPICK_ROUTE_CLEARANCE,
             nav_target_sd=NAV_TARGET_SD,
             nav_target_x=NAV_TARGET_X,
             prepick_attempts=PREPICK_ATTEMPTS,
             init_max_cmd_step=INIT_MAX_CMD_STEP,
             left_hand_mode=LEFT_HAND_MODE,
             joint_final_label_prefixes=JOINT_FINAL_LABEL_PREFIXES,
             preinit_recovery=PREINIT_RECOVERY,
             preinit_right_shoulder_roll=PREINIT_RIGHT_SHOULDER_ROLL,
             preinit_max_cmd_step=PREINIT_MAX_CMD_STEP,
             preinit_track_tol=PREINIT_TRACK_TOL,
             preinit_left_park=PREINIT_LEFT_PARK,
             preinit_left_park_q=PREINIT_LEFT_PARK_Q,
             initial_hand_settle_timeout=INITIAL_HAND_SETTLE_TIMEOUT,
             run_scaled_init=RUN_SCALED_INIT,
             scaled_init_duration=SCALED_INIT_DURATION,
             scaled_init_exec_scale=SCALED_INIT_EXEC_SCALE,
             post_grasp_lift_z=POST_GRASP_LIFT_Z,
             post_grasp_lift_duration=POST_GRASP_LIFT_DURATION,
             place_duration=PLACE_DURATION,
             place_return_duration=PLACE_RETURN_DURATION,
             place_target_index=PLACE_TARGET_INDEX,
             skip_pick_return_before_place=SKIP_PICK_RETURN_BEFORE_PLACE,
             skip_place_return_after_release=SKIP_PLACE_RETURN_AFTER_RELEASE,
             success_target_dist=SUCCESS_TARGET_DIST,
             place_require_pickable=PLACE_REQUIRE_PICKABLE)
        if hasattr(demo.camera, "stop"):
            demo.camera.stop()
        demo.camera = FileCamera()
        demo.estimator.camera = demo.camera
        if hasattr(demo, "localizer"):
            demo.localizer.intrinsics = demo.camera.intrinsics
        demo.planner.waypoints_per_second = PLANNER_WAYPOINTS_PER_SECOND
        jlog("planner_config", waypoints_per_second=demo.planner.waypoints_per_second)
        twist = G1TwistCmdNode("lo")

        if INITIAL_RIGHT_HAND_MODE != "none":
            if not set_initial_hands(demo):
                return 5
            time.sleep(0.5)
            if not wait_settled("after_initial_hand", limit=0.10, timeout=INITIAL_HAND_SETTLE_TIMEOUT):
                jlog("abort_initial_hand_not_settled", qvel=maxqvel(), upper_max_dq=upper_max_dq(demo.dds))
                return 5
        else:
            set_initial_hands(demo)
        if PREINIT_RECOVERY:
            if PREINIT_LEFT_PARK:
                if not preinit_left_park(demo):
                    jlog("abort_preinit_left_park", qvel=maxqvel(), upper_max_dq=upper_max_dq(demo.dds), contacts=contact_snapshot())
                    return 5
                time.sleep(0.5)
                if not wait_settled("after_preinit_left_park", limit=0.10, timeout=12):
                    jlog("abort_preinit_left_park_not_settled", qvel=maxqvel(), upper_max_dq=upper_max_dq(demo.dds), contacts=contact_snapshot())
                    return 5
            if not preinit_recovery(demo):
                jlog("abort_preinit_recovery", qvel=maxqvel(), upper_max_dq=upper_max_dq(demo.dds), contacts=contact_snapshot())
                return 5
            time.sleep(0.5)
            if not wait_settled("after_preinit_recovery", limit=0.10, timeout=12):
                jlog("abort_preinit_recovery_not_settled", qvel=maxqvel(), upper_max_dq=upper_max_dq(demo.dds), contacts=contact_snapshot())
                return 5
            if os.environ.get("STOP_AFTER_PREINIT", "0") == "1":
                jlog("stop_after_preinit", qvel=maxqvel(), upper_max_dq=upper_max_dq(demo.dds), bricks=brick_states(), contacts=contact_snapshot())
                return 0
        if RUN_SCALED_INIT:
            ok, info = scaled_init(demo)
            jlog("scaled_init_done", ok=ok, info=info, qvel=maxqvel(), contacts=contact_snapshot())
            if not ok:
                return 2
            set_hands(demo, "open_after_scaled_init", "open")
            time.sleep(1.0)
            if not wait_settled("after_scaled_init_open", limit=0.10, timeout=30):
                jlog("abort_scaled_init_open_not_settled", qvel=maxqvel(), contacts=contact_snapshot())
                return 5
            if os.environ.get("STOP_AFTER_INIT", "0") == "1":
                jlog("stop_after_scaled_init", qvel=maxqvel(), bricks=brick_states(), contacts=contact_snapshot())
                return 0
        elif os.environ.get("RUN_INIT", "0") == "1":
            ok, info = servo_execute_replanned(
                demo,
                lambda: plan_init(demo),
                demo.ik_solver.right_q_idx,
                "init",
                attempts=2,
                dt=0.10,
                track_tol=INIT_TRACK_TOL,
                max_cmd_step=INIT_MAX_CMD_STEP,
            )
            jlog("init_done", ok=ok, info=info, qvel=maxqvel())
            if not ok:
                return 2
            if os.environ.get("STOP_AFTER_INIT", "0") == "1":
                jlog("stop_after_init", qvel=maxqvel(), bricks=brick_states(), contacts=contact_snapshot())
                return 0
        else:
            hold_measured(demo, "init_skipped_hold", duration=2.0)
            jlog("init_skipped", reason="init servo can be unsafe; pick starts from measured rest pose")

        q_nav_hold, _ = demo.dds.get_upper_body_state()
        with HoldPublisher(demo.dds, q_nav_hold, demo.urdf_model, hz=NAV_HOLD_HZ):
            jlog("nav_hold_start", q=q_nav_hold, qvel=maxqvel(), upper_max_dq=upper_max_dq(demo.dds))
            pose, pickable, sd = estimate_best(demo, "after_init")
            for i in range(12):
                nav_ready = (
                    pose is not None
                    and (pickable or not NAV_REQUIRE_PICKABLE)
                    and sd <= NAV_TARGET_SD
                    and float(pose.position[0]) <= NAV_TARGET_X
                )
                jlog(
                    "nav_readiness",
                    step=i,
                    has_pose=pose is not None,
                    pickable=pickable,
                    sd=sd,
                    pose_x=None if pose is None else float(pose.position[0]),
                    ready=nav_ready,
                )
                if nav_ready:
                    break
                if pose is None or not np.isfinite(sd):
                    x, dur = NAV_NOPOSE_X, NAV_NOPOSE_DURATION
                elif sd > 0.11:
                    x, dur = 1.0, 2.4
                elif sd > 0.06:
                    x, dur = 0.45, 1.2
                elif sd > 0.025:
                    x, dur = 0.25, 0.8
                else:
                    x, dur = 0.12, 0.5
                bd, qv = pulse_and_check(twist, f"nav{i}", x, 0.0, dur, b0, demo.dds)
                qv, nav_settled = confirm_settled_after_nav(f"nav{i}", qv, dds=demo.dds)
                if bd > 0.015 or not nav_settled:
                    jlog("abort_nav", pulse=i, max_brick_disp=bd, qvel=qv)
                    return 3
                pose, pickable, sd = estimate_best(demo, f"after_nav{i}")
            jlog("nav_hold_done", qvel=maxqvel(), upper_max_dq=upper_max_dq(demo.dds))
        if pose is None or sd > NAV_TARGET_SD or float(pose.position[0]) > NAV_TARGET_X:
            jlog("abort_not_reachable", sd=sd, has_pose=pose is not None,
                 pose_x=None if pose is None else float(pose.position[0]),
                 nav_target_sd=NAV_TARGET_SD,
                 nav_target_x=NAV_TARGET_X)
            return 4

        if POST_NAV_DEMO_INIT:
            ok, info = servo_execute_replanned(
                demo,
                lambda: plan_demo_init(demo, duration=25.0),
                demo.ik_solver.right_q_idx,
                "post_nav_demo_init",
                attempts=2,
                dt=0.15,
                track_tol=INIT_TRACK_TOL,
                max_cmd_step=INIT_MAX_CMD_STEP,
            )
            jlog("post_nav_demo_init_done", ok=ok, info=info, qvel=maxqvel(), contacts=contact_snapshot())
            if not ok:
                return 5
            if not wait_settled("after_post_nav_demo_init", limit=0.10, timeout=20):
                jlog("abort_post_nav_demo_init_not_settled", qvel=maxqvel(), upper_max_dq=upper_max_dq(demo.dds), contacts=contact_snapshot())
                return 5

        pick_pose_pelvis = pose.transform.copy()
        jlog("pick_execute_start", pose=pose.position, transform=pick_pose_pelvis, sd=sd, qvel=maxqvel(), bricks=brick_states())
        if not apply_pick_gains(demo.dds):
            jlog("abort_pick_gain_settle", qvel=maxqvel(), upper_max_dq=upper_max_dq(demo.dds))
            return 5
        if PICK_FREE_MODE == "right_waist":
            pick_free = demo.ik_solver.right_q_idx + demo.ik_solver.waist_q_idx
        else:
            pick_free = demo.ik_solver.right_q_idx
        jlog("pick_free_mode", mode=PICK_FREE_MODE, free_joints=pick_free)
        if CLOSE_FOR_PREPICK:
            set_hands(demo, "close_for_prepick", "close")
            time.sleep(0.5)
            if not wait_settled("after_close_for_prepick", limit=0.10, timeout=12):
                jlog("abort_hand_not_settled", stage="prepick_high", qvel=maxqvel())
                return 5
        else:
            jlog("prepick_hand_kept_open")
        if PREPICK_LIFT_Z > 0:
            ok, info = servo_execute_replanned(
                demo,
                lambda: plan_right_lift_current(demo, PREPICK_LIFT_Z, duration=20.0),
                pick_free,
                "prepick_lift",
                attempts=PREPICK_LIFT_ATTEMPTS,
                dt=PICK_DT,
                track_tol=PICK_TRACK_TOL,
            )
            if not ok:
                jlog("pick_execute_done", stage="prepick_lift", ok=ok, info=info, qvel=maxqvel(), bricks=brick_states())
                return 5
        if PREPICK_ROUTE_MODE == "staged":
            ok, info = execute_staged_prepick(demo, pose.transform, [0.02, 0.0, PREPICK_HIGH_Z], pick_free)
        else:
            ok, info = servo_execute_replanned(
                demo,
                lambda: (
                    plan_right_to_brick_offset_segmented(demo, pose.transform, [0.02, 0.0, PREPICK_HIGH_Z], duration=35.0)
                    if PREPICK_ROUTE_MODE == "segmented"
                    else plan_right_to_brick_offset(demo, pose.transform, [0.02, 0.0, PREPICK_HIGH_Z], duration=35.0)
                ),
                pick_free,
                "prepick_high",
                attempts=PREPICK_ATTEMPTS,
                dt=PICK_DT,
                track_tol=PICK_TRACK_TOL,
            )
        if not ok:
            jlog("pick_execute_done", stage="prepick_high", ok=ok, info=info, qvel=maxqvel(), bricks=brick_states())
            return 5
        set_hands(demo, "open_for_descent", "open")
        time.sleep(0.5)
        if not wait_settled("after_open_for_descent", limit=0.10, timeout=12):
            jlog("abort_hand_not_settled", stage="pick_descend", qvel=maxqvel())
            return 5
        ok, info = servo_execute_replanned(
            demo,
            lambda: plan_right_to_brick_offset(demo, pose.transform, HAND_PICK_OFFSET, duration=25.0),
            pick_free,
            "pick_descend",
            attempts=3,
            dt=PICK_DT,
            track_tol=PICK_TRACK_TOL,
        )
        jlog("pick_execute_done", stage="pick_descend", ok=ok, info=info, qvel=maxqvel(), bricks=brick_states(), contacts=contact_snapshot())
        if not ok:
            return 5
        if STOP_AFTER_PICK_DESCEND:
            q_now, _ = demo.dds.get_upper_body_state()
            right_ee = demo.urdf_model.get_frame_transform(q_now, "right_palm_link", use_reduced=True)
            jlog("stop_after_pick_descend", right_ee=right_ee, target=_brick_to_grasp_pose(pose.transform, HAND_PICK_OFFSET),
                 brick_pose=pose.transform, bricks=brick_states(), contacts=contact_snapshot(), qvel=maxqvel())
            return 0
        set_hands(demo, "grasp", GRASP_HAND_MODE)
        time.sleep(1.0)
        jlog("grasped", bricks=brick_states(), qvel=maxqvel(), contacts=contact_snapshot())
        ok, info = servo_execute_replanned(
            demo,
            lambda: plan_right_lift_current(demo, POST_GRASP_LIFT_Z, duration=POST_GRASP_LIFT_DURATION),
            pick_free,
            "post_grasp_lift",
            attempts=2,
            dt=PICK_DT,
            track_tol=PICK_TRACK_TOL,
        )
        b_lift = brick_states()
        lift_disp = [float(np.linalg.norm(np.array(b_lift[i][:3]) - np.array(b0[i][:3]))) for i in range(5)]
        lift_z_delta = [float(np.array(b_lift[i][:3])[2] - np.array(b0[i][:3])[2]) for i in range(5)]
        carried_idx = int(np.argmax(lift_disp))
        carried_ok = bool(ok and lift_disp[carried_idx] >= MIN_CARRY_DISP and lift_z_delta[carried_idx] >= MIN_CARRY_Z_DELTA)
        world_from_pelvis_translation = np.array(b0[carried_idx][:3], dtype=float) - pick_pose_pelvis[:3, 3]
        T_world_from_pelvis = mujoco_freejoint_pose(b0[carried_idx]) @ np.linalg.inv(pick_pose_pelvis)
        jlog("post_grasp_lift_done", ok=ok, info=info, brick_disp=lift_disp, carried_idx=carried_idx,
             lift_z_delta=lift_z_delta, carried_ok=carried_ok,
             min_carry_disp=MIN_CARRY_DISP, min_carry_z_delta=MIN_CARRY_Z_DELTA,
             world_from_pelvis_translation=world_from_pelvis_translation, T_world_from_pelvis=T_world_from_pelvis,
             bricks=b_lift, qvel=maxqvel(), contacts=contact_snapshot())
        if not carried_ok:
            return 6
        if SKIP_PICK_RETURN_BEFORE_PLACE:
            b_after = b_lift
            disp = lift_disp
            jlog("pick_return_skipped", reason="place_directly_from_lifted_pose", brick_disp=disp, bricks=b_after, qvel=maxqvel(), contacts=contact_snapshot())
        else:
            ok, info = servo_execute_replanned(
                demo,
                lambda: plan_init(demo, duration=30.0),
                pick_free,
                "pick_return",
                attempts=4,
                dt=0.20,
            )
            b_after = brick_states()
            disp = [float(np.linalg.norm(np.array(b_after[i][:3]) - np.array(b0[i][:3]))) for i in range(5)]
            jlog("pick_return_done", ok=ok, info=info, brick_disp=disp, bricks=b_after, qvel=maxqvel())
            if not ok or max(disp) <= 0.03:
                return 6

        if not capture_external_frame("before_place_localize", timeout=20):
            jlog("place_localize_capture_failed", qvel=maxqvel(), contacts=contact_snapshot())
            return 7
        loc = demo.localize_table()
        if loc is None:
            jlog("place_localize_failed", qvel=maxqvel(), contacts=contact_snapshot())
            return 7
        T_pelvis_to_table, loc_result = loc
        place_row = selected_place_row()
        T_pelvis_to_place = T_pelvis_to_table @ _brick_to_table_pose(place_row)
        place_target_world_est = (T_world_from_pelvis @ T_pelvis_to_place)[:3, 3]
        place_pickable, place_sd = demo.is_pickable(T_pelvis_to_place)
        jlog("place_target", row=place_row, pose=T_pelvis_to_place, pickable=place_pickable, sd=place_sd,
             target_world_est=place_target_world_est, carried_idx=carried_idx,
             markers=loc_result.n_markers_used, reprojection_error=loc_result.reprojection_error)
        if not place_pickable and PLACE_REQUIRE_PICKABLE:
            return 7
        if not place_pickable:
            jlog("place_pickable_gate_ignored", sd=place_sd, reason="place target uses table localization, not brick pickability")
        ok, info = servo_execute_replanned(
            demo,
            lambda: plan_place_to_target(demo, T_pelvis_to_place, duration=PLACE_DURATION),
            pick_free,
            "place",
            attempts=2,
            dt=PICK_DT,
            track_tol=PICK_TRACK_TOL,
        )
        b_place = brick_states()
        place_disp = [float(np.linalg.norm(np.array(b_place[i][:3]) - np.array(b0[i][:3]))) for i in range(5)]
        jlog("place_execute_done", ok=ok, info=info, brick_disp=place_disp, bricks=b_place, qvel=maxqvel(), contacts=contact_snapshot())
        if not ok:
            return 8
        set_hands(demo, "release", "open")
        time.sleep(1.0)
        if not wait_settled("after_release", limit=0.12, timeout=15):
            jlog("release_not_settled", qvel=maxqvel(), contacts=contact_snapshot())
            return 8
        b_release = brick_states()
        release_disp = [float(np.linalg.norm(np.array(b_release[i][:3]) - np.array(b0[i][:3]))) for i in range(5)]
        target_dist_pelvis_bug = [
            float(np.linalg.norm(np.array(b_release[i][:3]) - T_pelvis_to_place[:3, 3]))
            for i in range(5)
        ]
        target_dist_world_est = [
            float(np.linalg.norm(np.array(b_release[i][:3]) - place_target_world_est))
            for i in range(5)
        ]
        placed_idx = carried_idx
        jlog("release_done", brick_disp=release_disp, target_dist=target_dist_world_est,
             target_dist_world_est=target_dist_world_est, target_dist_pelvis_bug=target_dist_pelvis_bug,
             target_world_est=place_target_world_est, placed_idx=placed_idx,
             bricks=b_release, qvel=maxqvel(), contacts=contact_snapshot())
        if SKIP_PLACE_RETURN_AFTER_RELEASE:
            ok_success = release_disp[placed_idx] >= MIN_CARRY_DISP and target_dist_world_est[placed_idx] <= SUCCESS_TARGET_DIST
            jlog("place_return_skipped", reason="success_checked_after_release", ok=ok_success,
                 placed_idx=placed_idx, placed_dist=target_dist_world_est[placed_idx], success_target_dist=SUCCESS_TARGET_DIST,
                 qvel=maxqvel(), contacts=contact_snapshot(), bricks=brick_states())
            return 0 if ok_success else 9
        ok, info = servo_execute_replanned(
            demo,
            lambda: plan_init(demo, duration=PLACE_RETURN_DURATION),
            pick_free,
            "place_return",
            attempts=2,
            dt=PICK_DT,
            track_tol=PICK_TRACK_TOL,
        )
        jlog("place_return_done", ok=ok, info=info, qvel=maxqvel(), contacts=contact_snapshot(), bricks=brick_states())
        return 0 if ok and max(release_disp) > 0.03 else 9
    except BaseException as e:
        jlog("base_exception", err=repr(e), type=type(e).__name__, tb=traceback.format_exc())
        return 99
    finally:
        if demo is not None:
            try:
                demo.shutdown()
            except Exception as e:
                jlog("shutdown_error", err=repr(e))
        jlog("end")


if __name__ == "__main__":
    raise SystemExit(main())
