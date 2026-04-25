"""Live MuJoCo controller for the Panda+cube scene.

Runs the viewer plus a small HTTP control server on 127.0.0.1:8765.
Submit commands as JSON; observations are returned as JSON.

Launch:
    uv run --with mujoco python controller.py

Endpoints:
    GET  /info, /state, /health
    POST /move_ee, /move_ee_delta, /move_ee_path,
         /set_joints, /nudge, /gripper, /home, /reset, /wait,
         /cancel, /await
"""

from __future__ import annotations

import json
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Optional

import mujoco
import mujoco.viewer
import numpy as np


# ============================================================ Config

HOST = "127.0.0.1"
PORT = 8765
ROOT = Path(__file__).resolve().parent
SCENE = ROOT / "models" / "panda_cube" / "scene.xml"

# Pinch point in the hand body's local frame: midpoint between fingertip pads.
# Hand frame z-axis points out of the gripper toward the fingers; ~10.3 cm down
# from hand origin sits between the closed pads.
PINCH_LOCAL = np.array([0.0, 0.0, 0.1034])

DEFAULT_POS_TOL = 5e-3                 # 5 mm
DEFAULT_ROT_TOL = float(np.deg2rad(2)) # 2 deg
DEFAULT_LIN_VEL = 0.25                 # m/s
DEFAULT_ANG_VEL = 1.0                  # rad/s
DEFAULT_JOINT_VEL = 1.5                # rad/s

ACTION_GRACE_S = 1.5                   # extra settle time after planned duration
HTTP_BLOCK_TIMEOUT_S = 120.0
GRASP_FORCE_THRESHOLD = 0.5            # N, per-finger min for ee_grasping=true


# ============================================================ Quat / SO(3)

def quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    out = np.zeros(4)
    mujoco.mju_mulQuat(out, q1, q2)
    return out


def quat_conj(q: np.ndarray) -> np.ndarray:
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_to_mat(q: np.ndarray) -> np.ndarray:
    out = np.zeros(9)
    mujoco.mju_quat2Mat(out, q)
    return out.reshape(3, 3)


def mat_to_quat(m: np.ndarray) -> np.ndarray:
    out = np.zeros(4)
    mujoco.mju_mat2Quat(out, np.ascontiguousarray(m).flatten())
    return out


def quat_normalize(q: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(q)
    return q / n if n > 0 else np.array([1.0, 0.0, 0.0, 0.0])


def quat_slerp(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    if dot > 0.9995:
        return quat_normalize(q0 + t * (q1 - q0))
    theta_0 = np.arccos(np.clip(dot, -1.0, 1.0))
    theta = theta_0 * t
    sin_t0 = np.sin(theta_0)
    s0 = np.cos(theta) - dot * np.sin(theta) / sin_t0
    s1 = np.sin(theta) / sin_t0
    return s0 * q0 + s1 * q1


def axis_angle_from_mat(R: np.ndarray) -> np.ndarray:
    """Convert a 3x3 rotation matrix to an axis*angle 3-vector."""
    q = mat_to_quat(R)
    w = float(q[0])
    v = q[1:]
    s = float(np.linalg.norm(v))
    if s < 1e-10:
        return np.zeros(3)
    angle = 2.0 * np.arctan2(s, w)
    if angle > np.pi:
        angle -= 2.0 * np.pi
    return (v / s) * angle


def axis_angle_to_quat(aa: np.ndarray) -> np.ndarray:
    angle = float(np.linalg.norm(aa))
    if angle < 1e-10:
        return np.array([1.0, 0.0, 0.0, 0.0])
    axis = aa / angle
    half = angle * 0.5
    return np.array([np.cos(half), *(axis * np.sin(half))])


# ============================================================ Action

@dataclass
class Action:
    kind: str
    payload: dict
    done: threading.Event = field(default_factory=threading.Event)
    result: dict = field(default_factory=dict)
    started: bool = False
    start_time_sim: float = 0.0


# ============================================================ Controller

class Controller:
    def __init__(self) -> None:
        self.model = mujoco.MjModel.from_xml_path(str(SCENE))
        self.data = mujoco.MjData(self.model)
        self.ik_data = mujoco.MjData(self.model)

        self.home_key = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "home")
        if self.home_key < 0:
            raise RuntimeError("home keyframe not found")
        self._reset_to_home()

        def nid(otype, name):
            i = mujoco.mj_name2id(self.model, otype, name)
            if i < 0:
                raise RuntimeError(f"missing {name}")
            return i

        self.hand_body_id = nid(mujoco.mjtObj.mjOBJ_BODY, "hand")
        self.left_finger_body_id = nid(mujoco.mjtObj.mjOBJ_BODY, "left_finger")
        self.right_finger_body_id = nid(mujoco.mjtObj.mjOBJ_BODY, "right_finger")

        # Discover every free-jointed body (i.e. movable object) in the scene.
        # name -> {body_id, qpos_adr, qvel_adr}
        self.blocks: dict[str, dict] = {}
        for bid in range(self.model.nbody):
            if int(self.model.body_jntnum[bid]) != 1:
                continue
            jid = int(self.model.body_jntadr[bid])
            if int(self.model.jnt_type[jid]) != int(mujoco.mjtJoint.mjJNT_FREE):
                continue
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, bid)
            if not name:
                continue
            self.blocks[name] = {
                "body_id": bid,
                "qpos_adr": int(self.model.jnt_qposadr[jid]),
                "qvel_adr": int(self.model.jnt_dofadr[jid]),
            }
        self._block_body_ids: dict[int, str] = {info["body_id"]: name for name, info in self.blocks.items()}

        # Arm joint limits: prefer actuator ctrlrange; fall back to jnt_range.
        self.joint_low = np.zeros(7)
        self.joint_high = np.zeros(7)
        for i in range(7):
            lo, hi = self.model.actuator_ctrlrange[i]
            if hi <= lo:
                jid = nid(mujoco.mjtObj.mjOBJ_JOINT, f"joint{i+1}")
                if self.model.jnt_limited[jid]:
                    lo, hi = self.model.jnt_range[jid]
                else:
                    lo, hi = -2.8973, 2.8973
            self.joint_low[i] = lo
            self.joint_high[i] = hi

        # Default EE quat = orientation at the home pose (gripper pointing down).
        _, ee_quat_home, _ = self._ee_pose_from(self.data)
        self.default_ee_quat = ee_quat_home.copy()

        self.queue: deque[Action] = deque()
        # Re-entrant: the sim loop wraps step_action+mj_step under this lock,
        # and step_action acquires it again internally for queue management.
        self.lock = threading.RLock()
        self.cancel_flag = False
        self.current_action: Optional[Action] = None

    # -------- world reset

    def _reset_to_home(self) -> None:
        # mj_resetData puts every qpos at qpos0 (so the cube freejoint sits at
        # its compiled pose on the table). We then overlay only the arm + finger
        # slots from the home keyframe — the keyframe has 9 qpos entries and
        # would zero-fill the cube freejoint otherwise.
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:9] = self.model.key_qpos[self.home_key, :9]
        self.data.ctrl[:] = self.model.key_ctrl[self.home_key]
        mujoco.mj_forward(self.model, self.data)

    # -------- kinematics

    def _ee_pose_from(self, d) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        hand_pos = d.xpos[self.hand_body_id].copy()
        hand_mat = d.xmat[self.hand_body_id].reshape(3, 3).copy()
        pinch_pos = hand_pos + hand_mat @ PINCH_LOCAL
        pinch_quat = mat_to_quat(hand_mat)
        return pinch_pos, pinch_quat, hand_mat

    def ee_pose(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self._ee_pose_from(self.data)

    def ik_solve(
        self,
        target_pos: np.ndarray,
        target_quat: np.ndarray,
        q_init: np.ndarray,
        max_iters: int = 200,
        lam: float = 0.05,
        pos_tol: float = 1e-4,
        rot_tol: float = float(np.deg2rad(0.2)),
    ) -> tuple[np.ndarray, float, float]:
        self.ik_data.qpos[:] = self.data.qpos
        self.ik_data.qvel[:] = 0.0
        self.ik_data.qpos[:7] = q_init

        target_mat = quat_to_mat(quat_normalize(target_quat))
        q = q_init.copy()
        pos_err_norm = float("inf")
        rot_err_norm = float("inf")

        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))

        for _ in range(max_iters):
            self.ik_data.qpos[:7] = q
            mujoco.mj_kinematics(self.model, self.ik_data)
            mujoco.mj_comPos(self.model, self.ik_data)

            hand_pos = self.ik_data.xpos[self.hand_body_id]
            hand_mat = self.ik_data.xmat[self.hand_body_id].reshape(3, 3)
            pinch_pos = hand_pos + hand_mat @ PINCH_LOCAL

            pos_err = target_pos - pinch_pos
            pos_err_norm = float(np.linalg.norm(pos_err))

            rot_err = axis_angle_from_mat(target_mat @ hand_mat.T)
            rot_err_norm = float(np.linalg.norm(rot_err))

            if pos_err_norm < pos_tol and rot_err_norm < rot_tol:
                break

            err6 = np.concatenate([pos_err, rot_err])
            mujoco.mj_jac(self.model, self.ik_data, jacp, jacr, pinch_pos, self.hand_body_id)
            J = np.vstack([jacp[:, :7], jacr[:, :7]])
            H = J.T @ J + (lam ** 2) * np.eye(7)
            dq = np.linalg.solve(H, J.T @ err6)

            sn = float(np.linalg.norm(dq))
            if sn > 0.3:
                dq *= 0.3 / sn
            q = np.clip(q + dq, self.joint_low, self.joint_high)

        return q, pos_err_norm, rot_err_norm

    # -------- step loop

    def step_action(self) -> None:
        with self.lock:
            if self.cancel_flag:
                while self.queue:
                    a = self.queue.popleft()
                    a.result = {"ok": False, "error": "cancelled"}
                    a.done.set()
                if self.current_action is not None:
                    self.current_action.result = {"ok": False, "error": "cancelled"}
                    self.current_action.done.set()
                    self.current_action = None
                self.cancel_flag = False

            if self.current_action is None and self.queue:
                self.current_action = self.queue.popleft()

            action = self.current_action

        if action is None:
            return

        if not action.started:
            action.started = True
            action.start_time_sim = float(self.data.time)
            try:
                self._init_action(action)
            except Exception as e:
                action.result = {"ok": False, "error": f"init: {type(e).__name__}: {e}"}
                action.done.set()
                with self.lock:
                    self.current_action = None
                return

        try:
            done = self._advance_action(action)
        except Exception as e:
            action.result = {"ok": False, "error": f"step: {type(e).__name__}: {e}"}
            done = True

        if done:
            if not action.result:
                action.result = {"ok": True, "final_state": self.get_state()}
            action.done.set()
            with self.lock:
                self.current_action = None

    # -------- action init / step

    def _init_action(self, a: Action) -> None:
        if a.kind in ("move_ee", "move_ee_delta"):
            cur_pos, cur_quat, _ = self.ee_pose()
            if a.kind == "move_ee":
                target_pos = np.array(a.payload["pos"], dtype=float)
                if a.payload.get("quat") is not None:
                    target_quat = np.array(a.payload["quat"], dtype=float)
                else:
                    target_quat = self.default_ee_quat.copy()
            else:
                dpos = np.array(a.payload.get("dpos", [0, 0, 0]), dtype=float)
                drot = a.payload.get("drot")
                if drot is None:
                    drot_q = np.array([1.0, 0.0, 0.0, 0.0])
                else:
                    drot_arr = np.array(drot, dtype=float)
                    drot_q = axis_angle_to_quat(drot_arr) if drot_arr.shape == (3,) else drot_arr
                frame = a.payload.get("frame", "world")
                if frame == "ee":
                    cur_mat = quat_to_mat(cur_quat)
                    target_pos = cur_pos + cur_mat @ dpos
                    target_quat = quat_mul(cur_quat, drot_q)
                else:
                    target_pos = cur_pos + dpos
                    target_quat = quat_mul(drot_q, cur_quat)

            target_quat = quat_normalize(target_quat)
            if float(np.dot(target_quat, cur_quat)) < 0:
                target_quat = -target_quat

            max_lin = float(a.payload.get("max_lin_vel", DEFAULT_LIN_VEL))
            max_ang = float(a.payload.get("max_ang_vel", DEFAULT_ANG_VEL))
            dist = float(np.linalg.norm(target_pos - cur_pos))
            ang = 2.0 * float(np.arccos(np.clip(abs(np.dot(target_quat, cur_quat)), -1.0, 1.0)))
            duration = max(dist / max_lin, ang / max_ang, 0.5)

            a.payload["_start_pos"] = cur_pos
            a.payload["_start_quat"] = cur_quat
            a.payload["_target_pos"] = target_pos
            a.payload["_target_quat"] = target_quat
            a.payload["_duration"] = duration
            a.payload["_q_warm"] = self.data.qpos[:7].copy()

        elif a.kind == "set_joints":
            target_q = np.clip(np.array(a.payload["q"], dtype=float),
                               self.joint_low, self.joint_high)
            start_q = self.data.ctrl[:7].copy()
            max_vel = float(a.payload.get("max_vel", DEFAULT_JOINT_VEL))
            duration = max(float(np.max(np.abs(target_q - start_q))) / max_vel, 0.3)
            a.payload["_target_q"] = target_q
            a.payload["_start_q"] = start_q
            a.payload["_duration"] = duration

        elif a.kind == "nudge":
            j = int(a.payload["joint"])
            delta = float(a.payload["delta"])
            start_q = self.data.ctrl[:7].copy()
            target_q = start_q.copy()
            target_q[j] = float(np.clip(target_q[j] + delta, self.joint_low[j], self.joint_high[j]))
            max_vel = float(a.payload.get("max_vel", DEFAULT_JOINT_VEL))
            duration = max(abs(delta) / max_vel, 0.2)
            a.payload["_target_q"] = target_q
            a.payload["_start_q"] = start_q
            a.payload["_duration"] = duration

        elif a.kind == "home":
            target_q = np.array(self.model.key_ctrl[self.home_key, :7], dtype=float)
            start_q = self.data.ctrl[:7].copy()
            max_vel = float(a.payload.get("max_vel", DEFAULT_JOINT_VEL))
            duration = max(float(np.max(np.abs(target_q - start_q))) / max_vel, 0.5)
            a.payload["_target_q"] = target_q
            a.payload["_start_q"] = start_q
            a.payload["_duration"] = duration
            a.payload["_gripper_target"] = float(self.model.key_ctrl[self.home_key, 7])

        elif a.kind == "gripper":
            a.payload["_target_ctrl"] = float(self._gripper_to_ctrl(a.payload))

        elif a.kind == "reset":
            self._reset_to_home()

        elif a.kind == "wait":
            a.payload["_duration"] = float(a.payload.get("duration", 0.5))

        else:
            raise ValueError(f"unknown action kind: {a.kind}")

    def _gripper_to_ctrl(self, p: dict) -> float:
        if "ctrl" in p:
            return float(np.clip(float(p["ctrl"]), 0.0, 255.0))
        if "width" in p:
            w = float(np.clip(float(p["width"]), 0.0, 0.08))
            return w / 0.08 * 255.0
        return 255.0 if p.get("action", "open") == "open" else 0.0

    def _advance_action(self, a: Action) -> bool:
        elapsed = float(self.data.time) - a.start_time_sim

        if a.kind in ("move_ee", "move_ee_delta"):
            duration = a.payload["_duration"]
            t = min(elapsed / duration, 1.0)
            s = 0.5 * (1.0 - np.cos(np.pi * t))

            target_pos = a.payload["_start_pos"] + s * (a.payload["_target_pos"] - a.payload["_start_pos"])
            target_quat = quat_slerp(a.payload["_start_quat"], a.payload["_target_quat"], s)

            q_warm = a.payload["_q_warm"]
            q_sol, _, _ = self.ik_solve(target_pos, target_quat, q_warm, max_iters=60)
            a.payload["_q_warm"] = q_sol
            self.data.ctrl[:7] = q_sol

            if t >= 1.0 and elapsed > duration + ACTION_GRACE_S:
                cur_pos, cur_quat, _ = self.ee_pose()
                pos_e = float(np.linalg.norm(a.payload["_target_pos"] - cur_pos))
                rot_e = float(np.linalg.norm(
                    axis_angle_from_mat(quat_to_mat(a.payload["_target_quat"]) @ quat_to_mat(cur_quat).T)
                ))
                pos_tol = float(a.payload.get("pos_tol", DEFAULT_POS_TOL))
                rot_tol = float(a.payload.get("rot_tol", DEFAULT_ROT_TOL))
                ok = (pos_e <= pos_tol) and (rot_e <= rot_tol)
                a.result = {
                    "ok": ok,
                    "error": None if ok else "tolerance not met",
                    "pos_err_m": pos_e,
                    "rot_err_rad": rot_e,
                    "took_s": elapsed,
                    "final_state": self.get_state(),
                }
                return True
            return False

        if a.kind in ("set_joints", "nudge", "home"):
            duration = a.payload["_duration"]
            t = min(elapsed / duration, 1.0)
            s = 0.5 * (1.0 - np.cos(np.pi * t))
            target_q = a.payload["_target_q"]
            start_q = a.payload["_start_q"]
            self.data.ctrl[:7] = start_q + s * (target_q - start_q)
            if a.kind == "home":
                self.data.ctrl[7] = a.payload["_gripper_target"]
            if t >= 1.0 and elapsed > duration + 0.5:
                err = float(np.max(np.abs(self.data.qpos[:7] - target_q)))
                a.result = {
                    "ok": err < 0.05,
                    "joint_err_max_rad": err,
                    "took_s": elapsed,
                    "final_state": self.get_state(),
                }
                return True
            return False

        if a.kind == "gripper":
            self.data.ctrl[7] = a.payload["_target_ctrl"]
            a.result = {"ok": True, "final_state": self.get_state()}
            return True

        if a.kind == "reset":
            a.result = {"ok": True, "final_state": self.get_state()}
            return True

        if a.kind == "wait":
            if elapsed >= a.payload["_duration"]:
                a.result = {"ok": True, "took_s": elapsed}
                return True
            return False

        a.result = {"ok": False, "error": f"unknown kind: {a.kind}"}
        return True

    # -------- observation

    def get_state(self) -> dict:
        # All MuJoCo reads must happen under the sim lock — otherwise the sim
        # loop can mutate self.data mid-call and the dict construction throws.
        with self.lock:
            return self._get_state_locked()

    def _get_state_locked(self) -> dict:
        d, m = self.data, self.model

        ee_pos, ee_quat, hand_mat = self.ee_pose()

        hand_vel = np.zeros(6)
        mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, self.hand_body_id, hand_vel, 0)
        ee_angvel = hand_vel[:3].copy()
        hand_origin_linvel = hand_vel[3:].copy()
        # transport linear velocity from hand origin to pinch point
        ee_linvel = hand_origin_linvel + np.cross(ee_angvel, hand_mat @ PINCH_LOCAL)

        # Per-block pose, twist, in-EE-frame pose.
        blocks: dict[str, dict] = {}
        block_vel = np.zeros(6)
        for name, info in self.blocks.items():
            bid = info["body_id"]
            pos = d.xpos[bid].copy()
            quat = d.xquat[bid].copy()
            mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, bid, block_vel, 0)
            angvel = block_vel[:3].copy()
            linvel = block_vel[3:].copy()
            blocks[name] = {
                "pos": list(map(float, pos)),
                "quat": list(map(float, quat)),
                "linvel": list(map(float, linvel)),
                "angvel": list(map(float, angvel)),
                "in_ee_frame": {
                    "pos": list(map(float, hand_mat.T @ (pos - ee_pos))),
                    "quat": list(map(float, quat_mul(quat_conj(ee_quat), quat))),
                },
                "contacts": [],
            }

        # Contacts involving any block, plus per-block per-finger force tally.
        left_force = {name: 0.0 for name in self.blocks}
        right_force = {name: 0.0 for name in self.blocks}
        f6 = np.zeros(6)
        for i in range(d.ncon):
            c = d.contact[i]
            b1 = int(m.geom_bodyid[c.geom1])
            b2 = int(m.geom_bodyid[c.geom2])
            if b1 in self._block_body_ids:
                block_name = self._block_body_ids[b1]
                other = b2
            elif b2 in self._block_body_ids:
                block_name = self._block_body_ids[b2]
                other = b1
            else:
                continue
            other_name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, other) or f"body_{other}"
            mujoco.mj_contactForce(m, d, i, f6)
            nf = float(abs(f6[0]))
            blocks[block_name]["contacts"].append({
                "other_body": other_name,
                "normal_force_n": nf,
                "normal_world": list(map(float, c.frame[:3])),
                "pos_world": list(map(float, c.pos)),
            })
            if other == self.left_finger_body_id:
                left_force[block_name] += nf
            elif other == self.right_finger_body_id:
                right_force[block_name] += nf

        ee_grasping = False
        ee_grasping_block: Optional[str] = None
        for name in self.blocks:
            if (left_force[name] > GRASP_FORCE_THRESHOLD and
                    right_force[name] > GRASP_FORCE_THRESHOLD):
                ee_grasping = True
                ee_grasping_block = name
                break

        gripper_width = float(d.qpos[7] + d.qpos[8])

        with self.lock:
            queue_depth = len(self.queue) + (1 if self.current_action is not None else 0)
            current_kind = self.current_action.kind if self.current_action else None

        return {
            "time": float(d.time),
            "nstep": int(round(d.time / m.opt.timestep)),
            "dt": float(m.opt.timestep),
            "q": list(map(float, d.qpos[:7])),
            "qd": list(map(float, d.qvel[:7])),
            "gripper_width_m": gripper_width,
            "ctrl": list(map(float, d.ctrl)),
            "actuator_force": list(map(float, d.actuator_force[:7])),
            "ee_pos": list(map(float, ee_pos)),
            "ee_quat": list(map(float, ee_quat)),
            "ee_linvel": list(map(float, ee_linvel)),
            "ee_angvel": list(map(float, ee_angvel)),
            "blocks": blocks,
            "ee_grasping": bool(ee_grasping),
            "ee_grasping_block": ee_grasping_block,
            "queue_depth": queue_depth,
            "current_action": current_kind,
        }

    def get_info(self) -> dict:
        m = self.model
        return {
            "block_names": list(self.blocks.keys()),
            "joint_names": [f"joint{i+1}" for i in range(7)],
            "joint_limits_rad": [[float(self.joint_low[i]), float(self.joint_high[i])] for i in range(7)],
            "actuator_ctrl_ranges": m.actuator_ctrlrange.tolist(),
            "n_arm_joints": 7,
            "ee_frame": "pinch — midpoint between finger pads, +0.1034 m along hand-local z",
            "default_ee_quat": list(map(float, self.default_ee_quat)),
            "gripper_ctrl_range": [0.0, 255.0],
            "gripper_width_range_m": [0.0, 0.08],
            "home_qpos": list(map(float, m.key_qpos[self.home_key, :9])),
            "home_ctrl": list(map(float, m.key_ctrl[self.home_key])),
            "units": {"distance": "m", "angle": "rad", "time": "s"},
            "quat_order": "[w, x, y, z]",
            "scene_path": str(SCENE),
        }

    # -------- queue control

    def submit(self, kind: str, payload: dict, blocking: bool, timeout: float) -> dict:
        a = Action(kind=kind, payload=dict(payload))
        with self.lock:
            self.queue.append(a)
        if not blocking:
            return {"ok": True, "queued": True}
        if a.done.wait(timeout=timeout):
            return a.result
        return {"ok": False, "error": "timeout waiting for action"}

    def cancel(self) -> dict:
        with self.lock:
            self.cancel_flag = True
        return {"ok": True}

    def await_drain(self, timeout: float) -> dict:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self.lock:
                if not self.queue and self.current_action is None:
                    return {"ok": True}
            time.sleep(0.01)
        return {"ok": False, "error": "timeout"}


# ============================================================ HTTP

def make_handler(controller: Controller):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            return  # quiet

        def _send_json(self, code: int, body: Any) -> None:
            data = json.dumps(body).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _read_json(self) -> dict:
            length = int(self.headers.get("Content-Length", "0"))
            if length == 0:
                return {}
            return json.loads(self.rfile.read(length))

        def do_GET(self):
            try:
                if self.path == "/state":
                    self._send_json(200, controller.get_state())
                elif self.path == "/info":
                    self._send_json(200, controller.get_info())
                elif self.path == "/health":
                    self._send_json(200, {"ok": True})
                else:
                    self._send_json(404, {"ok": False, "error": "not found"})
            except Exception as e:
                self._send_json(500, {"ok": False, "error": f"{type(e).__name__}: {e}"})

        def do_POST(self):
            try:
                body = self._read_json()
            except Exception as e:
                self._send_json(400, {"ok": False, "error": f"bad json: {e}"})
                return
            blocking = bool(body.pop("blocking", True))
            timeout = float(body.pop("timeout", HTTP_BLOCK_TIMEOUT_S))

            try:
                if self.path == "/move_ee_path":
                    poses = body.pop("poses", [])
                    actions = []
                    with controller.lock:
                        for pose in poses:
                            payload = dict(body)
                            payload.update(pose)
                            a = Action(kind="move_ee", payload=payload)
                            controller.queue.append(a)
                            actions.append(a)
                    if not blocking:
                        result = {"ok": True, "queued": len(actions)}
                    else:
                        results = []
                        deadline = time.monotonic() + timeout
                        for a in actions:
                            remaining = max(0.05, deadline - time.monotonic())
                            if a.done.wait(remaining):
                                results.append(a.result)
                            else:
                                results.append({"ok": False, "error": "timeout"})
                                break
                        result = {"ok": all(r.get("ok") for r in results), "results": results}
                elif self.path in ("/move_ee", "/move_ee_delta", "/set_joints",
                                   "/nudge", "/gripper", "/home", "/reset", "/wait"):
                    kind = self.path.lstrip("/")
                    result = controller.submit(kind, body, blocking, timeout)
                elif self.path == "/cancel":
                    result = controller.cancel()
                elif self.path == "/await":
                    result = controller.await_drain(timeout)
                else:
                    self._send_json(404, {"ok": False, "error": "not found"})
                    return
                self._send_json(200, result)
            except Exception as e:
                self._send_json(500, {"ok": False, "error": f"{type(e).__name__}: {e}"})

    return Handler


def start_server(controller: Controller) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer((HOST, PORT), make_handler(controller))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print(f"control server: http://{HOST}:{PORT}", flush=True)
    return server


# ============================================================ Main

def main() -> None:
    controller = Controller()
    start_server(controller)

    last = time.time()
    with mujoco.viewer.launch_passive(controller.model, controller.data) as viewer:
        while viewer.is_running():
            with controller.lock:
                controller.step_action()
                mujoco.mj_step(controller.model, controller.data)
            viewer.sync()
            now = time.time()
            sleep_for = controller.model.opt.timestep - (now - last)
            if sleep_for > 0:
                time.sleep(sleep_for)
            last = time.time()


if __name__ == "__main__":
    main()
