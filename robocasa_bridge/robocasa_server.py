"""Robocasa-backed Clawblox bridge.

Mirrors ``server.py`` (FastAPI /observe + /input) but the underlying sim is
a robocasa kitchen task running through robosuite. The realtime loop calls
``env.step(current_action)`` at ``control_freq``; SetControl updates that
action vector; Reset calls ``env.reset()``.

Beyond the Panda bridge, this server exposes the **full** robosuite
observation dict (proprio, eef pose, per-object pose, eef-relative
distances, ...), the latest reward / done / success signal, and the
composite-controller action layout. Static info that doesn't change tick
to tick (env/robot name, action bounds, observation key shapes, fixture
names) lives at GET /spec.

Run:

    uv run python robocasa_bridge/robocasa_server.py
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
import io
import os
import threading
import time
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse, Response
from pydantic import BaseModel, Field
import mujoco
import numpy as np
import uvicorn

try:
    from PIL import Image
except ImportError:  # /render will 503 if Pillow is missing
    Image = None  # type: ignore[assignment]

from robocasa_setup import (
    DEFAULT_CONTROL_FREQ,
    DEFAULT_ENV,
    DEFAULT_ROBOT,
    RobocasaEnv,
    make_env,
)


ROOT = Path(__file__).resolve().parent
API_DOC = ROOT.parent / "API.md"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8080

# Snapshot covers everything that mj_step touches plus user data — i.e.
# enough to deterministically resume the sim from a saved slot.
STATE_SPEC = mujoco.mjtState.mjSTATE_INTEGRATION

# Panda max reach (link sum, excluding the EE flange) — used by /reach_check
# as a fast yes/no for "can the arm touch this from where it's parked?".
PANDA_REACH_RADIUS = 0.855

RENDER_DEFAULT_W = 320
RENDER_DEFAULT_H = 240
RENDER_MAX_DIM = 1024


class InputAction(BaseModel):
    type: str
    data: dict[str, Any] = Field(default_factory=dict)


def _to_jsonable(value: Any) -> Any:
    """Convert numpy / robosuite obs values into JSON-safe Python types."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, (bool, int, float, str)) or value is None:
        return value
    # Last-ditch: fall back to str so we never blow up the response.
    return str(value)


def _serialize_obs(obs: dict[str, Any] | None) -> dict[str, Any]:
    if not obs:
        return {}
    return {str(k): _to_jsonable(v) for k, v in obs.items()}


def _obs_key_shapes(obs: dict[str, Any] | None) -> dict[str, list[int]]:
    """Shapes of every key in the obs dict — for the static /spec output."""
    if not obs:
        return {}
    out: dict[str, list[int]] = {}
    for k, v in obs.items():
        if isinstance(v, np.ndarray):
            out[str(k)] = list(v.shape)
        elif isinstance(v, (list, tuple)):
            out[str(k)] = [len(v)]
        else:
            out[str(k)] = []
    return out


class RobocasaSim:
    """Threaded wrapper that drives a robocasa env at its control_freq.

    The realtime thread calls ``env.step(current_action)`` once per control
    period, captures the returned (obs, reward, done, info), and stores it
    so /observe can return it without re-stepping.
    """

    def __init__(self, env: RobocasaEnv) -> None:
        self.env = env
        self.model = env.model
        self.data = env.data
        self.action_dim = env.action_dim
        self.control_freq = env.control_freq
        self.action_low = env.action_low
        self.action_high = env.action_high
        self.action_layout = env.action_layout
        self.env_name = env.env_name
        self.robot_name = env.robot_name

        self.lock = threading.RLock()
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None

        self.actuator_names = self._names(mujoco.mjtObj.mjOBJ_ACTUATOR, self.model.nu)
        self.joint_names = self._names(mujoco.mjtObj.mjOBJ_JOINT, self.model.njnt)
        self.body_names = self._names(mujoco.mjtObj.mjOBJ_BODY, self.model.nbody)
        self.body_id_by_name = {n: i for i, n in enumerate(self.body_names)}
        self.camera_names = self._names(mujoco.mjtObj.mjOBJ_CAMERA, self.model.ncam)
        self.object_body_ids = self._discover_object_bodies()
        self.arm_qpos_addrs, self.arm_qvel_addrs, self.arm_joint_names = (
            self._discover_arm_addrs()
        )

        self.current_action = np.zeros(self.action_dim, dtype=float)
        self.last_obs: dict[str, Any] = env.initial_obs
        self.last_reward: float = 0.0
        self.last_done: bool = False
        self.last_info: dict[str, Any] = {}
        self.last_success: bool = self._check_success_safe()
        self.step_count: int = 0

        # /scene is heavy — clients should cache and only refetch when
        # scene_version changes (after Reset, RestoreState, etc).
        self.scene_version: int = 1

        # Episode metrics: useful for "I've been driving for ages and
        # gotten nowhere" heuristics in higher-level planners.
        self.episode_reward: float = 0.0
        self.episode_start_time: float = time.monotonic()
        self.base_distance: float = 0.0
        self._last_base_pos: np.ndarray | None = None

        # Named save/restore slots — full physics state plus wrapper bits.
        self._state_size = int(mujoco.mj_stateSize(self.model, STATE_SPEC))
        self.state_slots: dict[str, dict[str, Any]] = {}

        # Reset is routed through the step loop so env.reset() always runs on
        # the same thread that owns env.step() / env.render(). Racing those
        # crashes robosuite (env.sim is briefly None during reset).
        self._reset_pending = threading.Event()
        self._reset_done = threading.Event()

    # ---- threaded sim driver ----

    def start(self) -> None:
        if self.thread is not None and self.thread.is_alive():
            return
        self.stop_event.clear()
        self.thread = threading.Thread(target=self._run_realtime, name="robocasa-sim", daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=2.0)

    # ---- public commands ----

    def reset(self) -> dict[str, Any]:
        # Signal the step loop to do the actual env.reset(); never call it
        # directly from the HTTP thread (see _reset_pending docstring).
        self._reset_done.clear()
        self._reset_pending.set()
        self._reset_done.wait(timeout=10.0)
        with self.lock:
            return self.observe_locked()

    def _do_reset_locked(self) -> None:
        obs = self.env.reset()
        with self.lock:
            self.current_action = np.zeros(self.action_dim, dtype=float)
            self.last_obs = obs
            self.last_reward = 0.0
            self.last_done = False
            self.last_info = {}
            self.last_success = self._check_success_safe()
            self.step_count = 0
            # robosuite resets re-randomize object placements, so object body
            # IDs may shift if a new layout swaps in different props.
            self.object_body_ids = self._discover_object_bodies()
            # New layout = bump scene_version so cached /scene fetches
            # invalidate on the client side.
            self.scene_version += 1
            self.episode_reward = 0.0
            self.episode_start_time = time.monotonic()
            self.base_distance = 0.0
            self._last_base_pos = None

    def set_control(self, ctrl: list[float]) -> dict[str, Any]:
        if len(ctrl) != self.action_dim:
            raise HTTPException(
                status_code=400,
                detail=f"ctrl length must be {self.action_dim} (env.action_dim), got {len(ctrl)}",
            )
        with self.lock:
            self.current_action = np.asarray(ctrl, dtype=float)
            return self.observe_locked()

    # ---- observe / spec ----

    def observe(self) -> dict[str, Any]:
        with self.lock:
            return self.observe_locked()

    def observe_locked(self) -> dict[str, Any]:
        objects = self.objects_locked()
        contacts = self.contacts_locked()
        return {
            # raw mujoco view (matches the Panda bridge for parity)
            "time": float(self.data.time),
            "qpos": self.data.qpos.tolist(),
            "qvel": self.data.qvel.tolist(),
            "ctrl": self.data.ctrl.tolist(),
            "action": self.current_action.tolist(),
            "model": {
                "nq": int(self.model.nq),
                "nv": int(self.model.nv),
                "nu": int(self.model.nu),
                "action_dim": int(self.action_dim),
                "control_freq": int(self.control_freq),
            },
            "names": {
                "actuators": self.actuator_names,
                "joints": self.joint_names,
                "bodies": self.body_names,
            },
            "objects": objects,
            "blocks": objects,  # alias kept for pre-robocasa clients
            "contacts": contacts,
            "scene_version": int(self.scene_version),
            "episode": {
                "reward": float(self.episode_reward),
                "base_distance": float(self.base_distance),
                "wall_time": float(time.monotonic() - self.episode_start_time),
                "step_count": int(self.step_count),
            },

            # robosuite-enriched view
            "obs": _serialize_obs(self.last_obs),
            "reward": float(self.last_reward),
            "done": bool(self.last_done),
            "success": bool(self.last_success),
            "info": _to_jsonable(self.last_info),
            "step_count": int(self.step_count),
            "robot": self._robot_summary_locked(),
        }

    def spec(self) -> dict[str, Any]:
        """Static info — fetch once on connect, not every tick."""
        return {
            "env_name": self.env_name,
            "robot_name": self.robot_name,
            "control_freq": int(self.control_freq),
            "action": {
                "dim": int(self.action_dim),
                "low": self.action_low.tolist(),
                "high": self.action_high.tolist(),
                "layout": self.action_layout,
            },
            "observation": {
                "keys": _obs_key_shapes(self.last_obs),
            },
            "model": {
                "nq": int(self.model.nq),
                "nv": int(self.model.nv),
                "nu": int(self.model.nu),
                "nbody": int(self.model.nbody),
                "njnt": int(self.model.njnt),
            },
            "names": {
                "actuators": self.actuator_names,
                "joints": self.joint_names,
                "bodies": self.body_names,
                "free_jointed_objects": [self.body_names[i] for i in self.object_body_ids],
            },
        }

    # ---- per-tick details ----

    def objects_locked(self) -> list[dict[str, Any]]:
        return [
            {
                "name": self.body_names[body_id],
                "position": self.data.xpos[body_id].tolist(),
                "quaternion": self.data.xquat[body_id].tolist(),
            }
            for body_id in self.object_body_ids
        ]

    def contacts_locked(self) -> list[dict[str, Any]]:
        """Active contacts this tick — what's actually touching what.

        Lets a client detect "I'm stuck" in one round trip instead of
        inferring it from velocity stalls."""
        out: list[dict[str, Any]] = []
        force6 = np.zeros(6, dtype=np.float64)
        for i in range(int(self.data.ncon)):
            c = self.data.contact[i]
            mujoco.mj_contactForce(self.model, self.data, i, force6)
            b1 = int(self.model.geom_bodyid[int(c.geom1)])
            b2 = int(self.model.geom_bodyid[int(c.geom2)])
            out.append({
                "body1": self.body_names[b1],
                "body2": self.body_names[b2],
                "pos": c.pos.tolist(),
                "dist": float(c.dist),
                "normal_force": float(force6[0]),
            })
        return out

    def scene_locked(self) -> dict[str, Any]:
        """Static obstacle map: every body's current world pose plus its
        collision geoms (contype != 0). Fetch once after Reset; rebuild
        only when the layout actually changes."""
        bodies: list[dict[str, Any]] = []
        for body_id in range(self.model.nbody):
            adr = int(self.model.body_geomadr[body_id])
            num = int(self.model.body_geomnum[body_id])
            geoms: list[dict[str, Any]] = []
            for gid in range(adr, adr + num):
                # skip pure-visual geoms (no collision class)
                if int(self.model.geom_contype[gid]) == 0 and int(self.model.geom_conaffinity[gid]) == 0:
                    continue
                geoms.append({
                    "type": int(self.model.geom_type[gid]),
                    "pos": self.model.geom_pos[gid].tolist(),
                    "quat": self.model.geom_quat[gid].tolist(),
                    "size": self.model.geom_size[gid].tolist(),
                    "rbound": float(self.model.geom_rbound[gid]),
                })
            if not geoms:
                continue  # bodies with no collision geoms can't block anything
            bodies.append({
                "name": self.body_names[body_id],
                "pos": self.data.xpos[body_id].tolist(),
                "quat": self.data.xquat[body_id].tolist(),
                "geoms": geoms,
            })
        return {"version": int(self.scene_version), "bodies": bodies}

    def body_pose_locked(self, name: str) -> dict[str, Any] | None:
        """Per-body world pose by name — convenience over walking /observe."""
        body_id = self.body_id_by_name.get(name)
        if body_id is None:
            return None
        return {
            "name": name,
            "id": int(body_id),
            "pos": self.data.xpos[body_id].tolist(),
            "quat": self.data.xquat[body_id].tolist(),
        }

    def reach_check_locked(self, target_pos: list[float]) -> dict[str, Any]:
        """Heuristic reachability for the right arm.

        Distance from the arm's shoulder body (robot0_link0) to the world
        target, compared to Panda's link sum (~0.855 m). This is a
        first-pass filter for the path planner — it tells you whether the
        base is parked close enough that *some* IK solution can exist.
        Doesn't run IK or check joint limits."""
        target = np.asarray(target_pos, dtype=float)
        if target.shape != (3,):
            raise HTTPException(400, "target_pos must be [x, y, z]")
        shoulder_id = (
            self.body_id_by_name.get("robot0_link0")
            or self.body_id_by_name.get("robot0_base")
            or 0
        )
        shoulder = self.data.xpos[shoulder_id].copy()
        delta = target - shoulder
        dist = float(np.linalg.norm(delta))
        return {
            "shoulder_body": self.body_names[shoulder_id],
            "shoulder_pos": shoulder.tolist(),
            "target_pos": target.tolist(),
            "delta": delta.tolist(),
            "distance": dist,
            "horizontal_distance": float(np.linalg.norm(delta[:2])),
            "max_reach": float(PANDA_REACH_RADIUS),
            "reachable": dist <= PANDA_REACH_RADIUS,
            "margin": float(PANDA_REACH_RADIUS - dist),
        }

    def save_state_locked(self, slot: str) -> dict[str, Any]:
        """Snapshot enough state to deterministically resume — physics
        (qpos/qvel/act/mocap/userdata via mj_getState) plus our wrapper
        bookkeeping (current_action, episode metrics, last_obs)."""
        if not slot:
            raise HTTPException(400, "slot must be a non-empty string")
        state = np.zeros(self._state_size, dtype=np.float64)
        mujoco.mj_getState(self.model, self.data, state, STATE_SPEC)
        self.state_slots[slot] = {
            "physics": state,
            "scene_version": int(self.scene_version),
            "current_action": self.current_action.copy(),
            "step_count": self.step_count,
            "last_obs": self.last_obs,
            "last_reward": self.last_reward,
            "last_done": self.last_done,
            "last_success": self.last_success,
            "last_info": self.last_info,
            "episode_reward": self.episode_reward,
            "episode_start_time": self.episode_start_time,
            "base_distance": self.base_distance,
            "saved_wall_time": time.monotonic(),
        }
        return {
            "slot": slot,
            "physics_size": int(self._state_size),
            "saved_at_step": int(self.step_count),
            "slots": list(self.state_slots.keys()),
        }

    def restore_state_locked(self, slot: str) -> dict[str, Any]:
        """Restore a previously saved slot. Calls mj_forward so derived
        kinematics (xpos, xquat) are valid before the next observation."""
        s = self.state_slots.get(slot)
        if s is None:
            raise HTTPException(404, f"unknown state slot: {slot!r}")
        # If a Reset happened between save and restore, body IDs may have
        # shifted and mj_setState would silently corrupt the scene.
        if int(s["scene_version"]) != int(self.scene_version):
            raise HTTPException(
                409,
                f"slot {slot!r} was saved against scene_version="
                f"{s['scene_version']} but current is {self.scene_version}; "
                "a Reset invalidated the snapshot.",
            )
        mujoco.mj_setState(self.model, self.data, s["physics"], STATE_SPEC)
        mujoco.mj_forward(self.model, self.data)
        self.current_action = s["current_action"].copy()
        self.step_count = s["step_count"]
        self.last_obs = s["last_obs"]
        self.last_reward = s["last_reward"]
        self.last_done = s["last_done"]
        self.last_success = s["last_success"]
        self.last_info = s["last_info"]
        self.episode_reward = s["episode_reward"]
        self.episode_start_time = s["episode_start_time"]
        self.base_distance = s["base_distance"]
        # Restart distance tracking from the restored base pose so the
        # next step doesn't add a phantom jump from wherever we were.
        self._last_base_pos = None
        return self.observe_locked()

    def list_states_locked(self) -> dict[str, Any]:
        return {
            "slots": [
                {
                    "name": name,
                    "saved_at_step": int(s["step_count"]),
                    "saved_wall_time": float(s["saved_wall_time"]),
                }
                for name, s in self.state_slots.items()
            ]
        }

    def delete_state_locked(self, slot: str) -> dict[str, Any]:
        if slot not in self.state_slots:
            raise HTTPException(404, f"unknown state slot: {slot!r}")
        del self.state_slots[slot]
        return {"slot": slot, "slots": list(self.state_slots.keys())}

    def set_arm_qpos_locked(self, qpos: list[float]) -> dict[str, Any]:
        """Joint-space teleport for the 7 arm joints. Useful escape hatch
        when OSC has wound itself into a bad pose. Caller is responsible
        for following up with a SetControl that targets the new EEF (or
        zeroing the action) — the next env.step still runs the OSC
        controller against whatever current_action holds."""
        n = len(self.arm_qpos_addrs)
        if len(qpos) != n:
            raise HTTPException(
                400,
                f"qpos must have {n} entries (arm joints {self.arm_joint_names}), got {len(qpos)}",
            )
        for addr, val in zip(self.arm_qpos_addrs, qpos):
            self.data.qpos[addr] = float(val)
        for vaddr in self.arm_qvel_addrs:
            self.data.qvel[vaddr] = 0.0
        mujoco.mj_forward(self.model, self.data)
        return self.observe_locked()

    def render_locked(self, camera: str | int, width: int, height: int) -> bytes:
        """Single-frame PNG snapshot from a named camera. Lazy GL
        context — first call may be slow, and on hosts without an EGL/
        OSMesa backend it raises 503."""
        if Image is None:
            raise HTTPException(503, "Pillow not installed; cannot encode PNG")
        if width <= 0 or height <= 0 or width > RENDER_MAX_DIM or height > RENDER_MAX_DIM:
            raise HTTPException(
                400, f"width/height must be in (0, {RENDER_MAX_DIM}]"
            )
        try:
            renderer = mujoco.Renderer(self.model, height=height, width=width)
        except Exception as exc:
            raise HTTPException(
                503, f"renderer init failed ({type(exc).__name__}: {exc}). "
                "Set MUJOCO_GL=egl (or osmesa) and ensure the GL backend is installed.",
            )
        try:
            renderer.update_scene(self.data, camera=camera)
            img = renderer.render()
        except Exception as exc:
            raise HTTPException(400, f"render failed: {type(exc).__name__}: {exc}")
        buf = io.BytesIO()
        Image.fromarray(img, mode="RGB").save(buf, format="PNG")
        return buf.getvalue()

    def _robot_summary_locked(self) -> dict[str, Any]:
        """Pull the most useful robot fields out of last_obs for clients
        that don't want to walk the full obs dict."""
        obs = self.last_obs or {}

        def _arr(key: str) -> list[float] | None:
            v = obs.get(key)
            if isinstance(v, np.ndarray):
                return v.tolist()
            if isinstance(v, (list, tuple)):
                return [float(x) for x in v]
            return None

        return {
            "eef_pos": _arr("robot0_eef_pos"),
            "eef_quat": _arr("robot0_eef_quat"),
            "base_pos": _arr("robot0_base_pos"),
            "base_quat": _arr("robot0_base_quat"),
            "base_to_eef_pos": _arr("robot0_base_to_eef_pos"),
            "base_to_eef_quat": _arr("robot0_base_to_eef_quat"),
            "joint_pos": _arr("robot0_joint_pos"),
            "joint_vel": _arr("robot0_joint_vel"),
            "gripper_qpos": _arr("robot0_gripper_qpos"),
            "gripper_qvel": _arr("robot0_gripper_qvel"),
        }

    def _check_success_safe(self) -> bool:
        check = getattr(self.env.env, "_check_success", None)
        if check is None:
            return False
        try:
            return bool(check())
        except Exception:
            return False

    # ---- step loop ----

    def step_once(self) -> None:
        if self._reset_pending.is_set():
            try:
                self._do_reset_locked()
            except Exception as exc:
                print(f"[robocasa-sim] reset error: {exc}", flush=True)
            finally:
                self._reset_pending.clear()
                self._reset_done.set()
            return
        with self.lock:
            action = self.current_action.copy()
        try:
            ret = self.env.step(action)
        except Exception as exc:
            print(f"[robocasa-sim] step error: {exc}", flush=True)
            return

        # robosuite native is 4-tuple; gym 0.26+ uses 5-tuple. Be defensive.
        if len(ret) == 4:
            obs, reward, done, info = ret
        elif len(ret) == 5:
            obs, reward, terminated, truncated, info = ret
            done = bool(terminated) or bool(truncated)
        else:
            print(f"[robocasa-sim] unexpected step tuple len={len(ret)}", flush=True)
            return

        with self.lock:
            self.last_obs = obs
            self.last_reward = float(reward)
            self.last_done = bool(done)
            self.last_info = info if isinstance(info, dict) else {"raw": info}
            self.last_success = self._check_success_safe()
            self.step_count += 1
            self.episode_reward += float(reward)
            base_pos = obs.get("robot0_base_pos") if isinstance(obs, dict) else None
            if base_pos is not None:
                bp = np.asarray(base_pos, dtype=float)
                if self._last_base_pos is not None:
                    self.base_distance += float(
                        np.linalg.norm(bp[:2] - self._last_base_pos[:2])
                    )
                self._last_base_pos = bp

    def _run_realtime(self) -> None:
        period = 1.0 / float(self.control_freq)
        next_step = time.perf_counter()

        while not self.stop_event.is_set():
            self.step_once()
            next_step += period
            sleep_time = next_step - time.perf_counter()
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                next_step = time.perf_counter()

    # ---- name / body helpers ----

    def _names(self, obj_type: mujoco.mjtObj, count: int) -> list[str]:
        names: list[str] = []
        for obj_id in range(count):
            name = mujoco.mj_id2name(self.model, obj_type, obj_id)
            names.append(name or f"{obj_type.name.lower()}_{obj_id}")
        return names

    def _discover_object_bodies(self) -> list[int]:
        """Bodies driven by a single freejoint = movable scene props."""
        ids: list[int] = []
        for body_id in range(self.model.nbody):
            if int(self.model.body_jntnum[body_id]) != 1:
                continue
            jnt_id = int(self.model.body_jntadr[body_id])
            if int(self.model.jnt_type[jnt_id]) != int(mujoco.mjtJoint.mjJNT_FREE):
                continue
            ids.append(body_id)
        return ids

    def _discover_arm_addrs(self) -> tuple[list[int], list[int], list[str]]:
        """Find qpos/qvel addresses for the right-arm joints, ordered
        by joint number (robot0_joint1, robot0_joint2, ...). Returns
        empty lists if the model isn't a Panda-style arm."""
        triples: list[tuple[int, str, int]] = []
        for jnt_id in range(self.model.njnt):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, jnt_id) or ""
            if not name.startswith("robot0_joint"):
                continue
            suffix = name[len("robot0_joint"):]
            if not suffix.isdigit():
                continue
            triples.append((int(suffix), name, jnt_id))
        triples.sort()
        qpos_addrs = [int(self.model.jnt_qposadr[t[2]]) for t in triples]
        qvel_addrs = [int(self.model.jnt_dofadr[t[2]]) for t in triples]
        names = [t[1] for t in triples]
        return qpos_addrs, qvel_addrs, names


def create_app(sim: RobocasaSim, manage_sim: bool = True) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if manage_sim:
            sim.start()
        try:
            yield
        finally:
            if manage_sim:
                sim.stop()

    @asynccontextmanager
    async def noop_lifespan(app: FastAPI):
        yield

    app = FastAPI(
        title="MuJoCo Robocasa API",
        lifespan=lifespan if manage_sim else noop_lifespan,
    )

    @app.get("/observe")
    def observe() -> dict[str, Any]:
        return sim.observe()

    @app.get("/spec")
    def spec() -> dict[str, Any]:
        return sim.spec()

    @app.get("/scene")
    def scene() -> dict[str, Any]:
        with sim.lock:
            return sim.scene_locked()

    @app.get("/body/{name}")
    def body(name: str) -> dict[str, Any]:
        with sim.lock:
            pose = sim.body_pose_locked(name)
        if pose is None:
            raise HTTPException(404, f"unknown body: {name!r}")
        return pose

    @app.get("/states")
    def states() -> dict[str, Any]:
        with sim.lock:
            return sim.list_states_locked()

    @app.post("/reach_check")
    def reach_check(payload: dict[str, Any]) -> dict[str, Any]:
        target = payload.get("target_pos")
        if not isinstance(target, list):
            raise HTTPException(400, "body must include target_pos: [x, y, z]")
        with sim.lock:
            return sim.reach_check_locked(target)

    @app.get("/render")
    def render(
        camera: str = "robot0_agentview_center",
        w: int = RENDER_DEFAULT_W,
        h: int = RENDER_DEFAULT_H,
    ) -> Response:
        with sim.lock:
            png = sim.render_locked(camera, w, h)
        return Response(content=png, media_type="image/png")

    @app.post("/input")
    def input_action(action: InputAction) -> dict[str, Any]:
        if action.type == "SetControl":
            ctrl = action.data.get("ctrl")
            if not isinstance(ctrl, list) or not all(isinstance(v, int | float) for v in ctrl):
                raise HTTPException(
                    status_code=400,
                    detail="SetControl requires data.ctrl as a list of numbers",
                )
            return sim.set_control([float(v) for v in ctrl])

        if action.type == "Reset":
            return sim.reset()

        if action.type == "SaveState":
            slot = action.data.get("slot")
            if not isinstance(slot, str):
                raise HTTPException(400, "SaveState requires data.slot (string)")
            with sim.lock:
                return sim.save_state_locked(slot)

        if action.type == "RestoreState":
            slot = action.data.get("slot")
            if not isinstance(slot, str):
                raise HTTPException(400, "RestoreState requires data.slot (string)")
            with sim.lock:
                return sim.restore_state_locked(slot)

        if action.type == "DeleteState":
            slot = action.data.get("slot")
            if not isinstance(slot, str):
                raise HTTPException(400, "DeleteState requires data.slot (string)")
            with sim.lock:
                return sim.delete_state_locked(slot)

        if action.type == "SetArmJointPos":
            qpos = action.data.get("qpos")
            if not isinstance(qpos, list) or not all(isinstance(v, int | float) for v in qpos):
                raise HTTPException(
                    400, "SetArmJointPos requires data.qpos as a list of numbers"
                )
            with sim.lock:
                return sim.set_arm_qpos_locked([float(v) for v in qpos])

        raise HTTPException(status_code=400, detail=f"unknown input type: {action.type}")

    @app.get("/api.md", response_class=PlainTextResponse)
    def api_doc() -> str:
        return API_DOC.read_text()

    return app


def _build_sim() -> RobocasaSim:
    env = make_env(
        env_name=os.environ.get("ROBOCASA_ENV", DEFAULT_ENV),
        robot=os.environ.get("ROBOCASA_ROBOT", DEFAULT_ROBOT),
        control_freq=int(os.environ.get("ROBOCASA_CONTROL_FREQ", DEFAULT_CONTROL_FREQ)),
    )
    return RobocasaSim(env)


def main() -> None:
    sim = _build_sim()
    app = create_app(sim)
    uvicorn.run(app, host=DEFAULT_HOST, port=DEFAULT_PORT)


if __name__ == "__main__":
    main()
