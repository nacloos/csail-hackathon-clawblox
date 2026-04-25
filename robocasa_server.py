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

    uv run python robocasa_server.py
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
import os
import threading
import time
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field
import mujoco
import numpy as np
import uvicorn

from robocasa_setup import (
    DEFAULT_CONTROL_FREQ,
    DEFAULT_ENV,
    DEFAULT_ROBOT,
    RobocasaEnv,
    make_env,
)


ROOT = Path(__file__).resolve().parent
API_DOC = ROOT / "API.md"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8080


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
        self.object_body_ids = self._discover_object_bodies()

        self.current_action = np.zeros(self.action_dim, dtype=float)
        self.last_obs: dict[str, Any] = env.initial_obs
        self.last_reward: float = 0.0
        self.last_done: bool = False
        self.last_info: dict[str, Any] = {}
        self.last_success: bool = self._check_success_safe()
        self.step_count: int = 0
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
