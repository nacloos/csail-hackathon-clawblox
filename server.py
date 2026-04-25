from __future__ import annotations

import argparse
from contextlib import asynccontextmanager
from pathlib import Path
import threading
import time
from uuid import uuid4
from typing import Any

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field
import mujoco
import uvicorn

from panda_setup import set_panda_home


ROOT = Path(__file__).resolve().parent
SCENE = ROOT / "models" / "panda_cube" / "scene.xml"
API_DOC = ROOT / "API.md"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8080


class InputAction(BaseModel):
    type: str
    data: dict[str, Any] = Field(default_factory=dict)


class Session:
    def __init__(self, name: str, session_id: str | None = None) -> None:
        self.name = name
        self.session_id = session_id or str(uuid4())
        self.agent_id = str(uuid4())
        self.created_at = time.time()
        self.last_seen = self.created_at


class SimState:
    def __init__(self, scene: Path) -> None:
        self.model = mujoco.MjModel.from_xml_path(str(scene))
        self.data = mujoco.MjData(self.model)
        self.lock = threading.RLock()
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None
        self.actuator_names = self._names(mujoco.mjtObj.mjOBJ_ACTUATOR, self.model.nu)
        self.joint_names = self._names(mujoco.mjtObj.mjOBJ_JOINT, self.model.njnt)
        self.body_names = self._names(mujoco.mjtObj.mjOBJ_BODY, self.model.nbody)
        self.object_body_ids = [
            body_id
            for body_id, name in enumerate(self.body_names)
            if name.startswith(("block_", "brick_", "plank_", "pillar_"))
        ]
        self.sessions: dict[str, Session] = {}
        self.reset()

    def start(self) -> None:
        if self.thread is not None and self.thread.is_alive():
            return
        self.stop_event.clear()
        self.thread = threading.Thread(target=self._run_realtime, name="mujoco-sim", daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=2.0)

    def reset(self) -> dict[str, Any]:
        with self.lock:
            mujoco.mj_resetData(self.model, self.data)
            set_panda_home(self.model, self.data)
            return self.observe_locked()

    def set_control(self, ctrl: list[float]) -> dict[str, Any]:
        if len(ctrl) != self.model.nu:
            raise HTTPException(status_code=400, detail=f"ctrl length must be {self.model.nu}, got {len(ctrl)}")

        with self.lock:
            self.data.ctrl[:] = ctrl
            return self.observe_locked()

    def observe(self) -> dict[str, Any]:
        with self.lock:
            return self.observe_locked()

    def observe_locked(self) -> dict[str, Any]:
        objects = self.objects_locked()
        return {
            "time": float(self.data.time),
            "qpos": self.data.qpos.tolist(),
            "qvel": self.data.qvel.tolist(),
            "ctrl": self.data.ctrl.tolist(),
            "model": {
                "nq": int(self.model.nq),
                "nv": int(self.model.nv),
                "nu": int(self.model.nu),
            },
            "names": {
                "actuators": self.actuator_names,
                "joints": self.joint_names,
                "bodies": self.body_names,
            },
            "objects": objects,
            "blocks": objects,
        }

    def join(self, name: str, existing_session_id: str | None = None) -> dict[str, Any]:
        with self.lock:
            if existing_session_id and existing_session_id in self.sessions:
                session = self.sessions[existing_session_id]
                session.name = name
                session.last_seen = time.time()
            else:
                session = Session(name, existing_session_id)
                self.sessions[session.session_id] = session
            return {
                "session": session.session_id,
                "agent_id": session.agent_id,
                "name": session.name,
            }

    def leave(self, session_id: str | None) -> dict[str, Any]:
        if not session_id:
            return {"ok": True}
        with self.lock:
            self.sessions.pop(session_id, None)
            return {"ok": True}

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            return {
                "time": float(self.data.time),
                "qpos": self.data.qpos.tolist(),
                "qvel": self.data.qvel.tolist(),
                "ctrl": self.data.ctrl.tolist(),
                "sessions": [
                    {
                        "session": session.session_id,
                        "agent_id": session.agent_id,
                        "name": session.name,
                        "created_at": session.created_at,
                        "last_seen": session.last_seen,
                    }
                    for session in self.sessions.values()
                ],
            }

    def objects_locked(self) -> list[dict[str, Any]]:
        return [
            {
                "name": self.body_names[body_id],
                "position": self.data.xpos[body_id].tolist(),
                "quaternion": self.data.xquat[body_id].tolist(),
            }
            for body_id in self.object_body_ids
        ]

    def _run_realtime(self) -> None:
        next_step = time.perf_counter()
        timestep = float(self.model.opt.timestep)

        while not self.stop_event.is_set():
            with self.lock:
                mujoco.mj_step(self.model, self.data)

            next_step += timestep
            sleep_time = next_step - time.perf_counter()
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                next_step = time.perf_counter()

    def _names(self, obj_type: mujoco.mjtObj, count: int) -> list[str]:
        names: list[str] = []
        for obj_id in range(count):
            name = mujoco.mj_id2name(self.model, obj_type, obj_id)
            names.append(name or f"{obj_type.name.lower()}_{obj_id}")
        return names


def create_app(sim: SimState, manage_sim: bool = True) -> FastAPI:
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

    app = FastAPI(title="MuJoCo Panda API", lifespan=lifespan if manage_sim else noop_lifespan)

    @app.post("/join")
    def join(name: str = "agent", x_session: str | None = Header(default=None)) -> dict[str, Any]:
        return sim.join(name, x_session)

    @app.post("/leave")
    def leave(x_session: str | None = Header(default=None)) -> dict[str, Any]:
        return sim.leave(x_session)

    @app.get("/observe")
    def observe() -> dict[str, Any]:
        return sim.observe()

    @app.post("/input")
    def input_action(action: InputAction) -> dict[str, Any]:
        if action.type == "SetControl":
            ctrl = action.data.get("ctrl")
            if not isinstance(ctrl, list) or not all(isinstance(value, int | float) for value in ctrl):
                raise HTTPException(status_code=400, detail="SetControl requires data.ctrl as a list of numbers")
            return sim.set_control([float(value) for value in ctrl])

        if action.type == "Reset":
            return sim.reset()

        raise HTTPException(status_code=400, detail=f"unknown input type: {action.type}")

    @app.get("/api.md", response_class=PlainTextResponse)
    def api_doc() -> str:
        return API_DOC.read_text()

    @app.get("/skill.md", response_class=PlainTextResponse)
    def skill_doc() -> str:
        return API_DOC.read_text()

    @app.get("/snapshot")
    def snapshot() -> dict[str, Any]:
        return sim.snapshot()

    return app


sim = SimState(SCENE)
app = create_app(sim)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the MuJoCo Panda API server.")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
