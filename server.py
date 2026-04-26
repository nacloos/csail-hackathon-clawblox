from __future__ import annotations

import argparse
from collections import deque
from contextlib import asynccontextmanager
from datetime import UTC, datetime
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

from dual_panda_scene import DUAL_SCENE, ensure_dual_panda_scene
from mujoco_recording import RecordingConfig, RecordingWriter, timestamped_recording_path
from panda_setup import find_panda_arms, set_panda_home


ROOT = Path(__file__).resolve().parent
SCENE = ROOT / "models" / "panda_cube" / "scene.xml"
API_DOC = ROOT / "API.md"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8080
DEFAULT_RECORD_DIR = ROOT / "recordings"

GEOM_TYPES = {
    int(mujoco.mjtGeom.mjGEOM_PLANE): "plane",
    int(mujoco.mjtGeom.mjGEOM_HFIELD): "hfield",
    int(mujoco.mjtGeom.mjGEOM_SPHERE): "sphere",
    int(mujoco.mjtGeom.mjGEOM_CAPSULE): "capsule",
    int(mujoco.mjtGeom.mjGEOM_ELLIPSOID): "ellipsoid",
    int(mujoco.mjtGeom.mjGEOM_CYLINDER): "cylinder",
    int(mujoco.mjtGeom.mjGEOM_BOX): "box",
    int(mujoco.mjtGeom.mjGEOM_MESH): "mesh",
    int(mujoco.mjtGeom.mjGEOM_SDF): "sdf",
}

JOINT_TYPES = {
    int(mujoco.mjtJoint.mjJNT_FREE): "free",
    int(mujoco.mjtJoint.mjJNT_BALL): "ball",
    int(mujoco.mjtJoint.mjJNT_SLIDE): "slide",
    int(mujoco.mjtJoint.mjJNT_HINGE): "hinge",
}

JOINT_QPOS_WIDTHS = {"free": 7, "ball": 4, "slide": 1, "hinge": 1}
JOINT_QVEL_WIDTHS = {"free": 6, "ball": 3, "slide": 1, "hinge": 1}

ACTUATOR_TRANSMISSION_TYPES = {
    int(mujoco.mjtTrn.mjTRN_JOINT): "joint",
    int(mujoco.mjtTrn.mjTRN_JOINTINPARENT): "jointinparent",
    int(mujoco.mjtTrn.mjTRN_SLIDERCRANK): "slidercrank",
    int(mujoco.mjtTrn.mjTRN_TENDON): "tendon",
    int(mujoco.mjtTrn.mjTRN_SITE): "site",
    int(mujoco.mjtTrn.mjTRN_BODY): "body",
}


class InputAction(BaseModel):
    type: str
    data: dict[str, Any] = Field(default_factory=dict)


class RecordStart(BaseModel):
    path: str | None = None
    preview_hz: float = 30.0
    checkpoint_seconds: float = 1.0


class ChatPost(BaseModel):
    content: str


class Session:
    def __init__(self, name: str, session_id: str | None = None, robot: str | None = None) -> None:
        self.name = name
        self.session_id = session_id or str(uuid4())
        self.agent_id = str(uuid4())
        self.robot = robot
        self.created_at = time.time()
        self.last_seen = self.created_at


class SimState:
    def __init__(
        self,
        scene: Path,
        *,
        record_path: Path | None = None,
        record_config: RecordingConfig | None = None,
    ) -> None:
        self.scene = scene
        self.model = mujoco.MjModel.from_xml_path(str(scene))
        self.data = mujoco.MjData(self.model)
        self.lock = threading.RLock()
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None
        self.tick = 0
        self.recorder: RecordingWriter | None = None
        self.last_recording_flush = time.perf_counter()
        self.instance_id = str(uuid4())
        self.chat_next_seq = 0
        self.chat_messages: deque[dict[str, Any]] = deque(maxlen=1000)
        self.actuator_names = self._names(mujoco.mjtObj.mjOBJ_ACTUATOR, self.model.nu)
        self.joint_names = self._names(mujoco.mjtObj.mjOBJ_JOINT, self.model.njnt)
        self.body_names = self._names(mujoco.mjtObj.mjOBJ_BODY, self.model.nbody)
        self.arms = find_panda_arms(self.model)
        self.arms_by_name = {arm.name: arm for arm in self.arms}
        self.object_body_ids = [
            body_id
            for body_id, name in enumerate(self.body_names)
            if name.startswith(("block_", "brick_", "plank_", "pillar_"))
        ]
        self.sessions: dict[str, Session] = {}
        self.reset()
        if record_path is not None:
            self.start_recording(record_path, record_config)

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
        self.stop_recording()

    def reset(self, session_id: str | None = None) -> dict[str, Any]:
        with self.lock:
            mujoco.mj_resetData(self.model, self.data)
            set_panda_home(self.model, self.data)
            self.record_event_locked("Reset", {}, session_id)
            return self.observe_locked(session_id)

    def set_control(self, ctrl: list[float], session_id: str | None = None) -> dict[str, Any]:
        with self.lock:
            if len(self.arms) > 1:
                if not session_id:
                    raise HTTPException(status_code=401, detail="SetControl requires X-Session in multi-robot worlds")
                session = self.sessions.get(session_id)
                if session is None:
                    raise HTTPException(status_code=401, detail="unknown session")
                if session.robot is None:
                    raise HTTPException(status_code=403, detail="session has no assigned robot")
                arm = self.arms_by_name[session.robot]
                if len(ctrl) != len(arm.actuator_ids):
                    raise HTTPException(
                        status_code=400,
                        detail=f"ctrl length must be {len(arm.actuator_ids)} for your assigned robot",
                    )
                for actuator_id, value in zip(arm.actuator_ids, ctrl, strict=True):
                    self.data.ctrl[actuator_id] = value
                self.record_event_locked("SetControl", {"robot": arm.name, "ctrl": ctrl}, session_id)
                return self.observe_locked(session_id)

            if len(ctrl) != self.model.nu:
                raise HTTPException(
                    status_code=400,
                    detail=f"ctrl length must be {self.model.nu}, got {len(ctrl)}",
                )
            self.data.ctrl[:] = ctrl
            self.record_event_locked("SetControl", {"ctrl": ctrl}, session_id)
            return self.observe_locked(session_id)

    def observe(self, session_id: str | None = None) -> dict[str, Any]:
        with self.lock:
            return self.observe_locked(session_id)

    def observe_locked(self, session_id: str | None = None) -> dict[str, Any]:
        objects = self.objects_locked()
        session = self.sessions.get(session_id or "")
        return {
            "time": float(self.data.time),
            "state": {
                "qpos": self.data.qpos.tolist(),
                "qvel": self.data.qvel.tolist(),
                "ctrl": self.data.ctrl.tolist(),
            },
            "model": self.model_payload_locked(),
            "objects": objects,
            "blocks": objects,
            "robots": self.robots_locked(),
            "session": self.session_payload(session) if session else None,
        }

    def join(self, name: str, existing_session_id: str | None = None) -> dict[str, Any]:
        with self.lock:
            if existing_session_id and existing_session_id in self.sessions:
                session = self.sessions[existing_session_id]
                session.name = name
                session.last_seen = time.time()
            else:
                session = Session(name, existing_session_id, self.next_available_robot_locked())
                self.sessions[session.session_id] = session
            return {
                "session": session.session_id,
                "agent_id": session.agent_id,
                "name": session.name,
                "robot": session.robot,
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
                "tick": self.tick,
                "time": float(self.data.time),
                "qpos": self.data.qpos.tolist(),
                "qvel": self.data.qvel.tolist(),
                "ctrl": self.data.ctrl.tolist(),
                "sessions": [
                    {
                        "session": session.session_id,
                        "agent_id": session.agent_id,
                        "name": session.name,
                        "robot": session.robot,
                        "created_at": session.created_at,
                        "last_seen": session.last_seen,
                    }
                    for session in self.sessions.values()
                ],
                "chat": {
                    "next_seq": self.chat_next_seq,
                    "messages": list(self.chat_messages),
                },
            }

    def start_recording(
        self,
        path: Path | None = None,
        config: RecordingConfig | None = None,
    ) -> dict[str, Any]:
        with self.lock:
            if self.recorder is not None:
                raise HTTPException(status_code=409, detail="recording is already active")
            record_path = path or timestamped_recording_path(DEFAULT_RECORD_DIR)
            try:
                self.recorder = RecordingWriter(
                    record_path,
                    scene=self.scene,
                    model=self.model,
                    config=config,
                )
            except RuntimeError as exc:
                raise HTTPException(status_code=500, detail=str(exc)) from exc
            self.recorder.record_initial(self.tick, self.model, self.data)
            self.recorder.record_event(
                {
                    "type": "RecordingStarted",
                    "tick": self.tick,
                    "sim_time": float(self.data.time),
                    "path": str(record_path),
                }
            )
            self.recorder.flush()
            return self.recording_status_locked()

    def stop_recording(self) -> dict[str, Any]:
        with self.lock:
            recorder = self.recorder
            self.recorder = None
            if recorder is None:
                return {"active": False}
            status = recorder.status()
            recorder.close()
            return {"active": False, **status}

    def recording_status(self) -> dict[str, Any]:
        with self.lock:
            return self.recording_status_locked()

    def recording_status_locked(self) -> dict[str, Any]:
        if self.recorder is None:
            return {"active": False}
        return {"active": True, **self.recorder.status()}

    def objects_locked(self) -> list[dict[str, Any]]:
        return [
            {
                "name": self.body_names[body_id],
                "position": self.data.xpos[body_id].tolist(),
                "quaternion": self.data.xquat[body_id].tolist(),
            }
            for body_id in self.object_body_ids
        ]

    def model_payload_locked(self) -> dict[str, Any]:
        bodies = []
        for body_id, name in enumerate(self.body_names):
            geom_start = int(self.model.body_geomadr[body_id])
            geom_count = int(self.model.body_geomnum[body_id])
            joint_start = int(self.model.body_jntadr[body_id])
            joint_count = int(self.model.body_jntnum[body_id])
            parent_id = int(self.model.body_parentid[body_id])
            bodies.append(
                {
                    "id": body_id,
                    "name": name,
                    "parent_id": parent_id if body_id != 0 else None,
                    "parent_name": self.body_names[parent_id] if body_id != 0 else None,
                    "position": self.data.xpos[body_id].tolist(),
                    "quaternion": self.data.xquat[body_id].tolist(),
                    "geom_ids": list(range(geom_start, geom_start + geom_count)),
                    "joint_ids": list(range(joint_start, joint_start + joint_count)),
                }
            )

        joints = []
        for joint_id, name in enumerate(self.joint_names):
            joint_type = JOINT_TYPES.get(int(self.model.jnt_type[joint_id]), str(int(self.model.jnt_type[joint_id])))
            qpos_address = int(self.model.jnt_qposadr[joint_id])
            qvel_address = int(self.model.jnt_dofadr[joint_id])
            qpos_width = JOINT_QPOS_WIDTHS.get(joint_type, 1)
            qvel_width = JOINT_QVEL_WIDTHS.get(joint_type, 1)
            body_id = int(self.model.jnt_bodyid[joint_id])
            joints.append(
                {
                    "id": joint_id,
                    "name": name,
                    "type": joint_type,
                    "body_id": body_id,
                    "body_name": self.body_names[body_id],
                    "qpos_address": qpos_address,
                    "qvel_address": qvel_address,
                    "qpos_width": qpos_width,
                    "qvel_width": qvel_width,
                    "position": self.data.qpos[qpos_address : qpos_address + qpos_width].tolist(),
                    "velocity": self.data.qvel[qvel_address : qvel_address + qvel_width].tolist(),
                    "range": self.model.jnt_range[joint_id].tolist(),
                }
            )

        materials = []
        material_names = self._names(mujoco.mjtObj.mjOBJ_MATERIAL, self.model.nmat)
        for material_id, name in enumerate(material_names):
            materials.append(
                {
                    "id": material_id,
                    "name": name,
                    "rgba": self.model.mat_rgba[material_id].tolist(),
                }
            )

        geoms = []
        for geom_id in range(self.model.ngeom):
            body_id = int(self.model.geom_bodyid[geom_id])
            material_id = int(self.model.geom_matid[geom_id])
            geoms.append(
                {
                    "id": geom_id,
                    "name": mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or f"geom_{geom_id}",
                    "body_id": body_id,
                    "body_name": self.body_names[body_id],
                    "type": GEOM_TYPES.get(int(self.model.geom_type[geom_id]), str(int(self.model.geom_type[geom_id]))),
                    "size": self.model.geom_size[geom_id].tolist(),
                    "local_position": self.model.geom_pos[geom_id].tolist(),
                    "local_quaternion": self.model.geom_quat[geom_id].tolist(),
                    "world_position": self.data.geom_xpos[geom_id].tolist(),
                    "world_xmat": self.data.geom_xmat[geom_id].reshape(3, 3).tolist(),
                    "material_id": material_id if material_id >= 0 else None,
                    "material_name": material_names[material_id] if material_id >= 0 else None,
                    "rgba": (
                        self.model.mat_rgba[material_id].tolist()
                        if material_id >= 0
                        else self.model.geom_rgba[geom_id].tolist()
                    ),
                }
            )

        actuators = []
        for actuator_id, name in enumerate(self.actuator_names):
            trn_type_id = int(self.model.actuator_trntype[actuator_id])
            trn_type = ACTUATOR_TRANSMISSION_TYPES.get(trn_type_id, str(trn_type_id))
            trn_ids = [int(value) for value in self.model.actuator_trnid[actuator_id].tolist()]
            actuators.append(
                {
                    "id": actuator_id,
                    "name": name,
                    "ctrl_index": actuator_id,
                    "ctrl": float(self.data.ctrl[actuator_id]),
                    "ctrlrange": self.model.actuator_ctrlrange[actuator_id].tolist(),
                    "forcerange": self.model.actuator_forcerange[actuator_id].tolist(),
                    "transmission_type": trn_type,
                    "transmission_ids": trn_ids,
                    "target": self.actuator_target_name(trn_type, trn_ids),
                }
            )

        return {
            "nq": int(self.model.nq),
            "nv": int(self.model.nv),
            "nu": int(self.model.nu),
            "bodies": bodies,
            "joints": joints,
            "geoms": geoms,
            "actuators": actuators,
            "materials": materials,
        }

    def post_chat(self, content: str, session_id: str | None) -> dict[str, Any]:
        content = content.strip()
        if not content:
            raise HTTPException(status_code=400, detail="content must be 1-500 non-whitespace characters")
        if len(content) > 500:
            raise HTTPException(status_code=400, detail="content exceeds 500 characters")

        with self.lock:
            session = self.require_session_locked(session_id)
            self.chat_next_seq += 1
            heard_by = [active.agent_id for active in self.sessions.values()]
            audibility = [
                {
                    "agent_id": active.agent_id,
                    "agent_name": active.name,
                    "heard": True,
                }
                for active in self.sessions.values()
            ]
            message = {
                "id": f"local_{self.chat_next_seq}",
                "instance_id": self.instance_id,
                "agent_id": session.agent_id,
                "agent_name": session.name,
                "message_type": "text",
                "content": content,
                "created_at": datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
                "delivery_mode": "global",
                "hearing_radius": None,
                "heard_by": heard_by,
                "audibility": audibility,
            }
            self.chat_messages.append(message)
            self.record_event_locked("ChatMessage", message, session.session_id)
            return {
                "id": message["id"],
                "created_at": message["created_at"],
            }

    def list_chat_messages(
        self,
        *,
        session_id: str | None,
        after: str | None = None,
        limit: int = 50,
    ) -> dict[str, Any]:
        if limit < 1:
            limit = 1
        if limit > 100:
            limit = 100
        after = after.strip() if after else None

        with self.lock:
            self.require_session_locked(session_id)
            messages = [
                message
                for message in self.chat_messages
                if after is None or message["created_at"] > after
            ]
            return {"messages": messages[-limit:]}

    def require_session_locked(self, session_id: str | None) -> Session:
        if not session_id:
            raise HTTPException(status_code=401, detail="X-Session is required")
        session = self.sessions.get(session_id)
        if session is None:
            raise HTTPException(status_code=401, detail="unknown session")
        return session

    def robots_locked(self) -> list[dict[str, Any]]:
        result = []
        for arm in self.arms:
            entry: dict[str, Any] = {
                "name": arm.name,
                "control_indices": list(arm.actuator_ids),
                "ctrl": [float(self.data.ctrl[actuator_id]) for actuator_id in arm.actuator_ids],
                "assigned": any(session.robot == arm.name for session in self.sessions.values()),
            }
            # Add end-effector world position from the "hand" body
            hand_body_name = f"{arm.prefix}hand" if arm.prefix else "hand"
            hand_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, hand_body_name)
            if hand_body_id >= 0:
                entry["ee_pos"] = self.data.xpos[hand_body_id].tolist()
            result.append(entry)
        return result

    def session_payload(self, session: Session) -> dict[str, Any]:
        payload = {
            "session": session.session_id,
            "agent_id": session.agent_id,
            "name": session.name,
            "robot": session.robot,
        }
        if session.robot and session.robot in self.arms_by_name:
            payload["control_indices"] = list(self.arms_by_name[session.robot].actuator_ids)
        return payload

    def actuator_target_name(self, trn_type: str, trn_ids: list[int]) -> dict[str, Any] | None:
        target_id = trn_ids[0]
        if target_id < 0:
            return None
        if trn_type in ("joint", "jointinparent", "slidercrank"):
            return {
                "type": "joint",
                "id": target_id,
                "name": self.joint_names[target_id] if target_id < len(self.joint_names) else None,
            }
        if trn_type == "tendon":
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_TENDON, target_id)
            return {"type": "tendon", "id": target_id, "name": name}
        if trn_type == "site":
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_SITE, target_id)
            return {"type": "site", "id": target_id, "name": name}
        if trn_type == "body":
            return {
                "type": "body",
                "id": target_id,
                "name": self.body_names[target_id] if target_id < len(self.body_names) else None,
            }
        return {"type": trn_type, "id": target_id, "name": None}

    def next_available_robot_locked(self) -> str | None:
        assigned = {session.robot for session in self.sessions.values()}
        for arm in self.arms:
            if arm.name not in assigned:
                return arm.name
        if len(self.arms) == 1:
            return self.arms[0].name
        if self.arms:
            raise HTTPException(status_code=409, detail="all robots already have assigned sessions")
        return None

    def record_event_locked(
        self,
        event_type: str,
        data: dict[str, Any],
        session_id: str | None = None,
    ) -> None:
        if self.recorder is None:
            return
        agent_id = ""
        agent_name = ""
        if session_id and session_id in self.sessions:
            session = self.sessions[session_id]
            agent_id = session.agent_id
            agent_name = session.name
        self.recorder.record_event(
            {
                "type": event_type,
                "tick": self.tick,
                "sim_time": float(self.data.time),
                "session": session_id or "",
                "agent_id": agent_id,
                "agent_name": agent_name,
                "data": data,
            }
        )

    def _run_realtime(self) -> None:
        next_step = time.perf_counter()
        timestep = float(self.model.opt.timestep)

        while not self.stop_event.is_set():
            with self.lock:
                mujoco.mj_step(self.model, self.data)
                self.tick += 1
                if self.recorder is not None:
                    self.recorder.record_step(self.tick, self.model, self.data)
                    if time.perf_counter() - self.last_recording_flush > 5.0:
                        self.recorder.flush()
                        self.last_recording_flush = time.perf_counter()

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

    @app.post("/chat")
    def chat(message: ChatPost, x_session: str | None = Header(default=None)) -> dict[str, Any]:
        return sim.post_chat(message.content, x_session)

    @app.get("/chat/messages")
    def chat_messages(
        after: str | None = None,
        limit: int = 50,
        x_session: str | None = Header(default=None),
    ) -> dict[str, Any]:
        return sim.list_chat_messages(session_id=x_session, after=after, limit=limit)

    @app.get("/observe")
    def observe(x_session: str | None = Header(default=None)) -> dict[str, Any]:
        return sim.observe(x_session)

    @app.post("/input")
    def input_action(action: InputAction, x_session: str | None = Header(default=None)) -> dict[str, Any]:
        if action.type == "SetControl":
            ctrl = action.data.get("ctrl")
            if not isinstance(ctrl, list) or not all(isinstance(value, int | float) for value in ctrl):
                raise HTTPException(status_code=400, detail="SetControl requires data.ctrl as a list of numbers")
            return sim.set_control([float(value) for value in ctrl], x_session)

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

    @app.post("/record/start")
    def record_start(request: RecordStart) -> dict[str, Any]:
        path = Path(request.path) if request.path else timestamped_recording_path(DEFAULT_RECORD_DIR)
        if not path.is_absolute():
            path = ROOT / path
        return sim.start_recording(
            path,
            RecordingConfig(
                preview_hz=request.preview_hz,
                checkpoint_seconds=request.checkpoint_seconds,
            ),
        )

    @app.post("/record/stop")
    def record_stop() -> dict[str, Any]:
        return sim.stop_recording()

    @app.get("/record/status")
    def record_status() -> dict[str, Any]:
        return sim.recording_status()

    @app.get("/recordings")
    def recordings() -> dict[str, Any]:
        DEFAULT_RECORD_DIR.mkdir(parents=True, exist_ok=True)
        files = sorted(DEFAULT_RECORD_DIR.glob("*.h5"), key=lambda item: item.stat().st_mtime, reverse=True)
        return {"recordings": [str(path) for path in files]}

    return app


sim = SimState(SCENE)
app = create_app(sim)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the MuJoCo Panda API server.")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--scene", type=Path, default=SCENE)
    parser.add_argument("--dual-panda", action="store_true", help="Run one shared world with left/right Panda arms.")
    parser.add_argument("--record", action="store_true", help="Record the session to HDF5.")
    parser.add_argument("--record-dir", type=Path, default=DEFAULT_RECORD_DIR)
    parser.add_argument("--record-path", type=Path)
    parser.add_argument("--preview-hz", type=float, default=30.0)
    parser.add_argument("--checkpoint-seconds", type=float, default=1.0)
    args = parser.parse_args()
    record_path = None
    if args.record:
        record_path = args.record_path or timestamped_recording_path(args.record_dir)
    scene = ensure_dual_panda_scene(DUAL_SCENE) if args.dual_panda else args.scene
    local_sim = SimState(
        scene,
        record_path=record_path,
        record_config=RecordingConfig(
            preview_hz=args.preview_hz,
            checkpoint_seconds=args.checkpoint_seconds,
        ),
    )
    uvicorn.run(create_app(local_sim), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
