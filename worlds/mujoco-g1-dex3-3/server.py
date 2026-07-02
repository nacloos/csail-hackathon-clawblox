"""MuJoCo G1 world with a Unitree SDK2 DDS control interface.

This world speaks the same wire protocol as the physical Unitree G1: a
CycloneDDS bus carrying `unitree_hg` IDL messages. A controller subscribes
`rt/lowstate` and publishes `rt/lowcmd` (per-motor q / dq / kp / kd / tau); the
same controller binary can drive the real robot.

Actuate injects the per-run bus coordinates as WORLD_DDS_DOMAIN_ID /
WORLD_DDS_INTERFACE and encloses this process and the agent in one network
namespace, so concurrent runs share no bus. See the Actuate `interfaces` and
`isolation` docs.

HTTP here is only the lifecycle layer: readiness (`GET /api.md`), session
(`POST /join`), and a read-only `GET /observe` for humans/debugging. Robot
control is DDS, not HTTP.
"""
from __future__ import annotations

import argparse
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import mujoco
from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel
import uvicorn

ROOT = Path(__file__).resolve().parent
# Vendored, cyclonedds-only unitree_hg IDL (no unitree_sdk2py, no native libs).
sys.path.insert(0, str(ROOT / "vendor"))

from unitree_sdk2py.idl.unitree_hg.msg.dds_ import (  # noqa: E402
    LowCmd_,
    LowState_,
    MotorState_,
    IMUState_,
)
from cyclonedds.domain import DomainParticipant  # noqa: E402
from cyclonedds.topic import Topic  # noqa: E402
from cyclonedds.pub import DataWriter  # noqa: E402
from cyclonedds.sub import DataReader  # noqa: E402

DEFAULT_SCENE = ROOT / "models" / "g1" / "scene_29dof.xml"
API_DOC = ROOT / "API.md"
TOPIC_LOWCMD = "rt/lowcmd"
TOPIC_LOWSTATE = "rt/lowstate"
MOTOR_SLOTS = 35  # IDL fixed array length (unused tail is zero-filled)
TIMESTEP = 0.002  # 500 Hz


def env_str(name: str, default: str) -> str:
    value = os.environ.get(name)
    return value if value else default


class G1DDSWorld:
    def __init__(self, scene: Path, domain_id: int, interface: str) -> None:
        self.model = mujoco.MjModel.from_xml_path(str(scene))
        self.model.opt.timestep = TIMESTEP
        self.data = mujoco.MjData(self.model)
        self.nu = int(self.model.nu)
        self.domain_id = domain_id
        self.interface = interface

        # Map each actuator to the qpos/qvel address of the joint it drives,
        # using MuJoCo's own tables so the mapping holds for any scene.
        self.qadr = np.zeros(self.nu, dtype=int)
        self.vadr = np.zeros(self.nu, dtype=int)
        for act in range(self.nu):
            joint = int(self.model.actuator_trnid[act, 0])
            self.qadr[act] = int(self.model.jnt_qposadr[joint])
            self.vadr[act] = int(self.model.jnt_dofadr[joint])
        # Root free-joint state (floating base) for the IMU message.
        self.root_qadr = 0
        self.root_vadr = 0

        self.lock = threading.RLock()
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None
        self.latest_cmd: LowCmd_ | None = None
        self.tick = 0

        # DDS interface is set up on the sim thread (CycloneDDS is thread-bound).
        self._participant: DomainParticipant | None = None
        self._reader: DataReader | None = None
        self._writer: DataWriter | None = None

    # -- lifecycle -----------------------------------------------------------
    def start(self) -> None:
        self.thread = threading.Thread(target=self._run, name="g1-dds-sim", daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=2.0)

    # -- realtime loop -------------------------------------------------------
    def _setup_dds(self) -> None:
        self._participant = DomainParticipant(self.domain_id)
        cmd_topic = Topic(self._participant, TOPIC_LOWCMD, LowCmd_)
        state_topic = Topic(self._participant, TOPIC_LOWSTATE, LowState_)
        self._reader = DataReader(self._participant, cmd_topic)
        self._writer = DataWriter(self._participant, state_topic)
        print(
            f"DDS active: domain={self.domain_id} interface={self.interface} "
            f"topics={TOPIC_LOWCMD},{TOPIC_LOWSTATE} actuators={self.nu}",
            flush=True,
        )

    def _run(self) -> None:
        # CycloneDDS reads WORLD_DDS_INTERFACE via env config; domain is explicit.
        os.environ.setdefault("CYCLONEDDS_URI", self._cyclonedds_uri())
        self._setup_dds()
        next_step = time.perf_counter()
        while not self.stop_event.is_set():
            samples = self._reader.take(N=16)
            if samples:
                self.latest_cmd = samples[-1]
            with self.lock:
                self._apply_control_locked()
                mujoco.mj_step(self.model, self.data)
                self.tick += 1
                state = self._build_lowstate_locked()
            self._writer.write(state)

            next_step += TIMESTEP
            sleep = next_step - time.perf_counter()
            if sleep > 0:
                time.sleep(sleep)
            else:
                next_step = time.perf_counter()

    def _cyclonedds_uri(self) -> str:
        # `lo` is not multicast-capable, so discovery uses unicast peers on
        # localhost rather than SPDP multicast. Discovery is locked to the
        # assigned interface; the per-run domain id provides isolation. Agents
        # must use the same config (see API.md) to interoperate.
        return (
            "<CycloneDDS><Domain><General>"
            f"<Interfaces><NetworkInterface name='{self.interface}'/></Interfaces>"
            "<AllowMulticast>false</AllowMulticast></General>"
            "<Discovery><ParticipantIndex>auto</ParticipantIndex>"
            "<Peers><Peer address='localhost'/></Peers></Discovery>"
            "</Domain></CycloneDDS>"
        )

    def _apply_control_locked(self) -> None:
        cmd = self.latest_cmd
        if cmd is None:
            return
        # Onboard-style joint PD over torque actuators, matching the real G1.
        for i in range(self.nu):
            mc = cmd.motor_cmd[i]
            q = self.data.qpos[self.qadr[i]]
            dq = self.data.qvel[self.vadr[i]]
            self.data.ctrl[i] = mc.kp * (mc.q - q) + mc.kd * (mc.dq - dq) + mc.tau

    def _build_lowstate_locked(self) -> LowState_:
        motors = []
        for i in range(MOTOR_SLOTS):
            if i < self.nu:
                q = float(self.data.qpos[self.qadr[i]])
                dq = float(self.data.qvel[self.vadr[i]])
                tau = float(self.data.actuator_force[i])
            else:
                q = dq = tau = 0.0
            motors.append(
                MotorState_(
                    mode=1, q=q, dq=dq, ddq=0.0, tau_est=tau,
                    temperature=[0, 0], vol=0.0, sensor=[0, 0],
                    motorstate=0, reserve=[0, 0, 0, 0],
                )
            )
        quat = [float(x) for x in self.data.qpos[3:7]]      # MuJoCo root quaternion (w,x,y,z)
        gyro = [float(x) for x in self.data.qvel[3:6]]      # root angular velocity
        imu = IMUState_(
            quaternion=quat, gyroscope=gyro, accelerometer=[0.0, 0.0, 0.0],
            rpy=[0.0, 0.0, 0.0], temperature=0,
        )
        return LowState_(
            version=[0, 0], mode_pr=0, mode_machine=0, tick=self.tick & 0xFFFFFFFF,
            imu_state=imu, motor_state=motors, wireless_remote=[0] * 40,
            reserve=[0, 0, 0, 0], crc=0,
        )

    # -- read-only observation (HTTP debug) ----------------------------------
    def observe(self) -> dict[str, Any]:
        with self.lock:
            return {
                "time": float(self.data.time),
                "tick": self.tick,
                "state": {
                    "qpos": self.data.qpos.tolist(),
                    "qvel": self.data.qvel.tolist(),
                    "ctrl": self.data.ctrl.tolist(),
                },
                "dds": {
                    "domain_id": self.domain_id,
                    "interface": self.interface,
                    "topics": {"command": TOPIC_LOWCMD, "state": TOPIC_LOWSTATE},
                    "have_command": self.latest_cmd is not None,
                },
            }


class ChatPost(BaseModel):
    content: str


def create_app(world: G1DDSWorld) -> FastAPI:
    sessions: dict[str, str] = {}

    app = FastAPI(title="G1 DDS World")

    @app.get("/api.md", response_class=PlainTextResponse)
    def api_doc() -> str:
        return API_DOC.read_text()

    @app.get("/skill.md", response_class=PlainTextResponse)
    def skill_doc() -> str:
        return API_DOC.read_text()

    @app.post("/join")
    def join(name: str = "agent", x_session: str | None = Header(default=None)) -> dict[str, Any]:
        session = x_session or uuid4().hex
        sessions[session] = name
        return {"session": session, "name": name, "robot": "g1",
                "dds": {"domain_id": world.domain_id, "interface": world.interface,
                        "command_topic": TOPIC_LOWCMD, "state_topic": TOPIC_LOWSTATE}}

    @app.post("/leave")
    def leave(x_session: str | None = Header(default=None)) -> dict[str, Any]:
        if x_session:
            sessions.pop(x_session, None)
        return {"ok": True}

    @app.get("/observe")
    def observe() -> dict[str, Any]:
        return world.observe()

    @app.post("/input")
    def input_action() -> dict[str, Any]:
        raise HTTPException(
            status_code=400,
            detail="Robot control is direct DDS in this world; publish rt/lowcmd "
                   "(unitree_hg LowCmd_) and subscribe rt/lowstate.",
        )

    return app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", type=Path, default=DEFAULT_SCENE)
    parser.add_argument("--host", default=env_str("WORLD_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(env_str("WORLD_PORT", "8080")))
    parser.add_argument("--dds-domain", type=int,
                        default=int(env_str("WORLD_DDS_DOMAIN_ID", "0")))
    parser.add_argument("--dds-interface", default=env_str("WORLD_DDS_INTERFACE", "lo"))
    args = parser.parse_args()

    world = G1DDSWorld(args.scene, args.dds_domain, args.dds_interface)
    world.start()
    app = create_app(world)
    print(
        f"HTTP lifecycle on http://{args.host}:{args.port}"
        " (/api.md, /join, /observe) — robot control is DDS",
        flush=True,
    )
    try:
        uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    finally:
        world.stop()


if __name__ == "__main__":
    main()
