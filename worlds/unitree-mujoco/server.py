from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import signal
import socket
import shutil
import sys
import subprocess
import threading
import time
from typing import Any
from urllib.parse import parse_qs, urlparse
from uuid import uuid4


ROOT = Path(__file__).resolve().parent
SIM_PYTHON = ROOT / "Unitree-Mujoco-Dex3" / "simulate_python"
LOCAL_UNITREE_SDK = ROOT / ".deps" / "unitree_sdk2_python"
BRICKLAYING_SRC = ROOT / "G1-Bricklaying-Simulation" / "src"
API_DOC = ROOT / "API.md"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8080
DEFAULT_SPECTATOR_OFFSET = 1000
CONDA_ENV = os.environ.get("UNITREE_MUJOCO_CONDA_ENV", "unitree-mujoco")
VIEWER_DT = float(os.environ.get("UNITREE_VIEWER_DT", "0.2"))


def ensure_conda_python() -> None:
    if os.environ.get("UNITREE_MUJOCO_SKIP_CONDA_REEXEC") == "1":
        return
    if CONDA_ENV in Path(sys.executable).parts:
        return
    conda = shutil.which("conda")
    if conda is None:
        return
    try:
        base = subprocess.check_output(
            [conda, "info", "--base"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return
    python = Path(base) / "envs" / CONDA_ENV / "bin" / "python"
    if not python.is_file() or python.resolve() == Path(sys.executable).resolve():
        return
    env = os.environ.copy()
    env.pop("VIRTUAL_ENV", None)
    env["CONDA_PREFIX"] = str(python.parents[1])
    env["PATH"] = os.pathsep.join(
        [
            str(python.parent),
            str(Path(base) / "bin"),
            *[
                item
                for item in env.get("PATH", "").split(os.pathsep)
                if item and ".venv" not in Path(item).parts
            ],
        ]
    )
    os.execve(str(python), [str(python), *sys.argv], env)


@dataclass
class Session:
    name: str
    session: str = field(default_factory=lambda: str(uuid4()))
    agent_id: str = field(default_factory=lambda: str(uuid4()))
    created_at: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)


class UnitreeMujocoRuntime:
    def __init__(
        self,
        *,
        scene: Path | None,
        interface: str,
        domain_id: int,
        spectator_host: str,
        spectator_port: int,
        spectator_public_host: str,
        enable_cmd_vel: bool,
        print_scene_info: bool,
    ) -> None:
        self.scene_arg = scene
        self.interface = interface
        self.domain_id = domain_id
        self.spectator_host = spectator_host
        self.spectator_port = spectator_port
        self.spectator_public_host = spectator_public_host
        self.enable_cmd_vel = enable_cmd_vel
        self.print_scene_info = print_scene_info

        self.lock = threading.RLock()
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None
        self.tick = 0
        self.started_at = time.time()
        self.ready = False
        self.error: str | None = None
        self.model = None
        self.data = None
        self.viewer_server = None
        self.viewer_scene = None
        self.spectator_url = f"http://{self.spectator_public_host}:{self.spectator_port}/"

    def start(self) -> None:
        os.environ.setdefault("MUJOCO_GL", "egl")
        os.environ.setdefault("ROS_LOCALHOST_ONLY", "1")
        os.environ.setdefault("RMW_IMPLEMENTATION", "rmw_fastrtps_cpp")
        os.environ["UNITREE_DDS_DOMAIN_ID"] = str(self.domain_id)
        os.environ["UNITREE_DDS_INTERFACE"] = self.interface
        for path in (BRICKLAYING_SRC, LOCAL_UNITREE_SDK, SIM_PYTHON):
            if path.exists():
                sys.path.insert(0, str(path))

        import mujoco
        import viser
        from mjviser.scene import ViserMujocoScene
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize

        import config
        import unitree_sdk2py_bridge
        from unitree_sdk2py_bridge import ElasticBand, UnitreeSdk2Bridge

        scene = self._resolve_scene(config)
        self.model = mujoco.MjModel.from_xml_path(str(scene))
        self.data = mujoco.MjData(self.model)
        self.model.opt.timestep = config.SIMULATE_DT

        unitree_sdk2py_bridge.motion = self.enable_cmd_vel
        if not self.print_scene_info:
            UnitreeSdk2Bridge.PrintSceneInformation = lambda self: None

        ChannelFactoryInitialize(self.domain_id, self.interface)
        self.bridge = UnitreeSdk2Bridge(self.model, self.data, self.lock)

        self.viewer_server = viser.ViserServer(
            host=self.spectator_host,
            port=self.spectator_port,
            label="unitree-mujoco",
        )
        self.viewer_scene = ViserMujocoScene(self.viewer_server, self.model, num_envs=1)
        print(f"Spectator frontend: {self.spectator_url}", flush=True)
        print(
            "DDS active: "
            f"domain={self.domain_id} interface={self.interface} "
            "topics=rt/lowstate,rt/lowcmd,rt/dex3/*,rt/realsense/*",
            flush=True,
        )

        elastic_band = ElasticBand()
        band_attached_link = 1
        viewer_data = mujoco.MjData(self.model)
        viewer_lock = threading.RLock()

        def run_loop() -> None:
            last_viewer_snapshot = 0.0
            try:
                while not self.stop_event.is_set():
                    step_start = time.perf_counter()
                    with self.lock:
                        if config.ENABLE_ELASTIC_BAND and elastic_band.enable:
                            self.data.xfrc_applied[band_attached_link, :3] = elastic_band.Advance(
                                self.data.qpos[:3], self.data.qvel[:3]
                            )
                        mujoco.mj_step(self.model, self.data)
                        self.tick += 1

                        now = time.perf_counter()
                        if now - last_viewer_snapshot >= VIEWER_DT:
                            with viewer_lock:
                                viewer_data.qpos[:] = self.data.qpos
                                viewer_data.qvel[:] = self.data.qvel
                                viewer_data.ctrl[:] = self.data.ctrl
                                viewer_data.mocap_pos[:] = self.data.mocap_pos
                                viewer_data.mocap_quat[:] = self.data.mocap_quat
                            last_viewer_snapshot = now

                    sleep_s = float(self.model.opt.timestep) - (time.perf_counter() - step_start)
                    if sleep_s > 0:
                        time.sleep(sleep_s)
            except BaseException as exc:
                self.error = f"{type(exc).__name__}: {exc}"
                print(f"simulation loop failed: {self.error}", flush=True)

        def viewer_loop() -> None:
            try:
                while not self.stop_event.is_set():
                    loop_start = time.perf_counter()
                    with viewer_lock:
                        mujoco.mj_forward(self.model, viewer_data)
                        self.viewer_scene.update_from_mjdata(viewer_data)
                    sleep_s = VIEWER_DT - (time.perf_counter() - loop_start)
                    if sleep_s > 0:
                        self.stop_event.wait(sleep_s)
            except BaseException as exc:
                self.error = f"{type(exc).__name__}: {exc}"
                print(f"viewer loop failed: {self.error}", flush=True)

        self.thread = threading.Thread(target=run_loop, name="unitree-mujoco-sim", daemon=True)
        self.thread.start()
        self.viewer_thread = threading.Thread(target=viewer_loop, name="unitree-mujoco-viewer", daemon=True)
        self.viewer_thread.start()
        self.ready = True

    def stop(self) -> None:
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=2.0)
        if getattr(self, "viewer_thread", None) is not None:
            self.viewer_thread.join(timeout=2.0)
        if self.viewer_server is not None:
            self.viewer_server.stop()

    def observe(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "ok": self.error is None,
            "error": self.error,
            "ready": self.ready,
            "robot": "g1",
            "dds": {
                "domain_id": self.domain_id,
                "interface": self.interface,
                "topics": [
                    "rt/lowstate",
                    "rt/lowcmd",
                    "rt/dex3/left/state",
                    "rt/dex3/left/cmd",
                    "rt/dex3/right/state",
                    "rt/dex3/right/cmd",
                    "rt/realsense/color",
                    "rt/realsense/depth",
                ],
            },
            "spectator_url": self.spectator_url,
            "uptime_seconds": time.time() - self.started_at,
            "tick": self.tick,
        }
        if self.data is not None:
            with self.lock:
                payload["time"] = float(self.data.time)
                payload["state"] = {
                    "qpos": self.data.qpos.tolist(),
                    "qvel": self.data.qvel.tolist(),
                    "ctrl": self.data.ctrl.tolist(),
                    "mocap_pos": self.data.mocap_pos.tolist(),
                    "mocap_quat": self.data.mocap_quat.tolist(),
                }
                payload["contacts"] = self._contact_snapshot()
        return payload

    def _contact_snapshot(self, limit: int = 64) -> list[dict[str, Any]]:
        import mujoco

        if self.model is None or self.data is None:
            return []
        contacts: list[dict[str, Any]] = []
        for i in range(min(int(self.data.ncon), limit)):
            contact = self.data.contact[i]
            geom1 = int(contact.geom1)
            geom2 = int(contact.geom2)
            body1 = int(self.model.geom_bodyid[geom1])
            body2 = int(self.model.geom_bodyid[geom2])
            contacts.append(
                {
                    "geom1": geom1,
                    "geom2": geom2,
                    "geom1_name": mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom1),
                    "geom2_name": mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom2),
                    "body1": body1,
                    "body2": body2,
                    "body1_name": mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body1),
                    "body2_name": mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body2),
                    "dist": float(contact.dist),
                    "pos": contact.pos.tolist(),
                }
            )
        return contacts

    def _resolve_scene(self, config: Any) -> Path:
        if self.scene_arg is not None:
            return self.scene_arg if self.scene_arg.is_absolute() else ROOT / self.scene_arg
        configured = Path(config.ROBOT_SCENE)
        if configured.is_absolute():
            return configured
        return (SIM_PYTHON / configured).resolve()


class UnitreeWorldState:
    def __init__(self, runtime: UnitreeMujocoRuntime) -> None:
        self.runtime = runtime
        self.sessions: dict[str, Session] = {}
        self.lock = threading.RLock()

    def join(self, name: str, existing_session: str | None) -> dict[str, Any]:
        with self.lock:
            if existing_session and existing_session in self.sessions:
                session = self.sessions[existing_session]
                session.name = name
                session.last_seen = time.time()
            else:
                session = Session(name=name)
                self.sessions[session.session] = session
            return {
                "session": session.session,
                "agent_id": session.agent_id,
                "name": session.name,
                "robot": "g1",
            }

    def leave(self, session_id: str | None) -> dict[str, Any]:
        if session_id:
            with self.lock:
                self.sessions.pop(session_id, None)
        return {"ok": True}

    def observe(self) -> dict[str, Any]:
        with self.lock:
            sessions = [
                {
                    "session": item.session,
                    "agent_id": item.agent_id,
                    "name": item.name,
                    "created_at": item.created_at,
                    "last_seen": item.last_seen,
                }
                for item in self.sessions.values()
            ]
        payload = self.runtime.observe()
        payload["sessions"] = sessions
        return payload


class Handler(BaseHTTPRequestHandler):
    server_version = "UnitreeMujocoClawblox/1.0"
    protocol_version = "HTTP/1.1"

    @property
    def world(self) -> UnitreeWorldState:
        return self.server.world  # type: ignore[attr-defined]

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path in {"/api.md", "/skill.md"}:
            self.respond_text(200, API_DOC.read_text(encoding="utf-8"), "text/markdown")
            return
        if parsed.path in {"/observe", "/snapshot"}:
            self.respond_json(200, self.world.observe())
            return
        if parsed.path == "/":
            self.respond_json(200, {"ok": True, "api": "/api.md", "observe": "/observe"})
            return
        self.respond_json(404, {"error": f"unknown path: {parsed.path}"})

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/join":
            query = parse_qs(parsed.query)
            name = query.get("name", ["agent"])[0] or "agent"
            existing = self.headers.get("X-Session")
            self.respond_json(200, self.world.join(name, existing))
            return
        if parsed.path == "/leave":
            self.respond_json(200, self.world.leave(self.headers.get("X-Session")))
            return
        if parsed.path == "/input":
            self._consume_body()
            self.respond_json(
                400,
                {
                    "error": "Robot control is direct DDS in this world; use unitree_sdk2py topics.",
                    "docs": "/api.md",
                },
            )
            return
        self._consume_body()
        self.respond_json(404, {"error": f"unknown path: {parsed.path}"})

    def log_message(self, fmt: str, *args: object) -> None:
        print(
            f"{datetime.now(timezone.utc).isoformat()} {self.address_string()} {fmt % args}",
            flush=True,
        )

    def respond_json(self, status: int, payload: dict[str, Any]) -> None:
        data = (json.dumps(payload, separators=(",", ":")) + "\n").encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(data)

    def respond_text(self, status: int, text: str, content_type: str) -> None:
        data = text.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", f"{content_type}; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(data)

    def _consume_body(self) -> bytes:
        length = int(self.headers.get("Content-Length", "0") or "0")
        return self.rfile.read(length) if length else b""


class UnitreeHTTPServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, addr: tuple[str, int], world: UnitreeWorldState) -> None:
        self.world = world
        super().__init__(addr, Handler)


def env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return default
    return int(value)


def choose_port(preferred: int) -> int:
    with socket.socket() as sock:
        try:
            sock.bind(("127.0.0.1", preferred))
            return preferred
        except OSError:
            pass
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Unitree MuJoCo as a Clawblox world.")
    parser.add_argument("--host", default=os.environ.get("WORLD_HOST", DEFAULT_HOST))
    parser.add_argument("--port", type=int, default=env_int("WORLD_PORT", DEFAULT_PORT))
    parser.add_argument("--scene", type=Path)
    parser.add_argument("--interface", default=os.environ.get("UNITREE_DDS_INTERFACE", "lo"))
    parser.add_argument("--domain-id", type=int, default=env_int("UNITREE_DDS_DOMAIN_ID", 0))
    parser.add_argument("--spectator-host", default=os.environ.get("WORLD_SPECTATOR_HOST", DEFAULT_HOST))
    parser.add_argument("--spectator-public-host", default=os.environ.get("WORLD_SPECTATOR_PUBLIC_HOST", DEFAULT_HOST))
    parser.add_argument("--spectator-port", type=int)
    parser.add_argument("--enable-cmd-vel", action="store_true")
    parser.add_argument("--print-scene-info", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    spectator_port = args.spectator_port or choose_port(args.port + DEFAULT_SPECTATOR_OFFSET)
    runtime = UnitreeMujocoRuntime(
        scene=args.scene,
        interface=args.interface,
        domain_id=args.domain_id,
        spectator_host=args.spectator_host,
        spectator_port=spectator_port,
        spectator_public_host=args.spectator_public_host,
        enable_cmd_vel=args.enable_cmd_vel,
        print_scene_info=args.print_scene_info,
    )
    world = UnitreeWorldState(runtime)
    httpd = UnitreeHTTPServer((args.host, args.port), world)

    def stop(_signum: int, _frame: object) -> None:
        threading.Thread(target=httpd.shutdown, daemon=True).start()

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)

    def start_runtime() -> None:
        try:
            runtime.start()
        except BaseException as exc:
            runtime.error = f"{type(exc).__name__}: {exc}"
            print(f"runtime startup failed: {runtime.error}", flush=True)

    runtime_thread = threading.Thread(target=start_runtime, name="unitree-mujoco-start", daemon=True)
    runtime_thread.start()

    print(f"Clawblox API: http://{args.host}:{args.port}", flush=True)
    try:
        httpd.serve_forever(poll_interval=0.2)
    finally:
        httpd.server_close()
        runtime.stop()
        runtime_thread.join(timeout=2.0)
    return 0


if __name__ == "__main__":
    ensure_conda_python()
    raise SystemExit(main())
