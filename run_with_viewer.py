from __future__ import annotations

import argparse
from pathlib import Path
import threading
import time

import mujoco.viewer
import uvicorn

from mujoco_recording import RecordingConfig, timestamped_recording_path
from server import (
    DEFAULT_HOST,
    DEFAULT_PORT,
    DEFAULT_RECORD_DIR,
    DEFAULT_SPECTATOR_PORT_OFFSET,
    DUAL_SCENE,
    SCENE,
    LiveSpectator,
    SimState,
    create_app,
    ensure_dual_panda_scene,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the MuJoCo API server with a passive viewer.")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--scene", type=Path, default=SCENE)
    parser.add_argument("--api-doc", type=Path)
    parser.add_argument("--no-spectator", action="store_true", help="Disable the live browser spectator.")
    parser.add_argument("--spectator-host", default=DEFAULT_HOST)
    parser.add_argument("--spectator-public-host", default=DEFAULT_HOST)
    parser.add_argument("--spectator-port", type=int)
    parser.add_argument("--spectator-hz", type=float, default=30.0)
    parser.add_argument("--dual-panda", action="store_true", help="Run one shared world with left/right Panda arms.")
    parser.add_argument("--record", action="store_true", help="Record the session to HDF5.")
    parser.add_argument("--record-dir", type=Path, default=DEFAULT_RECORD_DIR)
    parser.add_argument("--record-path", type=Path)
    parser.add_argument("--preview-hz", type=float, default=30.0)
    parser.add_argument("--checkpoint-seconds", type=float, default=1.0)
    args = parser.parse_args()

    api_doc = args.api_doc or (Path.cwd() / "API.md" if (Path.cwd() / "API.md").exists() else None)
    if api_doc is not None and not api_doc.is_absolute():
        api_doc = Path.cwd() / api_doc
    record_path = None
    if args.record:
        record_dir = args.record_dir if args.record_dir.is_absolute() else Path.cwd() / args.record_dir
        record_path = args.record_path or timestamped_recording_path(record_dir)
        if not record_path.is_absolute():
            record_path = Path.cwd() / record_path
    scene = ensure_dual_panda_scene(DUAL_SCENE) if args.dual_panda else args.scene
    if not scene.is_absolute():
        scene = Path.cwd() / scene
    sim = SimState(
        scene,
        record_path=record_path,
        record_config=RecordingConfig(
            preview_hz=args.preview_hz,
            checkpoint_seconds=args.checkpoint_seconds,
        ),
    )
    spectator = None
    if not args.no_spectator:
        spectator = LiveSpectator(
            sim,
            host=args.spectator_host,
            port=args.spectator_port or (args.port + DEFAULT_SPECTATOR_PORT_OFFSET),
            public_host=args.spectator_public_host,
            update_hz=args.spectator_hz,
        )
    app = (
        create_app(sim, manage_sim=False, api_doc_path=api_doc, spectator=spectator)
        if api_doc
        else create_app(sim, manage_sim=False, spectator=spectator)
    )
    server = uvicorn.Server(
        uvicorn.Config(app, host=args.host, port=args.port, log_level="info")
    )
    server_thread = threading.Thread(target=server.run, name="mujoco-api", daemon=True)

    sim.start()
    if spectator is not None:
        spectator.start()
    server_thread.start()

    try:
        with mujoco.viewer.launch_passive(sim.model, sim.data) as viewer:
            while viewer.is_running():
                with sim.lock:
                    viewer.sync()
                time.sleep(sim.model.opt.timestep)
    finally:
        server.should_exit = True
        if spectator is not None:
            spectator.stop()
        sim.stop()
        server_thread.join(timeout=2.0)


if __name__ == "__main__":
    main()
