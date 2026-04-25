from __future__ import annotations

import argparse
from pathlib import Path
import threading
import time

import mujoco.viewer
import uvicorn

from mujoco_recording import RecordingConfig, timestamped_recording_path
from dual_panda_scene import DUAL_SCENE, ensure_dual_panda_scene
from server import DEFAULT_HOST, DEFAULT_PORT, DEFAULT_RECORD_DIR, SCENE, SimState, create_app


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the MuJoCo Panda API server with a passive viewer.")
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
    sim = SimState(
        scene,
        record_path=record_path,
        record_config=RecordingConfig(
            preview_hz=args.preview_hz,
            checkpoint_seconds=args.checkpoint_seconds,
        ),
    )
    app = create_app(sim, manage_sim=False)
    server = uvicorn.Server(
        uvicorn.Config(app, host=args.host, port=args.port, log_level="info")
    )
    server_thread = threading.Thread(target=server.run, name="mujoco-api", daemon=True)

    sim.start()
    server_thread.start()

    try:
        with mujoco.viewer.launch_passive(sim.model, sim.data) as viewer:
            while viewer.is_running():
                with sim.lock:
                    viewer.sync()
                time.sleep(sim.model.opt.timestep)
    finally:
        server.should_exit = True
        sim.stop()
        server_thread.join(timeout=2.0)


if __name__ == "__main__":
    main()
