from __future__ import annotations

import argparse
import threading
import time

import mujoco.viewer
import uvicorn

from server import DEFAULT_HOST, DEFAULT_PORT, SCENE, SimState, create_app


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the MuJoCo Panda API server with a passive viewer.")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = parser.parse_args()

    sim = SimState(SCENE)
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
