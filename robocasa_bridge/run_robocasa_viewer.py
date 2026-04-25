"""Run robocasa_server.py with robosuite's native mjviewer.

Why not ``mujoco.viewer.launch_passive`` like the Panda flow? The kitchen
scene is too heavy — on WSLg the launch_passive renderer gets through about
30 seconds of geometry/texture upload and then dies silently. Robosuite's
own ``mjviewer`` is what the robocasa demos use and it handles the scene
fine.

Architecture: the env is built with ``has_renderer=True``. The HTTP server
runs in a daemon thread. The main thread is the sim+render loop:

    while window open:
        env.step(current_action)
        env.render()

SetControl/Reset over /input mutate ``current_action`` and a pending-reset
flag, which the main loop drains.

Run:

    DISPLAY=:0 uv run python robocasa_bridge/run_robocasa_viewer.py
"""

from __future__ import annotations

import os
import threading
import time

import uvicorn

from robocasa_server import (
    DEFAULT_HOST,
    DEFAULT_PORT,
    RobocasaSim,
    create_app,
)
from robocasa_setup import (
    DEFAULT_CONTROL_FREQ,
    DEFAULT_ENV,
    DEFAULT_ROBOT,
    make_env,
)


def main() -> None:
    print("[viewer] building robocasa env (has_renderer=True)...", flush=True)
    env = make_env(
        env_name=os.environ.get("ROBOCASA_ENV", DEFAULT_ENV),
        robot=os.environ.get("ROBOCASA_ROBOT", DEFAULT_ROBOT),
        control_freq=int(os.environ.get("ROBOCASA_CONTROL_FREQ", DEFAULT_CONTROL_FREQ)),
        has_renderer=True,
    )
    sim = RobocasaSim(env)
    print(f"[viewer] env ready (action_dim={sim.action_dim}, nq={sim.model.nq})", flush=True)

    # The HTTP server lives in a daemon thread and writes to sim.current_action
    # via /input. The main thread owns env.step + env.render, which is the
    # invariant robosuite's mjviewer expects.
    app = create_app(sim, manage_sim=False)
    server = uvicorn.Server(
        uvicorn.Config(app, host=DEFAULT_HOST, port=DEFAULT_PORT, log_level="info")
    )
    server_thread = threading.Thread(target=server.run, name="robocasa-api", daemon=True)
    server_thread.start()
    print("[viewer] http server thread started; entering render loop", flush=True)

    period = 1.0 / float(sim.control_freq)
    next_step = time.perf_counter()
    try:
        while True:
            sim.step_once()
            env.render()

            next_step += period
            sleep_time = next_step - time.perf_counter()
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                next_step = time.perf_counter()
    except KeyboardInterrupt:
        print("[viewer] interrupted", flush=True)
    except Exception as exc:
        # Closing the mjviewer window typically raises; treat that as a
        # clean exit rather than a crash.
        print(f"[viewer] render loop ended: {type(exc).__name__}: {exc}", flush=True)
    finally:
        server.should_exit = True
        server_thread.join(timeout=2.0)


if __name__ == "__main__":
    main()
