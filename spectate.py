"""Out-of-process spectator for a running world.

Renders a live browser view without touching the world's sim thread: it polls
the world's lightweight ``/state`` (qpos) and updates a viser MuJoCo scene in
*this* separate process, so the world can run ``--no-spectator`` at full realtime
(no shared GIL) while you still watch live.

    python spectate.py --scene worlds/mujoco-g1-dex3-3/models/g1/scene_29dof.xml --port 8185

Any Python with mujoco + viser + mjviser works (the repo uv venv or the conda
env). The spectator URL is printed on start.
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.request

import mujoco
import viser
from mjviser.scene import ViserMujocoScene


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True, help="Scene XML the world runs.")
    ap.add_argument("--port", type=int, default=8080, help="World HTTP port.")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--spectator-port", type=int, default=0, help="0 = pick a free port.")
    ap.add_argument("--hz", type=float, default=30.0)
    args = ap.parse_args()

    model = mujoco.MjModel.from_xml_path(args.scene)
    data = mujoco.MjData(model)
    server = viser.ViserServer(
        host=args.host,
        port=args.spectator_port or None,
        label="MuJoCo spectator",
    )
    scene = ViserMujocoScene(server, model, num_envs=1)
    print(
        f"Spectator: http://{args.host}:{server.get_port()}/  "
        f"(watching world http://{args.host}:{args.port})",
        flush=True,
    )

    url = f"http://{args.host}:{args.port}/state"
    delay = 1.0 / max(1.0, args.hz)
    while True:
        try:
            with urllib.request.urlopen(url, timeout=2) as resp:
                s = json.load(resp)
            data.qpos[:] = s["qpos"]
            mujoco.mj_forward(model, data)  # qpos -> geom world poses (this process)
            scene.update_from_mjdata(data)
        except Exception:
            time.sleep(0.5)
            continue
        time.sleep(delay)


if __name__ == "__main__":
    main()
