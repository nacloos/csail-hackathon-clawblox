# Franka Panda MuJoCo Sandbox

Tiny MuJoCo setup for a Franka Emika Panda arm manipulating a cube.

The Panda model is vendored from DeepMind's MuJoCo Menagerie in
`models/franka_emika_panda`. The custom scene in `models/panda_cube/scene.xml`
adds a table, cube, lighting, and camera around that robot.

## Run

Standalone viewer only:

```bash
DISPLAY=:0 uv run --with mujoco python run_viewer.py
```

If the viewer opens, you should see the Panda arm in its home pose with a red
cube on the table. The MuJoCo viewer control panel can edit the actuator
controls directly.

## Agent API

Run the real-time simulation server:

```bash
uv run --with mujoco --with fastapi --with uvicorn python server.py
```

Run the same API server with an attached viewer:

```bash
DISPLAY=:0 uv run --with mujoco --with fastapi --with uvicorn python run_with_viewer.py
```

Use `run_with_viewer.py` when you want to see the exact simulation controlled by
`/input`. Running `server.py` and `run_viewer.py` separately starts two separate
MuJoCo simulations.

Then use the Clawblox-style endpoints:

```bash
curl http://localhost:8080/observe
curl -X POST http://localhost:8080/input \
  -H 'Content-Type: application/json' \
  -d '{"type":"SetControl","data":{"ctrl":[0,0,0,-1.57079,0,1.57079,-0.7853,255]}}'
```

For headless rendering/debugging on WSL:

```bash
MUJOCO_GL=egl uv run --with mujoco python smoke_test.py
```

## Robocasa kitchen flow

This branch also ships a parallel bridge that wraps a [robocasa.ai](https://robocasa.ai)
kitchen task instead of the bare Panda+cube scene. The HTTP surface is the
same `/observe` + `/input` shape, so an existing Clawblox-style agent can
point at either backend.

One-shot install:

```bash
./robocasa_bridge/install.sh
```

See [robocasa_bridge/README.md](robocasa_bridge/README.md) for run
instructions, the action vector layout, and a Python client example.
