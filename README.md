# Franka Panda MuJoCo Sandbox

Tiny MuJoCo setup for a Franka Emika Panda arm manipulating construction
objects.

The Panda model is vendored from DeepMind's MuJoCo Menagerie in
`models/franka_emika_panda`. The custom scene in `models/panda_cube/scene.xml`
adds blocks, bricks, planks, pillars, lighting, and camera around that robot.

## Run

Standalone viewer only:

```bash
DISPLAY=:0 uv run --with mujoco python run_viewer.py
```

If the viewer opens, you should see the Panda arm in its home pose with objects
placed around the robot. The MuJoCo viewer control panel can edit the actuator
controls directly.

## Agent API

Run the real-time simulation server:

```bash
uv run --with mujoco --with fastapi --with uvicorn python server.py
```

Run the server and record an optimized replay artifact:

```bash
uv run --with mujoco --with h5py --with fastapi --with uvicorn \
  python server.py --record
```

Run the same API server with an attached viewer:

```bash
DISPLAY=:0 uv run --with mujoco --with fastapi --with uvicorn python run_with_viewer.py
```

Run one shared server with two Panda arms:

```bash
uv run --with mujoco --with fastapi --with uvicorn python server.py --dual-panda
```

In the dual-arm world, each `/join` response assigns a session to one robot
(`left` or `right`). Agents use `SetControl` with an 8-value vector; the server
applies it only to the robot owned by that session.

The viewer runner accepts the same recording flags:

```bash
DISPLAY=:0 uv run --with mujoco --with h5py --with fastapi --with uvicorn \
  python run_with_viewer.py --record --preview-hz 30 --checkpoint-seconds 1
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

## Recordings and Replay

Recordings are written to `recordings/*.h5` by default, with a sibling
`*.events.jsonl` file for inputs and session events. The HDF5 file stores
downsampled preview arrays (`qpos`, `qvel`, `ctrl`) for fast scrubbing plus
periodic full MuJoCo integration-state checkpoints for exact recovery work.

Replay a recording with the native MuJoCo viewer:

```bash
DISPLAY=:0 uv run --with mujoco --with h5py python run_replay.py recordings/<file>.h5
```

Replay controls: space toggles play/pause, arrow keys seek by one simulated
second, Home/End jump to the start/end, and `[` / `]` adjust speed.

Validate a recording without opening a viewer:

```bash
uv run --with mujoco --with h5py python run_replay.py recordings/<file>.h5 --check
```

## Claude Agent

Run one simulator world with one Claude agent and an attached viewer:

```bash
bash agent/launch_multi_claude.sh --world-dir worlds/mujoco-panda --base-port 8085 --tmux-session mujoco-panda-agent --run-id mujoco-panda-test --model claude-opus-4-7 --sandbox --world-server-cmd 'DISPLAY=:0 uv run --with mujoco --with fastapi --with uvicorn python run_with_viewer.py'
```

Run one simulator world with two Panda arms and two Claude agents:

```bash
bash agent/launch_multi_claude.sh \
  --world-dir worlds/mujoco-dual-panda \
  --base-port 8085 \
  --tmux-session mujoco-dual-panda-agents \
  --run-id mujoco-dual-panda-test \
  --agents-per-world 2 \
  --model claude-opus-4-7 \
  --sandbox \
  --world-server-cmd 'DISPLAY=:0 uv run --with mujoco --with fastapi --with uvicorn python run_with_viewer.py --dual-panda'
```

Attach to the tmux session:

```bash
tmux attach -t mujoco-panda-agent
```
