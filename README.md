# MuJoCo Clawblox Worlds

MuJoCo worlds plus Clawblox-style launchers for running Claude agents against
HTTP-controlled simulations. Worlds live under `worlds/`, each with its own
`world.toml`, prompt, API docs, scene assets, and run results.

## Layout

```text
agent/                         Claude agent launcher/runtime
worlds/mujoco-panda/           Single Franka Panda world
worlds/mujoco-dual-panda/      Two-arm Panda world
server.py                      Generic MuJoCo HTTP server
mujoco_recording.py            HDF5 recording reader/writer
run_replay.py                  Native MuJoCo recording replay
run_web_replay.py              Browser-based 3D recording replay
launch_multi_generations_claude.sh
```

The Panda model is vendored from MuJoCo Menagerie under
`worlds/mujoco-panda/models/franka_emika_panda`.

## Setup

```bash
uv sync
```

For Claude agent runs, the launcher expects a usable Claude Code install and
auth token. A local `.env` file can provide:

```bash
CLAUDE_CODE_OAUTH_TOKEN=...
```

## Running a World

Each world supplies a single run command in `<world>/world.toml`. The launcher
runs that command from the world directory and appends the port and recording
options.

Run the Panda server directly:

```bash
cd worlds/mujoco-panda
uv run --project ../.. python ../../server.py --scene models/panda_cube/scene.xml
```

Then inspect the state:

```bash
curl http://127.0.0.1:8080/observe
```

The server starts a live browser spectator by default on `port + 1000` and
prints its URL as:

```text
Spectator frontend: http://127.0.0.1:9080/?spectator_token=...
```

Send a control vector:

```bash
curl -X POST http://127.0.0.1:8080/input \
  -H 'Content-Type: application/json' \
  -d '{"type":"SetControl","data":{"ctrl":[0,0,0,-1.57079,0,1.57079,-0.7853,255]}}'
```

Open the native MuJoCo viewer for the default Panda scene:

```bash
DISPLAY=:0 uv run python run_viewer.py
```

Run the server with an attached native viewer:

```bash
DISPLAY=:0 uv run python run_with_viewer.py
```

## Agent Runs

Single Panda world with one agent:

```bash
bash agent/launch_multi_claude.sh \
  --world-dir worlds/mujoco-panda \
  --base-port 8085 \
  --tmux-session mujoco-panda-agent \
  --run-id mujoco-panda-test \
  --sandbox
```

Dual Panda world with two agents:

```bash
bash agent/launch_multi_claude.sh \
  --world-dir worlds/mujoco-dual-panda \
  --base-port 8085 \
  --tmux-session mujoco-dual-panda-agents \
  --run-id mujoco-dual-panda-test \
  --agents-per-world 2 \
  --sandbox
```

Attach to a run:

```bash
tmux attach -t mujoco-panda-agent
```

The launcher also prints a `spectator:` URL for each world once it is ready.
Open that URL to watch the live simulation in the browser.

Stop a run:

```bash
bash agent/launch_multi_claude.sh --tmux-session mujoco-panda-agent --stop
```

Each agent gets a workspace under the run directory. The selected world source
is copied into `<agent-workspace>/world`, excluding `results`, so the agent can
read scene XML, API docs, and model assets without access to the whole repo.

## Multi-Generation Runs

Use the root launcher for generation chains:

```bash
bash launch_multi_generations_claude.sh \
  --world-dir worlds/mujoco-panda \
  --experiment-id panda-build \
  --generations 5 \
  --generation-duration 2h \
  --base-port 8085 \
  --sandbox
```

Results are stored under the selected world:

```text
worlds/mujoco-panda/results/<experiment-id>/
```

## Recordings

Agent runs record by default. Per-world recordings are saved at:

```text
worlds/<world-name>/results/<run-id>/worlds/world-0/recordings/
```

Each recording has:

- `<timestamp>.h5` for MuJoCo preview frames and checkpoints
- `<timestamp>.events.jsonl` for session/input/chat events

To record when running the server directly:

```bash
cd worlds/mujoco-panda
uv run --project ../.. python ../../server.py \
  --scene models/panda_cube/scene.xml \
  --record \
  --record-dir recordings
```

## Replay

Native MuJoCo viewer:

```bash
uv run python run_replay.py path/to/recording.h5
```

Native replay keys: space toggles play/pause, arrow keys seek, Home/End jump,
and `[` / `]` adjust speed.

Browser replay with interactive 3D camera:

```bash
uv run python run_web_replay.py path/to/recording.h5 --port 8081
```

Open `http://127.0.0.1:8081`. The browser view supports camera orbit, pan,
zoom, pause/play, seeking, speed changes, and scene-visibility controls.

Validate a recording without opening a viewer:

```bash
uv run python run_web_replay.py path/to/recording.h5 --check
```

## World Convention

A world is a directory with:

```text
world.toml
API.md
system_prompt.md
models/...
```

Minimal `world.toml`:

```toml
name = "MuJoCo Panda"

[run]
command = [
  "uv", "run", "--project", "../..",
  "python", "../../server.py",
  "--scene", "models/panda_cube/scene.xml",
]
```

Keep world-specific assets and instructions inside the world directory. Keep
shared runtime behavior in the generic engine files at the repo root.

External worlds launched through Clawblox receive runtime defaults through:

```text
WORLD_HOST
WORLD_PORT
WORLD_BASE_PATH
WORLD_RECORD
WORLD_RECORD_DIR
WORLD_RESUME_PATH
CLAWBLOX_BIN
```

Server CLI flags still override these environment defaults.
