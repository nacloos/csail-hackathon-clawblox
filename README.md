# MuJoCo Clawblox Worlds

MuJoCo worlds plus runners for driving agents against HTTP-controlled
simulations. Worlds live under `worlds/`, each with its own `world.toml`, system
prompt, API docs, and scene assets.

## Setup

Install the Clawblox CLI from the `dev` branch:

```bash
git clone --branch dev https://github.com/nacloos/clawblox.git
cd clawblox
cargo install --path .
```

Install the Python dependencies:

```bash
uv sync
```

Provide a Claude auth token in a local `.env` file:

```bash
CLAUDE_CODE_OAUTH_TOKEN=...
```

## Running Agents

Run agents on a world:

```bash
uv run run_agent_generations.py --generations 4 --generation-duration 5h --base-port 8185 --tmux-prefix claude-1 --world-dir worlds/mujoco-panda-2
```

Each generation runs in its own tmux session (`<tmux-prefix>-g001`, ...). Use
`tmux ls` to list them and `tmux kill-session -t <session>` to stop one.
Results are written under `worlds/<world>/results/<experiment-id>/`.

### Multiple agents

Use `--agents-per-world` to run several agents in the same world. The world must
have one controllable robot per agent — each agent is assigned its own
controller, and requesting more agents than controllers fails. The two-arm
`mujoco-dual-panda` world supports two agents:

```bash
uv run run_agent_generations.py --agents-per-world 2 --generations 4 --generation-duration 5h --base-port 8185 --tmux-prefix claude-1 --world-dir worlds/mujoco-dual-panda
```

## Replay

Browse a world's recordings and open them in the replay UI:

```bash
clawblox app worlds/mujoco-panda-2 --port 8081
```

Open `http://127.0.0.1:8081` for an interactive 3D replay.
