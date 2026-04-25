# RoboCasa Bridge

HTTP control surface for [robocasa.ai](https://robocasa.ai) kitchen tasks.
Wraps a robosuite environment (Franka-class arm on a mobile base) and
exposes the same `/observe` + `/input` shape as the Panda+cube sandbox in
this repo, so a single agent can target either backend.

## Install

```bash
./robocasa_bridge/install.sh
```

That single command will:

1. Verify `uv` is on `PATH` (and print the install hint if it isn't).
2. Run `uv sync` to install everything in `pyproject.toml`, including
   robosuite (from git) and robocasa (vendored under `vendor/robocasa`).
3. Create `vendor/robocasa/robocasa/macros_private.py` if it doesn't
   already exist (robocasa's upstream `setup_macros` script is interactive
   — `install.sh` does the equivalent without prompting).
4. Download the ~10 GB kitchen asset bundle (textures, fixtures, objaverse
   objects, etc.) into `vendor/robocasa/robocasa/models/assets/` if it
   isn't already there.

Flags:

```bash
./robocasa_bridge/install.sh --skip-assets   # skip the multi-GB download (most scenes will fail to load)
./robocasa_bridge/install.sh --force-assets  # re-download even if assets are already present
./robocasa_bridge/install.sh --help
```

The asset step is gated on whether `models/assets/textures` and
`models/assets/objects/objaverse` are non-empty, so re-running the
installer after a successful install is a fast no-op.

## Run

**Headless server** — HTTP only, no window, runs anywhere:

```bash
uv run python robocasa_bridge/robocasa_server.py
```

**Server with viewer** — opens robosuite's native `mjviewer` window. On
WSLg you'll need `DISPLAY` exported (`:0` is the usual value):

```bash
DISPLAY=:0 uv run python robocasa_bridge/run_robocasa_viewer.py
```

The viewer uses robosuite's `mjviewer` rather than `mujoco.viewer.launch_passive`
because the kitchen scene is too heavy for `launch_passive` on WSLg — it
silently dies during texture upload.

Both entry points listen on `http://127.0.0.1:8080`.

### Picking a task or robot

Set env vars before launching:

```bash
ROBOCASA_ENV=PickPlaceCounterToCabinet \
ROBOCASA_ROBOT=PandaOmron \
ROBOCASA_CONTROL_FREQ=20 \
  uv run python robocasa_bridge/robocasa_server.py
```

Defaults are `PickPlaceCounterToCabinet` / `PandaOmron` / `20 Hz`. Any
robosuite-registered env name and Franka-class robot will work; robocasa
adds the kitchen tasks to that registry on import.

## HTTP API

Full schema lives in [../API.md](../API.md). The two endpoints you'll
touch most:

```bash
# Observation (raw mujoco state + full robosuite obs dict + robot summary)
curl http://localhost:8080/observe

# Static spec (action layout, control freq, observation key shapes) — fetch once
curl http://localhost:8080/spec

# Apply a control (length must match spec.action.dim)
curl -X POST http://localhost:8080/input \
  -H 'Content-Type: application/json' \
  -d '{"type":"SetControl","data":{"ctrl":[0,0,0, 0,0,0,  0,0,  0,0,0,  0]}}'

# Reset the episode
curl -X POST http://localhost:8080/input \
  -H 'Content-Type: application/json' \
  -d '{"type":"Reset","data":{}}'
```

`Reset` is routed through the simulation step loop (the main thread owns
the env), so concurrent HTTP requests don't race with `env.step` /
`env.render`.

### Action vector — `PickPlaceCounterToCabinet` + `PandaOmron`

The composite controller exposes a 12-dim action. `make_ctrl()` in any
client helper just slices into this layout:

| Indices  | Part    | Controller                          | Meaning |
|----------|---------|-------------------------------------|---------|
| `0:6`    | right arm  | OSC pose                         | dx, dy, dz, drx, dry, drz (in robot base frame) |
| `6:8`    | gripper    | SimpleGrip                       | both fingers; -1 open, +1 close |
| `8:11`   | base       | LegacyMobileBaseJointVelocity    | side-vel, yaw-vel, torso (see note) |
| `11`     | torso      | JointPosition                    | (no-op on this build — torso lives in the base channels) |

The `/spec` endpoint returns the layout that robosuite's introspection
reports; the empirical mapping above was determined by probing one
actuator at a time. Always treat `spec.action.dim` /
`spec.action.low` / `spec.action.high` as authoritative for sizing and
clipping.

## Python client example

```python
import json, urllib.request

URL = "http://127.0.0.1:8080"

def post(path, body):
    req = urllib.request.Request(
        f"{URL}{path}",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=10) as r:
        return json.loads(r.read())

def get(path):
    with urllib.request.urlopen(f"{URL}{path}", timeout=5) as r:
        return json.loads(r.read())

spec = get("/spec")
dim = spec["action"]["dim"]

post("/input", {"type": "Reset", "data": {}})
obs = get("/observe")
print("eef:", obs["robot"]["eef_pos"], "obj:", obs["obs"]["obj_pos"])

# Open gripper, hold for one tick
ctrl = [0.0] * dim
ctrl[6] = ctrl[7] = -1.0
post("/input", {"type": "SetControl", "data": {"ctrl": ctrl}})
```

## Useful observation keys (PandaOmron + PickPlace)

`/observe` includes the raw MuJoCo state, the full robosuite obs dict
under `obs`, and a `robot` summary with the most-used fields
pre-extracted. Common fields:

| Path                              | Meaning |
|-----------------------------------|---------|
| `robot.eef_pos`, `robot.eef_quat` | World end-effector pose (quat is `[x, y, z, w]`) |
| `robot.base_pos`, `robot.base_quat` | World mobile-base pose |
| `robot.gripper_qpos`              | Two finger joint positions |
| `obs.obj_pos`, `obs.obj_to_robot0_eef_pos` | Object pose, EE-relative |
| `obs.distr_cab_pos`, `obs.distr_cab_to_robot0_eef_pos` | Placement target (cabinet) |
| `success`                         | Task-defined success predicate |
| `reward`, `done`, `step_count`    | Per-step training signals |

## Troubleshooting

* **`No module named 'robocasa'`** — re-run `./robocasa_bridge/install.sh`. The vendored
  package is installed editable, so a partial `uv sync` may have skipped
  it.
* **Viewer is blank or crashes on WSL** — make sure `DISPLAY` is set
  (`echo $DISPLAY`) and that you're running `run_robocasa_viewer.py`,
  not `mujoco.viewer.launch_passive`. `launch_passive` doesn't survive
  the kitchen scene's geometry upload on WSLg.
* **Asset download dies partway** — re-run `./robocasa_bridge/install.sh --force-assets`.
  The download script overwrites partial extractions.
* **`SetControl` returns 422** — your `ctrl` array length doesn't match
  `spec.action.dim`. Fetch `/spec` once and use that as the length.
* **Env feels frozen after a long run** — process drift from a stuck
  controller; `POST /input {"type":"Reset"}` returns the env to its
  initial state without restarting the server.
