# RoboClaw

Two products in one repo:

1. **MuJoCo Panda sim** (`server.py`, `run_viewer.py`, etc.) — a deterministic
   robotics sandbox that a Claude agent can drive via HTTP.
2. **Reskin pipeline** (`webapp.py`, `pipeline.py`) — turns a short clip into
   N stylistic variations sharing the same motion/geometry. Useful for
   synthetic perception training data.

The two are independent — pick the section you need.

---

## MuJoCo sandbox

Tiny MuJoCo setup for a Franka Emika Panda arm manipulating construction
objects. The Panda model is vendored from DeepMind's MuJoCo Menagerie in
`models/franka_emika_panda`. The custom scene in `models/panda_cube/scene.xml`
adds blocks, bricks, planks, pillars, lighting, and camera around that robot.

### Run

Standalone viewer:

```bash
DISPLAY=:0 uv run --with mujoco python run_viewer.py
```

Real-time simulation server:

```bash
uv run --with mujoco --with fastapi --with uvicorn python server.py
```

Server with attached viewer:

```bash
DISPLAY=:0 uv run --with mujoco --with fastapi --with uvicorn python run_with_viewer.py
```

Dual-arm world (each session is assigned to one robot):

```bash
uv run --with mujoco --with fastapi --with uvicorn python server.py --dual-panda
```

Server endpoints:

```bash
curl http://localhost:8080/observe
curl -X POST http://localhost:8080/input \
  -H 'Content-Type: application/json' \
  -d '{"type":"SetControl","data":{"ctrl":[0,0,0,-1.57079,0,1.57079,-0.7853,255]}}'
```

Headless rendering for WSL:

```bash
MUJOCO_GL=egl uv run --with mujoco python smoke_test.py
```

### Recordings and replay

Recordings are written to `recordings/*.h5` with sibling `*.events.jsonl`
files for inputs and session events. The HDF5 file stores downsampled
preview arrays (`qpos`, `qvel`, `ctrl`) for fast scrubbing plus periodic
full integration-state checkpoints for exact recovery.

Replay with the native viewer:

```bash
DISPLAY=:0 uv run --with mujoco --with h5py python run_replay.py recordings/<file>.h5
```

Replay controls: space toggles play/pause, arrow keys seek by one simulated
second, Home/End jump to the start/end, `[` / `]` adjust speed.

Validate without opening the viewer:

```bash
uv run --with mujoco --with h5py python run_replay.py recordings/<file>.h5 --check
```

### MuJoCo Claude agent (separate from the pipeline agent)

The `agent/` directory hosts a Claude agent that drives the mujoco world via
HTTP. Run one world + one agent + viewer:

```bash
bash agent/launch_multi_claude.sh \
  --world-dir worlds/mujoco-panda \
  --base-port 8085 \
  --tmux-session mujoco-panda-agent \
  --run-id mujoco-panda-test \
  --model claude-opus-4-7 \
  --sandbox \
  --world-server-cmd 'DISPLAY=:0 uv run --with mujoco --with fastapi --with uvicorn python run_with_viewer.py'
```

Two arms, two agents:

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

Attach: `tmux attach -t mujoco-panda-agent`

---

## Reskin pipeline

Turn a short clip into N stylistic variations that share its motion and
geometry — synthetic perception data with identical action but radically
different lighting, textures, environments, and floors.

### Pipeline at a glance

```
source.mp4
  └─> Pegasus 1.5 (≤5s segments)
        └─> pick one segment
              └─> GPT-5.4-nano analyzes the first frame
                    └─> 8 reskin prompts (you keep 3, regenerate boring ones)
                          └─> GPT-5.4-image-2 paints each prompt on the frame
                                └─> {Lucy Edit 2.1, Runway Aleph} propagates
                                    each painted frame across the original motion
                                      └─> 3 mp4s, same length as input segment
```

### Two ways to run it

**Manual mode** — you walk through each gate yourself. Default.

**Agent mode** — a Claude Code agent reads `SOUL.md` + `PIPELINE_AGENT.md`
and drives the whole pipeline autonomously, posting decisions to the same
endpoints the UI uses. Auth is via your **Claude Code subscription** (long
lived OAuth token), so the agent's reasoning costs nothing per token — you
only pay for the external APIs the pipeline already calls (TwelveLabs,
OpenRouter, Decart/Runway). The agent's audit trail lives at
`runs/<name>/agent_decisions.log` and surfaces in the UI under
"agent reasoning · decisions."

### Two render models

Pick at upload time, locked for the run:

- **Lucy · Decart** (`lucy-2.1`) — fast, established, no per-call duration cap.
- **Runway · Aleph** (`gen4_aleph`) — preserves motion well, novel reskins.
  Caps each output at 5 seconds, which is why segments are constrained to
  ≤5s upstream.

### Run the web UI (recommended)

```bash
uv run --with fastapi --with 'uvicorn[standard]' --with python-multipart \
    --with httpx --with runwayml python webapp.py
```

Open http://localhost:8000. The default landing screen shows the model
picker (Lucy / Runway), dropzone, and mode toggle (manual / agent).

Drag a clip onto the dropzone, then either drive the gates yourself or
let the agent do it. Either way the canvas shows segments → pick → analyze
→ count (always 3 in agent mode) → approve prompts → image variations →
final render.

### Run the CLI

```bash
uv run --with httpx --with runwayml python pipeline.py --input sim.mp4
```

Resume any in-flight run with `--run-dir runs/<existing>`; completed
stages are skipped automatically.

### Capturing a clip from the sim

On macOS the easy path is **Cmd+Shift+5 → Record Selected Portion**, drag
the MuJoCo window, click Record, stop after **≥ 4 seconds** (TwelveLabs
minimum). The recording lands as a `.mov` on your Desktop.

CLI alternative (full display, edit input index for your setup):

```bash
ffmpeg -f avfoundation -i "1" -t 8 -an -c:v libx264 -pix_fmt yuv420p sim.mp4
```

Audio is stripped on ingest regardless.

### Where the artifacts land

Every run gets a timestamped directory under `runs/` (gitignored):

```
runs/<YYYYMMDD-HHMMSS>_<clipname>/
  00_input/         source.mp4, metadata.json (model, dims, duration)
  01_twelvelabs/    asset.json, segments.json     (≤5s clips)
  02_segment/       chosen.json, segment.mp4, first_frame.png
  03_analysis/      prompts.json                  (invariants + axes + 8 prompts)
  04_variations/    var_NN/{prompt.txt, image.png}
  05_lucy_edit/     var_NN/output.mp4             ← Lucy outputs (if model=lucy)
  05_runway/        var_NN/output.mp4             ← Runway outputs (if model=runway)
  agent.log         live event stream from the Claude agent
  agent.events.jsonl    raw stream-json events (debugging)
  agent_decisions.log   one-line audit per gate decision
  pipeline.log      pipeline subprocess output
```

### Required keys

Create `.env` at the project root (gitignored) with whichever of these you
need for the path you'll use:

```
# always required
OPENROUTER_API_KEY=sk-or-...
TWELVELABS_API_KEY=tlk_...

# render model — pick the one(s) you'll actually run
DECART_API_KEY=dct_...                    # for Lucy
RUNWAYML_API_SECRET=key_...               # for Runway Aleph

# only required for agent mode
CLAUDE_CODE_OAUTH_TOKEN=sk-ant-oat01-...  # `claude setup-token`
```

The pipeline only enforces the renderer key for the model the run was
created with — i.e. you don't need a Decart key to run a Runway-only
session, and vice versa.

### Notes

- **Minimum clip length: 4s** — TwelveLabs' floor. The pipeline errors out
  cleanly before any API call if the input is too short.
- **Segment cap: 5s** — Pegasus is asked for ≈5s segments and any over-cap
  scenes get post-split. This is what lets a single Aleph call cover the
  whole segment 1:1 without truncation.
- **Concurrency** — image gen and final render run with bounded concurrency
  (4 in parallel) plus exponential backoff on 429/5xx.
- **Caching** — every stage is keyed on disk; resuming a run skips work
  that's already done. Only re-renders downstream stages when their inputs
  change.

### The agent's identity

`SOUL.md` is the durable identity ("you are an autonomous robotics data
engineer; optimize for signal, diversity, transferability"); `PIPELINE_AGENT.md`
is the operational manual (gate rubrics, endpoints, decision log format).
Both get prepended to the agent's prompt by `pipeline_agent.sh` on launch.
Edit either to reshape the agent's behavior; no code change needed.
