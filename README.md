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

<<<<<<< Updated upstream
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
=======
## From simulator to synthetic data

Use the MuJoCo sim as a deterministic source clip, then push it through the
reskin pipeline (`pipeline.py` / `webapp.py`) to generate N styled variants
of the same motion — synthetic perception data with identical geometry but
different lighting / textures / environments.

### 1. Record a clip from the viewer

Launch the sim, set up the camera you want, then capture a window recording.

```bash
uv run --with mujoco python run_viewer.py     # macOS: prefix with mjpython
```

On macOS the easy path is **Cmd+Shift+5 → Record Selected Portion**, drag the
MuJoCo window, click Record, stop after **≥ 4 seconds** (TwelveLabs minimum
clip length). The recording lands as a `.mov` on your Desktop.

CLI alternative (full display, edit input index for your setup):

```bash
ffmpeg -f avfoundation -i "1" -t 8 -an -c:v libx264 -pix_fmt yuv420p sim.mp4
```

Audio is stripped on ingest regardless, so don't worry about muting.

### 2. Drop the clip into the pipeline

**Web UI** (recommended — interactive segment pick + prompt approval on a
node canvas):

```bash
uv run --with fastapi --with 'uvicorn[standard]' --with python-multipart \
    --with httpx python webapp.py
```

Open http://localhost:8000, drag the clip onto the dropzone, then walk
through the canvas: Pegasus 1.5 splits the clip into scenes → pick one →
GPT-5.4-nano writes 8 reskin prompts → choose how many to render (1/3/8)
→ edit/regen/approve → GPT-5.4-image-2 paints each on top of your chosen
frame → Decart Lucy Edit 2.1 propagates each variant across the original
motion.

**CLI** (same flow, terminal prompts at each gate):

```bash
uv run --with httpx python pipeline.py --input sim.mp4
```

Resume any in-flight run by passing `--run-dir runs/<existing>`; completed
stages are skipped (the web UI does this automatically).

### 3. Where the artifacts land

Every run gets a timestamped directory under `runs/` (gitignored):

```
runs/<YYYYMMDD-HHMMSS>_<clipname>/
  00_input/        source.mp4              (audio stripped at the door)
  01_twelvelabs/   asset.json, segments.json
  02_segment/      chosen.json, segment.mp4, first_frame.png
  03_analysis/     prompts.json            (variation axes + 8 prompts)
  04_variations/   var_NN/{prompt.txt, image.png}
  05_lucy_edit/    var_NN/output.mp4       ← synthetic clips
```

The final variants for downstream training live at
`runs/.../05_lucy_edit/var_NN/output.mp4`. Intermediate frames and prompts
are preserved so you can re-render only the downstream stages of an
existing run.

### Required keys

Create `.env` at the project root (gitignored) with:

```
OPENROUTER_API_KEY=sk-or-...
TWELVELABS_API_KEY=tlk_...
DECART_API_KEY=dct_...
```

### Notes

- **Minimum clip length: 4 s** — Pegasus 1.5 will reject anything shorter.
  The pipeline errors out cleanly before any API call if the input is too
  short.
- **Cost rough estimate**, ~5 s segment × 8 variations: ≈ $2 (TwelveLabs
  analyze + 8× `gpt-5.4-image-2` + 8× Lucy 720p).
- Image gen and Lucy Edit run with bounded concurrency (4 in parallel) and
  exponential backoff on 429 / 5xx, so ramping the variation count to 8
  stays inside per-key rate limits.

>>>>>>> Stashed changes
