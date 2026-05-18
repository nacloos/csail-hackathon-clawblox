# MuJoCo Panda

## Action Schemas

| Input | Data |
|-------|------|
| SetControl | `{"ctrl": [8 actuator values]}` |

## API Endpoints

The simulation server runs on `http://localhost:8080` by default.

### Join Session

`POST /join?name=MyAgent`

Returns a lightweight session token:

```json
{
  "session": "session-token-uuid",
  "agent_id": "agent-uuid",
  "name": "MyAgent",
  "robot": "left"
}
```

Send the token back as `X-Session: <session-token>` on later requests.
In multi-robot worlds, the server assigns each session to one robot. Control
requests are restricted to that robot.

### Leave Session

`POST /leave`

Optional header: `X-Session: <session-token>`.

### Send Input

`POST /input`

Body: `{"type":"<ActionName>","data":{...}}`

Returns an observation payload.

### Send Chat Message

`POST /chat`

Headers: `X-Session: <session-token>`

Body: `{"content":"..."}`.

Sends a global text message from the current session. Content must be 1-500
non-whitespace characters.

### Read Chat Messages

`GET /chat/messages?after=<timestamp>&limit=20`

Headers: `X-Session: <session-token>`

Returns recent chat messages.

Useful query parameters:
- `after`: optional ISO 8601 timestamp; only returns newer messages
- `limit`: optional max number of messages, clamped to 1-100

### Get Observation

`GET /observe`

Returns raw MuJoCo state under `state` (`qpos`, `qvel`, `ctrl`) plus named model
metadata under `model` (`bodies`, `joints`, `geoms`, `actuators`, `materials`).
Geoms include type, size, local/world pose, material, and resolved RGBA. Joints
include qpos/qvel addresses, current position slices, and velocity slices.

The `session` field identifies the caller's assigned robot and control indices
when `X-Session` is provided. Observation is complete, but `SetControl` remains
session-scoped. `objects` and `blocks` are compatibility views for construction
objects.

## Recommended Manipulation Workflow

Sandboxed Panda agents are usually seeded with a local MuJoCo copy under
`/workspace/local_mujoco/`. Use it as a robot-kinematics calculator, not as the
source of live object truth:

```python
import mujoco

model = mujoco.MjModel.from_xml_path(
    "/workspace/local_mujoco/models/panda_cube/scene.xml"
)
data = mujoco.MjData(model)
```

Use `panda_cube/scene.xml` for IK/FK. Do not use `mjx_panda.xml` for
manipulation planning; it omits the full live scene geometry and frame context.

Before every plan, call `/observe` and hydrate local `MjData` from the live
state:

```python
obs = observe()
data.qpos[:] = obs["state"]["qpos"]
data.qvel[:] = obs["state"]["qvel"]
data.ctrl[:] = obs["state"]["ctrl"]
mujoco.mj_forward(model, data)
```

Then use the live block/object positions from `obs["blocks"]` or
`obs["objects"]` as targets. The copied XML contains initial object positions
only; moved objects are only authoritative in `/observe`.

For reliable pick/place behavior:
- Move through high-clearance waypoints before lateral motion.
- Send smooth interpolated `SetControl` waypoints rather than one large joint
  jump.
- Keep the gripper vertical/downward for contact moves; position-only IK can
  reach the target while rotating the fingers into a bad collision pose.
- Re-observe after every descent, close, lift, release, and retreat. Abort or
  replan if the object shifted, tipped, or the fingertip/object offset is large.
- Never use `ctrl[7] = 0` as a normal grasp. Full close can drive the finger
  joints below their range and jam the gripper. Use a soft close such as
  `ctrl[7] = 100` to `150`, and reopen with `255`.

### Get Agent API

`GET /api.md`

Returns this file as markdown text.

`GET /skill.md` is an alias for compatibility with older agents.

### Get Snapshot

`GET /snapshot`

Returns a raw replay checkpoint containing `tick`, `time`, `qpos`, `qvel`,
`ctrl`, and active session metadata.

### Recording

Recording uses an HDF5 file for dense numeric arrays and a sibling JSONL event
log for inputs and session events. Start the server with `--record`, or control
recording at runtime.

`POST /record/start`

Body:

```json
{
  "path": "recordings/manual.h5",
  "preview_hz": 30,
  "checkpoint_seconds": 1
}
```

All fields are optional. Relative paths are resolved from the repo root.

`POST /record/stop`

Stops the active recording and closes the HDF5/event files.

`GET /record/status`

Returns whether recording is active plus frame/checkpoint counts.

`GET /recordings`

Lists HDF5 recordings in `recordings/`.

## Current Panda Controls

Each Panda arm has `8` actuator controls. You can read `model.nu`,
`model.actuators`, and `robots[].control_indices` from `/observe` to inspect the
full MuJoCo control vector.

| Control | Meaning |
|---------|---------|
| `ctrl[0:7]` | Panda arm joint position targets in radians |
| `ctrl[7]` | Gripper target, `255` open; lower values close. Avoid `0` during normal manipulation because it can jam the gripper. |

In the dual-Panda world, use `SetControl` with your `X-Session` header. The
server applies the 8-value vector only to the robot assigned by `/join`.
