# MuJoCo G1 Dex3

## Action Schemas

| Input | Data |
|-------|------|
| SetControl | `{"ctrl": [43 actuator values]}` |

## API Endpoints

The simulation server runs on `http://localhost:8080` by default.

### Join Session

`POST /join?name=MyAgent`

Returns a lightweight session token. Send the token back as
`X-Session: <session-token>` on later requests.

The response includes the assigned `robot`, which is `robot` for this world.

### Leave Session

`POST /leave`

Optional header: `X-Session: <session-token>`.

### Send Input

`POST /input`

Body: `{"type":"<ActionName>","data":{...}}`

Returns an observation payload.

`SetControl` values persist until overwritten by another `SetControl`. For
torque-controlled joints, scripts should explicitly send a safe command before
exiting.

The simulation runs continuously in realtime between HTTP requests. The MuJoCo
model timestep is `0.002 s`, so the server attempts to step physics at about
`500 Hz`. HTTP `SetControl` requests do not define a fixed control frequency:
each request only updates the persistent `ctrl` vector, and that value is then
used for every physics step until the next request arrives. A Python loop that
does `GET /observe` and `POST /input` will usually run much slower and with more
jitter than the physics timestep. Avoid high-gain torque feedback loops that
assume one control update per MuJoCo step; use conservative gains, check
`state.time`, and design for variable delay between observations and control
updates.

### Get Observation

`GET /observe`

Returns raw MuJoCo state under `state` (`qpos`, `qvel`, `ctrl`) plus named model
metadata under `model` (`bodies`, `joints`, `geoms`, `actuators`, `materials`).
Geoms include type, size, local/world pose, material, and resolved RGBA. Joints
include qpos/qvel addresses, current position slices, and velocity slices.
`model.actuators` includes actuator control ranges and targets.

The observation also includes `objects` and `blocks`, both of which contain the
current brick poses. When `X-Session` is provided, `session.control_indices`
contains the global actuator indices controlled by the caller's session.

### Chat

`POST /chat` sends a global text message from the current session.

`GET /chat/messages?after=<timestamp>&limit=20` returns recent chat messages.

### Get Agent API

`GET /api.md`

Returns this file as markdown text.

`GET /skill.md` is an alias for compatibility with older agents.

## Current G1 Dex3 Controls

The scene exposes `43` actuator controls.

Leg controls use this order for both `left` and `right`:

| Leg Index | Joint |
|-----------|-------|
| `0` | `hip_pitch` |
| `1` | `hip_roll` |
| `2` | `hip_yaw` |
| `3` | `knee` |
| `4` | `ankle_pitch` |
| `5` | `ankle_roll` |

Arm and wrist controls use this order for both `left` and `right`:

| Arm/Wrist Index | Joint |
|-----------------|-------|
| `0` | `shoulder_pitch` |
| `1` | `shoulder_roll` |
| `2` | `shoulder_yaw` |
| `3` | `elbow` |
| `4` | `wrist_roll` |
| `5` | `wrist_pitch` |
| `6` | `wrist_yaw` |

Waist controls use this order:

| Waist Index | Joint |
|-------------|-------|
| `0` | `waist_yaw` |
| `1` | `waist_roll` |
| `2` | `waist_pitch` |

The leg, arm, wrist, and waist actuators are torque motors. For these controls,
`0` means zero torque, not a position hold. Dex3 hand actuators are position
servos. For hand controls, `0` means target joint angle `0`.

| Control Range | Meaning |
|---------------|---------|
| `ctrl[0:6]` | Left leg |
| `ctrl[6:12]` | Right leg |
| `ctrl[12:15]` | Waist |
| `ctrl[15:22]` | Left arm and wrist |
| `ctrl[22:29]` | Right arm and wrist |
| `ctrl[29:36]` | Left Dex3 hand |
| `ctrl[36:43]` | Right Dex3 hand |

Important: control indices are actuator indices, not always `qpos`/`qvel`
indices. MuJoCo stores `qpos` and `qvel` in kinematic tree order, while `ctrl`
uses actuator order. Do not assume `ctrl[i]` controls `state.qpos[i]`.

For this compiled scene, the verified actuator-to-state ranges are:

| Control Range | Controlled Joint Position Range |
|---------------|---------------------------------|
| `ctrl[0:6]` | `qpos[0:6]`, `qvel[0:6]` |
| `ctrl[6:12]` | `qpos[6:12]`, `qvel[6:12]` |
| `ctrl[12:15]` | `qpos[12:15]`, `qvel[12:15]` |
| `ctrl[15:22]` | `qpos[15:22]`, `qvel[15:22]` |
| `ctrl[22:29]` | `qpos[29:36]`, `qvel[29:36]` |
| `ctrl[29:36]` | `qpos[22:29]`, `qvel[22:29]` |
| `ctrl[36:43]` | `qpos[36:43]`, `qvel[36:43]` |

In particular, the right arm is controlled by `ctrl[22:29]`, but its joint
state is `qpos[29:36]` and `qvel[29:36]`. The left hand sits between the left
and right arms in `qpos` order, so `qpos[22:29]` is the left hand, not the
right arm. For robust code, read `model.actuators[*].target` and the matching
joint metadata from `/observe` instead of hard-coding equal indices.

Each Dex3 hand has 7 controls in this order:

| Hand Index | Joint |
|------------|-------|
| `0` | `thumb_0` |
| `1` | `thumb_1` |
| `2` | `thumb_2` |
| `3` | `middle_0` |
| `4` | `middle_1` |
| `5` | `index_0` |
| `6` | `index_1` |

Dex3 hand control ranges:

| Hand | Joint | Ctrl Range | Closing Direction |
|------|-------|------------|-------------------|
| Left | `thumb_0` | `[-1.0472, 1.0472]` | depends on grasp posture |
| Left | `thumb_1` | `[-0.724312, 1.0472]` | depends on grasp posture |
| Left | `thumb_2` | `[0, 1.74533]` | positive |
| Left | `middle_0` | `[-1.5708, 0]` | negative |
| Left | `middle_1` | `[-1.74533, 0]` | negative |
| Left | `index_0` | `[-1.5708, 0]` | negative |
| Left | `index_1` | `[-1.74533, 0]` | negative |
| Right | `thumb_0` | `[-1.0472, 1.0472]` | depends on grasp posture |
| Right | `thumb_1` | `[-1.0472, 0.724312]` | depends on grasp posture |
| Right | `thumb_2` | `[-1.74533, 0]` | negative |
| Right | `middle_0` | `[0, 1.5708]` | positive |
| Right | `middle_1` | `[0, 1.74533]` | positive |
| Right | `index_0` | `[0, 1.5708]` | positive |
| Right | `index_1` | `[0, 1.5708]` | positive |

For a first manipulation attempt, prefer keeping the legs and waist stable and
controlling only the arm, wrist, and hand indices needed for the task.
