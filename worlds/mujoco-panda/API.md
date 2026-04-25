# MuJoCo Panda

## Action Schemas

| Input | Data |
|-------|------|
| SetControl | `{"ctrl": [n actuator values]}` |
| Reset | `{}` |

## API Endpoints

The simulation server runs on `http://localhost:8080` by default.

### Join Session

`POST /join?name=MyAgent`

Returns a lightweight session token:

```json
{
  "session": "session-token-uuid",
  "agent_id": "agent-uuid",
  "name": "MyAgent"
}
```

Send the token back as `X-Session: <session-token>` on later requests.
The simulator currently accepts unauthenticated control requests too; the
session is used so generic agents can keep the same convention across worlds.

### Leave Session

`POST /leave`

Optional header: `X-Session: <session-token>`.

### Send Input

`POST /input`

Body: `{"type":"<ActionName>","data":{...}}`

Returns an observation payload.

### Get Observation

`GET /observe`

Returns raw MuJoCo state:

```json
{
  "time": 0.0,
  "qpos": [],
  "qvel": [],
  "ctrl": [],
  "model": {"nq": 0, "nv": 0, "nu": 0},
  "names": {
    "actuators": [],
    "joints": [],
    "bodies": []
  },
  "objects": [
    {
      "name": "block_red",
      "position": [x, y, z],
      "quaternion": [w, x, y, z]
    }
  ],
  "blocks": []
}
```

`objects` contains construction objects such as blocks, bricks, planks,
and pillars. `blocks` is currently an alias for compatibility with earlier
agents.

### Get Agent API

`GET /api.md`

Returns this file as markdown text.

`GET /skill.md` is an alias for compatibility with older agents.

### Get Snapshot

`GET /snapshot`

Returns a raw replay checkpoint containing `time`, `qpos`, `qvel`, `ctrl`, and
active session metadata.

## Current Panda Controls

The current robot has `8` actuator controls. You can also read
`model.nu` and `names.actuators` from `/observe` to discover the active
control vector.

| Control | Meaning |
|---------|---------|
| `ctrl[0:7]` | Panda arm joint position targets in radians |
| `ctrl[7]` | Gripper target, `0` closed and `255` open |
