# MuJoCo Panda

## Action Schemas

| Input | Data |
|-------|------|
| SetControl | `{"ctrl": [n actuator values]}` |
| Reset | `{}` |

## API Endpoints

The simulation server runs on `http://localhost:8080` by default.

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

## Current Panda Controls

The current robot has `8` actuator controls. You can also read
`model.nu` and `names.actuators` from `/observe` to discover the active
control vector.

| Control | Meaning |
|---------|---------|
| `ctrl[0:7]` | Panda arm joint position targets in radians |
| `ctrl[7]` | Gripper target, `0` closed and `255` open |
