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

## Robocasa Backend

When the bridge is `robocasa_server.py`, the `SetControl` `ctrl` array is
forwarded to robosuite's composite controller (OSC pose + gripper, plus
mobile-base channels for `PandaOmron`). Its length must equal
`model.action_dim`, not `model.nu`.

### Extra `/observe` fields (robocasa only)

In addition to the raw MuJoCo fields above, the robocasa bridge surfaces
the full robosuite enrichment:

| Field | Type | Meaning |
|-------|------|---------|
| `obs` | object | Full robosuite observation dict — `robot0_proprio-state`, `robot0_eef_pos`/`_quat`, `robot0_base_pos`/`_quat`, `robot0_joint_pos`/`_vel`/`_acc`, `robot0_gripper_qpos`/`_qvel`, plus per-task object keys like `obj_pos`, `obj_to_robot0_eef_pos`, etc. Numpy arrays are JSON lists. |
| `reward` | float | Reward from the most recent `env.step` |
| `done` | bool | Episode-done flag from the most recent step |
| `success` | bool | `env._check_success()` — task-defined success predicate |
| `info` | object | The `info` dict returned by `env.step` |
| `step_count` | int | Number of `env.step` calls since last reset |
| `robot` | object | Pre-extracted robot summary: `eef_pos`, `eef_quat`, `base_pos`, `base_quat`, `base_to_eef_pos`/`_quat`, `joint_pos`/`_vel`, `gripper_qpos`/`_qvel` |

### `GET /spec`

Static info that doesn't change tick-to-tick. Fetch once on connect:

```json
{
  "env_name": "PickPlaceCounterToCabinet",
  "robot_name": "PandaOmron",
  "control_freq": 20,
  "action": {
    "dim": 12,
    "low":  [-1, -1, ..., -1],
    "high": [+1, +1, ..., +1],
    "layout": [
      {"part": "right", "indices": [0, 7],  "controller": "JointPosition", "dim": 7},
      {"part": "base",  "indices": [7, 11], "controller": "MobileBase",    "dim": 4},
      ...
    ]
  },
  "observation": { "keys": {"robot0_eef_pos": [3], "robot0_proprio-state": [68], ...} },
  "model": {"nq": 126, "nv": 123, "nu": 13, "nbody": 384, "njnt": 108},
  "names": {
    "actuators": [...], "joints": [...], "bodies": [...],
    "free_jointed_objects": ["obj_main", "distr_counter_main", "distr_cab_main"]
  }
}
```

`action.layout` is best-effort introspection of robosuite's composite
controller. If the install's controller class names don't match what the
probe expects, the entire action vector falls under a single
`{"part": "composite"}` entry — clients should still treat
`action.dim` / `action.low` / `action.high` as authoritative.
