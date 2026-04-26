# RoboCasa Bridge API

HTTP control surface for `robocasa_bridge/robocasa_server.py`. Wraps a
robosuite kitchen environment and exposes the same `/observe` + `/input` +
session shape as the Panda+cube sandbox documented in
[../API.md](../API.md). This document covers what's *additional* or
*different* on the robocasa side.

## Action Schemas

| Input | Data |
|-------|------|
| SetControl | `{"ctrl": [n actuator values]}` |
| Reset | `{}` |
| SaveState | `{"slot": "<name>"}` |
| RestoreState | `{"slot": "<name>"}` |
| DeleteState | `{"slot": "<name>"}` |
| SetArmJointPos | `{"qpos": [7 joint angles, rad]}` |

`SetControl`'s `ctrl` array is forwarded to robosuite's composite controller
(OSC pose + gripper, plus mobile-base channels for `PandaOmron`). Its
length must equal `model.action_dim`, not `model.nu`.

## Extra `/observe` fields

In addition to the raw MuJoCo fields documented in [../API.md](../API.md),
the robocasa bridge surfaces the full robosuite enrichment:

| Field | Type | Meaning |
|-------|------|---------|
| `obs` | object | Full robosuite observation dict — `robot0_proprio-state`, `robot0_eef_pos`/`_quat`, `robot0_base_pos`/`_quat`, `robot0_joint_pos`/`_vel`/`_acc`, `robot0_gripper_qpos`/`_qvel`, plus per-task object keys like `obj_pos`, `obj_to_robot0_eef_pos`, etc. Numpy arrays are JSON lists. |
| `reward` | float | Reward from the most recent `env.step` |
| `done` | bool | Episode-done flag from the most recent step |
| `success` | bool | `env._check_success()` — task-defined success predicate |
| `info` | object | The `info` dict returned by `env.step` |
| `step_count` | int | Number of `env.step` calls since last reset |
| `robot` | object | Pre-extracted robot summary: `eef_pos`, `eef_quat`, `base_pos`, `base_quat`, `base_to_eef_pos`/`_quat`, `joint_pos`/`_vel`, `gripper_qpos`/`_qvel` |
| `contacts` | array | List of active mujoco contacts this tick (`body1`, `body2`, `pos`, `dist`, `normal_force`). Non-empty when the robot is touching something. |
| `scene_version` | int | Increments whenever the layout changes (Reset). Compare against your cached `/scene` to know when to refetch. |
| `episode` | object | `{reward, base_distance, wall_time, step_count}` since the last Reset. Useful for "are we making progress?" heuristics. |

## `GET /spec`

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

## Extra robocasa-only endpoints

| Method / Path | Purpose |
|---------------|---------|
| `GET /scene` | Static obstacle map: every body with collision geoms (type/pos/quat/size/rbound). Heavy — cache it and refetch only when `scene_version` changes. |
| `GET /body/{name}` | Convenience world pose for a single body: `{name, id, pos, quat}`. 404 if unknown. |
| `GET /states` | List currently saved state slots: `{slots: [{name, saved_at_step, saved_wall_time}]}`. |
| `GET /render?camera=<name>&w=<W>&h=<H>` | PNG snapshot from a named camera (defaults to `robot0_agentview_center`, 320×240, max 1024). Returns 503 if no GL backend is available — set `MUJOCO_GL=egl` (or `osmesa`) before launching. |
| `POST /reach_check` | Body `{"target_pos": [x, y, z]}`. Returns whether the target is within Panda link-sum reach (~0.855 m) of `robot0_link0`. Heuristic only — no IK, no joint-limit check. |

## Save / restore semantics

`SaveState` / `RestoreState` snapshot the full mujoco physics state
(`mjSTATE_INTEGRATION` — qpos/qvel/act/mocap/userdata/ctrl/applied
forces) plus the wrapper's bookkeeping (`current_action`, `step_count`,
`last_obs`, episode metrics). Restoring a slot calls `mj_forward` so
derived kinematics (`xpos`, `xquat`) are valid before the next observe.

A `Reset` between save and restore returns 409: re-randomization can
shift body identities, and silently restoring against a different
layout would corrupt the scene. Re-save after a Reset.

## `SetArmJointPos`

Joint-space teleport for the 7 right-arm joints (`robot0_joint1..7`).
Writes `qpos` directly and zeroes `qvel`, then `mj_forward`. Doesn't
change `current_action` — the next `env.step` will run the OSC
controller against whatever you last `SetControl`ed, so follow up by
sending an OSC target that matches the new EEF (or a no-op zero).
Useful as an escape hatch when OSC has wound the arm into a bad pose.
