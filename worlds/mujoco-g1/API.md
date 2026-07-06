# MuJoCo G1 Dex3 (DDS)

This world uses the stock Unitree G1 29-DOF MuJoCo body with official Dex3-1
hand meshes and joint ordering at `models/g1/scene_dex3.xml`, with active
3-DOF waist, the table and brick scene, and two extra bricks lying on the
floor. The untouched upstream body model remains at `models/g1/g1_29dof.xml`;
`models/g1/g1_29dof_dex3.xml` is the Dex3-1 hand integration.

Control is over Unitree SDK2 DDS, not HTTP. The server and model source are
available in the workspace for inspection.

## DDS

Use `/sandbox-deps/envs/g1/bin/python` for DDS control scripts.

```python
import os
from unitree_sdk2py.core.channel import ChannelFactoryInitialize

ChannelFactoryInitialize(
    int(os.environ["WORLD_DDS_DOMAIN_ID"]),
    os.environ.get("WORLD_DDS_INTERFACE", "lo"),
)
```

| Direction | Topic | Message |
| --- | --- | --- |
| publish body command | `rt/lowcmd` | `LowCmd_` |
| subscribe body state | `rt/lowstate` | `LowState_` |
| publish left hand command | `rt/dex3/left/cmd` | `HandCmd_` |
| subscribe left hand state | `rt/dex3/left/state` | `HandState_` |
| publish right hand command | `rt/dex3/right/cmd` | `HandCmd_` |
| subscribe right hand state | `rt/dex3/right/state` | `HandState_` |

`LowCmd_.motor_cmd` has 35 slots. This 29-DOF G1 body uses indices `0..28`;
the remaining body slots are ignored. The Dex3-1 hands are not controlled via
`LowCmd_`; use the `HandCmd_` topics.

For each motor:

```text
tau_applied = kp * (q_target - q_measured)
            + kd * (dq_target - dq_measured)
            + tau_ff
```

Use `mode = 1` for enabled motor commands. Physics runs at 500 Hz.

## Motor Order

```text
0-5    left leg    hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll
6-11   right leg   hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll
12-14  waist       yaw, roll, pitch
15-21  left arm    shoulder_pitch, shoulder_roll, shoulder_yaw, elbow,
                   wrist_roll, wrist_pitch, wrist_yaw
22-28  right arm   shoulder_pitch, shoulder_roll, shoulder_yaw, elbow,
                   wrist_roll, wrist_pitch, wrist_yaw
```

## Dex3-1 Hand Order

Each `HandCmd_.motor_cmd` has 7 slots in the official Unitree Dex3-1 order:

```text
0 thumb_0
1 thumb_1
2 thumb_2
3 index_0
4 index_1
5 middle_0
6 middle_1
```

The hand actuators are MuJoCo position actuators. Command joint targets in
radians. Left index and middle finger closing uses negative angles; right index
and middle finger closing uses positive angles.

Full joint names are in `models/g1/g1_joint_index_dds.md` and
`models/g1/g1_29dof_dex3.xml`.

## HTTP

HTTP is only for lifecycle and debugging:

```text
GET  /api.md
POST /join
GET  /observe
POST /leave
```

`POST /input` is disabled.
