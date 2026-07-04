# MuJoCo G1 (DDS)

This world uses the stock Unitree G1 29-DOF MuJoCo model at
`models/g1/scene_29dof.xml`, with fixed rubber hands, active 3-DOF waist, the
table and brick scene, and two extra bricks lying on the floor.

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
| publish command | `rt/lowcmd` | `LowCmd_` |
| subscribe state | `rt/lowstate` | `LowState_` |

`LowCmd_.motor_cmd` has 35 slots. This 29-DOF G1 uses indices `0..28`; the
remaining slots are ignored.

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

Full joint names are in `models/g1/g1_joint_index_dds.md` and
`models/g1/g1_29dof.xml`.

## HTTP

HTTP is only for lifecycle and debugging:

```text
GET  /api.md
POST /join
GET  /observe
POST /leave
```

`POST /input` is disabled.
