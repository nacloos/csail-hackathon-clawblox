# MuJoCo G1 (DDS)

You control a free-base Unitree G1 humanoid in MuJoCo over the **Unitree SDK2
DDS interface** — the same wire protocol as the physical robot. A controller
written against this interface can drive the simulator or the real G1 unchanged.

Robot control is **direct DDS**, not HTTP. The HTTP server is only for docs,
session lifecycle, and read-only debugging.

## Control Interface (DDS)

CycloneDDS, carrying `unitree_hg` IDL messages:

| Direction | Topic | Message |
|-----------|-------|---------|
| command (you publish) | `rt/lowcmd` | `unitree_hg::msg::dds_::LowCmd_` |
| state (you subscribe) | `rt/lowstate` | `unitree_hg::msg::dds_::LowState_` |

The bus coordinates for your run are provided in the environment and echoed by
`POST /join` and `GET /observe`:

```text
WORLD_DDS_DOMAIN_ID    the DDS domain id for your run
WORLD_DDS_INTERFACE    the network interface the bus binds (e.g. lo)
```

### Runtime for DDS control

Run your DDS control scripts with the Python that has CycloneDDS and the
`unitree_hg` IDL installed:

```text
/sandbox-deps/envs/unitree-mujoco/bin/python
```

The default `python3` / `/sandbox-deps/bin/python` has `mujoco` (use it for
kinematics and model inspection) but **not** CycloneDDS, so `import cyclonedds`
fails there. `unitree_sdk2py.idl.unitree_hg.msg.dds_` (`LowCmd_`, `LowState_`,
`MotorCmd_`) is available in the env python above. Read the DDS coordinates from
the environment (`WORLD_DDS_DOMAIN_ID` / `WORLD_DDS_INTERFACE`) and initialize
your participant on that domain.

### Discovery config (required)

`lo` is not multicast-capable, so you must configure **unicast discovery** or you
will never see the world's participant. Set `CYCLONEDDS_URI` before creating any
participant, to exactly what the world uses:

```xml
<CycloneDDS><Domain><General>
  <Interfaces><NetworkInterface name='lo'/></Interfaces>
  <AllowMulticast>false</AllowMulticast>
</General><Discovery>
  <ParticipantIndex>auto</ParticipantIndex>
  <Peers><Peer address='localhost'/></Peers>
</Discovery></Domain></CycloneDDS>
```

(Equivalently, `unitree_sdk2py.core.channel.ChannelFactoryInitialize(domain,
interface)` sets up compatible loopback discovery for you.) The world and your
controller run the **same** CycloneDDS (0.10.2) and IDL, so types match — mixing
CycloneDDS versions crashes DDS type discovery.

Initialize your DDS participant on that domain and interface. Your run has its
own private network namespace, so the bus is yours alone.

## Command: `LowCmd_`

`LowCmd_.motor_cmd` is a fixed array of 35 motor commands. For this 29-DOF G1,
indices `0..28` are active (leg, waist, arm joints); the tail is ignored. Each
`MotorCmd_` is an onboard PD setpoint applied every physics step:

```text
tau_applied = kp * (q - q_measured) + kd * (dq - dq_measured) + tau
```

| Field | Meaning |
|-------|---------|
| `q`   | target joint position (rad) |
| `dq`  | target joint velocity (rad/s) |
| `kp`  | position gain |
| `kd`  | velocity gain |
| `tau` | feedforward torque (N·m) |
| `mode`| set to `1` (enabled) |

Set `kp`/`kd` to `0` and use `tau` alone for pure torque control. The base is
free: the robot balances only through the torques you command and ground
contact.

## State: `LowState_`

Published every physics step (500 Hz):

- `motor_state[i]`: `q`, `dq`, `tau_est` per joint (indices `0..28`).
- `imu_state`: `quaternion` (w, x, y, z) and `gyroscope` of the floating base.

## Motor Order

`motor_cmd[i]` / `motor_state[i]` follow the Unitree G1 29-DOF IDL order:

```text
0-5    left leg    hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll
6-11   right leg   hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll
12-14  waist       yaw, roll, pitch
15-21  left arm    shoulder_pitch, shoulder_roll, shoulder_yaw, elbow,
                   wrist_roll, wrist_pitch, wrist_yaw
22-28  right arm   shoulder_pitch, shoulder_roll, shoulder_yaw, elbow,
                   wrist_roll, wrist_pitch, wrist_yaw
```

See `models/g1/g1_joint_index_dds.md` for the full index table.

## Control Rate

Physics runs at 500 Hz (`0.002 s` timestep). Publish `rt/lowcmd` at up to that
rate; the most recent command is applied each step. Run a persistent controller
process rather than issuing one command per shell call.

## HTTP (lifecycle only)

```text
GET  /api.md      this document
POST /join        returns a session token and the DDS coordinates
GET  /observe     read-only qpos/qvel/ctrl + dds info (debug)
POST /leave       end the session
```

`POST /input` is disabled: control is DDS.
