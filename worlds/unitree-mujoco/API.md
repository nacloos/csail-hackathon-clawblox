# Unitree MuJoCo

Unitree G1 MuJoCo simulation with the Unitree SDK2 DDS interface.

Robot control is direct DDS. The Clawblox HTTP server only exposes lifecycle, status, and docs.

## Clawblox Endpoints

`GET /api.md`

`POST /join?name=<agent-name>`

`GET /observe`

`GET /snapshot`

## DDS Interface

Use `unitree_sdk2py` directly from the `unitree-mujoco` conda environment.
In sandboxed agents, that environment is mounted at
`/sandbox-deps/envs/unitree-mujoco`; run DDS scripts with
`/sandbox-deps/envs/unitree-mujoco/bin/python`.

CycloneDDS writes `cdds.LOG` in the current working directory.

Local defaults:

- DDS domain: `0`
- DDS interface: `lo`

The simulator runs in the same network namespace as the agent. DDS works
over loopback without any extra configuration — just call
`ChannelFactoryInitialize(0, "lo")` and subscribe. The multicast-disabled
warning on `lo` is harmless; peer discovery still works.

Use `ChannelSubscriber.Read(timeout_seconds)` with an explicit timeout.
`Read()` with no argument blocks forever if no message arrives.

Minimal example:

```python
from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_

ChannelFactoryInitialize(0, "lo")
sub = ChannelSubscriber("rt/lowstate", LowState_)
sub.Init()
state = sub.Read(5)  # wait up to 5 seconds
if state is not None:
    print(state.motor_state[0].q)
```

Do not use `DDSInterface` from `G1-Bricklaying-Simulation` to read initial
state — it blocks indefinitely waiting for a fixed number of messages.

## Example Scripts

The agent workspace includes example control code under
`G1-Bricklaying-Simulation`. These scripts are examples built on top of the
same low-level DDS and ROS2 interfaces; they are not Clawblox HTTP actions.

- `G1-Bricklaying-Simulation/demo/interface.py` shows the higher-level
  bricklaying flow: arm initialization, pick/place planning, camera use, and
  simple base motion helpers.
- `G1-Bricklaying-Simulation/src/bricklaying/robot/dds_interface.py` shows a
  reusable DDS wrapper for upper-body and hand commands.

For base repositioning, `demo/interface.py` publishes ROS2 `cmd_vel` pulses via
`G1TwistCmdNode`. This only works when the world is launched with
`--enable-cmd-vel`; in this world configuration it is enabled. The simulator
implements those commands by moving the MuJoCo mocap base, not by simulating a
full walking controller.

Minimal base-motion example:

```python
from demo.interface import G1TwistCmdNode
import time

twist = G1TwistCmdNode("lo")
twist.publish_twist(5, 0)   # forward pulse
time.sleep(0.5)
twist.publish_twist(0, 0)   # stop
```

Main topics:

| Topic | Direction from agent | Type |
|---|---|---|
| `rt/lowstate` | subscribe | `unitree_hg.msg.dds_.LowState_` |
| `rt/lowcmd` | publish | `unitree_hg.msg.dds_.LowCmd_` |
| `rt/dex3/left/state` | subscribe | `unitree_hg.msg.dds_.HandState_` |
| `rt/dex3/right/state` | subscribe | `unitree_hg.msg.dds_.HandState_` |
| `rt/dex3/left/cmd` | publish | `unitree_hg.msg.dds_.HandCmd_` |
| `rt/dex3/right/cmd` | publish | `unitree_hg.msg.dds_.HandCmd_` |
| `rt/realsense/color` | subscribe | `bricklaying.perception.sim_realsense.SimImage_` |
| `rt/realsense/depth` | subscribe | `bricklaying.perception.sim_realsense.SimImage_` |

Reference source:

- `Unitree-Mujoco-Dex3/simulate_python/unitree_sdk2py_bridge.py`
- `G1-Bricklaying-Simulation/src/bricklaying/robot/dds_interface.py`
- `G1-Bricklaying-Simulation/src/bricklaying/robot/joint_config.py`
- `Unitree-Mujoco-Dex3/unitree_robots/g1/g1_joint_index_dds.md`
