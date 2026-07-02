"""Unitree SDK2 DDS control bridge for a MuJoCo world.

Optional add-on to ``server.py``'s ``SimState``: when a DDS domain is configured
(``--dds-domain`` / ``WORLD_DDS_DOMAIN_ID``), the world speaks the same wire
protocol as the physical Unitree G1. It subscribes ``rt/lowcmd`` (``unitree_hg``
``LowCmd_``: per-motor q/dq/kp/kd/tau), applies onboard-style joint PD to the
torque actuators, and publishes ``rt/lowstate`` — so the same controller drives
the sim or the real robot. Discovery is unicast on loopback (``lo`` cannot
multicast). Requires cyclonedds + unitree_sdk2py; imported lazily so worlds
without DDS never need them.
"""
from __future__ import annotations

import os
from typing import Any

import numpy as np

MOTOR_SLOTS = 35  # unitree_hg fixed motor-array length


class G1DdsBridge:
    def __init__(self, model: Any, domain_id: int, interface: str = "lo") -> None:
        self.model = model
        self.domain_id = int(domain_id)
        self.interface = interface
        self.nu = int(model.nu)
        # Map each actuator to the qpos/qvel address of the joint it drives,
        # using MuJoCo's own tables so it holds for any scene.
        self.qadr = np.zeros(self.nu, dtype=int)
        self.vadr = np.zeros(self.nu, dtype=int)
        for act in range(self.nu):
            joint = int(model.actuator_trnid[act, 0])
            self.qadr[act] = int(model.jnt_qposadr[joint])
            self.vadr[act] = int(model.jnt_dofadr[joint])
        self.latest_cmd: Any = None
        self._reader = None
        self._writer = None

    def cyclonedds_uri(self) -> str:
        # lo is not multicast-capable, so discovery uses unicast peers on
        # localhost. Agents must use the same config (see API.md).
        return (
            "<CycloneDDS><Domain><General>"
            f"<Interfaces><NetworkInterface name='{self.interface}'/></Interfaces>"
            "<AllowMulticast>false</AllowMulticast></General>"
            "<Discovery><ParticipantIndex>auto</ParticipantIndex>"
            "<Peers><Peer address='localhost'/></Peers></Discovery>"
            "</Domain></CycloneDDS>"
        )

    def setup(self) -> None:
        """Create the DDS participant/reader/writer. Call on the sim thread
        (CycloneDDS entities are thread-affine)."""
        os.environ.setdefault("CYCLONEDDS_URI", self.cyclonedds_uri())
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import (
            LowCmd_,
            LowState_,
            MotorState_,
            IMUState_,
        )
        from cyclonedds.domain import DomainParticipant
        from cyclonedds.topic import Topic
        from cyclonedds.pub import DataWriter
        from cyclonedds.sub import DataReader

        self._LowCmd_ = LowCmd_
        dp = DomainParticipant(self.domain_id)
        self._reader = DataReader(dp, Topic(dp, "rt/lowcmd", LowCmd_))
        self._writer = DataWriter(dp, Topic(dp, "rt/lowstate", LowState_))
        # Pre-allocate the outbound LowState and mutate it in place each step:
        # rebuilding 35 MotorState_ objects per step (at 500Hz) is enough Python
        # allocation to push the realtime loop past its 2ms budget.
        self._state = LowState_(
            version=[0, 0], mode_pr=0, mode_machine=0, tick=0,
            imu_state=IMUState_(
                quaternion=[1.0, 0.0, 0.0, 0.0], gyroscope=[0.0, 0.0, 0.0],
                accelerometer=[0.0, 0.0, 0.0], rpy=[0.0, 0.0, 0.0], temperature=0,
            ),
            motor_state=[
                MotorState_(
                    mode=1, q=0.0, dq=0.0, ddq=0.0, tau_est=0.0,
                    temperature=[0, 0], vol=0.0, sensor=[0, 0],
                    motorstate=0, reserve=[0, 0, 0, 0],
                )
                for _ in range(MOTOR_SLOTS)
            ],
            wireless_remote=[0] * 40, reserve=[0, 0, 0, 0], crc=0,
        )
        print(
            f"DDS active: domain={self.domain_id} interface={self.interface} "
            f"topics=rt/lowcmd,rt/lowstate actuators={self.nu}",
            flush=True,
        )

    def apply_control(self, data: Any) -> None:
        """Read the latest LowCmd and apply onboard-style PD to `data.ctrl`."""
        # take() also yields InvalidSample (dispose/unregister) notifications,
        # e.g. when a controller exits; keep only real LowCmd_ data.
        for sample in self._reader.take(N=16):
            if isinstance(sample, self._LowCmd_):
                self.latest_cmd = sample
        cmd = self.latest_cmd
        if cmd is None:
            return
        for i in range(self.nu):
            mc = cmd.motor_cmd[i]
            q = data.qpos[self.qadr[i]]
            dq = data.qvel[self.vadr[i]]
            data.ctrl[i] = mc.kp * (mc.q - q) + mc.kd * (mc.dq - dq) + mc.tau

    def publish_state(self, data: Any, tick: int) -> None:
        """Publish LowState (motor q/dq/tau + floating-base IMU), mutating the
        pre-allocated message in place to avoid per-step allocation."""
        qpos = data.qpos
        qvel = data.qvel
        force = data.actuator_force
        motors = self._state.motor_state
        for i in range(self.nu):
            m = motors[i]
            m.q = float(qpos[self.qadr[i]])
            m.dq = float(qvel[self.vadr[i]])
            m.tau_est = float(force[i])
        imu = self._state.imu_state
        imu.quaternion = [float(x) for x in qpos[3:7]]  # free-base quaternion
        imu.gyroscope = [float(x) for x in qvel[3:6]]
        self._state.tick = tick & 0xFFFFFFFF
        self._writer.write(self._state)

    @property
    def have_command(self) -> bool:
        return self.latest_cmd is not None
