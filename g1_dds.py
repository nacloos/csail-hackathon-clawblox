"""Unitree SDK2 DDS control bridge for a MuJoCo G1 world.

Optional add-on to ``server.py``'s ``SimState``: when a DDS domain is configured
(``--dds-domain`` / ``WORLD_DDS_DOMAIN_ID``), the world speaks the same wire
protocol as the physical Unitree G1, so the same controller drives the sim or
the real robot.

- Body (legs/waist/arms, torque motors): subscribe ``rt/lowcmd`` (``LowCmd_``,
  per-motor q/dq/kp/kd/tau), apply onboard-style joint PD, publish ``rt/lowstate``.
- Dex3 hands (position servos), when the scene has them: subscribe
  ``rt/dex3/{left,right}/cmd`` (``HandCmd_``, per-motor target q), publish
  ``rt/dex3/{left,right}/state`` (``HandState_``).

Discovery is unicast on loopback (``lo`` cannot multicast). Requires cyclonedds
+ unitree_sdk2py; imported lazily so worlds without DDS never need them.
"""
from __future__ import annotations

import os
from typing import Any

import mujoco
import numpy as np

MOTOR_SLOTS = 35  # unitree_hg fixed body motor-array length
# Dex3 hand motor order in the unitree_hg IDL (index before middle) — the MuJoCo
# scene orders the actuators differently, so hands are mapped by joint name.
HAND_IDL_ORDER = ["thumb_0", "thumb_1", "thumb_2", "index_0", "index_1", "middle_0", "middle_1"]


def _actuator_state_addr(model: Any, act: int) -> tuple[int, int]:
    joint = int(model.actuator_trnid[act, 0])
    return int(model.jnt_qposadr[joint]), int(model.jnt_dofadr[joint])


class G1DdsBridge:
    def __init__(self, model: Any, domain_id: int, interface: str = "lo") -> None:
        self.model = model
        self.domain_id = int(domain_id)
        self.interface = interface

        # Classify actuators: Dex3 hand actuators (name `<side>_hand_<joint>_joint`)
        # vs. body torque motors (everything else). Hands map by joint name to
        # the IDL order; body keeps actuator order (== the LowCmd body order).
        self.body_acts: list[int] = []
        self.hand_acts: dict[str, list[int | None]] = {
            "left": [None] * 7,
            "right": [None] * 7,
        }
        for act in range(int(model.nu)):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, act) or ""
            side = "left" if name.startswith("left_hand_") else (
                "right" if name.startswith("right_hand_") else None
            )
            if side is not None:
                for slot, joint in enumerate(HAND_IDL_ORDER):
                    if name == f"{side}_hand_{joint}_joint":
                        self.hand_acts[side][slot] = act
            else:
                self.body_acts.append(act)

        self.n_body = len(self.body_acts)
        self.body_qadr = np.array([_actuator_state_addr(model, a)[0] for a in self.body_acts])
        self.body_vadr = np.array([_actuator_state_addr(model, a)[1] for a in self.body_acts])
        self.has_hands = {
            side: all(a is not None for a in acts) for side, acts in self.hand_acts.items()
        }
        self.hand_addr = {
            side: [
                _actuator_state_addr(model, a) if a is not None else (0, 0)
                for a in self.hand_acts[side]
            ]
            for side in ("left", "right")
        }

        self.latest_cmd: Any = None
        self.latest_hand: dict[str, Any] = {"left": None, "right": None}

    # -- lifecycle -----------------------------------------------------------
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
        """Create the DDS entities. Call on the sim thread (CycloneDDS is
        thread-affine)."""
        os.environ.setdefault("CYCLONEDDS_URI", self.cyclonedds_uri())
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_, HandCmd_, HandState_
        from unitree_sdk2py.idl.default import (
            unitree_hg_msg_dds__LowState_ as low_state_default,
            unitree_hg_msg_dds__HandState_ as hand_state_default,
        )
        from cyclonedds.domain import DomainParticipant
        from cyclonedds.topic import Topic
        from cyclonedds.pub import DataWriter
        from cyclonedds.sub import DataReader

        self._LowCmd_ = LowCmd_
        self._HandCmd_ = HandCmd_
        dp = DomainParticipant(self.domain_id)
        self._reader = DataReader(dp, Topic(dp, "rt/lowcmd", LowCmd_))
        self._writer = DataWriter(dp, Topic(dp, "rt/lowstate", LowState_))
        # Pre-allocate outbound messages, mutated in place each step (rebuilding
        # them at 500Hz allocates enough to blow the realtime budget).
        self._state = low_state_default()

        self._hand_reader: dict[str, Any] = {}
        self._hand_writer: dict[str, Any] = {}
        self._hand_state: dict[str, Any] = {}
        for side in ("left", "right"):
            if not self.has_hands[side]:
                continue
            self._hand_reader[side] = DataReader(
                dp, Topic(dp, f"rt/dex3/{side}/cmd", HandCmd_)
            )
            self._hand_writer[side] = DataWriter(
                dp, Topic(dp, f"rt/dex3/{side}/state", HandState_)
            )
            self._hand_state[side] = hand_state_default()

        hands = [s for s in ("left", "right") if self.has_hands[s]]
        print(
            f"DDS active: domain={self.domain_id} interface={self.interface} "
            f"body_motors={self.n_body} hands={hands or 'none'} "
            f"topics=rt/lowcmd,rt/lowstate"
            + (",rt/dex3/*" if hands else ""),
            flush=True,
        )

    # -- control -------------------------------------------------------------
    def apply_control(self, data: Any) -> None:
        # Body: onboard-style joint PD over torque actuators.
        for sample in self._reader.take(N=16):
            if isinstance(sample, self._LowCmd_):
                self.latest_cmd = sample
        cmd = self.latest_cmd
        if cmd is not None:
            for i in range(self.n_body):
                mc = cmd.motor_cmd[i]
                q = data.qpos[self.body_qadr[i]]
                dq = data.qvel[self.body_vadr[i]]
                data.ctrl[self.body_acts[i]] = mc.kp * (mc.q - q) + mc.kd * (mc.dq - dq) + mc.tau
        # Hands: position servos, target = HandCmd motor_cmd[slot].q.
        for side in self._hand_reader:
            for sample in self._hand_reader[side].take(N=8):
                if isinstance(sample, self._HandCmd_):
                    self.latest_hand[side] = sample
            hc = self.latest_hand[side]
            if hc is None:
                continue
            for slot, act in enumerate(self.hand_acts[side]):
                data.ctrl[act] = hc.motor_cmd[slot].q

    def publish_state(self, data: Any, tick: int) -> None:
        qpos, qvel, force = data.qpos, data.qvel, data.actuator_force
        motors = self._state.motor_state
        for i in range(self.n_body):
            m = motors[i]
            m.q = float(qpos[self.body_qadr[i]])
            m.dq = float(qvel[self.body_vadr[i]])
            m.tau_est = float(force[self.body_acts[i]])
        imu = self._state.imu_state
        imu.quaternion = [float(x) for x in qpos[3:7]]  # free-base quaternion
        imu.gyroscope = [float(x) for x in qvel[3:6]]
        self._state.tick = tick & 0xFFFFFFFF
        self._writer.write(self._state)

        for side in self._hand_writer:
            hs = self._hand_state[side].motor_state
            for slot, act in enumerate(self.hand_acts[side]):
                qadr, vadr = self.hand_addr[side][slot]
                hs[slot].q = float(qpos[qadr])
                hs[slot].dq = float(qvel[vadr])
                hs[slot].tau_est = float(force[act])
            self._hand_writer[side].write(self._hand_state[side])

    @property
    def have_command(self) -> bool:
        return self.latest_cmd is not None
