from __future__ import annotations

from dataclasses import dataclass

import mujoco


PANDA_HOME_QPOS = (0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853, 0.04, 0.04)
PANDA_HOME_CTRL = (0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853, 255.0)


@dataclass(frozen=True)
class PandaArm:
    name: str
    prefix: str
    actuator_ids: tuple[int, ...]


def find_panda_arms(model: mujoco.MjModel) -> list[PandaArm]:
    arms: list[PandaArm] = []
    for name, prefix in (("left", "left_"), ("right", "right_"), ("panda", "")):
        actuator_ids = []
        for index in range(1, 9):
            actuator_id = mujoco.mj_name2id(
                model,
                mujoco.mjtObj.mjOBJ_ACTUATOR,
                f"{prefix}actuator{index}",
            )
            if actuator_id < 0:
                break
            actuator_ids.append(actuator_id)
        if len(actuator_ids) == 8:
            arms.append(PandaArm(name=name, prefix=prefix, actuator_ids=tuple(actuator_ids)))
    return arms


def set_panda_home(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    """Put Panda joints at home, leaving free objects like the cube alone."""
    for arm in find_panda_arms(model):
        for index, value in enumerate(PANDA_HOME_QPOS, start=1):
            joint_name = f"{arm.prefix}joint{index}"
            if index > 7:
                joint_name = f"{arm.prefix}finger_joint{index - 7}"
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            if joint_id < 0:
                continue

            qpos_address = model.jnt_qposadr[joint_id]
            data.qpos[qpos_address] = value

        for actuator_id, value in zip(arm.actuator_ids, PANDA_HOME_CTRL, strict=True):
            data.ctrl[actuator_id] = value
    mujoco.mj_forward(model, data)
