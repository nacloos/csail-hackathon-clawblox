from __future__ import annotations

import mujoco


PANDA_HOME_QPOS = (0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853, 0.04, 0.04)
PANDA_HOME_CTRL = (0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853, 255.0)


def set_panda_home(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    """Put only the Panda joints at home, leaving free objects like the cube alone."""
    for index, value in enumerate(PANDA_HOME_QPOS, start=1):
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"joint{index}")
        if joint_id < 0:
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"finger_joint{index - 7}")

        qpos_address = model.jnt_qposadr[joint_id]
        data.qpos[qpos_address] = value

    data.ctrl[:] = PANDA_HOME_CTRL
    mujoco.mj_forward(model, data)

