from pathlib import Path
import time

import mujoco
import mujoco.viewer


ROOT = Path(__file__).resolve().parent
SCENE = ROOT / "models" / "panda_cube" / "scene.xml"


def main() -> None:
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    data = mujoco.MjData(model)

    home_key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    if home_key >= 0:
        mujoco.mj_resetDataKeyframe(model, data, home_key)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            # The Panda actuators are position-like servos. Holding ctrl at the
            # keyframe values keeps the robot stable while the cube remains free.
            if home_key >= 0:
                data.ctrl[:] = model.key_ctrl[home_key]

            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(model.opt.timestep)


if __name__ == "__main__":
    main()

