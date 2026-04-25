from pathlib import Path
import time

import mujoco
import mujoco.viewer

from panda_setup import set_panda_home


ROOT = Path(__file__).resolve().parent
SCENE = ROOT / "models" / "panda_cube" / "scene.xml"


def main() -> None:
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    data = mujoco.MjData(model)
    set_panda_home(model, data)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            # The viewer control panel edits data.ctrl directly. Do not overwrite
            # it here, or the sliders will appear to do nothing.
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(model.opt.timestep)


if __name__ == "__main__":
    main()
