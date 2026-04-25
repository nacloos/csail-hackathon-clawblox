from pathlib import Path
import os

import mujoco

from panda_setup import set_panda_home


ROOT = Path(__file__).resolve().parent
SCENE = ROOT / "models" / "panda_cube" / "scene.xml"


def main() -> None:
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    data = mujoco.MjData(model)
    set_panda_home(model, data)

    renderer = mujoco.Renderer(model, height=240, width=320)
    for _ in range(50):
        mujoco.mj_step(model, data)

    renderer.update_scene(data, camera="overview")
    image = renderer.render()
    print(f"ok: {model.nq=} {model.nv=} {model.nu=} render_shape={image.shape} mean_pixel={int(image.mean())}", flush=True)

    # EGL teardown can hang on some WSL/ARM setups; this is only a smoke test.
    os._exit(0)


if __name__ == "__main__":
    main()
