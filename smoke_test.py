from pathlib import Path
import os

import mujoco


ROOT = Path(__file__).resolve().parent
SCENE = ROOT / "models" / "panda_cube" / "scene.xml"


def main() -> None:
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    data = mujoco.MjData(model)

    home_key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    if home_key >= 0:
        mujoco.mj_resetDataKeyframe(model, data, home_key)
        data.ctrl[:] = model.key_ctrl[home_key]

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
