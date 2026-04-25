# Franka Panda MuJoCo Sandbox

Tiny MuJoCo setup for a Franka Emika Panda arm manipulating a cube.

The Panda model is vendored from DeepMind's MuJoCo Menagerie in
`models/franka_emika_panda`. The custom scene in `models/panda_cube/scene.xml`
adds a table, cube, lighting, and camera around that robot.

## Run

```bash
uv run --with mujoco python run_viewer.py
```

If the viewer opens, you should see the Panda arm in its home pose with a red
cube on the table. The script keeps the arm at the home joint targets and leaves
the gripper open so the scene is stable while you inspect it.

For headless rendering/debugging on WSL:

```bash
MUJOCO_GL=egl uv run --with mujoco python smoke_test.py
```

