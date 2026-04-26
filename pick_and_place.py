"""Pick-and-place demo using sensor data for proprioception, EE pose, and touch."""

from pathlib import Path
import time
import numpy as np
import mujoco
import mujoco.viewer

from panda_setup import set_panda_home

ROOT = Path(__file__).resolve().parent
SCENE = ROOT / "models" / "panda_cube" / "scene.xml"

DROP_XY = np.array([0.60, -0.10])

GRASP_OFFSET_Z  = 0.01
HOVER_CLEARANCE = 0.15

GRIPPER_OPEN   = 255.0
GRIPPER_CLOSED = 30.0   # closer to 0 = tighter grip

IK_STEPS     = 1000
IK_ALPHA     = 0.5
IK_DAMPING   = 1e-4
SETTLE_STEPS = 300
POS_WEIGHT   = 1.0
ROT_WEIGHT   = 0.1


def sensor(model, data, name):
    sid  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, name)
    adr  = model.sensor_adr[sid]
    dim  = model.sensor_dim[sid]
    return data.sensordata[adr:adr + dim].copy()


def rotation_error(R_curr, R_target):
    R_err = R_target @ R_curr.T
    return 0.5 * np.array([
        R_err[2, 1] - R_err[1, 2],
        R_err[0, 2] - R_err[2, 0],
        R_err[1, 0] - R_err[0, 1],
    ])


def solve_ik(model, data, site_id, target_pos, target_mat, n_arm=7):
    """Solve IK on a scratch copy of data. Returns target joint angles."""
    scratch = mujoco.MjData(model)
    scratch.qpos[:] = data.qpos[:]
    mujoco.mj_forward(model, scratch)
    for _ in range(IK_STEPS):
        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))
        mujoco.mj_jacSite(model, scratch, jacp, jacr, site_id)
        Jp  = jacp[:, :n_arm] * POS_WEIGHT
        Jr  = jacr[:, :n_arm] * ROT_WEIGHT
        J   = np.vstack([Jp, Jr])
        curr_mat = scratch.site_xmat[site_id].reshape(3, 3)
        err6 = np.concatenate([
            (target_pos - scratch.site_xpos[site_id]) * POS_WEIGHT,
            rotation_error(curr_mat, target_mat) * ROT_WEIGHT,
        ])
        pos_err = np.linalg.norm(err6[:3])
        if pos_err < 0.002:
            break
        dq = J.T @ np.linalg.solve(J @ J.T + IK_DAMPING * np.eye(6), err6)
        scratch.qpos[:n_arm] += IK_ALPHA * dq
        mujoco.mj_forward(model, scratch)
    actual = scratch.site_xpos[site_id]
    print(f"  IK target {target_pos.round(4)}  solved {actual.round(4)}  err {np.linalg.norm(target_pos - actual):.4f}")
    return scratch.qpos[:n_arm].copy()


def move_to(model, data, viewer, site_id, target_pos, target_mat, gripper_ctrl,
            tol=0.005):
    """Solve IK then drive ctrl toward target joints, simulating physics."""
    target_joints = solve_ik(model, data, site_id, target_pos, target_mat)
    # drive until the arm reaches the target or we time out
    for _ in range(IK_STEPS * 3):
        data.ctrl[:7] = target_joints
        data.ctrl[7]  = gripper_ctrl
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(model.opt.timestep)
        if np.linalg.norm(data.site_xpos[site_id] - target_pos) < tol:
            break


def close_until_contact(model, data, viewer):
    """Close gripper and stop as soon as both fingertips report touch."""
    for _ in range(SETTLE_STEPS * 3):
        data.ctrl[:7] = data.qpos[:7]
        data.ctrl[7]  = GRIPPER_CLOSED
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(model.opt.timestep)
        left  = sensor(model, data, "touch_left")[0]
        right = sensor(model, data, "touch_right")[0]
        if left > 0.01 or right > 0.01:
            print(f"  contact! left={left:.2f}  right={right:.2f}")
            break


def print_state(model, data):
    ee_pos  = sensor(model, data, "ee_pos")
    ee_quat = sensor(model, data, "ee_quat")
    c_pos   = sensor(model, data, "cube_pos")
    joints  = np.array([sensor(model, data, f"q{i}")[0] for i in range(1, 8)])
    f1      = sensor(model, data, "finger1_pos")[0]
    f2      = sensor(model, data, "finger2_pos")[0]
    print(f"  EE pos:    {ee_pos.round(4)}")
    print(f"  EE quat:   {ee_quat.round(4)}")
    print(f"  cube pos:  {c_pos.round(4)}")
    print(f"  joints:    {joints.round(3)}")
    print(f"  fingers:   {f1:.4f}  {f2:.4f}")


def main():
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    data  = mujoco.MjData(model)
    set_panda_home(model, data)

    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "ee")

    # read initial EE orientation from sensors — maintain this throughout
    target_mat = data.site_xmat[site_id].reshape(3, 3).copy()

    import threading
    go = threading.Event()

    def on_key(keycode):
        if keycode == 32:  # Space
            go.set()

    with mujoco.viewer.launch_passive(model, data, key_callback=on_key) as viewer:
        print("=== Initial state ===")
        print_state(model, data)
        print("\nMove the cube wherever you want, then press SPACE in the viewer to start.")

        while not go.is_set() and viewer.is_running():
            data.ctrl[:7] = data.qpos[:7]
            data.ctrl[7]  = GRIPPER_OPEN
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(model.opt.timestep)

        # read cube position from sensor at the moment Space was pressed
        cube_pos = sensor(model, data, "cube_pos")
        ee_pos   = sensor(model, data, "ee_pos")
        print(f"\ncube at {cube_pos.round(4)},  EE at {ee_pos.round(4)}")

        grasp_z   = cube_pos[2] + GRASP_OFFSET_Z
        hover_z   = grasp_z + HOVER_CLEARANCE
        predrop_z = grasp_z + HOVER_CLEARANCE

        print("\nPhase 1: hover above cube")
        move_to(model, data, viewer, site_id,
                np.array([cube_pos[0], cube_pos[1], hover_z]),
                target_mat, GRIPPER_OPEN)

        print("Phase 2: descend to cube")
        move_to(model, data, viewer, site_id,
                np.array([cube_pos[0], cube_pos[1], grasp_z]),
                target_mat, GRIPPER_OPEN)

        print("Phase 3: close gripper (touch-triggered)")
        close_until_contact(model, data, viewer)
        # hold grip and let it settle before lifting
        for _ in range(SETTLE_STEPS * 2):
            data.ctrl[:7] = data.ctrl[:7]
            data.ctrl[7]  = GRIPPER_CLOSED
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(model.opt.timestep)

        print("Phase 4: lift")
        move_to(model, data, viewer, site_id,
                np.array([cube_pos[0], cube_pos[1], hover_z]),
                target_mat, GRIPPER_CLOSED)

        print("Phase 5: move to drop XY")
        move_to(model, data, viewer, site_id,
                np.array([DROP_XY[0], DROP_XY[1], predrop_z]),
                target_mat, GRIPPER_CLOSED)

        print("Phase 6: lower to table")
        move_to(model, data, viewer, site_id,
                np.array([DROP_XY[0], DROP_XY[1], grasp_z]),
                target_mat, GRIPPER_CLOSED)

        print("Phase 7: release")
        for _ in range(SETTLE_STEPS * 2):
            data.ctrl[:7] = data.qpos[:7]
            data.ctrl[7]  = GRIPPER_OPEN
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(model.opt.timestep)

        print("Phase 8: retreat")
        move_to(model, data, viewer, site_id,
                np.array([DROP_XY[0], DROP_XY[1], predrop_z]),
                target_mat, GRIPPER_OPEN)

        print("\n=== Final state ===")
        print_state(model, data)
        print("Done!")

        while viewer.is_running():
            data.ctrl[:7] = data.qpos[:7]
            data.ctrl[7]  = GRIPPER_OPEN
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(model.opt.timestep)


if __name__ == "__main__":
    main()
