"""
IK controller WebSocket bridge over the MuJoCo REST API (server.py / run_with_viewer.py).

Architecture:
  run_with_viewer.py  →  REST API on http://localhost:8080  (sim + viewer)
  sim_server.py       →  WebSocket on ws://localhost:8765   (IK + commands)

This server polls /observe for state, runs IK locally, and drives the arm
by calling POST /input?SetControl. It accepts high-level commands over
WebSocket so Claude (or any client) can issue natural-language-style instructions.

WebSocket protocol:
  Client → Server:
    {"cmd": "pick",      "body": "block_red"}           # pick named block
    {"cmd": "pick",      "body": "block_red", "drop": [x, y]}
    {"cmd": "move_ee",   "pos": [x, y, z]}              # move EE to world pos
    {"cmd": "set_ctrl",  "ctrl": [8 floats]}             # raw control
    {"cmd": "reset"}
    {"cmd": "observe"}                                   # get state snapshot

  Server → Client:
    {"type": "state",   ...}                             # broadcast ~20 Hz
    {"type": "done",    "cmd": "..."}
    {"type": "error",   "msg": "..."}
"""

from __future__ import annotations
import asyncio
import json
import queue
import threading
import time
from pathlib import Path

import httpx
import mujoco
import numpy as np
import websockets

from panda_setup import set_panda_home

ROOT       = Path(__file__).resolve().parent
SCENE      = ROOT / "models" / "panda_cube" / "scene.xml"
REST_URL   = "http://localhost:8080"
WS_HOST    = "localhost"
WS_PORT    = 8765

GRIPPER_OPEN   = 255.0
GRIPPER_CLOSED = 30.0
GRASP_OFFSET_Z = 0.015   # EE site above block centre
HOVER_Z_ABS    = 0.25    # absolute hover height
IK_STEPS       = 1000
IK_ALPHA       = 0.5
IK_DAMPING     = 1e-4
POS_WEIGHT     = 1.0
ROT_WEIGHT     = 0.1
SETTLE_STEPS   = 200

_clients: set = set()
_loop: asyncio.AbstractEventLoop | None = None
_cmd_queue: queue.Queue = queue.Queue()


# ── REST helpers ──────────────────────────────────────────────────────────────

def rest_observe() -> dict:
    r = httpx.get(f"{REST_URL}/observe", timeout=5)
    r.raise_for_status()
    return r.json()


def rest_set_ctrl(ctrl: list[float]) -> dict:
    r = httpx.post(f"{REST_URL}/input",
                   json={"type": "SetControl", "data": {"ctrl": ctrl}},
                   timeout=5)
    r.raise_for_status()
    return r.json()


def rest_reset() -> dict:
    r = httpx.post(f"{REST_URL}/input", json={"type": "Reset", "data": {}}, timeout=5)
    r.raise_for_status()
    return r.json()


# ── IK (local, uses MuJoCo model for Jacobians) ───────────────────────────────

def rotation_error(R_curr, R_target):
    R = R_target @ R_curr.T
    return 0.5 * np.array([R[2,1]-R[1,2], R[0,2]-R[2,0], R[1,0]-R[0,1]])


def solve_ik(model, site_id, qpos_init, target_pos, target_mat, n_arm=7):
    scratch = mujoco.MjData(model)
    scratch.qpos[:len(qpos_init)] = qpos_init
    mujoco.mj_forward(model, scratch)
    for _ in range(IK_STEPS):
        jacp, jacr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
        mujoco.mj_jacSite(model, scratch, jacp, jacr, site_id)
        J = np.vstack([jacp[:,:n_arm]*POS_WEIGHT, jacr[:,:n_arm]*ROT_WEIGHT])
        R = scratch.site_xmat[site_id].reshape(3,3)
        e = np.concatenate([(target_pos - scratch.site_xpos[site_id])*POS_WEIGHT,
                             rotation_error(R, target_mat)*ROT_WEIGHT])
        if np.linalg.norm(e[:3]) < 0.002:
            break
        scratch.qpos[:n_arm] += IK_ALPHA * J.T @ np.linalg.solve(J@J.T + IK_DAMPING*np.eye(6), e)
        mujoco.mj_forward(model, scratch)
    return scratch.qpos[:n_arm].tolist()


# ── push to WS clients ────────────────────────────────────────────────────────

def push(msg: dict):
    if not _clients or _loop is None:
        return
    asyncio.run_coroutine_threadsafe(_push_async(json.dumps(msg)), _loop)


async def _push_async(msg: str):
    dead = set()
    for ws in list(_clients):
        try:
            await ws.send(msg)
        except Exception:
            dead.add(ws)
    _clients.difference_update(dead)


# ── controller thread ─────────────────────────────────────────────────────────

def obs_to_state(obs: dict) -> dict:
    """Summarise REST /observe into a compact state dict."""
    qpos = obs["qpos"]
    return {
        "type":    "state",
        "t":       obs["time"],
        "joints":  [round(q, 4) for q in qpos[:7]],
        "fingers": [round(qpos[7], 4), round(qpos[8], 4)] if len(qpos) > 8 else [],
        "objects": obs.get("objects", []),
    }


def find_object(obs: dict, name: str) -> list[float] | None:
    for obj in obs.get("objects", []):
        if obj["name"] == name:
            return obj["position"]
    return None


def ctrl_from_joints(joints7: list[float], gripper: float) -> list[float]:
    return list(joints7) + [gripper]


def move_to(model, site_id, target_mat, target_pos, gripper_ctrl, tol=0.006):
    """IK → REST SetControl loop until EE reaches target."""
    for _ in range(IK_STEPS * 4):
        obs      = rest_observe()
        qpos     = obs["qpos"]
        joints7  = solve_ik(model, site_id, qpos, target_pos, target_mat)
        ctrl     = ctrl_from_joints(joints7, gripper_ctrl)
        rest_set_ctrl(ctrl)
        # check EE position via model forward
        scratch = mujoco.MjData(model)
        scratch.qpos[:len(qpos)] = qpos
        mujoco.mj_forward(model, scratch)
        ee = scratch.site_xpos[site_id]
        if np.linalg.norm(ee - target_pos) < tol:
            break
        time.sleep(model.opt.timestep * 5)
    push(obs_to_state(rest_observe()))


def settle(gripper_ctrl: float, steps: int = SETTLE_STEPS):
    obs     = rest_observe()
    joints7 = obs["qpos"][:7]
    ctrl    = ctrl_from_joints(joints7, gripper_ctrl)
    for _ in range(steps):
        rest_set_ctrl(ctrl)
        time.sleep(model_global.opt.timestep * 2)


def do_pick(model, site_id, target_mat, obj_pos: list[float], drop_xy: list[float]):
    cx, cy, cz = obj_pos
    dx, dy     = drop_xy
    gz         = cz + GRASP_OFFSET_Z
    hz         = HOVER_Z_ABS

    print(f"[ctrl] pick ({cx:.3f},{cy:.3f}) → drop ({dx:.3f},{dy:.3f})")
    move_to(model, site_id, target_mat, np.array([cx, cy, hz]),  GRIPPER_OPEN)
    move_to(model, site_id, target_mat, np.array([cx, cy, gz]),  GRIPPER_OPEN)

    # close until contact (poll touch via model forward isn't available here;
    # just close for enough time)
    obs     = rest_observe()
    joints7 = obs["qpos"][:7]
    for _ in range(SETTLE_STEPS * 3):
        rest_set_ctrl(ctrl_from_joints(joints7, GRIPPER_CLOSED))
        time.sleep(model.opt.timestep * 2)

    settle(GRIPPER_CLOSED, SETTLE_STEPS * 2)
    move_to(model, site_id, target_mat, np.array([cx, cy, hz]),  GRIPPER_CLOSED)
    move_to(model, site_id, target_mat, np.array([dx, dy, hz]),  GRIPPER_CLOSED)
    move_to(model, site_id, target_mat, np.array([dx, dy, gz]),  GRIPPER_CLOSED)
    settle(GRIPPER_OPEN, SETTLE_STEPS * 2)
    move_to(model, site_id, target_mat, np.array([dx, dy, hz]),  GRIPPER_OPEN)
    print("[ctrl] pick done")


model_global: mujoco.MjModel | None = None


def controller_thread():
    global model_global
    model      = mujoco.MjModel.from_xml_path(str(SCENE))
    model_global = model
    scratch    = mujoco.MjData(model)
    set_panda_home(model, scratch)
    site_id    = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "ee")
    target_mat = scratch.site_xmat[site_id].reshape(3,3).copy()
    drop_xy    = [0.55, -0.15]

    print(f"[ctrl] ready — waiting for commands on ws://{WS_HOST}:{WS_PORT}")
    broadcast_t = 0.0

    while True:
        # broadcast state periodically
        now = time.time()
        if now - broadcast_t > 0.05:
            try:
                push(obs_to_state(rest_observe()))
                broadcast_t = now
            except Exception:
                pass

        try:
            cmd  = _cmd_queue.get_nowait()
            name = cmd.get("cmd", "")
            try:
                if name == "observe":
                    obs = rest_observe()
                    push(obs_to_state(obs))
                    push({"type": "done", "cmd": "observe"})

                elif name == "reset":
                    rest_reset()
                    push({"type": "done", "cmd": "reset"})

                elif name == "set_drop":
                    drop_xy = cmd["xy"]
                    push({"type": "done", "cmd": "set_drop"})

                elif name == "set_ctrl":
                    rest_set_ctrl(cmd["ctrl"])
                    push({"type": "done", "cmd": "set_ctrl"})

                elif name == "move_ee":
                    target = np.array(cmd["pos"])
                    move_to(model, site_id, target_mat, target, GRIPPER_OPEN)
                    push({"type": "done", "cmd": "move_ee"})

                elif name == "pick":
                    body_name = cmd.get("body", "block_red")
                    if "drop" in cmd:
                        drop_xy = cmd["drop"]
                    obs     = rest_observe()
                    obj_pos = find_object(obs, body_name)
                    if obj_pos is None:
                        push({"type": "error", "msg": f"body '{body_name}' not found"})
                    else:
                        do_pick(model, site_id, target_mat, obj_pos, drop_xy)
                        push({"type": "done", "cmd": "pick"})

            except Exception as e:
                push({"type": "error", "msg": str(e)})
                import traceback; traceback.print_exc()

        except queue.Empty:
            time.sleep(0.005)


# ── WebSocket server ──────────────────────────────────────────────────────────

async def ws_handler(websocket):
    _clients.add(websocket)
    print(f"[ws] connected ({len(_clients)} clients)")
    try:
        async for raw in websocket:
            try:
                cmd = json.loads(raw)
                print(f"[ws] → {cmd}")
                _cmd_queue.put(cmd)
            except Exception as e:
                await websocket.send(json.dumps({"type": "error", "msg": str(e)}))
    except websockets.exceptions.ConnectionClosed:
        pass
    finally:
        _clients.discard(websocket)
        print(f"[ws] disconnected ({len(_clients)} clients)")


async def main_async():
    global _loop
    _loop = asyncio.get_running_loop()
    threading.Thread(target=controller_thread, daemon=True).start()
    async with websockets.serve(ws_handler, WS_HOST, WS_PORT):
        print(f"[ws] listening on ws://{WS_HOST}:{WS_PORT}")
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main_async())
