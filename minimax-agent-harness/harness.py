"""
MiniMax M2.7 agent harness for the MuJoCo Panda sim.

Connects MiniMax (or any Anthropic-compatible model) to the existing sim
REST API at http://localhost:8080. Follows the same session protocol as
the Claude agents — /join first, X-Session on every request.

Usage:
    export MINIMAX_API_KEY=...
    python harness.py "stack the red and blue blocks"
    python harness.py --model minimax-m2.7 --world http://localhost:8080 "observe"

Environment:
    MINIMAX_API_KEY   required
    WORLD_URL         sim base URL (default http://localhost:8080)
    MODEL             model override (default minimax-m2.7)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import anthropic
import httpx
import mujoco
import numpy as np

LOG_FILE = Path(__file__).parent / "harness.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("minimax")

# ── config ────────────────────────────────────────────────────────────────────

WORLD_URL       = os.getenv("WORLD_URL", "http://localhost:8080")
MODEL           = os.getenv("MODEL", "minimax-m2.7")
BASE_URL        = "https://api.minimax.io/anthropic"
API_KEY         = os.getenv("MINIMAX_API_KEY", "")
MAX_TURNS       = 40
AGENT_NAME      = "minimax-agent"
THINKING_BUDGET = int(os.getenv("THINKING_BUDGET", "0"))  # 0 = off, >0 = on

# ── IK model (loaded once at startup) ─────────────────────────────────────────

_SCENE = (Path(__file__).parent.parent / "models" / "panda_cube" / "scene.xml").resolve()
_ik_model = mujoco.MjModel.from_xml_path(str(_SCENE))
_ik_data  = mujoco.MjData(_ik_model)

_HAND_ID = mujoco.mj_name2id(_ik_model, mujoco.mjtObj.mjOBJ_BODY, "hand")
_LEFT_ID = mujoco.mj_name2id(_ik_model, mujoco.mjtObj.mjOBJ_BODY, "left_finger")

# Compute fingertip-to-hand offset at home pose so z targets are in fingertip space
_home_q = [0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853]
_JOINT_ADDRS = [
    int(_ik_model.jnt_qposadr[mujoco.mj_name2id(_ik_model, mujoco.mjtObj.mjOBJ_JOINT, f"joint{i}")])
    for i in range(1, 8)
]
for _idx, _addr in enumerate(_JOINT_ADDRS):
    _ik_data.qpos[_addr] = _home_q[_idx]
mujoco.mj_kinematics(_ik_model, _ik_data)
_hand_z_home = float(_ik_data.xpos[_HAND_ID][2])
_tip_z_home  = float(min(
    _ik_data.geom_xpos[i][2]
    for i in range(_ik_model.ngeom)
    if _ik_model.geom_bodyid[i] == _LEFT_ID
))
# How much higher the hand body is vs fingertips
HAND_TO_TIP_Z = _hand_z_home - _tip_z_home   # ≈ 0.108 m

JOINT_LIMITS = np.array([
    [-2.897, 2.897],
    [-1.763, 1.763],
    [-2.897, 2.897],
    [-3.072, -0.069],
    [-2.897, 2.897],
    [-0.018, 3.752],
    [-2.897, 2.897],
])

log.info("IK model loaded. HAND_TO_TIP_Z=%.4f m", HAND_TO_TIP_Z)

# ── sim session ───────────────────────────────────────────────────────────────

class SimSession:
    def __init__(self, world_url: str, name: str) -> None:
        self.world_url  = world_url
        self.session_id: str | None = None
        self.robot:      str | None = None
        self._client    = httpx.Client(timeout=10)

    def join(self, name: str) -> dict:
        resp = self._client.post(f"{self.world_url}/join", params={"name": name}).json()
        self.session_id = resp.get("session")
        self.robot      = resp.get("robot")
        return resp

    def leave(self) -> None:
        if self.session_id:
            self._client.post(f"{self.world_url}/leave",
                              headers={"X-Session": self.session_id})

    def observe(self) -> dict:
        headers = {"X-Session": self.session_id} if self.session_id else {}
        return self._client.get(f"{self.world_url}/observe", headers=headers).json()

    def set_control(self, ctrl: list[float]) -> dict:
        return self._client.post(
            f"{self.world_url}/input",
            headers={"X-Session": self.session_id} if self.session_id else {},
            json={"type": "SetControl", "data": {"ctrl": ctrl}},
        ).json()

    def reset(self) -> dict:
        return self._client.post(
            f"{self.world_url}/input",
            headers={"X-Session": self.session_id} if self.session_id else {},
            json={"type": "Reset", "data": {}},
        ).json()

    def chat(self, content: str) -> None:
        if self.session_id:
            self._client.post(
                f"{self.world_url}/chat",
                headers={"X-Session": self.session_id},
                json={"content": content},
            )

    def api_doc(self) -> str:
        return self._client.get(f"{self.world_url}/api.md").text


# ── tool definitions ───────────────────────────────────────────────────────────

TOOLS: list[dict] = [
    {
        "name": "observe",
        "description": "Get current sim state: object positions, joint angles, touch sensors.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "set_control",
        "description": (
            "Send raw joint position targets to the arm. "
            "ctrl is 8 values: [joint0..joint6 in radians, gripper]. "
            "Gripper: 0=closed, 255=open."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ctrl": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "8 actuator values",
                }
            },
            "required": ["ctrl"],
        },
    },
    {
        "name": "reset",
        "description": "Reset the robot arm to its home position. Does not move objects.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "move_ee",
        "description": (
            "Move the end-effector to a target world position (x, y, z) using numerical IK. "
            "Much easier than set_control — just specify where you want the gripper to go. "
            "gripper: 0=closed, 255=open (default 255)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "x": {"type": "number", "description": "target x in metres"},
                "y": {"type": "number", "description": "target y in metres"},
                "z": {"type": "number", "description": "target z in metres (table surface ~0.0, block top ~0.07)"},
                "gripper": {"type": "number", "description": "gripper: 0=closed, 255=open", "default": 255},
            },
            "required": ["x", "y", "z"],
        },
    },
    {
        "name": "chat",
        "description": "Broadcast a text message visible to all agents in the sim.",
        "input_schema": {
            "type": "object",
            "properties": {
                "message": {"type": "string", "description": "1-500 chars"}
            },
            "required": ["message"],
        },
    },
]


# ── tool execution ─────────────────────────────────────────────────────────────

def fmt_obs(obs: dict) -> str:
    lines = []

    # End-effector world position (from robots array)
    for robot in obs.get("robots", []):
        ee = robot.get("ee_pos")
        if ee:
            lines.append(f"EE pos (world): ({ee[0]:.3f}, {ee[1]:.3f}, {ee[2]:.3f})")
        ctrl = robot.get("ctrl", [])
        if ctrl:
            lines.append(f"joint ctrl: {[round(v,3) for v in ctrl]}")

    # Object positions
    lines.append("Objects:")
    for o in obs.get("objects", []):
        p = o["position"]
        lines.append(f"  {o['name']:<16} ({p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f})")

    touch = obs.get("touch", {})
    if touch:
        lines.append(f"touch L={touch.get('left',0):.2f} R={touch.get('right',0):.2f}")

    lines.append(f"t={obs.get('time',0):.2f}s")
    return "\n".join(lines)


_DOF_ADDRS = [
    int(_ik_model.jnt_dofadr[mujoco.mj_name2id(_ik_model, mujoco.mjtObj.mjOBJ_JOINT, f"joint{i}")])
    for i in range(1, 8)
]


def move_ee_to(target: list[float], gripper: float, sim: SimSession) -> str:
    """6-DOF IK using mujoco analytic Jacobian (position + orientation).

    Keeps the hand pointing straight down while reaching the target position.
    Target [x, y, z] is in FINGERTIP space.
    """
    TARGET_POS   = 0.012   # m
    TARGET_ORI   = 0.05    # rad (sin of angle)
    MAX_STEPS    = 200
    DAMP         = 0.01    # damping for DLS
    ORI_WEIGHT   = 0.3     # scale orientation error vs position
    # Desired hand z-axis = pointing down
    DESIRED_ZAXIS = np.array([0.0, 0.0, -1.0])

    tx, ty, tz = target
    hand_target = np.array([tx, ty, tz + HAND_TO_TIP_Z])

    obs = sim.observe()
    ctrl_now = obs.get("robots", [{}])[0].get("ctrl", [0.0] * 8)
    joints = np.array(ctrl_now[:7])

    jacp = np.zeros((3, _ik_model.nv))
    jacr = np.zeros((3, _ik_model.nv))

    for _ in range(MAX_STEPS):
        for idx, addr in enumerate(_JOINT_ADDRS):
            _ik_data.qpos[addr] = joints[idx]
        mujoco.mj_forward(_ik_model, _ik_data)

        pos  = _ik_data.xpos[_HAND_ID].copy()
        zax  = _ik_data.xmat[_HAND_ID].reshape(3, 3)[:, 2].copy()

        pos_err = hand_target - pos
        ori_err = np.cross(zax, DESIRED_ZAXIS) * ORI_WEIGHT

        if np.linalg.norm(pos_err) < TARGET_POS and np.linalg.norm(ori_err) < TARGET_ORI * ORI_WEIGHT:
            break

        mujoco.mj_jacBody(_ik_model, _ik_data, jacp, jacr, _HAND_ID)
        Jp = jacp[:, _DOF_ADDRS]   # 3×7
        Jr = jacr[:, _DOF_ADDRS]   # 3×7

        # Stack position + weighted orientation rows
        J6 = np.vstack([Jp, Jr * ORI_WEIGHT])          # 6×7
        e6 = np.concatenate([pos_err, ori_err])         # 6

        # Damped least squares: dq = Jᵀ(JJᵀ + λI)⁻¹ e
        A = J6 @ J6.T + DAMP * np.eye(6)
        dq = J6.T @ np.linalg.solve(A, e6)
        joints = np.clip(joints + dq, JOINT_LIMITS[:, 0], JOINT_LIMITS[:, 1])

    # Final state
    for idx, addr in enumerate(_JOINT_ADDRS):
        _ik_data.qpos[addr] = joints[idx]
    mujoco.mj_forward(_ik_model, _ik_data)
    final_pos = _ik_data.xpos[_HAND_ID].copy()
    final_zax = _ik_data.xmat[_HAND_ID].reshape(3, 3)[:, 2].copy()
    final_tip_z = final_pos[2] - HAND_TO_TIP_Z
    pos_err_final = np.linalg.norm(hand_target - final_pos)

    sim.set_control(joints.tolist() + [gripper])

    return (
        f"moved: fingertips≈({final_pos[0]:.3f},{final_pos[1]:.3f},{final_tip_z:.3f}) "
        f"target=({tx:.3f},{ty:.3f},{tz:.3f}) err={pos_err_final:.3f}m "
        f"hand_down={final_zax[2]:.2f}"
    )


def execute_tool(name: str, inputs: dict, sim: SimSession) -> str:
    if name == "observe":
        return fmt_obs(sim.observe())

    if name == "move_ee":
        x = float(inputs.get("x", 0))
        y = float(inputs.get("y", 0))
        z = float(inputs.get("z", 0.3))
        gripper = float(inputs.get("gripper", 255))
        return move_ee_to([x, y, z], gripper, sim)

    if name == "set_control":
        ctrl = inputs.get("ctrl", [])
        if len(ctrl) != 8:
            return f"error: ctrl must have 8 values, got {len(ctrl)}"
        sim.set_control([float(v) for v in ctrl])
        return "control sent"

    if name == "reset":
        sim.reset()
        return "reset done"

    if name == "chat":
        sim.chat(inputs["message"])
        return "message sent"

    return f"unknown tool: {name}"


# ── agent loop ─────────────────────────────────────────────────────────────────

def build_system(sim: SimSession, api_doc: str) -> str:
    return f"""You control a Franka Panda robot arm in a MuJoCo simulation.
Your assigned robot: {sim.robot or "panda"}
Your session: {sim.session_id}

## Primary tool: move_ee
Use move_ee(x, y, z, gripper) to move the fingertips to world position (x, y, z).
The harness solves inverse kinematics automatically.
- gripper=255 → open, gripper=0 → closed
- z coordinates are fingertip heights. Table surface = z=0.0.
- Blocks are cubes with half-size 0.035m: centre z=0.035, top z=0.07.
- To grasp a block: fingertips at z=0.015 (just above table, around the block sides).
- Lift to z=0.25 before moving sideways.

## Typical pick-and-place workflow
1. observe() → note block position (bx, by, bz)
2. move_ee(bx, by, 0.15, 255)   — approach above block, open
3. move_ee(bx, by, 0.015, 255)  — lower fingers beside block
4. move_ee(bx, by, 0.015, 0)    — close gripper around block
5. move_ee(bx, by, 0.25, 0)     — lift
6. move_ee(tx, ty, 0.25, 0)     — move over target
7. move_ee(tx, ty, 0.085, 0)    — lower onto stack (1st block top = 0.07)
8. move_ee(tx, ty, 0.085, 255)  — open gripper, release
9. move_ee(tx, ty, 0.25, 255)   — retract

## Stacking heights (fingertip z to release)
- Place on table:        z=0.085  (block top at 0.07 + small gap)
- Place on 1st block:    z=0.155  (top at 0.14)
- Place on 2nd block:    z=0.225  (top at 0.21)
Each block adds ~0.07m.

## World layout (from observe)
Table centre ~(0.4, 0.0). Arm reach: x 0.1–0.8, y ±0.5, z 0.0–0.9.
"""


def run(task: str, model: str = MODEL, world_url: str = WORLD_URL,
        verbose: bool = True, thinking_budget: int = THINKING_BUDGET) -> str:
    if not API_KEY:
        print("MINIMAX_API_KEY not set", file=sys.stderr)
        sys.exit(1)

    sim = SimSession(world_url, AGENT_NAME)
    join_info = sim.join(AGENT_NAME)
    log.info("joined: name=%s robot=%s session=%s",
             join_info.get("name"), join_info.get("robot"),
             (join_info.get("session") or "")[:8])

    api_doc  = sim.api_doc()
    system   = build_system(sim, api_doc)
    client   = anthropic.Anthropic(api_key=API_KEY, base_url=BASE_URL)
    messages = [{"role": "user", "content": task}]
    log.info("task: %s  thinking_budget=%d", task, thinking_budget)

    # thinking params — same pattern as swarm-controller/llm_utils.py
    thinking_kwargs: dict = {}
    if thinking_budget > 0:
        thinking_kwargs["thinking"] = {"type": "enabled", "budget_tokens": thinking_budget}

    try:
        for turn in range(MAX_TURNS):
            log.info("turn %d — calling %s (thinking=%s)", turn + 1, model,
                     "on" if thinking_budget > 0 else "off")
            resp = client.messages.create(
                model=model,
                max_tokens=max(1024, thinking_budget + 1024) if thinking_budget > 0 else 1024,
                system=system,
                tools=TOOLS,
                messages=messages,
                **thinking_kwargs,
            )
            log.info("turn %d — stop_reason=%s input_tokens=%s output_tokens=%s",
                     turn + 1, resp.stop_reason,
                     resp.usage.input_tokens, resp.usage.output_tokens)

            messages.append({"role": "assistant", "content": resp.content})

            for block in resp.content:
                if block.type == "thinking":
                    snippet = block.thinking[:200].replace("\n", " ")
                    log.info("[think] %s%s", snippet, "…" if len(block.thinking) > 200 else "")
                elif hasattr(block, "text") and block.text:
                    log.info("[agent] %s", block.text)
                elif block.type == "tool_use":
                    log.info("[tool]  %s(%s)", block.name,
                             json.dumps(block.input, separators=(',', ':')))

            if resp.stop_reason == "end_turn":
                final = next((b.text for b in resp.content if hasattr(b, "text")), "done")
                log.info("done: %s", final)
                return final

            if resp.stop_reason != "tool_use":
                log.warning("unexpected stop: %s", resp.stop_reason)
                return f"stopped: {resp.stop_reason}"

            tool_results = []
            for block in resp.content:
                if block.type != "tool_use":
                    continue
                result = execute_tool(block.name, block.input, sim)
                log.info("[result] %s", result[:300])
                tool_results.append({
                    "type":        "tool_result",
                    "tool_use_id": block.id,
                    "content":     result,
                })

            messages.append({"role": "user", "content": tool_results})

    finally:
        sim.leave()
        log.info("session left")

    return "max turns reached"


# ── CLI ────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="MiniMax agent for MuJoCo Panda sim")
    parser.add_argument("task", nargs="+")
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--world", default=WORLD_URL)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--thinking", type=int, default=THINKING_BUDGET,
                        metavar="BUDGET", help="thinking budget tokens (0=off)")
    args = parser.parse_args()

    result = run(" ".join(args.task), model=args.model,
                 world_url=args.world, verbose=not args.quiet,
                 thinking_budget=args.thinking)
    print(f"\n[done] {result}")


if __name__ == "__main__":
    main()
