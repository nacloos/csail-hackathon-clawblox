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

WORLD_URL  = os.getenv("WORLD_URL", "http://localhost:8080")
MODEL      = os.getenv("MODEL", "minimax-m2.7")
BASE_URL   = "https://api.minimax.io/anthropic"
API_KEY    = os.getenv("MINIMAX_API_KEY", "")
MAX_TURNS  = 40
AGENT_NAME = "minimax-agent"

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
    for o in obs.get("objects", []):
        p = o["position"]
        lines.append(f"  {o['name']:<16} ({p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f})")
    touch = obs.get("touch", {})
    if touch:
        lines.append(f"touch L={touch.get('left',0):.2f} R={touch.get('right',0):.2f}")
    state = obs.get("state", {})
    ctrl  = state.get("ctrl", [])
    if ctrl:
        lines.append(f"ctrl: {[round(v,3) for v in ctrl]}")
    lines.append(f"t={obs.get('time',0):.2f}s")
    return "\n".join(lines)


def execute_tool(name: str, inputs: dict, sim: SimSession) -> str:
    if name == "observe":
        return fmt_obs(sim.observe())

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

Use tools to observe and act. Be direct — call tools immediately.

{api_doc}"""


def run(task: str, model: str = MODEL, world_url: str = WORLD_URL,
        verbose: bool = True) -> str:
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
    log.info("task: %s", task)

    try:
        for turn in range(MAX_TURNS):
            log.info("turn %d — calling %s", turn + 1, model)
            resp = client.messages.create(
                model=model,
                max_tokens=1024,
                system=system,
                tools=TOOLS,
                messages=messages,
            )
            log.info("turn %d — stop_reason=%s input_tokens=%s output_tokens=%s",
                     turn + 1, resp.stop_reason,
                     resp.usage.input_tokens, resp.usage.output_tokens)

            messages.append({"role": "assistant", "content": resp.content})

            for block in resp.content:
                if hasattr(block, "text") and block.text:
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
    args = parser.parse_args()

    result = run(" ".join(args.task), model=args.model,
                 world_url=args.world, verbose=not args.quiet)
    print(f"\n[done] {result}")


if __name__ == "__main__":
    main()
