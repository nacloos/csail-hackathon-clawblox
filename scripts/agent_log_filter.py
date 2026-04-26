#!/usr/bin/env python3
"""Translate claude --output-format stream-json into a human-readable log.

stdin: JSONL stream-json events from `claude --print --output-format stream-json`.
stdout: one short line per meaningful event, suitable for tailing agent.log.

Falls through any line that isn't valid JSON so the operator never loses output.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime


def ts() -> str:
    return datetime.now().strftime("%H:%M:%S")


def trunc(s: str, n: int = 220) -> str:
    s = (s or "").replace("\n", " ").replace("\r", " ").strip()
    return s if len(s) <= n else s[: n - 1] + "…"


def emit(line: str) -> None:
    print(line, flush=True)


def render_assistant(message: dict) -> None:
    for block in message.get("content") or []:
        btype = block.get("type")
        if btype == "text":
            text = block.get("text") or ""
            if text.strip():
                emit(f"{ts()} [say]   {trunc(text, 400)}")
        elif btype == "tool_use":
            name = block.get("name") or "?"
            inp = block.get("input") or {}
            if name == "Bash":
                cmd = inp.get("command", "")
                emit(f"{ts()} [bash]  {trunc(cmd, 220)}")
            elif name == "Read":
                emit(f"{ts()} [read]  {inp.get('file_path', '')}")
            elif name == "Write":
                fp = inp.get("file_path", "")
                content = inp.get("content", "")
                emit(f"{ts()} [write] {fp}  ({len(content)} chars)")
            elif name == "Edit":
                emit(f"{ts()} [edit]  {inp.get('file_path', '')}")
            else:
                emit(f"{ts()} [tool:{name}] {trunc(json.dumps(inp), 220)}")


def render_user(message: dict) -> None:
    for block in message.get("content") or []:
        if block.get("type") != "tool_result":
            continue
        content = block.get("content")
        if isinstance(content, list):
            text = " | ".join(
                (b.get("text") or "")[:200]
                for b in content
                if isinstance(b, dict) and b.get("type") == "text"
            )
        else:
            text = str(content) if content is not None else ""
        marker = "err" if block.get("is_error") else "out"
        emit(f"{ts()} [{marker}]   {trunc(text, 280)}")


def main() -> None:
    for raw in sys.stdin:
        raw = raw.rstrip("\n")
        if not raw.strip():
            continue
        try:
            ev = json.loads(raw)
        except json.JSONDecodeError:
            emit(raw)
            continue
        t = ev.get("type")
        if t == "system":
            sub = ev.get("subtype")
            if sub == "init":
                model = ev.get("model") or "?"
                emit(f"{ts()} [init]  model={model} session={ev.get('session_id','?')[:8]}")
        elif t == "assistant":
            render_assistant(ev.get("message") or {})
        elif t == "user":
            render_user(ev.get("message") or {})
        elif t == "result":
            sub = ev.get("subtype") or "?"
            cost = ev.get("total_cost_usd")
            turns = ev.get("num_turns")
            cost_s = f" cost=${cost:.4f}" if isinstance(cost, (int, float)) else ""
            turns_s = f" turns={turns}" if turns is not None else ""
            emit(f"{ts()} [done]  {sub}{cost_s}{turns_s}")
        # Other event types (stream_event partial deltas etc.) are skipped to
        # keep the log readable. Raw JSONL is kept in agent.events.jsonl.


if __name__ == "__main__":
    try:
        main()
    except (BrokenPipeError, KeyboardInterrupt):
        pass
