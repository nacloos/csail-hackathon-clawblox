from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import shutil
import signal
import string
import time

from clawblox import Agent, World


ROOT = Path(__file__).resolve().parent
DEFAULT_TEMPLATE = ROOT / "agent" / "template" / "agent"
DEFAULT_MODEL = "claude-opus-4-6"
DEFAULT_SAVE_PROMPT = "You will be reset in 5 minutes. Update your workspace memory files now."


def parse_duration(value: str) -> int:
    value = value.strip()
    if not value:
        raise argparse.ArgumentTypeError("duration cannot be empty")
    unit = value[-1]
    if unit in {"s", "m", "h"}:
        amount_text = value[:-1]
        multiplier = {"s": 1, "m": 60, "h": 3600}[unit]
    else:
        amount_text = value
        multiplier = 1
    if not amount_text.isdigit() or int(amount_text) <= 0:
        raise argparse.ArgumentTypeError("duration must be positive, e.g. 3600, 60m, or 1h")
    return int(amount_text) * multiplier


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def copy_template(src: Path, dst: Path) -> None:
    if not src.is_dir():
        raise SystemExit(f"agent template directory not found: {src}")
    shutil.copytree(src, dst, dirs_exist_ok=True)


def copy_world_source(src: Path, dst: Path) -> None:
    def ignore(_dir: str, names: list[str]) -> set[str]:
        return {
            name
            for name in names
            if name in {"results", "__pycache__", ".pytest_cache"} or name.endswith(".pyc")
        }

    shutil.copytree(src, dst, ignore=ignore, dirs_exist_ok=True)


def render_prompt(template_path: Path, values: dict[str, str]) -> str:
    if not template_path.is_file():
        raise SystemExit(f"system prompt template not found: {template_path}")
    template = string.Template(template_path.read_text(encoding="utf-8"))
    return template.safe_substitute(values)


def wait_for(duration_seconds: int, stop_requested: list[bool]) -> None:
    deadline = time.monotonic() + duration_seconds
    while not stop_requested[0] and time.monotonic() < deadline:
        time.sleep(min(5, max(0.1, deadline - time.monotonic())))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one Clawblox Python world and one agent.")
    parser.add_argument("--world-dir", type=Path, default=Path("worlds/mujoco-panda"))
    parser.add_argument("--duration", type=parse_duration, default=parse_duration("1h"))
    parser.add_argument("--base-port", type=int, default=8085)
    parser.add_argument("--run-id", default=f"python-agent-{utc_stamp()}")
    parser.add_argument("--tmux-session")
    parser.add_argument("--agent-name", default="Eko")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--permission-mode", default="bypassPermissions")
    parser.add_argument("--record", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--sandbox", action="store_true")
    parser.add_argument("--bare", action="store_true")
    parser.add_argument("--goal", default="")
    parser.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE)
    parser.add_argument("--system-prompt", type=Path)
    parser.add_argument("--results-root", type=Path)
    parser.add_argument("--health-timeout", type=int, default=30)
    parser.add_argument("--port-wait-seconds", type=int, default=120)
    parser.add_argument("--stop-grace-seconds", type=int, default=10)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    world_dir = args.world_dir if args.world_dir.is_absolute() else ROOT / args.world_dir
    if not world_dir.is_dir():
        raise SystemExit(f"world directory not found: {world_dir}")

    results_root = args.results_root or world_dir / "results"
    if not results_root.is_absolute():
        results_root = ROOT / results_root
    run_dir = results_root / args.run_id
    if run_dir.exists():
        if not args.force:
            raise SystemExit(f"run directory already exists: {run_dir} (use --force to replace it)")
        shutil.rmtree(run_dir)

    world_root = run_dir / "worlds" / "world-0"
    agent_dir = world_root / "agents" / args.agent_name
    workspace_dir = agent_dir / "workspace"
    runtime_dir = agent_dir / "runtime"
    recordings_dir = world_root / "recordings"
    logs_dir = world_root / "logs"
    for path in (workspace_dir, runtime_dir, recordings_dir, logs_dir):
        path.mkdir(parents=True, exist_ok=True)

    copy_template(args.template if args.template.is_absolute() else ROOT / args.template, workspace_dir)
    copy_world_source(world_dir, workspace_dir / "world")

    world = World(dir=world_dir)
    agent = Agent(
        agent="claude",
        name=args.agent_name,
        dir=agent_dir,
        model=args.model,
        permission_mode=args.permission_mode,
        tmux=args.tmux_session or args.run_id,
        sandbox=args.sandbox,
        bare=args.bare,
    )

    stop_requested = [False]

    def request_stop(_signum: int, _frame: object) -> None:
        stop_requested[0] = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    started_agent = None
    try:
        started_world = world.start(
            port=args.base_port,
            record=args.record,
            record_dir=recordings_dir,
            health_timeout=args.health_timeout,
            port_wait=args.port_wait_seconds,
            command_file=runtime_dir / "world.sh",
            log_file=logs_dir / "world.log",
        )
        access = world.connect(agent=agent)
        session = access["session"]
        session_line = f"X-Session: {session}"
        system_prompt_template = args.system_prompt or world_dir / "system_prompt.md"
        system_prompt = render_prompt(
            system_prompt_template,
            {
                "WORLD_AGENT_NAME": args.agent_name,
                "WORKSPACE_DIR": agent.visible_workspace_dir,
                "SESSION_LINE": session_line,
                "WORLD_BASE_URL": started_world.url,
                "SKILL_CURL": f"curl -H 'X-Session: {session}'",
                "SKILL_URL": f"{started_world.url}/api.md",
            },
        )
        initial_prompt = args.goal or "Begin"
        started_agent = agent.start(
            initial_prompt=initial_prompt,
            system_prompt=system_prompt,
            env={
                "WORLD_BASE_URL": started_world.url,
                "WORLD_INTERNAL_BASE_URL": started_world.url,
                "WORLD_AGENT_NAME": args.agent_name,
                "WORKSPACE_DIR": str(workspace_dir),
            },
        )

        print(f"world: {started_world.url}", flush=True)
        print(f"world log: {started_world.log_file}", flush=True)
        print(f"agent: {started_agent.attach_command}", flush=True)
        print(f"agent dir: {started_agent.agent_dir}", flush=True)
        wait_for(args.duration, stop_requested)
    finally:
        if started_agent is not None:
            agent.stop(grace_seconds=args.stop_grace_seconds)
        world.stop(grace_seconds=args.stop_grace_seconds)


if __name__ == "__main__":
    main()
