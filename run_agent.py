from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import shlex
import string
import subprocess
import sys
import time
from urllib.error import URLError
from urllib.request import Request, urlopen

from clawblox import Agent, World


ROOT = Path(__file__).resolve().parent
DEFAULT_TEMPLATE = ROOT / "agent" / "template" / "agent"
DEFAULT_MODELS = {
    "claude": "claude-opus-4-7",
    "codex": "gpt-5.5",
}
DEFAULT_CLAUDE_CODE_VERSION_PIN = "2.1.116"
HEALTH_TIMEOUT_SECONDS = 30
PORT_WAIT_SECONDS = 120
STOP_GRACE_SECONDS = 1
WORLD_POLL_SECONDS = 2
WORLD_FAILURES_BEFORE_ABORT = 3
WORLD_LOG_TAIL_LINES = 80


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


def claude_auth_env() -> dict[str, str]:
    key = "CLAUDE_CODE_OAUTH_TOKEN"
    if token := os.environ.get(key):
        return {key: token}
    env_file = Path(os.environ.get("CLAWBLOX_ENV_FILE", ROOT / ".env"))
    if not env_file.is_file():
        return {}
    for line in env_file.read_text(encoding="utf-8").splitlines():
        if line.startswith(f"{key}="):
            return {key: line.split("=", 1)[1].strip().strip("'\"")}
    return {}


def resolve_claude_binary() -> Path | None:
    explicit = os.environ.get("CLAWBLOX_CLAUDE_BIN", "").strip()
    if explicit:
        path = Path(explicit).expanduser()
        if not path.is_file():
            raise SystemExit(f"CLAWBLOX_CLAUDE_BIN does not exist: {path}")
        if not os.access(path, os.X_OK):
            raise SystemExit(f"CLAWBLOX_CLAUDE_BIN is not executable: {path}")
        return path

    version_pin = os.environ.get(
        "CLAWBLOX_CLAUDE_CODE_VERSION_PIN",
        DEFAULT_CLAUDE_CODE_VERSION_PIN,
    ).strip()
    if not version_pin:
        return None

    path = Path.home() / ".local" / "share" / "claude" / "versions" / version_pin
    if path.is_file() and os.access(path, os.X_OK):
        return path

    raise SystemExit(
        f"pinned Claude Code {version_pin} not found at {path}. "
        "Set CLAWBLOX_CLAUDE_BIN=/path/to/claude, or set "
        "CLAWBLOX_CLAUDE_CODE_VERSION_PIN= to use claude from PATH."
    )


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


def world_is_healthy(url: str) -> bool:
    request = Request(f"{url.rstrip('/')}/api.md", method="GET")
    try:
        with urlopen(request, timeout=2) as response:
            return response.status == 200
    except (OSError, URLError):
        return False


def tail_text(path: Path, max_lines: int) -> str:
    if not path.is_file():
        return "(log file does not exist)"
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(lines[-max_lines:]) or "(log file is empty)"


def read_events(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    events = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return events


def toml_string(value: str) -> str:
    return json.dumps(value)


def latest_agent_transcript(agent_dir: Path) -> Path | None:
    session_dir = agent_dir / "session"
    transcripts = sorted(
        session_dir.glob("*.jsonl"),
        key=lambda path: path.stat().st_mtime if path.exists() else 0,
        reverse=True,
    )
    if transcripts:
        return transcripts[0]
    return None


def enrich_replay_manifests(world_root: Path) -> None:
    agents_dir = world_root / "agents"
    recordings_dir = world_root / "recordings"
    if not agents_dir.is_dir() or not recordings_dir.is_dir():
        return

    agents = []
    for agent_dir in sorted(path for path in agents_dir.iterdir() if path.is_dir()):
        transcript = latest_agent_transcript(agent_dir)
        if transcript is None:
            continue
        agents.append((agent_dir.name, transcript))
    if not agents:
        return

    for manifest in sorted(recordings_dir.glob("*.replay.toml")):
        raw = manifest.read_text(encoding="utf-8")
        if "[[agents]]" in raw:
            continue
        lines = [raw.rstrip(), ""]
        for agent_id, transcript in agents:
            rel_transcript = os.path.relpath(transcript, manifest.parent)
            lines.extend(
                [
                    "[[agents]]",
                    f"id = {toml_string(agent_id)}",
                    f"name = {toml_string(agent_id)}",
                    f"transcript = {toml_string(rel_transcript)}",
                    "",
                ]
            )
        manifest.write_text("\n".join(lines), encoding="utf-8")


def agent_failure_diagnostics(agent_dir: Path) -> str:
    events_file = agent_dir / "events.jsonl"
    runtime_dir = agent_dir / "runtime"
    diagnostics = []

    events = read_events(events_file)
    process_events = [
        event for event in events if event.get("operation") == "agent.process"
    ]
    failed_events = [
        event for event in events if event.get("status") == "failed" or event.get("error")
    ]
    if process_events:
        event = process_events[-1]
        diagnostics.append(
            "Last process event: "
            f"{event.get('status', 'unknown')} "
            f"exit_code={event.get('exit_code', 'unknown')} "
            f"error={event.get('error') or 'none'}"
        )
    if failed_events:
        event = failed_events[-1]
        diagnostics.append(
            "Last failure event: "
            f"{event.get('operation', 'unknown')} "
            f"error={event.get('error') or 'none'}"
        )

    settings_file = runtime_dir / "claude_sandbox_settings.json"
    if settings_file.is_file():
        try:
            settings = json.loads(settings_file.read_text(encoding="utf-8"))
            sandbox = settings.get("sandbox", {})
            diagnostics.append(
                "Claude native sandbox: "
                f"enabled={sandbox.get('enabled')} "
                f"failIfUnavailable={sandbox.get('failIfUnavailable')}"
            )
        except json.JSONDecodeError:
            diagnostics.append(f"Claude sandbox settings could not be parsed: {settings_file}")

    start_file = runtime_dir / "start.sh"
    if start_file.is_file():
        start_text = start_file.read_text(encoding="utf-8", errors="replace")
        has_deps_mount = "/sandbox-deps" in start_text
        diagnostics.append(f"Agent command file: {start_file}")
        diagnostics.append(f"Sandbox deps mounted: {has_deps_mount}")
        if "failIfUnavailable" in tail_text(settings_file, 20) and not has_deps_mount:
            diagnostics.append(
                "Hint: Claude native sandbox is strict, but sandbox deps are not mounted. "
                "Set SANDBOX_DEPS_ROOT or use a Clawblox build with automatic sandbox deps detection."
            )
    else:
        diagnostics.append(f"Agent command file missing: {start_file}")

    if not diagnostics:
        return "(no structured agent diagnostics found)"
    return "\n".join(diagnostics)


def tmux_window_exists(session: str, window: str) -> bool:
    result = subprocess.run(
        ["tmux", "list-windows", "-t", session, "-F", "#{window_name}"],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        check=False,
    )
    return result.returncode == 0 and window in result.stdout.splitlines()


def abort_with_log(title: str, log_file: Path, *, agent_dir: Path | None = None) -> None:
    print("", file=sys.stderr, flush=True)
    print(title, file=sys.stderr, flush=True)
    print(f"Log: {log_file}", file=sys.stderr, flush=True)
    print("", file=sys.stderr, flush=True)
    print(f"Last {WORLD_LOG_TAIL_LINES} log lines", file=sys.stderr, flush=True)
    print("-------------------------", file=sys.stderr, flush=True)
    print(tail_text(log_file, WORLD_LOG_TAIL_LINES), file=sys.stderr, flush=True)
    if agent_dir is not None:
        print("", file=sys.stderr, flush=True)
        print("Structured diagnostics", file=sys.stderr, flush=True)
        print("----------------------", file=sys.stderr, flush=True)
        print(agent_failure_diagnostics(agent_dir), file=sys.stderr, flush=True)
    raise SystemExit(1)


def wait_for_world(
    duration_seconds: int,
    stop_requested: list[bool],
    *,
    world_url: str,
    world_log: Path,
    tmux_session: str,
    world_window: str,
    agent_window: str,
    agent_log: Path,
    agent_dir: Path,
) -> None:
    deadline = time.monotonic() + duration_seconds
    failures = 0
    while not stop_requested[0] and time.monotonic() < deadline:
        if not tmux_window_exists(tmux_session, world_window):
            abort_with_log(f"World tmux window exited: {tmux_session}:{world_window}", world_log)
        if not tmux_window_exists(tmux_session, agent_window):
            abort_with_log(
                f"Agent tmux window exited: {tmux_session}:{agent_window}",
                agent_log,
                agent_dir=agent_dir,
            )
        if world_is_healthy(world_url):
            failures = 0
        else:
            failures += 1
            if failures >= WORLD_FAILURES_BEFORE_ABORT:
                abort_with_log(f"World stopped responding: {world_url}", world_log)
        time.sleep(min(WORLD_POLL_SECONDS, max(0.1, deadline - time.monotonic())))


def format_duration(seconds: int) -> str:
    if seconds % 3600 == 0:
        return f"{seconds // 3600}h"
    if seconds % 60 == 0:
        return f"{seconds // 60}m"
    return f"{seconds}s"


def remove_host_launch_artifacts(*paths: Path) -> None:
    for path in paths:
        try:
            path.unlink()
        except FileNotFoundError:
            pass


def spectator_url_from_log(path: Path) -> str | None:
    if not path.is_file():
        return None
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        marker = "Spectator frontend:"
        if marker in line:
            return line.split(marker, 1)[1].strip()
    return None


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def print_run_summary(
    *,
    run_id: str,
    tmux_session: str,
    duration_seconds: int,
    started_world,
    started_agent,
    world_base_url: str,
    world_internal_base_url: str,
    spectator_url: str | None,
    run_dir: Path,
    recordings_dir: Path,
    workspace_dir: Path,
    session: str,
    backend: str,
    model: str,
) -> None:
    print("", flush=True)
    print("Clawblox run started", flush=True)
    print("--------------------", flush=True)
    print(f"Run:       {run_id}", flush=True)
    print(f"Agent:     {backend} ({model})", flush=True)
    print(f"Duration:  {format_duration(duration_seconds)}", flush=True)
    print(f"World:     {world_internal_base_url}", flush=True)
    if world_base_url != world_internal_base_url:
        print(f"Agent API: {world_base_url}", flush=True)
    if spectator_url:
        print(f"Spectator: {spectator_url}", flush=True)
    print(f"Tmux:      tmux attach -t {shlex.quote(tmux_session)}", flush=True)
    if getattr(started_agent, "tmux_pane_id", None):
        print(f"Pane:      {started_agent.tmux_pane_id}", flush=True)
    print("", flush=True)
    print("Windows:   world-0 (server), agent-0-0 (agent)", flush=True)
    print(f"Results:   {rel(run_dir)}", flush=True)
    print(f"Workspace: {rel(workspace_dir)}", flush=True)
    if (workspace_dir / ".venv").is_dir():
        print(f"Python:    {rel(workspace_dir / '.venv')} ready", flush=True)
    print(f"Logs:      {rel(Path(started_world.log_file))}", flush=True)
    print(f"           {rel(Path(started_agent.agent_dir) / 'logs' / 'agent.log')}", flush=True)
    print("", flush=True)
    print(f"API:       curl -H 'X-Session: {session}' {world_base_url}/observe", flush=True)
    print(f"Replay:    recordings in {rel(recordings_dir)}", flush=True)
    print("", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one Clawblox Python world and one agent.")
    parser.add_argument("--world-dir", type=Path, default=Path("worlds/mujoco-panda"))
    parser.add_argument("--duration", type=parse_duration, default=parse_duration("1h"))
    parser.add_argument("--base-port", type=int, default=8085)
    parser.add_argument("--run-id", default=f"python-agent-{utc_stamp()}")
    parser.add_argument("--tmux-session")
    parser.add_argument("--backend", choices=sorted(DEFAULT_MODELS), default="claude")
    parser.add_argument("--agent-name", default="Eko")
    parser.add_argument("--model")
    parser.add_argument("--permission-mode", default="bypassPermissions")
    parser.add_argument("--record", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--sandbox", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--bare", action="store_true")
    parser.add_argument("--goal", default="")
    parser.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE)
    parser.add_argument("--system-prompt", type=Path)
    parser.add_argument("--results-root", type=Path)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    backend = args.backend
    model = args.model or DEFAULT_MODELS[backend]

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
    tmux_session = args.tmux_session or args.run_id
    agent = Agent(
        agent=backend,
        name=args.agent_name,
        dir=agent_dir,
        model=model,
        binary=resolve_claude_binary() if backend == "claude" else None,
        permission_mode=args.permission_mode,
        tmux=tmux_session,
        sandbox=args.sandbox,
        bare=args.bare,
    )

    stop_requested = [False]
    started_agent = None
    try:
        started_world = world.start(
            port=args.base_port,
            record=args.record,
            record_dir=recordings_dir,
            health_timeout=HEALTH_TIMEOUT_SECONDS,
            port_wait=PORT_WAIT_SECONDS,
            command_file=runtime_dir / "world.sh",
            log_file=logs_dir / "world.log",
            _tmux_session=tmux_session,
        )
        access = world.connect(agent=agent)
        session = access["session"]
        world_base_url = access.get("base_url", started_world.url)
        world_internal_base_url = access.get("internal_base_url", started_world.url)
        session_line = f"X-Session: {session}"
        system_prompt_template = args.system_prompt or world_dir / "system_prompt.md"
        system_prompt = render_prompt(
            system_prompt_template,
            {
                "WORLD_AGENT_NAME": args.agent_name,
                "WORKSPACE_DIR": agent.visible_workspace_dir,
                "SESSION_LINE": session_line,
                "WORLD_BASE_URL": world_base_url,
                "SKILL_CURL": f"curl -H 'X-Session: {session}'",
                "SKILL_URL": f"{world_base_url}/api.md",
            },
        )
        initial_prompt = args.goal or "Begin"
        agent_env = claude_auth_env()
        started_agent = agent.start(
            initial_prompt=initial_prompt,
            system_prompt=system_prompt,
            env=agent_env or None,
        )
        remove_host_launch_artifacts(
            Path(started_world.command_file),
        )

        print_run_summary(
            run_id=args.run_id,
            tmux_session=tmux_session,
            duration_seconds=args.duration,
            started_world=started_world,
            started_agent=started_agent,
            world_base_url=world_base_url,
            world_internal_base_url=world_internal_base_url,
            spectator_url=spectator_url_from_log(Path(started_world.log_file)),
            run_dir=run_dir,
            recordings_dir=recordings_dir,
            workspace_dir=workspace_dir,
            session=session,
            backend=backend,
            model=model,
        )
        wait_for_world(
            args.duration,
            stop_requested,
            world_url=started_world.url,
            world_log=Path(started_world.log_file),
            tmux_session=tmux_session,
            world_window="world-0",
            agent_window="agent-0-0",
            agent_log=Path(started_agent.agent_dir) / "logs" / "agent.log",
            agent_dir=Path(started_agent.agent_dir),
        )
    finally:
        if started_agent is not None:
            agent.stop(grace_seconds=STOP_GRACE_SECONDS)
        world.stop(grace_seconds=STOP_GRACE_SECONDS)
        if args.record:
            enrich_replay_manifests(world_root)


if __name__ == "__main__":
    main()
