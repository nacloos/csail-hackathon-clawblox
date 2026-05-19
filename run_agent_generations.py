from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import os
from pathlib import Path
import re
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
DEFAULT_MODELS = {"claude": "claude-opus-4-7", "codex": "gpt-5.5"}
CHAIN_HEADER = [
    "generation",
    "agent_index",
    "agent_name",
    "run_id",
    "tmux_session",
    "template_in",
    "workspace_out",
    "start_utc",
    "end_utc",
    "duration_seconds",
]

HEALTH_TIMEOUT_SECONDS = int(os.environ.get("HEALTH_TIMEOUT", "30"))
PORT_WAIT_SECONDS = int(os.environ.get("PORT_WAIT_SECONDS", "120"))
STOP_GRACE_SECONDS = int(os.environ.get("STOP_GRACE_SECONDS", "10"))
SAVE_GRACE_SECONDS = int(os.environ.get("SAVE_GRACE_SECONDS", "300"))
GENERATION_RETRIES = int(os.environ.get("MAX_GENERATION_RETRIES", "3"))
RETRY_DELAY_SECONDS = int(os.environ.get("RETRY_DELAY_SECONDS", "15"))
SAVE_PROMPT = os.environ.get(
    "SAVE_PROMPT",
    "You will be reset in 5 minutes. Update your workspace memory files now.",
)
WORLD_POLL_SECONDS = 2


def parse_duration(value: str) -> int:
    value = value.strip()
    match = re.fullmatch(r"([1-9][0-9]*)([smh]?)", value)
    if not match:
        raise argparse.ArgumentTypeError("duration must be positive, e.g. 3600, 60m, or 1h")
    amount = int(match.group(1))
    unit = match.group(2)
    return amount * {"": 1, "s": 1, "m": 60, "h": 3600}[unit]


def utc_stamp(compact: bool = False) -> str:
    fmt = "%Y%m%dT%H%M%SZ" if compact else "%Y-%m-%dT%H:%M:%SZ"
    return datetime.now(timezone.utc).strftime(fmt)


def root_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", value)


def agent_auth_env() -> dict[str, str]:
    if os.environ.get("CLAUDE_CODE_OAUTH_TOKEN"):
        return {"CLAUDE_CODE_OAUTH_TOKEN": os.environ["CLAUDE_CODE_OAUTH_TOKEN"]}
    env_file = Path(os.environ.get("CLAWBLOX_ENV_FILE", ROOT / ".env"))
    if not env_file.is_file():
        return {}
    for raw in env_file.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if line.startswith("CLAUDE_CODE_OAUTH_TOKEN="):
            return {"CLAUDE_CODE_OAUTH_TOKEN": line.split("=", 1)[1].strip().strip("'\"")}
    return {}


def copy_workspace(template: Path, workspace: Path, world_dir: Path) -> None:
    if not template.is_dir():
        raise SystemExit(f"agent template directory not found: {template}")

    def ignore_workspace(_dir: str, names: list[str]) -> set[str]:
        return {
            name
            for name in names
            if name in {".venv", "__pycache__", ".pytest_cache"} or name.endswith(".pyc")
        }

    shutil.copytree(template, workspace, ignore=ignore_workspace, dirs_exist_ok=True)

    def ignore(_dir: str, names: list[str]) -> set[str]:
        return {
            name
            for name in names
            if name in {"results", "__pycache__", ".pytest_cache"} or name.endswith(".pyc")
        }

    shutil.copytree(world_dir, workspace / "world", ignore=ignore, dirs_exist_ok=True)


def render_system_prompt(path: Path, values: dict[str, str]) -> str:
    if not path.is_file():
        raise SystemExit(f"system prompt template not found: {path}")
    return string.Template(path.read_text(encoding="utf-8")).safe_substitute(values)


def world_ok(url: str) -> bool:
    try:
        with urlopen(Request(f"{url.rstrip('/')}/api.md"), timeout=2) as response:
            return response.status == 200
    except (OSError, URLError):
        return False


def tmux_window_exists(session: str, window: str) -> bool:
    result = subprocess.run(
        ["tmux", "list-windows", "-t", session, "-F", "#{window_name}"],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        check=False,
    )
    return result.returncode == 0 and window in result.stdout.splitlines()


def wait_while_running(
    seconds: int,
    *,
    url: str,
    tmux_session: str,
    world_log: Path,
    agent_log: Path,
    require_agent: bool = True,
) -> None:
    deadline = time.monotonic() + seconds
    failures = 0
    while time.monotonic() < deadline:
        if not tmux_window_exists(tmux_session, "world-0"):
            raise RuntimeError(f"world window exited; see {world_log}")
        if require_agent and not tmux_window_exists(tmux_session, "agent-0-0"):
            raise RuntimeError(f"agent window exited; see {agent_log}")
        failures = 0 if world_ok(url) else failures + 1
        if failures >= 3:
            raise RuntimeError(f"world stopped responding; see {world_log}")
        time.sleep(min(WORLD_POLL_SECONDS, max(0.1, deadline - time.monotonic())))


def load_resume(chain_csv: Path) -> tuple[int, Path, str | None, str | None]:
    with chain_csv.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return 1, Path(), None, None

    latest = max(rows, key=lambda row: int(row["generation"]))
    workspace = Path(latest["workspace_out"])
    if not workspace.is_dir():
        raise SystemExit(f"recorded workspace does not exist: {workspace}")
    run_prefix = re.sub(r"-g[0-9]{3}$", "", latest["run_id"])
    tmux_prefix = re.sub(r"-g[0-9]{3}$", "", latest["tmux_session"])
    return int(latest["generation"]) + 1, workspace, run_prefix, tmux_prefix


def validate_agent_dir(agent_dir: Path, backend: str) -> None:
    session_name = "codex_session_id.txt" if backend == "codex" else "claude_session_id.txt"
    for path in (
        agent_dir / "logs" / "agent.log",
        agent_dir / "runtime" / "system_prompt.md",
        agent_dir / session_name,
        agent_dir / "world_session.txt",
    ):
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"missing or empty required artifact: {path}")
    if not any((agent_dir / "workspace").iterdir()):
        raise RuntimeError(f"agent workspace is empty: {agent_dir / 'workspace'}")


def remove_tree(path: Path) -> None:
    if not path.exists():
        return

    def onerror(func: object, failed_path: str, _exc_info: object) -> None:
        os.chmod(failed_path, 0o700)
        func(failed_path)

    last_error: Exception | None = None
    for _ in range(5):
        try:
            shutil.rmtree(path, onerror=onerror)
            return
        except OSError as err:
            last_error = err
            time.sleep(0.2)
    if last_error is not None:
        raise last_error


def run_generation(args: argparse.Namespace, *, generation: int, template: Path, run_id: str, tmux_session: str) -> Path:
    world_dir = root_path(args.world_dir)
    run_dir = args.runs_dir / run_id
    if run_dir.exists():
        raise RuntimeError(f"run directory already exists: {run_dir}")

    agent_name = safe_name(f"Eko-r{run_id}-w0-a0")
    world_root = run_dir / "worlds" / "world-0"
    agent_dir = world_root / "agents" / agent_name
    workspace = agent_dir / "workspace"
    runtime = agent_dir / "runtime"
    recordings = world_root / "recordings"
    logs = world_root / "logs"
    for path in (workspace, runtime, recordings, logs):
        path.mkdir(parents=True, exist_ok=True)

    copy_workspace(template, workspace, world_dir)
    world = World(dir=world_dir)
    agent = Agent(
        agent=args.backend,
        name=agent_name,
        dir=agent_dir,
        model=args.model or DEFAULT_MODELS[args.backend],
        permission_mode=args.permission_mode,
        tmux=tmux_session,
        sandbox=args.sandbox,
        bare=args.bare,
    )

    started_agent = None
    started_at = utc_stamp()
    try:
        started_world = world.start(
            port=args.base_port,
            record=args.record,
            record_dir=recordings,
            health_timeout=HEALTH_TIMEOUT_SECONDS,
            port_wait=PORT_WAIT_SECONDS,
            command_file=runtime / "world.sh",
            log_file=logs / "world.log",
            _tmux_session=tmux_session,
        )
        access = world.connect(agent=agent)
        session = access["session"]
        base_url = access.get("base_url", started_world.url)
        system_prompt = render_system_prompt(
            root_path(args.system_prompt) if args.system_prompt else world_dir / "system_prompt.md",
            {
                "WORLD_AGENT_NAME": "Eko",
                "WORKSPACE_DIR": agent.visible_workspace_dir,
                "SESSION_LINE": f"X-Session: {session}",
                "WORLD_BASE_URL": base_url,
                "SKILL_CURL": f"curl -H 'X-Session: {session}'",
                "SKILL_URL": f"{base_url}/api.md",
            },
        )
        warning = min(SAVE_GRACE_SECONDS, args.generation_duration - 1) if args.generation_duration > 1 else 0
        active = args.generation_duration - warning
        auth_env = agent_auth_env()
        started_agent = agent.start(
            initial_prompt=args.goal or "Begin",
            system_prompt=system_prompt,
            env=auth_env or None,
        )

        for path in (Path(started_world.command_file), Path(started_agent.command_file)):
            path.unlink(missing_ok=True)

        print(f"  workspace: {workspace}", flush=True)
        print(f"  tmux:      tmux attach -t {shlex.quote(tmux_session)}", flush=True)
        print(f"  world:     {started_world.url}", flush=True)

        if warning > 0:
            wait_while_running(
                active,
                url=started_world.url,
                tmux_session=tmux_session,
                world_log=Path(started_world.log_file),
                agent_log=agent_dir / "logs" / "agent.log",
            )
            agent.stop(grace_seconds=STOP_GRACE_SECONDS)
            started_agent = None
            started_agent = agent.send(prompt=SAVE_PROMPT, env=auth_env or None)
            wait_while_running(
                warning,
                url=started_world.url,
                tmux_session=tmux_session,
                world_log=Path(started_world.log_file),
                agent_log=agent_dir / "logs" / "agent.log",
                require_agent=False,
            )
        else:
            wait_while_running(
                args.generation_duration,
                url=started_world.url,
                tmux_session=tmux_session,
                world_log=Path(started_world.log_file),
                agent_log=agent_dir / "logs" / "agent.log",
            )

        validate_agent_dir(agent_dir, args.backend)
        with args.chain_csv.open("a", encoding="utf-8", newline="") as handle:
            csv.writer(handle, quoting=csv.QUOTE_ALL).writerow(
                [
                    generation,
                    0,
                    agent_name,
                    run_id,
                    tmux_session,
                    str(template),
                    str(workspace),
                    started_at,
                    utc_stamp(),
                    args.generation_duration,
                ]
            )
        return workspace
    finally:
        if started_agent is not None:
            agent.stop(grace_seconds=STOP_GRACE_SECONDS)
        world.stop(grace_seconds=STOP_GRACE_SECONDS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a single-agent Clawblox generation chain.")
    parser.add_argument("--generations", type=int, default=int(os.environ.get("GENERATIONS", "5")))
    parser.add_argument("--generation-duration", type=parse_duration, default=parse_duration(os.environ.get("GENERATION_DURATION", "2h")))
    parser.add_argument("--experiment-id", default=os.environ.get("EXPERIMENT_ID", f"genexp-python-{utc_stamp(True)}"))
    parser.add_argument("--run-prefix", default=os.environ.get("RUN_PREFIX"))
    parser.add_argument("--tmux-prefix", default=os.environ.get("TMUX_PREFIX", "clawblox-gen-python"))
    parser.add_argument("--world-dir", type=Path, default=Path(os.environ.get("WORLD_DIR", "worlds/mujoco-panda")))
    parser.add_argument("--base-port", type=int, default=int(os.environ.get("BASE_PORT", "8085")))
    parser.add_argument("--backend", choices=sorted(DEFAULT_MODELS), default=os.environ.get("BACKEND", "claude"))
    parser.add_argument("--model", default=os.environ.get("MODEL"))
    parser.add_argument("--permission-mode", default=os.environ.get("PERMISSION_MODE", "bypassPermissions"))
    parser.add_argument("--record", action=argparse.BooleanOptionalAction, default=os.environ.get("RECORD", "true") != "false")
    parser.add_argument("--sandbox", action=argparse.BooleanOptionalAction, default=os.environ.get("SANDBOX", "true") != "false")
    parser.add_argument("--bare", action="store_true", default=os.environ.get("BARE", "false") == "true")
    parser.add_argument("--goal", default=os.environ.get("GOAL", ""))
    parser.add_argument("--template", type=Path, default=Path(os.environ.get("TEMPLATE", DEFAULT_TEMPLATE)))
    parser.add_argument("--system-prompt", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.generations <= 0:
        raise SystemExit("--generations must be positive")
    if args.base_port <= 0 or args.base_port > 65535:
        raise SystemExit("--base-port must be in [1, 65535]")
    return args


def main() -> None:
    args = parse_args()
    world_dir = root_path(args.world_dir)
    if not world_dir.is_dir():
        raise SystemExit(f"world directory not found: {world_dir}")

    experiment_dir = world_dir / "results" / safe_name(args.experiment_id)
    args.runs_dir = experiment_dir / "runs"
    args.chain_csv = experiment_dir / "generation_chain.csv"
    run_prefix = args.run_prefix or args.experiment_id
    tmux_prefix = args.tmux_prefix
    start_generation = 1
    template = root_path(args.template)

    if args.resume:
        if not args.chain_csv.is_file():
            raise SystemExit(f"--resume requested but metadata CSV is missing: {args.chain_csv}")
        start_generation, resumed_template, inferred_run, inferred_tmux = load_resume(args.chain_csv)
        template = resumed_template
        run_prefix = args.run_prefix or inferred_run or run_prefix
        tmux_prefix = inferred_tmux or tmux_prefix
    else:
        if experiment_dir.exists():
            if not args.force:
                raise SystemExit(f"experiment directory already exists: {experiment_dir} (use --resume or --force)")
            remove_tree(experiment_dir)
        args.runs_dir.mkdir(parents=True)
        with args.chain_csv.open("w", encoding="utf-8", newline="") as handle:
            csv.writer(handle).writerow(CHAIN_HEADER)

    if not template.is_dir():
        raise SystemExit(f"agent template directory not found: {template}")

    print("Starting Python generation chain", flush=True)
    print(f"  Generations: {args.generations}", flush=True)
    print(f"  Duration each: {args.generation_duration}s", flush=True)
    print(f"  Experiment metadata: {experiment_dir}", flush=True)
    print("", flush=True)

    for generation in range(start_generation, args.generations + 1):
        run_id = safe_name(f"{run_prefix}-g{generation:03d}")
        tmux_session = safe_name(f"{tmux_prefix}-g{generation:03d}")
        print(f"=== Generation {generation}/{args.generations} ===", flush=True)
        print(f"  run_id: {run_id}", flush=True)
        print(f"  template: {template}", flush=True)

        for attempt in range(GENERATION_RETRIES + 1):
            try:
                template = run_generation(
                    args,
                    generation=generation,
                    template=template,
                    run_id=run_id,
                    tmux_session=tmux_session,
                )
                break
            except Exception as err:
                try:
                    remove_tree(args.runs_dir / run_id)
                except Exception as cleanup_err:
                    print(f"  cleanup warning: {cleanup_err}", flush=True)
                if attempt == GENERATION_RETRIES:
                    raise SystemExit(f"generation {generation} failed: {err}") from err
                print(f"  attempt {attempt + 1} failed: {err}", flush=True)
                time.sleep(RETRY_DELAY_SECONDS)
        print("", flush=True)

    print("Python generation chain complete.", flush=True)
    print(f"Metadata CSV: {args.chain_csv}", flush=True)


if __name__ == "__main__":
    main()
