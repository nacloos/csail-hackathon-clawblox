from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
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

from clawblox import Agent, World, load_checkpoint, read_metadata, save_checkpoint
from clawblox.checkpoint import CheckpointError


ROOT = Path(__file__).resolve().parent
DEFAULT_TEMPLATE = ROOT / "agent" / "template" / "agent"
DEFAULT_MODELS = {"claude": "claude-opus-4-8", "codex": "gpt-5.5"}
AGENT_DISPLAY_NAMES = ("Eko", "Moa", "Rua", "Tavi", "Oni", "Zev", "Ika", "Pala")
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
STOP_GRACE_SECONDS = int(os.environ.get("STOP_GRACE_SECONDS", "1"))
SAVE_GRACE_SECONDS = int(os.environ.get("SAVE_GRACE_SECONDS", "300"))
GENERATION_RETRIES = int(os.environ.get("MAX_GENERATION_RETRIES", "3"))
RETRY_DELAY_SECONDS = int(os.environ.get("RETRY_DELAY_SECONDS", "15"))
SAVE_PROMPT = os.environ.get(
    "SAVE_PROMPT",
    "You will be reset in 5 minutes. Update your workspace memory files now.",
)
WORLD_POLL_SECONDS = 2
CHECKPOINT_EVERY_SECONDS = int(os.environ.get("CHECKPOINT_EVERY", "900"))
CHECKPOINT_IDLE_QUIET_SECONDS = 5
CHECKPOINT_IDLE_WAIT_SECONDS = 60
# The exact prompt the blocking Stop hook uses for unattended auto-continue.
# Resuming with the same text keeps a restored run indistinguishable from a
# routine idle nudge: checkpoints stay invisible to the agent.
AUTO_CONTINUE_PROMPT = "Please don't stop. Just continue doing whatever you want to do."


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


def copy_workspace(template: Path, workspace: Path, world_dir: Path, *, copy_world: bool) -> None:
    if not template.is_dir():
        raise SystemExit(f"agent template directory not found: {template}")

    def ignore_workspace(_dir: str, names: list[str]) -> set[str]:
        return {
            name
            for name in names
            if name in {".venv", "__pycache__", ".pytest_cache"} or name.endswith(".pyc")
        }

    shutil.copytree(template, workspace, ignore=ignore_workspace, dirs_exist_ok=True)

    if copy_world:
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


class Checkpointer:
    """Periodic, agent-invisible checkpoints of a running generation.

    World state is captured via ``world.save`` (exact, lock-protected);
    agent state is archived from disk without sending anything to the agent.
    Snapshots prefer an idle moment (agent log quiet) but fall back to a
    crash-consistent archive: transcripts are append-only, so a mid-turn
    snapshot resumes from the last completed message.
    """

    def __init__(
        self,
        *,
        world: World,
        run_dir: Path,
        experiment_dir: Path,
        generation: int,
        agents: dict[str, Path],
        agent_logs: list[Path],
        backend: str,
        metadata: dict[str, object],
        interval: int = CHECKPOINT_EVERY_SECONDS,
        elapsed_offset: int = 0,
    ) -> None:
        self.world = world
        self.run_dir = run_dir
        self.experiment_dir = experiment_dir
        self.generation = generation
        self.agents = agents
        self.agent_logs = agent_logs
        self.backend = backend
        self.metadata = metadata
        self.interval = interval
        self.elapsed_offset = elapsed_offset
        self.started_mono = time.monotonic()
        self.next_due = self.started_mono + interval if interval > 0 else None
        self.idle_wait_until: float | None = None

    def elapsed_seconds(self) -> int:
        return self.elapsed_offset + int(time.monotonic() - self.started_mono)

    def agents_idle(self) -> bool:
        now = time.time()
        for log in self.agent_logs:
            try:
                if now - log.stat().st_mtime < CHECKPOINT_IDLE_QUIET_SECONDS:
                    return False
            except OSError:
                return False
        return True

    def maybe_checkpoint(self) -> None:
        if self.next_due is None or time.monotonic() < self.next_due:
            return
        if not self.agents_idle():
            if self.idle_wait_until is None:
                self.idle_wait_until = time.monotonic() + CHECKPOINT_IDLE_WAIT_SECONDS
            if time.monotonic() < self.idle_wait_until:
                return
        self.idle_wait_until = None
        self.next_due = time.monotonic() + self.interval
        try:
            self.checkpoint()
        except Exception as err:
            print(f"  checkpoint warning: {err}", flush=True)

    def checkpoint(self) -> Path:
        elapsed = self.elapsed_seconds()
        checkpoints_dir = self.run_dir / "checkpoints"
        snapshot = self.world.save(dir=checkpoints_dir)
        ckpt_path = save_checkpoint(
            checkpoints_dir / f"gen{self.generation:03d}-t{elapsed:06d}.ckpt",
            world_snapshot=snapshot["path"],
            agents=self.agents,
            backend=self.backend,
            metadata={
                "generation": self.generation,
                "elapsed_seconds": elapsed,
                **self.metadata,
            },
        )
        Path(snapshot["path"]).unlink(missing_ok=True)
        pointer = {
            "checkpoint": str(ckpt_path),
            "generation": self.generation,
            "elapsed_seconds": elapsed,
            "written_at": utc_stamp(),
        }
        pointer_path = self.experiment_dir / "checkpoint.json"
        pointer_tmp = pointer_path.with_suffix(".json.tmp")
        pointer_tmp.write_text(json.dumps(pointer, indent=2) + "\n", encoding="utf-8")
        pointer_tmp.replace(pointer_path)
        print(f"  checkpoint: {ckpt_path} (t={elapsed}s)", flush=True)
        return ckpt_path


def wait_while_running(
    seconds: int,
    *,
    url: str,
    tmux_session: str,
    world_log: Path,
    agent_logs: list[Path],
    require_agent: bool = True,
    agent_windows: list[str] | None = None,
    checkpointer: Checkpointer | None = None,
) -> None:
    deadline = time.monotonic() + seconds
    failures = 0
    while time.monotonic() < deadline:
        if not tmux_window_exists(tmux_session, "world-0"):
            raise RuntimeError(f"world window exited; see {world_log}")
        if require_agent:
            windows = agent_windows or [f"agent-0-{idx}" for idx in range(len(agent_logs))]
            for idx, (window, agent_log) in enumerate(zip(windows, agent_logs)):
                if not tmux_window_exists(tmux_session, window):
                    raise RuntimeError(f"agent window {window} exited; see {agent_log}")
        failures = 0 if world_ok(url) else failures + 1
        if failures >= 3:
            raise RuntimeError(f"world stopped responding; see {world_log}")
        if checkpointer is not None:
            checkpointer.maybe_checkpoint()
        time.sleep(min(WORLD_POLL_SECONDS, max(0.1, deadline - time.monotonic())))


def load_resume(
    chain_csv: Path, agents_per_world: int
) -> tuple[int, list[Path], str | None, str | None]:
    with chain_csv.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise SystemExit(
            f"--resume requested but no completed generation is recorded in {chain_csv}; "
            "to resume a generation that crashed mid-flight, use --resume-from with a "
            "checkpoint from the run's checkpoints/ directory"
        )

    latest_generation = max(int(row["generation"]) for row in rows)
    latest_rows = [row for row in rows if int(row["generation"]) == latest_generation]
    if len(latest_rows) != agents_per_world:
        raise SystemExit(
            f"resume metadata has {len(latest_rows)} agents for generation "
            f"{latest_generation}, expected {agents_per_world}"
        )

    templates: list[Path] = [Path()] * agents_per_world
    for row in latest_rows:
        agent_index = int(row["agent_index"])
        if agent_index < 0 or agent_index >= agents_per_world:
            raise SystemExit(f"invalid agent index in resume metadata: {agent_index}")
        workspace = Path(row["workspace_out"])
        if not workspace.is_dir():
            raise SystemExit(f"recorded workspace does not exist: {workspace}")
        templates[agent_index] = workspace

    sample = max(latest_rows, key=lambda row: int(row["agent_index"]))
    run_prefix = re.sub(r"-g[0-9]{3}$", "", sample["run_id"])
    tmux_prefix = re.sub(r"-g[0-9]{3}$", "", sample["tmux_session"])
    return latest_generation + 1, templates, run_prefix, tmux_prefix


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


def discard_failed_run_dir(run_dir: Path) -> None:
    """Remove a failed run dir, unless it holds checkpoints worth keeping."""
    if not run_dir.exists():
        return
    if any((run_dir / "checkpoints").glob("*.ckpt")):
        preserved = run_dir.with_name(f"{run_dir.name}.failed-{utc_stamp(True)}")
        run_dir.rename(preserved)
        print(f"  preserved failed run dir (has checkpoints): {preserved}", flush=True)
        return
    remove_tree(run_dir)


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


def prune_reconstructable_agent_artifacts(agent_dir: Path) -> list[Path]:
    pruned: list[Path] = []
    for path in (
        agent_dir / "sandbox-home" / ".local" / "share" / "claude" / "versions",
        agent_dir / "runtime" / "python",
        agent_dir / "workspace" / ".venv",
    ):
        if path.exists():
            remove_tree(path)
            pruned.append(path)
    return pruned


def run_generation(
    args: argparse.Namespace,
    *,
    generation: int,
    templates: list[Path],
    run_id: str,
    tmux_session: str,
    resume_ckpt: Path | None = None,
) -> list[Path]:
    world_dir = root_path(args.world_dir)
    run_dir = args.runs_dir / run_id
    if run_dir.exists():
        raise RuntimeError(f"run directory already exists: {run_dir}")

    world_runtime = run_dir / "runtime"
    recordings = run_dir / "recordings"
    logs = run_dir / "logs"
    agents_root = run_dir / "agents"
    for path in (world_runtime, recordings, logs, agents_root):
        path.mkdir(parents=True, exist_ok=True)

    world = World(dir=world_dir)
    copy_world_source = bool(world.config.get("agent", {}).get("source_workspace", True))
    agents: list[Agent] = []
    started_agents: list[object] = []
    agent_dirs: list[Path] = []
    workspaces: list[Path] = []
    auth_env = agent_auth_env()
    started_at = utc_stamp()
    completed = False

    resumed = None
    elapsed_offset = 0
    resume_agent_names: list[str] = []
    if resume_ckpt is not None:
        resumed = load_checkpoint(resume_ckpt, run_dir)
        meta = resumed["metadata"]
        elapsed_offset = int(meta.get("elapsed_seconds", 0))
        resume_agent_names = sorted(
            resumed["agents"],
            key=lambda name: int(re.search(r"-a([0-9]+)$", name).group(1))
            if re.search(r"-a([0-9]+)$", name)
            else 0,
        )
        if len(resume_agent_names) != args.agents_per_world:
            raise RuntimeError(
                f"checkpoint has {len(resume_agent_names)} agents, "
                f"expected {args.agents_per_world}"
            )
        print(f"  resuming from: {resume_ckpt} (t={elapsed_offset}s)", flush=True)

    try:
        started_world = world.start(
            port=args.base_port,
            record=args.record,
            record_dir=recordings,
            resume=resumed["world_snapshot"] if resumed else None,
            health_timeout=HEALTH_TIMEOUT_SECONDS,
            port_wait=PORT_WAIT_SECONDS,
            command_file=world_runtime / "world.sh",
            log_file=logs / "world.log",
            _tmux_session=tmux_session,
        )

        for idx in range(args.agents_per_world):
            if resumed is not None:
                agent_name = resume_agent_names[idx]
                agent_dir = Path(resumed["agents"][agent_name])
                workspace = agent_dir / "workspace"
            else:
                display_name = AGENT_DISPLAY_NAMES[idx] if idx < len(AGENT_DISPLAY_NAMES) else f"Agent{idx}"
                agent_name = safe_name(f"{display_name}-r{run_id}-w0-a{idx}")
                agent_dir = agents_root / agent_name
                workspace = agent_dir / "workspace"
                copy_workspace(templates[idx], workspace, world_dir, copy_world=copy_world_source)
            agent = Agent(
                agent=args.backend,
                name=agent_name,
                dir=agent_dir,
                model=args.model or DEFAULT_MODELS[args.backend],
                permission_mode=args.permission_mode,
                tmux=tmux_session,
                sandbox=args.sandbox,
                claude_native_sandbox=args.claude_native_sandbox,
                bare=args.bare,
            )
            agents.append(agent)
            agent_dirs.append(agent_dir)
            workspaces.append(workspace)

            if resumed is not None:
                # world.connect rejoins the recorded world session id; the agent's
                # original system prompt (restored with the session tier) stays
                # valid. The send prompt is the standard auto-continue nudge, so
                # the restored agent observes nothing unusual.
                world.connect(agent=agent)
                started_agents.append(
                    agent.send(prompt=AUTO_CONTINUE_PROMPT, env=auth_env or None)
                )
                continue

            access = world.connect(agent=agent)
            session = access["session"]
            base_url = access.get("base_url", started_world.url)
            system_prompt = render_system_prompt(
                root_path(args.system_prompt) if args.system_prompt else world_dir / "system_prompt.md",
                {
                    "WORLD_AGENT_NAME": display_name,
                    "WORKSPACE_DIR": agent.visible_workspace_dir,
                    "SESSION_LINE": f"X-Session: {session}",
                    "WORLD_BASE_URL": base_url,
                    "SKILL_CURL": f"curl -H 'X-Session: {session}'",
                    "SKILL_URL": f"{base_url}/api.md",
                },
            )
            started_agents.append(
                agent.start(
                    initial_prompt=args.goal or "Begin",
                    system_prompt=system_prompt,
                    env=auth_env or None,
                )
            )

        remaining_duration = max(1, args.generation_duration - elapsed_offset)
        warning = min(SAVE_GRACE_SECONDS, remaining_duration - 1) if remaining_duration > 1 else 0
        active = remaining_duration - warning

        checkpointer = None
        if CHECKPOINT_EVERY_SECONDS > 0:
            checkpointer = Checkpointer(
                world=world,
                run_dir=run_dir,
                experiment_dir=args.chain_csv.parent,
                generation=generation,
                agents={agent.name: agent_dir for agent, agent_dir in zip(agents, agent_dirs)},
                agent_logs=[agent_dir / "logs" / "agent.log" for agent_dir in agent_dirs],
                backend=args.backend,
                metadata={
                    "run_id": run_id,
                    "tmux_session": tmux_session,
                    "world_dir": str(args.world_dir),
                    "model": args.model or DEFAULT_MODELS[args.backend],
                    "agents_per_world": args.agents_per_world,
                    "generation_duration": args.generation_duration,
                },
                elapsed_offset=elapsed_offset,
            )

        for path in [Path(started_world.command_file)] + [
            Path(started.command_file) for started in started_agents
        ]:
            path.unlink(missing_ok=True)

        if len(workspaces) == 1:
            print(f"  workspace: {workspaces[0]}", flush=True)
        print(f"  tmux:      tmux attach -t {shlex.quote(tmux_session)}", flush=True)
        print(f"  world:     {started_world.url}", flush=True)
        if len(workspaces) > 1:
            for idx, workspace in enumerate(workspaces):
                print(f"  workspace[a{idx}]: {workspace}", flush=True)

        agent_windows = [
            getattr(started, "tmux_window", None) or f"agent-0-{idx}"
            for idx, started in enumerate(started_agents)
        ]
        if warning > 0:
            wait_while_running(
                active,
                url=started_world.url,
                tmux_session=tmux_session,
                world_log=Path(started_world.log_file),
                agent_logs=[agent_dir / "logs" / "agent.log" for agent_dir in agent_dirs],
                agent_windows=agent_windows,
                checkpointer=checkpointer,
            )
            for agent in agents:
                agent.stop(grace_seconds=STOP_GRACE_SECONDS)
            started_agents.clear()
            for agent in agents:
                started_agents.append(agent.send(prompt=SAVE_PROMPT, env=auth_env or None))
            wait_while_running(
                warning,
                url=started_world.url,
                tmux_session=tmux_session,
                world_log=Path(started_world.log_file),
                agent_logs=[agent_dir / "logs" / "agent.log" for agent_dir in agent_dirs],
                require_agent=False,
            )
        else:
            wait_while_running(
                remaining_duration,
                url=started_world.url,
                tmux_session=tmux_session,
                world_log=Path(started_world.log_file),
                agent_logs=[agent_dir / "logs" / "agent.log" for agent_dir in agent_dirs],
                agent_windows=agent_windows,
                checkpointer=checkpointer,
            )

        ended_at = utc_stamp()
        for idx, agent_dir in enumerate(agent_dirs):
            validate_agent_dir(agent_dir, args.backend)
            template_in = (
                f"checkpoint:{resume_ckpt}" if resumed is not None else str(templates[idx])
            )
            with args.chain_csv.open("a", encoding="utf-8", newline="") as handle:
                csv.writer(handle, quoting=csv.QUOTE_ALL).writerow(
                    [
                        generation,
                        idx,
                        agents[idx].name,
                        run_id,
                        tmux_session,
                        template_in,
                        str(workspaces[idx]),
                        started_at,
                        ended_at,
                        args.generation_duration,
                    ]
                )
        completed = True
        return workspaces
    finally:
        for agent in agents:
            try:
                agent.stop(grace_seconds=STOP_GRACE_SECONDS)
            except Exception:
                pass
        try:
            world.stop(grace_seconds=STOP_GRACE_SECONDS)
        except Exception:
            pass
        if completed and args.prune_reconstructable_artifacts:
            pruned_count = 0
            for agent_dir in agent_dirs:
                pruned_count += len(prune_reconstructable_agent_artifacts(agent_dir))
            if pruned_count:
                print(f"  pruned reconstructable artifacts: {pruned_count}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a Clawblox generation chain.")
    parser.add_argument("--generations", type=int, default=int(os.environ.get("GENERATIONS", "30")))
    parser.add_argument("--generation-duration", type=parse_duration, default=parse_duration(os.environ.get("GENERATION_DURATION", "30m")))
    parser.add_argument("--experiment-id", default=os.environ.get("EXPERIMENT_ID", f"genexp-python-{utc_stamp(True)}"))
    parser.add_argument("--run-prefix", default=os.environ.get("RUN_PREFIX"))
    parser.add_argument("--tmux-prefix", default=os.environ.get("TMUX_PREFIX", "clawblox-gen-python"))
    parser.add_argument("--world-dir", type=Path, default=Path(os.environ.get("WORLD_DIR", "worlds/mujoco-panda")))
    parser.add_argument("--base-port", type=int, default=int(os.environ.get("BASE_PORT", "8085")))
    parser.add_argument("--agents-per-world", type=int, default=int(os.environ.get("AGENTS_PER_WORLD", "1")))
    parser.add_argument("--backend", choices=sorted(DEFAULT_MODELS), default=os.environ.get("BACKEND", "claude"))
    parser.add_argument("--model", default=os.environ.get("MODEL"))
    parser.add_argument("--permission-mode", default=os.environ.get("PERMISSION_MODE", "bypassPermissions"))
    parser.add_argument("--record", action=argparse.BooleanOptionalAction, default=os.environ.get("RECORD", "true") != "false")
    parser.add_argument("--sandbox", action=argparse.BooleanOptionalAction, default=os.environ.get("SANDBOX", "true") != "false")
    parser.add_argument("--claude-native-sandbox", action=argparse.BooleanOptionalAction, default=os.environ.get("CLAUDE_NATIVE_SANDBOX", "true") != "false")
    parser.add_argument("--bare", action="store_true", default=os.environ.get("BARE", "false") == "true")
    parser.add_argument("--goal", default=os.environ.get("GOAL", ""))
    parser.add_argument("--template", type=Path, default=Path(os.environ.get("TEMPLATE", DEFAULT_TEMPLATE)))
    parser.add_argument("--system-prompt", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--resume-from",
        type=Path,
        help=(
            "Resume a generation mid-flight from a .ckpt checkpoint archive "
            "(see the run's checkpoints/ directory and the experiment's "
            "checkpoint.json pointer). Requires the original --experiment-id."
        ),
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--prune-reconstructable-artifacts",
        action=argparse.BooleanOptionalAction,
        default=os.environ.get("PRUNE_RECONSTRUCTABLE_ARTIFACTS", "true") != "false",
        help=(
            "After successful generations, delete replay-unnecessary artifacts "
            "that can be reconstructed: Claude version caches, embedded Python "
            "runtimes, and per-workspace virtualenvs."
        ),
    )
    args = parser.parse_args()
    if args.generations <= 0:
        raise SystemExit("--generations must be positive")
    if args.base_port <= 0 or args.base_port > 65535:
        raise SystemExit("--base-port must be in [1, 65535]")
    if args.agents_per_world <= 0:
        raise SystemExit("--agents-per-world must be positive")
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
    initial_template = root_path(args.template)
    templates = [initial_template for _ in range(args.agents_per_world)]

    resume_ckpt: Path | None = None
    if args.resume_from:
        resume_ckpt = root_path(args.resume_from)
        if not resume_ckpt.is_file():
            raise SystemExit(f"checkpoint not found: {resume_ckpt}")
        ckpt_meta = read_metadata(resume_ckpt)
        start_generation = int(ckpt_meta.get("generation", 1))
        ckpt_agents = int(ckpt_meta.get("agents_per_world", len(ckpt_meta.get("agents", {})) or 1))
        if ckpt_agents != args.agents_per_world:
            raise SystemExit(
                f"checkpoint has {ckpt_agents} agents per world, "
                f"but --agents-per-world is {args.agents_per_world}"
            )
        args.runs_dir.mkdir(parents=True, exist_ok=True)
        if not args.chain_csv.is_file():
            with args.chain_csv.open("w", encoding="utf-8", newline="") as handle:
                csv.writer(handle).writerow(CHAIN_HEADER)
    elif args.resume:
        if not args.chain_csv.is_file():
            raise SystemExit(f"--resume requested but metadata CSV is missing: {args.chain_csv}")
        start_generation, resumed_templates, inferred_run, inferred_tmux = load_resume(
            args.chain_csv, args.agents_per_world
        )
        templates = resumed_templates
        run_prefix = args.run_prefix or inferred_run or run_prefix
        tmux_prefix = inferred_tmux or tmux_prefix
    else:
        if experiment_dir.exists():
            if not args.force:
                raise SystemExit(f"experiment directory already exists: {experiment_dir} (use --resume, --resume-from, or --force)")
            remove_tree(experiment_dir)
        args.runs_dir.mkdir(parents=True)
        with args.chain_csv.open("w", encoding="utf-8", newline="") as handle:
            csv.writer(handle).writerow(CHAIN_HEADER)

    if resume_ckpt is None:
        for template in templates:
            if not template.is_dir():
                raise SystemExit(f"agent template directory not found: {template}")

    print("Starting Python generation chain", flush=True)
    print(f"  Generations: {args.generations}", flush=True)
    if args.agents_per_world > 1:
        print(f"  Agents per generation: {args.agents_per_world}", flush=True)
    print(f"  Duration each: {args.generation_duration}s", flush=True)
    print(f"  Experiment metadata: {experiment_dir}", flush=True)
    print("", flush=True)

    for generation in range(start_generation, args.generations + 1):
        is_resumed = resume_ckpt is not None and generation == start_generation
        suffix = f"-r{utc_stamp(True)}" if is_resumed else ""
        run_id = safe_name(f"{run_prefix}-g{generation:03d}{suffix}")
        tmux_session = safe_name(f"{tmux_prefix}-g{generation:03d}{suffix}")
        print(f"=== Generation {generation}/{args.generations} ===", flush=True)
        print(f"  run_id: {run_id}", flush=True)
        if is_resumed:
            print(f"  checkpoint: {resume_ckpt}", flush=True)
        elif len(templates) == 1:
            print(f"  template: {templates[0]}", flush=True)
        else:
            for idx, template in enumerate(templates):
                print(f"  template[a{idx}]: {template}", flush=True)

        for attempt in range(GENERATION_RETRIES + 1):
            try:
                templates = run_generation(
                    args,
                    generation=generation,
                    templates=templates,
                    run_id=run_id,
                    tmux_session=tmux_session,
                    resume_ckpt=resume_ckpt if is_resumed else None,
                )
                break
            except Exception as err:
                try:
                    discard_failed_run_dir(args.runs_dir / run_id)
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
