#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
LAUNCH_SCRIPT="${LAUNCH_SCRIPT:-$ROOT_DIR/agent/launch_multi_claude.sh}"

GENERATIONS="${GENERATIONS:-5}"
GENERATION_DURATION="${GENERATION_DURATION:-2h}"
EXPERIMENT_ID="${EXPERIMENT_ID:-genexp-claude-$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_PREFIX="${RUN_PREFIX:-$EXPERIMENT_ID}"
TMUX_PREFIX="${TMUX_PREFIX:-clawblox-gen-claude}"
RUN_PREFIX_EXPLICIT=0
TMUX_PREFIX_EXPLICIT=0
WORLD_DIR="${WORLD_DIR:-worlds/mesa-world}"
BASE_PORT="${BASE_PORT:-8085}"
AGENTS_PER_WORLD="${AGENTS_PER_WORLD:-1}"
RECORD="${RECORD:-true}"
HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-30}"
STOP_GRACE_SECONDS="${STOP_GRACE_SECONDS:-10}"
PORT_WAIT_SECONDS="${PORT_WAIT_SECONDS:-120}"
SAVE_GRACE_SECONDS="${SAVE_GRACE_SECONDS:-300}"
MAX_GENERATION_RETRIES="${MAX_GENERATION_RETRIES:-3}"
RETRY_DELAY_SECONDS="${RETRY_DELAY_SECONDS:-15}"
RESUME_MODE=0
RESUME_PRUNE_PARTIAL=0
SAVE_PROMPT="${SAVE_PROMPT:-You will be reset in 5 minutes. Update your workspace memory files now.}"
AGENT_NAME_PREFIX="${AGENT_NAME_PREFIX:-agent}"
GOAL="${GOAL:-}"
INITIAL_TEMPLATE="${INITIAL_TEMPLATE:-}"
SYSTEM_PROMPT_TEMPLATE="${SYSTEM_PROMPT_TEMPLATE:-}"
WORLD_SERVER_CMD="${WORLD_SERVER_CMD:-uv run --with mujoco --with fastapi --with uvicorn python server.py}"
CLAUDE_MODEL="${CLAUDE_MODEL:-claude-opus-4-6}"
CLAUDE_PERMISSION_MODE="${CLAUDE_PERMISSION_MODE:-bypassPermissions}"
CLAUDE_BARE="${CLAUDE_BARE:-0}"
CLAUDE_EXTRA_ARGS="${CLAUDE_EXTRA_ARGS:-}"
CLAUDE_USE_ENV_AUTH="${CLAUDE_USE_ENV_AUTH:-0}"
SANDBOX="${SANDBOX:-0}"
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-1800}"
KEEP_WORLD="${KEEP_WORLD:-0}"
declare -a CURRENT_TEMPLATES=()

sanitize_name() {
  printf '%s' "$1" | tr -c 'A-Za-z0-9._-' '_'
}

is_positive_int() {
  [[ "$1" =~ ^[1-9][0-9]*$ ]]
}

is_nonnegative_int() {
  [[ "$1" =~ ^[0-9]+$ ]]
}

require_value() {
  local opt="$1"
  local maybe_value="${2:-}"
  if [[ -z "$maybe_value" ]]; then
    echo "Error: $opt requires a value."
    usage
    exit 1
  fi
}

parse_duration_seconds() {
  local value="$1"
  if [[ ! "$value" =~ ^([0-9]+)([smh]?)$ ]]; then
    echo "Error: invalid duration '$value'. Use 3600, 60m, or 1h." >&2
    exit 1
  fi

  local amount="${BASH_REMATCH[1]}"
  local unit="${BASH_REMATCH[2]}"
  case "$unit" in
    ""|"s")
      printf '%s\n' "$amount"
      ;;
    "m")
      printf '%s\n' $((amount * 60))
      ;;
    "h")
      printf '%s\n' $((amount * 3600))
      ;;
    *)
      echo "Error: unsupported duration unit in '$value'." >&2
      exit 1
      ;;
  esac
}

to_world_abs_dir() {
  local value="$1"
  if [[ "$value" = /* ]]; then
    printf '%s\n' "$value"
  else
    printf '%s\n' "$ROOT_DIR/$value"
  fi
}

to_root_abs_path() {
  local value="$1"
  if [[ "$value" = /* ]]; then
    printf '%s\n' "$value"
  else
    printf '%s\n' "$ROOT_DIR/$value"
  fi
}

csv_escape() {
  printf '%s' "$1" | sed 's/"/""/g'
}

csv_unquote() {
  local value="$1"
  value="${value#\"}"
  value="${value%\"}"
  printf '%s' "${value//\"\"/\"}"
}

utc_now() {
  date -u +%Y-%m-%dT%H:%M:%SZ
}

sleep_until_deadline() {
  local deadline="$1"
  while ((SECONDS < deadline)); do
    local remaining=$((deadline - SECONDS))
    if ((remaining <= 0)); then
      break
    fi
    local sleep_for=5
    if ((remaining < sleep_for)); then
      sleep_for="$remaining"
    fi
    sleep "$sleep_for"
  done
}

set_all_current_templates() {
  local template_path="$1"
  CURRENT_TEMPLATES=()
  for ((idx = 0; idx < AGENTS_PER_WORLD; idx++)); do
    CURRENT_TEMPLATES[$idx]="$template_path"
  done
}

archive_existing_directory() {
  local dir_path="$1"
  local ts archive_dir

  [[ -e "$dir_path" ]] || return 0
  ts="$(date -u +%Y%m%dT%H%M%SZ)"
  archive_dir="${dir_path}.resume-prev-${ts}"
  mv "$dir_path" "$archive_dir"
  echo "Archived existing directory to: $archive_dir"
}

validate_agent_generation_output() {
  local agent_dir="$1"
  local workspace_dir="$agent_dir/workspace"
  local log_file="$agent_dir/logs/agent.log"
  local prompt_file="$agent_dir/runtime/system_prompt.md"
  local claude_session_file="$agent_dir/claude_session_id.txt"
  local world_session_file="$agent_dir/world_session.txt"

  if [[ ! -s "$log_file" ]]; then
    echo "Error: agent log missing or empty: $log_file"
    return 1
  fi
  if [[ ! -s "$prompt_file" ]]; then
    echo "Error: agent system prompt missing or empty: $prompt_file"
    return 1
  fi
  if [[ ! -s "$claude_session_file" ]]; then
    echo "Error: Claude session id missing or empty: $claude_session_file"
    return 1
  fi
  if [[ ! -s "$world_session_file" ]]; then
    echo "Error: world session id missing or empty: $world_session_file"
    return 1
  fi
  if [[ ! -d "$workspace_dir" ]]; then
    echo "Error: agent workspace missing: $workspace_dir"
    return 1
  fi
  if ! find "$workspace_dir" -mindepth 1 -maxdepth 1 | read -r _; then
    echo "Error: agent workspace is empty: $workspace_dir"
    return 1
  fi
}

wait_for_tmux_session_exit() {
  local tmux_session="$1"
  local timeout_seconds="$2"
  local deadline=$((SECONDS + timeout_seconds))

  while tmux has-session -t "$tmux_session" 2>/dev/null; do
    if ((SECONDS >= deadline)); then
      return 1
    fi
    sleep 5
  done
  return 0
}

wait_for_reset_cycle() {
  local log_file="$1"
  local cycle_index="$2"
  local timeout_seconds="$3"
  local start_line="${4:-1}"
  local deadline=$((SECONDS + timeout_seconds))
  local marker="cycle ${cycle_index})"

  while ((SECONDS < deadline)); do
    if [[ -f "$log_file" ]] && sed -n "${start_line},\$p" "$log_file" | grep -F "$marker" >/dev/null 2>&1; then
      return 0
    fi
    sleep 2
  done
  return 1
}

copy_agent_generation_snapshot() {
  local src_agent_dir="$1"
  local dest_agent_dir="$2"

  mkdir -p "$dest_agent_dir/logs" "$dest_agent_dir/runtime"
  cp -a "$src_agent_dir/workspace" "$dest_agent_dir/"
  cp -a "$src_agent_dir/logs/." "$dest_agent_dir/logs/"
}

scrub_resumed_agent_runtime_metadata() {
  local live_run_dir="$1"
  local agent_dir

  while IFS= read -r -d '' agent_dir; do
    rm -f "$agent_dir/claude_session_id.txt"
    rm -rf "$agent_dir/runtime" "$agent_dir/session"
    mkdir -p "$agent_dir/runtime"
  done < <(find "$live_run_dir/worlds" -type d -path '*/agents/*' ! -path '*/agents/*/*' -print0)
}

write_generation_run_metadata() {
  local target="$1"
  local resume_run_id="$2"
  local resume_run_dir="$3"
  local tmux_session="$4"
  local reset_every_seconds="$5"

  mkdir -p "$(dirname "$target")"
  cat >"$target" <<EOF
RUN_ID=$(printf '%q' "$resume_run_id")
RUN_SAFE_ID=$(printf '%q' "$(sanitize_name "$resume_run_id")")
RUN_DIR=$(printf '%q' "$resume_run_dir")
WORLDS_ROOT=$(printf '%q' "$resume_run_dir/worlds")
WORLD_ABS_DIR=$(printf '%q' "$WORLD_ABS_DIR")
NUM_WORLDS=$(printf '%q' "1")
AGENTS_PER_WORLD=$(printf '%q' "$AGENTS_PER_WORLD")
BASE_PORT=$(printf '%q' "$BASE_PORT")
RECORD=$(printf '%q' "$RECORD")
RESET_EVERY=$(printf '%q' "$reset_every_seconds")
WORLD_SERVER_CMD=$(printf '%q' "$WORLD_SERVER_CMD")
TMUX_SESSION=$(printf '%q' "$tmux_session")
EOF
}

save_generation_world_snapshot() {
  local src_world_root="$1"
  local dest_run_dir="$2"
  local world_index="$3"
  local port="$4"
  local dest_world_root="$dest_run_dir/worlds/world-$world_index"
  local resume_dir="$dest_world_root/resume"
  local snapshot_file="$resume_dir/latest.json"
  local tmp_file="$snapshot_file.tmp"
  local token_file="$src_world_root/spectator_token.txt"
  local spectator_token=""
  local ts
  local -a curl_args=()

  mkdir -p "$resume_dir"
  if [[ -f "$token_file" ]]; then
    spectator_token="$(tr -d '[:space:]' <"$token_file")"
    if [[ -n "$spectator_token" ]]; then
      curl_args=(-H "X-Spectator-Token: $spectator_token")
    fi
  fi

  if ! curl -fsS "${curl_args[@]}" "http://localhost:${port}/snapshot" >"$tmp_file"; then
    rm -f "$tmp_file"
    echo "Error: failed to save generation snapshot for world-$world_index on port $port" >&2
    return 1
  fi

  ts="$(date -u +%Y%m%dT%H%M%SZ)"
  cp "$tmp_file" "$resume_dir/checkpoint_${ts}.json"
  mv "$tmp_file" "$snapshot_file"
  if [[ -f "$token_file" ]]; then
    cp -a "$token_file" "$dest_world_root/"
  fi
}

copy_generation_world_snapshot() {
  local src_world_root="$1"
  local dest_run_dir="$2"
  local world_index="$3"
  local dest_world_root="$dest_run_dir/worlds/world-$world_index"
  local src_resume_dir="$src_world_root/resume"
  local dest_resume_dir="$dest_world_root/resume"
  local token_file="$src_world_root/spectator_token.txt"

  if [[ ! -f "$src_resume_dir/latest.json" ]]; then
    echo "Error: missing saved world snapshot at $src_resume_dir/latest.json" >&2
    return 1
  fi

  mkdir -p "$dest_resume_dir"
  cp -a "$src_resume_dir/." "$dest_resume_dir/"
  if [[ -f "$token_file" ]]; then
    cp -a "$token_file" "$dest_world_root/"
  fi
}

prepare_persistent_world_resume_run() {
  local checkpoint_run_dir="$1"
  local live_run_dir="$2"
  local live_tmux_session="$3"
  local checkpoint_run_id
  local checkpoint_reset_every
  local checkpoint_world_abs_dir
  local checkpoint_record
  local checkpoint_clawblox_bin

  if [[ ! -d "$checkpoint_run_dir" ]]; then
    echo "Error: persistent resume checkpoint directory not found: $checkpoint_run_dir"
    return 1
  fi
  if [[ ! -f "$checkpoint_run_dir/run.env" ]]; then
    echo "Error: persistent resume checkpoint missing run.env: $checkpoint_run_dir/run.env"
    return 1
  fi
  if [[ ! -f "$checkpoint_run_dir/worlds/world-0/resume/latest.json" ]]; then
    echo "Error: persistent resume checkpoint missing world snapshot: $checkpoint_run_dir/worlds/world-0/resume/latest.json"
    return 1
  fi

  checkpoint_run_id="$(
    # shellcheck disable=SC1090
    (
      source "$checkpoint_run_dir/run.env"
      printf '%s' "$RUN_ID"
    )
  )"
  checkpoint_reset_every="$(
    # shellcheck disable=SC1090
    (
      source "$checkpoint_run_dir/run.env"
      printf '%s' "${RESET_EVERY:-}"
    )
  )"
  checkpoint_world_abs_dir="$(
    # shellcheck disable=SC1090
    (
      source "$checkpoint_run_dir/run.env"
      printf '%s' "${WORLD_ABS_DIR:-}"
    )
  )"
  checkpoint_record="$(
    # shellcheck disable=SC1090
    (
      source "$checkpoint_run_dir/run.env"
      printf '%s' "${RECORD:-}"
    )
  )"
  checkpoint_world_server_cmd="$(
    # shellcheck disable=SC1090
    (
      source "$checkpoint_run_dir/run.env"
      printf '%s' "${WORLD_SERVER_CMD:-}"
    )
  )"

  if [[ -z "$checkpoint_run_id" ]]; then
    echo "Error: persistent resume checkpoint run.env is missing RUN_ID."
    return 1
  fi
  if [[ -n "$checkpoint_world_abs_dir" && "$checkpoint_world_abs_dir" != "$WORLD_ABS_DIR" ]]; then
    echo "Error: checkpoint world directory mismatch: $checkpoint_world_abs_dir != $WORLD_ABS_DIR"
    return 1
  fi
  if [[ -z "$WORLD_SERVER_CMD" && -n "$checkpoint_world_server_cmd" ]]; then
    WORLD_SERVER_CMD="$checkpoint_world_server_cmd"
  fi
  if [[ -n "$checkpoint_record" ]]; then
    RECORD="$checkpoint_record"
  fi

  archive_existing_directory "$live_run_dir"
  mkdir -p "$live_run_dir"
  cp -a "$checkpoint_run_dir/." "$live_run_dir/"
  scrub_resumed_agent_runtime_metadata "$live_run_dir"
  write_generation_run_metadata \
    "$live_run_dir/run.env" \
    "$checkpoint_run_id" \
    "$live_run_dir" \
    "$live_tmux_session" \
    "$checkpoint_reset_every"
}

usage() {
  cat <<EOF
Usage: $(basename "$0") [options]

Run a Claude Code generation chain where each agent keeps its own workspace lineage
across sessions.

Options:
  --generations N               Target generation count (default: 5)
  --generation-duration D       Per-generation runtime (e.g. 3600, 60m, 1h; default: 2h)
  --generation-retries N        Automatic retries per failed generation (default: 3)
  --retry-delay-seconds N       Wait between generation retries (default: 15)
  --experiment-id ID            Metadata directory under results/ (default: timestamped)
  --run-prefix PREFIX           Prefix for per-generation run ids (default: experiment id)
  --tmux-prefix PREFIX          Prefix for per-generation tmux sessions (default: clawblox-gen-claude)
  --world-dir PATH              World directory (relative to project root or absolute)
  --base-port PORT              World server port for each generation (default: 8085)
  --agents-per-world N          Agents per generation/session (default: 1)
  --record true|false           Recording toggle forwarded to launch_multi_claude.sh
  --health-timeout SECONDS      Health timeout forwarded to launch_multi_claude.sh
  --stop-grace-seconds N        Graceful stop wait forwarded to launch_multi_claude.sh
  --port-wait-seconds N         Port wait timeout forwarded to launch_multi_claude.sh (default: 120)
  --save-grace-seconds N        Final save window before stop (default: 300)
  --save-prompt TEXT            Warning prompt sent during Claude save window
  --agent-name-prefix PREFIX    Agent name prefix forwarded to launch_multi_claude.sh
  --goal TEXT                   Objective forwarded to launch_multi_claude.sh
  --model MODEL_ID              Claude model id forwarded to launch_multi_claude.sh
  --permission-mode MODE        Claude permission mode forwarded to launch_multi_claude.sh
  --claude-extra-args TEXT      Extra Claude CLI args forwarded to launch_multi_claude.sh
  --use-env-auth                Forward auth from env to launch_multi_claude.sh
  --sandbox                     Enable sandbox mode in launch_multi_claude.sh
  --checkpoint-interval N       Forwarded to launch_multi_claude.sh
  --keep-world                  Keep one world/tmux session alive and reset only Claude between generations
  --template PATH               Initial template directory for generation 1
  --system-prompt PATH          System prompt forwarded to launch_multi_claude.sh
  --resume                      Resume an existing experiment id from generation_chain.csv
  --resume-prune-partial        With --resume, delete existing run dir for next generation before relaunch
  --world-server-cmd CMD        Simulator command forwarded to launch_multi_claude.sh
  --help                        Show this message

Environment overrides:
  GENERATIONS, GENERATION_DURATION, EXPERIMENT_ID, RUN_PREFIX, TMUX_PREFIX,
  WORLD_DIR, BASE_PORT, AGENTS_PER_WORLD, RECORD, HEALTH_TIMEOUT, STOP_GRACE_SECONDS,
  PORT_WAIT_SECONDS, SAVE_GRACE_SECONDS, MAX_GENERATION_RETRIES, RETRY_DELAY_SECONDS,
  SAVE_PROMPT, AGENT_NAME_PREFIX, GOAL, INITIAL_TEMPLATE, SYSTEM_PROMPT_TEMPLATE,
  WORLD_SERVER_CMD, CLAUDE_MODEL, CLAUDE_PERMISSION_MODE, CLAUDE_BARE, KEEP_WORLD,
  CLAUDE_EXTRA_ARGS, CLAUDE_USE_ENV_AUTH, CLAUDE_CODE_OAUTH_TOKEN,
  CLAWBLOX_ENV_FILE, SANDBOX, CHECKPOINT_INTERVAL
EOF
}

if [[ ! -f "$LAUNCH_SCRIPT" ]]; then
  echo "Error: launch script not found at $LAUNCH_SCRIPT"
  exit 1
fi

while [[ $# -gt 0 ]]; do
  case "$1" in
    --generations)
      require_value "$1" "${2:-}"
      GENERATIONS="$2"
      shift 2
      ;;
    --generation-duration)
      require_value "$1" "${2:-}"
      GENERATION_DURATION="$2"
      shift 2
      ;;
    --generation-retries)
      require_value "$1" "${2:-}"
      MAX_GENERATION_RETRIES="$2"
      shift 2
      ;;
    --retry-delay-seconds)
      require_value "$1" "${2:-}"
      RETRY_DELAY_SECONDS="$2"
      shift 2
      ;;
    --experiment-id)
      require_value "$1" "${2:-}"
      EXPERIMENT_ID="$2"
      shift 2
      ;;
    --run-prefix)
      require_value "$1" "${2:-}"
      RUN_PREFIX="$2"
      RUN_PREFIX_EXPLICIT=1
      shift 2
      ;;
    --tmux-prefix)
      require_value "$1" "${2:-}"
      TMUX_PREFIX="$2"
      TMUX_PREFIX_EXPLICIT=1
      shift 2
      ;;
    --world-dir)
      require_value "$1" "${2:-}"
      WORLD_DIR="$2"
      shift 2
      ;;
    --base-port)
      require_value "$1" "${2:-}"
      BASE_PORT="$2"
      shift 2
      ;;
    --agents-per-world)
      require_value "$1" "${2:-}"
      AGENTS_PER_WORLD="$2"
      shift 2
      ;;
    --record)
      require_value "$1" "${2:-}"
      RECORD="$2"
      shift 2
      ;;
    --health-timeout)
      require_value "$1" "${2:-}"
      HEALTH_TIMEOUT="$2"
      shift 2
      ;;
    --stop-grace-seconds)
      require_value "$1" "${2:-}"
      STOP_GRACE_SECONDS="$2"
      shift 2
      ;;
    --port-wait-seconds)
      require_value "$1" "${2:-}"
      PORT_WAIT_SECONDS="$2"
      shift 2
      ;;
    --save-grace-seconds)
      require_value "$1" "${2:-}"
      SAVE_GRACE_SECONDS="$2"
      shift 2
      ;;
    --save-prompt)
      require_value "$1" "${2:-}"
      SAVE_PROMPT="$2"
      shift 2
      ;;
    --agent-name-prefix)
      require_value "$1" "${2:-}"
      AGENT_NAME_PREFIX="$2"
      shift 2
      ;;
    --goal)
      require_value "$1" "${2:-}"
      GOAL="$2"
      shift 2
      ;;
    --model)
      require_value "$1" "${2:-}"
      CLAUDE_MODEL="$2"
      shift 2
      ;;
    --permission-mode)
      require_value "$1" "${2:-}"
      CLAUDE_PERMISSION_MODE="$2"
      shift 2
      ;;
    --claude-extra-args)
      require_value "$1" "${2:-}"
      CLAUDE_EXTRA_ARGS="$2"
      shift 2
      ;;
    --use-env-auth)
      CLAUDE_USE_ENV_AUTH=1
      shift
      ;;
    --sandbox)
      SANDBOX=1
      shift
      ;;
    --checkpoint-interval)
      require_value "$1" "${2:-}"
      CHECKPOINT_INTERVAL="$2"
      shift 2
      ;;
    --keep-world|--persistent-world)
      KEEP_WORLD=1
      shift
      ;;
    --template|--initial-template)
      require_value "$1" "${2:-}"
      INITIAL_TEMPLATE="$2"
      shift 2
      ;;
    --system-prompt)
      require_value "$1" "${2:-}"
      SYSTEM_PROMPT_TEMPLATE="$2"
      shift 2
      ;;
    --resume)
      RESUME_MODE=1
      shift
      ;;
    --resume-prune-partial)
      RESUME_MODE=1
      RESUME_PRUNE_PARTIAL=1
      shift
      ;;
    --world-server-cmd)
      require_value "$1" "${2:-}"
      WORLD_SERVER_CMD="$2"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      usage
      exit 1
      ;;
  esac
done

if ! is_positive_int "$GENERATIONS"; then
  echo "Error: --generations must be a positive integer (got '$GENERATIONS')."
  exit 1
fi
if ! is_positive_int "$BASE_PORT" || ((BASE_PORT > 65535)); then
  echo "Error: --base-port must be an integer in [1, 65535] (got '$BASE_PORT')."
  exit 1
fi
if ! is_positive_int "$AGENTS_PER_WORLD"; then
  echo "Error: --agents-per-world must be a positive integer (got '$AGENTS_PER_WORLD')."
  exit 1
fi
if ! is_positive_int "$HEALTH_TIMEOUT"; then
  echo "Error: --health-timeout must be a positive integer (got '$HEALTH_TIMEOUT')."
  exit 1
fi
if ! is_nonnegative_int "$STOP_GRACE_SECONDS"; then
  echo "Error: --stop-grace-seconds must be a non-negative integer (got '$STOP_GRACE_SECONDS')."
  exit 1
fi
if ! is_nonnegative_int "$PORT_WAIT_SECONDS"; then
  echo "Error: --port-wait-seconds must be a non-negative integer (got '$PORT_WAIT_SECONDS')."
  exit 1
fi
if ! is_nonnegative_int "$SAVE_GRACE_SECONDS"; then
  echo "Error: --save-grace-seconds must be a non-negative integer (got '$SAVE_GRACE_SECONDS')."
  exit 1
fi
if ! is_nonnegative_int "$MAX_GENERATION_RETRIES"; then
  echo "Error: --generation-retries must be a non-negative integer (got '$MAX_GENERATION_RETRIES')."
  exit 1
fi
if ! is_nonnegative_int "$RETRY_DELAY_SECONDS"; then
  echo "Error: --retry-delay-seconds must be a non-negative integer (got '$RETRY_DELAY_SECONDS')."
  exit 1
fi
if [[ "$RECORD" != "true" && "$RECORD" != "false" ]]; then
  echo "Error: --record must be 'true' or 'false' (got '$RECORD')."
  exit 1
fi
if [[ "$CLAUDE_BARE" != "0" && "$CLAUDE_BARE" != "1" ]]; then
  echo "Error: CLAUDE_BARE must be 0 or 1 (got '$CLAUDE_BARE')."
  exit 1
fi
if [[ "$CLAUDE_USE_ENV_AUTH" != "0" && "$CLAUDE_USE_ENV_AUTH" != "1" ]]; then
  echo "Error: CLAUDE_USE_ENV_AUTH must be 0 or 1 (got '$CLAUDE_USE_ENV_AUTH')."
  exit 1
fi
if [[ "$SANDBOX" != "0" && "$SANDBOX" != "1" ]]; then
  echo "Error: SANDBOX must be 0 or 1 (got '$SANDBOX')."
  exit 1
fi
if [[ "$KEEP_WORLD" != "0" && "$KEEP_WORLD" != "1" ]]; then
  echo "Error: KEEP_WORLD must be 0 or 1 (got '$KEEP_WORLD')."
  exit 1
fi
if ! is_positive_int "$CHECKPOINT_INTERVAL"; then
  echo "Error: --checkpoint-interval must be a positive integer (got '$CHECKPOINT_INTERVAL')."
  exit 1
fi

GENERATION_DURATION_SECONDS="$(parse_duration_seconds "$GENERATION_DURATION")"
if ((GENERATION_DURATION_SECONDS <= 0)); then
  echo "Error: --generation-duration must be greater than zero."
  exit 1
fi
if ((KEEP_WORLD == 1)) && ((SAVE_GRACE_SECONDS <= 0 || GENERATION_DURATION_SECONDS <= 1)); then
  echo "Error: --keep-world requires --save-grace-seconds > 0 and --generation-duration > 1s."
  exit 1
fi

WORLD_ABS_DIR="$(to_world_abs_dir "$WORLD_DIR")"
if [[ ! -d "$WORLD_ABS_DIR" ]]; then
  echo "Error: world directory not found at $WORLD_ABS_DIR"
  exit 1
fi

if [[ -n "$INITIAL_TEMPLATE" ]]; then
  initial_template_abs="$(to_root_abs_path "$INITIAL_TEMPLATE")"
  if [[ ! -d "$initial_template_abs" ]]; then
    echo "Error: initial template directory not found at $initial_template_abs"
    exit 1
  fi
  set_all_current_templates "$initial_template_abs"
fi

EXPERIMENT_SAFE_ID="$(sanitize_name "$EXPERIMENT_ID")"
EXPERIMENT_DIR="$WORLD_ABS_DIR/results/$EXPERIMENT_SAFE_ID"
CHAIN_CSV="$EXPERIMENT_DIR/generation_chain.csv"
EXPERIMENT_RUNS_DIR="$EXPERIMENT_DIR/runs"
START_GENERATION=1
COMPLETED_GENERATIONS=0

if ((RESUME_MODE == 1)); then
  if [[ ! -d "$EXPERIMENT_DIR" ]]; then
    echo "Error: --resume requested but experiment directory does not exist: $EXPERIMENT_DIR"
    exit 1
  fi
  if [[ ! -f "$CHAIN_CSV" ]]; then
    echo "Error: --resume requested but metadata CSV is missing: $CHAIN_CSV"
    exit 1
  fi
  if [[ ! -d "$EXPERIMENT_RUNS_DIR" ]]; then
    echo "Error: --resume requested but runs directory is missing: $EXPERIMENT_RUNS_DIR"
    exit 1
  fi

  csv_header="$(head -n 1 "$CHAIN_CSV" || true)"
  if [[ "$csv_header" != "generation,agent_index,agent_name,run_id,tmux_session,template_in,workspace_out,start_utc,end_utc,duration_seconds" ]]; then
    echo "Error: unrecognized metadata CSV header in $CHAIN_CSV"
    exit 1
  fi

  mapfile -t chain_rows < <(tail -n +2 "$CHAIN_CSV" | sed '/^[[:space:]]*$/d')
  if ((${#chain_rows[@]} > 0)); then
    declare -A latest_workspaces=()
    max_generation=0
    last_run_id=""
    last_tmux_session=""
    inferred_agents_per_world=0

    for row in "${chain_rows[@]}"; do
      IFS=',' read -r gen_raw agent_index_raw _ run_id_raw tmux_raw _ workspace_raw _ _ _ <<<"$row"

      if [[ -z "${workspace_raw:-}" || -z "${tmux_raw:-}" || -z "${run_id_raw:-}" || -z "${agent_index_raw:-}" || -z "${gen_raw:-}" ]]; then
        echo "Error: failed to parse metadata row in $CHAIN_CSV"
        exit 1
      fi

      generation_value="$(csv_unquote "$gen_raw")"
      agent_index_value="$(csv_unquote "$agent_index_raw")"
      if [[ ! "$generation_value" =~ ^[0-9]+$ ]]; then
        echo "Error: invalid generation value in CSV: '$generation_value'"
        exit 1
      fi
      if [[ ! "$agent_index_value" =~ ^[0-9]+$ ]]; then
        echo "Error: invalid agent_index value in CSV: '$agent_index_value'"
        exit 1
      fi

      if ((generation_value > max_generation)); then
        max_generation="$generation_value"
        latest_workspaces=()
        last_run_id="$(csv_unquote "$run_id_raw")"
        last_tmux_session="$(csv_unquote "$tmux_raw")"
        inferred_agents_per_world=0
      fi

      if ((generation_value == max_generation)); then
        workspace_value="$(csv_unquote "$workspace_raw")"
        latest_workspaces[$agent_index_value]="$workspace_value"
        last_run_id="$(csv_unquote "$run_id_raw")"
        last_tmux_session="$(csv_unquote "$tmux_raw")"
        if ((inferred_agents_per_world < agent_index_value + 1)); then
          inferred_agents_per_world=$((agent_index_value + 1))
        fi
      fi
    done

    if ((inferred_agents_per_world != AGENTS_PER_WORLD)); then
      echo "Error: resume metadata expects $inferred_agents_per_world agents per generation, but --agents-per-world is $AGENTS_PER_WORLD."
      exit 1
    fi

    COMPLETED_GENERATIONS="$max_generation"
    START_GENERATION=$((COMPLETED_GENERATIONS + 1))
    CURRENT_TEMPLATES=()
    for ((idx = 0; idx < AGENTS_PER_WORLD; idx++)); do
      workspace_value="${latest_workspaces[$idx]:-}"
      if [[ -z "$workspace_value" ]]; then
        echo "Error: missing workspace lineage for agent index $idx in generation $COMPLETED_GENERATIONS."
        exit 1
      fi
      if [[ ! -d "$workspace_value" ]]; then
        echo "Error: recorded workspace for agent index $idx does not exist: $workspace_value"
        exit 1
      fi
      CURRENT_TEMPLATES[$idx]="$workspace_value"
    done

    if ((RUN_PREFIX_EXPLICIT == 0)); then
      inferred_run_prefix="$(printf '%s' "$last_run_id" | sed -E 's/-g[0-9]{3}$//')"
      if [[ "$inferred_run_prefix" != "$last_run_id" && -n "$inferred_run_prefix" ]]; then
        RUN_PREFIX="$inferred_run_prefix"
      fi
    fi
    if ((TMUX_PREFIX_EXPLICIT == 0)); then
      inferred_tmux_prefix="$(printf '%s' "$last_tmux_session" | sed -E 's/-g[0-9]{3}$//')"
      if [[ "$inferred_tmux_prefix" != "$last_tmux_session" && -n "$inferred_tmux_prefix" ]]; then
        TMUX_PREFIX="$inferred_tmux_prefix"
      fi
    fi

    if [[ -n "$INITIAL_TEMPLATE" ]]; then
      echo "Note: ignoring --template in --resume mode because completed generations already exist."
    fi
  fi
else
  if [[ -e "$EXPERIMENT_DIR" ]]; then
    echo "Error: experiment directory already exists: $EXPERIMENT_DIR"
    echo "Use --experiment-id with a unique value or rerun with --resume."
    exit 1
  fi
  mkdir -p "$EXPERIMENT_DIR"
  mkdir -p "$EXPERIMENT_RUNS_DIR"
  printf 'generation,agent_index,agent_name,run_id,tmux_session,template_in,workspace_out,start_utc,end_utc,duration_seconds\n' >"$CHAIN_CSV"
fi

CURRENT_SESSION=""
audit_generation_event() {
  local event="$1"
  local details="${2:-}"
  local audit_file parent_cmd self_cmd

  if [[ -z "${EXPERIMENT_DIR:-}" ]]; then
    return 0
  fi

  audit_file="$EXPERIMENT_DIR/generation_audit.log"
  mkdir -p "$(dirname "$audit_file")"
  parent_cmd="$(ps -o args= -p "$PPID" 2>/dev/null | tr '\n' ' ' || true)"
  self_cmd="$(ps -o args= -p "$$" 2>/dev/null | tr '\n' ' ' || true)"
  printf '%s event=%s pid=%s ppid=%s current_session=%q experiment_id=%q parent_cmd=%q self_cmd=%q details=%q\n' \
    "$(utc_now)" \
    "$event" \
    "$$" \
    "$PPID" \
    "$CURRENT_SESSION" \
    "$EXPERIMENT_ID" \
    "$parent_cmd" \
    "$self_cmd" \
    "$details" >>"$audit_file"
}

cleanup_active_session() {
  if [[ -z "$CURRENT_SESSION" ]]; then
    return
  fi

  local session="$CURRENT_SESSION"
  local stop_status=0

  echo
  echo "Stopping active session '$session'..."
  audit_generation_event "cleanup_active_session" "stop_grace_seconds=$STOP_GRACE_SECONDS"
  set +e
  CLAWBLOX_STOP_REASON="${CLAWBLOX_STOP_REASON:-generation_cleanup}" \
    CLAWBLOX_STOP_SOURCE="${CLAWBLOX_STOP_SOURCE:-launch_multi_generations_claude_cleanup}" \
    bash "$LAUNCH_SCRIPT" --tmux-session "$session" --stop --force --stop-grace-seconds "$STOP_GRACE_SECONDS"
  stop_status=$?
  set -e

  if ((stop_status != 0)); then
    echo "Warning: stop command for '$session' exited with status $stop_status." >&2
    audit_generation_event "cleanup_active_session_failed" "session=$session status=$stop_status"
    return "$stop_status"
  fi

  if tmux has-session -t "$session" 2>/dev/null; then
    echo "Warning: tmux session '$session' still exists after cleanup." >&2
    audit_generation_event "cleanup_active_session_leftover" "session=$session"
    return 1
  fi

  echo "Cleanup verified: tmux session '$session' is gone."
  audit_generation_event "cleanup_active_session_verified" "session=$session"
  CURRENT_SESSION=""
}

cleanup_on_signal() {
  local signal_name="$1"
  echo
  echo "Signal $signal_name received."
  audit_generation_event "signal_received" "signal=$signal_name"
  CLAWBLOX_STOP_REASON="generation_wrapper_signal_${signal_name}" \
    CLAWBLOX_STOP_SOURCE="launch_multi_generations_claude_signal"
  cleanup_active_session || true
  exit 130
}

cleanup_failed_generation_attempt() {
  local run_dir="$1"
  cleanup_active_session
  if [[ -d "$run_dir" ]]; then
    echo "Removing failed generation run directory: $run_dir"
    rm -rf "$run_dir"
  fi
}

run_generation_once() {
  local gen="$1"
  local run_id="$2"
  local tmux_session="$3"
  local start_utc="$4"
  shift 4
  local template_inputs=("$@")
  local run_dir="$EXPERIMENT_RUNS_DIR/$run_id"
  local idx
  local template_in
  local launch_duration=""
  local launch_warning="$SAVE_GRACE_SECONDS"
  local total_wait_seconds
  local agents_root
  local workspace_out
  local agent_dir
  local agent_name
  local agent_index
  local end_utc
  local launch_cmd=()
  local -a workspaces=()
  local -A next_templates=()
  local -A next_agent_names=()

  if [[ -e "$run_dir" ]]; then
    if ((RESUME_MODE == 1 && RESUME_PRUNE_PARTIAL == 1 && gen == START_GENERATION)); then
      echo "  removing partial run directory: $run_dir"
      rm -rf "$run_dir"
    else
      echo "Error: run directory already exists: $run_dir"
      if ((RESUME_MODE == 1)); then
        echo "Use --resume-prune-partial to remove this partial directory and continue."
      fi
      return 1
    fi
  fi

  launch_cmd=(
    bash "$LAUNCH_SCRIPT"
    --force
    --run-id "$run_id"
    --tmux-session "$tmux_session"
    --num-worlds 1
    --agents-per-world "$AGENTS_PER_WORLD"
    --base-port "$BASE_PORT"
    --record "$RECORD"
    --world-dir "$WORLD_DIR"
    --results-root "$EXPERIMENT_RUNS_DIR"
    --health-timeout "$HEALTH_TIMEOUT"
    --agent-name-prefix "$AGENT_NAME_PREFIX"
    --stop-grace-seconds "$STOP_GRACE_SECONDS"
    --port-wait-seconds "$PORT_WAIT_SECONDS"
    --model "$CLAUDE_MODEL"
    --permission-mode "$CLAUDE_PERMISSION_MODE"
    --checkpoint-interval "$CHECKPOINT_INTERVAL"
  )
  if [[ -n "$GOAL" ]]; then
    launch_cmd+=(--goal "$GOAL")
  fi
  if [[ -n "$WORLD_SERVER_CMD" ]]; then
    launch_cmd+=(--world-server-cmd "$WORLD_SERVER_CMD")
  fi
  if [[ -n "$SYSTEM_PROMPT_TEMPLATE" ]]; then
    launch_cmd+=(--system-prompt "$SYSTEM_PROMPT_TEMPLATE")
  fi
  if [[ -n "$CLAUDE_EXTRA_ARGS" ]]; then
    launch_cmd+=(--claude-extra-args "$CLAUDE_EXTRA_ARGS")
  fi
  if [[ "$CLAUDE_USE_ENV_AUTH" == "1" ]]; then
    launch_cmd+=(--use-env-auth)
  fi
  if [[ "$SANDBOX" == "1" ]]; then
    launch_cmd+=(--sandbox)
  fi
  for ((idx = 0; idx < AGENTS_PER_WORLD; idx++)); do
    template_in="${template_inputs[$idx]:-}"
    if [[ -n "$template_in" ]]; then
      launch_cmd+=(--agent-template "$template_in")
    fi
  done

  if ((SAVE_GRACE_SECONDS > 0 && GENERATION_DURATION_SECONDS > 1)); then
    launch_warning="$SAVE_GRACE_SECONDS"
    if ((launch_warning >= GENERATION_DURATION_SECONDS)); then
      launch_warning=$((GENERATION_DURATION_SECONDS - 1))
    fi
    launch_duration="$((GENERATION_DURATION_SECONDS - launch_warning))"
  fi

  if [[ -n "$launch_duration" ]]; then
    total_wait_seconds=$((launch_duration + launch_warning + STOP_GRACE_SECONDS + 60))
    CHECKPOINT_WARNING_SECONDS="$launch_warning" \
      CHECKPOINT_WARNING_PROMPT="$SAVE_PROMPT" \
      CLAUDE_BARE="$CLAUDE_BARE" \
      "${launch_cmd[@]}" --duration "$launch_duration" || return 1
  else
    total_wait_seconds=$((GENERATION_DURATION_SECONDS + STOP_GRACE_SECONDS + 60))
    CLAUDE_BARE="$CLAUDE_BARE" \
      "${launch_cmd[@]}" || return 1
  fi

  CURRENT_SESSION="$tmux_session"

  if [[ -n "$launch_duration" ]]; then
    if ! wait_for_tmux_session_exit "$tmux_session" "$total_wait_seconds"; then
      echo "Error: timed out waiting for session '$tmux_session' to stop."
      return 1
    fi
  else
    local manual_deadline=$((SECONDS + GENERATION_DURATION_SECONDS))
    sleep_until_deadline "$manual_deadline"
    echo "  stopping session: $tmux_session"
    bash "$LAUNCH_SCRIPT" --tmux-session "$tmux_session" --stop --force --stop-grace-seconds "$STOP_GRACE_SECONDS" || return 1
    if ! wait_for_tmux_session_exit "$tmux_session" $((STOP_GRACE_SECONDS + 60)); then
      echo "Error: timed out waiting for session '$tmux_session' to stop."
      return 1
    fi
  fi
  CURRENT_SESSION=""

  agents_root="$run_dir/worlds/world-0/agents"
  if [[ ! -d "$agents_root" ]]; then
    echo "Error: expected agents directory not found: $agents_root"
    return 1
  fi

  mapfile -t workspaces < <(find "$agents_root" -mindepth 2 -maxdepth 2 -type d -name workspace | sort)
  if ((${#workspaces[@]} == 0)); then
    echo "Error: no workspace directory found under $agents_root"
    return 1
  fi

  for workspace_out in "${workspaces[@]}"; do
    agent_dir="$(dirname "$workspace_out")"
    agent_name="$(basename "$agent_dir")"
    if [[ ! "$agent_name" =~ -a([0-9]+)$ ]]; then
      echo "Error: could not infer agent index from agent directory '$agent_name'"
      return 1
    fi
    agent_index="${BASH_REMATCH[1]}"
    if ((agent_index >= AGENTS_PER_WORLD)); then
      echo "Error: found workspace for unexpected agent index $agent_index under $agents_root"
      return 1
    fi
    if [[ -n "${next_templates[$agent_index]:-}" ]]; then
      echo "Error: duplicate workspace discovered for agent index $agent_index under $agents_root"
      return 1
    fi
    validate_agent_generation_output "$agent_dir" || return 1
    next_templates[$agent_index]="$workspace_out"
    next_agent_names[$agent_index]="$agent_name"
  done

  for ((idx = 0; idx < AGENTS_PER_WORLD; idx++)); do
    if [[ -z "${next_templates[$idx]:-}" ]]; then
      echo "Error: missing workspace for agent index $idx under $agents_root"
      return 1
    fi
  done

  end_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  CURRENT_TEMPLATES=()
  for ((idx = 0; idx < AGENTS_PER_WORLD; idx++)); do
    template_in="${template_inputs[$idx]:-}"
    workspace_out="${next_templates[$idx]}"
    agent_name="${next_agent_names[$idx]}"
    CURRENT_TEMPLATES[$idx]="$workspace_out"

    printf '"%s","%s","%s","%s","%s","%s","%s","%s","%s","%s"\n' \
      "$(csv_escape "$gen")" \
      "$(csv_escape "$idx")" \
      "$(csv_escape "$agent_name")" \
      "$(csv_escape "$run_id")" \
      "$(csv_escape "$tmux_session")" \
      "$(csv_escape "$template_in")" \
      "$(csv_escape "$workspace_out")" \
      "$(csv_escape "$start_utc")" \
      "$(csv_escape "$end_utc")" \
      "$(csv_escape "$GENERATION_DURATION_SECONDS")" >>"$CHAIN_CSV"

    echo "  workspace_out[a${idx}]: $workspace_out"
  done

  return 0
}

run_persistent_world_generations() {
  local live_run_id
  local live_tmux_session
  local live_run_dir
  local checkpoint_run_dir=""
  local agents_root
  local idx
  local gen
  local reset_cycle_index
  local template_in
  local start_utc
  local end_utc
  local run_id
  local launch_warning="$SAVE_GRACE_SECONDS"
  local launch_duration="$GENERATION_DURATION_SECONDS"
  local total_duration_seconds
  local remaining_generations="$GENERATIONS"
  local launch_cmd=()
  local workspace_out
  local agent_dir
  local agent_name
  local agent_index
  local snapshot_agent_dir
  local snapshot_agents_root
  local snapshot_run_dir
  local live_world_root
  local log_file
  local log_start_line
  local final_wait_seconds
  local boundary_wait_seconds
  local -a template_inputs=()
  local -a workspaces=()
  local -A next_templates=()
  local -A next_agent_names=()

  if ((launch_warning >= GENERATION_DURATION_SECONDS)); then
    launch_warning=$((GENERATION_DURATION_SECONDS - 1))
  fi
  launch_duration="$((GENERATION_DURATION_SECONDS - launch_warning))"
  if ((RESUME_MODE == 1)); then
    if ((COMPLETED_GENERATIONS <= 0)); then
      echo "Error: --keep-world --resume requires at least one committed generation checkpoint."
      return 1
    fi
    remaining_generations=$((GENERATIONS - COMPLETED_GENERATIONS))
  fi
  total_duration_seconds=$((remaining_generations * GENERATION_DURATION_SECONDS - launch_warning))
  boundary_wait_seconds=$((GENERATION_DURATION_SECONDS + STOP_GRACE_SECONDS + 120))
  final_wait_seconds=$((GENERATION_DURATION_SECONDS + STOP_GRACE_SECONDS + 180))

  live_run_id="$(sanitize_name "${RUN_PREFIX}-live")"
  live_tmux_session="$(sanitize_name "${TMUX_PREFIX}-live")"
  live_run_dir="$EXPERIMENT_RUNS_DIR/$live_run_id"
  live_world_root="$live_run_dir/worlds/world-0"
  agents_root="$live_run_dir/worlds/world-0/agents"

  template_inputs=("${CURRENT_TEMPLATES[@]}")
  if ((RESUME_MODE == 1)); then
    checkpoint_run_dir="$EXPERIMENT_RUNS_DIR/$(sanitize_name "$(printf '%s-g%03d' "$RUN_PREFIX" "$COMPLETED_GENERATIONS")")"
    if ! prepare_persistent_world_resume_run "$checkpoint_run_dir" "$live_run_dir" "$live_tmux_session"; then
      return 1
    fi
    launch_cmd=(
      bash "$LAUNCH_SCRIPT"
      --force
      --resume "$live_run_dir"
      --tmux-session "$live_tmux_session"
      --num-worlds 1
      --agents-per-world "$AGENTS_PER_WORLD"
      --base-port "$BASE_PORT"
      --record "$RECORD"
      --health-timeout "$HEALTH_TIMEOUT"
      --agent-name-prefix "$AGENT_NAME_PREFIX"
      --stop-grace-seconds "$STOP_GRACE_SECONDS"
      --port-wait-seconds "$PORT_WAIT_SECONDS"
      --model "$CLAUDE_MODEL"
      --permission-mode "$CLAUDE_PERMISSION_MODE"
      --checkpoint-interval "$CHECKPOINT_INTERVAL"
      --reset-every "$launch_duration"
      --duration "$total_duration_seconds"
    )
  else
    launch_cmd=(
      bash "$LAUNCH_SCRIPT"
      --force
      --run-id "$live_run_id"
      --tmux-session "$live_tmux_session"
      --num-worlds 1
      --agents-per-world "$AGENTS_PER_WORLD"
      --base-port "$BASE_PORT"
      --record "$RECORD"
      --world-dir "$WORLD_DIR"
      --results-root "$EXPERIMENT_RUNS_DIR"
      --health-timeout "$HEALTH_TIMEOUT"
      --agent-name-prefix "$AGENT_NAME_PREFIX"
      --stop-grace-seconds "$STOP_GRACE_SECONDS"
      --port-wait-seconds "$PORT_WAIT_SECONDS"
      --model "$CLAUDE_MODEL"
      --permission-mode "$CLAUDE_PERMISSION_MODE"
      --checkpoint-interval "$CHECKPOINT_INTERVAL"
      --reset-every "$launch_duration"
      --duration "$total_duration_seconds"
    )
  fi
  if [[ -n "$GOAL" ]]; then
    launch_cmd+=(--goal "$GOAL")
  fi
  if [[ -n "$WORLD_SERVER_CMD" ]]; then
    launch_cmd+=(--world-server-cmd "$WORLD_SERVER_CMD")
  fi
  if [[ -n "$SYSTEM_PROMPT_TEMPLATE" ]]; then
    launch_cmd+=(--system-prompt "$SYSTEM_PROMPT_TEMPLATE")
  fi
  if [[ -n "$CLAUDE_EXTRA_ARGS" ]]; then
    launch_cmd+=(--claude-extra-args "$CLAUDE_EXTRA_ARGS")
  fi
  if [[ "$CLAUDE_USE_ENV_AUTH" == "1" ]]; then
    launch_cmd+=(--use-env-auth)
  fi
  if [[ "$SANDBOX" == "1" ]]; then
    launch_cmd+=(--sandbox)
  fi
  for ((idx = 0; idx < AGENTS_PER_WORLD; idx++)); do
    template_in="${template_inputs[$idx]:-}"
    if [[ -n "$template_in" ]]; then
      launch_cmd+=(--agent-template "$template_in")
    fi
  done

  echo "Persistent world mode"
  echo "  live_run_id: $live_run_id"
  echo "  live_tmux_session: $live_tmux_session"
  echo "  active Claude window: ${launch_duration}s"
  echo "  save grace: ${launch_warning}s"
  echo
  audit_generation_event "persistent_world_launch" "live_run_id=$live_run_id tmux_session=$live_tmux_session total_duration_seconds=$total_duration_seconds launch_duration=$launch_duration save_grace=$launch_warning"

  start_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  CHECKPOINT_WARNING_SECONDS="$launch_warning" \
    CHECKPOINT_WARNING_PROMPT="$SAVE_PROMPT" \
    CLAUDE_BARE="$CLAUDE_BARE" \
    "${launch_cmd[@]}" || return 1

  CURRENT_SESSION="$live_tmux_session"

  local startup_deadline=$((SECONDS + HEALTH_TIMEOUT + 120))
  while ((SECONDS < startup_deadline)); do
    if [[ -d "$agents_root" ]]; then
      mapfile -t workspaces < <(find "$agents_root" -mindepth 2 -maxdepth 2 -type d -name workspace | sort)
      if ((${#workspaces[@]} == AGENTS_PER_WORLD)); then
        break
      fi
    fi
    sleep 2
  done
  if ((${#workspaces[@]} != AGENTS_PER_WORLD)); then
    echo "Error: expected $AGENTS_PER_WORLD live workspaces under $agents_root, found ${#workspaces[@]}."
    return 1
  fi

  for ((idx = 0; idx < AGENTS_PER_WORLD; idx++)); do
    if [[ -z "${template_inputs[$idx]:-}" ]]; then
      template_inputs[$idx]=""
    fi
  done

  for ((gen = START_GENERATION; gen <= GENERATIONS; gen++)); do
    run_id="$(sanitize_name "$(printf '%s-g%03d' "$RUN_PREFIX" "$gen")")"

    echo "=== Generation $gen/$GENERATIONS ==="
    echo "  run_id: $run_id"
    echo "  tmux_session: $live_tmux_session"
    for ((idx = 0; idx < AGENTS_PER_WORLD; idx++)); do
      template_in="${template_inputs[$idx]:-}"
      if [[ -n "$template_in" ]]; then
        echo "  template_in[a${idx}]: $template_in"
      else
        echo "  template_in[a${idx}]: (default agent template)"
      fi
    done

    if ((gen < GENERATIONS)); then
      reset_cycle_index=$((gen - START_GENERATION + 1))
      echo "  waiting for reset cycle $gen..."
      audit_generation_event "waiting_for_reset_cycle" "generation=$gen reset_cycle_index=$reset_cycle_index timeout_seconds=$boundary_wait_seconds"
      mapfile -t workspaces < <(find "$agents_root" -mindepth 2 -maxdepth 2 -type d -name workspace | sort)
      if ((${#workspaces[@]} != AGENTS_PER_WORLD)); then
        echo "Error: expected $AGENTS_PER_WORLD live workspaces under $agents_root, found ${#workspaces[@]}."
        return 1
      fi
      for workspace_out in "${workspaces[@]}"; do
        agent_dir="$(dirname "$workspace_out")"
        log_file="$agent_dir/logs/agent.log"
        log_start_line=1
        if [[ -f "$log_file" ]]; then
          log_start_line=$(($(wc -l <"$log_file") + 1))
        fi
        if ! wait_for_reset_cycle "$log_file" "$reset_cycle_index" "$boundary_wait_seconds" "$log_start_line"; then
          audit_generation_event "reset_cycle_timeout" "generation=$gen reset_cycle_index=$reset_cycle_index log_file=$log_file timeout_seconds=$boundary_wait_seconds"
          echo "Error: timed out waiting for reset cycle $reset_cycle_index in $log_file"
          return 1
        fi
      done
      audit_generation_event "reset_cycle_observed" "generation=$gen reset_cycle_index=$reset_cycle_index"
    else
      echo "  waiting for final session shutdown..."
      audit_generation_event "waiting_for_final_shutdown" "generation=$gen timeout_seconds=$final_wait_seconds"
      if ! wait_for_tmux_session_exit "$live_tmux_session" "$final_wait_seconds"; then
        audit_generation_event "final_shutdown_timeout" "generation=$gen timeout_seconds=$final_wait_seconds"
        echo "Error: timed out waiting for persistent session '$live_tmux_session' to stop."
        return 1
      fi
      CURRENT_SESSION=""
      audit_generation_event "final_shutdown_observed" "generation=$gen"
    fi

    mapfile -t workspaces < <(find "$agents_root" -mindepth 2 -maxdepth 2 -type d -name workspace | sort)
    if ((${#workspaces[@]} == 0)); then
      echo "Error: no live workspace directory found under $agents_root"
      return 1
    fi

    snapshot_run_dir="$EXPERIMENT_RUNS_DIR/$run_id"
    snapshot_agents_root="$snapshot_run_dir/worlds/world-0/agents"
    mkdir -p "$snapshot_agents_root"
    next_templates=()
    next_agent_names=()

    if ((gen < GENERATIONS)); then
      if ! save_generation_world_snapshot "$live_world_root" "$snapshot_run_dir" 0 "$BASE_PORT"; then
        echo "Error: failed to snapshot live world for generation $gen."
        return 1
      fi
    else
      if ! copy_generation_world_snapshot "$live_world_root" "$snapshot_run_dir" 0; then
        echo "Error: failed to copy saved live world snapshot for generation $gen."
        return 1
      fi
    fi
    write_generation_run_metadata \
      "$snapshot_run_dir/run.env" \
      "$live_run_id" \
      "$snapshot_run_dir" \
      "$live_tmux_session" \
      "$launch_duration"

    for workspace_out in "${workspaces[@]}"; do
      agent_dir="$(dirname "$workspace_out")"
      agent_name="$(basename "$agent_dir")"
      if [[ ! "$agent_name" =~ -a([0-9]+)$ ]]; then
        echo "Error: could not infer agent index from live agent directory '$agent_name'"
        return 1
      fi
      agent_index="${BASH_REMATCH[1]}"
      if ((agent_index >= AGENTS_PER_WORLD)); then
        echo "Error: found live workspace for unexpected agent index $agent_index under $agents_root"
        return 1
      fi
      if [[ -n "${next_templates[$agent_index]:-}" ]]; then
        echo "Error: duplicate live workspace discovered for agent index $agent_index under $agents_root"
        return 1
      fi

      validate_agent_generation_output "$agent_dir" || return 1
      snapshot_agent_dir="$snapshot_agents_root/$agent_name"
      copy_agent_generation_snapshot "$agent_dir" "$snapshot_agent_dir"
      next_templates[$agent_index]="$snapshot_agent_dir/workspace"
      next_agent_names[$agent_index]="$agent_name"
    done

    for ((idx = 0; idx < AGENTS_PER_WORLD; idx++)); do
      if [[ -z "${next_templates[$idx]:-}" ]]; then
        echo "Error: missing workspace for agent index $idx under $agents_root"
        return 1
      fi
    done

    end_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    CURRENT_TEMPLATES=()
    for ((idx = 0; idx < AGENTS_PER_WORLD; idx++)); do
      template_in="${template_inputs[$idx]:-}"
      workspace_out="${next_templates[$idx]}"
      agent_name="${next_agent_names[$idx]}"
      CURRENT_TEMPLATES[$idx]="$workspace_out"

      printf '"%s","%s","%s","%s","%s","%s","%s","%s","%s","%s"\n' \
        "$(csv_escape "$gen")" \
        "$(csv_escape "$idx")" \
        "$(csv_escape "$agent_name")" \
        "$(csv_escape "$run_id")" \
        "$(csv_escape "$live_tmux_session")" \
        "$(csv_escape "$template_in")" \
        "$(csv_escape "$workspace_out")" \
        "$(csv_escape "$start_utc")" \
        "$(csv_escape "$end_utc")" \
        "$(csv_escape "$GENERATION_DURATION_SECONDS")" >>"$CHAIN_CSV"

      echo "  workspace_out[a${idx}]: $workspace_out"
    done

    template_inputs=("${CURRENT_TEMPLATES[@]}")
    start_utc="$end_utc"
    audit_generation_event "generation_snapshot_written" "generation=$gen run_id=$run_id"
    echo
  done

  return 0
}

trap 'cleanup_on_signal INT' INT
trap 'cleanup_on_signal TERM' TERM
trap 'cleanup_on_signal HUP' HUP
trap cleanup_active_session EXIT

echo "Starting Claude generation chain"
echo "  Generations: $GENERATIONS"
echo "  Agents/session: $AGENTS_PER_WORLD"
echo "  Duration each: ${GENERATION_DURATION_SECONDS}s"
echo "  Generation retries: $MAX_GENERATION_RETRIES"
echo "  Retry delay: ${RETRY_DELAY_SECONDS}s"
echo "  Save grace: ${SAVE_GRACE_SECONDS}s"
echo "  Keep world: $([[ "$KEEP_WORLD" -eq 1 ]] && echo "true" || echo "false")"
echo "  Resume: $([[ "$RESUME_MODE" -eq 1 ]] && echo "true" || echo "false")"
if ((RESUME_MODE == 1)); then
  echo "  Completed: $COMPLETED_GENERATIONS"
  echo "  Next generation: $START_GENERATION"
fi
echo "  Experiment metadata: $EXPERIMENT_DIR"
echo
audit_generation_event "wrapper_started" "generations=$GENERATIONS keep_world=$KEEP_WORLD base_port=$BASE_PORT"

if ((START_GENERATION > GENERATIONS)); then
  echo "Nothing to run: completed generations already reach target ($GENERATIONS)."
  echo "Metadata CSV: $CHAIN_CSV"
  exit 0
fi

if ((KEEP_WORLD == 1)); then
  if ! run_persistent_world_generations; then
    cleanup_active_session
    exit 1
  fi
  echo "Claude generation chain complete."
  echo "Metadata CSV: $CHAIN_CSV"
  exit 0
fi

for ((gen = START_GENERATION; gen <= GENERATIONS; gen++)); do
  gen_tag="$(printf 'g%03d' "$gen")"
  run_id="$(sanitize_name "${RUN_PREFIX}-${gen_tag}")"
  tmux_session="$(sanitize_name "${TMUX_PREFIX}-${gen_tag}")"
  template_inputs=("${CURRENT_TEMPLATES[@]}")

  echo "=== Generation $gen/$GENERATIONS ==="
  echo "  run_id: $run_id"
  echo "  tmux_session: $tmux_session"
  for ((idx = 0; idx < AGENTS_PER_WORLD; idx++)); do
    template_in="${template_inputs[$idx]:-}"
    if [[ -n "$template_in" ]]; then
      echo "  template_in[a${idx}]: $template_in"
    else
      echo "  template_in[a${idx}]: (default agent template)"
    fi
  done

  attempt=0
  success=0
  while ((attempt <= MAX_GENERATION_RETRIES)); do
    start_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    if ((attempt > 0)); then
      echo "  retry attempt $attempt/$MAX_GENERATION_RETRIES"
    fi

    if run_generation_once "$gen" "$run_id" "$tmux_session" "$start_utc" "${template_inputs[@]}"; then
      success=1
      break
    fi

    cleanup_failed_generation_attempt "$EXPERIMENT_RUNS_DIR/$run_id"
    if ((attempt == MAX_GENERATION_RETRIES)); then
      break
    fi
    if ((RETRY_DELAY_SECONDS > 0)); then
      echo "  waiting ${RETRY_DELAY_SECONDS}s before retry..."
      sleep "$RETRY_DELAY_SECONDS"
    fi
    attempt=$((attempt + 1))
  done

  if ((success == 0)); then
    echo "Error: generation $gen failed after $((MAX_GENERATION_RETRIES + 1)) attempt(s)."
    exit 1
  fi
  echo
done

echo "Claude generation chain complete."
echo "Metadata CSV: $CHAIN_CSV"
