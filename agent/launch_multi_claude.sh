#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

load_claude_code_oauth_token_env() {
  local env_file="${CLAWBLOX_ENV_FILE:-$ROOT_DIR/.env}"
  local line value
  if [[ -n "${CLAUDE_CODE_OAUTH_TOKEN:-}" || ! -f "$env_file" ]]; then
    return 0
  fi
  while IFS= read -r line || [[ -n "$line" ]]; do
    if [[ "$line" =~ ^[[:space:]]*(export[[:space:]]+)?CLAUDE_CODE_OAUTH_TOKEN[[:space:]]*=[[:space:]]*(.*)$ ]]; then
      value="${BASH_REMATCH[2]}"
      value="${value%%#*}"
      value="${value#"${value%%[![:space:]]*}"}"
      value="${value%"${value##*[![:space:]]}"}"
      if [[ "$value" == \"*\" && "$value" == *\" ]]; then
        value="${value:1:${#value}-2}"
      elif [[ "$value" == \'*\' && "$value" == *\' ]]; then
        value="${value:1:${#value}-2}"
      fi
      export CLAUDE_CODE_OAUTH_TOKEN="$value"
      return 0
    fi
  done <"$env_file"
}

load_claude_code_oauth_token_env

NUM_WORLDS="${NUM_WORLDS:-1}"
AGENTS_PER_WORLD="${AGENTS_PER_WORLD:-1}"
BASE_PORT="${BASE_PORT:-8080}"
TMUX_SESSION="${TMUX_SESSION:-clawblox-claude}"
RECORD="${RECORD:-true}"
WORLD_DIR="${WORLD_DIR:-}"
HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-30}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
LAUNCH_INSTANCE_ID="${CLAWBLOX_LAUNCH_INSTANCE_ID:-$(date -u +%Y%m%dT%H%M%SZ)-$$-$RANDOM}"
AGENT_NAME_PREFIX="${AGENT_NAME_PREFIX:-agent}"
STOP_GRACE_SECONDS="${STOP_GRACE_SECONDS:-10}"
GOAL="${GOAL:-}"
TEMPLATE_DIR="${TEMPLATE_DIR:-$SCRIPT_DIR/template/agent}"
RESULTS_ROOT="${RESULTS_ROOT:-}"
PORT_WAIT_SECONDS="${PORT_WAIT_SECONDS:-120}"
RESUME_PATH="${RESUME_PATH:-}"
RESET_EVERY="${RESET_EVERY:-}"
DURATION="${DURATION:-}"
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-1800}"
CHECKPOINT_WARNING_SECONDS="${CHECKPOINT_WARNING_SECONDS:-300}"
CHECKPOINT_WARNING_PROMPT="${CHECKPOINT_WARNING_PROMPT:-You will be reset in 5 minutes. Update your workspace memory files now.}"
CLAUDE_MODEL="${CLAUDE_MODEL:-claude-opus-4-6}"
CLAUDE_PERMISSION_MODE="${CLAUDE_PERMISSION_MODE:-bypassPermissions}"
CLAUDE_BARE="${CLAUDE_BARE:-0}"
CLAUDE_EXTRA_ARGS="${CLAUDE_EXTRA_ARGS:-}"
CLAUDE_USE_ENV_AUTH="${CLAUDE_USE_ENV_AUTH:-0}"
CLAWBLOX_CLAUDE_BIN="${CLAWBLOX_CLAUDE_BIN:-}"
CLAWBLOX_CLAUDE_CODE_VERSION_PIN="${CLAWBLOX_CLAUDE_CODE_VERSION_PIN:-2.1.116}"
SANDBOX="${SANDBOX:-0}"
SKIP_SOUL="${SKIP_SOUL:-0}"
WORLD_CAPABILITY_PROXY="${WORLD_CAPABILITY_PROXY:-auto}"
WORLD_SERVER_CMD="${WORLD_SERVER_CMD:-uv run --with mujoco --with fastapi --with uvicorn python server.py}"
CLAUDE_FAILURE_IDLE_GRACE_SECONDS="${CLAUDE_FAILURE_IDLE_GRACE_SECONDS:-180}"
CLAUDE_FAILURE_STALL_GRACE_SECONDS="${CLAUDE_FAILURE_STALL_GRACE_SECONDS:-300}"
CLAUDE_FAILURE_RESTART_BACKOFF_SECONDS="${CLAUDE_FAILURE_RESTART_BACKOFF_SECONDS:-5}"
EXPECTED_RUN_ID="${CLAWBLOX_EXPECT_RUN_ID:-}"
EXPECTED_LAUNCH_INSTANCE_ID="${CLAWBLOX_EXPECT_LAUNCH_INSTANCE_ID:-}"
CURRENT_AUTO_TIMER_PID="${CLAWBLOX_AUTO_TIMER_PID:-}"
AUTO_STOP_PID="${AUTO_STOP_PID:-}"
declare -a AGENT_TEMPLATE_DIRS=()

AGENT_SCRIPT="$SCRIPT_DIR/run_claude_agent.sh"
WATCHDOG_SCRIPT="$SCRIPT_DIR/watch_claude_recovery.sh"
if [[ ! -f "$AGENT_SCRIPT" ]]; then
  echo "Error: Claude agent runner not found at $AGENT_SCRIPT"
  exit 1
fi

parse_duration_seconds() {
  local input="$1"
  if [[ "$input" =~ ^[0-9]+$ ]]; then
    printf '%s' "$input"
    return 0
  fi
  local total=0 rest="$input"
  while [[ -n "$rest" ]]; do
    if [[ "$rest" =~ ^([0-9]+)([hms]) ]]; then
      local val="${BASH_REMATCH[1]}" unit="${BASH_REMATCH[2]}"
      case "$unit" in
        h) total=$((total + val * 3600)) ;;
        m) total=$((total + val * 60)) ;;
        s) total=$((total + val)) ;;
      esac
      rest="${rest#"${BASH_REMATCH[0]}"}"
    else
      echo "Error: invalid duration format '$input' (use e.g. 2h, 30m, 1h30m, or seconds)" >&2
      return 1
    fi
  done
  printf '%s' "$total"
}

sanitize_name() {
  printf '%s' "$1" | tr -c 'A-Za-z0-9._-' '_'
}

utc_now() {
  date -u +%Y-%m-%dT%H:%M:%SZ
}

refresh_afs_tokens() {
  if [[ -z "${KRB5CCNAME:-}" ]] || ! command -v aklog >/dev/null 2>&1; then
    return 0
  fi
  if ! aklog >/dev/null 2>&1; then
    echo "Warning: aklog failed for KRB5CCNAME=$KRB5CCNAME; AFS paths may be unavailable." >&2
  fi
}

append_launcher_audit() {
  local event="$1"
  local details="${2:-}"
  local meta_file run_dir audit_file tmux_socket parent_cmd self_cmd

  meta_file="$(session_meta_file)"
  run_dir="${RUN_DIR:-}"
  if [[ -f "$meta_file" ]]; then
    meta_run_dir="$(
      bash --noprofile --norc -c '
        set -euo pipefail
        # shellcheck disable=SC1090
        source "$1"
        printf "%s" "${RUN_DIR:-}"
      ' _ "$meta_file" 2>/dev/null || true
    )"
    run_dir="${meta_run_dir:-$run_dir}"
  fi
  if [[ -z "$run_dir" ]]; then
    return 0
  fi

  audit_file="$run_dir/launcher_audit.log"
  mkdir -p "$(dirname "$audit_file")"
  tmux_socket="${TMUX:-}"
  parent_cmd="$(ps -o args= -p "$PPID" 2>/dev/null | tr '\n' ' ' || true)"
  self_cmd="$(ps -o args= -p "$$" 2>/dev/null | tr '\n' ' ' || true)"
  printf '%s event=%s pid=%s ppid=%s tmux_session=%q tmux_env=%q stop_reason=%q stop_source=%q parent_cmd=%q self_cmd=%q details=%q\n' \
    "$(utc_now)" \
    "$event" \
    "$$" \
    "$PPID" \
    "$TMUX_SESSION" \
    "$tmux_socket" \
    "${CLAWBLOX_STOP_REASON:-}" \
    "${CLAWBLOX_STOP_SOURCE:-}" \
    "$parent_cmd" \
    "$self_cmd" \
    "$details" >>"$audit_file"
}

write_agent_pane_command() {
  local target="$1"
  local root_dir="$2"
  local auth_env_prefix="$3"
  local world_url="$4"
  local world_internal_url="$5"
  local agent_display_name="$6"
  local agent_dir="$7"
  local agent_workspace="$8"
  local session_id_file="$9"
  local world_session_file="${10}"
  local reset_every_seconds="${11}"
  local duration_seconds="${12}"
  local checkpoint_warning_seconds="${13}"
  local checkpoint_warning_prompt="${14}"
  local claude_model="${15}"
  local claude_permission_mode="${16}"
  local claude_bare="${17}"
  local claude_extra_args="${18}"
  local sandbox_env_prefix="${19}"
  local template_env_prefix="${20}"
  local startup_env_prefix="${21}"
  local claude_bin="${22}"
  local claude_code_version_pin="${23}"

  cat >"$target" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd $(printf '%q' "$root_dir")
EOF
  cat >>"$target" <<'EOF'
refresh_afs_tokens() {
  if [[ -z "${KRB5CCNAME:-}" ]] || ! command -v aklog >/dev/null 2>&1; then
    return 0
  fi
  if ! aklog >/dev/null 2>&1; then
    echo "Warning: aklog failed for KRB5CCNAME=$KRB5CCNAME; AFS paths may be unavailable." >&2
  fi
}
refresh_afs_tokens
EOF
  cat >>"$target" <<'EOF'
load_claude_code_oauth_token_env() {
  local env_file="${CLAWBLOX_ENV_FILE:-.env}"
  local line value
  if [[ -n "${CLAUDE_CODE_OAUTH_TOKEN:-}" || ! -f "$env_file" ]]; then
    return 0
  fi
  while IFS= read -r line || [[ -n "$line" ]]; do
    if [[ "$line" =~ ^[[:space:]]*(export[[:space:]]+)?CLAUDE_CODE_OAUTH_TOKEN[[:space:]]*=[[:space:]]*(.*)$ ]]; then
      value="${BASH_REMATCH[2]}"
      value="${value%%#*}"
      value="${value#"${value%%[![:space:]]*}"}"
      value="${value%"${value##*[![:space:]]}"}"
      if [[ "$value" == \"*\" && "$value" == *\" ]]; then
        value="${value:1:${#value}-2}"
      elif [[ "$value" == \'*\' && "$value" == *\' ]]; then
        value="${value:1:${#value}-2}"
      fi
      export CLAUDE_CODE_OAUTH_TOKEN="$value"
      return 0
    fi
  done <"$env_file"
}
load_claude_code_oauth_token_env
EOF
  cat >>"$target" <<EOF
${auth_env_prefix}WORLD_BASE_URL=$(printf '%q' "$world_url") WORLD_INTERNAL_BASE_URL=$(printf '%q' "$world_internal_url") WORLD_AGENT_NAME=$(printf '%q' "$agent_display_name") AGENT_DIR=$(printf '%q' "$agent_dir") WORKSPACE_DIR=$(printf '%q' "$agent_workspace") SESSION_ID_FILE=$(printf '%q' "$session_id_file") WORLD_SESSION_FILE=$(printf '%q' "$world_session_file") RESET_EVERY=$(printf '%q' "$reset_every_seconds") DURATION_SECONDS=$(printf '%q' "$duration_seconds") CHECKPOINT_WARNING_SECONDS=$(printf '%q' "$checkpoint_warning_seconds") CHECKPOINT_WARNING_PROMPT=$(printf '%q' "$checkpoint_warning_prompt") CLAUDE_MODEL=$(printf '%q' "$claude_model") CLAUDE_PERMISSION_MODE=$(printf '%q' "$claude_permission_mode") CLAUDE_BARE=$(printf '%q' "$claude_bare") CLAUDE_EXTRA_ARGS=$(printf '%q' "$claude_extra_args") CLAWBLOX_CLAUDE_BIN=$(printf '%q' "$claude_bin") CLAWBLOX_CLAUDE_CODE_VERSION_PIN=$(printf '%q' "$claude_code_version_pin") ${sandbox_env_prefix}${template_env_prefix}${startup_env_prefix}bash $(printf '%q' "$AGENT_SCRIPT")
EOF
  chmod +x "$target"
}

world_capability_proxy_enabled() {
  case "$WORLD_CAPABILITY_PROXY" in
    1|true|yes|on) return 0 ;;
    0|false|no|off) return 1 ;;
    auto) [[ "$SANDBOX" == "1" ]] ;;
    *)
      echo "Error: WORLD_CAPABILITY_PROXY must be auto, true, or false." >&2
      exit 1
      ;;
  esac
}

start_world_capability_proxy() {
  local agent_runtime_dir="$1"
  local agent_log_dir="$2"
  local target_url="$3"
  local public_host="$4"
  local session_file="$5"
  local port_file="$agent_runtime_dir/world_proxy_port.txt"
  local pid_file="$agent_runtime_dir/world_proxy.pid"
  local log_file="$agent_log_dir/world_proxy.log"
  local proxy_script="$SCRIPT_DIR/world_capability_proxy.py"
  local proxy_pid

  if [[ ! -x "$proxy_script" ]]; then
    chmod +x "$proxy_script" 2>/dev/null || true
  fi
  if [[ ! -f "$proxy_script" ]]; then
    echo "Error: missing world capability proxy at $proxy_script" >&2
    exit 1
  fi
  rm -f "$port_file" "$pid_file"
  python3 "$proxy_script" \
    --listen-host 127.0.0.1 \
    --listen-port 0 \
    --target-base-url "$target_url" \
    --public-host "$public_host" \
    --session-token-file "$session_file" \
    --session-header X-Session \
    --port-file "$port_file" \
    --pid-file "$pid_file" \
    --log-file "$log_file" \
    >>"$log_file" 2>&1 &
  proxy_pid=$!

  for _ in {1..50}; do
    if [[ -s "$port_file" ]]; then
      printf '%s\n' "$proxy_pid"
      return 0
    fi
    if ! kill -0 "$proxy_pid" 2>/dev/null; then
      echo "Error: world capability proxy exited early; see $log_file" >&2
      exit 1
    fi
    sleep 0.1
  done
  echo "Error: world capability proxy did not publish a port; see $log_file" >&2
  kill "$proxy_pid" 2>/dev/null || true
  exit 1
}

start_agent_recovery_watchdog() {
  local pane_id="$1"
  local agent_dir="$2"
  local command_file="$3"
  local agent_log_file="$4"
  local runtime_dir pid_file existing_pid watcher_pid

  [[ -x "$WATCHDOG_SCRIPT" ]] || return 0

  runtime_dir="$agent_dir/runtime"
  pid_file="$runtime_dir/recovery_watchdog.pid"
  while IFS= read -r existing_pid; do
    [[ -n "$existing_pid" ]] || continue
    if kill -0 "$existing_pid" 2>/dev/null; then
      kill "$existing_pid" 2>/dev/null || true
      append_launcher_audit "watchdog_cleanup_killed" "agent_dir=$agent_dir watchdog_pid=$existing_pid"
    fi
  done < <(
    ps -eo pid=,args= | awk -v script="$WATCHDOG_SCRIPT" -v agent_dir="$agent_dir" '
      index($0, script) && index($0, agent_dir) { print $1 }
    '
  )
  if [[ -f "$pid_file" ]]; then
    existing_pid="$(tr -d '[:space:]' <"$pid_file" 2>/dev/null || true)"
    if [[ -n "$existing_pid" ]] && kill -0 "$existing_pid" 2>/dev/null; then
      kill "$existing_pid" 2>/dev/null || true
    fi
    rm -f "$pid_file"
  fi

  CLAUDE_FAILURE_IDLE_GRACE_SECONDS="$CLAUDE_FAILURE_IDLE_GRACE_SECONDS" \
    CLAUDE_FAILURE_STALL_GRACE_SECONDS="$CLAUDE_FAILURE_STALL_GRACE_SECONDS" \
    CLAUDE_FAILURE_RESTART_BACKOFF_SECONDS="$CLAUDE_FAILURE_RESTART_BACKOFF_SECONDS" \
    nohup bash "$WATCHDOG_SCRIPT" \
      --pane-id "$pane_id" \
      --agent-dir "$agent_dir" \
      --command-file "$command_file" \
      --agent-log-file "$agent_log_file" \
      >/dev/null 2>&1 &
  watcher_pid="$!"
  printf '%s\n' "$watcher_pid" >"$pid_file"
  append_launcher_audit "watchdog_started" "pane_id=$pane_id agent_dir=$agent_dir watchdog_pid=$watcher_pid"
}

archive_existing_run_dir() {
  local run_dir="$1"
  local ts archive_dir
  ts="$(date -u +%Y%m%dT%H%M%SZ)"
  archive_dir="${run_dir}.preforce-${ts}"
  mv "$run_dir" "$archive_dir"
  echo "Archived existing run directory to: $archive_dir"
}

agent_display_name_for_index() {
  local idx="$1"
  local -a creature_names=(Eko Moa Rua Tavi Oni Zev Ika Pala Sori Nyx)
  if ((idx < ${#creature_names[@]})); then
    printf '%s' "${creature_names[$idx]}"
  else
    printf '%s' "${AGENT_NAME_PREFIX}-a${idx}"
  fi
}

session_meta_file() {
  printf '%s/.clawblox-multi/%s.env' "$ROOT_DIR" "$(sanitize_name "$TMUX_SESSION")"
}

write_run_metadata() {
  local target="$1"
  mkdir -p "$(dirname "$target")"
  cat >"$target" <<EOF
RUN_ID=$(printf '%q' "$RUN_ID")
RUN_SAFE_ID=$(printf '%q' "$RUN_SAFE_ID")
RUN_DIR=$(printf '%q' "$RUN_DIR")
WORLDS_ROOT=$(printf '%q' "$WORLDS_ROOT")
WORLD_ABS_DIR=$(printf '%q' "$WORLD_ABS_DIR")
NUM_WORLDS=$(printf '%q' "$NUM_WORLDS")
AGENTS_PER_WORLD=$(printf '%q' "$AGENTS_PER_WORLD")
BASE_PORT=$(printf '%q' "$BASE_PORT")
RECORD=$(printf '%q' "$RECORD")
RESET_EVERY=$(printf '%q' "$RESET_EVERY")
WORLD_SERVER_CMD=$(printf '%q' "$WORLD_SERVER_CMD")
TMUX_SESSION=$(printf '%q' "$TMUX_SESSION")
EOF
}

write_session_metadata() {
  local target="$1"
  write_run_metadata "$target"
  cat >>"$target" <<EOF
LAUNCH_INSTANCE_ID=$(printf '%q' "$LAUNCH_INSTANCE_ID")
AUTO_STOP_PID=$(printf '%q' "$AUTO_STOP_PID")
EOF
}

# Serialize the full launch configuration to a machine-readable JSON file.
# Consumed by downstream replay/eval tooling (e.g. analysis/eval_goal_by_episode.py)
# to inherit agent-config flags from a saved run without re-parsing bash state.
write_run_config() {
  local target="$1"
  mkdir -p "$(dirname "$target")"
  local agent_template_dirs_joined=""
  if ((${#AGENT_TEMPLATE_DIRS[@]} > 0)); then
    agent_template_dirs_joined="$(printf '%s\n' "${AGENT_TEMPLATE_DIRS[@]}")"
  fi
  RUN_CONFIG_TARGET="$target" \
  RUN_ID="$RUN_ID" \
  LAUNCH_INSTANCE_ID="$LAUNCH_INSTANCE_ID" \
  NUM_WORLDS="$NUM_WORLDS" \
  AGENTS_PER_WORLD="$AGENTS_PER_WORLD" \
  BASE_PORT="$BASE_PORT" \
  WORLD_ABS_DIR="$WORLD_ABS_DIR" \
  RESULTS_ABS_ROOT="$RESULTS_ABS_ROOT" \
  TMUX_SESSION="$TMUX_SESSION" \
  GOAL="$GOAL" \
  CLAUDE_MODEL="$CLAUDE_MODEL" \
  CLAUDE_PERMISSION_MODE="$CLAUDE_PERMISSION_MODE" \
  CLAUDE_BARE="$CLAUDE_BARE" \
  CLAUDE_EXTRA_ARGS="$CLAUDE_EXTRA_ARGS" \
  CLAUDE_USE_ENV_AUTH="$CLAUDE_USE_ENV_AUTH" \
  CLAUDE_CODE_OAUTH_TOKEN_PRESENT="$([[ -n "${CLAUDE_CODE_OAUTH_TOKEN:-}" ]] && printf 1 || printf 0)" \
  CLAWBLOX_CLAUDE_BIN="$CLAWBLOX_CLAUDE_BIN" \
  CLAWBLOX_CLAUDE_CODE_VERSION_PIN="$CLAWBLOX_CLAUDE_CODE_VERSION_PIN" \
  SANDBOX="$SANDBOX" \
  SKIP_SOUL="$SKIP_SOUL" \
  WORLD_CAPABILITY_PROXY="$WORLD_CAPABILITY_PROXY" \
  SYSTEM_PROMPT_TEMPLATE="${SYSTEM_PROMPT_TEMPLATE:-}" \
  AGENT_NAME_PREFIX="$AGENT_NAME_PREFIX" \
  AGENT_TEMPLATE_DIRS_JOINED="$agent_template_dirs_joined" \
  DURATION="$DURATION" \
  RESET_EVERY="$RESET_EVERY" \
  CHECKPOINT_INTERVAL="$CHECKPOINT_INTERVAL" \
  RECORD="$RECORD" \
  HEALTH_TIMEOUT="$HEALTH_TIMEOUT" \
  STOP_GRACE_SECONDS="$STOP_GRACE_SECONDS" \
  PORT_WAIT_SECONDS="$PORT_WAIT_SECONDS" \
  WORLD_SERVER_CMD="$WORLD_SERVER_CMD" \
  python3 <<'PY'
import json
import os

def _bool(val: str) -> bool:
    return val.strip() in {"1", "true", "True", "yes"}

config = {
    "schema_version": 1,
    "run_id": os.environ["RUN_ID"],
    "launch_instance_id": os.environ["LAUNCH_INSTANCE_ID"],
    "num_worlds": int(os.environ["NUM_WORLDS"]),
    "agents_per_world": int(os.environ["AGENTS_PER_WORLD"]),
    "base_port": int(os.environ["BASE_PORT"]),
    "world_dir": os.environ["WORLD_ABS_DIR"],
    "results_root": os.environ["RESULTS_ABS_ROOT"],
    "tmux_session": os.environ["TMUX_SESSION"],
    "goal": os.environ.get("GOAL", ""),
    "claude": {
        "model": os.environ["CLAUDE_MODEL"],
        "permission_mode": os.environ["CLAUDE_PERMISSION_MODE"],
        "bare": _bool(os.environ["CLAUDE_BARE"]),
        "extra_args": os.environ.get("CLAUDE_EXTRA_ARGS", ""),
        "use_env_auth": _bool(os.environ["CLAUDE_USE_ENV_AUTH"]),
        "uses_claude_code_oauth_token": _bool(os.environ["CLAUDE_CODE_OAUTH_TOKEN_PRESENT"]),
        "binary_override": os.environ.get("CLAWBLOX_CLAUDE_BIN", ""),
        "version_pin": os.environ.get("CLAWBLOX_CLAUDE_CODE_VERSION_PIN", ""),
    },
    "agent": {
        "sandbox": _bool(os.environ["SANDBOX"]),
        "skip_soul": _bool(os.environ["SKIP_SOUL"]),
        "world_capability_proxy": os.environ["WORLD_CAPABILITY_PROXY"],
        "system_prompt_template": os.environ.get("SYSTEM_PROMPT_TEMPLATE", ""),
        "agent_name_prefix": os.environ["AGENT_NAME_PREFIX"],
        "agent_template_dirs": [
            line for line in os.environ.get("AGENT_TEMPLATE_DIRS_JOINED", "").splitlines() if line
        ],
    },
    "session_cycling": {
        "duration": os.environ.get("DURATION", ""),
        "reset_every": os.environ.get("RESET_EVERY", ""),
        "checkpoint_interval": os.environ.get("CHECKPOINT_INTERVAL", ""),
    },
    "plumbing": {
        "record": os.environ["RECORD"],
        "health_timeout": os.environ["HEALTH_TIMEOUT"],
        "stop_grace_seconds": os.environ["STOP_GRACE_SECONDS"],
        "port_wait_seconds": os.environ["PORT_WAIT_SECONDS"],
        "world_server_cmd": os.environ["WORLD_SERVER_CMD"],
    },
}

with open(os.environ["RUN_CONFIG_TARGET"], "w", encoding="utf-8") as f:
    json.dump(config, f, indent=2, sort_keys=True)
    f.write("\n")
PY
}

load_session_metadata() {
  local meta_file="$1"
  local -n out_run_id_ref="$2"
  local -n out_run_dir_ref="$3"
  local -n out_num_worlds_ref="$4"
  local -n out_base_port_ref="$5"
  local -n out_launch_instance_ref="$6"
  local -n out_auto_stop_pid_ref="$7"
  local -a meta_fields=()

  out_run_id_ref=""
  out_run_dir_ref=""
  out_num_worlds_ref=""
  out_base_port_ref=""
  out_launch_instance_ref=""
  out_auto_stop_pid_ref=""

  [[ -f "$meta_file" ]] || return 1

  mapfile -t meta_fields < <(
    bash --noprofile --norc -c '
      set -euo pipefail
      # shellcheck disable=SC1090
      source "$1"
      printf "%s\n%s\n%s\n%s\n%s\n%s\n" \
        "${RUN_ID:-}" \
        "${RUN_DIR:-}" \
        "${NUM_WORLDS:-}" \
        "${BASE_PORT:-}" \
        "${LAUNCH_INSTANCE_ID:-}" \
        "${AUTO_STOP_PID:-}"
    ' _ "$meta_file"
  )
  out_run_id_ref="${meta_fields[0]:-}"
  out_run_dir_ref="${meta_fields[1]:-}"
  out_num_worlds_ref="${meta_fields[2]:-}"
  out_base_port_ref="${meta_fields[3]:-}"
  out_launch_instance_ref="${meta_fields[4]:-}"
  out_auto_stop_pid_ref="${meta_fields[5]:-}"
}

stop_requires_metadata_match() {
  [[ -n "$EXPECTED_RUN_ID" || -n "$EXPECTED_LAUNCH_INSTANCE_ID" ]]
}

kill_recorded_auto_stop_pid() {
  local meta_file="$1"
  local skip_pid="${2:-}"
  local meta_run_id meta_run_dir meta_num_worlds meta_base_port meta_launch_instance meta_auto_stop_pid
  local pid_cmd

  if ! load_session_metadata \
    "$meta_file" \
    meta_run_id \
    meta_run_dir \
    meta_num_worlds \
    meta_base_port \
    meta_launch_instance \
    meta_auto_stop_pid; then
    return 0
  fi

  [[ -n "$meta_auto_stop_pid" ]] || return 0
  if [[ -n "$skip_pid" && "$meta_auto_stop_pid" == "$skip_pid" ]]; then
    return 0
  fi
  if ! kill -0 "$meta_auto_stop_pid" 2>/dev/null; then
    return 0
  fi
  pid_cmd="$(ps -o args= -p "$meta_auto_stop_pid" 2>/dev/null | tr '\n' ' ' || true)"
  if [[ "$pid_cmd" != *"$(basename "$0")"* ]]; then
    append_launcher_audit "auto_stop_cleanup_skipped" "pid=$meta_auto_stop_pid reason=command_mismatch cmd=$pid_cmd"
    return 0
  fi
  kill "$meta_auto_stop_pid" 2>/dev/null || true
  append_launcher_audit "auto_stop_cleanup_killed" "pid=$meta_auto_stop_pid launch_instance_id=$meta_launch_instance run_id=$meta_run_id"
}

world_root_for_index() {
  local world_index="$1"
  printf '%s/worlds/world-%s' "$RUN_DIR" "$world_index"
}

world_log_for_index() {
  local world_index="$1"
  printf '%s/logs/world.log' "$(world_root_for_index "$world_index")"
}

world_spectator_token_file() {
  local world_index="$1"
  printf '%s/spectator_token.txt' "$(world_root_for_index "$world_index")"
}

extract_spectator_token_from_log() {
  local world_log="$1"
  [[ -f "$world_log" ]] || return 1
  sed -n 's|^Spectator frontend: .*spectator_token=\([A-Za-z0-9-][A-Za-z0-9-]*\)$|\1|p' "$world_log" | tail -n 1
}

extract_spectator_token_from_tmux_pane() {
  local world_index="$1"
  tmux capture-pane -J -p -S -2000 -t "${TMUX_SESSION}:world-${world_index}" 2>/dev/null \
    | sed -n 's|^Spectator frontend: .*spectator_token=\([A-Za-z0-9-][A-Za-z0-9-]*\)$|\1|p' \
    | tail -n 1
}

cache_world_spectator_token() {
  local world_index="$1"
  local token_file
  local token
  token_file="$(world_spectator_token_file "$world_index")"
  token="$(extract_spectator_token_from_log "$(world_log_for_index "$world_index")")" || true
  if [[ -z "$token" ]]; then
    token="$(extract_spectator_token_from_tmux_pane "$world_index")" || return 1
  fi
  [[ -n "$token" ]] || return 1
  printf '%s\n' "$token" >"$token_file"
}

load_world_spectator_token() {
  local world_index="$1"
  local token_file
  token_file="$(world_spectator_token_file "$world_index")"
  if [[ -f "$token_file" ]]; then
    tr -d '[:space:]' <"$token_file"
    return 0
  fi
  cache_world_spectator_token "$world_index" >/dev/null 2>&1 || return 1
  tr -d '[:space:]' <"$token_file"
}

save_world_snapshot() {
  local world_index="$1"
  local port="$2"
  local world_root
  local spectator_token=""
  local -a curl_args=()
  world_root="$(world_root_for_index "$world_index")"
  local resume_dir="$world_root/resume"
  local snapshot_file="$resume_dir/latest.json"
  local tmp_file="$snapshot_file.tmp"
  mkdir -p "$resume_dir"
  if spectator_token="$(load_world_spectator_token "$world_index" 2>/dev/null)"; then
    curl_args=(-H "X-Spectator-Token: $spectator_token")
  fi
  if curl -fsS "${curl_args[@]}" "http://localhost:${port}/snapshot" >"$tmp_file"; then
    local ts
    ts="$(date -u +%Y%m%dT%H%M%SZ)"
    cp "$tmp_file" "$resume_dir/checkpoint_${ts}.json"
    mv "$tmp_file" "$snapshot_file"
    echo "Saved snapshot: $snapshot_file (checkpoint_${ts}.json)"
    return 0
  fi
  rm -f "$tmp_file"
  echo "Error: failed to save snapshot for world-$world_index on port $port" >&2
  return 1
}

kill_world_capability_proxies() {
  local root="${1:-}"
  local pid_file pid
  [[ -n "$root" && -d "$root" ]] || return 0
  while IFS= read -r -d '' pid_file; do
    pid="$(tr -d '[:space:]' < "$pid_file" 2>/dev/null || true)"
    [[ -n "$pid" ]] || continue
    if kill -0 "$pid" 2>/dev/null; then
      kill "$pid" 2>/dev/null || true
      append_launcher_audit "world_capability_proxy_killed" "pid=$pid pid_file=$pid_file"
    fi
    rm -f "$pid_file"
  done < <(find "$root" -type f -path '*/runtime/world_proxy.pid' -print0 2>/dev/null)
  return 0
}

usage() {
  cat <<EOF
Usage: $(basename "$0") [--stop|--status] [options]

Launch a simulator world with Claude Code agents in tmux.

Options include:
  --world-server-cmd CMD        Command used to start the simulator; --port is appended
  --skip-soul, --no-soul        Do not seed or append SOUL.md for Claude agents
EOF
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

is_nonnegative_int() { [[ "$1" =~ ^[0-9]+$ ]]; }
is_positive_int() { [[ "$1" =~ ^[1-9][0-9]*$ ]]; }

verify_recording_db() {
  local recording_file="$1"
  if [[ ! -f "$recording_file" ]]; then
    return 0
  fi
  if [[ -f "${recording_file}-wal" || -f "${recording_file}-shm" ]]; then
    echo "Warning: recording not finalized cleanly: $recording_file"
  fi
}

collect_session_recordings() {
  local window_name
  tmux list-windows -t "$TMUX_SESSION" -F "#{window_name}" 2>/dev/null | while IFS= read -r window_name; do
    [[ "$window_name" == world-* ]] || continue
    tmux capture-pane -J -p -S -2000 -t "${TMUX_SESSION}:${window_name}" 2>/dev/null | sed -n 's/^Recording to: //p' | tail -n 1
  done | awk 'NF && !seen[$0]++'
}

do_stop() {
  local recordings=()
  local meta_file
  local i
  local port
  local window_names=()
  local world_window_count=0
  local snapshot_failed=0
  local meta_run_id=""
  local meta_run_dir=""
  local meta_num_worlds=""
  local meta_base_port=""
  local meta_launch_instance_id=""
  local meta_auto_stop_pid=""
  refresh_afs_tokens
  append_launcher_audit "stop_requested" "force=$FORCE stop_grace_seconds=$STOP_GRACE_SECONDS"
  if tmux has-session -t "$TMUX_SESSION" 2>/dev/null; then
    echo "Cleanup: tmux session '$TMUX_SESSION' exists."
    mapfile -t recordings < <(collect_session_recordings)
    mapfile -t window_names < <(tmux list-windows -t "$TMUX_SESSION" -F "#{window_name}" 2>/dev/null)
    meta_file="$(session_meta_file)"
    echo "Cleanup: metadata file: $meta_file"
    if load_session_metadata \
      "$meta_file" \
      meta_run_id \
      meta_run_dir \
      meta_num_worlds \
      meta_base_port \
      meta_launch_instance_id \
      meta_auto_stop_pid; then
      if [[ -n "$EXPECTED_RUN_ID" && "$meta_run_id" != "$EXPECTED_RUN_ID" ]]; then
        echo "Skipping stop for '$TMUX_SESSION': expected run '$EXPECTED_RUN_ID', current owner is '${meta_run_id:-unknown}'."
        append_launcher_audit "stop_skipped_run_mismatch" "expected_run_id=$EXPECTED_RUN_ID current_run_id=${meta_run_id:-} force=$FORCE"
        return 0
      fi
      if [[ -n "$EXPECTED_LAUNCH_INSTANCE_ID" && "$meta_launch_instance_id" != "$EXPECTED_LAUNCH_INSTANCE_ID" ]]; then
        echo "Skipping stop for '$TMUX_SESSION': expected launch instance '$EXPECTED_LAUNCH_INSTANCE_ID', current owner is '${meta_launch_instance_id:-unknown}'."
        append_launcher_audit "stop_skipped_launch_instance_mismatch" "expected_launch_instance_id=$EXPECTED_LAUNCH_INSTANCE_ID current_launch_instance_id=${meta_launch_instance_id:-} force=$FORCE"
        return 0
      fi
      echo "Cleanup: owner run_id=${meta_run_id:-unknown} run_dir=${meta_run_dir:-unknown} base_port=${meta_base_port:-unknown} worlds=${meta_num_worlds:-unknown}."
      kill_recorded_auto_stop_pid "$meta_file" "$CURRENT_AUTO_TIMER_PID"
      if [[ -n "$meta_run_dir" && -n "$meta_num_worlds" && -n "$meta_base_port" ]]; then
        RUN_DIR="$meta_run_dir"
        WORLDS_ROOT="$RUN_DIR/worlds"
        echo "Cleanup: saving world snapshots..."
        for ((i = 0; i < meta_num_worlds; i++)); do
          port=$((meta_base_port + i))
          echo "Cleanup: snapshot world-$i via port $port."
          if ! save_world_snapshot "$i" "$port"; then
            snapshot_failed=1
          fi
        done
        if ((snapshot_failed != 0)); then
          append_launcher_audit "stop_snapshot_failed" "num_worlds=$meta_num_worlds base_port=$meta_base_port"
          if ((FORCE == 1)); then
            echo "Warning: snapshot capture failed, continuing with --force." >&2
          else
            echo "Snapshot capture failed. Aborting shutdown so the live run remains resumable." >&2
            exit 1
          fi
        fi
      fi
    elif stop_requires_metadata_match; then
      echo "Skipping stop for '$TMUX_SESSION': expected ownership metadata is missing or unreadable."
      append_launcher_audit "stop_skipped_missing_metadata" "expected_run_id=$EXPECTED_RUN_ID expected_launch_instance_id=$EXPECTED_LAUNCH_INSTANCE_ID force=$FORCE meta_file=$meta_file"
      return 0
    fi
    echo "Cleanup: stopping world servers and agent panes..."
    kill_world_capability_proxies "$meta_run_dir" || true
    for window_name in "${window_names[@]}"; do
      [[ "$window_name" == world-* ]] || continue
      ((++world_window_count))
      echo "Cleanup: sending Ctrl-C to ${TMUX_SESSION}:${window_name}."
      tmux send-keys -t "${TMUX_SESSION}:${window_name}" C-c 2>/dev/null || true
    done
    echo "Cleanup: sent Ctrl-C to $world_window_count world window(s)."
    echo "Cleanup: waiting ${STOP_GRACE_SECONDS}s for tmux panes to exit..."
    sleep "$STOP_GRACE_SECONDS"
    if tmux has-session -t "$TMUX_SESSION" 2>/dev/null; then
      echo "Cleanup: killing tmux session '$TMUX_SESSION'..."
      tmux kill-session -t "$TMUX_SESSION" 2>/dev/null || true
    fi
    if tmux has-session -t "$TMUX_SESSION" 2>/dev/null; then
      echo "Warning: tmux session '$TMUX_SESSION' still exists after kill-session." >&2
    else
      echo "Cleanup: tmux session '$TMUX_SESSION' is gone."
    fi
    if ((${#recordings[@]} > 0)); then
      echo "Cleanup: verifying ${#recordings[@]} recording database(s)..."
      for recording_file in "${recordings[@]}"; do
        verify_recording_db "$recording_file"
      done
    fi
    echo "Cleanup: stopped."
    append_launcher_audit "stop_completed" "recordings=${#recordings[@]} snapshot_failed=$snapshot_failed"
    rm -f "$meta_file"
  else
    echo "No tmux session '$TMUX_SESSION' found."
    append_launcher_audit "stop_no_session" "force=$FORCE"
  fi
}

do_status() {
  if ! tmux has-session -t "$TMUX_SESSION" 2>/dev/null; then
    echo "No tmux session '$TMUX_SESSION' running."
    exit 1
  fi
  echo "Session: $TMUX_SESSION"
  echo "Windows:"
  tmux list-windows -t "$TMUX_SESSION" -F "  #{window_index}: #{window_name} (#{window_panes} panes)"
}

wait_for_server() {
  local port="$1"
  local deadline=$((SECONDS + HEALTH_TIMEOUT))
  while ((SECONDS < deadline)); do
    if curl -sf "http://localhost:${port}/api.md" >/dev/null 2>&1; then
      return 0
    fi
    sleep 1
  done
  return 1
}

port_is_in_use() {
  local port="$1"
  (echo >/dev/tcp/127.0.0.1/"$port") >/dev/null 2>&1
}

ensure_ports_available() {
  local i port waited=0 deadline=0
  local busy_ports=() pids=()
  while true; do
    busy_ports=()
    for ((i = 0; i < NUM_WORLDS; i++)); do
      port=$((BASE_PORT + i))
      if port_is_in_use "$port"; then
        busy_ports+=("$port")
      fi
    done
    if ((${#busy_ports[@]} == 0)); then
      return 0
    fi
    if ((PORT_WAIT_SECONDS == 0)); then
      break
    fi
    if ((waited == 0)); then
      deadline=$((SECONDS + PORT_WAIT_SECONDS))
      echo "Target ports are busy (${busy_ports[*]}). Waiting up to ${PORT_WAIT_SECONDS}s for release..."
    fi
    if ((SECONDS >= deadline)); then
      break
    fi
    waited=1
    sleep 1
  done
  for port in "${busy_ports[@]}"; do
    echo "Error: target port $port is already in use."
    if command -v lsof >/dev/null 2>&1; then
      mapfile -t pids < <(lsof -nP -t -iTCP:"$port" -sTCP:LISTEN 2>/dev/null | sort -u)
      if ((${#pids[@]} > 0)); then
        echo "  listening pids: ${pids[*]}"
      fi
    fi
  done
  exit 1
}

require_claude_auth() {
  local claude_bin=""

  if [[ -n "$CLAWBLOX_CLAUDE_BIN" ]]; then
    if [[ -x "$CLAWBLOX_CLAUDE_BIN" ]]; then
      claude_bin="$CLAWBLOX_CLAUDE_BIN"
    else
      echo "Error: CLAWBLOX_CLAUDE_BIN is not executable: $CLAWBLOX_CLAUDE_BIN"
      exit 1
    fi
  elif [[ -n "$CLAWBLOX_CLAUDE_CODE_VERSION_PIN" ]]; then
    local version_bin="$HOME/.local/share/claude/versions/$CLAWBLOX_CLAUDE_CODE_VERSION_PIN"
    if [[ -x "$version_bin" ]]; then
      claude_bin="$version_bin"
    fi
  fi

  if [[ -z "$claude_bin" ]]; then
    if ! claude_bin="$(command -v claude 2>/dev/null)"; then
      echo "Error: claude CLI was not found in PATH."
      exit 1
    fi
  fi

  if ! "$claude_bin" auth status --text >/dev/null 2>&1; then
    echo "Error: Claude Code is not logged in for $claude_bin. Run: $claude_bin auth login"
    exit 1
  fi
}

require_sandbox_runtime() {
  local native_sandbox="${CLAUDE_NATIVE_SANDBOX:-$SANDBOX}"
  if [[ "$SANDBOX" != "1" ]]; then
    return 0
  fi
  if ! command -v bwrap >/dev/null 2>&1; then
    echo "Error: --sandbox requires bubblewrap (bwrap)." >&2
    exit 1
  fi
  if [[ "$native_sandbox" == "1" ]] && ! command -v socat >/dev/null 2>&1; then
    echo "Error: --sandbox requires socat for Claude Code's native Linux sandbox." >&2
    echo "Install socat, or set CLAUDE_NATIVE_SANDBOX=0 only for non-secure local debugging." >&2
    exit 1
  fi
}

FORCE=0
ACTION="run"
CLI_NUM_WORLDS=0
CLI_AGENTS_PER_WORLD=0
CLI_BASE_PORT=0
CLI_TMUX_SESSION=0
CLI_RECORD=0
CLI_RESET_EVERY=0
CLI_WORLD_SERVER_CMD=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --stop) ACTION="stop"; shift ;;
    --status) ACTION="status"; shift ;;
    --force) FORCE=1; shift ;;
    --num-worlds) require_value "$1" "${2:-}"; NUM_WORLDS="$2"; CLI_NUM_WORLDS=1; shift 2 ;;
    --agents-per-world) require_value "$1" "${2:-}"; AGENTS_PER_WORLD="$2"; CLI_AGENTS_PER_WORLD=1; shift 2 ;;
    --base-port) require_value "$1" "${2:-}"; BASE_PORT="$2"; CLI_BASE_PORT=1; shift 2 ;;
    --tmux-session) require_value "$1" "${2:-}"; TMUX_SESSION="$2"; CLI_TMUX_SESSION=1; shift 2 ;;
    --record) require_value "$1" "${2:-}"; RECORD="$2"; CLI_RECORD=1; shift 2 ;;
    --world-dir) require_value "$1" "${2:-}"; WORLD_DIR="$2"; shift 2 ;;
    --health-timeout) require_value "$1" "${2:-}"; HEALTH_TIMEOUT="$2"; shift 2 ;;
    --run-id) require_value "$1" "${2:-}"; RUN_ID="$2"; shift 2 ;;
    --agent-name-prefix) require_value "$1" "${2:-}"; AGENT_NAME_PREFIX="$2"; shift 2 ;;
    --stop-grace-seconds) require_value "$1" "${2:-}"; STOP_GRACE_SECONDS="$2"; shift 2 ;;
    --port-wait-seconds) require_value "$1" "${2:-}"; PORT_WAIT_SECONDS="$2"; shift 2 ;;
    --world-server-cmd) require_value "$1" "${2:-}"; WORLD_SERVER_CMD="$2"; CLI_WORLD_SERVER_CMD=1; shift 2 ;;
    --goal) require_value "$1" "${2:-}"; GOAL="$2"; shift 2 ;;
    --reset-every) require_value "$1" "${2:-}"; RESET_EVERY="$2"; CLI_RESET_EVERY=1; shift 2 ;;
    --duration) require_value "$1" "${2:-}"; DURATION="$2"; shift 2 ;;
    --model) require_value "$1" "${2:-}"; CLAUDE_MODEL="$2"; shift 2 ;;
    --permission-mode) require_value "$1" "${2:-}"; CLAUDE_PERMISSION_MODE="$2"; shift 2 ;;
    --claude-extra-args) require_value "$1" "${2:-}"; CLAUDE_EXTRA_ARGS="$2"; shift 2 ;;
    --use-env-auth) CLAUDE_USE_ENV_AUTH=1; shift ;;
    --sandbox) SANDBOX=1; shift ;;
    --skip-soul|--no-soul) SKIP_SOUL=1; shift ;;
    --world-capability-proxy) WORLD_CAPABILITY_PROXY=true; shift ;;
    --no-world-capability-proxy) WORLD_CAPABILITY_PROXY=false; shift ;;
    --template|--template-dir) require_value "$1" "${2:-}"; TEMPLATE_DIR="$2"; shift 2 ;;
    --system-prompt) require_value "$1" "${2:-}"; SYSTEM_PROMPT_TEMPLATE="$2"; shift 2 ;;
    --agent-template) require_value "$1" "${2:-}"; AGENT_TEMPLATE_DIRS+=("$2"); shift 2 ;;
    --results-root) require_value "$1" "${2:-}"; RESULTS_ROOT="$2"; shift 2 ;;
    --resume) require_value "$1" "${2:-}"; RESUME_PATH="$2"; shift 2 ;;
    --checkpoint-interval) require_value "$1" "${2:-}"; CHECKPOINT_INTERVAL="$2"; shift 2 ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown option: $1"; usage; exit 1 ;;
  esac
done

case "$ACTION" in
  stop) do_stop; exit 0 ;;
  status) do_status; exit 0 ;;
esac

if [[ "$SKIP_SOUL" != "0" && "$SKIP_SOUL" != "1" ]]; then
  echo "Error: SKIP_SOUL must be 0 or 1 (got '$SKIP_SOUL')."
  exit 1
fi

refresh_afs_tokens
require_claude_auth
require_sandbox_runtime

if tmux has-session -t "$TMUX_SESSION" 2>/dev/null; then
  if [[ "$FORCE" -eq 1 ]]; then
    echo "Replacing existing session '$TMUX_SESSION'..."
    kill_recorded_auto_stop_pid "$(session_meta_file)"
    do_stop
  else
    echo "Tmux session '$TMUX_SESSION' already exists."
    exit 1
  fi
fi

if [[ -f "$(session_meta_file)" ]]; then
  kill_recorded_auto_stop_pid "$(session_meta_file)"
fi

if [[ -n "$RESUME_PATH" ]]; then
  requested_num_worlds="$NUM_WORLDS"
  requested_agents_per_world="$AGENTS_PER_WORLD"
  requested_base_port="$BASE_PORT"
  requested_tmux_session="$TMUX_SESSION"
  requested_record="$RECORD"
  requested_reset_every="$RESET_EVERY"
  requested_world_server_cmd="${WORLD_SERVER_CMD:-}"
  if [[ "$RESUME_PATH" = /* ]]; then
    RUN_DIR="$RESUME_PATH"
  else
    RUN_DIR="$ROOT_DIR/$RESUME_PATH"
  fi
  if [[ ! -d "$RUN_DIR" ]]; then
    echo "Error: resume run directory not found at $RUN_DIR"
    exit 1
  fi
  RUN_META_FILE="$RUN_DIR/run.env"
  if [[ ! -f "$RUN_META_FILE" ]]; then
    echo "Error: missing run metadata at $RUN_META_FILE"
    exit 1
  fi
  # shellcheck disable=SC1090
  source "$RUN_META_FILE"
  if ((CLI_NUM_WORLDS == 1)); then
    NUM_WORLDS="$requested_num_worlds"
  fi
  if ((CLI_AGENTS_PER_WORLD == 1)); then
    AGENTS_PER_WORLD="$requested_agents_per_world"
  fi
  if ((CLI_BASE_PORT == 1)); then
    BASE_PORT="$requested_base_port"
  fi
  if ((CLI_TMUX_SESSION == 1)); then
    TMUX_SESSION="$requested_tmux_session"
  fi
  if ((CLI_RECORD == 1)); then
    RECORD="$requested_record"
  fi
  if ((CLI_RESET_EVERY == 1)); then
    RESET_EVERY="$requested_reset_every"
  fi
  if ((CLI_WORLD_SERVER_CMD == 1)); then
    WORLD_SERVER_CMD="$requested_world_server_cmd"
  fi
  RESUME_PATH="$RUN_DIR"
  RESULTS_ABS_ROOT="$(dirname "$RUN_DIR")"
  for ((i = 0; i < NUM_WORLDS; i++)); do
    if [[ ! -f "$RUN_DIR/worlds/world-$i/resume/latest.json" ]]; then
      echo "Error: missing resume snapshot for world-$i at $RUN_DIR/worlds/world-$i/resume/latest.json"
      exit 1
    fi
  done
else
  if [[ -z "$WORLD_DIR" ]]; then
    echo "Error: --world-dir is required"
    exit 1
  fi
  if [[ "$WORLD_DIR" = /* ]]; then
    WORLD_ABS_DIR="$WORLD_DIR"
  else
    WORLD_ABS_DIR="$ROOT_DIR/${WORLD_DIR#./}"
  fi
fi

if [[ ! -d "$WORLD_ABS_DIR" ]]; then
  echo "Error: world directory not found at $WORLD_ABS_DIR"
  exit 1
fi

TEMPLATE_ABS_DIR="$TEMPLATE_DIR"
if [[ "$TEMPLATE_ABS_DIR" != /* ]]; then
  TEMPLATE_ABS_DIR="$ROOT_DIR/$TEMPLATE_ABS_DIR"
fi

if [[ -z "${SYSTEM_PROMPT_TEMPLATE:-}" ]]; then
  if [[ -f "$WORLD_ABS_DIR/system_prompt.md" ]]; then
    SYSTEM_PROMPT_TEMPLATE="$WORLD_ABS_DIR/system_prompt.md"
  elif [[ -f "$ROOT_DIR/worlds/mujoco-panda/system_prompt.md" ]]; then
    SYSTEM_PROMPT_TEMPLATE="$ROOT_DIR/worlds/mujoco-panda/system_prompt.md"
  elif [[ -f "$TEMPLATE_ABS_DIR/system_prompt.md" ]]; then
    SYSTEM_PROMPT_TEMPLATE="$TEMPLATE_ABS_DIR/system_prompt.md"
  else
    echo "Error: no system_prompt.md found in world dir ($WORLD_ABS_DIR) or template dir ($TEMPLATE_ABS_DIR)" >&2
    exit 1
  fi
fi

if ((${#AGENT_TEMPLATE_DIRS[@]} > AGENTS_PER_WORLD)); then
  echo "Error: received ${#AGENT_TEMPLATE_DIRS[@]} --agent-template values for only $AGENTS_PER_WORLD agents."
  exit 1
fi

declare -a AGENT_TEMPLATE_ABS_DIRS=()
if ((${#AGENT_TEMPLATE_DIRS[@]} > 0)); then
  for template_dir in "${AGENT_TEMPLATE_DIRS[@]}"; do
    template_abs_dir="$template_dir"
    if [[ "$template_abs_dir" != /* ]]; then
      template_abs_dir="$ROOT_DIR/$template_abs_dir"
    fi
    if [[ ! -d "$template_abs_dir" ]]; then
      echo "Error: agent template directory not found at $template_abs_dir"
      exit 1
    fi
    AGENT_TEMPLATE_ABS_DIRS+=("$template_abs_dir")
  done
fi

RUN_SAFE_ID="$(sanitize_name "$RUN_ID")"
if [[ -z "${RESULTS_ABS_ROOT:-}" ]]; then
  if [[ -n "$RESULTS_ROOT" ]]; then
    if [[ "$RESULTS_ROOT" = /* ]]; then
      RESULTS_ABS_ROOT="$RESULTS_ROOT"
    else
      RESULTS_ABS_ROOT="$ROOT_DIR/$RESULTS_ROOT"
    fi
  else
    RESULTS_ABS_ROOT="$WORLD_ABS_DIR/results"
  fi
fi

if [[ -z "${RUN_DIR:-}" ]]; then
  RUN_DIR="$RESULTS_ABS_ROOT/$RUN_SAFE_ID"
fi
if [[ -z "${WORLDS_ROOT:-}" ]]; then
  WORLDS_ROOT="$RUN_DIR/worlds"
fi

if [[ -n "$RESUME_PATH" ]]; then
  for ((i = 0; i < NUM_WORLDS; i++)); do
    for ((j = 0; j < AGENTS_PER_WORLD; j++)); do
      agent_display_name="$(agent_display_name_for_index "$j")"
      agent_name="$(sanitize_name "${agent_display_name}-r${RUN_SAFE_ID}-w${i}-a${j}")"
      agent_dir="$RUN_DIR/worlds/world-$i/agents/$agent_name"
      if [[ ! -d "$agent_dir" ]]; then
        echo "Error: missing agent directory for resume at $agent_dir"
        exit 1
      fi
      if [[ ! -f "$agent_dir/world_session.txt" ]]; then
        echo "Error: missing world session file for resume at $agent_dir/world_session.txt"
        exit 1
      fi
    done
  done
fi

if [[ -z "$RESUME_PATH" && -e "$RUN_DIR" ]]; then
  if ((FORCE == 1)); then
    archive_existing_run_dir "$RUN_DIR"
  else
    echo "Error: run directory already exists: $RUN_DIR"
    echo "Set RUN_ID to a unique value and retry."
    exit 1
  fi
fi

ensure_ports_available
mkdir -p "$WORLDS_ROOT"
printf '%s\n' "$RUN_DIR" >"$RESULTS_ABS_ROOT/latest_run.txt"
write_run_metadata "$RUN_DIR/run.env"
write_run_config "$RUN_DIR/run_config.json"
write_session_metadata "$(session_meta_file)"
append_launcher_audit "run_metadata_written" "run_id=$RUN_ID launch_instance_id=$LAUNCH_INSTANCE_ID base_port=$BASE_PORT num_worlds=$NUM_WORLDS agents_per_world=$AGENTS_PER_WORLD"

echo "Starting $NUM_WORLDS world servers..."
tmux new-session -d -s "$TMUX_SESSION" -n "world-0"
tmux set-option -t "$TMUX_SESSION" remain-on-exit on
if [[ -n "${KRB5CCNAME:-}" ]]; then
  tmux set-environment -t "$TMUX_SESSION" KRB5CCNAME "$KRB5CCNAME"
fi
append_launcher_audit "tmux_session_created" "session=$TMUX_SESSION"

for ((i = 0; i < NUM_WORLDS; i++)); do
  port=$((BASE_PORT + i))
  world_root="$WORLDS_ROOT/world-$i"
  world_log_dir="$world_root/logs"
  world_log="$world_log_dir/world.log"
  world_record_dir="$world_root/recordings"
  world_agents_dir="$world_root/agents"
  world_resume_dir="$world_root/resume"
  world_resume_file="$world_resume_dir/latest.json"
  world_resume_arg=""
  mkdir -p "$world_log_dir" "$world_record_dir" "$world_agents_dir" "$world_resume_dir"
  if ((i > 0)); then
    tmux new-window -t "$TMUX_SESSION" -n "world-$i"
  fi
  if [[ -n "$RESUME_PATH" ]]; then
    world_resume_arg="--resume '$world_resume_file' "
  fi
  if [[ "$RECORD" == "true" ]]; then
    tmux send-keys -t "${TMUX_SESSION}:world-$i" \
      "cd '$ROOT_DIR' && $WORLD_SERVER_CMD --port $port 2>&1 | tee -a '$world_log'" Enter
  else
    tmux send-keys -t "${TMUX_SESSION}:world-$i" \
      "cd '$ROOT_DIR' && $WORLD_SERVER_CMD --port $port 2>&1 | tee -a '$world_log'" Enter
  fi
done

echo "Waiting for servers to become healthy..."
for ((i = 0; i < NUM_WORLDS; i++)); do
  port=$((BASE_PORT + i))
  printf "  world-%d (port %d)... " "$i" "$port"
  if wait_for_server "$port"; then
    cache_world_spectator_token "$i" >/dev/null 2>&1 || true
    echo "ready"
    append_launcher_audit "world_ready" "world_index=$i port=$port"
  else
    echo "TIMEOUT (${HEALTH_TIMEOUT}s)"
    append_launcher_audit "world_health_timeout" "world_index=$i port=$port health_timeout=$HEALTH_TIMEOUT"
  fi
done

DURATION_SECONDS=""
if [[ -n "$DURATION" ]]; then
  DURATION_SECONDS="$(parse_duration_seconds "$DURATION")"
fi

RESET_EVERY_SECONDS=""
if [[ -n "$RESET_EVERY" ]]; then
  RESET_EVERY_SECONDS="$(parse_duration_seconds "$RESET_EVERY")"
fi

echo "Launching $AGENTS_PER_WORLD Claude agents per world..."
for ((i = 0; i < NUM_WORLDS; i++)); do
  port=$((BASE_PORT + i))
  url="http://localhost:${port}"
  startup_instructions=""
  if [[ -n "$GOAL" ]]; then
    startup_instructions="$GOAL"
  fi
  tmux new-window -t "$TMUX_SESSION" -n "agents-$i"
  for ((j = 0; j < AGENTS_PER_WORLD; j++)); do
    pane_target="${TMUX_SESSION}:agents-$i"
    if ((j > 0)); then
      pane_target="$(tmux split-window -P -F '#{pane_id}' -t "${TMUX_SESSION}:agents-$i")"
      tmux select-layout -t "${TMUX_SESSION}:agents-$i" tiled
    else
      pane_target="$(tmux display-message -p -t "${TMUX_SESSION}:agents-$i" '#{pane_id}')"
    fi
    agent_display_name="$(agent_display_name_for_index "$j")"
    agent_name="$(sanitize_name "${agent_display_name}-r${RUN_SAFE_ID}-w${i}-a${j}")"
    world_agents_dir="$WORLDS_ROOT/world-$i/agents"
    agent_dir="$world_agents_dir/$agent_name"
    agent_log_dir="$agent_dir/logs"
    agent_runtime_dir="$agent_dir/runtime"
    agent_workspace="$agent_dir/workspace"
    agent_session_file="$agent_dir/claude_session_id.txt"
    agent_log_file="$agent_log_dir/agent.log"
    agent_world_session_file="$agent_dir/world_session.txt"
    mkdir -p "$agent_log_dir" "$agent_workspace" "$agent_runtime_dir"

    agent_template_abs_dir="$TEMPLATE_ABS_DIR"
    if ((j < ${#AGENT_TEMPLATE_ABS_DIRS[@]})); then
      agent_template_abs_dir="${AGENT_TEMPLATE_ABS_DIRS[$j]}"
    fi

    template_env_prefix=""
    if [[ -n "$agent_template_abs_dir" ]]; then
      template_env_prefix="TEMPLATE_DIR=$(printf '%q' "$agent_template_abs_dir") "
    fi
    if [[ -n "$SYSTEM_PROMPT_TEMPLATE" ]]; then
      template_env_prefix+="SYSTEM_PROMPT_TEMPLATE=$(printf '%q' "$SYSTEM_PROMPT_TEMPLATE") "
    fi
    if [[ "$SKIP_SOUL" == "1" ]]; then
      template_env_prefix+="SKIP_SOUL=1 "
    fi
    startup_env_prefix=""
    if [[ -n "$startup_instructions" ]]; then
      startup_instructions_q="$(printf '%q' "$startup_instructions")"
      startup_env_prefix="WORLD_STARTUP_INSTRUCTIONS=${startup_instructions_q} "
    fi

    auth_env_prefix="unset ANTHROPIC_API_KEY ANTHROPIC_OAUTH_TOKEN ANTHROPIC_AUTH_TOKEN; "
    if [[ "$CLAUDE_USE_ENV_AUTH" == "1" ]]; then
      if [[ -n "${CLAUDE_CODE_OAUTH_TOKEN:-}" ]]; then
        auth_env_prefix="unset ANTHROPIC_API_KEY ANTHROPIC_OAUTH_TOKEN ANTHROPIC_AUTH_TOKEN; "
      elif [[ -n "${ANTHROPIC_OAUTH_TOKEN:-}" ]]; then
        auth_env_prefix="unset ANTHROPIC_API_KEY ANTHROPIC_AUTH_TOKEN; ANTHROPIC_OAUTH_TOKEN=$(printf '%q' "$ANTHROPIC_OAUTH_TOKEN") "
      fi
    fi

    sandbox_env_prefix=""
    if [[ "$SANDBOX" == "1" ]]; then
      sandbox_env_prefix="SANDBOX=1 "
    fi
    agent_world_url="$url"
    agent_internal_world_url="$url"
    if world_capability_proxy_enabled; then
      proxy_public_host="world"
      proxy_pid="$(
        start_world_capability_proxy \
          "$agent_runtime_dir" \
          "$agent_log_dir" \
          "$url" \
          "$proxy_public_host" \
          "$agent_world_session_file"
      )"
      proxy_port="$(tr -d '[:space:]' < "$agent_runtime_dir/world_proxy_port.txt")"
      agent_world_url="http://${proxy_public_host}"
      sandbox_env_prefix+="CLAWBLOX_WORLD_HTTP_PROXY_PORT=$(printf '%q' "$proxy_port") "
      printf '%s\n' "$proxy_public_host" >"$agent_runtime_dir/world_proxy_public_host.txt"
      append_launcher_audit "world_capability_proxy_started" "agent_dir=$agent_dir pid=$proxy_pid public_host=$proxy_public_host listen_port=$proxy_port target=$url"
    fi

    pane_command_file="$agent_runtime_dir/launch_agent_pane.sh"
    write_agent_pane_command \
      "$pane_command_file" \
      "$ROOT_DIR" \
      "$auth_env_prefix" \
      "$agent_world_url" \
      "$agent_internal_world_url" \
      "$agent_display_name" \
      "$agent_dir" \
      "$agent_workspace" \
      "$agent_session_file" \
      "$agent_world_session_file" \
      "$RESET_EVERY_SECONDS" \
      "$DURATION_SECONDS" \
      "$CHECKPOINT_WARNING_SECONDS" \
      "$CHECKPOINT_WARNING_PROMPT" \
      "$CLAUDE_MODEL" \
      "$CLAUDE_PERMISSION_MODE" \
      "$CLAUDE_BARE" \
      "$CLAUDE_EXTRA_ARGS" \
      "$sandbox_env_prefix" \
      "$template_env_prefix" \
      "$startup_env_prefix" \
      "$CLAWBLOX_CLAUDE_BIN" \
      "$CLAWBLOX_CLAUDE_CODE_VERSION_PIN"
    printf '%s\n' "$pane_target" >"$agent_runtime_dir/pane_id.txt"
    printf '%s\n' "$pane_command_file" >"$agent_runtime_dir/pane_command_file.txt"

    agent_log_file_q="$(printf '%q' "$agent_log_file")"
    pane_command_file_q="$(printf '%q' "$pane_command_file")"
    tmux pipe-pane -o -t "$pane_target" "cat >> $agent_log_file_q"
    tmux send-keys -t "$pane_target" "bash $pane_command_file_q" Enter
    start_agent_recovery_watchdog "$pane_target" "$agent_dir" "$pane_command_file" "$agent_log_file"
  done
done
if [[ -n "$DURATION" ]]; then
  duration_secs="$(parse_duration_seconds "$DURATION")"
  # Agent gets duration + warning time to save workspace
  total_wait=$(( duration_secs + CHECKPOINT_WARNING_SECONDS + 30 ))
  echo "Run will auto-stop after ${DURATION} (${duration_secs}s + ${CHECKPOINT_WARNING_SECONDS}s warning)..."
  append_launcher_audit "auto_stop_scheduled" "duration=$DURATION duration_seconds=$duration_secs warning_seconds=$CHECKPOINT_WARNING_SECONDS total_wait=$total_wait launch_instance_id=$LAUNCH_INSTANCE_ID"
  (
    expected_run_id="$RUN_ID"
    expected_launch_instance_id="$LAUNCH_INSTANCE_ID"
    sleep "$total_wait"
    echo "Duration ${DURATION} + warning elapsed. Stopping..."
    CLAWBLOX_EXPECT_RUN_ID="$expected_run_id" \
      CLAWBLOX_EXPECT_LAUNCH_INSTANCE_ID="$expected_launch_instance_id" \
      CLAWBLOX_AUTO_TIMER_PID="$BASHPID" \
      CLAWBLOX_STOP_REASON="duration_elapsed" \
      CLAWBLOX_STOP_SOURCE="launch_multi_claude_auto_timer" \
      bash "$0" --stop --tmux-session "$TMUX_SESSION"
  ) &
  AUTO_STOP_PID="$!"
  write_session_metadata "$(session_meta_file)"
  append_launcher_audit "auto_stop_pid_recorded" "auto_stop_pid=$AUTO_STOP_PID launch_instance_id=$LAUNCH_INSTANCE_ID"
  disown
fi

echo "tmux attach -t $TMUX_SESSION"
