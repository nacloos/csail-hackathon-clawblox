#!/usr/bin/env bash
set -euo pipefail

PANE_ID=""
AGENT_DIR=""
COMMAND_FILE=""
AGENT_LOG_FILE=""
IDLE_GRACE_SECONDS="${CLAUDE_FAILURE_IDLE_GRACE_SECONDS:-180}"
STALL_GRACE_SECONDS="${CLAUDE_FAILURE_STALL_GRACE_SECONDS:-300}"
RESTART_BACKOFF_SECONDS="${CLAUDE_FAILURE_RESTART_BACKOFF_SECONDS:-5}"
POLL_SECONDS="${CLAUDE_FAILURE_POLL_SECONDS:-2}"

usage() {
  cat <<'EOF'
Usage: watch_claude_recovery.sh --pane-id PANE --agent-dir DIR --command-file FILE --agent-log-file FILE [options]

Options:
  --idle-grace-seconds N     Wait after idle_prompt before respawn (default: 180)
  --stall-grace-seconds N    Wait after stop_failure with no progress (default: 300)
  --restart-backoff-seconds N  Sleep after respawn before resuming watch (default: 5)
  --poll-seconds N           Poll interval (default: 2)
EOF
}

while (($# > 0)); do
  case "$1" in
    --pane-id) PANE_ID="$2"; shift 2 ;;
    --agent-dir) AGENT_DIR="$2"; shift 2 ;;
    --command-file) COMMAND_FILE="$2"; shift 2 ;;
    --agent-log-file) AGENT_LOG_FILE="$2"; shift 2 ;;
    --idle-grace-seconds) IDLE_GRACE_SECONDS="$2"; shift 2 ;;
    --stall-grace-seconds) STALL_GRACE_SECONDS="$2"; shift 2 ;;
    --restart-backoff-seconds) RESTART_BACKOFF_SECONDS="$2"; shift 2 ;;
    --poll-seconds) POLL_SECONDS="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 1 ;;
  esac
done

if [[ -z "$PANE_ID" || -z "$AGENT_DIR" || -z "$COMMAND_FILE" || -z "$AGENT_LOG_FILE" ]]; then
  usage >&2
  exit 1
fi

LOG_DIR="$AGENT_DIR/logs"
RUNTIME_DIR="$AGENT_DIR/runtime"
HOOK_METADATA_FILE="$RUNTIME_DIR/hook_session.env"
SESSION_ID_FILE="$AGENT_DIR/claude_session_id.txt"
RECOVERY_LOG_FILE="$RUNTIME_DIR/recovery.log"
SANDBOX_HOME_DIR="$AGENT_DIR/sandbox-home"
WATCH_STARTED_EPOCH="$(date +%s)"

utc_now() {
  date -u +%Y-%m-%dT%H:%M:%SZ
}

log_event() {
  local event="$1"
  local details="${2:-}"
  printf '%s event=%s pid=%s pane_id=%q details=%q\n' \
    "$(utc_now)" \
    "$event" \
    "$$" \
    "$PANE_ID" \
    "$details" >>"$RECOVERY_LOG_FILE"
}

log_event "watch_start" "agent_dir=$AGENT_DIR command_file=$COMMAND_FILE idle_grace_seconds=$IDLE_GRACE_SECONDS stall_grace_seconds=$STALL_GRACE_SECONDS restart_backoff_seconds=$RESTART_BACKOFF_SECONDS poll_seconds=$POLL_SECONDS"

file_mtime_epoch() {
  local path="$1"
  if [[ ! -e "$path" ]]; then
    printf '0\n'
    return 0
  fi
  stat -c %Y "$path" 2>/dev/null || printf '0\n'
}

map_runtime_path() {
  local src="${1:-}"
  if [[ -z "$src" ]]; then
    return 0
  fi
  if [[ "$src" == /home/agent/* ]]; then
    printf '%s\n' "${SANDBOX_HOME_DIR}${src#/home/agent}"
    return 0
  fi
  printf '%s\n' "$src"
}

pane_exists() {
  tmux display-message -p -t "$PANE_ID" '#{pane_id}' >/dev/null 2>&1
}

current_session_id() {
  local SESSION_ID="" TRANSCRIPT_PATH=""
  if [[ -f "$HOOK_METADATA_FILE" ]]; then
    # shellcheck disable=SC1090
    source "$HOOK_METADATA_FILE"
    if [[ -n "${SESSION_ID:-}" ]]; then
      printf '%s\n' "$SESSION_ID"
      return 0
    fi
  fi
  if [[ -f "$SESSION_ID_FILE" ]]; then
    tr -d '[:space:]' <"$SESSION_ID_FILE"
    return 0
  fi
  return 1
}

session_transcript_path() {
  local session_id="$1"
  local session_start_file="$LOG_DIR/session_start.${session_id}.json"
  local src=""

  if [[ -f "$session_start_file" ]]; then
    src="$(
      INPUT_JSON="$(cat "$session_start_file")" node <<'EOF'
const data = JSON.parse(process.env.INPUT_JSON ?? "{}");
process.stdout.write(String(data.transcript_path ?? ""));
EOF
    )"
  elif [[ -f "$HOOK_METADATA_FILE" ]]; then
    local SESSION_ID="" TRANSCRIPT_PATH=""
    # shellcheck disable=SC1090
    source "$HOOK_METADATA_FILE"
    if [[ "${SESSION_ID:-}" == "$session_id" ]]; then
      src="${TRANSCRIPT_PATH:-}"
    fi
  fi

  map_runtime_path "$src"
}

stop_failure_fields() {
  local path="$1"
  INPUT_JSON="$(cat "$path")" node <<'EOF'
const data = JSON.parse(process.env.INPUT_JSON ?? "{}");
const fields = [
  data.error ?? "",
  data.transcript_path ?? "",
];
for (const value of fields) process.stdout.write(String(value) + "\n");
EOF
}

session_start_epoch_for_session() {
  local session_id="$1"
  local session_start_file="$LOG_DIR/session_start.${session_id}.json"
  file_mtime_epoch "$session_start_file"
}

session_end_epoch_for_session() {
  local session_id="$1"
  local session_end_file="$LOG_DIR/session_end.${session_id}.json"
  file_mtime_epoch "$session_end_file"
}

respawn_agent() {
  local reason="$1"
  local session_id="$2"
  local stop_failure_epoch="$3"
  local respawn_cmd

  respawn_cmd="bash $(printf '%q' "$COMMAND_FILE")"
  log_event "respawn_start" "reason=$reason session_id=$session_id stop_failure_epoch=$stop_failure_epoch"
  tmux respawn-pane -k -t "$PANE_ID" "$respawn_cmd"
  tmux pipe-pane -o -t "$PANE_ID" "cat >> $(printf '%q' "$AGENT_LOG_FILE")"
  sleep "$RESTART_BACKOFF_SECONDS"
}

last_handled_key=""

while pane_exists; do
  session_id="$(current_session_id || true)"
  if [[ -z "$session_id" ]]; then
    sleep "$POLL_SECONDS"
    continue
  fi

  stop_failure_file="$LOG_DIR/stop_failure.${session_id}.json"
  if [[ ! -f "$stop_failure_file" ]]; then
    sleep "$POLL_SECONDS"
    continue
  fi

  stop_failure_epoch="$(file_mtime_epoch "$stop_failure_file")"
  session_start_epoch="$(session_start_epoch_for_session "$session_id")"
  if (( stop_failure_epoch < WATCH_STARTED_EPOCH || stop_failure_epoch < session_start_epoch )); then
    sleep "$POLL_SECONDS"
    continue
  fi

  current_key="${session_id}:${stop_failure_epoch}"
  if [[ "$current_key" == "$last_handled_key" ]]; then
    sleep "$POLL_SECONDS"
    continue
  fi

  mapfile -t failure_fields < <(stop_failure_fields "$stop_failure_file" || true)
  failure_error="${failure_fields[0]:-unknown}"
  transcript_path="$(map_runtime_path "${failure_fields[1]:-}")"
  if [[ -z "$transcript_path" ]]; then
    transcript_path="$(session_transcript_path "$session_id")"
  fi

  idle_prompt_file="$LOG_DIR/notification.idle_prompt.${session_id}.json"
  idle_prompt_epoch="$(file_mtime_epoch "$idle_prompt_file")"
  session_end_epoch="$(session_end_epoch_for_session "$session_id")"
  progress_epoch=0
  transcript_epoch=0
  if [[ -n "$transcript_path" && -f "$transcript_path" ]]; then
    transcript_epoch="$(file_mtime_epoch "$transcript_path")"
  fi
  (( transcript_epoch > progress_epoch )) && progress_epoch="$transcript_epoch"
  if (( progress_epoch == 0 )); then
    agent_log_epoch="$(file_mtime_epoch "$AGENT_LOG_FILE")"
    (( agent_log_epoch > progress_epoch )) && progress_epoch="$agent_log_epoch"
  fi
  now_epoch="$(date +%s)"

  if (( session_end_epoch >= stop_failure_epoch )); then
    last_handled_key="$current_key"
    log_event "recover_session_end" "session_id=$session_id error=$failure_error stop_failure_epoch=$stop_failure_epoch session_end_epoch=$session_end_epoch"
    respawn_agent "stop_failure_session_end" "$session_id" "$stop_failure_epoch"
    continue
  fi

  if (( idle_prompt_epoch >= stop_failure_epoch && now_epoch - idle_prompt_epoch >= IDLE_GRACE_SECONDS )); then
    last_handled_key="$current_key"
    log_event "recover_idle_prompt" "session_id=$session_id error=$failure_error stop_failure_epoch=$stop_failure_epoch idle_prompt_epoch=$idle_prompt_epoch"
    respawn_agent "stop_failure_idle_prompt" "$session_id" "$stop_failure_epoch"
    continue
  fi

  if (( now_epoch - stop_failure_epoch >= STALL_GRACE_SECONDS && progress_epoch <= stop_failure_epoch )); then
    last_handled_key="$current_key"
    log_event "recover_stalled" "session_id=$session_id error=$failure_error stop_failure_epoch=$stop_failure_epoch progress_epoch=$progress_epoch"
    respawn_agent "stop_failure_stalled" "$session_id" "$stop_failure_epoch"
    continue
  fi

  sleep "$POLL_SECONDS"
done

log_event "watch_exit" "reason=pane_missing"
