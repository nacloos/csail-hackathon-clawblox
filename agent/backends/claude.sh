#!/usr/bin/env bash

agent_backend_name() {
  printf 'claude\n'
}

agent_set_defaults() {
  CLAUDE_MODEL="${CLAUDE_MODEL:-${AGENT_MODEL:-claude-opus-4-6}}"
  CLAUDE_PERMISSION_MODE="${CLAUDE_PERMISSION_MODE:-bypassPermissions}"
  CLAUDE_BARE="${CLAUDE_BARE:-0}"
  CLAUDE_EXTRA_ARGS="${CLAUDE_EXTRA_ARGS:-}"
  CLAUDE_USE_ENV_AUTH="${CLAUDE_USE_ENV_AUTH:-0}"
  CLAWBLOX_CLAUDE_BIN="${CLAWBLOX_CLAUDE_BIN:-}"
  CLAWBLOX_CLAUDE_CODE_VERSION_PIN="${CLAWBLOX_CLAUDE_CODE_VERSION_PIN:-2.1.116}"
  CLAUDE_FAILURE_IDLE_GRACE_SECONDS="${CLAUDE_FAILURE_IDLE_GRACE_SECONDS:-180}"
  CLAUDE_FAILURE_STALL_GRACE_SECONDS="${CLAUDE_FAILURE_STALL_GRACE_SECONDS:-300}"
  CLAUDE_FAILURE_RESTART_BACKOFF_SECONDS="${CLAUDE_FAILURE_RESTART_BACKOFF_SECONDS:-5}"
  AGENT_RUNNER_SCRIPT="${AGENT_RUNNER_SCRIPT:-$SCRIPT_DIR/run_claude_agent.sh}"
  AGENT_WATCHDOG_SCRIPT="${AGENT_WATCHDOG_SCRIPT:-$SCRIPT_DIR/watch_claude_recovery.sh}"

  _claude_load_oauth_token_env

  if [[ ! -f "$AGENT_RUNNER_SCRIPT" ]]; then
    echo "Error: Claude agent runner not found at $AGENT_RUNNER_SCRIPT" >&2
    exit 1
  fi
}

agent_parse_arg() {
  local opt="$1"
  local maybe_value="${2:-}"
  AGENT_PARSE_CONSUMED=0

  case "$opt" in
    --model)
      [[ -n "$maybe_value" ]] || return 2
      AGENT_MODEL="$maybe_value"
      CLAUDE_MODEL="$maybe_value"
      AGENT_PARSE_CONSUMED=2
      ;;
    --permission-mode)
      [[ -n "$maybe_value" ]] || return 2
      CLAUDE_PERMISSION_MODE="$maybe_value"
      AGENT_PARSE_CONSUMED=2
      ;;
    --claude-extra-args)
      [[ -n "$maybe_value" ]] || return 2
      CLAUDE_EXTRA_ARGS="$maybe_value"
      AGENT_PARSE_CONSUMED=2
      ;;
    --use-env-auth)
      CLAUDE_USE_ENV_AUTH=1
      AGENT_PARSE_CONSUMED=1
      ;;
    --skip-soul|--no-soul)
      SKIP_SOUL=1
      AGENT_PARSE_CONSUMED=1
      ;;
    *)
      return 1
      ;;
  esac
}

agent_validate() {
  _claude_require_auth
  _claude_require_sandbox_runtime
}

agent_start_command() {
  local agent_dir="$1"
  local context_file="$2"
  local command_file="$3"
  local plugin_dir

  if plugin_dir="$(_claude_prepare_agent_plugin "$agent_dir")"; then
    :
  else
    return 1
  fi

  _claude_write_agent_command "$agent_dir" "$context_file" "$command_file" "$plugin_dir"
}

agent_after_start() {
  local pane_id="$1"
  local agent_dir="$2"
  local command_file="$3"
  local agent_log_file="$4"

  _claude_start_recovery_watchdog "$pane_id" "$agent_dir" "$command_file" "$agent_log_file"
}

agent_message_send() {
  local agent_dir="$1"
  local message="$2"
  local pane_id_file="$agent_dir/runtime/pane_id.txt"
  local pane_id

  if [[ ! -f "$pane_id_file" ]]; then
    echo "Error: pane id file not found at $pane_id_file" >&2
    return 1
  fi
  pane_id="$(tr -d '[:space:]' <"$pane_id_file")"
  tmux send-keys -t "$pane_id" "$message" Enter
}

agent_session_ref() {
  local agent_dir="$1"
  local session_file="$agent_dir/claude_session_id.txt"

  [[ -f "$session_file" ]] || return 1
  tr -d '[:space:]' <"$session_file"
}

agent_write_run_metadata() {
  local target="$1"
  cat >>"$target" <<EOF
CLAUDE_MODEL=$(printf '%q' "$CLAUDE_MODEL")
CLAUDE_PERMISSION_MODE=$(printf '%q' "$CLAUDE_PERMISSION_MODE")
CLAUDE_BARE=$(printf '%q' "$CLAUDE_BARE")
CLAUDE_EXTRA_ARGS=$(printf '%q' "$CLAUDE_EXTRA_ARGS")
CLAUDE_USE_ENV_AUTH=$(printf '%q' "$CLAUDE_USE_ENV_AUTH")
CLAWBLOX_CLAUDE_BIN=$(printf '%q' "$CLAWBLOX_CLAUDE_BIN")
CLAWBLOX_CLAUDE_CODE_VERSION_PIN=$(printf '%q' "$CLAWBLOX_CLAUDE_CODE_VERSION_PIN")
EOF
}

agent_hook_set() {
  local agent_dir="$1"
  local hook_name="$2"
  local hook_command="$3"
  local hooks_dir="$agent_dir/runtime/agent-hooks"

  agent_hook_validate "$hook_name" || return 1

  mkdir -p "$hooks_dir"
  printf '%s\n' "$hook_command" >"$hooks_dir/${hook_name}.cmd"
}

agent_hook_validate() {
  local hook_name="$1"

  case "$hook_name" in
    session_start|session_end|user_message|user_prompt_submit|stop|turn_end|stop_failure|notification)
      return 0
      ;;
    *)
      echo "Error: unsupported Claude hook '$hook_name'." >&2
      echo "Supported hooks: session_start, session_end, user_message, user_prompt_submit, stop, turn_end, stop_failure, notification" >&2
      return 1
      ;;
  esac
}

_claude_load_oauth_token_env() {
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

_claude_require_auth() {
  local claude_bin=""

  if [[ -n "$CLAWBLOX_CLAUDE_BIN" ]]; then
    if [[ -x "$CLAWBLOX_CLAUDE_BIN" ]]; then
      claude_bin="$CLAWBLOX_CLAUDE_BIN"
    else
      echo "Error: CLAWBLOX_CLAUDE_BIN is not executable: $CLAWBLOX_CLAUDE_BIN" >&2
      return 1
    fi
  elif [[ -n "$CLAWBLOX_CLAUDE_CODE_VERSION_PIN" ]]; then
    local version_bin="$HOME/.local/share/claude/versions/$CLAWBLOX_CLAUDE_CODE_VERSION_PIN"
    if [[ -x "$version_bin" ]]; then
      claude_bin="$version_bin"
    fi
  fi

  if [[ -z "$claude_bin" ]]; then
    if ! claude_bin="$(command -v claude 2>/dev/null)"; then
      echo "Error: claude CLI was not found in PATH." >&2
      return 1
    fi
  fi

  if ! "$claude_bin" auth status --text >/dev/null 2>&1; then
    echo "Error: Claude Code is not logged in for $claude_bin. Run: $claude_bin auth login" >&2
    return 1
  fi
}

_claude_require_sandbox_runtime() {
  local native_sandbox="${CLAUDE_NATIVE_SANDBOX:-$SANDBOX}"
  if [[ "$SANDBOX" != "1" ]]; then
    return 0
  fi
  if ! command -v bwrap >/dev/null 2>&1; then
    echo "Error: --sandbox requires bubblewrap (bwrap)." >&2
    return 1
  fi
  if [[ "$native_sandbox" == "1" ]] && ! command -v socat >/dev/null 2>&1; then
    echo "Error: --sandbox requires socat for Claude Code's native Linux sandbox." >&2
    echo "Install socat, or set CLAUDE_NATIVE_SANDBOX=0 only for non-secure local debugging." >&2
    return 1
  fi
}

_claude_write_agent_command() {
  local agent_dir="$1"
  local context_file="$2"
  local target="$3"
  local plugin_dir="$4"
  local auth_env_prefix="unset ANTHROPIC_API_KEY ANTHROPIC_OAUTH_TOKEN ANTHROPIC_AUTH_TOKEN; "
  local plugin_env_prefix=""

  if [[ "$CLAUDE_USE_ENV_AUTH" == "1" ]]; then
    if [[ -n "${CLAUDE_CODE_OAUTH_TOKEN:-}" ]]; then
      auth_env_prefix="unset ANTHROPIC_API_KEY ANTHROPIC_OAUTH_TOKEN ANTHROPIC_AUTH_TOKEN; "
    elif [[ -n "${ANTHROPIC_OAUTH_TOKEN:-}" ]]; then
      auth_env_prefix="unset ANTHROPIC_API_KEY ANTHROPIC_AUTH_TOKEN; ANTHROPIC_OAUTH_TOKEN=$(printf '%q' "$ANTHROPIC_OAUTH_TOKEN") "
    fi
  fi
  if [[ -n "$plugin_dir" ]]; then
    plugin_env_prefix="PLUGIN_DIR=$(printf '%q' "$plugin_dir") "
  fi

  cat >"$target" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd $(printf '%q' "$ROOT_DIR")
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
  cat >>"$target" <<EOF
# shellcheck disable=SC1090
source $(printf '%q' "$context_file")
${auth_env_prefix}WORLD_BASE_URL="\$AGENT_WORLD_BASE_URL" WORLD_INTERNAL_BASE_URL="\$AGENT_WORLD_INTERNAL_BASE_URL" WORLD_AGENT_NAME="\$AGENT_DISPLAY_NAME" WORLD_SOURCE_DIR="\$AGENT_WORLD_SOURCE_DIR" AGENT_DIR="\$AGENT_DIR" WORKSPACE_DIR="\$AGENT_WORKSPACE_DIR" SESSION_ID_FILE="\$AGENT_DIR/claude_session_id.txt" WORLD_SESSION_FILE="\$AGENT_WORLD_SESSION_FILE" RESET_EVERY="\$AGENT_RESET_EVERY_SECONDS" DURATION_SECONDS="\$AGENT_DURATION_SECONDS" CHECKPOINT_WARNING_SECONDS="\$AGENT_CHECKPOINT_WARNING_SECONDS" CHECKPOINT_WARNING_PROMPT="\$AGENT_CHECKPOINT_WARNING_PROMPT" CLAUDE_MODEL=$(printf '%q' "$CLAUDE_MODEL") CLAUDE_PERMISSION_MODE=$(printf '%q' "$CLAUDE_PERMISSION_MODE") CLAUDE_BARE=$(printf '%q' "$CLAUDE_BARE") CLAUDE_EXTRA_ARGS=$(printf '%q' "$CLAUDE_EXTRA_ARGS") CLAWBLOX_CLAUDE_BIN=$(printf '%q' "$CLAWBLOX_CLAUDE_BIN") CLAWBLOX_CLAUDE_CODE_VERSION_PIN=$(printf '%q' "$CLAWBLOX_CLAUDE_CODE_VERSION_PIN") SANDBOX="\$AGENT_SANDBOX" CLAWBLOX_WORLD_HTTP_PROXY_PORT="\$AGENT_WORLD_HTTP_PROXY_PORT" TEMPLATE_DIR="\$AGENT_TEMPLATE_DIR" SYSTEM_PROMPT_TEMPLATE="\$AGENT_SYSTEM_PROMPT_TEMPLATE" SKIP_SOUL="\$AGENT_SKIP_SOUL" WORLD_STARTUP_INSTRUCTIONS="\$AGENT_STARTUP_INSTRUCTIONS" ${plugin_env_prefix}bash $(printf '%q' "$AGENT_RUNNER_SCRIPT")
EOF
  chmod +x "$target"
}

_claude_prepare_agent_plugin() {
  local agent_dir="$1"
  local hooks_dir="$agent_dir/runtime/agent-hooks"
  local plugin_dir="$agent_dir/runtime/claude-plugin"
  local dispatcher="$plugin_dir/bin/agent_hook_dispatch.sh"
  local base_plugin="$SCRIPT_DIR/claude-plugin"

  [[ -d "$hooks_dir" ]] || return 0

  if [[ ! -d "$base_plugin" ]]; then
    echo "Error: Claude plugin directory not found at $base_plugin" >&2
    return 1
  fi

  rm -rf "$plugin_dir"
  cp -R "$base_plugin" "$plugin_dir"
  mkdir -p "$plugin_dir/agent-hooks"
  cp "$hooks_dir"/*.cmd "$plugin_dir/agent-hooks/"

  cat >"$dispatcher" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

hook_name="${1:-}"
plugin_root="${CLAUDE_PLUGIN_ROOT:-}"
command_file="$plugin_root/agent-hooks/${hook_name}.cmd"

if [[ -z "$hook_name" || ! -f "$command_file" ]]; then
  exit 0
fi

input="$(cat)"
hook_command="$(cat "$command_file")"
AGENT_HOOK_NAME="$hook_name" bash -c "$hook_command" <<<"$input"
EOF
  chmod +x "$dispatcher"

  python3 - "$plugin_dir/hooks/hooks.json" "$hooks_dir" <<'PY'
import json
import os
import sys

hooks_path, hooks_dir = sys.argv[1], sys.argv[2]
with open(hooks_path, encoding="utf-8") as f:
    data = json.load(f)

event_map = {
    "session_start": "SessionStart",
    "session_end": "SessionEnd",
    "user_message": "UserPromptSubmit",
    "user_prompt_submit": "UserPromptSubmit",
    "stop": "Stop",
    "turn_end": "Stop",
    "stop_failure": "StopFailure",
    "notification": "Notification",
}

hooks = data.setdefault("hooks", {})
for name in sorted(event_map):
    command_file = os.path.join(hooks_dir, f"{name}.cmd")
    if not os.path.exists(command_file):
        continue
    event = event_map[name]
    entry = {
        "hooks": [
            {
                "type": "command",
                "command": f"\"${{CLAUDE_PLUGIN_ROOT}}/bin/agent_hook_dispatch.sh\" {name}",
            }
        ]
    }
    hooks.setdefault(event, []).append(entry)

with open(hooks_path, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, sort_keys=True)
    f.write("\n")
PY

  printf '%s\n' "$plugin_dir"
}

_claude_start_recovery_watchdog() {
  local pane_id="$1"
  local agent_dir="$2"
  local command_file="$3"
  local agent_log_file="$4"
  local runtime_dir pid_file existing_pid watcher_pid

  [[ -x "$AGENT_WATCHDOG_SCRIPT" ]] || return 0

  runtime_dir="$agent_dir/runtime"
  pid_file="$runtime_dir/recovery_watchdog.pid"
  while IFS= read -r existing_pid; do
    [[ -n "$existing_pid" ]] || continue
    if kill -0 "$existing_pid" 2>/dev/null; then
      kill "$existing_pid" 2>/dev/null || true
      append_launcher_audit "watchdog_cleanup_killed" "agent_dir=$agent_dir watchdog_pid=$existing_pid"
    fi
  done < <(
    ps -eo pid=,args= | awk -v script="$AGENT_WATCHDOG_SCRIPT" -v agent_dir="$agent_dir" '
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
    nohup bash "$AGENT_WATCHDOG_SCRIPT" \
      --pane-id "$pane_id" \
      --agent-dir "$agent_dir" \
      --command-file "$command_file" \
      --agent-log-file "$agent_log_file" \
      >/dev/null 2>&1 &
  watcher_pid="$!"
  printf '%s\n' "$watcher_pid" >"$pid_file"
  append_launcher_audit "watchdog_started" "pane_id=$pane_id agent_dir=$agent_dir watchdog_pid=$watcher_pid"
}

agent_write_run_config() {
  :
}
