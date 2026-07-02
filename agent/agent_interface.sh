#!/usr/bin/env bash

agent_load() {
  local backend="$1"
  local backend_file

  case "$backend" in
    claude)
      backend_file="$SCRIPT_DIR/backends/claude.sh"
      ;;
    *)
      echo "Error: unsupported agent backend '$backend'." >&2
      return 1
      ;;
  esac

  if [[ ! -f "$backend_file" ]]; then
    echo "Error: agent backend not found at $backend_file" >&2
    return 1
  fi

  # shellcheck disable=SC1090
  source "$backend_file"
}

agent_validate() {
  echo "Error: no agent backend loaded." >&2
  return 1
}

agent_set_defaults() {
  :
}

agent_parse_arg() {
  AGENT_PARSE_CONSUMED=0
  return 1
}

agent_write_run_metadata() {
  :
}

agent_start_command() {
  echo "Error: no agent backend loaded." >&2
  return 1
}

agent_message_send() {
  echo "Error: no agent backend loaded." >&2
  return 1
}

agent_session_stop() {
  :
}

agent_session_ref() {
  :
}

agent_hook_set() {
  :
}

agent_hook_validate() {
  :
}

agent_artifacts_export() {
  :
}

agent_after_start() {
  :
}

agent_write_run_config() {
  :
}
