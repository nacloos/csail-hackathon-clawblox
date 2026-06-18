#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/krb-longrun run [--name NAME] [--check-minutes N] -- COMMAND [ARGS...]
  scripts/krb-longrun status [--name NAME]
  scripts/krb-longrun stop [--name NAME]

Run a long command with an explicit renewable Kerberos cache and a background
krenew process. This avoids stale KRB5CCNAME values inherited by old tmux
servers.

Environment:
  KRB_LONGRUN_DIR       Directory for cache and pid files.
  KRB_LONGRUN_NAME      Default name when --name is omitted.
  KRB_LONGRUN_CHECK_MINUTES
                        Default krenew check interval, in minutes.

Examples:
  scripts/krb-longrun run --name mesa -- bash worlds/mesa-world/launch_multi_generations_claude.sh ...
  scripts/krb-longrun status --name mesa
  scripts/krb-longrun stop --name mesa
EOF
}

die() {
  echo "Error: $*" >&2
  exit 1
}

warn() {
  echo "Warning: $*" >&2
}

info() {
  echo "Info: $*" >&2
}

RENEWER_PID=""
CHILD_PID=""
CHILD_STOP_REQUESTED=0
CHILD_SIGNAL_GRACE_SECONDS="${KRB_LONGRUN_SIGNAL_GRACE_SECONDS:-5}"
CHILD_KILL_GRACE_SECONDS="${KRB_LONGRUN_KILL_GRACE_SECONDS:-120}"

sanitize_name() {
  local value="$1"
  value="$(printf '%s' "$value" | tr -c 'A-Za-z0-9._-' '_')"
  value="${value##_}"
  value="${value%%_}"
  [[ -n "$value" ]] || value="longrun"
  printf '%s\n' "$value"
}

default_state_dir() {
  if [[ -n "${KRB_LONGRUN_DIR:-}" ]]; then
    printf '%s\n' "$KRB_LONGRUN_DIR"
  elif [[ -d "/storage/${USER:-}" ]]; then
    printf '%s\n' "/storage/${USER}/.krb5/longrun"
  elif [[ -n "${HOME:-}" && -d "$HOME" ]]; then
    printf '%s\n' "$HOME/.krb5/longrun"
  else
    printf '%s\n' "/tmp/krb5cc_${UID}_longrun_state"
  fi
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || die "required command not found: $1"
}

parse_common_opts() {
  NAME="$(sanitize_name "${KRB_LONGRUN_NAME:-longrun}")"
  CHECK_MINUTES="${KRB_LONGRUN_CHECK_MINUTES:-60}"
  STARTUP_VALIDATION_SECONDS="${KRB_LONGRUN_STARTUP_VALIDATION_SECONDS:-30}"
  CHILD_SIGNAL_GRACE_SECONDS="${KRB_LONGRUN_SIGNAL_GRACE_SECONDS:-5}"
  CHILD_KILL_GRACE_SECONDS="${KRB_LONGRUN_KILL_GRACE_SECONDS:-120}"

  while [[ $# -gt 0 ]]; do
    case "$1" in
      --name)
        [[ $# -ge 2 ]] || die "--name requires a value"
        NAME="$(sanitize_name "$2")"
        shift 2
        ;;
      --check-minutes)
        [[ $# -ge 2 ]] || die "--check-minutes requires a value"
        CHECK_MINUTES="$2"
        shift 2
        ;;
      --help|-h)
        usage
        exit 0
        ;;
      --)
        shift
        break
        ;;
      -*)
        die "unknown option: $1"
        ;;
      *)
        break
        ;;
    esac
  done

  [[ "$CHECK_MINUTES" =~ ^[1-9][0-9]*$ ]] || die "--check-minutes must be a positive integer"
  [[ "$STARTUP_VALIDATION_SECONDS" =~ ^[1-9][0-9]*$ ]] || die "KRB_LONGRUN_STARTUP_VALIDATION_SECONDS must be a positive integer"
  [[ "$CHILD_SIGNAL_GRACE_SECONDS" =~ ^[1-9][0-9]*$ ]] || die "KRB_LONGRUN_SIGNAL_GRACE_SECONDS must be a positive integer"
  [[ "$CHILD_KILL_GRACE_SECONDS" =~ ^[1-9][0-9]*$ ]] || die "KRB_LONGRUN_KILL_GRACE_SECONDS must be a positive integer"

  STATE_DIR="$(default_state_dir)"
  CACHE_FILE="$STATE_DIR/krb5cc_${UID}_${NAME}"
  PID_FILE="$STATE_DIR/krenew_${UID}_${NAME}.pid"
  LOG_FILE="$STATE_DIR/krenew_${UID}_${NAME}.log"
  export KRB5CCNAME="FILE:$CACHE_FILE"

  REMAINING_ARGS=("$@")
}

cache_ok() {
  [[ -f "$CACHE_FILE" ]] || return 1
  KRB5CCNAME="FILE:$CACHE_FILE" klist -s >/dev/null 2>&1
}

read_pid_file() {
  local pid_file="$1"
  [[ -s "$pid_file" ]] || return 1
  local pid
  pid="$(tr -d '[:space:]' <"$pid_file")"
  [[ -n "$pid" ]] || return 1
  printf '%s\n' "$pid"
}

log_tail() {
  local path="$1"
  local lines="${2:-40}"
  [[ -f "$path" ]] || return 0
  tail -n "$lines" "$path" 2>/dev/null || true
}

diagnose_renewer_failure() {
  local reason="$1"
  local pid="${2:-}"
  echo "krenew startup diagnostics:" >&2
  echo "  reason:      $reason" >&2
  echo "  cache:       FILE:$CACHE_FILE" >&2
  echo "  pid file:    $PID_FILE" >&2
  echo "  renew log:   $LOG_FILE" >&2
  if [[ -n "$pid" ]]; then
    echo "  renewer pid: $pid" >&2
  fi
  echo >&2
  echo "Kerberos cache details:" >&2
  KRB5CCNAME="FILE:$CACHE_FILE" klist -f >&2 || true
  echo >&2
  echo "Recent krenew log output:" >&2
  if ! log_tail "$LOG_FILE" 60 >&2; then
    true
  fi
}

preflight_renewal() {
  echo "Preflighting renewal and AFS access..."

  if ! KRB5CCNAME="FILE:$CACHE_FILE" krenew -H 120 -k "$CACHE_FILE" >/dev/null 2>&1; then
    die "cache is not healthy for at least 120 more minutes: FILE:$CACHE_FILE"
  fi

  if ! KRB5CCNAME="FILE:$CACHE_FILE" aklog >/dev/null 2>&1; then
    die "aklog failed for FILE:$CACHE_FILE; AFS access is not viable"
  fi
}

print_paths() {
  echo "Kerberos longrun:"
  echo "  name:        $NAME"
  echo "  cache:       FILE:$CACHE_FILE"
  echo "  pid file:    $PID_FILE"
  echo "  renew log:   $LOG_FILE"
}

stop_existing_renewer() {
  if [[ ! -s "$PID_FILE" ]]; then
    return 0
  fi

  local pid
  pid="$(tr -d '[:space:]' <"$PID_FILE")"
  if [[ -z "$pid" ]]; then
    rm -f "$PID_FILE"
    return 0
  fi
  if kill -0 "$pid" >/dev/null 2>&1; then
    echo "Stopping existing krenew pid $pid"
    kill "$pid" >/dev/null 2>&1 || true
    sleep 1
  fi
  rm -f "$PID_FILE"
}

start_renewer() {
  : >"$LOG_FILE"
  rm -f "$PID_FILE"
  if ! krenew -a -b -t -v -x -K "$CHECK_MINUTES" -k "$CACHE_FILE" -p "$PID_FILE" >>"$LOG_FILE" 2>&1; then
    diagnose_renewer_failure "krenew failed before daemonizing" ""
    die "krenew could not start a renewal daemon"
  fi

  local pid=""
  local pid_deadline=$((SECONDS + 8))
  local validation_deadline=$((SECONDS + STARTUP_VALIDATION_SECONDS))
  local saw_success=0

  echo "Validating first krenew renewal for up to ${STARTUP_VALIDATION_SECONDS}s..."

  while ((SECONDS < pid_deadline)); do
    if pid="$(read_pid_file "$PID_FILE" 2>/dev/null)"; then
      break
    fi
    sleep 1
  done

  if [[ -z "$pid" ]]; then
    diagnose_renewer_failure "pid file was not written" ""
    die "krenew did not write pid file: $PID_FILE"
  fi

  while ((SECONDS < validation_deadline)); do
    if ! kill -0 "$pid" >/dev/null 2>&1; then
      diagnose_renewer_failure "process exited during startup validation" "$pid"
      die "krenew pid $pid is not running"
    fi
    if grep -q "error renewing credentials" "$LOG_FILE"; then
      diagnose_renewer_failure "renewal failed during first daemon renewal attempt" "$pid"
      die "krenew could not renew credentials"
    fi
    if grep -Eq "aklog exited with status [1-9][0-9]*" "$LOG_FILE"; then
      diagnose_renewer_failure "aklog failed during first daemon renewal attempt" "$pid"
      die "krenew renewed Kerberos but failed to refresh AFS tokens"
    fi
    if grep -q "aklog exited with status 0" "$LOG_FILE"; then
      saw_success=1
      break
    fi
    sleep 1
  done

  if ! kill -0 "$pid" >/dev/null 2>&1; then
    diagnose_renewer_failure "process exited at the end of startup validation" "$pid"
    die "krenew pid $pid is not running"
  fi
  if [[ "$saw_success" -ne 1 ]]; then
    diagnose_renewer_failure "timed out waiting for the first successful daemon renewal" "$pid"
    die "krenew did not prove a successful first renewal within ${STARTUP_VALIDATION_SECONDS}s"
  fi

  RENEWER_PID="$pid"
  info "krenew pid $pid completed an initial renewal successfully"
}

kill_child_if_running() {
  if [[ -n "${CHILD_PID:-}" ]] && kill -0 "$CHILD_PID" >/dev/null 2>&1; then
    kill "$CHILD_PID" >/dev/null 2>&1 || true
  fi
}

child_is_running() {
  [[ -n "${CHILD_PID:-}" ]] || return 1
  kill -0 "$CHILD_PID" >/dev/null 2>&1 || return 1
  local stat
  stat="$(ps -o stat= -p "$CHILD_PID" 2>/dev/null | awk '{print $1}')"
  [[ -n "$stat" && "$stat" != Z* ]]
}

signal_child_group() {
  local signal_name="$1"
  [[ -n "${CHILD_PID:-}" ]] || return 0
  # The child is launched with setsid, so its PID is also its process group ID.
  kill "-$signal_name" -- "-$CHILD_PID" >/dev/null 2>&1 || true
}

wait_for_child_shutdown() {
  local grace_seconds="$1"
  local deadline=$((SECONDS + grace_seconds))
  while child_is_running; do
    ((SECONDS < deadline)) || return 1
    sleep 1
  done
  return 0
}

cleanup_renewer() {
  if [[ -s "${PID_FILE:-}" ]]; then
    stop_existing_renewer || true
  fi
}

forward_signal_to_child() {
  local signal_name="$1"
  CHILD_STOP_REQUESTED=1
  trap - INT TERM
  if child_is_running; then
    echo
    echo "Signal $signal_name received; forwarding to child process group $CHILD_PID."
    echo "Waiting up to ${CHILD_SIGNAL_GRACE_SECONDS}s for graceful child shutdown before TERM."
    signal_child_group "$signal_name"
  fi
}

wait_for_child_exit() {
  local status=0

  set +e
  while true; do
    wait "$CHILD_PID"
    status=$?
    if ((CHILD_STOP_REQUESTED == 1)); then
      if ! wait_for_child_shutdown "$CHILD_SIGNAL_GRACE_SECONDS"; then
        warn "child process group $CHILD_PID did not exit after ${CHILD_SIGNAL_GRACE_SECONDS}s; sending TERM"
        echo "Waiting up to ${CHILD_KILL_GRACE_SECONDS}s for cleanup after TERM before KILL."
        signal_child_group TERM
        if ! wait_for_child_shutdown "$CHILD_KILL_GRACE_SECONDS"; then
          warn "child process group $CHILD_PID did not exit after TERM; sending KILL"
          signal_child_group KILL
          wait_for_child_shutdown "$CHILD_KILL_GRACE_SECONDS" || true
        fi
      fi
      if child_is_running; then
        warn "child process $CHILD_PID is still present after KILL; exiting wrapper anyway"
      else
        echo "Child process group $CHILD_PID exited after signal handling."
        wait "$CHILD_PID" >/dev/null 2>&1 || true
      fi
      status=130
      break
    fi
    if ((status > 128)) && child_is_running; then
      continue
    fi
    break
  done
  set -e

  return "$status"
}

verify_credentials() {
  echo "Verifying Kerberos cache..."
  KRB5CCNAME="FILE:$CACHE_FILE" klist -f
  KRB5CCNAME="FILE:$CACHE_FILE" krenew -H 120 -k "$CACHE_FILE" >/dev/null

  if command -v tokens >/dev/null 2>&1; then
    echo
    echo "AFS tokens:"
    tokens || warn "tokens command failed"
  fi
}

cmd_run() {
  parse_common_opts "$@"
  [[ "${#REMAINING_ARGS[@]}" -gt 0 ]] || die "missing command; use -- COMMAND"

  require_command kinit
  require_command klist
  require_command krenew
  require_command aklog
  require_command setsid

  mkdir -p "$STATE_DIR"
  chmod 700 "$STATE_DIR"

  print_paths
  echo

  if cache_ok; then
    echo "Using existing valid cache."
  else
    echo "Initializing cache with kinit."
    kinit -c "$CACHE_FILE"
  fi

  preflight_renewal

  stop_existing_renewer
  echo "Starting krenew every ${CHECK_MINUTES} minutes."
  start_renewer

  echo
  verify_credentials

  echo
  echo "Launching command with KRB5CCNAME=FILE:$CACHE_FILE"
  echo "Stop renewer later with: $0 stop --name $NAME"
  echo

  trap 'forward_signal_to_child INT' INT
  trap 'forward_signal_to_child TERM' TERM
  trap 'cleanup_renewer' EXIT

  setsid env \
    KRB5CCNAME="FILE:$CACHE_FILE" \
    KRB_LONGRUN_CHILD_AKLOG_INTERVAL_SECONDS="$((CHECK_MINUTES * 60))" \
    bash --noprofile --norc -c '
      set -euo pipefail

      refresh_child_afs_tokens() {
        if [[ -z "${KRB5CCNAME:-}" ]] || ! command -v aklog >/dev/null 2>&1; then
          return 0
        fi
        if ! aklog >/dev/null 2>&1; then
          echo "Warning: child aklog failed for KRB5CCNAME=$KRB5CCNAME; AFS paths may be unavailable." >&2
        fi
      }

      aklog_loop_pid=""
      refresh_child_afs_tokens
      if [[ "${KRB_LONGRUN_CHILD_AKLOG_INTERVAL_SECONDS:-0}" =~ ^[1-9][0-9]*$ ]] && command -v aklog >/dev/null 2>&1; then
        (
          while true; do
            sleep "$KRB_LONGRUN_CHILD_AKLOG_INTERVAL_SECONDS" || exit 0
            refresh_child_afs_tokens
          done
        ) &
        aklog_loop_pid="$!"
      fi

      cleanup_child_helpers() {
        if [[ -n "$aklog_loop_pid" ]]; then
          kill "$aklog_loop_pid" >/dev/null 2>&1 || true
        fi
      }
      trap cleanup_child_helpers EXIT

      "$@" &
      cmd_pid="$!"
      set +e
      wait "$cmd_pid"
      cmd_status="$?"
      set -e
      exit "$cmd_status"
    ' _ "${REMAINING_ARGS[@]}" &
  CHILD_PID="$!"
  child_status=0
  wait_for_child_exit || child_status=$?
  return "$child_status"
}

cmd_status() {
  parse_common_opts "$@"
  print_paths
  echo

  if cache_ok; then
    KRB5CCNAME="FILE:$CACHE_FILE" klist -f
  else
    warn "cache is missing or invalid: $CACHE_FILE"
  fi

  echo
  if [[ -s "$PID_FILE" ]]; then
    local pid
    pid="$(tr -d '[:space:]' <"$PID_FILE")"
    if [[ -n "$pid" ]] && kill -0 "$pid" >/dev/null 2>&1; then
      echo "krenew: running pid $pid"
    else
      warn "krenew pid file exists but process is not running"
    fi
  else
    warn "no krenew pid file"
  fi

  if command -v tokens >/dev/null 2>&1; then
    echo
    tokens || true
  fi
}

cmd_stop() {
  parse_common_opts "$@"
  stop_existing_renewer
  echo "Stopped krenew for '$NAME'."
}

main() {
  [[ $# -gt 0 ]] || {
    usage
    exit 2
  }

  local subcommand="$1"
  shift
  case "$subcommand" in
    run) cmd_run "$@" ;;
    status) cmd_status "$@" ;;
    stop) cmd_stop "$@" ;;
    --help|-h|help) usage ;;
    *) die "unknown subcommand: $subcommand" ;;
  esac
}

main "$@"
