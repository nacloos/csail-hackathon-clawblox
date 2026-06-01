#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$ROOT_DIR/../.." && pwd)"
SCRIPT_PATH="$ROOT_DIR/$(basename "${BASH_SOURCE[0]}")"

PYTHON_BIN="${PYTHON_BIN:-/storage/nacloos/libraries/conda/envs/unitree-mujoco/bin/python}"
HARNESS="${HARNESS:-$ROOT_DIR/verified_pick_place_harness.py}"
DOMAIN_ID="${UNITREE_DDS_DOMAIN_ID:-160}"
API_HOST="${API_HOST:-127.0.0.1}"
API_PORT="${API_PORT:-18140}"
SPECTATOR_HOST="${SPECTATOR_HOST:-127.0.0.1}"
SPECTATOR_PUBLIC_HOST="${SPECTATOR_PUBLIC_HOST:-127.0.0.1}"
SPECTATOR_PORT="${SPECTATOR_PORT:-19140}"
LOG_DIR="${LOG_DIR:-/tmp}"
TMUX_SESSION="${TMUX_SESSION:-unitree-pick-place}"

usage() {
  cat <<USAGE
Usage: ./run_verified_pick_place.sh [--no-tmux] [--no-server]

Runs the verified Unitree G1 brick pick/place attempt from a clean set of
environment settings.

Options:
  --no-tmux      Run in the current shell instead of creating tmux panes.
  --no-server    Do not start the world server; assume it is already running.
  --server-only  Start only the world server. Used by the tmux server pane.
  -h, --help     Show this help.

Environment overrides:
  PYTHON_BIN             Python executable. Default: $PYTHON_BIN
  HARNESS                Pick/place harness. Default: $HARNESS
  UNITREE_DDS_DOMAIN_ID  DDS domain. Default: $DOMAIN_ID
  API_PORT               Clawblox API port. Default: $API_PORT
  SPECTATOR_PORT         spectator port. Default: $SPECTATOR_PORT
  LOG_DIR                copied result logs directory. Default: $LOG_DIR
  TMUX_SESSION           tmux session name. Default: $TMUX_SESSION

Spectator:
  http://$SPECTATOR_PUBLIC_HOST:$SPECTATOR_PORT/
USAGE
}

START_SERVER=1
USE_TMUX=1
SERVER_ONLY=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-tmux)
      USE_TMUX=0
      shift
      ;;
    --no-server)
      START_SERVER=0
      shift
      ;;
    --server-only)
      SERVER_ONLY=1
      USE_TMUX=0
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python executable not found or not executable: $PYTHON_BIN" >&2
  exit 1
fi

if [[ ! -f "$HARNESS" ]]; then
  cat >&2 <<EOF
Verified harness not found: $HARNESS

This runner currently wraps the verified temporary harness from run 79.
Set HARNESS=/path/to/codex_servo_pick_attempt.py if it has been moved.
EOF
  exit 1
fi

mkdir -p "$LOG_DIR" /tmp/ros_logs /tmp/ultralytics /tmp/matplotlib

export UNITREE_DDS_DOMAIN_ID="$DOMAIN_ID"
export ROS_LOCALHOST_ONLY="${ROS_LOCALHOST_ONLY:-1}"
export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_fastrtps_cpp}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export ROS_LOG_DIR="${ROS_LOG_DIR:-/tmp/ros_logs}"
export YOLO_CONFIG_DIR="${YOLO_CONFIG_DIR:-/tmp/ultralytics}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"

export PREPICK_ROUTE_MODE="${PREPICK_ROUTE_MODE:-staged}"
export PICK_TRACK_TOL="${PICK_TRACK_TOL:-0.18}"
export NAV_TARGET_SD="${NAV_TARGET_SD:-0.0095}"
export NAV_TARGET_X="${NAV_TARGET_X:-0.43}"
export IK_ABORT_STEP="${IK_ABORT_STEP:-0.35}"
export PREPICK_STAGE_ATTEMPTS="${PREPICK_STAGE_ATTEMPTS:-3}"
export PREPICK_ROUTE_CLEARANCE="${PREPICK_ROUTE_CLEARANCE:-0.18}"
export PREPICK_HIGH_Z="${PREPICK_HIGH_Z:-0.25}"
export PICK_FREE_MODE="${PICK_FREE_MODE:-right_waist}"
export PREINIT_LEFT_PARK="${PREINIT_LEFT_PARK:-0}"
export LEFT_HAND_MODE="${LEFT_HAND_MODE:-close}"
export INITIAL_RIGHT_HAND_MODE="${INITIAL_RIGHT_HAND_MODE:-open}"
export POST_NAV_DEMO_INIT="${POST_NAV_DEMO_INIT:-0}"
export RUN_SCALED_INIT="${RUN_SCALED_INIT:-0}"
export RUN_INIT="${RUN_INIT:-0}"
export MIN_CARRY_DISP="${MIN_CARRY_DISP:-0.08}"
export MIN_CARRY_Z_DELTA="${MIN_CARRY_Z_DELTA:-0.06}"
export PLACE_TARGET_ROW="${PLACE_TARGET_ROW:-0.09,-0.04,0.09,0,0,-25}"
export EXTERNAL_HOLD_KP="${EXTERNAL_HOLD_KP:-25}"
export EXTERNAL_HOLD_KD="${EXTERNAL_HOLD_KD:-1}"
export RIGHT_KP="${RIGHT_KP:-25}"
export RIGHT_KD="${RIGHT_KD:-1}"

api_url="http://$API_HOST:$API_PORT/"
observe_url="http://$API_HOST:$API_PORT/observe"

api_ready() {
  "$PYTHON_BIN" - "$observe_url" <<'PY' >/dev/null 2>&1
import json
import sys
import urllib.request

with urllib.request.urlopen(sys.argv[1], timeout=1.0) as resp:
    json.load(resp)
PY
}

listening_pids_for_port() {
  local port="$1"
  if command -v ss >/dev/null 2>&1; then
    ss -ltnp "sport = :$port" 2>/dev/null \
      | sed -n 's/.*pid=\([0-9][0-9]*\).*/\1/p' \
      | sort -u
  elif command -v lsof >/dev/null 2>&1; then
    timeout 2 lsof -tiTCP:"$port" -sTCP:LISTEN 2>/dev/null || true
  elif command -v fuser >/dev/null 2>&1; then
    timeout 2 fuser -n tcp "$port" 2>/dev/null || true
  fi
}

kill_process_tree() {
  local pid="$1"
  local child
  while read -r child; do
    [[ -n "$child" ]] || continue
    kill_process_tree "$child"
  done < <(pgrep -P "$pid" 2>/dev/null || true)
  kill "$pid" >/dev/null 2>&1 || true
}

clean_existing_world_server() {
  local pids=()
  local pid
  while read -r pid; do
    [[ -n "$pid" ]] && pids+=("$pid")
  done < <({ listening_pids_for_port "$API_PORT"; listening_pids_for_port "$SPECTATOR_PORT"; } | sort -u)

  if [[ "${#pids[@]}" -eq 0 ]]; then
    return
  fi

  echo "Stopping existing world server processes on ports $API_PORT/$SPECTATOR_PORT: ${pids[*]}"
  for pid in "${pids[@]}"; do
    kill_process_tree "$pid"
  done

  for _ in $(seq 1 20); do
    if ! api_ready && [[ -z "$(listening_pids_for_port "$API_PORT")" ]] && [[ -z "$(listening_pids_for_port "$SPECTATOR_PORT")" ]]; then
      return
    fi
    sleep 0.5
  done

  while read -r pid; do
    [[ -n "$pid" ]] || continue
    echo "Force-stopping process $pid still holding world server ports"
    kill -9 "$pid" >/dev/null 2>&1 || true
  done < <({ listening_pids_for_port "$API_PORT"; listening_pids_for_port "$SPECTATOR_PORT"; } | sort -u)
}

shell_quote() {
  printf '%q' "$1"
}

tmux_child_command() {
  local extra_args=("$@")
  local cmd
  cmd="cd $(shell_quote "$ROOT_DIR") &&"
  cmd+=" PYTHON_BIN=$(shell_quote "$PYTHON_BIN")"
  cmd+=" HARNESS=$(shell_quote "$HARNESS")"
  cmd+=" UNITREE_DDS_DOMAIN_ID=$(shell_quote "$DOMAIN_ID")"
  cmd+=" API_HOST=$(shell_quote "$API_HOST")"
  cmd+=" API_PORT=$(shell_quote "$API_PORT")"
  cmd+=" SPECTATOR_HOST=$(shell_quote "$SPECTATOR_HOST")"
  cmd+=" SPECTATOR_PUBLIC_HOST=$(shell_quote "$SPECTATOR_PUBLIC_HOST")"
  cmd+=" SPECTATOR_PORT=$(shell_quote "$SPECTATOR_PORT")"
  cmd+=" LOG_DIR=$(shell_quote "$LOG_DIR")"
  cmd+=" $(shell_quote "$SCRIPT_PATH") --no-tmux"
  for arg in "${extra_args[@]}"; do
    cmd+=" $(shell_quote "$arg")"
  done
  printf '%s' "$cmd"
}

if [[ "$START_SERVER" == "1" ]]; then
  clean_existing_world_server
fi

if [[ "$USE_TMUX" == "1" ]]; then
  if ! command -v tmux >/dev/null 2>&1; then
    echo "tmux not found; falling back to current-shell run." >&2
  else
    session="$TMUX_SESSION"
    if tmux has-session -t "$session" 2>/dev/null; then
      echo "Replacing existing tmux session: $session"
      tmux kill-session -t "$session"
    fi

    server_cmd="$(tmux_child_command --server-only)"
    runner_cmd="$(tmux_child_command --no-server)"

    tmux new-session -d -s "$session" -n pick-place "$server_cmd"
    tmux split-window -h -t "$session:0" "sleep 5; $runner_cmd"
    tmux select-layout -t "$session:0" even-horizontal >/dev/null
    tmux select-pane -t "$session:0.1"

    echo "Created tmux session: $session"
    echo "Left pane: world server. Right pane: verified pick/place harness."
    echo "Spectator: http://$SPECTATOR_PUBLIC_HOST:$SPECTATOR_PORT/"
    if [[ -t 0 ]]; then
      exec tmux attach-session -t "$session"
    fi
    echo "Attach with: tmux attach -t $session"
    exit 0
  fi
fi

server_pid=""
cleanup() {
  if [[ -n "$server_pid" ]]; then
    echo "Stopping world server pid $server_pid"
    kill "$server_pid" >/dev/null 2>&1 || true
    wait "$server_pid" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

if [[ "$START_SERVER" == "1" ]]; then
  echo "Starting clean world server at $api_url"
  if [[ "$SERVER_ONLY" == "1" ]]; then
    cd "$ROOT_DIR"
    exec "$PYTHON_BIN" server.py \
      --host "$API_HOST" \
      --port "$API_PORT" \
      --spectator-host "$SPECTATOR_HOST" \
      --spectator-public-host "$SPECTATOR_PUBLIC_HOST" \
      --spectator-port "$SPECTATOR_PORT" \
      --enable-cmd-vel
  fi
  (
    cd "$ROOT_DIR"
    exec "$PYTHON_BIN" server.py \
      --host "$API_HOST" \
      --port "$API_PORT" \
      --spectator-host "$SPECTATOR_HOST" \
      --spectator-public-host "$SPECTATOR_PUBLIC_HOST" \
      --spectator-port "$SPECTATOR_PORT" \
      --enable-cmd-vel
  ) &
  server_pid="$!"
  for _ in $(seq 1 60); do
    if api_ready; then
      break
    fi
    sleep 1
  done
  if ! api_ready; then
    echo "World server did not become ready at $api_url" >&2
    exit 1
  fi
fi

if [[ "$SERVER_ONLY" == "1" ]]; then
  echo "Server pane is keeping the world server alive."
  echo "Spectator: http://$SPECTATOR_PUBLIC_HOST:$SPECTATOR_PORT/"
  wait "$server_pid"
  exit $?
fi

if ! api_ready; then
  echo "Waiting for world server at $api_url"
  for _ in $(seq 1 60); do
    if api_ready; then
      break
    fi
    sleep 1
  done
  if ! api_ready; then
    echo "World server is not ready at $api_url" >&2
    exit 1
  fi
fi

echo "Spectator: http://$SPECTATOR_PUBLIC_HOST:$SPECTATOR_PORT/"
echo "Running verified pick/place harness on DDS domain $UNITREE_DDS_DOMAIN_ID"

(
  cd "$REPO_ROOT"
  exec "$PYTHON_BIN" "$HARNESS"
)

stamp="$(date -u +%Y%m%dT%H%M%SZ)"
if [[ -f /tmp/codex_servo_pick_attempt.jsonl ]]; then
  cp /tmp/codex_servo_pick_attempt.jsonl "$LOG_DIR/codex_servo_pick_attempt_${stamp}.jsonl"
  echo "Copied run log to $LOG_DIR/codex_servo_pick_attempt_${stamp}.jsonl"
fi
