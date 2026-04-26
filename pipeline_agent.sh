#!/usr/bin/env bash
# Long-lived poller agent that drives the ClawBlox video pipeline through its
# three human-input gates. Authenticates via CLAUDE_CODE_OAUTH_TOKEN
# (subscription billing, no API spend). Spawned by webapp.py when a run is
# created with mode=agent.
#
# Usage:
#   pipeline_agent.sh <run_name>
#
# Optional env:
#   WEBAPP_URL     base URL of the running webapp (default http://127.0.0.1:8000)
#   CLAUDE_MODEL   model id (default claude-opus-4-7)
#   CLAUDE_BIN     path to claude CLI (default: first 'claude' on PATH)

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <run_name>" >&2
  exit 2
fi

RUN_NAME="$1"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$SCRIPT_DIR"
RUN_DIR="$ROOT_DIR/runs/$RUN_NAME"
SOUL="$ROOT_DIR/SOUL.md"
METHODOLOGY="$ROOT_DIR/PIPELINE_AGENT.md"

if [[ ! -d "$RUN_DIR" ]]; then
  echo "Error: run dir does not exist: $RUN_DIR" >&2
  exit 1
fi
if [[ ! -f "$METHODOLOGY" ]]; then
  echo "Error: methodology doc missing: $METHODOLOGY" >&2
  exit 1
fi
if [[ ! -f "$SOUL" ]]; then
  echo "Error: soul doc missing: $SOUL" >&2
  exit 1
fi

# Strip any stray API key so the SDK can't fall through to paid billing.
unset ANTHROPIC_API_KEY 2>/dev/null || true

load_oauth_token() {
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

load_oauth_token

if [[ -z "${CLAUDE_CODE_OAUTH_TOKEN:-}" ]]; then
  echo "Error: CLAUDE_CODE_OAUTH_TOKEN not set (and not found in $ROOT_DIR/.env)." >&2
  echo "Generate one with: claude setup-token" >&2
  exit 1
fi

WEBAPP_URL="${WEBAPP_URL:-http://127.0.0.1:8000}"
CLAUDE_MODEL="${CLAUDE_MODEL:-claude-opus-4-7}"
CLAUDE_BIN="${CLAUDE_BIN:-$(command -v claude || true)}"
if [[ -z "$CLAUDE_BIN" ]]; then
  echo "Error: claude CLI not found in PATH." >&2
  exit 1
fi

# Per-run agent state.
AGENT_LOG="$RUN_DIR/agent.log"
AGENT_EVENTS="$RUN_DIR/agent.events.jsonl"
AGENT_PID_FILE="$RUN_DIR/agent.pid"
DECISIONS_LOG="$RUN_DIR/agent_decisions.log"
LOG_FILTER="$ROOT_DIR/scripts/agent_log_filter.py"
: >"$AGENT_LOG"
: >"$AGENT_EVENTS"
touch "$DECISIONS_LOG"

echo "$$" >"$AGENT_PID_FILE"
trap 'rm -f "$AGENT_PID_FILE"' EXIT

# Concatenate soul (identity + values) and methodology (operational manual)
# into one prompt. Soul comes first so the agent reads its values before the
# rules that operationalize them.
SOUL_BODY="$(cat "$SOUL")"
METHODOLOGY_BODY="$(cat "$METHODOLOGY")"

INITIAL_PROMPT="$SOUL_BODY

---

$METHODOLOGY_BODY

---

## This run

- WEBAPP_URL=$WEBAPP_URL
- RUN_NAME=$RUN_NAME
- RUN_DIR=$RUN_DIR

Begin the polling loop now. Confirm initial state with one curl, then proceed.
Do not summarize the soul or methodology back to me — just start working."

export WEBAPP_URL RUN_NAME RUN_DIR

# Headless, single long-lived session. The agent drives a Bash polling loop
# inside this turn until the pipeline completes or errors. Tools are scoped
# to what the loop needs.
# Stream every event (assistant text, tool calls, tool results) so the operator
# can watch the agent reason in real time. Raw events go to agent.events.jsonl
# verbatim; a human-readable digest is piped through the filter to agent.log.
# The pipeline below does:
#   claude --print --output-format stream-json --include-partial-messages \
#     | tee agent.events.jsonl  (raw JSONL, useful for debugging)
#     | agent_log_filter.py     (one short line per event)
#     >> agent.log              (what the UI tails)
set -o pipefail
"$CLAUDE_BIN" \
  --print \
  --model "$CLAUDE_MODEL" \
  --permission-mode bypassPermissions \
  --add-dir "$RUN_DIR" \
  --allowed-tools "Bash,Read,Write" \
  --disallowed-tools "WebFetch,WebSearch" \
  --output-format stream-json \
  --include-partial-messages \
  --verbose \
  -- "$INITIAL_PROMPT" 2>>"$AGENT_LOG" \
  | tee -a "$AGENT_EVENTS" \
  | python3 "$LOG_FILTER" >>"$AGENT_LOG"
