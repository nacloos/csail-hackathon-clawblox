#!/usr/bin/env bash
set -euo pipefail

input="$(cat)"
mapfile -t parsed_fields < <(
  INPUT_JSON="$input" node <<'EOF'
const data = JSON.parse(process.env.INPUT_JSON ?? "{}");
const fields = [
  data.session_id ?? "",
  data.transcript_path ?? "",
  data.cwd ?? "",
  data.source ?? "",
  data.model ?? "",
];
for (const value of fields) process.stdout.write(String(value) + "\n");
EOF
)
session_id="${parsed_fields[0]:-}"
transcript_path="${parsed_fields[1]:-}"
cwd_path="${parsed_fields[2]:-}"
source_kind="${parsed_fields[3]:-}"
model_name="${parsed_fields[4]:-}"

agent_dir="${CLAWBLOX_CLAUDE_AGENT_DIR:-}"
log_dir="${CLAWBLOX_CLAUDE_LOG_DIR:-}"
metadata_file="${CLAWBLOX_CLAUDE_METADATA_FILE:-}"

if [[ -n "$log_dir" ]]; then
  mkdir -p "$log_dir"
  printf '%s\n' "$input" >"$log_dir/session_start.json"
  if [[ -n "$session_id" ]]; then
    printf '%s\n' "$input" >"$log_dir/session_start.${session_id}.json"
  fi
fi

if [[ -n "$metadata_file" ]]; then
  mkdir -p "$(dirname "$metadata_file")"
  cat >"$metadata_file" <<EOF
SESSION_ID=$(printf '%q' "$session_id")
TRANSCRIPT_PATH=$(printf '%q' "$transcript_path")
CWD=$(printf '%q' "$cwd_path")
SOURCE=$(printf '%q' "$source_kind")
MODEL=$(printf '%q' "$model_name")
EOF
fi

if [[ -n "${CLAUDE_ENV_FILE:-}" ]]; then
  {
    printf 'export CLAWBLOX_SESSION_ID=%q\n' "$session_id"
    printf 'export CLAWBLOX_TRANSCRIPT_PATH=%q\n' "$transcript_path"
    printf 'export CLAWBLOX_AGENT_DIR=%q\n' "$agent_dir"
  } >>"$CLAUDE_ENV_FILE"
fi

INPUT_SESSION_ID="$session_id" \
INPUT_TRANSCRIPT_PATH="$transcript_path" \
INPUT_CWD_PATH="$cwd_path" \
INPUT_SOURCE_KIND="$source_kind" \
INPUT_MODEL_NAME="$model_name" \
node <<'EOF'
const payload = {
  hookSpecificOutput: {
    hookEventName: "SessionStart",
    additionalContext:
      "Runtime session metadata:\n" +
      `- session_id: ${process.env.INPUT_SESSION_ID ?? ""}\n` +
      `- transcript_path: ${process.env.INPUT_TRANSCRIPT_PATH ?? ""}\n` +
      `- cwd: ${process.env.INPUT_CWD_PATH ?? ""}\n` +
      `- source: ${process.env.INPUT_SOURCE_KIND ?? ""}\n` +
      `- model: ${process.env.INPUT_MODEL_NAME ?? ""}`,
  },
};
process.stdout.write(JSON.stringify(payload));
EOF
