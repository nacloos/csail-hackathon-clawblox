#!/usr/bin/env bash
set -euo pipefail

input="$(cat)"
log_dir="${CLAWBLOX_CLAUDE_LOG_DIR:-}"
session_id="$(
  INPUT_JSON="$input" node <<'EOF'
const data = JSON.parse(process.env.INPUT_JSON || "{}");
process.stdout.write(String(data.session_id || ""));
EOF
)"

if [[ -n "$log_dir" ]]; then
  mkdir -p "$log_dir"
  printf '%s\n' "$input" >"$log_dir/user_prompt_submit.json"
  if [[ -n "$session_id" ]]; then
    printf '%s\n' "$input" >"$log_dir/user_prompt_submit.${session_id}.json"
  fi
fi

exit 0
