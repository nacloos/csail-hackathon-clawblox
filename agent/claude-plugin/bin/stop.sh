#!/usr/bin/env bash
set -euo pipefail

input="$(cat)"
log_dir="${CLAWBLOX_CLAUDE_LOG_DIR:-}"
session_id="$(
  INPUT_JSON="$input" node <<'EOF'
const data = JSON.parse(process.env.INPUT_JSON ?? "{}");
process.stdout.write(String(data.session_id ?? ""));
EOF
)"

if [[ -n "$log_dir" ]]; then
  mkdir -p "$log_dir"
  printf '%s\n' "$input" >"$log_dir/stop.json"
  if [[ -n "$session_id" ]]; then
    printf '%s\n' "$input" >"$log_dir/stop.${session_id}.json"
  fi
fi

auto_loop="${CLAWBLOX_CLAUDE_AUTO_LOOP:-0}"

if [[ "$auto_loop" == "1" ]]; then
  node <<'EOF'
process.stdout.write(JSON.stringify({
  decision: "block",
  reason: "Please don't stop. Just continue doing whatever you want to do.",
}));
EOF
  exit 0
fi

exit 0
