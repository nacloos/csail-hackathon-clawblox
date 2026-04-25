#!/usr/bin/env bash
set -euo pipefail

input="$(cat)"
log_dir="${CLAWBLOX_CLAUDE_LOG_DIR:-}"
mapfile -t parsed_fields < <(
  INPUT_JSON="$input" node <<'EOF'
const data = JSON.parse(process.env.INPUT_JSON ?? "{}");
const fields = [
  data.session_id ?? "",
  data.notification_type ?? "",
];
for (const value of fields) process.stdout.write(String(value) + "\n");
EOF
)
session_id="${parsed_fields[0]:-}"
notification_type="${parsed_fields[1]:-}"

if [[ -n "$log_dir" ]]; then
  mkdir -p "$log_dir"
  printf '%s\n' "$input" >"$log_dir/notification.json"
  if [[ -n "$notification_type" ]]; then
    printf '%s\n' "$input" >"$log_dir/notification.${notification_type}.json"
    if [[ -n "$session_id" ]]; then
      printf '%s\n' "$input" >"$log_dir/notification.${notification_type}.${session_id}.json"
    fi
  elif [[ -n "$session_id" ]]; then
    printf '%s\n' "$input" >"$log_dir/notification.${session_id}.json"
  fi
fi

exit 0
