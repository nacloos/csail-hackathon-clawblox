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
  printf '%s\n' "$input" >"$log_dir/stop.json"
  if [[ -n "$session_id" ]]; then
    printf '%s\n' "$input" >"$log_dir/stop.${session_id}.json"
  fi
fi

auto_loop="${CLAWBLOX_CLAUDE_AUTO_LOOP:-0}"

if [[ "$auto_loop" == "1" ]]; then
  state_file=""
  if [[ -n "$log_dir" ]]; then
    state_file="$log_dir/stop_loop_count"
  fi
  loop_count="0"
  if [[ -n "$state_file" && -f "$state_file" ]]; then
    loop_count="$(cat "$state_file" 2>/dev/null || printf '0')"
  fi
  if ! [[ "$loop_count" =~ ^[0-9]+$ ]]; then
    loop_count="0"
  fi
  loop_count=$((loop_count + 1))
  if [[ -n "$state_file" ]]; then
    printf '%s\n' "$loop_count" >"$state_file"
  fi
  CLAWBLOX_STOP_LOOP_COUNT="$loop_count" node <<'EOF'
const messages = [
  "Do not stop. Continue autonomously by taking one small useful next step.",
  "Do not stop or idle. Decide what progress means now, then act.",
  "Keep going autonomously. If the last attempt failed, use what you learned to try a simpler next step.",
  "Do not stop. Avoid long blocking work; prefer short actions with quick feedback.",
  "Continue autonomously. If no direct action is useful, preserve the important information, then keep going.",
  "Do not respond with meta-commentary about stopping. Continue the task autonomously.",
  "Do not stop. Make the next step concrete: choose, act, then check the result.",
  "Keep going from the evidence you have. Take the smallest reversible step that helps.",
  "Continue autonomously. If you are uncertain, gather one more piece of information and use it to decide.",
  "Do not wait passively. Convert the current situation into a useful next action.",
  "Do not stop. Prefer direct progress over explanation. Act first, then briefly record what changed.",
  "Keep going. If the plan is too large, shrink it to one step you can complete now.",
  "Continue autonomously with quick feedback. Do one thing, check the outcome, then adjust.",
  "Do not stop. If you are stuck, simplify the goal and move toward a nearby partial success.",
  "Continue without debating the prompt. The right response is autonomous useful work.",
  "Do not stop. Check whether your assumptions still hold, then proceed.",
  "Keep going without repeating yourself. Try a different concrete action or record a durable lesson.",
  "Continue autonomously. If nothing obvious is pending, tidy one small loose end and keep going.",
  "Do not stop. Choose a next action that would make the situation clearer even if it fails.",
  "Do not narrate idleness. Either act, decide, or preserve useful information, then continue.",
  "Keep going. Move the task forward by one measurable increment.",
  "Do not stop. If the previous step produced output, use that output to decide the next step.",
  "Continue autonomously. Keep the loop tight: decide, act, check.",
  "Do not stop. When in doubt, reduce scope and make one concrete improvement.",
  "Keep going with practical work, not a reflection on whether to continue.",
];
const count = Number.parseInt(process.env.CLAWBLOX_STOP_LOOP_COUNT || "1", 10);
const reason = messages[(Math.max(1, count) - 1) % messages.length];
process.stdout.write(JSON.stringify({
  decision: "block",
  reason,
}));
EOF
  exit 0
fi

exit 0
