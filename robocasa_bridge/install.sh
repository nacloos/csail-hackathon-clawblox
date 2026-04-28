#!/usr/bin/env bash
# One-shot setup for the robocasa kitchen bridge.
#
# Installs Python deps via uv, copies robocasa's macros template into a
# private macros file, and downloads the ~10 GB kitchen asset bundle if
# it isn't already on disk. Safe to run from anywhere — the script anchors
# itself to the repo root via its own location.
#
# Usage:
#   ./robocasa_bridge/install.sh                # full setup
#   ./robocasa_bridge/install.sh --skip-assets  # skip the kitchen asset download
#   ./robocasa_bridge/install.sh --force-assets # re-download assets even if present
#   ./robocasa_bridge/install.sh --help

set -euo pipefail

cyan()   { printf '\033[36m%s\033[0m\n' "$*"; }
green()  { printf '\033[32m%s\033[0m\n' "$*"; }
yellow() { printf '\033[33m%s\033[0m\n' "$*"; }
red()    { printf '\033[31m%s\033[0m\n' "$*" >&2; }

usage() {
    sed -n '2,13p' "$0" | sed 's/^# \{0,1\}//'
    exit 0
}

SKIP_ASSETS=0
FORCE_ASSETS=0
for arg in "$@"; do
    case "$arg" in
        --skip-assets)  SKIP_ASSETS=1 ;;
        --force-assets) FORCE_ASSETS=1 ;;
        --help|-h)      usage ;;
        *) red "unknown arg: $arg"; usage ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

if [[ ! -f pyproject.toml ]]; then
    red "pyproject.toml not found in $REPO_ROOT — expected the bridge folder to live one level under the repo root."
    exit 1
fi

ROBOCASA_DIR="$REPO_ROOT/vendor/robocasa"
ROBOCASA_REPO_URL="https://github.com/robocasa/robocasa.git"

cyan "[1/5] checking prerequisites"
for cmd in git curl; do
    command -v "$cmd" >/dev/null || { red "missing prerequisite: $cmd"; exit 1; }
done
if ! command -v uv >/dev/null; then
    red "uv not found. Install with:"
    red "  curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi
green "  uv $(uv --version | awk '{print $2}') ok"

cyan "[2/5] vendoring robocasa"
if [[ -d "$ROBOCASA_DIR/.git" ]]; then
    green "  vendor/robocasa already cloned, skipping"
elif [[ -d "$ROBOCASA_DIR" ]]; then
    # Directory exists without a .git — likely a partial/broken clone or a
    # leftover from an earlier move. Bail rather than silently overwriting.
    red "  $ROBOCASA_DIR exists but isn't a git checkout."
    red "  Remove it (or restore the .git directory) and re-run."
    exit 1
else
    mkdir -p "$REPO_ROOT/vendor"
    yellow "  cloning $ROBOCASA_REPO_URL into vendor/robocasa"
    git clone "$ROBOCASA_REPO_URL" "$ROBOCASA_DIR"
    green "  clone complete"
fi

cyan "[3/5] syncing Python deps (uv sync)"
uv sync
green "  deps synced into .venv"

cyan "[4/5] setting up robocasa macros"
MACROS_SRC="$ROBOCASA_DIR/robocasa/macros.py"
MACROS_DST="$ROBOCASA_DIR/robocasa/macros_private.py"
if [[ ! -f "$MACROS_SRC" ]]; then
    red "  $MACROS_SRC missing — vendored robocasa looks broken."
    exit 1
fi
if [[ -f "$MACROS_DST" ]]; then
    green "  macros_private.py already present, skipping"
else
    cp "$MACROS_SRC" "$MACROS_DST"
    green "  created $MACROS_DST"
fi

cyan "[5/5] kitchen assets (~10 GB on first run)"
TEX_DIR="$ROBOCASA_DIR/robocasa/models/assets/textures"
OBJ_DIR="$ROBOCASA_DIR/robocasa/models/assets/objects/objaverse"
assets_present=0
if [[ -d "$TEX_DIR" ]] && [[ -n "$(ls -A "$TEX_DIR" 2>/dev/null)" ]] \
   && [[ -d "$OBJ_DIR" ]] && [[ -n "$(ls -A "$OBJ_DIR" 2>/dev/null)" ]]; then
    assets_present=1
fi

if [[ $SKIP_ASSETS -eq 1 ]]; then
    yellow "  --skip-assets passed; not downloading. The bridge will fail to load any kitchen scene that needs missing textures/objects."
elif [[ $assets_present -eq 1 && $FORCE_ASSETS -eq 0 ]]; then
    green "  assets already present (textures + objaverse non-empty); skipping. Pass --force-assets to re-download."
else
    yellow "  starting download — this is a multi-GB transfer and may take a while."
    # The upstream script prompts "Proceed? (y/n)" once; pipe a 'y' to bypass.
    printf 'y\n' | uv run python -m robocasa.scripts.download_kitchen_assets --type all
    green "  asset download complete"
fi

cyan "verifying install"
uv run python - <<'PY'
import importlib
mods = ["mujoco", "fastapi", "uvicorn", "robosuite", "robocasa"]
for m in mods:
    importlib.import_module(m)
print("import check ok:", ", ".join(mods))
PY

green ""
green "robocasa bridge is ready. Next steps:"
green "  headless server :  uv run python robocasa_bridge/robocasa_server.py"
green "  with viewer     :  DISPLAY=:0 uv run python robocasa_bridge/run_robocasa_viewer.py"
green "  client API docs :  see robocasa_bridge/README.md"
