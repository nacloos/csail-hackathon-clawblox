#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="${ENV_NAME:-unitree-mujoco}"
UV_CACHE_DIR="${UV_CACHE_DIR:-/storage/nacloos/.uv}"
DEPS_DIR="${DEPS_DIR:-$ROOT_DIR/.deps}"
UNITREE_SDK_DIR="${UNITREE_SDK_DIR:-$DEPS_DIR/unitree_sdk2_python}"
UNITREE_SDK_REPO="${UNITREE_SDK_REPO:-https://github.com/unitreerobotics/unitree_sdk2_python.git}"
UNITREE_SDK_REF="${UNITREE_SDK_REF:-master}"

RECREATE=0
RUN_SMOKE=1
DOWNLOAD_FASTSAM=0

usage() {
  cat <<USAGE
Usage: ./install.sh [options]

Creates a no-sudo conda environment for the Unitree MuJoCo + G1 bricklaying stack.

Options:
  --env NAME             Conda environment name. Default: unitree-mujoco
  --recreate            Remove and recreate the conda environment.
  --no-smoke            Skip import/runtime smoke tests.
  --download-fastsam    Also download/load FastSAM-x.pt during smoke tests.
  -h, --help            Show this help.

Environment variables:
  UV_CACHE_DIR          uv cache directory. Default: /storage/nacloos/.uv
  DEPS_DIR              dependency checkout directory. Default: ./.deps
  UNITREE_SDK_DIR       Unitree SDK checkout path. Default: ./.deps/unitree_sdk2_python
  UNITREE_SDK_REPO      Unitree SDK git URL. Default: official Unitree GitHub repo
  UNITREE_SDK_REF       Unitree SDK ref. Default: master

Notes:
  - No sudo or Docker is used.
  - Pinocchio is installed from conda-forge because pip wheels do not expose
    pinocchio.casadi, which the bricklaying IK code requires.
  - unitree_sdk2_python is installed from a fresh source checkout under .deps.
    It is installed non-editably so sandboxed agents can import it from the
    mounted conda environment without needing the source checkout mounted too.
  - Unitree's CycloneDDS config is patched to write cdds.LOG in the process
    working directory instead of /tmp/cdds.LOG, because Claude's native sandbox
    blocks writes to the sandbox-level /tmp path.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env)
      ENV_NAME="$2"
      shift 2
      ;;
    --recreate)
      RECREATE=1
      shift
      ;;
    --no-smoke)
      RUN_SMOKE=0
      shift
      ;;
    --download-fastsam)
      DOWNLOAD_FASTSAM=1
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

export UV_CACHE_DIR

need_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Missing required command: $1" >&2
    exit 1
  fi
}

need_cmd conda
need_cmd uv
need_cmd git

if [[ ! -d "$ROOT_DIR/Unitree-Mujoco-Dex3" || ! -d "$ROOT_DIR/G1-Bricklaying-Simulation" ]]; then
  echo "Run this script from the unitree-mujoco workspace containing both repos." >&2
  exit 1
fi

CONDA_BASE="$(conda info --base)"
ENV_DIR="$CONDA_BASE/envs/$ENV_NAME"
PYTHON="$ENV_DIR/bin/python"

if [[ "$RECREATE" == "1" && -d "$ENV_DIR" ]]; then
  conda env remove -y -n "$ENV_NAME"
fi

if [[ ! -x "$PYTHON" ]]; then
  conda create -y --solver libmamba -n "$ENV_NAME" -c conda-forge \
    python=3.10 \
    pip \
    pinocchio \
    casadi \
    numpy \
    scipy \
    matplotlib
fi

mkdir -p "$DEPS_DIR"
if [[ ! -d "$UNITREE_SDK_DIR/.git" ]]; then
  git clone "$UNITREE_SDK_REPO" "$UNITREE_SDK_DIR"
fi
git -C "$UNITREE_SDK_DIR" fetch --tags --quiet
git -C "$UNITREE_SDK_DIR" checkout "$UNITREE_SDK_REF" --quiet

"$PYTHON" - <<'PY'
import sys
if sys.version_info[:2] != (3, 10):
    raise SystemExit(f"expected Python 3.10, got {sys.version}")
PY

# PyTorch first: default PyPI currently selects CUDA 13 wheels, which do not run
# with the CUDA 12.5 NVIDIA driver on this machine. The cu121 wheels do.
uv pip install --python "$PYTHON" \
  --index-url https://download.pytorch.org/whl/cu121 \
  --extra-index-url https://pypi.org/simple \
  --index-strategy unsafe-best-match \
  "torch==2.5.1+cu121" \
  "torchvision==0.20.1+cu121"

# Core simulator, DDS, ROS, perception, and web-viewer dependencies.
uv pip install --python "$PYTHON" \
  mujoco \
  pygame \
  numpy-quaternion \
  opencv-python \
  open3d \
  trimesh \
  cvxpy \
  meshcat \
  ultralytics \
  transformers \
  flask \
  mjviser \
  viser \
  ros-rclpy \
  ros-rmw-fastrtps-cpp \
  ros-geometry-msgs

# Install the Unitree SDK into site-packages, not as an editable source path.
# Sandboxed agents see the conda environment as /sandbox-deps, but they do not
# see arbitrary source checkouts such as .deps/unitree_sdk2_python.
uv pip uninstall --python "$PYTHON" unitree_sdk2py >/dev/null 2>&1 || true
uv pip install --python "$PYTHON" "$UNITREE_SDK_DIR"

# Unitree's setup.py does not declare the native CRC shared libraries as package
# data, so a normal wheel install omits them. Copy them into the installed
# package to keep CRC available while still avoiding an editable install.
SDK_PACKAGE_DIR="$("$PYTHON" - <<'PY'
import pathlib
import unitree_sdk2py

print(pathlib.Path(unitree_sdk2py.__file__).resolve().parent)
PY
)"
mkdir -p "$SDK_PACKAGE_DIR/utils/lib"
cp -a "$UNITREE_SDK_DIR/unitree_sdk2py/utils/lib/." "$SDK_PACKAGE_DIR/utils/lib/"

# Unitree's default interface-specific CycloneDDS config writes tracing to
# /tmp/cdds.LOG. Claude's native sandbox mounts /tmp as an internal tmpfs and
# denies that write, which prevents ChannelFactoryInitialize(0, "lo") from
# creating a DDS domain. Use a relative log path so the file lands in the
# process working directory (/workspace for agents, the world dir for server
# checks) where writes are allowed.
CHANNEL_CONFIG="$SDK_PACKAGE_DIR/core/channel_config.py"
"$PYTHON" - <<PY
import pathlib

path = pathlib.Path("$CHANNEL_CONFIG")
text = path.read_text()
old = "<OutputFile>/tmp/cdds.LOG</OutputFile>"
new = "<OutputFile>cdds.LOG</OutputFile>"
if old in text:
    path.write_text(text.replace(old, new))
elif new not in text:
    raise SystemExit(f"could not patch CycloneDDS log path in {path}")
print("CycloneDDS log path OK:", path)
PY

"$PYTHON" - <<PY
import pathlib
import unitree_sdk2py
from unitree_sdk2py.utils.crc import CRC

env_dir = pathlib.Path("$ENV_DIR").resolve()
package_file = pathlib.Path(unitree_sdk2py.__file__).resolve()
try:
    package_file.relative_to(env_dir)
except ValueError as exc:
    raise SystemExit(
        "unitree_sdk2py must be installed inside the conda env, not as an "
        f"editable checkout: {package_file}"
    ) from exc

lib_dir = package_file.parent / "utils" / "lib"
channel_config = package_file.parent / "core" / "channel_config.py"
missing = [
    name
    for name in ("crc_amd64.so", "crc_aarch64.so")
    if not (lib_dir / name).is_file()
]
if missing:
    raise SystemExit(f"unitree_sdk2py missing native CRC libraries: {missing}")
if "/tmp/cdds.LOG" in channel_config.read_text():
    raise SystemExit(f"unitree_sdk2py still writes CycloneDDS logs to /tmp: {channel_config}")

CRC()
print("Unitree SDK install OK:", package_file)
PY

# Install the local bricklaying package into site-packages too. The sandbox has
# its own workspace copy of the source, but dependency imports should not depend
# on editable paths outside the sandbox.
uv pip uninstall --python "$PYTHON" g1-bricklaying >/dev/null 2>&1 || true
uv pip install --python "$PYTHON" "$ROOT_DIR/G1-Bricklaying-Simulation"

if [[ "$RUN_SMOKE" == "1" ]]; then
  echo "Running smoke tests..."
  MUJOCO_GL=egl "$PYTHON" - <<PY
import pathlib
import socket
import sys
import tempfile

root = pathlib.Path("$ROOT_DIR")

import pinocchio
from pinocchio import casadi as cpin
import casadi
import mujoco
import torch
import rclpy
from geometry_msgs.msg import Twist
from unitree_sdk2py.utils.crc import CRC
from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from bricklaying.robot.kinematics import DualArmIK
from bricklaying.segmentation.fastsam import FastSAMSegmentor
import viser
from mjviser.scene import ViserMujocoScene

crc = CRC()
print("CRC library:", crc.platform)
print("Pinocchio:", pinocchio.__version__, "casadi bindings OK")
print("Torch:", torch.__version__, "cuda:", torch.cuda.is_available(), "devices:", torch.cuda.device_count())

scene_path = root / "Unitree-Mujoco-Dex3/unitree_robots/g1/scene_hands_modified.xml"
model = mujoco.MjModel.from_xml_path(str(scene_path))
data = mujoco.MjData(model)
renderer = mujoco.Renderer(model, 120, 160)
renderer.update_scene(data, 0)
image = renderer.render()
renderer.close()
print("MuJoCo EGL render OK:", model.nbody, "bodies,", model.nu, "actuators,", image.shape)

with socket.socket() as sock:
    sock.bind(("127.0.0.1", 0))
    viser_port = sock.getsockname()[1]
server = viser.ViserServer(host="127.0.0.1", port=viser_port, label="unitree-install-smoke")
viewer = ViserMujocoScene(server, model, num_envs=1)
viewer.update_from_mjdata(data)
server.stop()
print("Viser/MJViser scene OK")

old_cwd = pathlib.Path.cwd()
import os
with tempfile.TemporaryDirectory(prefix="unitree-dds-smoke-") as dds_log_dir:
    try:
        os.chdir(dds_log_dir)
        ChannelFactoryInitialize(0, "lo")
    finally:
        os.chdir(old_cwd)
print("DDS initialize OK on lo")

ik = DualArmIK()
print("DualArmIK OK:", ik.model.nq, "DoF")

if $DOWNLOAD_FASTSAM:
    segmentor = FastSAMSegmentor()
    print("FastSAM OK:", type(segmentor.model).__name__)
else:
    print("FastSAM import OK; checkpoint download skipped")
PY
fi

cat <<DONE

Install complete.

Activate:
  conda activate $ENV_NAME

Useful env:
  export UV_CACHE_DIR=$UV_CACHE_DIR
  export MUJOCO_GL=egl

Known repo-level follow-up:
  Unitree-Mujoco-Dex3/simulate_python/unitree_mujoco.py still launches the native GLFW viewer.
  For this headless machine, use a headless/web-viewer runner or run from a session with DISPLAY.
DONE
