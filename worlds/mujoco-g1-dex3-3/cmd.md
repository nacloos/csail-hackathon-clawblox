scripts/krb-longrun.sh run --name g1-1 -- uv run -m actuate.experiment --generations 10 --generation-duration 2h --base-port 8185 --tmux-prefix claude-1 --world-dir worlds/mujoco-g1-dex3-3 --template worlds/mujoco-g1-dex3-3/template/agent_no_soul --model claude-fable-5 --claude-effort medium --no-claude-native-sandbox --env-file .env

# Notes for this DDS world (differs from the HTTP dex3-2 world):
#
# --no-claude-native-sandbox is REQUIRED. The agent controls the robot over a
#   CycloneDDS bus (rt/lowcmd / rt/lowstate), which is UDP on loopback. Claude's
#   native sandbox puts the agent in its own network namespace, so its DDS
#   packets can't reach the world (only the HTTP proxy bridges namespaces).
#   Dropping it makes the agent run in Actuate's bubblewrap only, which shares
#   the host network namespace with the world, so DDS reaches. Isolation between
#   concurrent runs then relies on the per-run DDS domain id + capability proxy
#   (soft boundary) rather than a network namespace.
#
# Setup (one-time, per machine): the world runs in the `unitree-mujoco` conda
#   env (same cyclonedds 0.10.2 + unitree_hg IDL the agent's mounted deps use —
#   a version mismatch segfaults DDS type discovery). That env also needs
#   fastapi + uvicorn for the world's HTTP lifecycle:
#     uv pip install --python "$(conda info --base)/envs/unitree-mujoco/bin/python" fastapi uvicorn
