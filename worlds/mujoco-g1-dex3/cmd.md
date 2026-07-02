scripts/krb-longrun.sh run --name g1-1 -- uv run run_agent_generations.py --generations 10 --generation-duration 1h --base-port 8185 --tmux-prefix claude-1 --world-dir worlds/mujoco-g1-dex3


CODEX_HOME=/var/tmp/codex-nacloos scripts/krb-longrun.sh run --name g1-2 -- uv run run_agent_generations.py --generations 10 --generation-duration 1h --base-port 8285 --tmux-prefix codex-1 --world-dir worlds/mujoco-g1-dex3 --backend codex