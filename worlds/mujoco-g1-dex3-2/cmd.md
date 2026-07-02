scripts/krb-longrun.sh run --name g1-1 -- uv run scripts/actuate_agents.py --generations 10 --generation-duration 2h --base-port 8185 --tmux-prefix claude-1 --world-dir worlds/mujoco-g1-dex3-2 --model claude-fable-5 --claude-effort medium

