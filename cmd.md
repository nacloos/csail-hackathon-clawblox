 bash launch_multi_generations_claude.sh --world-dir worlds/mujoco-panda --generations 10 --generation-duration 1h --base-port 8085 --sandbox


uv run run_agent.py --duration 10m

uv run run_agent_generations.py --generations 30 --generation-duration 30m --base-port 8285 --tmux-prefix claude-2

