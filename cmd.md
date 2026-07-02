 bash launch_multi_generations_claude.sh --world-dir worlds/mujoco-panda --generations 10 --generation-duration 1h --base-port 8085 --sandbox


uv run run_agent.py --duration 10m

uv run run_agent_generations.py --generations 30 --generation-duration 30m --base-port 8285 --tmux-prefix claude-2


scripts/krb-longrun.sh run --name panda-1 -- uv run run_agent_generations.py --generations 30 --generation-duration 30m --base-port 8185 --tmux-prefix claude-1 --world-dir worlds/mujoco-dual-panda --agents-per-world 2 --system-prompt worlds/mujoco-panda/system_prompt.md 


scripts/krb-longrun.sh run --name panda-1 -- uv run run_agent_generations.py --generations 30 --generation-duration 30m --base-port 8185 --tmux-prefix claude-1 --world-dir worlds/mujoco-panda --model claude-fable-5

scripts/krb-longrun.sh run --name panda-1 -- uv run run_agent_generations.py --generations 4 --generation-duration 5h --base-port 8185 --tmux-prefix claude-1 --world-dir worlds/mujoco-panda-2 --model claude-fable-5

scripts/krb-longrun.sh run --name g1-1 -- uv run run_agent_generations.py --generations 10 --generation-duration 1h --base-port 8185 --tmux-prefix claude-1 --world-dir worlds/mujoco-g1-dex3 --model claude-fable-5

---
unitree mujoco:
uv run run_agent_generations.py --world-dir worlds/unitree-mujoco --sandbox --template worlds/unitree-mujoco/agent-template --no-claude-native-sandbox