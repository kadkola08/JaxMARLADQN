"""
Visualize Jumanji RobotWarehouse using the built-in policy visualizer from Jumanji.

Supports both RobotWarehouse-v0 (4 agents, default map) and RobotWarehouseSmall-v0
(1 agent, small map). The small variant is provided by JaxMARL and uses the same
Jumanji viewer for animate().

Usage:
    python viz_jumanji.py                    # default (RobotWarehouse-v0)
    python viz_jumanji.py --small           # RobotWarehouseSmall-v0
    python viz_jumanji.py --small --steps 50 --output my.gif
"""

import argparse
import jax
import jumanji

# RobotWarehouseSmall-v0 is only in JaxMARL; get raw Jumanji env from wrapper
def get_env(use_small: bool):
    if use_small:
        from jaxmarl import make
        wrapped = make("RobotWarehouseSmall-v0")
        return wrapped._jumanji_env
    return jumanji.make("RobotWarehouse-v0")


def main():
    parser = argparse.ArgumentParser(description="Visualize RobotWarehouse rollout")
    parser.add_argument("--small", action="store_true", help="Use RobotWarehouseSmall-v0 (1 agent, small map)")
    parser.add_argument("--steps", type=int, default=30, help="Number of steps to animate")
    parser.add_argument("--output", type=str, default=None, help="Output GIF path (default: robot_warehouse_rollout.gif or robot_warehouse_small_rollout.gif)")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    args = parser.parse_args()

    env = get_env(args.small)
    default_output = "robot_warehouse_small_rollout.gif" if args.small else "robot_warehouse_rollout.gif"
    save_path = args.output or default_output

    key = jax.random.PRNGKey(args.seed)
    state, timestep = env.reset(key)

    states = [state]
    for _ in range(args.steps):
        key, key_step = jax.random.split(key)
        action = jax.random.randint(key_step, (env.num_agents,), 0, 5)
        state, timestep = env.step(state, action)
        breakpoint()
        states.append(state)

    anim = env.animate(states, interval=200, save_path=save_path)
    print(f"Saved {len(states)} frames to {save_path} (env: {'RobotWarehouseSmall-v0' if args.small else 'RobotWarehouse-v0'})")


if __name__ == "__main__":
    main()
