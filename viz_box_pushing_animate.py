"""
Box Pushing visualizer script.

Runs one or more episodes of BoxPushingSimple with random or custom actions,
collects states, and renders them to a GIF using BoxPushingVisualizer.

Usage:
    python viz_box_pushing_animate.py                    # random actions
    python viz_box_pushing_animate.py --actions_file seq.txt   # your action sequence
    python viz_box_pushing_animate.py --actions "0,0 3,3 3,3"   # inline: step0 (0,0), step1 (3,3), ...
"""

import argparse
import jax
import jax.numpy as jnp
from jaxmarl import make
from jaxmarl.environments.box_pushing import BoxPushingSimple
from jaxmarl.viz.box_pushing_visualizer import BoxPushingVisualizer


def load_actions_from_file(path):
    """Load action sequence from file. One line per step: act0,act1. Lines starting with # are ignored."""
    seq = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip().split("#")[0].strip()
            if not line:
                continue
            parts = line.replace(",", " ").split()
            if len(parts) >= 2:
                seq.append((int(parts[0]), int(parts[1])))
    return seq if seq else None


def parse_actions_inline(s):
    """Parse inline action string, e.g. '0,0 3,3 3,3'. Returns list of (act0, act1) per step."""
    if not s or not s.strip():
        return None
    seq = []
    for token in s.strip().split():
        parts = token.replace(",", " ").split()
        if len(parts) >= 2:
            seq.append((int(parts[0]), int(parts[1])))
    return seq if seq else None


def main():
    parser = argparse.ArgumentParser(description="Visualize Box Pushing episodes")
    parser.add_argument("--seed", type=int, default=0, help="Random seed (for reset when using custom actions)")
    parser.add_argument("--episodes", type=int, default=1, help="Number of episodes (only used for random)")
    parser.add_argument("--max_steps", type=int, default=50, help="Max steps per episode (random mode)")
    parser.add_argument("--output", type=str, default="box_pushing.gif", help="Output GIF filename")
    parser.add_argument("--duration", type=float, default=0.5, help="GIF frame duration (seconds)")
    parser.add_argument(
        "--actions_file",
        type=str,
        default=None,
        help="Path to file with one line per step: act0,act1 (0=stay 1=left 2=right 3=forward)",
    )
    parser.add_argument(
        "--actions",
        type=str,
        default=None,
        help='Inline action sequence, e.g. "0,0 3,3 3,3" for steps (0,0) then (3,3) then (3,3)',
    )
    args = parser.parse_args()

    action_sequence = None
    if args.actions_file:
        action_sequence = load_actions_from_file(args.actions_file)
        if not action_sequence:
            raise SystemExit(f"Empty or invalid actions file: {args.actions_file}")
    elif args.actions:
        action_sequence = parse_actions_inline(args.actions)
        if not action_sequence:
            raise SystemExit("Invalid --actions string (need e.g. '0,0 3,3')")

    env = make("BoxPushingSimple")
    viz = BoxPushingVisualizer()

    rng = jax.random.PRNGKey(args.seed)
    all_states = []

    if action_sequence is not None:
        # Single episode with your action sequence
        rng, rng_reset = jax.random.split(rng)
        obs, state = env.reset(rng_reset)
        state_list = [state]
        for act0, act1 in action_sequence:
            actions = {env.agents[0]: jnp.int32(act0), env.agents[1]: jnp.int32(act1)}
            rng, rng_step = jax.random.split(rng)
            obs, state, rewards, dones, info = env.step(rng_step, state, actions)
            state_list.append(state)
            print(rewards)
        all_states = state_list
    else:
        # Random actions
        for ep in range(args.episodes):
            rng, rng_reset = jax.random.split(rng)
            obs, state = env.reset(rng_reset)
            state_list = [state]
            done = False
            step = 0
            while not done and step < args.max_steps:
                rng, rng_step = jax.random.split(rng)
                key_act = jax.random.split(rng_step, env.num_agents)
                actions = {
                    agent: env.action_space(agent).sample(key_act[i])
                    for i, agent in enumerate(env.agents)
                }
                obs, state, rewards, dones, info = env.step(rng_step, state, actions)
                state_list.append(state)
                done = dones["__all__"]
                step += 1

            all_states.extend(state_list)

    # If multiple episodes, save one GIF with all states; use base name + episode for per-episode files
    if args.episodes == 1:
        out_path = args.output
    else:
        base = args.output.rsplit(".", 1)[0] if "." in args.output else args.output
        ext = "." + args.output.rsplit(".", 1)[1] if "." in args.output else ".gif"
        out_path = f"{base}_all{ext}"

    viz.animate(
        all_states,
        grid_size=env.grid_size,
        num_small_boxes=env.num_small_boxes,
        num_large_boxes=env.num_large_boxes,
        filename=out_path,
        duration=args.duration,
    )
    print(f"Saved animation to {out_path} ({len(all_states)} frames)")


if __name__ == "__main__":
    main()