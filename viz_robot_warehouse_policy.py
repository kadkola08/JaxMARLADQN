"""
Visualize a trained IQL (or other Q-learning) policy on Jumanji RobotWarehouse.

Loads a trained RNN Q-network from the models/ dir, runs rollouts with greedy actions,
and saves a GIF using Jumanji's built-in RobotWarehouseViewer.

Usage:
    python viz_robot_warehouse_policy.py
    python viz_robot_warehouse_policy.py --models_dir models --seed 0 --episodes 2 --max_steps 100
"""

import argparse
import jax
import jax.numpy as jnp
import numpy as np
import yaml
from functools import partial

import flax.linen as nn
from flax.linen.initializers import constant, orthogonal
from safetensors.flax import load_file

from jaxmarl import make
from jaxmarl.wrappers.baselines import LogWrapper, CTRolloutManager


# --- Network (must match iql_rnn.py) ---
class ScannedRNN(nn.Module):
    @partial(
        nn.scan,
        variable_broadcast="params",
        in_axes=0,
        out_axes=0,
        split_rngs={"params": False},
    )
    @nn.compact
    def __call__(self, carry, x):
        rnn_state = carry
        ins, resets = x
        hidden_size = ins.shape[-1]
        rnn_state = jnp.where(
            resets[:, np.newaxis],
            self.initialize_carry(hidden_size, *ins.shape[:-1]),
            rnn_state,
        )
        new_rnn_state, y = nn.GRUCell(hidden_size)(rnn_state, ins)
        return new_rnn_state, y

    @staticmethod
    def initialize_carry(hidden_size, *batch_size):
        return nn.GRUCell(hidden_size, parent=None).initialize_carry(
            jax.random.PRNGKey(0), (*batch_size, hidden_size)
        )


class RNNQNetwork(nn.Module):
    action_dim: int
    hidden_dim: int
    init_scale: float = 1.0

    @nn.compact
    def __call__(self, hidden, obs, dones):
        embedding = nn.Dense(
            self.hidden_dim,
            kernel_init=orthogonal(self.init_scale),
            bias_init=constant(0.0),
        )(obs)
        embedding = nn.relu(embedding)
        rnn_in = (embedding, dones)
        hidden, embedding = ScannedRNN()(hidden, rnn_in)
        q_vals = nn.Dense(
            self.action_dim,
            kernel_init=orthogonal(self.init_scale),
            bias_init=constant(0.0),
        )(embedding)
        return hidden, q_vals


def construct_nested_dict(flat_dict):
    """Convert flat keys like 'a,b,c' to nested dict."""
    nested_dict = {}
    for key, value in flat_dict.items():
        path = key.split(",")
        current_dict = nested_dict
        for part in path[:-1]:
            if part not in current_dict:
                current_dict[part] = {}
            current_dict = current_dict[part]
        current_dict[path[-1]] = value
    return nested_dict


def batchify(x: dict, agents):
    return jnp.stack([x[agent] for agent in agents], axis=0)


def unbatchify(x: jnp.ndarray, agents):
    return {agent: x[i] for i, agent in enumerate(agents)}


def get_greedy_actions(q_vals, valid_actions):
    unavail_actions = 1 - valid_actions
    q_vals = q_vals - (unavail_actions * 1e10)
    return jnp.argmax(q_vals, axis=-1)


def take_first_batch(tree):
    """Take first batch index from a pytree (batch_size=1)."""
    return jax.tree.map(lambda x: x[0] if hasattr(x, "ndim") and x.ndim > 0 else x, tree)


def main():
    parser = argparse.ArgumentParser(description="Visualize trained IQL policy on RobotWarehouse")
    parser.add_argument("--models_dir", type=str, default="models", help="Base dir for saved models")
    parser.add_argument("--env_name", type=str, default="RobotWarehouse-v0", help="Env name (for path)")
    parser.add_argument("--alg_name", type=str, default="iql_rnn", help="Algorithm name (for path)")
    parser.add_argument("--seed", type=int, default=0, help="Config/checkpoint seed")
    parser.add_argument("--vmap_index", type=int, default=0, help="Which vmap checkpoint to load")
    parser.add_argument("--episodes", type=int, default=1, help="Number of episodes to animate")
    parser.add_argument("--max_steps", type=int, default=80, help="Max steps per episode")
    parser.add_argument("--output", type=str, default="robot_warehouse_policy.gif", help="Output GIF path")
    parser.add_argument("--interval", type=int, default=200, help="Frame interval (ms) for GIF")
    args = parser.parse_args()

    model_path = f"{args.models_dir}/{args.env_name}/{args.alg_name}_{args.env_name}_seed{args.seed}_vmap{args.vmap_index}.safetensors"
    config_path = f"{args.models_dir}/{args.env_name}/{args.alg_name}_{args.env_name}_seed{args.seed}_config.yaml"

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    env_kwargs = config.get("ENV_KWARGS", {}) or {}
    hidden_dim = config.get("HIDDEN_SIZE", 64)

    # JaxMARL wrapped env (same as training: LogWrapper + CTRolloutManager)
    base_env = make(args.env_name, **env_kwargs)
    env = make(args.env_name, **env_kwargs)
    env = LogWrapper(env)
    env = CTRolloutManager(env, batch_size=1)

    action_dim = env.action_spaces["agent_0"].n
    network = RNNQNetwork(action_dim=action_dim, hidden_dim=hidden_dim)
    flat_params = load_file(model_path)
    nested_params = construct_nested_dict(flat_params)
    # IQL saves shared params at top level; ADQN/QMIX use nested ["agent"]
    params = nested_params.get("agent", nested_params)

    # Raw Jumanji env for rendering (RobotWarehouseWrapper is base_env's inner when using make)
    # CTRolloutManager -> LogWrapper -> RobotWarehouseWrapper
    jumanji_env = env._env._env._jumanji_env

    rng = jax.random.PRNGKey(config.get("SEED", 0) + 42)
    jumanji_states_all = []
    hidden = ScannedRNN.initialize_carry(hidden_dim, len(env.agents), 1)
    all_dones = {agent: jnp.zeros((1,), dtype=bool) for agent in env.agents}
    all_dones["__all__"] = jnp.zeros((1,), dtype=bool)

    for ep in range(args.episodes):
        rng, rng_reset = jax.random.split(rng)
        obs, state = env.batch_reset(rng_reset)
        single = take_first_batch(state)
        jumanji_states = [single.env_state.jumanji_state]
        done = False
        step = 0
        ep_reward = 0.0

        while not done and step < args.max_steps:
            rng, rng_step = jax.random.split(rng)

            batched_obs = batchify(obs, env.agents)[:, np.newaxis]  # (n_agents, 1, obs_dim)
            dones_arr = jnp.array([all_dones[agent] for agent in env.agents])[:, np.newaxis]
            hidden, q_vals = jax.vmap(network.apply, in_axes=(None, 0, 0, 0))(
                params, hidden, batched_obs, dones_arr
            )
            q_vals = q_vals.squeeze(1)
            valid_actions = env.get_valid_actions(state)
            actions = get_greedy_actions(q_vals, batchify(valid_actions, env.agents))
            action_dict = unbatchify(actions, env.agents)

            obs, state, rewards, all_dones, info = env.batch_step(rng_step, state, action_dict)
            single = take_first_batch(state)
            jumanji_states.append(single.env_state.jumanji_state)
            done = all_dones["__all__"][0]
            step += 1
            ep_reward += float(rewards["__all__"][0])

        jumanji_states_all.extend(jumanji_states)
        print(f"Episode {ep + 1}: steps={step}, return={ep_reward:.1f}")

        if ep < args.episodes - 1:
            hidden = ScannedRNN.initialize_carry(hidden_dim, len(env.agents), 1)
            all_dones = {agent: jnp.zeros((1,), dtype=bool) for agent in env.agents}
            all_dones["__all__"] = jnp.zeros((1,), dtype=bool)

    # Animate with Jumanji's viewer
    anim = jumanji_env.animate(
        jumanji_states_all,
        interval=args.interval,
        save_path=args.output,
    )
    print(f"Saved policy animation to {args.output} ({len(jumanji_states_all)} frames)")


if __name__ == "__main__":
    main()
