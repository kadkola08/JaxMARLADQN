"""
Visualize Box Pushing policy rollouts, mirroring viz_smax2 structure.
Loads a trained RNN Q-network, runs episodes, and animates with BoxPushingVisualizer.
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.linen.initializers import constant, orthogonal
from safetensors.flax import load_file
from jaxmarl import make
from jaxmarl.viz.box_pushing_visualizer import BoxPushingVisualizer
from jaxmarl.wrappers.baselines import CTRolloutManager

import numpy as np
from functools import partial
import yaml


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


def batchify(x: dict, agents):
    return jnp.stack([x[agent] for agent in agents], axis=0)


def unbatchify(x: jnp.ndarray, agents):
    return {agent: x[i] for i, agent in enumerate(agents)}


def construct_nested_dict(flat_dict):
    """Convert a dictionary with flat keys like 'a,b,c' to a nested dict."""
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


def load_model(model_path, action_dim=4, hidden_dim=128):
    agent_network = RNNQNetwork(
        action_dim=action_dim,
        hidden_dim=hidden_dim,
    )
    flat_params = load_file(model_path)
    nested_params = construct_nested_dict(flat_params)
    return agent_network, nested_params


def get_greedy_actions(q_vals, valid_actions):
    unavail_actions = 1 - valid_actions
    q_vals = q_vals - (unavail_actions * 1e10)
    return jnp.argmax(q_vals, axis=-1)


# --- Config (adjust paths / alg to match your runs) ---
models_dir = "models"
env_name = "BoxPushingSimple"
alg_name = "adqn_rnn2"
seed = 0
vmap_index = 0
date = ""

model_path = f"{models_dir}/{env_name}/{alg_name}_{env_name}_seed{seed}_vmap{vmap_index}{date}.safetensors"
config_path = f"{models_dir}/{env_name}/{alg_name}_{env_name}_seed{seed}_config{date}.yaml"

with open(config_path, "r") as file:
    config = yaml.safe_load(file)

env_kwargs = config.get("ENV_KWARGS", {})
if not env_kwargs:
    env_kwargs = {}

# Base env (no wrapper) for grid/state info
base_env = make(env_name, **env_kwargs)

# Wrapped env for rollout
env = make(env_name, **env_kwargs)
env = CTRolloutManager(env, batch_size=1)

hidden_dim = config.get("HIDDEN_SIZE", 128)
action_dim = env.action_spaces["agent_0"].n
agent_network, loaded_params = load_model(
    model_path, action_dim=action_dim, hidden_dim=hidden_dim
)

if alg_name.split("_")[0] == "adqn" or alg_name.split("_")[0] == "qmix":
    params = loaded_params["agent"]
else:
    params = loaded_params

env_seed = 0
rng = jax.random.PRNGKey(env_seed)
rng, rng_reset = jax.random.split(rng)

obs, state = env.batch_reset(rng_reset)
state_list = []
done = False
all_dones = {agent: jnp.zeros((1,), dtype=bool) for agent in env.agents}
all_dones["__all__"] = jnp.zeros((1,), dtype=bool)

hidden = ScannedRNN.initialize_carry(hidden_dim, len(env.agents), 1)

num_episodes = 5
episode_count = 0
max_steps = 128

while episode_count < num_episodes:
    step_count = 0

    while not done and step_count < max_steps:
        rng, rng_step = jax.random.split(rng)

        batched_obs = batchify(obs, env.agents)
        batched_obs = batched_obs[:, jnp.newaxis]

        dones = jnp.array([all_dones[agent] for agent in env.agents])[:, jnp.newaxis]

        hidden, q_vals = jax.vmap(agent_network.apply, in_axes=(None, 0, 0, 0))(
            params,
            hidden,
            batched_obs,
            dones,
        )

        q_vals = q_vals.squeeze(1)
        valid_actions = env.get_valid_actions(state)
        actions = get_greedy_actions(q_vals, batchify(valid_actions, env.agents))

        action_dict = unbatchify(actions, env.agents)

        # Unbatch state for visualization (BoxPushingVisualizer expects single env state)
        viz_state = jax.tree.map(lambda x: x[0] if x.ndim > 0 else x, state)
        state_list.append(viz_state)

        obs, state, rewards, all_dones, info = env.batch_step(
            rng_step, state, action_dict
        )
        print(rewards)

        done = all_dones["__all__"][0]
        step_count += 1

    episode_count += 1
    if episode_count < num_episodes:
        print(f"Episode {episode_count} completed in {step_count} steps.")
        rng, rng_reset = jax.random.split(rng)
        obs, state = env.batch_reset(rng_reset)
        done = False
        all_dones = {agent: jnp.zeros((1,), dtype=bool) for agent in env.agents}
        all_dones["__all__"] = jnp.zeros((1,), dtype=bool)
        hidden = ScannedRNN.initialize_carry(hidden_dim, len(env.agents), 1)
    else:
        print(f"All {num_episodes} episodes completed.")

# Animate with Box Pushing visualizer
grid_size = getattr(base_env, "grid_size", 7)
num_small_boxes = getattr(base_env, "num_small_boxes", 2)
num_large_boxes = getattr(base_env, "num_large_boxes", 1)

viz = BoxPushingVisualizer()
save_fname = f"{alg_name}_{env_name}_animation_envseed_{env_seed}_policy{seed}.gif"
viz.animate(
    state_list,
    grid_size=grid_size,
    num_small_boxes=num_small_boxes,
    num_large_boxes=num_large_boxes,
    filename=save_fname,
    duration=0.5,
)
print(f"Saved animation to {save_fname}")
