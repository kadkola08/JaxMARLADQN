import jax 
from jaxmarl import make
from jaxmarl.environments.mpe import MPEVisualizer
import jax.numpy as jnp
import flax.linen as nn
from safetensors.flax import load_file
from flax.linen.initializers import constant, orthogonal

import numpy as np
from functools import partial
import yaml

from jaxmarl.wrappers.baselines import (
    SMAXLogWrapper,
    MPELogWrapper,
    LogWrapper,
    CTRolloutManager,
)

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
        """Applies the module."""
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
        # Use a dummy key since the default state init fn is just zeros.
        return nn.GRUCell(hidden_size, parent=None).initialize_carry(
            jax.random.PRNGKey(0), (*batch_size, hidden_size)
        )


class RNNQNetwork(nn.Module):
    # homogenous agent for parameters sharing, assumes all agents have same obs and action dim
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
    """Convert a dictionary with flat keys like 'a,b,c' to a nested dict."""
    nested_dict = {}
    
    for key, value in flat_dict.items():
        # Split the key by comma
        path = key.split(',')
        
        # Navigate to the appropriate nested dictionary
        current_dict = nested_dict
        for part in path[:-1]:
            if part not in current_dict:
                current_dict[part] = {}
            current_dict = current_dict[part]
        
        # Set the value at the final location
        current_dict[path[-1]] = value
    
    return nested_dict


def load_model(model_path, action_dim=6, hidden_dim=512,):
    # Create network instances
    agent_network = RNNQNetwork(
        action_dim=action_dim,
        hidden_dim=hidden_dim,
    )
    
    # Load the saved parameters
    flat_params = load_file(model_path)
    
    # Convert flat parameter structure to nested structure
    nested_params = construct_nested_dict(flat_params)
    
    return agent_network, nested_params

def batchify(x: dict):
    return jnp.stack([x[agent] for agent in env.agents], axis=0)

def unbatchify(x: jnp.ndarray):
    return {agent: x[i] for i, agent in enumerate(env.agents)}

def squeeze_state_arrays(state_seq):
    # Apply squeeze to all attributes containing arrays
    return [(key, 
            type(state)(
                state=type(state)(
                    p_pos=state.p_pos.squeeze(),
                    p_vel=state.p_vel.squeeze(),
                    c=state.c.squeeze(),
                    done=state.done.squeeze(),
                    step=state.step.squeeze(),
                    goal=state.goal
                ),
            ),
            actions) 
            for key, state, actions in state_seq]

def squeeze_state(state):
    return type(state)(
            p_pos=state.p_pos.squeeze(0),
            p_vel=state.p_vel.squeeze(0),
            c=state.c.squeeze(0),
            done=state.done.squeeze(0),
            step=state.step.squeeze(0),
            goal=state.goal
        )

def get_greedy_actions(q_vals, valid_actions):
    unavail_actions = 1 - valid_actions
    q_vals = q_vals - (unavail_actions * 1e10)
    return jnp.argmax(q_vals, axis=-1)

models_dir = "models"
env_name = "MPE_simple_spread_v3"
alg_name = "qmix_rnn"
seed = 0
vmap_index = 0
# date = "_2025-08-15"
date = ""

model_path = f"{models_dir}/{env_name}/{alg_name}_{env_name}_seed{seed}_vmap{vmap_index}{date}.safetensors"
config_path = f"{models_dir}/{env_name}/{alg_name}_{env_name}_seed{seed}_config{date}.yaml"

with open(config_path, 'r') as file:
    config = yaml.safe_load(file)


# Parameters + random keys
max_steps = 25
key = jax.random.PRNGKey(0)
key, key_r, key_a = jax.random.split(key, 3)

# Instantiate environment
env_ = make("MPE_simple_spread_v3")
obs, state_ = env_.reset(key_r)

env = CTRolloutManager(env_, batch_size=1)
obs, state = env.batch_reset(key_r)

hidden_dim = config['HIDDEN_SIZE']
action_dim = env.action_spaces['agent_0'].n
agent_network, loaded_params = load_model(model_path, action_dim=action_dim, hidden_dim=hidden_dim)


if alg_name.split("_")[0] in ["adqn", "qmix"]: 
    params = loaded_params['agent']
else:
    params = loaded_params

rng = jax.random.PRNGKey(seed)
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
        
        batched_obs = batchify(obs, )
        batched_obs = batched_obs[:, jnp.newaxis]  # Add time and batch dimensions
        
        dones = jnp.array([all_dones[agent] for agent in env.agents])[:, jnp.newaxis]
        
        hidden, q_vals = jax.vmap(agent_network.apply, in_axes=(None, 0, 0, 0))(
            params,
            hidden, 
            batched_obs, 
            dones
        )
        
        q_vals = q_vals.squeeze(1)  # Remove time dimension
        valid_actions = env.get_valid_actions(state)
        # actions = jnp.argmax(q_vals, axis=-1)
        actions = get_greedy_actions(q_vals, batchify(valid_actions,))
        # breakpoint()
        
        action_dict = unbatchify(actions, )
        
        # state_list.append((rng_step, state, unbatchify(actions.squeeze(1))))
        state_list.append(squeeze_state(state))

        obs, state, rewards, all_dones, info = env.batch_step(rng_step, state, action_dict)
        
        done = all_dones['__all__'][0]
        # state_list.append(state)
        
        step_count += 1
        print(f"Step {step_count} - Actions: {action_dict}, Rewards: {rewards}")
    
    episode_count += 1
    if episode_count < num_episodes:
        print(f"Episode {episode_count} completed in {step_count} steps. Starting next episode.")
        rng, rng_reset = jax.random.split(rng)
        obs, state = env.batch_reset(rng_reset)
        # breakpoint()
        state_list.append(squeeze_state(state))
        done = False
        all_dones = {agent: jnp.zeros((1,), dtype=bool) for agent in env.agents}
        all_dones["__all__"] = jnp.zeros((1,), dtype=bool)
        hidden = ScannedRNN.initialize_carry(hidden_dim, len(env.agents), 1)
    else:
        print(f"All {num_episodes} episodes completed.")

# state_seq = squeeze_state_arrays(state_list)
viz = MPEVisualizer(env, state_list)
viz.animate(save_fname=f"mpe_{alg_name}.gif", view=True)  # can also save the animiation as a .gif file with save_fname="mpe.gif"


# Sample random actions
# key_a = jax.random.split(key_a, env.num_agents)
# actions = {agent: env.action_space(agent).sample(key_a[i]) for i, agent in enumerate(env.agents)}
# 
# state_seq = []
# for _ in range(max_steps):
#     state_seq.append(state)
#     # Iterate random keys and sample actions
#     key, key_s, key_a = jax.random.split(key, 3)
#     key_a = jax.random.split(key_a, env.num_agents)
#     actions = {agent: env.action_space(agent).sample(key_a[i]) for i, agent in enumerate(env.agents)}
# 
#     # Step environment
#     obs, state, rewards, dones, infos = env.step(key_s, state, actions)
# 
# # state_seq is a list of the jax env states passed to the step function
# # i.e. [state_t0, state_t1, ...]
# viz = MPEVisualizer(env, state_seq)
# viz.animate(save_fname="mpe_random.gif", view=True)  # can also save the animiation as a .gif file with save_fname="mpe.gif"