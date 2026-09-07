import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.training.train_state import TrainState
from safetensors.flax import load_file
from flax.linen.initializers import constant, orthogonal
from jaxmarl import make
from jaxmarl.wrappers.baselines import (
    LogWrapper,
    CTRolloutManager,
)

from jaxmarl.environments.box_pushing import BoxPushing, BoxPushingSimple

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


def batchify(x: dict):
    return jnp.stack([x[agent] for agent in env.agents], axis=0)

def unbatchify(x: jnp.ndarray):
    return {agent: x[i] for i, agent in enumerate(env.agents)}

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

def get_greedy_actions(q_vals, valid_actions):
    unavail_actions = 1 - valid_actions
    q_vals = q_vals - (unavail_actions * 1e10)
    return jnp.argmax(q_vals, axis=-1)

models_dir = "models"
env_name = "BoxPushingSimple"
alg_name = "iql_rnn"
seed = 0
vmap_index = 0
date = ""

model_path = f"{models_dir}/{env_name}/{alg_name}_{env_name}_seed{seed}_vmap{vmap_index}{date}.safetensors"
config_path = f"{models_dir}/{env_name}/{alg_name}_{env_name}_seed{seed}_config{date}.yaml"

with open(config_path, 'r') as file:
    config = yaml.safe_load(file)

env_kwargs = {}

env = make(env_name, **env_kwargs)
env = CTRolloutManager(env, batch_size=1)

env_seed = 322
rng = jax.random.PRNGKey(env_seed)
rng, rng_reset = jax.random.split(rng)

obs, state = env.batch_reset(rng_reset)

breakpoint()

print(env)
