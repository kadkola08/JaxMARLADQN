import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.training.train_state import TrainState
from safetensors.flax import load_file
from flax.linen.initializers import constant, orthogonal
from jaxmarl import make
from jaxmarl.environments.smax import SMAX
from jaxmarl.viz.visualizer import SMAXVisualizer
from jaxmarl.environments.smax import map_name_to_scenario
from jaxmarl.wrappers.baselines import (
    SMAXLogWrapper,
    MPELogWrapper,
    LogWrapper,
    CTRolloutManager,
)

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
map_name = "smacv2_5_units_allies_outside"
env_name = "HeuristicEnemySMAX"
alg_name = "adqn_rnn2"
seed = 0
vmap_index = 0
date = ""

model_path = f"{models_dir}/{env_name}_{map_name}/{alg_name}_{env_name}_{map_name}_seed{seed}_vmap{vmap_index}{date}.safetensors"
config_path = f"{models_dir}/{env_name}_{map_name}/{alg_name}_{env_name}_{map_name}_seed{seed}_config{date}.yaml"

with open(config_path, 'r') as file:
    config = yaml.safe_load(file)

# Environment setup - create SMAX environment with the appropriate scenario
scenario = map_name_to_scenario(map_name)
env_kwargs = {
    "scenario" : scenario,
    "walls_cause_death" : True,
    "see_enemy_actions" : True,
    "attack_mode" : "closest",
}

# Create the base environment without CTRolloutManager for visualization
base_env = make(env_name, **env_kwargs)

# Create wrapped environment for running the policy
env = make(env_name, **env_kwargs)
env = CTRolloutManager(env, batch_size=1)

hidden_dim = config['HIDDEN_SIZE']
action_dim = env.action_spaces['ally_0'].n
agent_network, loaded_params = load_model(model_path, action_dim=action_dim, hidden_dim=hidden_dim)

if alg_name.split("_")[0] == "adqn": 
    params = loaded_params['agent']
else:
    params = loaded_params

env_seed = 322
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
        
        batched_obs = batchify(obs)
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
        actions = get_greedy_actions(q_vals, batchify(valid_actions))
        
        # Create action dict keeping batch dimension for env.batch_step
        action_dict = unbatchify(actions)
        
        # For visualization, create action dict without batch dimension
        viz_action_dict = {agent: actions[i, 0] for i, agent in enumerate(env.agents)}
        
        # Store state for visualization
        # Extract the unbatched state from the batched state for visualization
        viz_state = jax.tree_map(lambda x: x[0] if x.ndim > 0 else x, state)
        state_list.append((rng_step, viz_state, viz_action_dict))

        obs, state, rewards, all_dones, info = env.batch_step(rng_step, state, action_dict)
        
        done = all_dones['__all__'][0]
        
        step_count += 1
        # print(f"Step {step_count} - Rewards: {rewards}")
    
    episode_count += 1
    if episode_count < num_episodes:
        print(f"Episode {episode_count} completed in {step_count} steps. Starting next episode.")
        rng, rng_reset = jax.random.split(rng)
        obs, state = env.batch_reset(rng_reset)
        done = False
        all_dones = {agent: jnp.zeros((1,), dtype=bool) for agent in env.agents}
        all_dones["__all__"] = jnp.zeros((1,), dtype=bool)
        hidden = ScannedRNN.initialize_carry(hidden_dim, len(env.agents), 1)
    else:
        print(f"All {num_episodes} episodes completed.")

# Create visualizer with the base environment (not wrapped)
viz = SMAXVisualizer(base_env, state_list)  
viz.animate(save_fname=f'{alg_name}_{map_name}_animation_envseed_{env_seed}policy{seed}_2.gif', view=True)