import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.training.train_state import TrainState
from safetensors.flax import load_file
from jaxmarl.environments.overcooked import Overcooked
from jaxmarl.environments.overcooked_v2 import OvercookedV2
from jaxmarl.viz.overcooked_visualizer import OvercookedVisualizer
from jaxmarl.viz.overcooked_v2_visualizer import OvercookedV2Visualizer
from jaxmarl.environments.overcooked.layouts import overcooked_layouts  
from jaxmarl.environments.overcooked_v2.layouts import overcooked_v2_layouts
from flax.linen.initializers import constant, orthogonal

import numpy as np
from functools import partial

class CNN(nn.Module):
    activation: str = "relu"
    num_features: int = 64
    num_agents: int = 1 # scale features by num agents

    @nn.compact
    def __call__(self, x):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh
        # x = nn.Conv(
            # features=32,
            # kernel_size=(5, 5),
        # )(x)
        # x = activation(x)
        x = nn.Conv(
            features=32 * self.num_agents,
            kernel_size=(3, 3),
        )(x)
        x = activation(x)
        x = nn.Conv(
            features=32 * self.num_agents,
            kernel_size=(3, 3),
        )(x)
        x = activation(x)
        x = x.reshape((x.shape[0], -1))  # Flatten 
        # x = x.reshape((x.shape[0], x.shape[1], -1))
        x = nn.Dense(
            # features=self.num_features * 10
            features=self.num_features * self.num_agents
            # features=64
        )(x)
        x = activation(x)

        return x


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

class CNNRNNQNetwork(nn.Module):
    # homogenous agent for parameters sharing, assumes all agents have same obs and action dim
    action_dim: int = 6
    hidden_dim: int = 512
    init_scale: float = 1.0

    @nn.compact
    def __call__(self, hidden, obs, dones):

        time_steps, batch_size = obs.shape[:2]
	    # obs shape (NUM_STEPS, BUFFER_BATCH_SIZE, H, W, C)
        obs_reshaped = obs.reshape(-1, *obs.shape[2:])
	    # obs shape (NUM_STEPS * BUFFER_BATCH_SZE, H, W, C)

        embedding = CNN(num_features=self.hidden_dim, num_agents=1)(obs_reshaped)
	    # embedding shape (NUM_STEPS * BUFFER_BATCH_SIZE, EMBEDDING_SIZE)
        embedding = embedding.reshape(time_steps, batch_size, -1)
	    # embedding shape (NUM_STEPS, BUFFER_BATCH_SIZE, EMBEDDING_SIZE)

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
    agent_network = CNNRNNQNetwork(
        action_dim=action_dim,
        hidden_dim=hidden_dim,
    )
    
    # Load the saved parameters
    flat_params = load_file(model_path)
    
    # Convert flat parameter structure to nested structure
    nested_params = construct_nested_dict(flat_params)
    
    return agent_network, nested_params

map_layouts = ['asymm_advantages',]
alg_names = ['iql_cnn_rnn', 'vdn_cnn_rnn', 'adqn_cnn_rnn']

for map_layout in map_layouts: 

    env = OvercookedV2(layout=overcooked_v2_layouts[map_layout], agent_view_size=2)
    viz =  OvercookedV2Visualizer()

    save_path = "models"
    env_name = f"overcooked_v2_{map_layout}"

    for alg_name in alg_names:

        for seed in range(5):
            vmap_index = 0

            model_path = f"{save_path}/{env_name}/{alg_name}_{env_name}_seed{seed}_vmap{vmap_index}.safetensors"

            hidden_dim = 512
            agent_network, loaded_params = load_model(model_path,)

            if alg_name.split("_")[0] == "adqn": 
                params = loaded_params['agent']
            else:
                params = loaded_params

            rng = jax.random.PRNGKey(seed)
            rng, rng_reset = jax.random.split(rng)

            state_list = []

            episode = 0
            while episode < 2:
                obs, state = env.reset(rng_reset)
                state_list.append(state)

                done = False
                all_dones = {agent: jnp.zeros((1,), dtype=bool) for agent in env.agents}
                all_dones["__all__"] = jnp.zeros((1,), dtype=bool)
                hidden = ScannedRNN.initialize_carry(hidden_dim, len(env.agents), 1)

                while not done:
                    rng, rng_step = jax.random.split(rng)
                    batched_obs = batchify(obs)
                    batched_obs = batched_obs[:, jnp.newaxis, jnp.newaxis]  # Add time and batch dimensions

                    dones = jnp.array([all_dones[agent] for agent in env.agents])[:, jnp.newaxis]
                    
                    hidden, q_vals = jax.vmap(agent_network.apply, in_axes=(None, 0, 0, 0))(
                        params, 
                        hidden, 
                        batched_obs, 
                        dones
                    )
                    q_vals = q_vals.squeeze(1)
                    actions = jnp.argmax(q_vals, axis=-1).squeeze()
                    actions = unbatchify(actions,)

                    obs, state, rewards, done, info = env.step(rng_step, state, actions)
                    done = done['__all__']
                    state_list.append(state)

                rng, rng_reset = jax.random.split(rng)
                episode += 1

            state_seq = jax.tree.map(lambda *xs: jnp.stack(xs), *state_list)
            viz.animate(state_seq, agent_view_size=env.agent_view_size, filename=f'{env_name}_{alg_name}_seed_{seed}.gif')