import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.training.train_state import TrainState
from safetensors.flax import load_file
from jaxmarl.environments.smax import SMAX
from jaxmarl.viz.smax_visualizer import SMAXVisualizer
from jaxmarl.environments.smax import map_name_to_scenario
from functools import partial
import numpy as np

# Import neural network components from qmix_rnn.py
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
            kernel_init=nn.initializers.orthogonal(self.init_scale),
            bias_init=nn.initializers.constant(0.0),
        )(obs)
        embedding = nn.relu(embedding)

        rnn_in = (embedding, dones)
        hidden, embedding = ScannedRNN()(hidden, rnn_in)

        q_vals = nn.Dense(
            self.action_dim,
            kernel_init=nn.initializers.orthogonal(self.init_scale),
            bias_init=nn.initializers.constant(0.0),
        )(embedding)

        return hidden, q_vals


class HyperNetwork(nn.Module):
    """HyperNetwork for generating weights of QMix' mixing network."""

    hidden_dim: int
    output_dim: int
    init_scale: float

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(
            self.hidden_dim,
            kernel_init=nn.initializers.orthogonal(self.init_scale),
            bias_init=nn.initializers.constant(0.0),
        )(x)
        x = nn.relu(x)
        x = nn.Dense(
            self.output_dim,
            kernel_init=nn.initializers.orthogonal(self.init_scale),
            bias_init=nn.initializers.constant(0.0),
        )(x)
        return x


class MixingNetwork(nn.Module):
    """
    Mixing network for projecting individual agent Q-values into Q_tot. Follows the original QMix implementation.
    """

    embedding_dim: int
    hypernet_hidden_dim: int
    init_scale: float

    @nn.compact
    def __call__(self, q_vals, states):
        n_agents, time_steps, batch_size = q_vals.shape
        q_vals = jnp.transpose(q_vals, (1, 2, 0))  # (time_steps, batch_size, n_agents)

        # hypernetwork
        w_1 = HyperNetwork(
            hidden_dim=self.hypernet_hidden_dim,
            output_dim=self.embedding_dim * n_agents,
            init_scale=self.init_scale,
        )(states)
        b_1 = nn.Dense(
            self.embedding_dim,
            kernel_init=nn.initializers.orthogonal(self.init_scale),
            bias_init=nn.initializers.constant(0.0),
        )(states)
        w_2 = HyperNetwork(
            hidden_dim=self.hypernet_hidden_dim,
            output_dim=self.embedding_dim,
            init_scale=self.init_scale,
        )(states)
        b_2 = HyperNetwork(
            hidden_dim=self.embedding_dim, output_dim=1, init_scale=self.init_scale
        )(states)

        # monotonicity and reshaping
        w_1 = jnp.abs(w_1.reshape(time_steps, batch_size, n_agents, self.embedding_dim))
        b_1 = b_1.reshape(time_steps, batch_size, 1, self.embedding_dim)
        w_2 = jnp.abs(w_2.reshape(time_steps, batch_size, self.embedding_dim, 1))
        b_2 = b_2.reshape(time_steps, batch_size, 1, 1)

        # mix
        hidden = nn.elu(jnp.matmul(q_vals[:, :, None, :], w_1) + b_1)
        q_tot = jnp.matmul(hidden, w_2) + b_2

        return q_tot.squeeze()  # (time_steps, batch_size)


# Helper function to construct nested dictionary from flat dictionary
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


# Function to load QMIX model parameters from safetensors
def load_model(model_path, obs_size, action_dim, hidden_dim=64, mixer_embedding_dim=32, mixer_hypernet_hidden_dim=64):
    # Create network instances
    agent_network = RNNQNetwork(
        action_dim=action_dim,
        hidden_dim=hidden_dim,
    )
    
    mixer = MixingNetwork(
        embedding_dim=mixer_embedding_dim,
        hypernet_hidden_dim=mixer_hypernet_hidden_dim,
        init_scale=1.0,
    )
    
    # Load the saved parameters
    flat_params = load_file(model_path)
    
    # Convert flat parameter structure to nested structure
    nested_params = construct_nested_dict(flat_params)
    
    return agent_network, mixer, nested_params


# Helper functions for batching and unbatching observations/actions
def batchify(x: dict, agents):
    return jnp.stack([x[agent] for agent in agents], axis=0)

def unbatchify(x: jnp.ndarray, agents):
    return {agent: x[i] for i, agent in enumerate(agents)}

if __name__ == "__main__":
    models_dir = "models"
    map_name = "2s3z"
    env_name = "HeuristicEnemySMAX"
    alg_name = "qmix_rnn"
    seed = 0
    vmap_index = 0

    model_path = f"{models_dir}/{env_name}_{map_name}/{alg_name}_{env_name}_{map_name}_seed{seed}_vmap{vmap_index}.safetensors"
    
    # Environment setup - create SMAX environment with the appropriate scenario
    scenario = map_name_to_scenario(map_name)
    env = SMAX(scenario=scenario)
    viz = SMAXVisualizer(env, [])  # Initialize with empty state sequence
    
    # Define config parameters for the model
    hidden_dim = 64
    mixer_embedding_dim = 32
    mixer_hypernet_hidden_dim = 64
    breakpoint()
    
    # Define your environment's observation shape and action dimension
    obs_size = env.observation_space().shape[-1]  # Last dimension is the feature size
    action_dim = env.action_space().n  # Number of discrete actions in SMAX
    
    # Load the model
    agent_network, mixer_network, loaded_params = load_model(
        model_path, 
        obs_size, 
        action_dim, 
        hidden_dim=hidden_dim,
        mixer_embedding_dim=mixer_embedding_dim,
        mixer_hypernet_hidden_dim=mixer_hypernet_hidden_dim
    )

    # Create initial RNG
    rng = jax.random.PRNGKey(0)
    rng, rng_reset = jax.random.split(rng)

    # Reset environment
    obs, state = env.reset(rng_reset)
    state_list = [state]
    
    # Check if episode is done
    done = False
    all_dones = {agent: jnp.zeros((1,), dtype=bool) for agent in env.agents}
    all_dones["__all__"] = jnp.zeros((1,), dtype=bool)

    # Initialize RNN hidden state for all agents
    hidden = ScannedRNN.initialize_carry(hidden_dim, len(env.agents), 1)
    
    # Define number of episodes to run
    num_episodes = 1
    episode_count = 0
    max_steps = 100  # Maximum steps per episode to prevent infinite loops

    # Run the environment using the loaded model
    while episode_count < num_episodes:
        step_count = 0
        
        while not done and step_count < max_steps:
            rng, rng_step = jax.random.split(rng)
            
            # Batchify observations and prepare for network
            batched_obs = batchify(obs, env.agents)
            batched_obs = batched_obs[:, jnp.newaxis, jnp.newaxis]  # Add time and batch dimensions
            
            # Prepare dones for RNN processing
            dones = jnp.array([all_dones[agent] for agent in env.agents])[:, jnp.newaxis]
            
            # Get Q-values from the agent network
            hidden, q_vals = jax.vmap(agent_network.apply, in_axes=(None, 0, 0, 0))(
                loaded_params['agent'], 
                hidden, 
                batched_obs, 
                dones
            )
            
            # Remove extra dimensions and get actions
            q_vals = q_vals.squeeze(1)  # Remove time dimension
            actions = jnp.argmax(q_vals, axis=-1)
            
            # Convert actions back to dictionary format
            action_dict = unbatchify(actions, env.agents)
            
            # Take a step in the environment
            obs, state, rewards, all_dones, info = env.step(rng_step, state, action_dict)
            breakpoint()
            
            # Check if episode is done
            done = all_dones['__all__'][0]
            
            # Store state for visualization
            state_list.append(state)
            
            step_count += 1
            # Print actions and rewards for debugging
            print(f"Step {step_count} - Actions: {action_dict}, Rewards: {rewards}")
        
        # Reset for next episode
        episode_count += 1
        if episode_count < num_episodes:
            print(f"Episode {episode_count} completed in {step_count} steps. Starting next episode.")
            rng, rng_reset = jax.random.split(rng)
            obs, state = env.reset(rng_reset)
            state_list.append(state)
            done = False
            all_dones = {agent: jnp.zeros((1,), dtype=bool) for agent in env.agents}
            all_dones["__all__"] = jnp.zeros((1,), dtype=bool)
            hidden = ScannedRNN.initialize_carry(hidden_dim, len(env.agents), 1)
        else:
            print(f"All {num_episodes} episodes completed.")
    
    # Update the visualizer with the collected states
    viz.state_seq = state_list
    
    # Create animation
    viz.animate(save_fname=f'smax_{map_name}_animation.gif', view=True)
