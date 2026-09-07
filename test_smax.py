import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.training.train_state import TrainState
from safetensors.flax import load_file
from jaxmarl.environments.smax import SMAX
from jaxmarl.viz.visualizer import SMAXVisualizer
from jaxmarl.environments.smax import map_name_to_scenario
from jaxmarl import make
from jaxmarl.wrappers.baselines import (
    SMAXLogWrapper,
    MPELogWrapper,
    LogWrapper,
    CTRolloutManager,
)
from functools import partial
import numpy as np
import pickle

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
def load_model(model_path, obs_size, action_dim, hidden_dim=512, mixer_embedding_dim=32, mixer_hypernet_hidden_dim=64):
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

def squeeze_state_arrays(state_seq):
    # Apply squeeze to all attributes containing arrays
    return [(key, 
             type(state)(
                 state=type(state.state)(
                     unit_positions=state.state.unit_positions.squeeze(),
                     unit_alive=state.state.unit_alive.squeeze(),
                     unit_teams=state.state.unit_teams.squeeze(),
                     unit_health=state.state.unit_health.squeeze(),
                     unit_types=state.state.unit_types.squeeze(),
                     unit_weapon_cooldowns=state.state.unit_weapon_cooldowns.squeeze(),
                     prev_movement_actions=state.state.prev_movement_actions.squeeze(),
                     prev_attack_actions=state.state.prev_attack_actions.squeeze(),
                     time=state.state.time.squeeze(),
                     terminal=state.state.terminal.squeeze()
                 ),
                 enemy_policy_state=type(state.enemy_policy_state)(
                     default_target=state.enemy_policy_state.default_target.squeeze(),
                     last_attacked_enemy=state.enemy_policy_state.last_attacked_enemy.squeeze()
                 )
             ),
             actions) 
            for key, state, actions in state_seq]

def get_greedy_actions(q_vals, valid_actions):
    unavail_actions = 1 - valid_actions
    q_vals = q_vals - (unavail_actions * 1e10)
    return jnp.argmax(q_vals, axis=-1)

if __name__ == "__main__":
    models_dir = "models2/models"
    maps = ["2s3z", "3s5z_vs_3s6z", "5m_vs_6m", "3s5z", "6h_vs_8z", "smacv2_5_units", "10m_vs_11m", "smacv2_10_units"]
    map_name = "2s3z"
    env_name = "HeuristicEnemySMAX"
    alg_name = "vanqmix_rnn"
    seeds = [10, 11, 12, 13, 14]
    seeds = [15, 16, 17, 18, 19]
    seed = 10
    vmap_index = 0

    for map_name in maps:
        for seed in seeds:
            print(f"Map : {map_name}, seed : {seed}")

            model_path = f"{models_dir}/{env_name}_{map_name}/{alg_name}_{env_name}_{map_name}_seed{seed}_vmap{vmap_index}.safetensors"
            
            # Environment setup - create SMAX environment with the appropriate scenario
            scenario = map_name_to_scenario(map_name)
            env_kwargs = {
                "scenario" : scenario,
                "walls_cause_death" : True,
                "see_enemy_actions" : True,
                "attack_mode" : "closest"
            }
            # _env = make(env_name, **env_kwargs)
            env = make(env_name, **env_kwargs)
            env_ = SMAX(scenario=scenario, see_enemy_actions=True, walls_cause_death=True,) 
            # env = SMAXLogWrapper(env)
            env = CTRolloutManager(env, batch_size=1)
            
            # Define config parameters for the model
            hidden_dim = 512
            mixer_embedding_dim = 64
            mixer_hypernet_hidden_dim = 256 
            
            # Define your environment's observation shape and action dimension
            obs_size = env.obs_size  # Last dimension is the feature size
            action_dim = env.max_action_space # Number of discrete actions in SMAX
            
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
            # obs, state = env.reset(rng_reset)
            obs, state = env.batch_reset(rng_reset)
            # obs_, state_ = _env.reset(rng_reset)
            state_list = []
            
            # Check if episode is done
            done = False
            all_dones = {agent: jnp.zeros((1,), dtype=bool) for agent in env.agents}
            all_dones["__all__"] = jnp.zeros((1,), dtype=bool)

            # Initialize RNN hidden state for all agents
            hidden = ScannedRNN.initialize_carry(hidden_dim, len(env.agents), 1)
            
            # Define number of episodes to run
            num_episodes = 1
            episode_count = 0
            max_steps = 75  # Maximum steps per episode to prevent infinite loops
            step_count = 0

            # Run the environment using the loaded model
            while episode_count < num_episodes and step_count < max_steps:
                
                while not done and step_count < max_steps:
                    rng, rng_step = jax.random.split(rng)
                    
                    # Batchify observations and prepare for network
                    batched_obs = batchify(obs, env.agents)[:, jnp.newaxis]
                    # batched_obs = batched_obs[:, jnp.newaxis, jnp.newaxis]  # Add time and batch dimensions
                    
                    # Prepare dones for RNN processing
                    dones = batchify(all_dones, env.agents)[:, jnp.newaxis]
                    # dones = jnp.array([all_dones[agent] for agent in env.agents])[:, jnp.newaxis]
                    
                    # Get Q-values from the agent network
                    hidden, q_vals = jax.vmap(agent_network.apply, in_axes=(None, 0, 0, 0))(
                        loaded_params['agent'], 
                        hidden, 
                        batched_obs, 
                        dones
                    )
                    
                    # Remove extra dimensions and get actions
                    q_vals = q_vals.squeeze(1)  # Remove time dimension
                    valid_actions = env.get_valid_actions(state)
                    breakpoint()
                    actions = get_greedy_actions(q_vals, batchify(valid_actions, env.agents))
                    # breakpoint()
                    
                    # Convert actions back to dictionary format
                    action_dict = unbatchify(actions, env.agents)
                    
                    # Take a step in the environment
                    obs, state, rewards, all_dones, info = env.batch_step(rng_step, state, action_dict)
                    
                    # Check if episode is done
                    done = all_dones['__all__'][0]
                    
                    # Store state for visualization
                    state_list.append((rng_step, state, unbatchify(actions.squeeze(1), env.agents)))
                    # state_list.append((rng_step, state, action_dict))
                    # state_list.append(state.state)
                    
                    step_count += 1
                    # Print actions and rewards for debugging
                    # print(f"Step {step_count} - Actions: {action_dict}, Rewards: {rewards}")
                
                # Reset for next episode
                episode_count += 1
                if episode_count < num_episodes:
                    # print(f"Episode {episode_count} completed in {step_count} steps. Starting next episode.")
                    rng, rng_reset = jax.random.split(rng)
                    obs, state = env.reset(rng_reset)
                    state_list.append(state)
                    done = False
                    all_dones = {agent: jnp.zeros((1,), dtype=bool) for agent in env.agents}
                    all_dones["__all__"] = jnp.zeros((1,), dtype=bool)
                    hidden = ScannedRNN.initialize_carry(hidden_dim, len(env.agents), 1)
                else:
                    print(f"All {num_episodes} episodes completed.")
            
            state_list = squeeze_state_arrays(state_list)
            with open(f'exps/state_seqs/smax_{map_name}_{alg_name}_{seed}_state_list.pkl', 'wb') as file:
                pickle.dump(state_list, file)

            # viz = SMAXVisualizer(env, state_list)  
            
            # Create animation
            # viz.animate(save_fname=f'exps/media/smax_{map_name}_{alg_name}_{seed}_animation_1.gif', view=True)
