import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.training.train_state import TrainState
from safetensors.flax import load_file
from jaxmarl.environments.overcooked import Overcooked
from jaxmarl.viz.overcooked_visualizer import OvercookedVisualizer
from jaxmarl.environments.overcooked.layouts import overcooked_layouts as layouts

# First, recreate the same CNN and QNetwork architecture
class CNN(nn.Module):
    activation: str = "relu"

    @nn.compact
    def __call__(self, x):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh
        x = nn.Conv(
            features=32,
            kernel_size=(5, 5),
        )(x)
        x = activation(x)
        x = nn.Conv(
            features=32,
            kernel_size=(3, 3),
        )(x)
        x = activation(x)
        x = nn.Conv(
            features=32,
            kernel_size=(3, 3),
        )(x)
        x = activation(x)
        x = x.reshape((x.shape[0], -1))  # Flatten

        x = nn.Dense(
            features=64 
        )(x)
        x = activation(x)

        return x

class QNetwork(nn.Module):
    action_dim: int
    hidden_size: int = 64

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        embedding = CNN()(x)
        x = nn.Dense(self.hidden_size)(embedding)
        x = nn.Dense(self.action_dim)(x)
        return x

# Instead of using TrainState, we can create a simple class for inference
class InferenceState:
    def __init__(self, apply_fn, params):
        self.apply_fn = apply_fn
        self.params = params

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

def load_model(model_path, observation_shape, action_dim, hidden_size=64):
    # Create network instance
    network = QNetwork(action_dim=action_dim, hidden_size=hidden_size)
    
    # Initialize with dummy parameters to get the correct structure
    rng = jax.random.PRNGKey(0)
    init_x = jnp.zeros((1, *observation_shape))
    init_params = network.init(rng, init_x)
    
    # Load the saved parameters
    flat_params = load_file(model_path)
    
    # Convert flat parameter structure to nested structure
    nested_params = construct_nested_dict(flat_params)

    # print("Reconstructed parameter structure:")
    # print(jax.tree_map(lambda x: x.shape if hasattr(x, 'shape') else type(x), nested_params))
    
    # The key issue: Make sure params have the correct structure
    # Depending on how parameters were saved, you might need to restructure them
    # Most likely they need to be in a {'params': ...} dictionary
    
    return network, nested_params

def print_param_structure(params, prefix=""):
    """Helper function to print the nested structure of parameters"""
    if isinstance(params, dict):
        for key, value in params.items():
            print(f"{prefix}{key}:")
            print_param_structure(value, prefix + "  ")
    else:
        print(f"{prefix}Shape: {params.shape if hasattr(params, 'shape') else type(params)}")

def batchify(x: dict):
    return jnp.stack([x[agent] for agent in env.agents], axis=0)

def unbatchify(x: jnp.ndarray):
    return {agent: x[i] for i, agent in enumerate(env.agents)}

# Example usage
if __name__ == "__main__":
    # Path to your saved model
    save_path = "models"
    env_name = "overcooked_cramped_room"
    alg_name = "iql_cnn"
    seed = 0
    vmap_index = 0

    model_path = f"{save_path}/{env_name}/{alg_name}_{env_name}_seed{seed}_vmap{vmap_index}.safetensors"
    
    env_id = "Overcooked"
    map_layout = "cramped_room"

    env = Overcooked(layout=layouts[map_layout])
    viz =  OvercookedVisualizer()

    # Define your environment's observation shape and action dimension
    observation_shape = env.observation_space().shape  # Example shape, replace with your env's actual shape
    action_dim = 6  # Example, replace with your env's actual action dimension

    # Load the model
    network, loaded_params = load_model(model_path, observation_shape, action_dim)

    # Now you can use the model for inference
    def get_action(obs, network, params):
        q_values = network.apply(params, obs)
        return jnp.argmax(q_values, axis=-1)

    # Example observation
    example_obs = jnp.zeros((1, *observation_shape))
    action = get_action(example_obs, network, loaded_params)
    # print(f"Predicted action: {action}")

    rng = jax.random.PRNGKey(0)
    rng, rng_reset = jax.random.split(rng)

    state_list = []
    obs, state = env.reset(rng_reset)
    state_list.append(state)
    done = False

    while not done:
        rng, rng_step = jax.random.split(rng)
        batched_obs = batchify(obs)

        q_vals = network.apply(loaded_params, batched_obs)
        actions = jnp.argmax(q_vals, axis=-1)
        actions = unbatchify(actions)
        print(actions)

        obs, state, rewards, done, info = env.step(rng_step, state, actions)
        breakpoint()
        print(rewards)
        done = done['__all__']
        state_list.append(state)
    
    viz.animate(state_list, env.agent_view_size, filename='animation.gif')
