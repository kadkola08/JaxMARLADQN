import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.training.train_state import TrainState
from safetensors.flax import load_file
from jaxmarl.environments.overcooked import Overcooked
from jaxmarl.environments.overcooked_v2 import OvercookedV2, overcooked_v2_layouts, Layout 
from jaxmarl.viz.overcooked_visualizer import OvercookedVisualizer
from jaxmarl.environments.overcooked.layouts import overcooked_layouts as layouts

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

def batchify(x: dict):
    return jnp.stack([x[agent] for agent in env.agents], axis=0)

def unbatchify(x: jnp.ndarray):
    return {agent: x[i] for i, agent in enumerate(env.agents)}

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

action_dim = 6
network = QNetwork(action_dim=action_dim, hidden_size=64)
params = load_file(model_path)

rng = jax.random.PRNGKey(42)
dummy_input = jnp.zeros((1, *env.observation_space().shape))
# network = QNetwork(action_dim=action_dim, hidden_size=64)
init_params = network.init(rng, dummy_input)


flat_params = load_file(model_path)
print("Loaded flat parameters with keys:", list(flat_params.keys()))

# Convert the flat parameters into nested structure
def unflatten_params(flat_params, model_params_struct):
    """Convert flat parameter dict to nested structure matching the model's structure"""
    nested_params = {}

    # Initialize with the structure from model_params_struct
    for key_path, value in flat_params.items():
        # Split the key path into parts
        parts = key_path.split(',')

        # Start with the root dict
        current_dict = nested_params

        # Navigate through the path
        for i, part in enumerate(parts[:-1]):  # All but the last part
            if part not in current_dict:
                current_dict[part] = {}
            current_dict = current_dict[part]

        # Set the value at the leaf
        current_dict[parts[-1]] = value

    return nested_params

# Create a properly structured parameter dict
structured_params = unflatten_params(flat_params, init_params)
print("Converted to structured parameters with keys:", list(structured_params.keys()))


rng = jax.random.PRNGKey(0)
rng, rng_reset = jax.random.split(rng)

state_list = []
obs, state = env.reset(rng_reset)
state_list.append(state)
done = False

while not done:
    rng, rng_step = jax.random.split(rng)
    batched_obs = batchify(obs)
    # q_vals = jax.vmap(network.apply, in_axes=(None, 0))({"params": params}, batched_obs)
    q_vals = jax.vmap(network.apply, in_axes=(None, 0))(structured_params, batched_obs)

    actions = jnp.argmax(q_vals, axis=-1)
    actions = unbatchify(actions)

    obs, state, rewards, done, info = env.step(rng_step, state, actions)
    done = done['__all__']
    state_list.append(state)

viz.animate(state_list, env.agent_view_size, filename='test_v1_1.gif')
