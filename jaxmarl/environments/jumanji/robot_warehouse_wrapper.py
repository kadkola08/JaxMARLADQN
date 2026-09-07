"""
Jumanji RobotWarehouse wrapped to match JaxMARL MultiAgentEnv interface.
Requires: pip install jumanji
"""

import jax
import jax.numpy as jnp
import chex
from typing import Dict, Tuple
from functools import partial
from flax import struct

from jaxmarl.environments.multi_agent_env import MultiAgentEnv
from jaxmarl.environments.spaces import Box, Discrete

try:
    import jumanji
    from jumanji.environments.routing.robot_warehouse import RobotWarehouse as JumanjiRobotWarehouse
    from jumanji.environments.routing.robot_warehouse.generator import RandomGenerator as JumanjiRandomGenerator
    JUMANJI_AVAILABLE = True
except ImportError:
    JUMANJI_AVAILABLE = False
    JumanjiRobotWarehouse = None
    JumanjiRandomGenerator = None


@struct.dataclass
class RobotWarehouseEnvState:
    """State for the wrapped RobotWarehouse env (holds Jumanji state + timestep)."""
    jumanji_state: chex.Array  # PyTree from jumanji
    timestep: chex.Array       # PyTree: observation, reward, discount, step_type
    step: int
    done: chex.Array


class RobotWarehouseWrapper(MultiAgentEnv):
    """
    Wraps Jumanji's RobotWarehouse as a JaxMARL MultiAgentEnv so it works
    with IQL/VDN/QMix and other baselines that expect agents, obs dict, action_spaces.
    """

    def __init__(self, env_id: str = "RobotWarehouse-v0", **env_kwargs):
        if not JUMANJI_AVAILABLE:
            raise ImportError(
                "Jumanji is required for RobotWarehouse. Install with: pip install jumanji"
            )
        # Allow custom generator (e.g. 1 agent, small map), time_limit, and sensor_range (view radius in cells)
        generator = env_kwargs.pop("generator", None)
        time_limit = env_kwargs.pop("time_limit", 150)
        sensor_range = env_kwargs.pop("sensor_range", None)
        # If no generator but sensor_range set, build default RobotWarehouse-v0 generator with that view radius
        if generator is None and sensor_range is not None and JumanjiRandomGenerator is not None:
            generator = JumanjiRandomGenerator(
                shelf_rows=2,
                shelf_columns=3,
                column_height=8,
                num_agents=4,
                sensor_range=sensor_range,
                request_queue_size=8,
            )
        if generator is not None:
            self._jumanji_env = JumanjiRobotWarehouse(
                generator=generator, time_limit=time_limit, **env_kwargs
            )
        else:
            self._jumanji_env = jumanji.make(env_id, **env_kwargs)

        # Infer num_agents and obs size from the Jumanji env (depends on generator's sensor_range).
        # Prefer env.num_obs_features / env.num_agents so we don't rely on spec shape API.
        if hasattr(self._jumanji_env, "num_obs_features") and hasattr(self._jumanji_env, "num_agents"):
            num_agents = int(self._jumanji_env.num_agents)
            num_obs_features = int(self._jumanji_env.num_obs_features)
        else:
            obs_spec = self._jumanji_env.observation_spec
            if hasattr(obs_spec, "agents_view") and hasattr(obs_spec.agents_view, "shape"):
                shp = obs_spec.agents_view.shape
                num_agents = int(shp[0]) if len(shp) > 0 else 4
                num_obs_features = int(shp[1]) if len(shp) > 1 else 128
            else:
                num_agents = 4
                num_obs_features = 128

        super().__init__(num_agents=num_agents)

        self.agents = [f"agent_{i}" for i in range(num_agents)]
        self._env_id = env_id

        # 5 actions: noop, forward, turn_left, turn_right, toggle_load
        n_actions = 5
        self.action_spaces = {a: Discrete(n_actions) for a in self.agents}

        # Observation: agents_view (num_obs_features from spec) + action_mask (5) + step_count (1)
        obs_dim = num_obs_features + 5 + 1
        self.observation_spaces = {
            a: Box(low=-jnp.inf, high=jnp.inf, shape=(obs_dim,), dtype=jnp.float32)
            for a in self.agents
        }

    @property
    def name(self) -> str:
        return "RobotWarehouse"

    def _timestep_to_obs(self, timestep) -> Dict[str, chex.Array]:
        """Convert Jumanji timestep.observation to per-agent obs dict."""
        obs = timestep.observation
        # Handle both NamedTuple and dict-like observation
        if hasattr(obs, "agents_view"):
            agents_view = obs.agents_view  # (num_agents, num_features)
            action_mask = obs.action_mask  # (num_agents, 5)
            step_count = obs.step_count    # ()
        else:
            agents_view = obs["agents_view"]
            action_mask = obs["action_mask"]
            step_count = obs["step_count"]

        num_agents = agents_view.shape[0]
        max_obs_len = agents_view.shape[1] + action_mask.shape[1] + 1

        def one_obs(i):
            view = agents_view[i].flatten().astype(jnp.float32)
            mask = action_mask[i].astype(jnp.float32)
            step = jnp.array(step_count, dtype=jnp.float32)
            if step.shape == ():
                step = step.reshape(1)
            vec = jnp.concatenate([view, mask, step])
            # Pad to fixed size for observation_spaces
            pad_len = max(0, max_obs_len - vec.shape[0])
            vec = jnp.pad(vec, (0, pad_len), mode="constant", constant_values=0)
            return vec[: self.observation_spaces[self.agents[0]].shape[0]]

        return {a: one_obs(i) for i, a in enumerate(self.agents)}

    def _timestep_to_done(self, timestep) -> chex.Array:
        """True when episode is done (LAST step)."""
        # StepType.LAST = 2
        step_type = timestep.step_type
        if hasattr(step_type, "shape") and step_type.shape == ():
            return step_type == 2
        return jnp.equal(step_type, 2)

    def _timestep_to_reward(self, timestep) -> float:
        r = timestep.reward
        if hasattr(r, "shape") and r.size > 1:
            r = jnp.sum(r)
        return r

    def get_world_state(self, state: RobotWarehouseEnvState) -> chex.Array:
        """
        Return the entire warehouse grid as the global state for QMIX.
        Grid shape from Jumanji: (2, grid_height, grid_width) — two channels (agents, shelves).
        Flattened to a 1D vector; batch dimension preserved when state is batched.
        """
        grid = state.jumanji_state.grid  # (2, H, W) or (batch, 2, H, W)
        grid_float = jnp.asarray(grid, dtype=jnp.float32)
        # Flatten to (..., 2*H*W)
        return grid_float.reshape(grid_float.shape[:-3] + (-1,))

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key: chex.PRNGKey) -> Tuple[Dict[str, chex.Array], RobotWarehouseEnvState]:
        state, timestep = self._jumanji_env.reset(key)
        obs = self._timestep_to_obs(timestep)
        done = self._timestep_to_done(timestep)
        wrapper_state = RobotWarehouseEnvState(
            jumanji_state=state,
            timestep=timestep,
            step=0,
            done=done,
        )
        obs["world_state"] = jax.lax.stop_gradient(self.get_world_state(wrapper_state))
        return obs, wrapper_state

    def get_obs(self, state: RobotWarehouseEnvState) -> Dict[str, chex.Array]:
        return self._timestep_to_obs(state.timestep)

    @partial(jax.jit, static_argnums=(0,))
    def get_avail_actions(self, state: RobotWarehouseEnvState) -> Dict[str, chex.Array]:
        """Return valid action mask per agent (for CTRolloutManager). Works when state is batched (vmap)."""
        timestep = state.timestep
        obs = timestep.observation
        if hasattr(obs, "action_mask"):
            action_mask = obs.action_mask  # (num_agents, 5) or (batch, num_agents, 5)
        else:
            action_mask = obs["action_mask"]
        # Use ... to support both unbatched and batched: (5,) or (batch_size, 5)
        masks = {
            a: action_mask[..., i, :].astype(jnp.float32)
            for i, a in enumerate(self.agents)
        }
        return masks

    @partial(jax.jit, static_argnums=(0,))
    def step_env(
        self,
        key: chex.PRNGKey,
        state: RobotWarehouseEnvState,
        actions: Dict[str, chex.Array],
    ) -> Tuple[Dict[str, chex.Array], RobotWarehouseEnvState, Dict[str, float], Dict[str, bool], Dict]:
        # Dict of agent -> action (scalar) -> (num_agents,) array
        action_list = jnp.stack([actions[a] for a in self.agents])
        new_jumanji_state, timestep = self._jumanji_env.step(state.jumanji_state, action_list)

        obs = self._timestep_to_obs(timestep)
        reward_scalar = self._timestep_to_reward(timestep)
        done_all = self._timestep_to_done(timestep)

        # Per-agent reward (RobotWarehouse uses shared global reward)
        rewards = {a: reward_scalar for a in self.agents}
        rewards["__all__"] = reward_scalar
        dones = {a: done_all for a in self.agents}
        dones["__all__"] = done_all

        wrapper_state = RobotWarehouseEnvState(
            jumanji_state=new_jumanji_state,
            timestep=timestep,
            step=state.step + 1,
            done=done_all,
        )
        obs["world_state"] = jax.lax.stop_gradient(self.get_world_state(wrapper_state))
        return obs, wrapper_state, rewards, dones, {}
