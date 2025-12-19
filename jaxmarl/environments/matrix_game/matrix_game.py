"""
Single-step matrix game environment.

Agents receive rewards based on a payoff matrix. The game terminates after one step.
"""

import jax
import jax.numpy as jnp
from typing import Dict, Tuple
import chex
from functools import partial
from flax import struct

from jaxmarl.environments.multi_agent_env import MultiAgentEnv, State
from jaxmarl.environments import spaces


@struct.dataclass
class MatrixGameState:
    """State for matrix game environment."""
    done: chex.Array
    step: int


class MatrixGame(MultiAgentEnv):
    """
    Single-step matrix game environment.
    
    Agents simultaneously choose actions from a discrete action space.
    Rewards are determined by a payoff matrix based on all agents' actions.
    The game terminates after one step.
    
    Args:
        num_agents: Number of agents in the game
        num_actions: Number of actions available to each agent
        payoff_matrix: Payoff matrix of shape (num_agents, num_actions, ..., num_actions)
                      where the last num_agents dimensions correspond to each agent's action.
                      For 2 agents: shape is (2, num_actions, num_actions)
                      For 3 agents: shape is (3, num_actions, num_actions, num_actions)
                      payoff_matrix[agent_i][a0][a1]...[aN] is the reward for agent_i
                      when agent_0 chooses a0, agent_1 chooses a1, etc.
        observation_type: Type of observation. Options: 'empty', 'action_indices'
                          - 'empty': Empty observations (zeros)
                          - 'action_indices': Observations contain action indices of all agents
    """
    
    def __init__(
        self,
        num_agents: int = 2,
        num_actions: int = 2,
        payoff_matrix: jnp.ndarray = None,
        observation_type: str = 'empty',
    ):
        super().__init__(num_agents=num_agents)
        self.agents = [f"agent_{i}" for i in range(num_agents)]
        self.num_actions = num_actions
        self.observation_type = observation_type
        
        # Default payoff matrix: Prisoner's Dilemma for 2 agents
        if payoff_matrix is None:
            if num_agents == 2 and num_actions == 2:
                payoff_matrix = jnp.array([
                    [[3, 0], [5, 1]],  
                    [[3, 5], [0, 1]]   
                ], dtype=jnp.float32)
            if num_agents == 2 and num_actions == 3:
                payoff_matrix = jnp.array([
                [[8, -12, -12],
                 [-12, 6, 0],
                 [-12, 0, 6]],

                [[8, -12, -12], 
                 [-12, 6, 0],
                 [-12, 0, 6]]
                ], dtype=jnp.float32)
                payoff_matrix = payoff_matrix / 2
            else:
                # Default: zero-sum game
                payoff_matrix = jnp.zeros((num_agents,) + (num_actions,) * num_agents)
        else:
            payoff_matrix = jnp.array(payoff_matrix)
        
        # Validate payoff matrix shape
        expected_shape = (num_agents,) + (num_actions,) * num_agents
        assert payoff_matrix.shape == expected_shape, \
            f"Payoff matrix shape {payoff_matrix.shape} does not match expected shape {expected_shape}"
        
        self.payoff_matrix = payoff_matrix
        
        # Precompute strides for row-major indexing (static, computed at init time)
        payoff_shape = payoff_matrix.shape[1:]
        self.strides = jnp.array([
            jnp.prod(jnp.array(payoff_shape[i+1:], dtype=jnp.int32)) 
            if i < len(payoff_shape) - 1 else 1
            for i in range(len(payoff_shape))
        ], dtype=jnp.int32)
        self.payoff_size = jnp.prod(jnp.array(payoff_shape, dtype=jnp.int32))
        
        # Define observation and action spaces
        if observation_type == 'empty':
            obs_shape = (1,)  # Empty observation
            obs_dtype = jnp.float32
            obs_low = 0
            obs_high = 1
        elif observation_type == 'action_indices':
            obs_shape = (num_agents,)  # Indices of all agents' actions
            obs_dtype = jnp.int32
            obs_low = 0
            obs_high = num_actions
        else:
            raise ValueError(f"Unknown observation_type: {observation_type}")
        
        for agent in self.agents:
            self.observation_spaces[agent] = spaces.Box(
                low=obs_low, high=obs_high, shape=obs_shape, dtype=obs_dtype
            )
            self.action_spaces[agent] = spaces.Discrete(num_actions)
    
    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key: chex.PRNGKey) -> Tuple[Dict[str, chex.Array], State]:
        """Reset the environment to initial state."""
        state = State(
            done=jnp.zeros((self.num_agents,), dtype=bool),
            step=jnp.array(0, dtype=jnp.int32)
        )
        obs = self.get_obs(state)
        return obs, state
    
    def get_obs(self, state: State) -> Dict[str, chex.Array]:
        """Get observations for all agents."""
        if self.observation_type == 'empty':
            # Return empty observations (zeros)
            obs = {agent: jnp.zeros((1,), dtype=jnp.float32) for agent in self.agents}
        elif self.observation_type == 'action_indices':
            # Return zeros initially (no actions taken yet)
            obs = {agent: jnp.zeros((self.num_agents,), dtype=jnp.int32) for agent in self.agents}
        else:
            raise ValueError(f"Unknown observation_type: {self.observation_type}")
        return obs
    
    @partial(jax.jit, static_argnums=(0,))
    def step_env(
        self, key: chex.PRNGKey, state: State, actions: Dict[str, chex.Array]
    ) -> Tuple[Dict[str, chex.Array], State, Dict[str, float], Dict[str, bool], Dict]:
        """
        Step the environment.
        
        Args:
            key: Random key (not used in matrix games)
            state: Current state
            actions: Dictionary of actions for each agent
            
        Returns:
            obs: Observations for all agents
            state: Next state
            rewards: Rewards for all agents
            dones: Done flags for all agents
            infos: Additional info
        """
        # Extract actions from dictionary
        action_list = jnp.array([actions[agent] for agent in self.agents])
        
        # Compute rewards from payoff matrix
        # For 2 agents: rewards[i] = payoff_matrix[i][action_0][action_1]
        # For n agents: rewards[i] = payoff_matrix[i][action_0][action_1]...[action_n-1]
        rewards = self._compute_rewards(action_list)
        
        # Game always terminates after one step
        done = jnp.ones((self.num_agents,), dtype=bool)
        
        # Update state
        next_state = State(
            done=done,
            step=state.step + 1
        )
        
        # Get observations (can include action information)
        obs = self.get_obs(next_state)
        if self.observation_type == 'action_indices':
            # Include action indices in observations
            obs = {agent: action_list for agent in self.agents}
        
        # Convert rewards to dictionary (keep as JAX arrays, don't convert to float)
        rewards_dict = {agent: rewards[i] for i, agent in enumerate(self.agents)}
        
        # Convert dones to dictionary (keep as JAX arrays, don't convert to bool)
        dones = {agent: done[i] for i, agent in enumerate(self.agents)}
        dones["__all__"] = jnp.all(done)
        
        infos = {}
        
        return obs, next_state, rewards_dict, dones, infos
    
    def _compute_rewards(self, actions: jnp.ndarray) -> jnp.ndarray:
        """
        Compute rewards from payoff matrix based on actions.
        
        Args:
            actions: Array of actions for each agent, shape (num_agents,)
            
        Returns:
            rewards: Array of rewards for each agent, shape (num_agents,)
        """
        # Convert actions to integer indices
        actions_int = actions.astype(jnp.int32)
        
        # Use a general approach that works for any number of agents
        # Reshape payoff matrix and use dynamic indexing
        rewards = self._compute_rewards_general(actions_int)
        
        return rewards
    
    def _compute_rewards_general(self, actions_int: jnp.ndarray) -> jnp.ndarray:
        """
        General reward computation for arbitrary number of agents.
        Uses precomputed strides to index into the payoff matrix.
        """
        # Compute flat index: sum of action[i] * stride[i]
        flat_idx = jnp.sum(actions_int * self.strides)
        
        # Reshape payoff matrix to (num_agents, -1) and index
        payoff_flat = self.payoff_matrix.reshape(self.num_agents, self.payoff_size)
        
        # Get rewards for all agents at this flat index
        rewards = payoff_flat[:, flat_idx]
        
        return rewards
    
    @property
    def name(self) -> str:
        """Environment name."""
        return "MatrixGame-v0"
    
    def action_space(self, agent: str = None) -> spaces.Discrete:
        """Action space for an agent."""
        return spaces.Discrete(self.num_actions)
    
    def observation_space(self, agent: str = None) -> spaces.Box:
        """Observation space for an agent."""
        if self.observation_type == 'empty':
            return spaces.Box(low=0, high=1, shape=(1,), dtype=jnp.float32)
        elif self.observation_type == 'action_indices':
            return spaces.Box(low=0, high=self.num_actions, shape=(self.num_agents,), dtype=jnp.int32)
        else:
            raise ValueError(f"Unknown observation_type: {self.observation_type}")
    
    @partial(jax.jit, static_argnums=(0,))
    def get_avail_actions(self, state: State) -> Dict[str, chex.Array]:
        """Returns the available actions for each agent.
        
        For matrix games, all actions are always available.
        Returns a one-hot mask where all actions are valid (all 1s).
        """
        # All actions are always available in matrix games
        avail_actions = jnp.ones((self.num_actions,), dtype=jnp.float32)
        return {agent: avail_actions for agent in self.agents}
    
    @property
    def agent_classes(self) -> dict:
        """Agent classes (all agents are homogeneous)."""
        return {"agent": self.agents}

