"""
Box Pushing Environment for JaxMARL

A multi-agent environment where agents must cooperate to push boxes to target locations.
Based on the MacroMARL box pushing implementation adapted for JAX.
"""

import jax
import jax.numpy as jnp
import chex
from typing import Tuple, Dict, Optional
from functools import partial
from flax import struct
from jaxmarl.environments.multi_agent_env import MultiAgentEnv
from jaxmarl.environments.spaces import Box, Discrete


@struct.dataclass
class State:
    """Box Pushing Environment State"""
    agent_pos: chex.Array  # Agent positions [num_agents, 2] (x, y)
    agent_orient: chex.Array  # Agent orientations [num_agents] (0=right, 1=up, 2=left, 3=down)
    box_pos: chex.Array  # Box positions [num_boxes, 2] (x, y)
    target_pos: chex.Array  # Target positions [num_targets, 2] (x, y)
    box_at_target: chex.Array  # Whether each box is at its target [num_boxes]
    step: int  # Current step
    done: chex.Array  # Whether episode is done


class BoxPushing(MultiAgentEnv):
    """
    Multi-agent box pushing environment.
    
    Agents must cooperate to push boxes to their target locations.
    Each agent can move in 4 directions and push boxes.
    """
    
    def __init__(
        self,
        num_agents: int = 2,
        num_boxes: int = 2,
        grid_size: int = 8,
        max_steps: int = 50,
        push_reward: float = 10.0,
        target_reward: float = 20.0,
        step_penalty: float = -0.1,
        collision_penalty: float = -1.0,
        **kwargs
    ):
        """
        Initialize the box pushing environment.
        
        Args:
            num_agents: Number of agents
            num_boxes: Number of boxes to push
            grid_size: Size of the grid (grid_size x grid_size)
            max_steps: Maximum steps per episode
            push_reward: Reward for pushing a box
            target_reward: Reward for getting a box to target
            step_penalty: Penalty per step
            collision_penalty: Penalty for collisions
        """
        super().__init__(num_agents=num_agents)
        
        self.num_boxes = num_boxes
        self.num_targets = num_boxes  # One target per box
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.push_reward = push_reward
        self.target_reward = target_reward
        self.step_penalty = step_penalty
        self.collision_penalty = collision_penalty
        
        # Action space: 0=noop, 1=right, 2=up, 3=left, 4=down, 5=push
        self.action_spaces = {f"agent_{i}": Discrete(6) for i in range(num_agents)}
        
        # Observation space: agent_pos, agent_orient, box_pos, target_pos, other_agents_pos
        obs_dim = 2 + 1 + num_boxes * 2 + num_boxes * 2 + (num_agents - 1) * 2
        self.observation_spaces = {
            f"agent_{i}": Box(-jnp.inf, jnp.inf, (obs_dim,)) 
            for i in range(num_agents)
        }
        
        # Direction vectors for movement
        self.directions = jnp.array([
            [1, 0],   # right
            [0, 1],   # up
            [-1, 0],  # left
            [0, -1]   # down
        ])
        
        # Agent names
        self.agents = [f"agent_{i}" for i in range(num_agents)]
        self.a_to_i = {a: i for i, a in enumerate(self.agents)}

    @partial(jax.jit, static_argnums=[0])
    def reset(self, key: chex.PRNGKey) -> Tuple[Dict[str, chex.Array], State]:
        """Reset the environment to initial state."""
        key_agent, key_box, key_target = jax.random.split(key, 3)
        
        # Initialize agent positions randomly
        agent_pos = jax.random.uniform(
            key_agent, (self.num_agents, 2), 
            minval=0, maxval=self.grid_size
        )
        
        # Initialize box positions randomly (avoid agent positions)
        box_pos = jax.random.uniform(
            key_box, (self.num_boxes, 2),
            minval=0, maxval=self.grid_size
        )
        
        # Initialize target positions randomly
        target_pos = jax.random.uniform(
            key_target, (self.num_targets, 2),
            minval=0, maxval=self.grid_size
        )
        
        # Initialize agent orientations randomly
        agent_orient = jax.random.randint(
            key_agent, (self.num_agents,), 0, 4
        )
        
        # Check initial box-target assignments
        box_at_target = self._check_box_targets(box_pos, target_pos)
        
        state = State(
            agent_pos=agent_pos,
            agent_orient=agent_orient,
            box_pos=box_pos,
            target_pos=target_pos,
            box_at_target=box_at_target,
            step=0,
            done=jnp.array(False)
        )
        
        return self.get_obs(state), state

    @partial(jax.jit, static_argnums=[0])
    def step_env(
        self, 
        key: chex.PRNGKey, 
        state: State, 
        actions: Dict[str, chex.Array]
    ) -> Tuple[Dict[str, chex.Array], State, Dict[str, float], Dict[str, bool], Dict]:
        """Step the environment."""
        # Convert actions to array
        action_array = jnp.array([actions[agent] for agent in self.agents])
        
        # Update agent positions and orientations
        new_agent_pos, new_agent_orient = self._update_agents(
            state.agent_pos, state.agent_orient, action_array
        )
        
        # Update box positions based on pushing actions
        new_box_pos = self._update_boxes(
            state.agent_pos, new_agent_pos, state.box_pos, action_array
        )
        
        # Check if boxes are at targets
        new_box_at_target = self._check_box_targets(new_box_pos, state.target_pos)
        
        # Calculate rewards
        rewards = self._calculate_rewards(
            state, new_agent_pos, new_box_pos, new_box_at_target, action_array
        )
        
        # Check if episode is done
        done = (state.step >= self.max_steps - 1) | jnp.all(new_box_at_target)
        
        new_state = state.replace(
            agent_pos=new_agent_pos,
            agent_orient=new_agent_orient,
            box_pos=new_box_pos,
            box_at_target=new_box_at_target,
            step=state.step + 1,
            done=done
        )
        
        obs = self.get_obs(new_state)
        
        # Convert rewards to dict
        reward_dict = {agent: rewards[i] for i, agent in enumerate(self.agents)}
        
        # Done dict
        dones = {agent: done for agent in self.agents}
        dones["__all__"] = done
        
        return obs, new_state, reward_dict, dones, {}

    def _update_agents(
        self, 
        agent_pos: chex.Array, 
        agent_orient: chex.Array, 
        actions: chex.Array
    ) -> Tuple[chex.Array, chex.Array]:
        """Update agent positions and orientations."""
        
        @partial(jax.vmap, in_axes=(0, 0, 0))
        def _update_single_agent(pos, orient, action):
            # Movement actions (1-4)
            move_action = (action >= 1) & (action <= 4)
            move_dir = action - 1
            # Ensure indices are within [0, 3] for safe indexing
            safe_move_dir = jnp.clip(move_dir, 0, 3)
            
            # Update orientation for movement
            new_orient = jnp.where(move_action, safe_move_dir, orient)
            
            # Calculate new position
            candidate_vec = self.directions[safe_move_dir]
            move_vector = jnp.where(move_action, candidate_vec, jnp.array([0, 0]))
            new_pos = pos + move_vector
            
            # Keep within bounds
            new_pos = jnp.clip(new_pos, 0, self.grid_size - 1)
            
            return new_pos, new_orient
        
        return _update_single_agent(agent_pos, agent_orient, actions)

    def _update_boxes(
        self, 
        old_agent_pos: chex.Array, 
        new_agent_pos: chex.Array, 
        box_pos: chex.Array, 
        actions: chex.Array
    ) -> chex.Array:
        """Update box positions based on pushing actions."""
        
        def _update_single_box(agent_old, agent_new, action, box_pos):
            # Check if agent is pushing (action == 5)
            is_pushing = action == 5
            
            # Check if agent moved and is adjacent to box
            agent_moved = jnp.any(agent_new != agent_old)
            agent_direction = agent_new - agent_old
            
            # Check if box is in front of agent
            box_in_front = jnp.all(box_pos == agent_new + agent_direction)
            
            # Push box if conditions are met
            can_push = is_pushing & agent_moved & box_in_front
            push_direction = agent_direction
            
            new_box_pos = jnp.where(
                can_push,
                box_pos + push_direction,
                box_pos
            )
            
            # Keep within bounds
            new_box_pos = jnp.clip(new_box_pos, 0, self.grid_size - 1)
            
            return new_box_pos
        
        # Update all boxes based on all agents using nested vmap (agents -> boxes)
        update_boxes_for_agent = jax.vmap(
            lambda bpos, a_old, a_new, act: _update_single_box(a_old, a_new, act, bpos),
            in_axes=(0, None, None, None),
        )
        new_box_positions = jax.vmap(
            lambda a_old, a_new, act: update_boxes_for_agent(box_pos, a_old, a_new, act),
            in_axes=(0, 0, 0),
        )(old_agent_pos, new_agent_pos, actions)
        
        # Take the first valid push for each box
        valid_pushes = jnp.any(new_box_positions != box_pos[None, :, :], axis=2)
        first_valid_agent = jnp.argmax(valid_pushes, axis=0)
        idx = first_valid_agent[None, :, None]
        final_box_pos = jnp.take_along_axis(new_box_positions, idx, axis=0)[0]
        
        return final_box_pos

    def _check_box_targets(self, box_pos: chex.Array, target_pos: chex.Array) -> chex.Array:
        """Check which boxes are at their targets."""
        
        @partial(jax.vmap, in_axes=(0, None))
        def _check_single_box(box_pos, target_pos):
            distances = jnp.linalg.norm(box_pos[None, :] - target_pos, axis=1)
            return jnp.any(distances < 0.5)  # Within 0.5 units of any target
        
        return _check_single_box(box_pos, target_pos)

    def _calculate_rewards(
        self, 
        state: State, 
        new_agent_pos: chex.Array, 
        new_box_pos: chex.Array, 
        new_box_at_target: chex.Array, 
        actions: chex.Array
    ) -> chex.Array:
        """Calculate rewards for all agents."""
        
        # Step penalty
        step_rewards = jnp.full(self.num_agents, self.step_penalty)
        
        # Push reward (if any box moved)
        box_moved = jnp.any(new_box_pos != state.box_pos, axis=1)
        push_rewards = jnp.where(box_moved, self.push_reward, 0.0)
        
        # Target reward (if any box reached target)
        new_targets_reached = jnp.sum(new_box_at_target) - jnp.sum(state.box_at_target)
        target_rewards = jnp.full(self.num_agents, new_targets_reached * self.target_reward)
        
        # Collision penalty (if agents are too close)
        collision_penalties = self._check_collisions(new_agent_pos)
        
        total_rewards = step_rewards + push_rewards + target_rewards + collision_penalties
        
        return total_rewards

    def _check_collisions(self, agent_pos: chex.Array) -> chex.Array:
        """Check for agent collisions and return penalties."""
        
        @partial(jax.vmap, in_axes=(0, None))
        def _check_single_agent_collision(pos, all_pos):
            distances = jnp.linalg.norm(pos[None, :] - all_pos, axis=1)
            # Exclude self
            distances = jnp.where(distances == 0, jnp.inf, distances)
            min_distance = jnp.min(distances)
            return jnp.where(min_distance < 0.5, self.collision_penalty, 0.0)
        
        return _check_single_agent_collision(agent_pos, agent_pos)

    @partial(jax.jit, static_argnums=[0])
    def get_obs(self, state: State) -> Dict[str, chex.Array]:
        """Get observations for all agents."""
        
        @partial(jax.vmap, in_axes=(0, None, None, None, None))
        def _get_single_obs(agent_idx, agent_pos, agent_orient, box_pos, target_pos):
            # Agent's own position and orientation
            own_pos = agent_pos[agent_idx]
            own_orient = agent_orient[agent_idx]
            
            # Other agents' positions (JAX-jittable: avoid boolean indexing)
            # Roll so that current agent is at position 0, then take the rest
            rolled_pos = jnp.roll(agent_pos, -agent_idx, axis=0)
            other_agents_pos = rolled_pos[1:]
            
            # Flatten all observations
            obs = jnp.concatenate([
                own_pos,                    # 2
                jnp.array([own_orient]),    # 1
                box_pos.flatten(),          # num_boxes * 2
                target_pos.flatten(),       # num_targets * 2
                other_agents_pos.flatten()  # (num_agents - 1) * 2
            ])
            
            return obs
        
        obs_array = _get_single_obs(
            jnp.arange(self.num_agents),
            state.agent_pos,
            state.agent_orient,
            state.box_pos,
            state.target_pos
        )
        
        return {agent: obs_array[i] for i, agent in enumerate(self.agents)}

    @property
    def name(self) -> str:
        return "BoxPushing"

    def agent_classes(self) -> Dict[str, list]:
        return {"agents": self.agents}


# Example usage and testing
if __name__ == "__main__":
    import jax.random as random
    
    # Create environment
    env = BoxPushing(num_agents=2, num_boxes=2, grid_size=8)
    
    # Test reset
    key = random.PRNGKey(0)
    obs, state = env.reset(key)
    
    print("Environment created successfully!")
    print(f"Number of agents: {env.num_agents}")
    print(f"Number of boxes: {env.num_boxes}")
    print(f"Grid size: {env.grid_size}")
    print(f"Observation space: {env.observation_spaces['agent_0']}")
    print(f"Action space: {env.action_spaces['agent_0']}")
    
    # Test step
    actions = {agent: env.action_space(agent).sample(random.PRNGKey(1)) 
               for agent in env.agents}
    
    obs, state, rewards, dones, info = env.step(key, state, actions)
    
    print("\nStep completed successfully!")
    print(f"Rewards: {rewards}")
    print(f"Dones: {dones}")
