"""
Box Pushing Environment for JaxMARL

- 2 agents, 3 boxes (2 small, 1 large)
- Grid size: 8x8
- Goal area: top row (y=7)
- Actions: turn left, turn right, move forward, stay
- Stochastic actions: 0.9 success probability
- Observations: discrete 5 values describing what's in front of agent
"""

import jax
import jax.numpy as jnp
import chex
from typing import Tuple, Dict
from functools import partial
from flax import struct
from jaxmarl.environments.multi_agent_env import MultiAgentEnv
from jaxmarl.environments.spaces import Box, Discrete


@struct.dataclass
class State:
    """Box Pushing Environment State"""
    agent_pos: chex.Array            # [num_agents, 2]
    agent_orient: chex.Array         # [num_agents] - 0=right, 1=up, 2=left, 3=down
    box_pos: chex.Array              # [num_boxes, max_cells, 2]
    box_at_goal: chex.Array          # [num_boxes] boolean
    step: int
    done: chex.Array                 # scalar boolean


class BoxPushing(MultiAgentEnv):
    """
    Multi-agent box pushing environment.
    
    Actions:
    0 = stay
    1 = turn left
    2 = turn right  
    3 = move forward
    """

    def __init__(
        self,
        num_agents: int = 2,
        num_boxes: int = 3,
        grid_size: int = 8,
        max_steps: int = 50,
        step_penalty: float = -0.1,
        bump_penalty: float = -5.0,
        small_box_goal_reward: float = 10.0,
        large_box_coop_goal_reward: float = 100.0,
        action_success_prob: float = 0.9,
        **kwargs
    ):
        super().__init__(num_agents=num_agents)

        # Environment config
        self.num_agents = num_agents
        self.num_boxes = num_boxes
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.step_penalty = step_penalty
        self.bump_penalty = bump_penalty
        self.small_box_goal_reward = small_box_goal_reward
        self.large_box_coop_goal_reward = large_box_coop_goal_reward
        self.action_success_prob = action_success_prob

        # Box sizes: box 0 = 1 cell, box 1 = 2 cells (large), box 2 = 1 cell
        self.box_sizes = jnp.array([1, 2, 1], dtype=jnp.int32)
        self.max_cells_per_box = 2

        # Action space: 0=stay, 1=turn_left, 2=turn_right, 3=move_forward
        self.action_spaces = {f"agent_{i}": Discrete(4) for i in range(num_agents)}

        # Observation space: discrete 5 values (empty, wall, other agent, small box, large box)
        # Return as 1D array for wrapper compatibility
        self.observation_spaces = {
            f"agent_{i}": Box(0, 4, (1,), dtype=jnp.int32)
            for i in range(num_agents)
        }

        # Direction vectors (right, up, left, down)
        self.directions = jnp.array([
            [1, 0],   # right (0)
            [0, 1],   # up    (1)
            [-1, 0],  # left  (2)
            [0, -1]   # down  (3)
        ])

        # Agents and mapping
        self.agents = [f"agent_{i}" for i in range(self.num_agents)]
        self.a_to_i = {a: i for i, a in enumerate(self.agents)}

    @partial(jax.jit, static_argnums=[0])
    def reset(self, key: chex.PRNGKey) -> Tuple[Dict[str, chex.Array], State]:
        """Reset environment to the requested initial configuration."""
        
        # Fixed starting positions
        # Agent 0 at (1,1) facing right (orient=0)
        # Agent 1 at (6,1) facing left (orient=2)
        agent_pos = jnp.array([
            [1.0, 1.0],  # agent_0
            [6.0, 1.0],  # agent_1
        ])
        agent_orient = jnp.array([0, 2], dtype=jnp.int32)  # right, left

        # Fixed boxes:
        # - Box 0: single cell at (1,4)
        # - Box 1: two cells at (3,4) and (4,4) - large box
        # - Box 2: single cell at (6,4)
        # For boxes smaller than max_cells_per_box, duplicate the occupied cell(s)
        box_pos = jnp.array([
            [[1.0, 4.0], [1.0, 4.0]],   # Box 0 (size 1; duplicate)
            [[3.0, 4.0], [4.0, 4.0]],   # Box 1 (size 2)
            [[6.0, 4.0], [6.0, 4.0]],   # Box 2 (size 1; duplicate)
        ])

        # Compute which boxes are in the goal area initially (top row: y == grid_size - 1)
        box_at_goal = self._check_box_goals(box_pos)

        state = State(
            agent_pos=agent_pos,
            agent_orient=agent_orient,
            box_pos=box_pos,
            box_at_goal=box_at_goal,
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
        """
        Step the environment.
        Returns obs, new_state, rewards dict, dones dict, info dict.
        """
        # Convert actions mapping to array
        action_array = jnp.array([actions[agent] for agent in self.agents])
        
        # Split key for stochastic action success
        key, key_success = jax.random.split(key)

        # Update agents with stochastic action success
        new_agent_pos, new_agent_orient, initial_bumped_mask = self._update_agents(
            state.agent_pos, state.agent_orient, action_array, state.box_pos, key_success
        )

        # Update boxes based on push actions
        new_box_pos, box_moved_by_count, final_agent_pos, final_bumped_mask = self._update_boxes(
            state.agent_pos, new_agent_pos, new_agent_orient, state.box_pos, action_array
        )
        
        # Use final agent positions (updated if boxes were pushed)
        new_agent_pos = final_agent_pos
        bumped_mask = final_bumped_mask

        # Check goal status for boxes
        new_box_at_goal = self._check_box_goals(new_box_pos)

        # Calculate rewards
        rewards = self._calculate_rewards(
            state, new_agent_pos, new_box_pos, new_box_at_goal, 
            action_array, bumped_mask, box_moved_by_count
        )

        # Done if reached max_steps or all boxes in goal
        done = (state.step >= self.max_steps - 1) | jnp.all(new_box_at_goal)

        new_state = state.replace(
            agent_pos=new_agent_pos,
            agent_orient=new_agent_orient,
            box_pos=new_box_pos,
            box_at_goal=new_box_at_goal,
            step=state.step + 1,
            done=done
        )

        obs = self.get_obs(new_state)
        reward_dict = {agent: rewards[i] for i, agent in enumerate(self.agents)}

        dones = {agent: done for agent in self.agents}
        dones["__all__"] = done

        return obs, new_state, reward_dict, dones, {}

    def _update_agents(
        self,
        agent_pos: chex.Array,
        agent_orient: chex.Array,
        actions: chex.Array,
        box_pos: chex.Array,
        key: chex.PRNGKey
    ) -> Tuple[chex.Array, chex.Array, chex.Array]:
        """
        Update agent positions and orientations with stochastic action success.
        
        Returns (new_agent_pos, new_agent_orient, bumped_mask) where bumped_mask[i] = 1.0 
        if agent i attempted a move but was blocked, else 0.0.
        """

        def _single_agent_update(pos, orient, action, success):
            # Determine intended action based on success probability
            # If action fails (success=0), treat as stay
            effective_action = jnp.where(success, action, 0)
            
            # Turn left: decrease orientation (wrap around)
            new_orient = jnp.where(
                effective_action == 1,
                (orient - 1) % 4,
                orient
            )
            
            # Turn right: increase orientation (wrap around)
            new_orient = jnp.where(
                effective_action == 2,
                (orient + 1) % 4,
                new_orient
            )
            
            # Move forward or stay
            move_vector = jnp.where(
                effective_action == 3,
                self.directions[orient],
                jnp.array([0.0, 0.0])
            )
            
            tentative_pos = pos + move_vector
            
            # Check wall collision
            clipped_pos = jnp.clip(tentative_pos, 0.0, self.grid_size - 1.0)
            bumped_wall = jnp.any(clipped_pos != tentative_pos)
            
            # Check box collision - allow moving into box cells (will be handled in box update)
            # Only block if hitting wall
            bumped = (effective_action == 3) & bumped_wall
            
            # Allow moving into box cells - if box can't be pushed, will be handled later
            new_pos = jnp.where(bumped, pos, clipped_pos)
            
            # Orientation always updates (even if movement blocked)
            return new_pos, new_orient, jnp.where(bumped, 1.0, 0.0)

        # Sample success for each agent
        success_mask = jax.random.bernoulli(
            key, p=self.action_success_prob, shape=(self.num_agents,)
        ).astype(jnp.float32)

        # Vectorize
        vmap_update = jax.vmap(_single_agent_update, in_axes=(0, 0, 0, 0))
        new_pos, new_orient, bumped_mask = vmap_update(
            agent_pos, agent_orient, actions, success_mask
        )
        return new_pos, new_orient, bumped_mask

    def _update_boxes(
        self,
        old_agent_pos: chex.Array,
        new_agent_pos: chex.Array,
        agent_orient: chex.Array,
        box_pos: chex.Array,
        actions: chex.Array
    ) -> Tuple[chex.Array, chex.Array, chex.Array, chex.Array]:
        """
        Update boxes based on push actions.
        
        Returns (new_box_pos, push_counts, final_agent_pos, bumped_mask) where:
        - new_box_pos: updated box positions
        - push_counts: how many agents pushed each box
        - final_agent_pos: agent positions after box pushing (agents at pushed box positions)
        - bumped_mask: agents that bumped into unmovable boxes
        """

        def per_box_update(box_cells, b_idx):
            size_b = self.box_sizes[b_idx]
            
            def agent_push_check(a_old, a_new, orient, act):
                # Check if agent moved forward (action 3) and actually moved
                is_move_forward = (act == 3)
                moved = jnp.any(a_new != a_old)
                
                # Check if agent moved into a box cell (pushing the box)
                box_cell_at_new_pos = jnp.any(jnp.all(box_cells == a_new, axis=1))
                
                # Get the push direction from agent's orientation
                push_dir = self.directions[orient]
                
                can_push = is_move_forward & moved & box_cell_at_new_pos
                return can_push, push_dir

            can_push_vec, push_dirs = jax.vmap(agent_push_check, in_axes=(0, 0, 0, 0))(
                old_agent_pos, new_agent_pos, agent_orient, actions
            )
            num_pushers = jnp.sum(jnp.where(can_push_vec, 1, 0))

            def no_push_case():
                return box_cells, num_pushers

            def some_push_case():
                # Get first valid push direction
                first_valid_idx = jnp.argmax(can_push_vec.astype(jnp.int32))
                first_dir = push_dirs[first_valid_idx]
                
                # Check if all valid push directions match
                dirs_equal_first = jnp.all(push_dirs == first_dir[None, :], axis=1)
                dirs_agree = jnp.all(jnp.where(can_push_vec, dirs_equal_first, True))
                
                def move_if_allowed():
                    px = first_dir
                    
                    # Large boxes require 2 pushers, small boxes need 1
                    required_count = jnp.where(size_b > 1, 2, 1)
                    allowed_by_count = num_pushers >= required_count
                    
                    def do_move():
                        candidate = box_cells + px
                        candidate_clipped = jnp.clip(candidate, 0.0, self.grid_size - 1.0)
                        out_of_bounds = jnp.any(candidate_clipped != candidate)
                        
                        # Check for agent collisions at new position
                        flat_candidate = candidate.reshape((-1, 2))
                        agent_collision = jnp.any(
                            jnp.all(flat_candidate[:, None, :] == new_agent_pos[None, :, :], axis=2)
                        )
                        
                        cannot_move = out_of_bounds | agent_collision
                        # cannot_move is a scalar, so broadcast it for the where operation
                        final = jnp.where(cannot_move, box_cells, candidate)
                        return final
                    
                    moved_cells = jax.lax.cond(allowed_by_count, do_move, lambda: box_cells)
                    return moved_cells
                
                moved = jax.lax.cond(dirs_agree, move_if_allowed, lambda: box_cells)
                return moved, num_pushers

            return jax.lax.cond(num_pushers == 0, no_push_case, some_push_case)

        # Vectorize across boxes
        num_boxes_actual = box_pos.shape[0]
        box_indices = jnp.arange(num_boxes_actual)
        tentative_results = jax.vmap(per_box_update, in_axes=(0, 0))(box_pos, box_indices)
        tentative_box_cells = tentative_results[0]
        tentative_push_counts = tentative_results[1]

        # Check for overlaps between boxes
        def check_box_overlap(b_idx, current_cells):
            # Check overlap with each other box individually
            def check_overlap_with_other(other_idx, other_cells):
                is_different_box = other_idx != b_idx
                # Check if any cells of current box overlap with any cells of other box
                cell_overlaps = jnp.any(
                    jnp.all(current_cells[:, None, :] == other_cells[None, :, :], axis=2),
                    axis=1
                )
                has_overlap = jnp.any(cell_overlaps)
                # Only count if it's a different box
                return jnp.where(is_different_box, has_overlap, False)
            
            # Check against all boxes
            overlap_checks = jax.vmap(check_overlap_with_other, in_axes=(0, 0))(
                jnp.arange(num_boxes_actual), tentative_box_cells
            )
            overlaps = jnp.any(overlap_checks)
            return overlaps

        overlaps_vec = jax.vmap(check_box_overlap, in_axes=(0, 0))(
            jnp.arange(num_boxes_actual), tentative_box_cells
        )

        # Revert boxes with overlaps
        final_box_cells = jnp.where(
            overlaps_vec[:, None, None], box_pos, tentative_box_cells
        )
        
        # Update agent positions: if agent pushed a box, agent ends up at box's old position
        # Check which boxes moved
        boxes_moved = jnp.any(jnp.all(final_box_cells != box_pos, axis=1), axis=1)  # [num_boxes]
        
        # For each agent, check if they're at a box position and update accordingly
        def update_agent_after_box_push(agent_old, agent_pos_new):
            # Find which box (if any) agent is at
            box_cell_matches = jnp.any(
                jnp.all(final_box_cells == agent_pos_new[None, None, :], axis=2), 
                axis=1
            )  # [num_boxes]
            box_idx = jnp.argmax(box_cell_matches.astype(jnp.int32))
            is_at_box = box_cell_matches[box_idx]
            this_box_moved = boxes_moved[box_idx]
            
            # If agent successfully pushed box, agent goes to box's old position
            # If agent tried to push but box didn't move, agent stays at old position (bumped)
            # Otherwise, agent stays at new position
            final_agent_pos = jnp.where(
                is_at_box & this_box_moved,
                box_pos[box_idx, 0],  # Box's old first cell
                jnp.where(is_at_box & (~this_box_moved), agent_old, agent_pos_new)
            )
            
            return final_agent_pos
        
        # Vectorize over agents
        final_agent_positions = jax.vmap(update_agent_after_box_push)(old_agent_pos, new_agent_pos)
        
        # Compute bumped mask: agents that moved into unmovable boxes
        def check_agent_bumped(agent_old, agent_new, agent_final):
            # Agent is bumped if they moved forward into a box that didn't move
            moved = jnp.any(agent_new != agent_old)
            
            # Check if agent is at an unmoved box
            flat_box_final = final_box_cells.reshape((-1, 2))
            agent_at_box = jnp.any(jnp.all(flat_box_final == agent_final[None, :], axis=1))
            
            # Find which box
            box_cell_matches = jnp.any(jnp.all(final_box_cells == agent_final[None, None, :], axis=2), axis=1)
            box_idx = jnp.argmax(box_cell_matches.astype(jnp.int32))
            box_didnt_move = ~boxes_moved[box_idx]
            
            bumped = moved & agent_at_box & box_didnt_move & box_cell_matches[box_idx]
            return jnp.where(bumped, 1.0, 0.0)
        
        bumped_mask = jax.vmap(check_agent_bumped)(old_agent_pos, new_agent_pos, final_agent_positions)

        return final_box_cells, tentative_push_counts, final_agent_positions, bumped_mask

    def _check_box_goals(self, box_pos: chex.Array) -> chex.Array:
        """
        Check which boxes are fully within the goal area (top row).
        Returns boolean array [num_boxes].
        """
        ys = box_pos[..., 1]
        all_top = jnp.all(ys == (self.grid_size - 1.0), axis=1)
        return all_top

    def _calculate_rewards(
        self,
        state: State,
        new_agent_pos: chex.Array,
        new_box_pos: chex.Array,
        new_box_at_goal: chex.Array,
        actions: chex.Array,
        bumped_mask: chex.Array,
        box_moved_by_count: chex.Array
    ) -> chex.Array:
        """
        Compute rewards per agent.
        - step_penalty: -0.1 per agent per timestep
        - bump_penalty: -5 per agent for bumping
        - small_box_goal_reward: +10 per agent when small box reaches goal
        - large_box_coop_goal_reward: +100 per agent when large box reaches goal cooperatively
        """
        # Base step penalty
        rewards = jnp.full(self.num_agents, self.step_penalty)

        # Bump penalties
        rewards = rewards + bumped_mask * self.bump_penalty

        # Newly reached goals
        prev = state.box_at_goal
        new = new_box_at_goal
        newly_reached = jnp.logical_and(new, jnp.logical_not(prev))

        # Small box rewards
        small_mask = self.box_sizes == 1
        newly_small_reached = jnp.logical_and(newly_reached, small_mask)
        n_small_new = jnp.sum(jnp.where(newly_small_reached, 1.0, 0.0))
        rewards = rewards + n_small_new * self.small_box_goal_reward

        # Large box cooperative reward
        large_mask = self.box_sizes > 1
        newly_large_reached = jnp.logical_and(newly_reached, large_mask)
        coop_counts = box_moved_by_count * jnp.where(newly_large_reached, 1, 0)
        coop_condition = jnp.any(coop_counts >= 2)
        rewards = rewards + jnp.where(coop_condition, self.large_box_coop_goal_reward, 0.0)

        return rewards

    @partial(jax.jit, static_argnums=[0])
    def get_obs(self, state: State) -> Dict[str, chex.Array]:
        """Construct observations for each agent - discrete 5 values.
        
        Observation values:
        0 = empty field
        1 = wall
        2 = other agent
        3 = small box
        4 = large box
        """
        
        def get_obs_for_agent(agent_idx, agent_pos, agent_orient, box_pos, all_agent_pos):
            # Get position in front of agent
            front_pos = agent_pos + self.directions[agent_orient]
            
            # Check bounds
            in_bounds = jnp.all((front_pos >= 0) & (front_pos < self.grid_size))
            
            def check_wall():
                return jnp.array(1, dtype=jnp.int32)  # wall
            
            def check_content():
                # Check for other agent - avoid boolean indexing
                # Check all agents and exclude current agent using where
                def check_other_agent(other_idx):
                    is_other = other_idx != agent_idx
                    other_pos = all_agent_pos[other_idx]
                    matches = jnp.all(other_pos == front_pos)
                    return jnp.where(is_other, matches, False)
                
                # Check all agents
                other_agent_checks = jax.vmap(check_other_agent)(jnp.arange(self.num_agents))
                is_other_agent = jnp.any(other_agent_checks)
                
                # Check all boxes at once using vmap
                def check_box(box_cells, box_size):
                    matches = jnp.any(jnp.all(box_cells == front_pos, axis=1))
                    is_large = box_size > 1
                    return jnp.where(matches & is_large, 4,  # large box
                                   jnp.where(matches, 3, 0))  # small box or no match
                
                box_types = jax.vmap(check_box, in_axes=(0, 0))(box_pos, self.box_sizes)
                # Get the maximum box type (prioritize large box if multiple match)
                box_type = jnp.max(box_types)
                
                # Priority: other agent > box > empty
                obs = jnp.where(is_other_agent, 2, box_type)
                return obs.astype(jnp.int32)
            
            obs = jax.lax.cond(in_bounds, check_content, check_wall)
            # Return as 1D array to allow concatenation in wrappers
            return obs[None]  # Shape: (1,)
        
        # Vectorize over agents
        obs_array = jax.vmap(
            get_obs_for_agent, 
            in_axes=(0, 0, 0, None, None)
        )(
            jnp.arange(self.num_agents),
            state.agent_pos,
            state.agent_orient,
            state.box_pos,
            state.agent_pos
        )
        
        return {agent: obs_array[i] for i, agent in enumerate(self.agents)}

    @property
    def name(self) -> str:
        return "BoxPushing"

    def agent_classes(self) -> Dict[str, list]:
        return {"agents": self.agents}

