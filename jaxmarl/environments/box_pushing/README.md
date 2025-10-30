# Box Pushing Environment

A multi-agent reinforcement learning environment where agents must cooperate to push boxes to target locations. This environment is implemented in JAX following the JaxMARL patterns and is inspired by the MacroMARL box pushing implementation.

## Environment Description

In this environment, multiple agents must work together to push boxes to their designated target locations. The environment features:

- **Cooperative Multi-Agent Task**: Agents must coordinate to successfully complete the task
- **Grid-Based World**: Discrete 2D grid where agents and boxes can move
- **Box Pushing Mechanics**: Agents can push boxes by moving into them from adjacent positions
- **Target-Based Rewards**: Agents receive rewards for pushing boxes to target locations
- **Collision Detection**: Agents are penalized for colliding with each other

## Environment Parameters

- `num_agents` (int): Number of agents in the environment (default: 2)
- `num_boxes` (int): Number of boxes to push (default: 2)
- `grid_size` (int): Size of the grid world (default: 8x8)
- `max_steps` (int): Maximum steps per episode (default: 50)
- `push_reward` (float): Reward for pushing a box (default: 10.0)
- `target_reward` (float): Reward for getting a box to target (default: 20.0)
- `step_penalty` (float): Penalty per step (default: -0.1)
- `collision_penalty` (float): Penalty for agent collisions (default: -1.0)

## Action Space

Each agent has a discrete action space with 6 actions:
- `0`: No operation (noop)
- `1`: Move right
- `2`: Move up
- `3`: Move left
- `4`: Move down
- `5`: Push (only works when adjacent to a box)

## Observation Space

Each agent observes:
- Own position (2D coordinates)
- Own orientation (0=right, 1=up, 2=left, 3=down)
- All box positions (num_boxes × 2D coordinates)
- All target positions (num_targets × 2D coordinates)
- Other agents' positions ((num_agents-1) × 2D coordinates)

## State Representation

The environment state includes:
- `agent_pos`: Agent positions [num_agents, 2]
- `agent_orient`: Agent orientations [num_agents]
- `box_pos`: Box positions [num_boxes, 2]
- `target_pos`: Target positions [num_targets, 2]
- `box_at_target`: Whether each box is at its target [num_boxes]
- `step`: Current step number
- `done`: Whether the episode is done

## Reward Structure

Agents receive rewards for:
- **Pushing boxes**: +10.0 when a box is moved
- **Reaching targets**: +20.0 when a box reaches its target
- **Step penalty**: -0.1 per step to encourage efficiency
- **Collision penalty**: -1.0 when agents are too close to each other

## Usage Example

```python
import jax
from jaxmarl.environments.box_pushing import BoxPushing

# Create environment
env = BoxPushing(num_agents=2, num_boxes=2, grid_size=8)

# Reset environment
key = jax.random.PRNGKey(0)
obs, state = env.reset(key)

# Take random actions
key, key_act = jax.random.split(key)
actions = {
    agent: env.action_space(agent).sample(key_act) 
    for agent in env.agents
}

# Step environment
obs, state, rewards, dones, info = env.step(key, state, actions)
```

## Key Features

1. **JAX Compatibility**: Fully implemented using JAX for high-performance computing
2. **JIT Compilation**: All functions are JIT-compiled for optimal performance
3. **Vectorized Operations**: Efficient vectorized operations for multi-agent scenarios
4. **Modular Design**: Follows JaxMARL environment patterns for easy integration
5. **Configurable**: Highly configurable parameters for different scenarios

## Implementation Details

The environment is built on top of the JaxMARL `MultiAgentEnv` base class and follows the same patterns as other environments in the repository. Key implementation features:

- **Functional Programming**: All state transitions are pure functions
- **Immutable State**: Uses Flax struct dataclass for immutable state representation
- **Efficient Box Pushing**: Sophisticated box pushing mechanics that handle multiple agents
- **Collision Detection**: Built-in collision detection and resolution
- **Target Assignment**: Automatic target-box assignment and completion tracking

## Testing

To test the environment:

```bash
# Basic syntax and structure test
python3 test_box_pushing.py

# Full functionality test (requires JAX)
python3 jaxmarl/environments/box_pushing/box_pushing.py
```

## Dependencies

- JAX
- JAXlib
- Chex
- Flax
- NumPy

## References

This implementation is inspired by the MacroMARL box pushing environment and follows the design patterns established in the JaxMARL repository.
