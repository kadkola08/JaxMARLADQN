"""
Simple test script for the MatrixGame environment.
"""

import jax
import jax.numpy as jnp
from jaxmarl.environments.matrix_game import MatrixGame

def test_matrix_game():
    """Test the MatrixGame environment with a simple 2-agent Prisoner's Dilemma."""

    payoff_matrix = jnp.array([
    [[8, -12, -12],
     [-12, 6, 0],
     [-12, 0, 6]],
    
    [[8, -12, -12],  # Both agents get same rewards (cooperative)
     [-12, 6, 0],
     [-12, 0, 6]]
    ], dtype=jnp.float32)
    
    # Create environment with default Prisoner's Dilemma payoff matrix
    env = MatrixGame(num_agents=2, num_actions=3, observation_type='empty', payoff_matrix=payoff_matrix)
    
    print("Testing MatrixGame environment...")
    print(f"Environment: {env.name}")
    print(f"Number of agents: {env.num_agents}")
    print(f"Number of actions per agent: {env.num_actions}")
    print(f"Agents: {env.agents}")
    print()
    
    # Test reset
    key = jax.random.PRNGKey(0)
    obs, state = env.reset(key)
    print("Reset successful!")
    print(f"Initial observations: {obs}")
    print(f"Initial state - done: {state.done}, step: {state.step}")
    print()
    
    # Test step with different action combinations
    print("Testing step function with different actions...")
    breakpoint()
    
    # Test case 1: Both agents cooperate (action 0)
    actions = {"agent_0": jnp.array(0), "agent_1": jnp.array(0)}
    obs, state, rewards, dones, infos = env.step_env(key, state, actions)
    print(f"Actions: agent_0=Cooperate (0), agent_1=Cooperate (0)")
    print(f"Rewards: {rewards}")
    print(f"Dones: {dones}")
    print(f"Expected: agent_0=3, agent_1=3 (mutual cooperation)")
    print()
    
    # Test case 2: Agent 0 defects, Agent 1 cooperates
    actions = {"agent_0": jnp.array(1), "agent_1": jnp.array(0)}
    obs, state, rewards, dones, infos = env.step_env(key, state, actions)
    print(f"Actions: agent_0=Defect (1), agent_1=Cooperate (0)")
    print(f"Rewards: {rewards}")
    print(f"Dones: {dones}")
    
    # Test case 3: Agent 0 cooperates, Agent 1 defects
    actions = {"agent_0": jnp.array(0), "agent_1": jnp.array(1)}
    obs, state, rewards, dones, infos = env.step_env(key, state, actions)
    print(f"Actions: agent_0=Cooperate (0), agent_1=Defect (1)")
    print(f"Rewards: {rewards}")
    print(f"Dones: {dones}")
    
    # Test case 4: Both agents defect (action 1)
    actions = {"agent_0": jnp.array(1), "agent_1": jnp.array(1)}
    obs, state, rewards, dones, infos = env.step_env(key, state, actions)
    print(f"Actions: agent_0=Defect (1), agent_1=Defect (1)")
    print(f"Rewards: {rewards}")
    print(f"Dones: {dones}")
    
    # Test that the game terminates after one step
    print("Testing that game terminates after one step...")
    # Convert JAX arrays to Python bools for assertions (outside JIT context)
    assert bool(jnp.all(dones["__all__"])), "Game should terminate after one step!"
    assert all(bool(jnp.all(dones[agent])) for agent in env.agents), "All agents should be done!"
    
    # Test with custom payoff matrix
    print("Testing with custom payoff matrix...")
    custom_payoff = jnp.array([
        [[10, 0], [20, 5]],  # Agent 0's payoffs
        [[10, 20], [0, 5]]   # Agent 1's payoffs
    ])
    env_custom = MatrixGame(num_agents=2, num_actions=2, payoff_matrix=custom_payoff)
    obs, state = env_custom.reset(key)
    actions = {"agent_0": jnp.array(0), "agent_1": jnp.array(0)}
    obs, state, rewards, dones, infos = env_custom.step_env(key, state, actions)
    print(f"Custom payoff - Both cooperate: {rewards}")
    print(f"Expected: agent_0=10, agent_1=10")

if __name__ == "__main__":
    test_matrix_game()

