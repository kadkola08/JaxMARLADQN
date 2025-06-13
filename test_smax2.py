#!/usr/bin/env python

import jax
import jax.numpy as jnp
from jaxmarl import make
from jaxmarl.environments.smax import map_name_to_scenario
from jaxmarl.viz.visualizer import SMAXVisualizer

# Initialize environment
scenario = map_name_to_scenario("2s3z")
env = make(
    "HeuristicEnemySMAX",
    enemy_shoots=True,
    scenario=scenario,
    use_self_play_reward=False,
    walls_cause_death=True,
    see_enemy_actions=False,
)

# Run random policy and collect state sequence
def run_random_policy(env, max_steps=100):
    rng = jax.random.PRNGKey(42)
    rng, reset_key = jax.random.split(rng)
    obs, state = env.reset(reset_key)
    breakpoint()
    
    state_seq = []
    done = False
    step_count = 0
    
    while not done and step_count < max_steps:
        rng, step_key, action_key = jax.random.split(rng, 3)
        
        # Get valid actions
        valid_actions = env.get_avail_actions(state)
        
        # Select random actions
        actions = {}
        for agent in env.agents:
            valid = valid_actions[agent]
            # Normalize valid actions to create a probability distribution
            probs = valid / jnp.sum(valid)
            action_key, subkey = jax.random.split(action_key)
            actions[agent] = jax.random.choice(subkey, jnp.arange(valid.shape[0]), p=probs)
        
        # Save state, key, and actions
        state_seq.append((step_key, state, actions))
        
        # Step the environment
        obs, state, rewards, dones, info = env.step(step_key, state, actions)
        
        # Check termination
        if "__all__" in dones:
            done = dones["__all__"]
        else:
            done = all(dones.values())
            
        step_count += 1
    
    return state_seq

# Run random policy and collect state sequence
state_seq = run_random_policy(env)
breakpoint()

# Visualize
viz = SMAXVisualizer(env, state_seq)
viz.animate(view=True, save_fname="random_policy.gif")
