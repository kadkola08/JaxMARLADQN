import os
import copy
import jax
jax.config.update('jax_default_matmul_precision', 'bfloat16')  # Use lower precision

import jax.numpy as jnp
import numpy as np
from functools import partial
from typing import NamedTuple, Dict, Union, Any

import chex
import optax
import flax.linen as nn
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState
from gymnax.wrappers.purerl import LogWrapper
import hydra
from omegaconf import OmegaConf
import gymnax
import flashbax as fbx
import wandb

from jaxmarl import make
from jaxmarl.environments.smax import map_name_to_scenario
from jaxmarl.environments.overcooked import overcooked_layouts
from jaxmarl.wrappers.baselines import (
    SMAXLogWrapper,
    MPELogWrapper,
    LogWrapper,
    CTRolloutManager,
)

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

# class RNNQNetwork(nn.Module):
#     # For COVDN, we output Q-values for continuous actions
#     action_dim: int  # Now represents the continuous action dimension
#     hidden_dim: int
#     init_scale: float = 1.0

#     @nn.compact
#     def __call__(self, hidden, obs, dones, actions=None):
#         """
#         For COVDN, the Q-network takes both observations and actions as input
#         to output Q(s,a) for continuous actions
#         """
#         embedding = nn.Dense(
#             self.hidden_dim,
#             kernel_init=orthogonal(self.init_scale),
#             bias_init=constant(0.0),
#         )(obs)
#         embedding = nn.relu(embedding)

#         rnn_in = (embedding, dones)
#         hidden, embedding = ScannedRNN()(hidden, rnn_in)

#         if actions is not None:
#             embedding = jnp.concatenate([embedding, actions], axis=-1)
#             embedding = nn.Dense(
#                 self.hidden_dim,
#                 kernel_init=orthogonal(self.init_scale),
#                 bias_init=constant(0.0),
#             )(embedding)
#             embedding = nn.relu(embedding)

#         # Output a single Q-value for the state-action pair
#         q_val = nn.Dense(
#             1,
#             kernel_init=orthogonal(self.init_scale),
#             bias_init=constant(0.0),
#         )(embedding)

#         return hidden, q_val.squeeze(-1)

class RNNQNetwork(nn.Module):
    # For COVDN, we output Q-values for continuous actions
    action_dim: int  # Now represents the continuous action dimension
    hidden_dim: int
    init_scale: float = 1.0

    @nn.compact
    def __call__(self, hidden, obs, dones, actions):
        """
        For COVDN, computes Q(s,a) for continuous actions.
        This method expects actions to be provided.
        """
        embedding = nn.Dense(
            self.hidden_dim,
            kernel_init=orthogonal(self.init_scale),
            bias_init=constant(0.0),
            name="obs_encoder"
        )(obs)
        embedding = nn.relu(embedding)

        rnn_in = (embedding, dones)
        hidden, embedding = ScannedRNN()(hidden, rnn_in)

        # Concatenate actions with the embedding
        combined = jnp.concatenate([embedding, actions], axis=-1)
        combined = nn.Dense(
            self.hidden_dim,
            kernel_init=orthogonal(self.init_scale),
            bias_init=constant(0.0),
            name="action_encoder"
        )(combined)
        combined = nn.relu(combined)

        # Output a single Q-value for the state-action pair
        q_val = nn.Dense(
            1,
            kernel_init=orthogonal(self.init_scale),
            bias_init=constant(0.0),
            name="q_output"
        )(combined)

        return hidden, q_val.squeeze(-1)
    
    @nn.compact
    def update_hidden(self, hidden, obs, dones):
        """
        Updates only the hidden state without computing Q-values.
        Used for stepping through the environment.
        """
        embedding = nn.Dense(
            self.hidden_dim,
            kernel_init=orthogonal(self.init_scale),
            bias_init=constant(0.0),
            name="obs_encoder"
        )(obs)
        embedding = nn.relu(embedding)

        rnn_in = (embedding, dones)
        hidden, _ = ScannedRNN()(hidden, rnn_in)
        
        return hidden

@chex.dataclass(frozen=True)
class Timestep:
    obs: dict
    actions: dict
    rewards: dict
    dones: dict


class CustomTrainState(TrainState):
    target_network_params: Any
    timesteps: int = 0
    n_updates: int = 0
    grad_steps: int = 0

def cem_optimization(rng, q_network_apply, params, hidden_states, obs, dones, 
                     action_dim, num_agents, batch_size, cem_config, init_mean=None, init_std=None):
    """
    Cross-Entropy Method (CEM) (Algorithm 1) from the FACMAC paper
    """
    n_iterations = 3
    n_samples = 51  
    n_top = 6

    rng, _rng = jax.random.split(rng)


    if init_mean is not None:
        mu = init_mean
    else:
        mu = jnp.zeros((num_agents, batch_size, action_dim))
    
    if init_std is not None:
        sigma = init_std
    else:
        sigma = jnp.ones((num_agents, batch_size, action_dim))

    def cem_iteration(carry, _):
        mu, sigma, rng = carry
        rng, sample_rng = jax.random.split(rng)

        sample_rngs = jax.random.split(sample_rng, num_agents)

        def sample_agent_actions(agent_rng, agent_mu, agent_sigma):
            # Sample N actions for this agent across all batches
            # Shape: (n_samples, batch_size, action_dim)
            # Sample to standard normal and scale to mean and standard deviation
            eps = jax.random.normal(agent_rng, (n_samples, batch_size, action_dim))
            sampled_actions = agent_mu[None, :, :] + agent_sigma[None, :, :] * eps
            sampled_actions = jnp.tanh(sampled_actions)  # Bound actions to [-1, 1]
            return sampled_actions

        sampled_actions = jax.vmap(sample_agent_actions)(sample_rngs, mu, sigma)
        # Shape: (num_agents, n_samples, batch_size, action_dim)

        # Evaluate Q-values for all samples
        def evaluate_sample(sample_idx):
            actions_sample = sampled_actions[:, sample_idx]

            q_vals = []
            for agent_idx in range(num_agents):
                agent_actions = actions_sample[agent_idx, np.newaxis]  # (1, batch_size, action_dim)
                agent_obs = obs[agent_idx, np.newaxis] # (1, batch_size, obs_dim)
                agent_dones = dones[agent_idx, np.newaxis]  # (1, batch_size)
                agent_hidden = jax.tree.map(lambda x: x[agent_idx], hidden_states)
                # agent_hidden = jax.tree.map(lambda x: x[agent_idx, np.newaxis], hidden_states)
                
                _, q = q_network_apply(
                    params,
                    agent_hidden,
                    agent_obs,
                    agent_dones,
                    agent_actions
                )
                # q_vals.append(q.squeeze())    
                q_vals.append(q.squeeze(0))    

            q_vals = jnp.stack(q_vals, axis=0)
            # total_q = jnp.sum(jnp.stack(q_vals, axis=0), axis=0)  # (batch_size,)
            return q_vals, actions_sample

        q_values, all_actions = jax.vmap(evaluate_sample)(jnp.arange(n_samples))
        # q_values shape: (n_samples, batch_size)

        # def update_distribution(batch_idx):
        #     batch_q_values = q_values[:, :, batch_idx] 
        #     top_indices = jnp.argsort(batch_q_values)[-n_top:]

        #     new_mus = []
        #     new_sigmas = []
        #     for agent_idx in range(num_agents):
        #         agent_top_actions = all_actions[top_indices, agent_idx, batch_idx]  # (n_top, action_dim)
        #         new_mu = jnp.mean(agent_top_actions, axis=0)
        #         new_sigma = jnp.std(agent_top_actions, axis=0) + 1e-4 

        #     return jnp.stack(new_mus, axis=0), jnp.stack(new_sigmas, axis=0)

        def update_agent_distribution(agent_idx):
            agent_q_values = q_values[:, agent_idx, :]  # (n_samples, batch_size)
            agent_actions = all_actions[:, agent_idx, :, :]  # (n_samples, batch_size, action_dim)

            def update_batch_distribution(batch_idx):
                batch_q_values = agent_q_values[:, batch_idx]  # (n_samples,)
                top_indices = jnp.argsort(batch_q_values)[-n_top:]  # (n_top,)

                top_actions = agent_actions[top_indices, batch_idx, :]  # (n_top, action_dim)

                new_mu = jnp.mean(top_actions, axis=0)  # (action_dim,)
                new_sigma = jnp.std(top_actions, axis=0) + 1e-4  # (action_dim,)
                
                return new_mu, new_sigma

            batch_mus, batch_sigmas = jax.vmap(update_batch_distribution)(jnp.arange(batch_size))
            # batch_mus shape: (batch_size, action_dim)
            # batch_sigmas shape: (batch_size, action_dim)
            
            return batch_mus, batch_sigmas

        # Update distributions for all batch elements
        # new_mu, new_sigma = jax.vmap(update_distribution)(jnp.arange(batch_size))
        new_mu, new_sigma = jax.vmap(update_agent_distribution)(jnp.arange(num_agents))
        # Transpose to get (num_agents, batch_size, action_dim)
        # new_mu = jnp.transpose(new_mu, (1, 0, 2))
        # new_sigma = jnp.transpose(new_sigma, (1, 0, 2))
        
        return (new_mu, new_sigma, rng), None

    (final_mu, _, _), _ = jax.lax.scan(cem_iteration, (mu, sigma, rng), None, n_iterations)
    
    return jnp.tanh(final_mu)  

def make_train(config, env):

    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    
    # CEM configuration
    cem_config = {
        "N_ITERATIONS": config.get("CEM_ITERATIONS", 3),
        "N_SAMPLES": config.get("CEM_SAMPLES", 64),
        "N_TOP": config.get("CEM_TOP", 6),
    }

    config["EXPLORATION_NOISE_START"] = 0.3  # Initial noise std
    config["EXPLORATION_NOISE_END"] = 0.01   # Final noise std
    config["EXPLORATION_NOISE_DECAY"] = 0.5  # Fraction of training for decay

    noise_std_scheduler = optax.linear_schedule(
    init_value=config["EXPLORATION_NOISE_START"],
    end_value=config["EXPLORATION_NOISE_END"],
    transition_steps=config["EXPLORATION_NOISE_DECAY"] * config["NUM_UPDATES"],
    )

    eps_scheduler = optax.linear_schedule(
        init_value=config["EPS_START"],
        end_value=config["EPS_FINISH"],
        transition_steps=config["EPS_DECAY"] * config["NUM_UPDATES"],
    )

    def batchify(x: dict):
        return jnp.stack([x[agent] for agent in env.agents], axis=0)

    def unbatchify(x: jnp.ndarray):
        return {agent: x[i] for i, agent in enumerate(env.agents)}

    def train(rng):

        # INIT ENV
        original_seed = rng[0]
        rng, _rng = jax.random.split(rng)
        wrapped_env = CTRolloutManager(env, batch_size=config["NUM_ENVS"])
        test_env = CTRolloutManager(
            env, batch_size=config["TEST_NUM_ENVS"]
        )  # batched env for testing

        # INIT NETWORK AND OPTIMIZER
        # For continuous actions, action_dim is the continuous action space dimension
        action_dim = env.action_spaces[env.agents[0]].shape[0] if hasattr(env.action_spaces[env.agents[0]], 'shape') else 1
        
        network = RNNQNetwork(
            action_dim=action_dim,
            hidden_dim=config["HIDDEN_SIZE"],
        )

        def create_agent(rng):
            init_x = (
                jnp.zeros((1, 1, wrapped_env.obs_size)),  # obs
                jnp.zeros((1, 1)),  # dones
                jnp.zeros((1, 1, action_dim)),  # actions for continuous case
            )
            init_hs = ScannedRNN.initialize_carry(config["HIDDEN_SIZE"], 1)
            network_params = network.init(rng, init_hs, *init_x)

            lr_scheduler = optax.linear_schedule(
                init_value=config["LR"],
                end_value=1e-10,
                transition_steps=(config["NUM_EPOCHS"]) * config["NUM_UPDATES"],
            )

            lr = lr_scheduler if config.get("LR_LINEAR_DECAY", False) else config["LR"]

            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.radam(learning_rate=lr),
            )

            train_state = CustomTrainState.create(
                apply_fn=network.apply,
                params=network_params,
                target_network_params=network_params,
                tx=tx,
            )
            return train_state

        rng, _rng = jax.random.split(rng)
        train_state = create_agent(rng)

        # INIT BUFFER
        def _env_sample_step(env_state, unused):
            rng, key_a, key_s = jax.random.split(jax.random.PRNGKey(0), 3)
            # For continuous actions, sample from action space
            key_a = jax.random.split(key_a, env.num_agents)
            actions = {}
            for i, agent in enumerate(env.agents):
                # Sample continuous actions
                actions[agent] = jax.random.uniform(
                    key_a[i], 
                    (config["NUM_ENVS"], action_dim),
                    minval=-1.0,
                    maxval=1.0
                )
            
            obs, env_state, rewards, dones, infos = wrapped_env.batch_step(
                key_s, env_state, actions
            )
            timestep = Timestep(
                obs=obs,
                actions=actions,
                rewards=rewards,
                dones=dones,
            )
            return env_state, timestep

        _, _env_state = wrapped_env.batch_reset(rng)
        _, sample_traj = jax.lax.scan(
            _env_sample_step, _env_state, None, config["NUM_STEPS"]
        )
        sample_traj_unbatched = jax.tree.map(
            lambda x: x[:, 0], sample_traj
        )  # remove the NUM_ENV dim
        buffer = fbx.make_trajectory_buffer(
            max_length_time_axis=int(config["BUFFER_SIZE"] // config["NUM_ENVS"]),
            min_length_time_axis=config["BUFFER_BATCH_SIZE"],
            sample_batch_size=config["BUFFER_BATCH_SIZE"],
            add_batch_size=config["NUM_ENVS"],
            sample_sequence_length=1,
            period=1,
        )
        buffer_state = buffer.init(sample_traj_unbatched)

        # TRAINING LOOP
        def _update_step(runner_state, unused):

            train_state, buffer_state, test_state, rng = runner_state

            # SAMPLE PHASE
            def _step_env(carry, _):
                hs, last_obs, last_dones, env_state, prev_actions, rng = carry
                rng, rng_a, rng_s, rng_explore, rng_noise = jax.random.split(rng, 5)

                _obs = batchify(last_obs)[:, np.newaxis]
                _dones = batchify(last_dones)[:, np.newaxis]

                eps = eps_scheduler(train_state.n_updates)
                
                use_random = jax.random.uniform(rng_explore) < eps

                noise_std = noise_std_scheduler(train_state.n_updates)

                # Reset previous actions where episodes ended
                reset_mask = last_dones["__all__"][np.newaxis, :, np.newaxis]

                init_mean = jnp.where(
                    reset_mask,
                    jnp.zeros((env.num_agents, config["NUM_ENVS"], action_dim)),
                    prev_actions * 0.7
                )
    
                init_std = jnp.where(
                    reset_mask,
                    jnp.ones((env.num_agents, config["NUM_ENVS"], action_dim)),
                    jnp.ones((env.num_agents, config["NUM_ENVS"], action_dim)) * 0.5
                )

                # CEM optimization for action selection
                optimal_actions = cem_optimization(
                    rng_a,
                    network.apply,
                    train_state.params,
                    hs,
                    _obs.squeeze(1),  # Remove time dimension
                    _dones.squeeze(1),
                    action_dim,
                    env.num_agents,
                    config["NUM_ENVS"],
                    cem_config,
                    init_mean=init_mean,
                    init_std=init_std
                )

                noise = jax.random.normal(
                    rng_noise, 
                    (env.num_agents, config["NUM_ENVS"], action_dim)
                ) * noise_std

                actions = jnp.clip(optimal_actions + noise, -1.0, 1.0)
                    
                # Random actions for exploration
                # random_actions = jax.random.uniform(
                    # rng_a,
                    # (env.num_agents, config["NUM_ENVS"], action_dim),
                    # minval=-1.0,
                    # maxval=1.0
                # )
                    # 
                # actions = jax.lax.cond(
                    # use_random,
                    # lambda _: random_actions,
                    # lambda _: optimal_actions,
                    # None
                # )

                actions = unbatchify(actions)

                noise = jax.random.normal(
                    rng_noise, 
                    (env.num_agents, config["NUM_ENVS"], action_dim)
                ) * noise_std

                new_hs = jax.vmap(
                    lambda p, h, o, d: network.apply(p, h, o, d, method=network.update_hidden),
                    in_axes=(None, 0, 0, 0)
                )(
                    train_state.params,
                    hs,
                    _obs,
                    _dones
                )

                new_obs, new_env_state, rewards, dones, infos = wrapped_env.batch_step(
                    rng_s, env_state, actions
                )
                timestep = Timestep(
                    obs=last_obs,
                    actions=actions,
                    rewards=jax.tree.map(lambda x: config.get("REW_SCALE", 1) * x, rewards),
                    dones=last_dones,
                )
                return (new_hs, new_obs, dones, new_env_state, batchify(actions), rng), (timestep, infos)

            # Step the environment
            rng, _rng = jax.random.split(rng)
            init_obs, env_state = wrapped_env.batch_reset(_rng)
            init_dones = {
                agent: jnp.zeros((config["NUM_ENVS"]), dtype=bool)
                for agent in env.agents + ["__all__"]
            }
            init_hs = ScannedRNN.initialize_carry(
                config["HIDDEN_SIZE"], len(env.agents), config["NUM_ENVS"]
            )
            init_prev_actions = jnp.zeros((env.num_agents, config["NUM_ENVS"], action_dim)) # or None??
            expl_state = (init_hs, init_obs, init_dones, env_state, init_prev_actions)
            rng, _rng = jax.random.split(rng)
            _, (timesteps, infos) = jax.lax.scan(
                _step_env,
                (*expl_state, _rng),
                None,
                config["NUM_STEPS"],
            )

            train_state = train_state.replace(
                timesteps=train_state.timesteps
                + config["NUM_STEPS"] * config["NUM_ENVS"]
            )

            # BUFFER UPDATE
            buffer_traj_batch = jax.tree.map(
                lambda x: jnp.swapaxes(x, 0, 1)[:, np.newaxis],
                timesteps,
            )
            buffer_state = buffer.add(buffer_state, buffer_traj_batch)

            # NETWORKS UPDATE
            def _learn_phase(carry, _):

                train_state, rng = carry
                rng, _rng = jax.random.split(rng)
                minibatch = buffer.sample(buffer_state, _rng).experience
                minibatch = jax.tree.map(
                    lambda x: jnp.swapaxes(x[:, 0], 0, 1),
                    minibatch,
                )

                # Preprocess network input
                init_hs = ScannedRNN.initialize_carry(
                    config["HIDDEN_SIZE"],
                    len(env.agents),
                    config["BUFFER_BATCH_SIZE"],
                )
                _obs = batchify(minibatch.obs)
                _dones = batchify(minibatch.dones)
                _actions = batchify(minibatch.actions)

                # Get next Q-values using CEM for optimal next actions
                if config.get("USE_CEM_TARGET", True):
                    # Use CEM to find optimal next actions for target values
                    next_optimal_actions = []
                    for t in range(_obs.shape[1] - 1):  # For each timestep except last
                        rng, cem_rng = jax.random.split(rng)
                        next_actions_t = cem_optimization(
                            cem_rng,
                            network.apply,
                            train_state.target_network_params,
                            init_hs,
                            _obs[:, t + 1],
                            _dones[:, t + 1],
                            action_dim,
                            env.num_agents,
                            config["BUFFER_BATCH_SIZE"],
                            cem_config
                        )
                        next_optimal_actions.append(next_actions_t)
                    next_optimal_actions = jnp.stack(next_optimal_actions, axis=1)
                else:
                    # Use current actions from buffer as approximation
                    next_optimal_actions = _actions[:, 1:]

                def _loss_fn(params):
                    q_vals = []
                    for agent_idx in range(env.num_agents):
                        _, agent_q = network.apply(
                            params,
                            init_hs[agent_idx],
                            _obs[agent_idx],
                            _dones[agent_idx],
                            _actions[agent_idx]
                        )
                        q_vals.append(agent_q)

                    q_vals = jnp.stack(q_vals, axis=0)  # (num_agents, timesteps, batch_size)

                    # Get target Q-values for next states
                    q_next_vals = []
                    for agent_idx in range(env.num_agents):
                        _, agent_q_next = network.apply(
                            train_state.target_network_params,
                            init_hs[agent_idx],
                            _obs[agent_idx, 1:],
                            _dones[agent_idx, 1:],
                            next_optimal_actions[agent_idx]
                        )
                        q_next_vals.append(agent_q_next)
                    
                    q_next_vals = jnp.stack(q_next_vals, axis=0)

                    # VDN: Sum Q-values across agents
                    vdn_q = jnp.sum(q_vals, axis=0)[:-1]  # Current Q-values
                    vdn_q_next = jnp.sum(q_next_vals, axis=0)  # Next Q-values

                    # Compute TD targets
                    vdn_target = (
                        minibatch.rewards["__all__"][:-1]
                        + (1 - minibatch.dones["__all__"][:-1])
                        * config["GAMMA"]
                        * vdn_q_next
                    )

                    loss = jnp.mean((vdn_q - jax.lax.stop_gradient(vdn_target)) ** 2)

                    return loss, (vdn_q.mean(), vdn_target.mean())

                (loss, aux), grads = jax.value_and_grad(_loss_fn, has_aux=True)(
                    train_state.params
                )
                qvals, vdn_target = aux
                train_state = train_state.apply_gradients(grads=grads)
                train_state = train_state.replace(
                    grad_steps=train_state.grad_steps + 1,
                )
                return (train_state, rng), (loss, qvals, vdn_target)

            rng, _rng = jax.random.split(rng)
            is_learn_time = (
                buffer.can_sample(buffer_state)
            ) & (
                train_state.timesteps > config["LEARNING_STARTS"]
            )
            (train_state, rng), (loss, qvals, vdn_target) = jax.lax.cond(
                is_learn_time,
                lambda train_state, rng: jax.lax.scan(
                    _learn_phase, (train_state, rng), None, config["NUM_EPOCHS"]
                ),
                lambda train_state, rng: (
                    (train_state, rng),
                    (
                        jnp.zeros(config["NUM_EPOCHS"]),
                        jnp.zeros(config["NUM_EPOCHS"]),
                        jnp.zeros(config["NUM_EPOCHS"]),
                    ),
                ),
                train_state,
                _rng,
            )

            # Update target network
            train_state = jax.lax.cond(
                train_state.n_updates % config["TARGET_UPDATE_INTERVAL"] == 0,
                lambda train_state: train_state.replace(
                    target_network_params=optax.incremental_update(
                        train_state.params,
                        train_state.target_network_params,
                        config["TAU"],
                    )
                ),
                lambda train_state: train_state,
                operand=train_state,
            )

            # UPDATE METRICS
            train_state = train_state.replace(n_updates=train_state.n_updates + 1)
            metrics = {
                "env_step": train_state.timesteps,
                "update_steps": train_state.n_updates,
                "grad_steps": train_state.grad_steps,
                "loss": loss.mean(),
                "qvals": qvals.mean(),
                "vdn_target": vdn_target.mean()
            }
            metrics.update(jax.tree.map(lambda x: x.mean(), infos))

            if config.get("TEST_DURING_TRAINING", True):
                rng, _rng = jax.random.split(rng)
                test_state = jax.lax.cond(
                    train_state.n_updates
                    % int(config["NUM_UPDATES"] * config["TEST_INTERVAL"])
                    == 0,
                    lambda _: get_greedy_metrics(_rng, train_state),
                    lambda _: test_state,
                    operand=None,
                )
                metrics.update({"test_" + k: v for k, v in test_state.items()})

            if config["WANDB_MODE"] != "disabled":

                def callback(metrics, original_seed):
                    if config.get('WANDB_LOG_ALL_SEEDS', False):
                        metrics.update(
                            {f"rng{int(original_seed)}/{k}": v for k, v in metrics.items()}
                        )
                    wandb.log(metrics)

                jax.debug.callback(callback, metrics, original_seed)

            runner_state = (train_state, buffer_state, test_state, rng)

            return runner_state, None

        def get_greedy_metrics(rng, train_state):
            """Test greedy policy during training using CEM for action selection"""
            if not config.get("TEST_DURING_TRAINING", True):
                return None

            params = train_state.params
            def _greedy_env_step(step_state, unused):
                params, env_state, last_obs, last_dones, hstate, rng = step_state
                rng, key_s, key_cem = jax.random.split(rng, 3)

                _obs = batchify(last_obs)
                _dones = batchify(last_dones)

                # Use CEM for greedy action selection (no exploration)
                optimal_actions = cem_optimization(
                    key_cem,
                    network.apply,
                    params,
                    hstate,
                    _obs,
                    _dones,
                    action_dim,
                    env.num_agents,
                    config["TEST_NUM_ENVS"],
                    cem_config
                )

                actions = unbatchify(optimal_actions)

                # Update hidden states
                # hstate, _ = jax.vmap(network.apply, in_axes=(None, 0, 0, 0, None))(
                #     params,
                #     hstate,
                #     _obs[:, np.newaxis],
                #     _dones[:, np.newaxis],
                #     None
                # )

                hstate = jax.vmap(
                    lambda p, h, o, d: network.apply(p, h, o, d, method=network.update_hidden),
                    in_axes=(None, 0, 0, 0)
                )(
                    params,
                    hstate,
                    _obs[:, np.newaxis],
                    _dones[:, np.newaxis],
                )

                obs, env_state, rewards, dones, infos = test_env.batch_step(
                    key_s, env_state, actions
                )
                step_state = (params, env_state, obs, dones, hstate, rng)
                return step_state, (rewards, dones, infos)

            rng, _rng = jax.random.split(rng)
            init_obs, env_state = test_env.batch_reset(_rng)
            init_dones = {
                agent: jnp.zeros((config["TEST_NUM_ENVS"]), dtype=bool)
                for agent in env.agents + ["__all__"]
            }
            rng, _rng = jax.random.split(rng)
            hstate = ScannedRNN.initialize_carry(
                config["HIDDEN_SIZE"], len(env.agents), config["TEST_NUM_ENVS"]
            )
            step_state = (
                params,
                env_state,
                init_obs,
                init_dones,
                hstate,
                _rng,
            )
            step_state, (rewards, dones, infos) = jax.lax.scan(
                _greedy_env_step, step_state, None, config["TEST_NUM_STEPS"]
            )
            metrics = jax.tree.map(
                lambda x: jnp.nanmean(
                    jnp.where(
                        infos["returned_episode"],
                        x,
                        jnp.nan,
                    )
                ),
                infos,
            )
            return metrics

        rng, _rng = jax.random.split(rng)
        test_state = get_greedy_metrics(_rng, train_state)

        # Train
        rng, _rng = jax.random.split(rng)
        runner_state = (train_state, buffer_state, test_state, _rng)

        runner_state, metrics = jax.lax.scan(
            _update_step, runner_state, None, config["NUM_UPDATES"]
        )

        return {"runner_state": runner_state, "metrics": metrics}

    return train


def env_from_config(config):
    env_name = config["ENV_NAME"]
    # smax init needs a scenario
    if "smax" in env_name.lower():
        config["ENV_KWARGS"]["scenario"] = map_name_to_scenario(config["MAP_NAME"])
        env_name = f"{config['ENV_NAME']}_{config['MAP_NAME']}"
        env = make(config["ENV_NAME"], **config["ENV_KWARGS"])
        env = SMAXLogWrapper(env)
    # overcooked needs a layout
    elif "overcooked" in env_name.lower():
        env_name = f"{config['ENV_NAME']}_{config['ENV_KWARGS']['layout']}"
        config["ENV_KWARGS"]["layout"] = overcooked_layouts[
            config["ENV_KWARGS"]["layout"]
        ]
        env = make(config["ENV_NAME"], **config["ENV_KWARGS"])
        env = LogWrapper(env)
    elif "mpe" in env_name.lower():
        env = make(config["ENV_NAME"], **config["ENV_KWARGS"])
        env = MPELogWrapper(env)
    else:
        env = make(config["ENV_NAME"], **config["ENV_KWARGS"])
        env = LogWrapper(env)
    return env, env_name


def single_run(config):

    config = {**config, **config["alg"]}  # merge the alg config with the main config
    
    # Add COVDN-specific configs
    config["ALG_NAME"] = "covdn_rnn"
    config["USE_CEM"] = config.get("USE_CEM", True)
    config["USE_CEM_TARGET"] = config.get("USE_CEM_TARGET", True)
    config["CEM_ITERATIONS"] = config.get("CEM_ITERATIONS", 3)
    config["CEM_SAMPLES"] = config.get("CEM_SAMPLES", 64)
    config["CEM_TOP"] = config.get("CEM_TOP", 6)
    
    print("Config:\n", OmegaConf.to_yaml(config))

    alg_name = config.get("ALG_NAME", "covdn_rnn")
    env, env_name = env_from_config(copy.deepcopy(config))

    wandb.init(
        entity=config["ENTITY"],
        project=config["PROJECT"],
        tags=[
            alg_name.upper(),
            env_name.upper(),
            f"jax_{jax.__version__}",
        ],
        name=f"{alg_name}_{env_name}",
        config=config,
        mode=config["WANDB_MODE"],
    )

    rng = jax.random.PRNGKey(config["SEED"])

    rngs = jax.random.split(rng, config["NUM_SEEDS"])
    train_vjit = jax.jit(jax.vmap(make_train(config, env)))
    outs = jax.block_until_ready(train_vjit(rngs))

    # Save params
    if config.get("SAVE_PATH", None) is not None:
        from jaxmarl.wrappers.baselines import save_params

        model_state = outs["runner_state"][0]
        save_dir = os.path.join(config["SAVE_PATH"], env_name)
        os.makedirs(save_dir, exist_ok=True)
        OmegaConf.save(
            config,
            os.path.join(
                save_dir, f'{alg_name}_{env_name}_seed{config["SEED"]}_config.yaml'
            ),
        )

        for i, rng in enumerate(rngs):
            params = jax.tree.map(lambda x: x[i], model_state.params)
            save_path = os.path.join(
                save_dir,
                f'{alg_name}_{env_name}_seed{config["SEED"]}_vmap{i}.safetensors',
            )
            save_params(params, save_path)


def tune(default_config):
    """Hyperparameter sweep with wandb."""

    default_config = {**default_config, **default_config["alg"]}
    default_config["ALG_NAME"] = "covdn_rnn"
    env_name = default_config["ENV_NAME"]
    alg_name = default_config.get("ALG_NAME", "covdn_rnn")
    env, env_name = env_from_config(default_config)

    def wrapped_make_train():
        wandb.init(project=default_config["PROJECT"])

        # Update the default params
        config = copy.deepcopy(default_config)
        for k, v in dict(wandb.config).items():
            config[k] = v

        print("running experiment with params:", config)

        rng = jax.random.PRNGKey(config["SEED"])
        rngs = jax.random.split(rng, config["NUM_SEEDS"])
        train_vjit = jax.jit(jax.vmap(make_train(config, env)))
        outs = jax.block_until_ready(train_vjit(rngs))

    sweep_config = {
        "name": f"{alg_name}_{env_name}",
        "method": "bayes",
        "metric": {
            "name": "test_returned_episode_returns",
            "goal": "maximize",
        },
        "parameters": {
            "LR": {
                "values": [0.005, 0.001, 0.0005, 0.0001, 0.00005]
            },
            "NUM_ENVS": {"values": [8, 32, 64, 128]},
            "CEM_ITERATIONS": {"values": [2, 3, 4, 5]},
            "CEM_SAMPLES": {"values": [32, 64, 128]},
            "CEM_TOP": {"values": [4, 6, 8, 10]},
        },
    }

    wandb.login()
    sweep_id = wandb.sweep(
        sweep_config, entity=default_config["ENTITY"], project=default_config["PROJECT"]
    )
    wandb.agent(sweep_id, wrapped_make_train, count=300)


@hydra.main(version_base=None, config_path="./config", config_name="config")
def main(config):
    config = OmegaConf.to_container(config)
    print("Config:\n", OmegaConf.to_yaml(config))
    if config["HYP_TUNE"]:
        tune(config)
    else:
        single_run(config)


if __name__ == "__main__":
    main()