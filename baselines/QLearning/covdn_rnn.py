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
#     # homogenous agent for parameters sharing, assumes all agents have same obs and action dim
#     action_dim: int
#     hidden_dim: int
#     init_scale: float = 1.0

#     @nn.compact
#     def __call__(self, hidden, obs, dones):
#         embedding = nn.Dense(
#             self.hidden_dim,
#             kernel_init=orthogonal(self.init_scale),
#             bias_init=constant(0.0),
#         )(obs)
#         embedding = nn.relu(embedding)

#         rnn_in = (embedding, dones)
#         hidden, embedding = ScannedRNN()(hidden, rnn_in)

#         q_vals = nn.Dense(
#             self.action_dim,
#             kernel_init=orthogonal(self.init_scale),
#             bias_init=constant(0.0),
#         )(embedding)

#         return hidden, q_vals


class RNNCriticNetwork(nn.Module):
    """Q-Network for evaluation state-action pair"""
    hidden_dim: int
    init_scale: float = 1.0

    @nn.compact
    def __call__(self, hidden, obs, dones, actions):
        embedding = nn.Dense(
            self.hidden_dim,
            kernel_init=orthogonal(self.init_scale),
            bias_init=constant(0.0)
        )(obs)
        embedding = nn.relu(embedding)

        rnn_in = (embedding, dones)
        hidden, embedding = ScannedRNN()(hidden, rnn_in)

        # Concatenate state embedding with action for Q(s,a)
        state_action = jnp.concatenate([embedding, actions], axis=-1)
        
        # Q-value computation for given action
        q_vals = nn.Dense(256, kernel_init=orthogonal(self.init_scale))(state_action)
        q_vals = nn.relu(q_vals)
        q_vals = nn.Dense(128, kernel_init=orthogonal(self.init_scale))(q_vals)
        q_vals = nn.relu(q_vals)
        q_vals = nn.Dense(1, kernel_init=orthogonal(self.init_scale))(q_vals)
        
        return hidden, q_vals

class RNNActorNetwork(nn.Module):
    """Actor network for selecting actions"""
    action_dim: int
    hidden_dim: int
    init_scale: float = 1.0
    action_scale: float = 1.0

    @nn.compact
    def __call__(self, hidden, obs, dones):
        embedding = nn.Dense(
            self.hidden_dim,
            kernel_init=orthogonal(self.init_scale),
            bias_init=constant(0.0)
        )(obs)
        embedding = nn.relu(embedding)

        rnn_in = (embedding, dones)
        hidden, embedding = ScannedRNN()(hidden, rnn_in)

        actor_actions = nn.Dense(256, kernel_init=orthogonal(self.init_scale))(embedding)
        actor_actions = nn.relu(actor_actions)
        actor_actions = nn.Dense(128, kernel_init=orthogonal(self.init_scale))(actor_actions)
        actor_actions = nn.relu(actor_actions)
        actor_actions = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01))(actor_actions)
        actor_actions = nn.tanh(actor_actions) * self.action_scale 
        
        return hidden, actor_actions


@chex.dataclass(frozen=True)
class Timestep:
    obs: dict
    actions: dict
    rewards: dict
    dones: dict


# class CustomTrainState(TrainState):
#     target_network_params: Any
#     timesteps: int = 0
#     n_updates: int = 0
#     grad_steps: int = 0

class CustomTrainState(TrainState):
    target_network_params: Any
    actor_params: Any
    target_actor_params: Any
    timesteps: int = 0
    n_updates: int = 0
    grad_steps: int = 0


def make_train(config, env):

    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )

    eps_scheduler = optax.linear_schedule(
        init_value=config["EPS_START"],
        end_value=config["EPS_FINISH"],
        transition_steps=config["EPS_DECAY"] * config["NUM_UPDATES"],
    )

    def get_greedy_actions(q_vals, valid_actions):
        unavail_actions = 1 - valid_actions
        q_vals = q_vals - (unavail_actions * 1e10)
        return jnp.argmax(q_vals, axis=-1)

    # epsilon-greedy exploration
    def eps_greedy_exploration(rng, q_vals, eps, valid_actions):

        rng_a, rng_e = jax.random.split(
            rng
        )  # a key for sampling random actions and one for picking

        greedy_actions = get_greedy_actions(q_vals, valid_actions)

        # pick random actions from the valid actions
        def get_random_actions(rng, val_action):
            return jax.random.choice(
                rng,
                jnp.arange(val_action.shape[-1]),
                p=val_action * 1.0 / jnp.sum(val_action, axis=-1),
            )

        _rngs = jax.random.split(rng_a, valid_actions.shape[0])
        random_actions = jax.vmap(get_random_actions)(_rngs, valid_actions)

        chosed_actions = jnp.where(
            jax.random.uniform(rng_e, greedy_actions.shape)
            < eps,  # pick the actions that should be random
            random_actions,
            greedy_actions,
        )
        return chosed_actions

    def batchify(x: dict):
        return jnp.stack([x[agent] for agent in env.agents], axis=0)

    def unbatchify(x: jnp.ndarray):
        return {agent: x[i] for i, agent in enumerate(env.agents)}

    def add_gaussian_noise(actions, rng, noise_scale):
        noise = jax.random.normal(rng, actions.shape) * noise_scale
        return jnp.clip(actions + noise, -1.0, 1.0)

    def add_explortaion_noise(actions, noise_state, rng, test_mode, ):
        """Add exploration noise to continuous actions using JAX conditionals"""

        def no_noise(actions, noise_state, rng):
            return actions, noise_state

        def gaussian_noise(actions, noise_state, rng):
            noise_scale = config.get("EXPLORATION_NOISE", 0.1)
            noisy_actions = add_gaussian_noise(actions, rng, noise_scale)

        return jax.lax.cond(
            test_mode,
            no_noise,
            gaussian_noise,
            actions, noise_state, rng
        )

    def cem_action_selection(critic_network, critic_params, actior_network, actor_params,
                            hidden_critic, hidden_actor, obs, dones, rng):
        """CEM optimization for finding best continuous actions"""
        num_samples = config.get("CEM_SAMPLES", 64)
        num_elites = config.get("CEM_ELITES", 6)
        num_iters = config.get("CEM_ITERS", 2)

        batch_size = obs.shape[0]
        action_dim = actor_network.action_dim

        _, init_actions = actor_network.apply(actor_params, hidden_actor, obs, dones)

        mu = init_actions
        std = jnp.ones_like(mu) * config.get("CEM_INIT_STD", 0.5)

        def cem_iteration(carry, _):
            mu, std, rng = carry
            rng, sample_rng = jax.random.split(rng)

            # Sample actions
            noise = jax.random.normal(sample_rng, (num_samples, *mu.shape))
            sampled_actions = mu[None, ...] + noise * std[None, ...]
            sampled_actions = jnp.clip(sampled_actions, -config["ACTION_SCALE"], config["ACTION_SCALE"])

            # Evaluate Q-values for all samples
            def eval_actions(actions):
                _, q_val = critic_network.apply(critic_params, hidden_critic, obs, dones, actions)
                return q_val.squeeze(-1)

            q_values = jax.vmap(eval_actions)(sampled_actions)

            elite_idxs = jnp.argsort(q_values, axis=0)[-num_elites:]
            elite_actions = sampled_actions[elite_idxs]

            # Update distribution
            new_mu = jnp.mean(elite_actions, axis=0)
            new_std = jnp.std(elite_actions, axis=0) + 1e-6
            
            return (new_mu, new_std, rng), (sampled_actions, q_values)

        # Run CEM iterations
        (final_mu, final_std, _), (all_actions, all_q_values) = jax.lax.scan(
            cem_iteration, (mu, std, rng), None, num_iters
        )
        
        # Return best action from last iteration
        best_idx = jnp.argmax(all_q_values[-1])
        return all_actions[-1][best_idx]


    def train(rng):

        # INIT ENV
        original_seed = rng[0]
        rng, _rng = jax.random.split(rng)
        wrapped_env = CTRolloutManager(env, batch_size=config["NUM_ENVS"])
        test_env = CTRolloutManager(
            env, batch_size=config["TEST_NUM_ENVS"]
        )  # batched env for testing (has different batch size)

        # # INIT NETWORK AND OPTIMIZER
        # network = RNNQNetwork(
        #     action_dim=wrapped_env.max_action_space,
        #     hidden_dim=config["HIDDEN_SIZE"],
        # )

        # INIT NETWORKS AND OPTIMIZERS
        critic_network = RNNCriticNetwork(
            hidden_dim=config["HIDDEN_SIZE"],
        )
        
        actor_network = RNNActorNetwork(
            action_dim=action_dim,
            hidden_dim=config["HIDDEN_SIZE"],
            action_scale=config.get("ACTION_SCALE", 1.0),
        )

        def create_agent(rng):
            rng, critic_rng, actor_rng = jax.random.split(rng, 3)

            init_obs = jnp.zeros((1, 1, wrapped_env.obs_size))
            init_dones = jnp.zeros((1, 1))
            init_actions = jnp.zeros((1, 1, action_dim))
            init_hs = ScannedRNN.initialize_carry(config["HIDDEN_SIZE"], 1)

            critic_params = critic_network.init(
                critic_rng, init_hs, init_obs, init_dones, init_actions
            )
            
            # Initialize actor
            actor_params = actor_network.init(
                actor_rng, init_hs, init_obs, init_dones
            )

            # Learning rate schedulers
            lr_scheduler = optax.linear_schedule(
                init_value=config["LR"],
                end_value=1e-10,
                transition_steps=(config["NUM_EPOCHS"]) * config["NUM_UPDATES"],
            )

            lr = jax.lax.cond(
                config.get("LR_LINEAR_DECAY", False),
                lambda _: lr_scheduler,
                lambda _: config["LR"],
                None
            )

            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=lr),
            )

            train_state = CustomTrainState.create(
                apply_fn=critic_network.apply,
                params=critic_params,
                target_network_params=critic_params,
                actor_params=actor_params,
                target_actor_params=actor_params,
                tx=tx,
                ou_state=None,
            )
            return train_state

            # init_x = (
            #     jnp.zeros(
            #         (1, 1, wrapped_env.obs_size)
            #     ),  # (time_step, batch_size, obs_size)
            #     jnp.zeros((1, 1)),  # (time_step, batch size)
            # )
            # init_hs = ScannedRNN.initialize_carry(
            #     config["HIDDEN_SIZE"], 1
            # )  # (batch_size, hidden_dim)
            # network_params = network.init(rng, init_hs, *init_x)

            # lr_scheduler = optax.linear_schedule(
            #     init_value=config["LR"],
            #     end_value=1e-10,
            #     transition_steps=(config["NUM_EPOCHS"]) * config["NUM_UPDATES"],
            # )

            # lr = lr_scheduler if config.get("LR_LINEAR_DECAY", False) else config["LR"]

            # tx = optax.chain(
            #     optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
            #     optax.radam(learning_rate=lr),
            # )

            # train_state = CustomTrainState.create(
            #     apply_fn=network.apply,
            #     params=network_params,
            #     target_network_params=network_params,
            #     tx=tx,
            # )
            # return train_state

        rng, _rng = jax.random.split(rng)
        train_state = create_agent(rng)

        # INIT BUFFER
        # to initalize the buffer is necessary to sample a trajectory to know its strucutre
        def _env_sample_step(env_state, unused):
            rng, key_a, key_s = jax.random.split(
                jax.random.PRNGKey(0), 3
            )  # use a dummy rng here
            key_a = jax.random.split(key_a, env.num_agents)
            actions = {}
            for i, agent in enumerate(env.agents):
                actions[agent] = jax.random.uniform(
                    key_a[i],
                    (config["NUM_ENVS"], action_dim),
                    minval=-1,
                    maxval=1
                )

            # actions = {
            #     agent: wrapped_env.batch_sample(key_a[i], agent)
            #     for i, agent in enumerate(env.agents)
            # }
            # avail_actions = wrapped_env.get_valid_actions(env_state.env_state)
            obs, env_state, rewards, dones, infos = wrapped_env.batch_step(
                key_s, env_state, actions
            )
            timestep = Timestep(
                obs=obs,
                actions=actions,
                rewards=rewards,
                dones=dones,
                avail_actions=avail_actions,
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

        exploration_strategy_int = 0 # Gaussian

        # TRAINING LOOP
        def _update_step(runner_state, unused):

            train_state, buffer_state, test_state, rng = runner_state

            # SAMPLE PHASE
            def _step_env(carry, _):
                hs_actor, hs_critic, last_obs, last_dones, env_state, rng = carry
                rng, rng_a, rng_s = jax.random.split(rng, 3)

                # (num_agents, 1 (dummy time), num_envs, obs_size)
                _obs = batchify(last_obs)[:, np.newaxis]
                _dones = batchify(last_dones)[:, np.newaxis]

                new_hs_actor, actions = jax.vmap(
                    actor_network.apply, in_axes=(None, 0, 0, 0)
                )(
                    train_state.actor_params,
                    hs_actor,
                    _obs,
                    _dones
                )
                actions = actions.squeeze(axis=1)

                def use_cem_actions(actions, rng):
                    cem_actions = []
                    rng, cem_rng = jax.random.split(rng)
                    cem_action = cem_action_selection(
                            critic_network,
                            train_state.params,
                            actor_network,
                            train_state.actor_params,
                            hs_critic[i:i+1],
                            hs_actor[i:i+1],
                            _obs[i],
                            _dones[i],
                            cem_rng
                        )
                        cem_actions.append(cem_action)
                    return jnp.stack(cem_actions)

                def use_actor_actions(actions, rng):
                    return actions
                
                actions = jax.lax.cond(
                    config.get("USE_CEM", False),
                    use_cem_actions,
                    use_actor_actions,
                    actions, rng
                )

                actions, new_ou_state = add_exploration_noise(
                    actions, ou_state, rng_a, False, 
                )
                
                actions_dict = unbatchify(actions)

                # new_hs, q_vals = jax.vmap(
                #     network.apply, in_axes=(None, 0, 0, 0)
                # )(  # vmap across the agent dim
                #     train_state.params,
                #     hs,
                #     _obs,
                #     _dones,
                # )
                # q_vals = q_vals.squeeze(
                #     axis=1
                # )  # (num_agents, num_envs, num_actions) remove the time dim

                # # explore
                # avail_actions = wrapped_env.get_valid_actions(env_state.env_state)

                # eps = eps_scheduler(train_state.n_updates)
                # _rngs = jax.random.split(rng_a, env.num_agents)
                # actions = jax.vmap(eps_greedy_exploration, in_axes=(0, 0, None, 0))(
                #     _rngs, q_vals, eps, batchify(avail_actions)
                # )
                # actions = unbatchify(actions)

                new_obs, new_env_state, rewards, dones, infos = wrapped_env.batch_step(
                    rng_s, env_state, actions
                )
                timestep = Timestep(
                    obs=last_obs,
                    actions=actions,
                    rewards=jax.tree.map(lambda x:config.get("REW_SCALE", 1)*x, rewards),
                    dones=last_dones,
                )
                new_hs_critic = hs_critic

                # return (new_hs, new_obs, dones, new_env_state, rng), (timestep, infos)
                return (new_hs_actor, new_hs_critic, new_obs, dones, new_env_state, rng), (timestep, infos)

            # step the env (should be a complete rollout)
            rng, _rng = jax.random.split(rng)
            init_obs, env_state = wrapped_env.batch_reset(_rng)
            init_dones = {
                agent: jnp.zeros((config["NUM_ENVS"]), dtype=bool)
                for agent in env.agents + ["__all__"]
            }
            # init_hs = ScannedRNN.initialize_carry(
            #     config["HIDDEN_SIZE"], len(env.agents), config["NUM_ENVS"]
            # )
            init_hs_actor = ScannedRNN.initialize_carry(
                config["HIDDEN_SIZE"], len(env.agents), config["NUM_ENVS"]
            )
            init_hs_critic = ScannedRNN.initialize_carry(
                config["HIDDEN_SIZE"], len(env.agents), config["NUM_ENVS"]
            )
            # Initialize OU state if needed
            # init_ou_state = jnp.zeros((len(env.agents), config["NUM_ENVS"], action_dim))

            expl_state = (init_hs_actor, init_hs_critic, init_obs, init_dones, env_state)
            rng, _rng = jax.random.split(rng)
            # _, (timesteps, infos) = jax.lax.scan(
            #     _step_env,
            #     (*expl_state, _rng),
            #     None,
            #     config["NUM_STEPS"],
            # )
            # (_, _, _, _, _, new_ou_state, _), (timesteps, infos) = jax.lax.scan(
            _ , (timesteps, infos) = jax.lax.scan(
                _step_env,
                (*expl_state, _rng),
                None,
                config["NUM_STEPS"],
            )

            train_state = train_state.replace(
                timesteps=train_state.timesteps + config["NUM_STEPS"] * config["NUM_ENVS"],
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
                    lambda x: jnp.swapaxes(
                        x[:, 0], 0, 1
                    ),  # remove the dummy sequence dim (1) and swap batch and temporal dims
                    minibatch,
                )  # (max_time_steps, batch_size, ...)

                # preprocess network input
                # init_hs = ScannedRNN.initialize_carry(
                #     config["HIDDEN_SIZE"],
                #     len(env.agents),
                #     config["BUFFER_BATCH_SIZE"],
                # )
                init_hs_critic = ScannedRNN.initialize_carry(
                    config["HIDDEN_SIZE"],
                    len(env.agents),
                    config["BUFFER_BATCH_SIZE"],
                )
                init_hs_actor = ScannedRNN.initialize_carry(
                    config["HIDDEN_SIZE"],
                    len(env.agents),
                    config["BUFFER_BATCH_SIZE"],
                )

                # num_agents, timesteps, batch_size, ...
                _obs = batchify(minibatch.obs)
                _dones = batchify(minibatch.dones)
                _actions = batchify(minibatch.actions)
                #_rewards = batchify(minibatch.rewards)

                # Get next actions from target actor
                _, next_actions = jax.vmap(actor_network.apply, in_axes=(None, 0, 0, 0))(
                    train_state.target_actor_params,
                    init_hs_actor,
                    _obs,
                    _dones,
                )

                # Get Q-values for next state-action pairs
                _, q_next_target = jax.vmap(critic_network.apply, in_axes=(None, 0, 0, 0, 0))(
                    train_state.target_network_params,
                    init_hs_critic,
                    _obs,
                    _dones,
                    next_actions,
                )

                # _, q_next_target = jax.vmap(network.apply, in_axes=(None, 0, 0, 0))(
                #     train_state.target_network_params,
                #     init_hs,
                #     _obs,
                #     _dones,
                # )  # (num_agents, timesteps, batch_size, num_actions)

                def _loss_fn(params):
                    # _, q_vals = jax.vmap(network.apply, in_axes=(None, 0, 0, 0))(
                    #     params,
                    #     init_hs,
                    #     _obs,
                    #     _dones,
                    # )  # (num_agents, timesteps, batch_size, num_actions)
                    _, q_vals = jax.vmap(critic_network.apply, in_axes=(None, 0, 0, 0, 0))(
                        params,
                        init_hs_critic,
                        _obs,
                        _dones,
                        _actions,
                    )

                    q_tot = jnp.sum(q_vals.squeeze(-1), axis=0)[:-1]
                    q_next_tot = jnp.sum(q_next_target.squeeze(-1), axis=0)[1:]

                    vdn_target = (
                        minibatch.rewards["__all__"][:-1]
                        + (1 - minibatch.dones["__all__"][:-1])
                        * config["GAMMA"]
                        * q_next_tot
                    )

                    loss = jnp.mean((q_tot - jax.lax.stop_gradient(vdn_target)) ** 2)
                    
                    return loss, q_tot.mean()

                    # # get logits of the chosen actions
                    # chosen_action_q_vals = jnp.take_along_axis(
                    #     q_vals,
                    #     _actions[..., np.newaxis],
                    #     axis=-1,
                    # ).squeeze(-1)  # (num_agents, timesteps, batch_size,)

                    # unavailable_actions = 1 - _avail_actions
                    # valid_q_vals = q_vals - (unavailable_actions * 1e10)

                    # # get the q values of the next state
                    # q_next = jnp.take_along_axis(
                    #     q_next_target,
                    #     jnp.argmax(valid_q_vals, axis=-1)[..., np.newaxis],
                    #     axis=-1,
                    # ).squeeze(-1)  # (num_agents, timesteps, batch_size,)

                    # vdn_target = (
                    #     minibatch.rewards["__all__"][:-1]
                    #     + (
                    #         1 - minibatch.dones["__all__"][:-1]
                    #     )  # use next done because last done was saved for rnn re-init
                    #     * config["GAMMA"]
                    #     * jnp.sum(q_next, axis=0)[1:]  # sum over agents
                    # )

                    # chosen_action_q_vals = jnp.sum(chosen_action_q_vals, axis=0)[:-1]
                    # loss = jnp.mean(
                    #     (chosen_action_q_vals - jax.lax.stop_gradient(vdn_target)) ** 2
                    # )

                    # return loss, (chosen_action_q_vals.mean(), vdn_target.mean())

                (loss, qvals), grads = jax.value_and_grad(_loss_fn, has_aux=True)(
                    train_state.params
                )
                train_state = train_state.apply_gradients(grads=grads)
                train_state = train_state.replace(
                    grad_steps=train_state.grad_steps + 1,
                )
                return (train_state, rng), (loss, qvals, vdn_target)

            rng, _rng = jax.random.split(rng)
            is_learn_time = (
                buffer.can_sample(buffer_state)
            ) & (  # enough experience in buffer
                train_state.timesteps > config["LEARNING_STARTS"]
            )

            # Learning phase
            def do_learning(train_state, rng):
                return jax.lax.scan(
                    _learn_phase, (train_state, rng), None, config["NUM_EPOCHS"]
                )
            
            def skip_learning(train_state, rng):
                return (
                    (train_state, rng),
                    (
                        jnp.zeros(config["NUM_EPOCHS"]),
                        jnp.zeros(config["NUM_EPOCHS"]),
                    ),
                )

            
            (train_state, rng), (loss, qvals, vdn_target) = jax.lax.cond(
                is_learn_time,
                do_learning,
                skip_learning
                train_state,
                _rng,
            )

            # Update target networks
            def update_target_networks(train_state):
                return train_state.replace(
                    target_network_params=optax.incremental_update(
                        train_state.params,
                        train_state.target_network_params,
                        config["TAU"],
                    ),
                    target_actor_params=optax.incremental_update(
                        train_state.actor_params,
                        train_state.target_actor_params,
                        config["TAU"],
                    )
                )

            # update target network
            train_state = jax.lax.cond(
                train_state.n_updates % config["TARGET_UPDATE_INTERVAL"] == 0,
                update_target_networks,
                lambda train_state: train_state,
                operand=train_state,
            )

            # UPDATE METRICS
            print(type(env))
            train_state = train_state.replace(n_updates=train_state.n_updates + 1)
            metrics = {
                "env_step": train_state.timesteps,
                "update_steps": train_state.n_updates,
                "grad_steps": train_state.grad_steps,
                "loss": loss.mean(),
                "qvals": qvals.mean(),
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

            # report on wandb if required
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
            """Help function to test greedy policy during training"""
            if not config.get("TEST_DURING_TRAINING", True):
                return None

            # params = train_state.params
            actor_params = train_state.actor_params
            def _greedy_env_step(step_state, unused):
                actor_params, env_state, last_obs, last_dones, hstate, rng = step_state
                rng, key_s = jax.random.split(rng)
                _obs = batchify(last_obs)[:, np.newaxis]
                _dones = batchify(last_dones)[:, np.newaxis]

                hstate, actions = jax.vmap(actor_network.apply, in_axes=(None, 0, 0, 0))(
                    actor_params,
                    hstate,
                    _obs,
                    _dones,
                )
                actions = actions.squeeze(axis=1)
                actions_dict = unbatchify(actions)

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
            )  # (n_agents*n_envs, hs_size)
            step_state = (
                # params,
                actor_params,
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

        # train
        rng, _rng = jax.random.split(rng)
        runner_state = (train_state, buffer_state, test_state, _rng)

        runner_state, metrics = jax.lax.scan(
            _update_step, runner_state, None, config["NUM_UPDATES"]
        )

        return {"runner_state": runner_state, "metrics": metrics}

    return train


def env_from_config(config):
    env_name = config["ENV_NAME"]
    # smax init neeeds a scenario
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
    print("Config:\n", OmegaConf.to_yaml(config))

    alg_name = config.get("ALG_NAME", "vdn_rnn")
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

    # save params
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

    default_config = {**default_config, **default_config["alg"]}  # merge the alg config with the main config
    env_name = default_config["ENV_NAME"]
    alg_name = default_config.get("ALG_NAME", "vdn_rnn")
    env, env_name = env_from_config(default_config)

    def wrapped_make_train():
        wandb.init(project=default_config["PROJECT"])

        # update the default params
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
                "values": [
                    0.005,
                    0.001,
                    0.0005,
                    0.0001,
                    0.00005,
                ]
            },
            "NUM_ENVS": {"values": [8, 32, 64, 128]},
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
