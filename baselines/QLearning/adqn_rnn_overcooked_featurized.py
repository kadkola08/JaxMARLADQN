"""
Specific to this implementation: CNN network and Reward Shaping Annealing as per Overcooked paper.
"""
import os
import copy
import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
from typing import Any

import flax
import chex
import optax
import flax.linen as nn
from flax.training.train_state import TrainState
from flax.linen.initializers import constant, orthogonal
import hydra
from omegaconf import OmegaConf
import flashbax as fbx
import wandb

from jaxmarl import make
from jaxmarl.environments.smax import map_name_to_scenario
from jaxmarl.wrappers.baselines import (
    SMAXLogWrapper,
    MPELogWrapper,
    LogWrapper,
    CTRolloutManager,
)
from jaxmarl.environments.overcooked import overcooked_layouts
from jaxmarl.environments.overcooked_v2 import overcooked_v2_layouts

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

class RNNQNetwork(nn.Module):
    # homogenous agent for parameters sharing, assumes all agents have same obs and action dim
    action_dim: int
    hidden_dim: int
    init_scale: float = 1.0

    @nn.compact
    def __call__(self, hidden, obs, dones):
        # time_steps, batch_size = obs.shape[:2]
        # obs_reshaped = obs.reshape(-1, *obs.shape[2:])

        # embedding = CNN(num_features=self.hidden_dim)(obs_reshaped)
        # embedding = embedding.reshape(time_steps, batch_size, -1)
        
        # embedding = nn.relu(embedding)
        embedding = nn.Dense(
            self.hidden_dim,
            kernel_init=orthogonal(self.init_scale),
            bias_init=constant(0.0),
        )(obs)
        embedding = nn.relu(embedding)

        rnn_in = (embedding, dones)
        hidden, embedding = ScannedRNN()(hidden, rnn_in)

        q_vals = nn.Dense(
            self.action_dim,
            kernel_init=orthogonal(self.init_scale),
            bias_init=constant(0.0),
        )(embedding)

        return hidden, q_vals
    
class MLP(nn.Module):
    """Simple Multi-Layer Perceptron with reLU nonlinearities."""

    features: list[int]
    init_scale: float = 1.0

    @nn.compact
    def __call__(self, x: jax.Array):
        for i, features in enumerate(self.features):
            if i != 0:
                x = nn.relu(x)

            x = nn.Dense(
                features,
                kernel_init=orthogonal(self.init_scale),
                bias_init=constant(0.0),
            )(x)

        return x


class MixingNetwork(nn.Module):
    """
    Mixing network for projecting joint histories, states and joint actions into Q_tot.
    """

    embedding_dim: int
    hidden_dim: int
    init_scale: float = 1.0
    num_agents: int = 1
    hidden_multiplier: int = 1
    num_encoding_layers: int = 1

    @nn.compact
    def __call__(self, hidden, joint_observation, state, joint_action, dones):
        
        # time_steps, batch_size = joint_observation.shape[:2]
        # joint_observation = joint_observation.reshape(-1, *joint_observation.shape[2:])
        # joint_observation = CNN(num_features=self.hidden_dim)(joint_observation)
        # joint_observation = joint_observation.reshape(time_steps, batch_size, -1)
        # joint_observation = nn.relu(joint_observation)
        joint_observation = nn.Dense(
            self.hidden_dim,
            kernel_init=orthogonal(self.init_scale),
            bias_init=constant(0.0),
        )(joint_observation)
        joint_observation = nn.relu(joint_observation)

        # embedding = jnp.concatenate([joint_observation, joint_action], axis=-1)
        # rnn_in = (embedding, dones)
        # hidden, embedding = ScannedRNN()(hidden, rnn_in)
        rnn_in = (joint_observation, dones)
        hidden, joint_observation= ScannedRNN()(hidden, rnn_in)

        # time_steps, batch_size = state.shape[:2]
        # state = state.reshape(-1, *state.shape[2:])
        # state = CNN(num_features=self.hidden_dim)(state)
        # state = state.reshape(time_steps, batch_size, -1)
        # state = nn.relu(state)
        state = nn.Dense(
            self.hidden_dim,
            kernel_init=orthogonal(self.init_scale),
            bias_init=constant(0.0),
        )(state)
        state = nn.relu(state)

        input = jnp.concatenate([joint_observation, state, joint_action], axis=-1)
        # input = jnp.concatenate([embedding, state], axis=-1)
        features = [int(self.hidden_dim) * int(self.hidden_multiplier) for _ in range(int(self.num_encoding_layers))]
        embedding = MLP(
            features=features,
	        init_scale=self.init_scale,
        )(input)
        # embedding = nn.Dense(
        #     # 512,
        #     self.hidden_dim * self.hidden_multiplier,
        #     kernel_init=orthogonal(self.init_scale),
        #     bias_init=constant(0.0),
        # )(input)
        # embedding = nn.relu(embedding)
        # embedding = nn.Dense(
        #     self.embedding_dim,
        #     kernel_init=orthogonal(self.init_scale),
        #     bias_init=constant(0.0),
        # )(embedding)
        # embedding = nn.relu(embedding)
        # embedding = nn.Dense(
        #     self.hidden_dim * 4,
        #     kernel_init=orthogonal(self.init_scale),
        #     bias_init=constant(0.0),
        # )(embedding)
        # embedding = nn.relu(embedding)

        # rnn_in = (embedding, dones)
        # hidden, embedding = ScannedRNN()(hidden, rnn_in)

        q_tot = nn.Dense(
            1,
            kernel_init=orthogonal(self.init_scale),
            bias_init=constant(0.0),
        )(embedding)

        return hidden, q_tot.squeeze()  # (time_steps, batch_size)


@chex.dataclass(frozen=True)
class Timestep:
    obs: dict
    actions: dict
    avail_actions: dict
    rewards: dict
    dones: dict


class CustomTrainState(TrainState):
    target_network_params: Any
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

    rew_shaping_anneal = optax.linear_schedule(
        init_value=1.0, end_value=0.0, transition_steps=config["REW_SHAPING_HORIZON"]
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
    
    def compute_action_metrics(actions_dict, avail_actions_dict):
        """Compute action distribution metrics for logging."""
        action_counts = {}
        action_probs = {}
        
        for agent in env.agents:
            agent_actions = actions_dict[agent]  # shape: (num_envs,)
            agent_avail = avail_actions_dict[agent]  # shape: (num_envs, num_actions)
            
            # Count occurrences of each action
            num_actions = agent_avail.shape[-1]
            counts = jnp.zeros(num_actions)
            for a in range(num_actions):
                counts = counts.at[a].set(jnp.sum(agent_actions == a))
            
            # Compute probabilities
            probs = counts / jnp.sum(counts)
            
            action_counts[f"action_counts/{agent}"] = counts
            action_probs[f"action_probs/{agent}"] = probs
            
            # Also compute entropy of action distribution
            # Avoid log(0) by adding small epsilon
            eps = 1e-8
            entropy = -jnp.sum(probs * jnp.log(probs + eps))
            action_probs[f"action_entropy/{agent}"] = entropy
        
        # Compute aggregate metrics
        all_actions = jnp.concatenate([actions_dict[agent] for agent in env.agents])
        total_counts = jnp.zeros(num_actions)
        for a in range(num_actions):
            total_counts = total_counts.at[a].set(jnp.sum(all_actions == a))
        
        total_probs = total_counts / jnp.sum(total_counts)
        total_entropy = -jnp.sum(total_probs * jnp.log(total_probs + 1e-8))
        
        action_counts["action_counts/all_agents"] = total_counts
        action_probs["action_probs/all_agents"] = total_probs
        action_probs["action_entropy/all_agents"] = total_entropy
        
        return {**action_counts, **action_probs}
    
    def train(rng):

        original_seed = rng[0]
        # INIT ENV
        rng, _rng = jax.random.split(rng)
        wrapped_env = CTRolloutManager(
            env, batch_size=config["NUM_ENVS"], preprocess_obs=False
        )
        test_env = CTRolloutManager(
            env, batch_size=config["TEST_NUM_ENVS"], preprocess_obs=False
        )  # batched env for testing (has different batch size)

        # INIT BUFFER
        # to initalize the buffer is necessary to sample a trajectory to know its strucutre
        def _env_sample_step(env_state, unused):
            rng, key_a, key_s = jax.random.split(
                jax.random.PRNGKey(0), 3
            )  # use a dummy rng here
            key_a = jax.random.split(key_a, env.num_agents)
            actions = {
                agent: wrapped_env.batch_sample(key_a[i], agent)
                for i, agent in enumerate(env.agents)
            }
            avail_actions = wrapped_env.get_valid_actions(env_state.env_state)
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
        # make sure that the dim being remove below is actually the NUM_ENV dim
        sample_traj_unbatched = jax.tree.map(
            lambda x: x[:, 0], sample_traj
        )  # remove the NUM_ENV dim
        buffer = fbx.make_trajectory_buffer(
            max_length_time_axis=int(config["BUFFER_SIZE"] // config["NUM_ENVS"]),
            min_length_time_axis=config["BUFFER_BATCH_SIZE"],
            sample_batch_size=config["BUFFER_BATCH_SIZE"],
            add_batch_size=config["NUM_ENVS"],
            sample_sequence_length=100,
            period=1,
        )
        buffer_state = buffer.init(sample_traj_unbatched)

        # INIT NETWORK AND OPTIMIZER
        network = RNNQNetwork(
            action_dim=wrapped_env.max_action_space,
            hidden_dim=config["HIDDEN_SIZE"],
        )

        mixer = MixingNetwork(
            embedding_dim=config["MIXER_EMBEDDING_DIM"],
            hidden_dim=config["HIDDEN_SIZE"],
            init_scale=config["MIXER_INIT_SCALE"],
            hidden_multiplier=config["HIDDEN_MULTIPLIER"],
            num_encoding_layers=config["NUM_ENCODING_LAYERS"]
        )

        def create_agent(rng):
            init_x = (
                jnp.zeros(
                    # (1, 1, wrapped_env.obs_size)
                    (1, 1, *env.observation_space().shape)
                ),  # (time_step, batch_size, obs_size)
                jnp.zeros((1, 1)),  # (time_step, batch size)
            )
            init_hs = ScannedRNN.initialize_carry(
                config["HIDDEN_SIZE"], 1
            )  # (batch_size, hidden_dim)
            agent_params = network.init(rng, init_hs, *init_x)

            # H, W, C = env.observation_space().shape
            # init_jt_obs = jnp.zeros((1, 1, H, W, C * len(env.agents)))
            init_jt_obs = jnp.zeros((1, 1, wrapped_env.obs_size * len(env.agents)))
            init_jt_act = jnp.zeros((1, 1, wrapped_env.max_action_space * len(env.agents)))
            init_mixer_hs = ScannedRNN.initialize_carry(
                # config["HIDDEN_SIZE"] * config["HIDDEN_MULTIPLIER"] + wrapped_env.max_action_space * len(env.agents), 1
                config["HIDDEN_SIZE"] * config["HIDDEN_MULTIPLIER"], 1
                # config["MIXER_EMBEDDING_DIM"], 1
                # 512, 1
            ) 
            # state_shape_unflattened = (env.height, env.width, len(env.agents) * (18 + 4 * (env.layout.num_ingredients + 2)))
            # state_shape_unflattened = (env.height, env.width, len(env.agents) * (18 + 4 * (0 + 2)))
            init_state = jnp.zeros((1, 1, wrapped_env.obs_size * len(env.agents)))
            init_dones = jnp.zeros((1, 1))
            mixer_params = mixer.init(_rng, init_mixer_hs, init_jt_obs, init_state, init_jt_act, init_dones)

            network_params = {'agent':agent_params, 'mixer':mixer_params}

            lr_scheduler = optax.linear_schedule(
                config["LR"],
                1e-10,
                (config["NUM_EPOCHS"]) * config["NUM_UPDATES"],
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

        # TRAINING LOOP
        def _update_step(runner_state, unused):

            train_state, buffer_state, expl_state, test_state, rng = runner_state

            # SAMPLE PHASE
            def _step_env(carry, _):
                hs, last_obs, last_dones, env_state, rng = carry
                rng, rng_a, rng_s = jax.random.split(rng, 3)

                # (num_agents, 1 (dummy time), num_envs, obs_size)
                _obs = batchify(last_obs)[:, np.newaxis]
                _dones = batchify(last_dones)[:, np.newaxis]

                new_hs, q_vals = jax.vmap(
                    network.apply, in_axes=(None, 0, 0, 0)
                )(  # vmap across the agent dim
                    train_state.params['agent'],
                    hs,
                    _obs,
                    _dones,
                )
                q_vals = q_vals.squeeze(
                    axis=1
                )  # (num_agents, num_envs, num_actions) remove the time dim

                # explore
                avail_actions = wrapped_env.get_valid_actions(env_state.env_state)

                eps = eps_scheduler(train_state.n_updates)
                _rngs = jax.random.split(rng_a, env.num_agents)
                actions = jax.vmap(eps_greedy_exploration, in_axes=(0, 0, None, 0))(
                    _rngs, q_vals, eps, batchify(avail_actions)
                )
                actions = unbatchify(actions)

                new_obs, new_env_state, rewards, dones, infos = wrapped_env.batch_step(
                    rng_s, env_state, actions
                )

                # add shaped reward
                shaped_reward = infos.pop("shaped_reward")
                shaped_reward["__all__"] = batchify(shaped_reward).sum(axis=0)
                rewards = jax.tree.map(
                    lambda x, y: x + y * rew_shaping_anneal(train_state.timesteps),
                    rewards,
                    shaped_reward,
                )

                timestep = Timestep(
                    obs=last_obs,
                    actions=actions,
                    rewards=rewards,
                    dones=last_dones,
                    avail_actions=avail_actions,
                )

                # Compute action metrics for logging
                action_metrics = compute_action_metrics(actions, avail_actions)

                            
                return (new_hs, new_obs, dones, new_env_state, rng), (timestep, infos, action_metrics)

            # expl_state = (init_hs, init_obs, init_dones, env_state)
            rng, _rng = jax.random.split(rng)
            carry, (timesteps, infos, action_metrics_over_time) = jax.lax.scan(
                _step_env,
                (*expl_state, _rng),
                None,
                config["NUM_STEPS"],
            )
            expl_state = carry[:4]

            # Aggregate action metrics over the rollout
            aggregated_action_metrics = {}
            for key in action_metrics_over_time:
                if "counts" in key:
                    # Sum counts over time steps
                    aggregated_action_metrics[key] = jnp.sum(action_metrics_over_time[key], axis=0)
                elif "probs" in key or "entropy" in key:
                    # Average probabilities and entropy over time steps
                    aggregated_action_metrics[key] = jnp.mean(action_metrics_over_time[key], axis=0)

            train_state = train_state.replace(
                timesteps=train_state.timesteps
                + config["NUM_STEPS"] * config["NUM_ENVS"]
            )  # update timesteps count

            # BUFFER UPDATE
            # jax.debug.breakpoint()
            buffer_traj_batch = jax.tree.map(
                lambda x: jnp.swapaxes(x, 0, 1)[
                    :, np.newaxis
                ],  # put the batch dim first and add a dummy sequence dim
                timesteps,
            )  # (num_envs, 1, time_steps, ...)
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
                init_hs = ScannedRNN.initialize_carry(
                    config["HIDDEN_SIZE"],
                    len(env.agents),
                    config["BUFFER_BATCH_SIZE"],
                )

                mixer_hs = ScannedRNN.initialize_carry(
                    # 512,
                    # config["HIDDEN_SIZE"] * config["HIDDEN_MULTIPLIER"] + wrapped_env.max_action_space * len(env.agents),
                    config["HIDDEN_SIZE"] * config["HIDDEN_MULTIPLIER"],
                    # len(env.agents), 
                    config["BUFFER_BATCH_SIZE"],
                )
                # num_agents, timesteps, batch_size, ...

                _obs = batchify(minibatch.obs)
                _dones = batchify(minibatch.dones)
                _actions = batchify(minibatch.actions)
                _rewards = batchify(minibatch.rewards)
                _avail_actions = batchify(minibatch.avail_actions)

                _, q_next_target = jax.vmap(network.apply, in_axes=(None, 0, 0, 0))(
                    train_state.target_network_params['agent'],
                    init_hs,
                    _obs,
                    _dones,
                )  # (num_agents, timesteps, batch_size, num_actions)

                def _mixer_loss_fn(params):
                    _, q_vals = jax.vmap(network.apply, in_axes=(None, 0, 0, 0))(
                        params['agent'],
                        init_hs,
                        _obs,
                        _dones,
                    )  # (num_agents, timesteps, batch_size, num_actions)

                    unavailable_actions = 1 - _avail_actions
                    valid_q_vals = q_vals - (unavailable_actions * 1e10)

                    target_actions = jnp.argmax(valid_q_vals, axis=-1)
                    target_actions_unbatched = unbatchify(target_actions)
                    one_hot_actions_target = {}
                    for agent, action_values in target_actions_unbatched.items():
                        one_hot_actions_target[agent] = jax.nn.one_hot(action_values,  wrapped_env.max_action_space)

                    one_hot_actions = {}
                    for agent, action_values in minibatch.actions.items():
                        # print("Action values: ", action_values)
                        one_hot_actions[agent] = jax.nn.one_hot(action_values,  wrapped_env.max_action_space)  

                    joint_action = jnp.concatenate([one_hot_actions[agent] for agent in minibatch.actions], axis=-1)  # Shape: (26, 32, 15)
                    joint_action_target = jnp.concatenate([one_hot_actions_target[agent] for agent in minibatch.actions], axis=-1)  # Shape: (26, 32, 15)

                    joint_observation = []
                    for agent, obs_values in minibatch.obs.items():
                        if agent != '__all__':
                            joint_observation.append(obs_values)
                    joint_observation = jnp.concatenate(joint_observation, axis=-1)  # Shape: (26, 32, 63)

                    state = minibatch.obs["__all__"]
                    # channels_per_agent = 18 + 4 * (env.layout.num_ingredients + 2)
                    # channels_per_agent = 18 + 4 * (0 + 2)

                    # state = state_flat.reshape(
                    #     state_flat.shape[0],
                    #     state_flat.shape[1],
                    #     len(env.agents),
                    #     env.height,
                    #     env.width,
                    #     channels_per_agent
                    # ).transpose(0, 1, 3, 4, 2, 5).reshape(
                    #     state_flat.shape[0],
                    #     state_flat.shape[1],
                    #     env.height,
                    #     env.width,
                    #     channels_per_agent * len(env.agents)
                    # )

                    _, q_tot_next = mixer.apply(
                        train_state.target_network_params['mixer'],
                        mixer_hs,
                        joint_observation, 
                        state, 
                        joint_action_target,
                        minibatch.dones["__all__"]
                    )
                    q_tot_target = (
                        minibatch.rewards["__all__"][:-1]
                        + (
                            1 - minibatch.dones["__all__"][:-1]
                        )
                        * config["GAMMA"]
                        * q_tot_next[1:]
                    )

                    _, q_tot_vals = mixer.apply(
                        params['mixer'], 
                        mixer_hs,
                        joint_observation, 
                        state,
                        joint_action,
                        minibatch.dones["__all__"]
                    )

                    mixer_loss = jnp.mean(
                        (q_tot_vals[:-1] - jax.lax.stop_gradient(q_tot_target)) ** 2
                    )

                    return mixer_loss, q_tot_vals.mean()
                
                (mixer_loss, q_tot_vals), mixer_grads = jax.value_and_grad(_mixer_loss_fn, has_aux=True)(
                    train_state.params
                )
                mixer_grads = {
                    'agent' : jax.tree.map(lambda x: jnp.zeros_like(x), mixer_grads['agent']),
                    'mixer': mixer_grads['mixer']  # Keep the mixer gradients
                }

                def _agent_loss_fn(params):

                    _, q_vals = jax.vmap(network.apply, in_axes=(None, 0, 0, 0))(
                        params['agent'],
                        init_hs,
                        _obs,
                        _dones,
                    )  # (num_agents, timesteps, batch_size, num_actions)

                    # get logits of the chosen actions
                    chosen_action_q_vals = jnp.take_along_axis(
                        q_vals,
                        _actions[..., np.newaxis],
                        axis=-1,
                    ).squeeze(
                        -1
                    )  # (num_agents, timesteps, batch_size,)

                    unavailable_actions = 1 - _avail_actions
                    valid_q_vals = q_vals - (unavailable_actions * 1e10)

                    target_actions = jnp.argmax(valid_q_vals, axis=-1)
                    target_actions_unbatched = unbatchify(target_actions)
                    one_hot_actions_target = {}
                    for agent, action_values in target_actions_unbatched.items():
                        one_hot_actions_target[agent] = jax.nn.one_hot(action_values,  wrapped_env.max_action_space)

                    one_hot_actions = {}
                    for agent, action_values in minibatch.actions.items():
                        # print("Action values: ", action_values)
                        one_hot_actions[agent] = jax.nn.one_hot(action_values,  wrapped_env.max_action_space)  

                    joint_action = jnp.concatenate([one_hot_actions[agent] for agent in minibatch.actions], axis=-1)  # Shape: (26, 32, 15)
                    joint_action_target = jnp.concatenate([one_hot_actions_target[agent] for agent in minibatch.actions], axis=-1)  # Shape: (26, 32, 15)

                    joint_observation = []
                    for agent, obs_values in minibatch.obs.items():
                        if agent != '__all__':
                            joint_observation.append(obs_values)
                    joint_observation = jnp.concatenate(joint_observation, axis=-1)  # Shape: (26, 32, 63)

                    state = minibatch.obs["__all__"]
                    # channels_per_agent = 18 + 4 * (env.layout.num_ingredients + 2)
                    # channels_per_agent = 18 + 4 * (0 + 2)

                    # state = state_flat.reshape(
                    #     state_flat.shape[0],
                    #     state_flat.shape[1],
                    #     len(env.agents),
                    #     env.height,
                    #     env.width,
                    #     channels_per_agent
                    # ).transpose(0, 1, 3, 4, 2, 5).reshape(
                    #     state_flat.shape[0],
                    #     state_flat.shape[1],
                    #     env.height,
                    #     env.width,
                    #     channels_per_agent * len(env.agents)
                    # )

                    _, q_tot_next = mixer.apply(
                        train_state.target_network_params['mixer'],
                        mixer_hs,
                        joint_observation, 
                        state, 
                        joint_action_target,
                        minibatch.dones["__all__"]
                    )
                    q_tot_target = (
                        minibatch.rewards["__all__"][:-1]
                        + (
                            1 - minibatch.dones["__all__"][:-1]
                        )
                        * config["GAMMA"]
                        * q_tot_next[1:]
                    )

                    chosen_action_q_vals = chosen_action_q_vals[:, :-1]

                    loss = jnp.mean(
                        (chosen_action_q_vals - jax.lax.stop_gradient(q_tot_target)) ** 2
                    )                    
                    
                    return loss, chosen_action_q_vals.mean()
                
                def _calculate_grad_state(grads):
                    flat_grads = [x.flatten() for x in jax.tree_util.tree_leaves(grads) if x.size > 0]
                    all_grads = jnp.concatenate([g.flatten() for g in flat_grads])
                    grad_norm = jnp.linalg.norm(all_grads)
                    return {
                        'grad_norm': grad_norm,
                    }

                (agent_loss, agent_qvals), agent_grads = jax.value_and_grad(_agent_loss_fn, has_aux=True)(
                    train_state.params
                )
                agent_grads = {
                    'agent': agent_grads['agent'], # Keep the agent gradients
                    'mixer' : jax.tree.map(lambda x: jnp.zeros_like(x), agent_grads['mixer'])
                }

                mixer_grad_stats = _calculate_grad_state(mixer_grads)
                agent_grad_stats = _calculate_grad_state(agent_grads)

                update_mixer = (train_state.grad_steps % config.get("AGENT_UPDATE_RATIO", 1)) == 0
                update_agent = (train_state.grad_steps % config.get("MIXER_UPDATE_RATIO", 1)) == 0

                train_state = jax.lax.cond(
                    update_mixer,
                    lambda train_state: train_state.apply_gradients(grads=mixer_grads),
                    lambda train_state: train_state,
                    operand=train_state,
                )
                train_state = jax.lax.cond(
                    update_agent,
                    lambda train_state: train_state.apply_gradients(grads=agent_grads),
                    lambda train_state: train_state,
                    operand=train_state,
                )
                # train_state = train_state.apply_gradients(grads=mixer_grads)
                # train_state = train_state.apply_gradients(grads=agent_grads)
                
                train_state = train_state.replace(
                    grad_steps=train_state.grad_steps + 1,
                )

                return (train_state, rng), (mixer_loss, agent_loss, agent_qvals, q_tot_vals, 
                                            agent_grad_stats['grad_norm'], mixer_grad_stats['grad_norm'], _rewards.mean())

            rng, _rng = jax.random.split(rng)
            is_learn_time = (
                buffer.can_sample(buffer_state)
            ) & (  # enough experience in buffer
                train_state.timesteps > config["LEARNING_STARTS"]
            )
            (train_state, rng), (mixer_loss, agent_loss, agent_qvals, q_tot_vals, agent_grad_norm, mixer_grad_norm, rewards) = jax.lax.cond(
                is_learn_time,
                lambda train_state, rng: jax.lax.scan(
                    _learn_phase, (train_state, rng), None, config["NUM_EPOCHS"]
                ),
                lambda train_state, rng: (
                    (train_state, rng),
                    (
                        jnp.zeros(config["NUM_EPOCHS"]),    # mixer_loss
                        jnp.zeros(config["NUM_EPOCHS"]),    # loss
                        jnp.zeros(config["NUM_EPOCHS"]),    # qvals
                        jnp.zeros(config["NUM_EPOCHS"]),    # q_tot_vals
                        jnp.zeros(config["NUM_EPOCHS"]),    # agent_grad_stats
                        jnp.zeros(config["NUM_EPOCHS"]),    # mixer_grad_stats
                        jnp.zeros(config["NUM_EPOCHS"]),    # sampled rewards
                    ),
                ),  # do nothing
                train_state,
                _rng,
            )

            # update target network
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
                "loss": agent_loss.mean(),
                "mixer_loss": mixer_loss.mean(),
                "qvals": agent_qvals.mean(),
                "q_tot_vals": q_tot_vals.mean(),
                "agent_grad_stats": agent_grad_norm.mean(),
                "mixer_grad_stats": mixer_grad_norm.mean(),
                "sampled_rewards": rewards.mean(),
                "epsilon": eps_scheduler(train_state.n_updates),
            }
            metrics.update(jax.tree.map(lambda x: x.mean(), infos))

            # Add action metrics to wandb logging
            # For action counts, we want the mean per step
            for key, value in aggregated_action_metrics.items():
                if "counts" in key:
                    # Normalize counts by number of steps to get average per step
                    metrics[key] = value / config["NUM_STEPS"]
                else:
                    metrics[key] = value

            # update the test metrics
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
                    # Convert any array metrics to scalars for wandb
                    scalar_metrics = {}
                    for k, v in metrics.items():
                        if hasattr(v, 'shape') and v.shape:
                            # If it's an array, convert to list or take mean
                            if "counts" in k or "probs" in k:
                                # For action distributions, log each action separately
                                if v.ndim == 1:
                                    for i, val in enumerate(v):
                                        scalar_metrics[f"{k}/action_{i}"] = float(val)
                                else:
                                    scalar_metrics[k] = float(v.mean())
                            else:
                                scalar_metrics[k] = float(v.mean())
                        else:
                            scalar_metrics[k] = float(v)
                    
                    if config.get('WANDB_LOG_ALL_SEEDS', False):
                        scalar_metrics.update(
                            {f"rng{int(original_seed)}/{k}": v for k, v in scalar_metrics.items()}
                        )
                    wandb.log(scalar_metrics)
                    # if config.get('WANDB_LOG_ALL_SEEDS', False):
                    #     metrics.update(
                    #         {f"rng{int(original_seed)}/{k}": v for k, v in metrics.items()}
                    #     )
                    # wandb.log(metrics)

                jax.debug.callback(callback, metrics, original_seed)

            runner_state = (train_state, buffer_state, expl_state, test_state, rng)

            return runner_state, None
        
        def get_greedy_metrics(rng, train_state):
            """Help function to test greedy policy during training"""
            if not config.get("TEST_DURING_TRAINING", True):
                return None
            
            def _greedy_env_step(step_state, unused):
                env_state, last_obs, last_dones, hstate, rng = step_state
                rng, key_s = jax.random.split(rng)
                _obs = batchify(last_obs)[:, np.newaxis]
                _dones = batchify(last_dones)[:, np.newaxis]
                hstate, q_vals = jax.vmap(network.apply, in_axes=(None, 0, 0, 0))(
                    train_state.params['agent'],
                    hstate,
                    _obs,
                    _dones,
                )
                q_vals = q_vals.squeeze(axis=1)
                valid_actions = test_env.get_valid_actions(env_state.env_state)
                actions = get_greedy_actions(q_vals, batchify(valid_actions))
                actions = unbatchify(actions)
                obs, env_state, rewards, dones, infos = test_env.batch_step(
                    key_s, env_state, actions
                )

                # Compute action metrics for test phase
                action_metrics = compute_action_metrics(actions, valid_actions)

                # jax.debug.print("timesteps: {}", train_state.timesteps)
                # jax.debug.print("actions: {}", actions)
                # jax.debug.print("rewards: {}", rewards)

                step_state = (env_state, obs, dones, hstate, rng)
                return step_state, (rewards, dones, infos, action_metrics)
                # return step_state, (rewards, dones, infos)

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
                env_state,
                init_obs,
                init_dones,
                hstate,
                _rng,
            )

            step_state, (rewards, dones, infos, test_action_metrics) = jax.lax.scan(
                _greedy_env_step, step_state, None, config["TEST_NUM_STEPS"]
            )

            # Aggregate test action metrics
            test_aggregated_metrics = {}
            for key in test_action_metrics:
                if "counts" in key:
                    test_aggregated_metrics[f"test_{key}"] = jnp.sum(test_action_metrics[key], axis=0) / config["TEST_NUM_STEPS"]
                elif "probs" in key or "entropy" in key:
                    test_aggregated_metrics[f"test_{key}"] = jnp.mean(test_action_metrics[key], axis=0)


            metrics = {
                "returned_episode_returns": jnp.nanmean(
                    jnp.where(
                        infos["returned_episode"],
                        infos["returned_episode_returns"],
                        jnp.nan,
                    )
                )
            }
            return metrics
        
        rng, _rng = jax.random.split(rng)
        test_state = get_greedy_metrics(_rng, train_state)

        rng, _rng = jax.random.split(rng)
        init_obs, env_state = wrapped_env.batch_reset(_rng)
        init_dones = {
            agent: jnp.zeros((config["NUM_ENVS"]), dtype=bool)
            for agent in env.agents + ["__all__"]
        }
        init_hs = ScannedRNN.initialize_carry(
            config["HIDDEN_SIZE"], len(env.agents), config["NUM_ENVS"]
        )

        expl_state = (init_hs, init_obs, init_dones, env_state)

        # obs, env_state = wrapped_env.batch_reset(_rng)
        # expl_state = (obs, env_state)

        # train
        rng, _rng = jax.random.split(rng)
        runner_state = (train_state, buffer_state, expl_state, test_state, _rng)

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
    elif "overcooked_v2" in env_name.lower():
        env_name = f"{config['ENV_NAME']}_{config['ENV_KWARGS']['layout']}"
        config["ENV_KWARGS"]["layout"] = overcooked_v2_layouts[
            config["ENV_KWARGS"]["layout"]
        ]
        env = make(config["ENV_NAME"], **config["ENV_KWARGS"])
        env = LogWrapper(env)
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

    alg_name = config.get("ALG_NAME", "adqn_cnn_rnn")
    env, env_name= env_from_config(copy.deepcopy(config))

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
    alg_name = default_config.get("ALG_NAME", "adqn_cnn_rnn") 
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