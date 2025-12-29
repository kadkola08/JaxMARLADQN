import copy
import os
from typing import Any
from functools import partial

import chex
import flashbax as fbx
import flax.linen as nn
import hydra
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training.train_state import TrainState
from flax.linen.initializers import orthogonal, constant
from omegaconf import OmegaConf

import wandb
from jaxmarl import make
from jaxmarl.environments.overcooked import overcooked_layouts
from jaxmarl.environments.overcooked_v2 import overcooked_v2_layouts
from jaxmarl.environments.smax import map_name_to_scenario
from jaxmarl.wrappers.baselines import (
    CTRolloutManager, LogWrapper, MPELogWrapper, SMAXLogWrapper
)

class CNN(nn.Module):
    """CNN encoder for Overcooked observations."""
    activation: str = "relu"
    num_features: int = 65

    @nn.compact
    def __call__(self, x):
        activation = nn.relu if self.activation == "relu" else nn.tanh
        x = nn.Conv(features=33, kernel_size=(5, 5))(x)
        x = activation(x)
        x = nn.Conv(features=33, kernel_size=(3, 3))(x)
        x = activation(x)
        x = nn.Conv(features=33, kernel_size=(3, 3))(x)
        x = activation(x)
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(features=self.num_features)(x)
        x = activation(x)
        return x


class CNNOvercooked(nn.Module):
    """CNN module for processing Overcooked global state."""
    activation: str = "relu"

    @nn.compact
    def __call__(self, x):
        activation = nn.relu if self.activation == "relu" else nn.tanh
        x = nn.Conv(features=32, kernel_size=(5, 5))(x)
        x = activation(x)
        x = nn.Conv(features=32, kernel_size=(3, 3))(x)
        x = activation(x)
        x = x.reshape(*x.shape[:-3], -1)
        return x

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
        return nn.GRUCell(hidden_size, parent=None).initialize_carry(
            jax.random.PRNGKey(0), (*batch_size, hidden_size)
        )


class CNNRNNQNetwork(nn.Module):
    """CNN-RNN Q-Network for Overcooked."""
    action_dim: int
    hidden_dim: int
    init_scale: float = 1.0

    @nn.compact
    def __call__(self, hidden, obs, dones):
        time_steps, batch_size = obs.shape[:2]
        obs_reshaped = obs.reshape(-1, *obs.shape[2:])
        embedding = CNN(num_features=self.hidden_dim)(obs_reshaped)
        embedding = embedding.reshape(time_steps, batch_size, -1)
        rnn_in = (embedding, dones)
        hidden, embedding = ScannedRNN()(hidden, rnn_in)
        q_vals = nn.Dense(
            self.action_dim,
            kernel_init=orthogonal(self.init_scale),
            bias_init=constant(0.0),
        )(embedding)
        return hidden, q_vals

class DMAQ_SI_Weight(nn.Module):
    """State-Independent weight module for advantage computation."""
    n_agents: int
    n_actions: int
    state_dim: int
    num_kernel: int = 10
    adv_hypernet_embed: int = 64
    adv_hypernet_layers: int = 3
    state_encoder: nn.Module = None

    def setup(self):
        def make_key_extractor():
            if self.adv_hypernet_layers == 1:
                return nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))
            elif self.adv_hypernet_layers == 2:
                return nn.Sequential([
                    nn.Dense(self.adv_hypernet_embed, kernel_init=orthogonal(1.0), bias_init=constant(0.0)),
                    nn.relu,
                    nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))
                ])
            else:
                return nn.Sequential([
                    nn.Dense(self.adv_hypernet_embed, kernel_init=orthogonal(1.0), bias_init=constant(0.0)),
                    nn.relu,
                    nn.Dense(self.adv_hypernet_embed, kernel_init=orthogonal(1.0), bias_init=constant(0.0)),
                    nn.relu,
                    nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))
                ])

        def make_agent_extractor():
            if self.adv_hypernet_layers == 1:
                return nn.Dense(self.n_agents, kernel_init=orthogonal(1.0), bias_init=constant(0.0))
            elif self.adv_hypernet_layers == 2:
                return nn.Sequential([
                    nn.Dense(self.adv_hypernet_embed, kernel_init=orthogonal(1.0), bias_init=constant(0.0)),
                    nn.relu,
                    nn.Dense(self.n_agents, kernel_init=orthogonal(1.0), bias_init=constant(0.0))
                ])
            else:
                return nn.Sequential([
                    nn.Dense(self.adv_hypernet_embed, kernel_init=orthogonal(1.0), bias_init=constant(0.0)),
                    nn.relu,
                    nn.Dense(self.adv_hypernet_embed, kernel_init=orthogonal(1.0), bias_init=constant(0.0)),
                    nn.relu,
                    nn.Dense(self.n_agents, kernel_init=orthogonal(1.0), bias_init=constant(0.0))
                ])

        def make_action_extractor():
            if self.adv_hypernet_layers == 1:
                return nn.Dense(self.n_agents, kernel_init=orthogonal(1.0), bias_init=constant(0.0))
            elif self.adv_hypernet_layers == 2:
                return nn.Sequential([
                    nn.Dense(self.adv_hypernet_embed, kernel_init=orthogonal(1.0), bias_init=constant(0.0)),
                    nn.relu,
                    nn.Dense(self.n_agents, kernel_init=orthogonal(1.0), bias_init=constant(0.0))
                ])
            else:
                return nn.Sequential([
                    nn.Dense(self.adv_hypernet_embed, kernel_init=orthogonal(1.0), bias_init=constant(0.0)),
                    nn.relu,
                    nn.Dense(self.adv_hypernet_embed, kernel_init=orthogonal(1.0), bias_init=constant(0.0)),
                    nn.relu,
                    nn.Dense(self.n_agents, kernel_init=orthogonal(1.0), bias_init=constant(0.0))
                ])

        self.key_extractors = [make_key_extractor() for _ in range(self.num_kernel)]
        self.agent_extractors = [make_agent_extractor() for _ in range(self.num_kernel)]
        self.action_extractors = [make_action_extractor() for _ in range(self.num_kernel)]

    def __call__(self, states, actions, state_encoded=None):
        batch_size = states.shape[0]
        
        # Encode state if encoder provided and not already encoded
        if state_encoded is not None:
            states_flat = state_encoded
        elif self.state_encoder is not None:
            states_flat = self.state_encoder(states)
        else:
            states_flat = states.reshape(batch_size, -1)
        
        # Concatenate states and actions
        data = jnp.concatenate([states_flat, actions], axis=-1)

        head_attend_weights = []
        for i in range(self.num_kernel):
            x_key = jnp.abs(self.key_extractors[i](states_flat)) + 1e-10
            x_key = jnp.repeat(x_key, self.n_agents, axis=-1)
            x_agents = nn.sigmoid(self.agent_extractors[i](states_flat))
            x_action = nn.sigmoid(self.action_extractors[i](data))
            weights = x_key * x_agents * x_action
            head_attend_weights.append(weights)

        head_attend = jnp.stack(head_attend_weights, axis=1)
        head_attend = jnp.sum(head_attend, axis=1)
        return head_attend

class DMAQMixerOvercooked(nn.Module):
    """QPLEX Mixer adapted for Overcooked with CNN state encoding."""
    n_agents: int
    n_actions: int
    state_dim: int
    mixing_embed_dim: int = 32
    hypernet_embed: int = 64
    num_kernel: int = 10
    is_minus_one: bool = True
    weighted_head: bool = True
    adv_hypernet_embed: int = 64
    adv_hypernet_layers: int = 3
    state_encoder: nn.Module = None

    def setup(self):
        self.hyper_w_final = nn.Sequential([
            nn.Dense(self.hypernet_embed, kernel_init=orthogonal(1.0), bias_init=constant(0.0)),
            nn.relu,
            nn.Dense(self.n_agents, kernel_init=orthogonal(1.0), bias_init=constant(0.0))
        ])
        
        self.V = nn.Sequential([
            nn.Dense(self.hypernet_embed, kernel_init=orthogonal(1.0), bias_init=constant(0.0)),
            nn.relu,
            nn.Dense(self.n_agents, kernel_init=orthogonal(1.0), bias_init=constant(0.0))
        ])
        
        self.si_weight = DMAQ_SI_Weight(
            n_agents=self.n_agents,
            n_actions=self.n_actions,
            state_dim=self.state_dim,
            num_kernel=self.num_kernel,
            adv_hypernet_embed=self.adv_hypernet_embed,
            adv_hypernet_layers=self.adv_hypernet_layers,
            state_encoder=None,  # We'll pass encoded state directly
        )

    def calc_v(self, agent_qs):
        return jnp.sum(agent_qs, axis=-1)

    def calc_adv(self, agent_qs, states_encoded, actions, max_q_i):
        adv_q = agent_qs - max_q_i
        adv_q = jax.lax.stop_gradient(adv_q)
        adv_w_final = self.si_weight(states_encoded, actions, state_encoded=states_encoded)
        if self.is_minus_one:
            adv_tot = jnp.sum(adv_q * (adv_w_final - 1.0), axis=-1)
        else:
            adv_tot = jnp.sum(adv_q * adv_w_final, axis=-1)
        return adv_tot

    def __call__(self, agent_qs, states, actions=None, max_q_i=None, is_v=False):
        batch_size = agent_qs.shape[0]
        
        # Encode state using CNN if provided
        if self.state_encoder is not None:
            time_steps, batch_sz = states.shape[:2]
            states_reshaped = states.reshape(-1, *states.shape[2:])
            states_encoded = self.state_encoder(states_reshaped)
            states_encoded = states_encoded.reshape(time_steps * batch_sz, -1)
            # For single batch processing
            if batch_size != time_steps * batch_sz:
                states_encoded = states_encoded[:batch_size]
        else:
            states_encoded = states.reshape(batch_size, -1)
        
        agent_qs = agent_qs.reshape(batch_size, self.n_agents)
        
        w_final = jnp.abs(self.hyper_w_final(states_encoded)) + 1e-10
        v = self.V(states_encoded)
        
        if self.weighted_head:
            agent_qs = w_final * agent_qs + v
        
        if not is_v:
            max_q_i = max_q_i.reshape(batch_size, self.n_agents)
            if self.weighted_head:
                max_q_i = w_final * max_q_i + v
        
        if is_v:
            y = self.calc_v(agent_qs)
        else:
            y = self.calc_adv(agent_qs, states_encoded, actions, max_q_i)
        
        return y.reshape(batch_size, 1)

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

    def eps_greedy_exploration(rng, q_vals, eps, valid_actions):
        rng_a, rng_e = jax.random.split(rng)
        greedy_actions = get_greedy_actions(q_vals, valid_actions)

        def get_random_actions(rng, val_action):
            return jax.random.choice(
                rng,
                jnp.arange(val_action.shape[-1]),
                p=val_action * 1.0 / jnp.sum(val_action, axis=-1),
            )

        _rngs = jax.random.split(rng_a, valid_actions.shape[0])
        random_actions = jax.vmap(get_random_actions)(_rngs, valid_actions)

        chosen_actions = jnp.where(
            jax.random.uniform(rng_e, greedy_actions.shape) < eps,
            random_actions,
            greedy_actions,
        )
        return chosen_actions

    def batchify(x: dict):
        return jnp.stack([x[agent] for agent in env.agents], axis=0)

    def unbatchify(x: jnp.ndarray):
        return {agent: x[i] for i, agent in enumerate(env.agents)}

    def train(rng):
        original_seed = rng[0]

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        wrapped_env = CTRolloutManager(env, batch_size=config["NUM_ENVS"], preprocess_obs=False)
        test_env = CTRolloutManager(env, batch_size=config["TEST_NUM_ENVS"], preprocess_obs=False)

        # INIT BUFFER
        def _env_sample_step(env_state, unused):
            rng, key_a, key_s = jax.random.split(jax.random.PRNGKey(0), 3)
            key_a = jax.random.split(key_a, env.num_agents)
            actions = {
                agent: wrapped_env.batch_sample(key_a[i], agent)
                for i, agent in enumerate(env.agents)
            }
            avail_actions = wrapped_env.get_valid_actions(env_state.env_state)
            obs, env_state, rewards, dones, infos = wrapped_env.batch_step(key_s, env_state, actions)
            timestep = Timestep(
                obs=obs, actions=actions, rewards=rewards, dones=dones, avail_actions=avail_actions
            )
            return env_state, timestep

        _, _env_state = wrapped_env.batch_reset(rng)
        _, sample_traj = jax.lax.scan(_env_sample_step, _env_state, None, config["NUM_STEPS"])
        sample_traj_unbatched = jax.tree.map(lambda x: x[:, 0], sample_traj)
        
        buffer = fbx.make_trajectory_buffer(
            max_length_time_axis=int(config["BUFFER_SIZE"] // config["NUM_ENVS"]),
            min_length_time_axis=config["BUFFER_BATCH_SIZE"],
            sample_batch_size=config["BUFFER_BATCH_SIZE"],
            add_batch_size=config["NUM_ENVS"],
            sample_sequence_length=100,
            period=1,
        )
        buffer_state = buffer.init(sample_traj_unbatched)

        # Network dimensions
        n_agents = len(env.agents)
        n_actions = wrapped_env.max_action_space
        channels_per_agent = 18 + 4 * (env.layout.num_ingredients + 2)
        state_shape = (env.height, env.width, channels_per_agent * n_agents)

        # INIT NETWORKS
        network = CNNRNNQNetwork(
            action_dim=wrapped_env.max_action_space,
            hidden_dim=config["HIDDEN_SIZE"],
        )

        # state_encoder = CNNOvercooked()
        state_encoder = CNN()
        mixer = DMAQMixerOvercooked(
            n_agents=n_agents,
            n_actions=n_actions,
            state_dim=np.prod(state_shape),
            mixing_embed_dim=config.get("MIXING_EMBED_DIM", 32),
            hypernet_embed=config.get("HYPERNET_EMBED", 64),
            num_kernel=config.get("NUM_KERNEL", 10),
            is_minus_one=config.get("IS_MINUS_ONE", True),
            weighted_head=config.get("WEIGHTED_HEAD", True),
            adv_hypernet_embed=config.get("ADV_HYPERNET_EMBED", 64),
            adv_hypernet_layers=config.get("ADV_HYPERNET_LAYERS", 3),
            state_encoder=state_encoder,
        )

        def create_agent(rng):
            init_x = (
                jnp.zeros((1, 1, *env.observation_space().shape)),
                jnp.zeros((1, 1)),
            )
            init_hs = ScannedRNN.initialize_carry(config["HIDDEN_SIZE"], 1)
            
            rng, rng_agent, rng_mixer = jax.random.split(rng, 3)
            agent_params = network.init(rng_agent, init_hs, *init_x)

            # Initialize mixer
            dummy_agent_qs = jnp.zeros((1, n_agents))
            dummy_states = jnp.zeros((1, 1, *state_shape))
            dummy_actions = jnp.zeros((1, n_agents * n_actions))
            dummy_max_q_i = jnp.zeros((1, n_agents))
            mixer_params = mixer.init(
                rng_mixer, dummy_agent_qs, dummy_states, dummy_actions, dummy_max_q_i, False
            )

            network_params = {"agent": agent_params, "mixer": mixer_params}

            lr_scheduler = optax.linear_schedule(
                config["LR"], 1e-10, (config["NUM_EPOCHS"]) * config["NUM_UPDATES"]
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
        train_state = create_agent(_rng)

        def _update_step(runner_state, unused):
            train_state, buffer_state, expl_state, test_state, rng = runner_state

            def _step_env(carry, _):
                hs, last_obs, last_dones, env_state, rng = carry
                rng, rng_a, rng_s = jax.random.split(rng, 3)

                _obs = batchify(last_obs)[:, np.newaxis]
                _dones = batchify(last_dones)[:, np.newaxis]

                new_hs, q_vals = jax.vmap(network.apply, in_axes=(None, 0, 0, 0))(
                    train_state.params['agent'], hs, _obs, _dones
                )
                q_vals = q_vals.squeeze(axis=1)

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

                # Add shaped reward
                shaped_reward = infos.pop("shaped_reward")
                shaped_reward["__all__"] = batchify(shaped_reward).sum(axis=0)
                rewards = jax.tree.map(
                    lambda x, y: x + y * rew_shaping_anneal(train_state.timesteps),
                    rewards, shaped_reward
                )

                timestep = Timestep(
                    obs=last_obs, actions=actions, rewards=rewards,
                    dones=last_dones, avail_actions=avail_actions
                )
                return (new_hs, new_obs, dones, new_env_state, rng), (timestep, infos)

            rng, _rng = jax.random.split(rng)
            carry, (timesteps, infos) = jax.lax.scan(
                _step_env, (*expl_state, _rng), None, config["NUM_STEPS"]
            )
            expl_state = carry[:4]

            train_state = train_state.replace(
                timesteps=train_state.timesteps + config["NUM_STEPS"] * config["NUM_ENVS"]
            )

            buffer_traj_batch = jax.tree.map(
                lambda x: jnp.swapaxes(x, 0, 1)[:, np.newaxis], timesteps
            )
            buffer_state = buffer.add(buffer_state, buffer_traj_batch)

            def _learn_phase(carry, _):
                train_state, rng = carry
                rng, _rng = jax.random.split(rng)
                minibatch = buffer.sample(buffer_state, _rng).experience
                minibatch = jax.tree.map(lambda x: jnp.swapaxes(x[:, 0], 0, 1), minibatch)

                init_hs = ScannedRNN.initialize_carry(
                    config["HIDDEN_SIZE"], len(env.agents), config["BUFFER_BATCH_SIZE"]
                )

                _obs = batchify(minibatch.obs)
                _dones = batchify(minibatch.dones)
                _actions = batchify(minibatch.actions)
                _avail_actions = batchify(minibatch.avail_actions)

                # Process global state for mixer
                state_flat = minibatch.obs["__all__"]
                state = state_flat.reshape(
                    state_flat.shape[0], state_flat.shape[1],
                    n_agents, env.height, env.width, channels_per_agent
                ).transpose(0, 1, 3, 4, 2, 5).reshape(
                    state_flat.shape[0], state_flat.shape[1],
                    env.height, env.width, channels_per_agent * n_agents
                )

                _, q_next_target = jax.vmap(network.apply, in_axes=(None, 0, 0, 0))(
                    train_state.target_network_params['agent'], init_hs, _obs, _dones
                )

                def _loss_fn(params):
                    _, q_vals = jax.vmap(network.apply, in_axes=(None, 0, 0, 0))(
                        params['agent'], init_hs, _obs, _dones
                    )

                    chosen_action_q_vals = jnp.take_along_axis(
                        q_vals, _actions[..., np.newaxis], axis=-1
                    ).squeeze(-1)

                    unavailable_actions = 1 - _avail_actions
                    valid_q_vals = q_vals - (unavailable_actions * 1e10)
                    max_action_indices = jnp.argmax(valid_q_vals, axis=-1)

                    q_next = jnp.take_along_axis(
                        q_next_target, max_action_indices[..., np.newaxis], axis=-1
                    ).squeeze(-1)

                    # Reshape for mixer: (timesteps, batch_size, num_agents)
                    chosen_action_q_vals_t = jnp.transpose(chosen_action_q_vals, (1, 2, 0))
                    q_next_t = jnp.transpose(q_next, (1, 2, 0))
                    max_q_i = jnp.transpose(jnp.max(valid_q_vals, axis=-1), (1, 2, 0))

                    # One-hot encode actions
                    actions_onehot = jax.nn.one_hot(
                        jnp.transpose(_actions, (1, 2, 0)), n_actions
                    ).reshape(
                        _actions.shape[1], _actions.shape[2], -1
                    )

                    # QPLEX mixing functions
                    def mix_qplex_v(agent_qs, state_batch):
                        agent_qs = agent_qs[None, None, ...]
                        state_batch = state_batch[None, None, ...]
                        return mixer.apply(
                            params['mixer'], agent_qs, state_batch, None, None, True
                        ).squeeze(0)

                    def mix_qplex_a(agent_qs, state_batch, actions_oh, max_q):
                        agent_qs = agent_qs[None, None, ...]
                        state_batch = state_batch[None, None, ...]
                        actions_oh = actions_oh[None, ...]
                        max_q = max_q[None, ...]
                        return mixer.apply(
                            params['mixer'], agent_qs, state_batch, actions_oh, max_q, False
                        ).squeeze(0)

                    def mix_qplex_v_target(agent_qs, state_batch):
                        agent_qs = agent_qs[None, None, ...]
                        state_batch = state_batch[None, None, ...]
                        return mixer.apply(
                            train_state.target_network_params['mixer'],
                            agent_qs, state_batch, None, None, True
                        ).squeeze(0)

                    def mix_qplex_a_target(agent_qs, state_batch, actions_oh, max_q):
                        agent_qs = agent_qs[None, None, ...]
                        state_batch = state_batch[None, None, ...]
                        actions_oh = actions_oh[None, ...]
                        max_q = max_q[None, ...]
                        return mixer.apply(
                            train_state.target_network_params['mixer'],
                            agent_qs, state_batch, actions_oh, max_q, False
                        ).squeeze(0)

                    # Compute Q_tot = V_tot + A_tot (dueling)
                    q_v_chosen = jax.vmap(jax.vmap(mix_qplex_v, in_axes=(0, 0)), in_axes=(0, 0))(
                        chosen_action_q_vals_t[:-1], state[:-1]
                    )
                    q_a_chosen = jax.vmap(jax.vmap(mix_qplex_a, in_axes=(0, 0, 0, 0)), in_axes=(0, 0, 0, 0))(
                        chosen_action_q_vals_t[:-1], state[:-1], actions_onehot[:-1], max_q_i[:-1]
                    )
                    q_tot_chosen = q_v_chosen + q_a_chosen

                    # Target Q_tot
                    target_max_actions = jnp.transpose(max_action_indices[:, 1:], (1, 2, 0))
                    target_actions_onehot = jax.nn.one_hot(target_max_actions, n_actions).reshape(
                        target_max_actions.shape[0], target_max_actions.shape[1], -1
                    )

                    target_v = jax.vmap(jax.vmap(mix_qplex_v_target, in_axes=(0, 0)), in_axes=(0, 0))(
                        q_next_t[1:], state[1:]
                    )
                    target_adv = jax.vmap(jax.vmap(mix_qplex_a_target, in_axes=(0, 0, 0, 0)), in_axes=(0, 0, 0, 0))(
                        q_next_t[1:], state[1:], target_actions_onehot, max_q_i[1:]
                    )
                    target_q_tot = target_v + target_adv

                    # TD target
                    dones_tot = minibatch.dones["__all__"][:-1]
                    target = (
                        minibatch.rewards["__all__"][..., np.newaxis][:-1]
                        + (1 - dones_tot[..., np.newaxis]) * config["GAMMA"] * target_q_tot
                    )

                    td_error = q_tot_chosen - jax.lax.stop_gradient(target)
                    loss = jnp.mean(td_error ** 2)

                    return loss, chosen_action_q_vals_t.mean()

                (loss, qvals), grads = jax.value_and_grad(_loss_fn, has_aux=True)(train_state.params)
                train_state = train_state.apply_gradients(grads=grads)
                train_state = train_state.replace(grad_steps=train_state.grad_steps + 1)
                return (train_state, rng), (loss, qvals)

            rng, _rng = jax.random.split(rng)
            is_learn_time = buffer.can_sample(buffer_state) & (train_state.timesteps > config["LEARNING_STARTS"])
            
            (train_state, rng), (loss, qvals) = jax.lax.cond(
                is_learn_time,
                lambda ts, r: jax.lax.scan(_learn_phase, (ts, r), None, config["NUM_EPOCHS"]),
                lambda ts, r: ((ts, r), (jnp.zeros(config["NUM_EPOCHS"]), jnp.zeros(config["NUM_EPOCHS"]))),
                train_state, _rng
            )

            train_state = jax.lax.cond(
                train_state.n_updates % config["TARGET_UPDATE_INTERVAL"] == 0,
                lambda ts: ts.replace(
                    target_network_params=optax.incremental_update(
                        ts.params, ts.target_network_params, config["TAU"]
                    )
                ),
                lambda ts: ts,
                operand=train_state,
            )

            train_state = train_state.replace(n_updates=train_state.n_updates + 1)

            metrics = {
                "env_step": train_state.timesteps,
                "update_steps": train_state.n_updates,
                "grad_steps": train_state.grad_steps,
                "loss": loss.mean(),
                "qvals": qvals.mean(),
                "epsilon": eps_scheduler(train_state.n_updates),
            }
            metrics.update(jax.tree.map(lambda x: x.mean(), infos))


            # Test metrics
            if config.get("TEST_DURING_TRAINING", True):
                rng, _rng = jax.random.split(rng)
                test_state = jax.lax.cond(
                    train_state.n_updates % int(config["NUM_UPDATES"] * config["TEST_INTERVAL"]) == 0,
                    lambda _: get_greedy_metrics(_rng, train_state),
                    lambda _: test_state,
                    operand=None,
                )
                metrics.update({"test_" + k: v for k, v in test_state.items()})

            if config["WANDB_MODE"] != "disabled":
                def callback(metrics, original_seed):
                    scalar_metrics = {}
                    for k, v in metrics.items():
                        if hasattr(v, 'shape') and v.shape:
                            scalar_metrics[k] = float(v.mean())
                        else:
                            scalar_metrics[k] = float(v)
                    if config.get('WANDB_LOG_ALL_SEEDS', False):
                        scalar_metrics = {f"rng{int(original_seed)}/{k}": v for k, v in scalar_metrics.items()}
                    wandb.log(scalar_metrics)
                jax.debug.callback(callback, metrics, original_seed)

            runner_state = (train_state, buffer_state, expl_state, test_state, rng)
            return runner_state, None

        def get_greedy_metrics(rng, train_state):
            if not config.get("TEST_DURING_TRAINING", True):
                return None

            def _greedy_env_step(step_state, unused):
                env_state, last_obs, last_dones, hstate, rng = step_state
                rng, key_s = jax.random.split(rng)
                _obs = batchify(last_obs)[:, np.newaxis]
                _dones = batchify(last_dones)[:, np.newaxis]
                hstate, q_vals = jax.vmap(network.apply, in_axes=(None, 0, 0, 0))(
                    train_state.params['agent'], hstate, _obs, _dones
                )
                q_vals = q_vals.squeeze(axis=1)
                valid_actions = test_env.get_valid_actions(env_state.env_state)
                actions = get_greedy_actions(q_vals, batchify(valid_actions))
                actions = unbatchify(actions)
                obs, env_state, rewards, dones, infos = test_env.batch_step(key_s, env_state, actions)
                step_state = (env_state, obs, dones, hstate, rng)
                return step_state, (rewards, dones, infos)

            rng, _rng = jax.random.split(rng)
            init_obs, env_state = test_env.batch_reset(_rng)
            init_dones = {agent: jnp.zeros((config["TEST_NUM_ENVS"]), dtype=bool) for agent in env.agents + ["__all__"]}
            rng, _rng = jax.random.split(rng)
            hstate = ScannedRNN.initialize_carry(config["HIDDEN_SIZE"], len(env.agents), config["TEST_NUM_ENVS"])
            step_state = (env_state, init_obs, init_dones, hstate, _rng)
            step_state, (rewards, dones, infos) = jax.lax.scan(
                _greedy_env_step, step_state, None, config["TEST_NUM_STEPS"]
            )

            metrics = {
                "returned_episode_returns": jnp.nanmean(
                    jnp.where(infos["returned_episode"], infos["returned_episode_returns"], jnp.nan)
                )
            }
            return metrics

        rng, _rng = jax.random.split(rng)
        test_state = get_greedy_metrics(_rng, train_state)

        rng, _rng = jax.random.split(rng)
        init_obs, env_state = wrapped_env.batch_reset(_rng)
        init_dones = {agent: jnp.zeros((config["NUM_ENVS"]), dtype=bool) for agent in env.agents + ["__all__"]}
        init_hs = ScannedRNN.initialize_carry(config["HIDDEN_SIZE"], len(env.agents), config["NUM_ENVS"])
        expl_state = (init_hs, init_obs, init_dones, env_state)

        rng, _rng = jax.random.split(rng)
        runner_state = (train_state, buffer_state, expl_state, test_state, _rng)
        runner_state, metrics = jax.lax.scan(_update_step, runner_state, None, config["NUM_UPDATES"])

        return {"runner_state": runner_state, "metrics": metrics}

    return train


def env_from_config(config):
    env_name = config["ENV_NAME"]
    if "smax" in env_name.lower():
        config["ENV_KWARGS"]["scenario"] = map_name_to_scenario(config["MAP_NAME"])
        env_name = f"{config['ENV_NAME']}_{config['MAP_NAME']}"
        env = make(config["ENV_NAME"], **config["ENV_KWARGS"])
        env = SMAXLogWrapper(env)
    elif "overcooked_v2" in env_name.lower():
        env_name = f"{config['ENV_NAME']}_{config['ENV_KWARGS']['layout']}"
        config["ENV_KWARGS"]["layout"] = overcooked_v2_layouts[config["ENV_KWARGS"]["layout"]]
        env = make(config["ENV_NAME"], **config["ENV_KWARGS"])
        env = LogWrapper(env)
    elif "overcooked" in env_name.lower():
        env_name = f"{config['ENV_NAME']}_{config['ENV_KWARGS']['layout']}"
        config["ENV_KWARGS"]["layout"] = overcooked_layouts[config["ENV_KWARGS"]["layout"]]
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
    config = {**config, **config["alg"]}
    print("Config:\n", OmegaConf.to_yaml(config))

    alg_name = config.get("ALG_NAME", "qplex_cnn_rnn_overcooked")
    env, env_name = env_from_config(copy.deepcopy(config))

    wandb.init(
        entity=config["ENTITY"],
        project=config["PROJECT"],
        tags=[alg_name.upper(), env_name.upper(), f"jax_{jax.__version__}"],
        name=f"{alg_name}_{env_name}",
        config=config,
        mode=config["WANDB_MODE"],
        save_code=True
    )

    rng = jax.random.PRNGKey(config["SEED"])
    rngs = jax.random.split(rng, config["NUM_SEEDS"])
    train_vjit = jax.jit(jax.vmap(make_train(config, env)))
    outs = jax.block_until_ready(train_vjit(rngs))

    if config.get("SAVE_PATH", None) is not None:
        from jaxmarl.wrappers.baselines import save_params
        model_state = outs["runner_state"][0]
        save_dir = os.path.join(config["SAVE_PATH"], env_name)
        os.makedirs(save_dir, exist_ok=True)
        OmegaConf.save(config, os.path.join(save_dir, f'{alg_name}_{env_name}_seed{config["SEED"]}_config.yaml'))
        for i, rng in enumerate(rngs):
            params = jax.tree.map(lambda x: x[i], model_state.params)
            save_path = os.path.join(save_dir, f'{alg_name}_{env_name}_seed{config["SEED"]}_vmap{i}.safetensors')
            save_params(params, save_path)


def tune(default_config):
    default_config = {**default_config, **default_config["alg"]}
    alg_name = default_config.get("ALG_NAME", "qplex_cnn_rnn_overcooked")
    env, env_name = env_from_config(default_config)

    def wrapped_make_train():
        wandb.init(project=default_config["PROJECT"])
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
        "metric": {"name": "test_returned_episode_returns", "goal": "maximize"},
        "parameters": {
            "LR": {"values": [0.005, 0.001, 0.0005, 0.0001, 0.00005]},
            "NUM_ENVS": {"values": [8, 32, 64, 128]},
        },
    }

    wandb.login()
    sweep_id = wandb.sweep(sweep_config, entity=default_config["ENTITY"], project=default_config["PROJECT"])
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