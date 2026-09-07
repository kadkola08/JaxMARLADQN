# https://github.com/wjh720/QPLEX/ used as reference 
import os
import copy
import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
from typing import Any, Optional

import chex
import optax
import flax.linen as nn
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState
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


class RNNQNetwork(nn.Module):
    # homogenous agent for parameters sharing, assumes all agents have same obs and action dim
    action_dim: int
    hidden_dim: int
    init_scale: float = 1.0

    @nn.compact
    def __call__(self, hidden, obs, dones):
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


class DMAQ_SI_Weight(nn.Module):
    """State-Independent weight module for advantage computation."""
    n_agents: int
    n_actions: int
    state_dim: int
    num_kernel: int = 10
    adv_hypernet_embed: int = 64
    adv_hypernet_layers: int = 3

    def setup(self):
        action_dim = self.n_agents * self.n_actions
        state_action_dim = self.state_dim + action_dim

        # Create extractors for each kernel
        def make_key_extractor():
            if self.adv_hypernet_layers == 1:
                return nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))
            elif self.adv_hypernet_layers == 2:
                return nn.Sequential([
                    nn.Dense(self.adv_hypernet_embed, kernel_init=orthogonal(1.0), bias_init=constant(0.0)),
                    nn.relu,
                    nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))
                ])
            else:  # 3 layers
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
            else:  # 3 layers
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
            else:  # 3 layers
                return nn.Sequential([
                    nn.Dense(self.adv_hypernet_embed, kernel_init=orthogonal(1.0), bias_init=constant(0.0)),
                    nn.relu,
                    nn.Dense(self.adv_hypernet_embed, kernel_init=orthogonal(1.0), bias_init=constant(0.0)),
                    nn.relu,
                    nn.Dense(self.n_agents, kernel_init=orthogonal(1.0), bias_init=constant(0.0))
                ])

        # Create extractors for each kernel
        self.key_extractors = [make_key_extractor() for _ in range(self.num_kernel)]
        self.agent_extractors = [make_agent_extractor() for _ in range(self.num_kernel)]
        self.action_extractors = [make_action_extractor() for _ in range(self.num_kernel)]

    def __call__(self, states, actions):
        # states: (batch_size, state_dim)
        # actions: (batch_size, n_agents * n_actions) - one-hot encoded
        batch_size = states.shape[0]
        action_dim = self.n_agents * self.n_actions

        # Concatenate states and actions
        data = jnp.concatenate([states, actions], axis=-1)  # (batch_size, state_action_dim)

        # Multi-kernel attention mechanism
        head_attend_weights = []
        
        for i in range(self.num_kernel):
            # Apply extractors
            x_key = jnp.abs(self.key_extractors[i](states)) + 1e-10  # (batch_size, 1)
            x_key = jnp.repeat(x_key, self.n_agents, axis=-1)  # (batch_size, n_agents)
            
            x_agents = nn.sigmoid(self.agent_extractors[i](states))  # (batch_size, n_agents)
            x_action = nn.sigmoid(self.action_extractors[i](data))  # (batch_size, n_agents)
            
            # Combine: key * agents * action
            weights = x_key * x_agents * x_action  # (batch_size, n_agents)
            head_attend_weights.append(weights)

        # Stack and sum across kernels
        head_attend = jnp.stack(head_attend_weights, axis=1)  # (batch_size, num_kernel, n_agents)
        head_attend = jnp.sum(head_attend, axis=1)  # (batch_size, n_agents)

        return head_attend


class DMAQMixer(nn.Module):
    """Duplex Multi-Agent Q-learning Mixer (QPLEX)."""
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

    def setup(self):
        # Hypernetwork for value function weights
        self.hyper_w_final = nn.Sequential([
            nn.Dense(self.hypernet_embed, kernel_init=orthogonal(1.0), bias_init=constant(0.0)),
            nn.relu,
            nn.Dense(self.n_agents, kernel_init=orthogonal(1.0), bias_init=constant(0.0))
        ])
        
        # State value function V(s)
        self.V = nn.Sequential([
            nn.Dense(self.hypernet_embed, kernel_init=orthogonal(1.0), bias_init=constant(0.0)),
            nn.relu,
            nn.Dense(self.n_agents, kernel_init=orthogonal(1.0), bias_init=constant(0.0))
        ])
        
        # SI weight module for advantage
        self.si_weight = DMAQ_SI_Weight(
            n_agents=self.n_agents,
            n_actions=self.n_actions,
            state_dim=self.state_dim,
            num_kernel=self.num_kernel,
            adv_hypernet_embed=self.adv_hypernet_embed,
            adv_hypernet_layers=self.adv_hypernet_layers
        )

    def calc_v(self, agent_qs):
        """Compute value function: V_tot = sum(Q_i)"""
        # agent_qs: (batch_size, n_agents)
        v_tot = jnp.sum(agent_qs, axis=-1)  # (batch_size,)
        return v_tot

    def calc_adv(self, agent_qs, states, actions, max_q_i):
        """Compute advantage function: A_tot = weighted sum of (Q_i - max Q_i)"""
        # agent_qs: (batch_size, n_agents)
        # states: (batch_size, state_dim)
        # actions: (batch_size, n_agents * n_actions) - one-hot
        # max_q_i: (batch_size, n_agents)
        
        # Compute advantage: Q_i(a_i) - max_a Q_i(a)
        adv_q = agent_qs - max_q_i  # (batch_size, n_agents)
        adv_q = jax.lax.stop_gradient(adv_q)  # Stop gradient in advantage computation
        
        # Get state-action dependent weights
        adv_w_final = self.si_weight(states, actions)  # (batch_size, n_agents)
        
        # Apply weighting (with optional minus_one trick)
        if self.is_minus_one:
            adv_tot = jnp.sum(adv_q * (adv_w_final - 1.0), axis=-1)  # (batch_size,)
        else:
            adv_tot = jnp.sum(adv_q * adv_w_final, axis=-1)  # (batch_size,)
        
        return adv_tot

    def __call__(self, agent_qs, states, actions=None, max_q_i=None, is_v=False):
        """
        Forward pass of QPLEX mixer.
        
        Args:
            agent_qs: (batch_size, n_agents) - Q-values for chosen actions
            states: (batch_size, state_dim) - Global state
            actions: (batch_size, n_agents * n_actions) - One-hot encoded actions (for advantage)
            max_q_i: (batch_size, n_agents) - Max Q-values (for advantage)
            is_v: bool - If True, compute only value; if False, compute advantage
        
        Returns:
            q_tot: (batch_size, 1) - Total Q-value
        """
        batch_size = agent_qs.shape[0]
        states = states.reshape(batch_size, -1)  # Ensure flattened
        agent_qs = agent_qs.reshape(batch_size, self.n_agents)
        
        # Compute state-dependent weights and value bias
        w_final = jnp.abs(self.hyper_w_final(states))  # (batch_size, n_agents)
        w_final = w_final + 1e-10
        v = self.V(states)  # (batch_size, n_agents)
        
        # Apply weighted head transformation
        if self.weighted_head:
            agent_qs = w_final * agent_qs + v
        
        if not is_v:
            # For advantage computation, also transform max_q_i
            max_q_i = max_q_i.reshape(batch_size, self.n_agents)
            if self.weighted_head:
                max_q_i = w_final * max_q_i + v
        
        # Compute value or advantage
        if is_v:
            y = self.calc_v(agent_qs)
        else:
            y = self.calc_adv(agent_qs, states, actions, max_q_i)
        
        return y.reshape(batch_size, 1)  # (batch_size, 1)


@chex.dataclass(frozen=True)
class Timestep:
    obs: dict
    actions: dict
    rewards: dict
    dones: dict
    avail_actions: dict


class CustomTrainState(TrainState):
    target_network_params: Any
    target_mixer_params: Any
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

    def count_params(params):
        def size(x):
            if hasattr(x, "size"):
                return x.size
            if isinstance(x, (list, tuple)):
                return sum(size(e) for e in x)
            return 0

        return sum(size(p) for p in jax.tree.flatten(params))

    def train(rng):

        # INIT ENV
        original_seed = rng[0]
        rng, _rng = jax.random.split(rng)
        wrapped_env = CTRolloutManager(env, batch_size=config["NUM_ENVS"])
        test_env = CTRolloutManager(
            env, batch_size=config["TEST_NUM_ENVS"]
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
        sample_traj_unbatched = jax.tree.map(
            lambda x: x[:, 0], sample_traj
        )  # remove the NUM_ENV dim
        buffer = fbx.make_trajectory_buffer(
            max_length_time_axis=config["BUFFER_SIZE"] // config["NUM_ENVS"],
            min_length_time_axis=config["BUFFER_BATCH_SIZE"],
            sample_batch_size=config["BUFFER_BATCH_SIZE"],
            add_batch_size=config["NUM_ENVS"],
            sample_sequence_length=1,
            period=1,
        )
        buffer_state = buffer.init(sample_traj_unbatched)

        # Get state dimension (for mixer)
        # We'll need to get this from the environment or config
        state_dim = config.get("STATE_DIM", wrapped_env.obs_size * len(env.agents))
        n_agents = len(env.agents)
        n_actions = wrapped_env.max_action_space

        # INIT NETWORK AND OPTIMIZER
        network = RNNQNetwork(
            action_dim=wrapped_env.max_action_space,
            hidden_dim=config["HIDDEN_SIZE"],
        )

        mixer = DMAQMixer(
            n_agents=n_agents,
            n_actions=n_actions,
            state_dim=state_dim,
            mixing_embed_dim=config.get("MIXING_EMBED_DIM", 32),
            hypernet_embed=config.get("HYPERNET_EMBED", 64),
            num_kernel=config.get("NUM_KERNEL", 10),
            is_minus_one=config.get("IS_MINUS_ONE", True),
            weighted_head=config.get("WEIGHTED_HEAD", True),
            adv_hypernet_embed=config.get("ADV_HYPERNET_EMBED", 64),
            adv_hypernet_layers=config.get("ADV_HYPERNET_LAYERS", 3),
        )

        def create_agent(rng):
            init_x = (
                jnp.zeros(
                    (1, 1, wrapped_env.obs_size)
                ),  # (time_step, batch_size, obs_size)
                jnp.zeros((1, 1)),  # (time_step, batch size)
            )
            init_hs = ScannedRNN.initialize_carry(
                config["HIDDEN_SIZE"], 1
            )  # (batch_size, hidden_dim)
            network_params = network.init(rng, init_hs, *init_x)

            state_size = sample_traj.obs["__all__"].shape[
                -1
            ]  # get the state shape from the buffer
            init_state = jnp.zeros((1, 1, state_size)) # (time_step, batch_size, obs_size)

            # Initialize mixer
            rng_mixer, rng = jax.random.split(rng)
            dummy_agent_qs = jnp.zeros((1, n_agents))
            # dummy_states = jnp.zeros((1, state_dim))
            dummy_states = jnp.zeros((1, 1, state_size)) # (time_step, batch_size, obs_size)
            dummy_actions = jnp.zeros((1, n_agents * n_actions))
            dummy_max_q_i = jnp.zeros((1, n_agents))
            mixer_params = mixer.init(
                rng_mixer,
                dummy_agent_qs,
                dummy_states,
                dummy_actions,
                dummy_max_q_i,
                False
            )

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
                target_mixer_params=mixer_params,
                tx=tx,
            )
            # Create separate optimizer for mixer
            tx_mixer = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.radam(learning_rate=lr),
            )
            mixer_opt_state = tx_mixer.init(mixer_params)
            return train_state, mixer_params, mixer_opt_state, tx_mixer

        rng, _rng = jax.random.split(rng)
        train_state, mixer_params, mixer_opt_state, tx_mixer = create_agent(rng)
        num_agent_params = count_params(train_state.params)
        num_mixer_params = count_params(mixer_params)
        jax.debug.breakpoint()

        # TRAINING LOOP
        def _update_step(runner_state, unused):

            train_state, mixer_params, mixer_opt_state, target_mixer_params, buffer_state, test_state, rng = runner_state

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
                    train_state.params,
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
                timestep = Timestep(
                    obs=last_obs,
                    actions=actions,
                    rewards=jax.tree.map(lambda x:config.get("REW_SCALE", 1)*x, rewards),
                    dones=last_dones,
                    avail_actions=avail_actions,
                )
                return (new_hs, new_obs, dones, new_env_state, rng), (timestep, infos)

            # step the env (should be a complete rollout)
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
            )  # update timesteps count

            # BUFFER UPDATE
            buffer_traj_batch = jax.tree.map(
                lambda x: jnp.swapaxes(x, 0, 1)[
                    :, np.newaxis
                ],  # put the batch dim first and add a dummy sequence dim
                timesteps,
            )  # (num_envs, 1, time_steps, ...)
            buffer_state = buffer.add(buffer_state, buffer_traj_batch)

            # NETWORKS UPDATE
            def _learn_phase(carry, _):

                train_state, mixer_params, mixer_opt_state, rng = carry
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
                # num_agents, timesteps, batch_size, ...
                _obs = batchify(minibatch.obs)
                _dones = batchify(minibatch.dones)
                _actions = batchify(minibatch.actions)
                _rewards = batchify(minibatch.rewards)
                _avail_actions = batchify(minibatch.avail_actions)

                # Get global state (concatenate all agent observations)
                # For simplicity, we'll use concatenated observations as state
                # In practice, you might want to use a proper global state
                _states = minibatch.obs["__all__"]
                # _states = jnp.concatenate(
                    # [_obs[i] for i in range(len(env.agents))], axis=-1
                # )  # (timesteps, batch_size, state_dim)

                # Get Q-values from target network
                _, q_next_target = jax.vmap(network.apply, in_axes=(None, 0, 0, 0))(
                    train_state.target_network_params,
                    init_hs,
                    _obs,
                    _dones,
                )  # (num_agents, timesteps, batch_size, num_actions)

                # Get Q-values from main network
                def _loss_fn(params, mixer_p):
                    _, q_vals = jax.vmap(network.apply, in_axes=(None, 0, 0, 0))(
                        params,
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

                    # get the q values of the next state (for double Q-learning)
                    max_action_indices = jnp.argmax(valid_q_vals, axis=-1)  # (num_agents, timesteps, batch_size)
                    q_next = jnp.take_along_axis(
                        q_next_target,
                        max_action_indices[..., np.newaxis],
                        axis=-1,
                    ).squeeze(
                        -1
                    )  # (num_agents, timesteps, batch_size,)

                    # Reshape for mixer: (timesteps, batch_size, num_agents)
                    chosen_action_q_vals = jnp.transpose(chosen_action_q_vals, (1, 2, 0))
                    q_next = jnp.transpose(q_next, (1, 2, 0))
                    max_q_i = jnp.transpose(jnp.max(valid_q_vals, axis=-1), (1, 2, 0))

                    # Convert actions to one-hot for mixer
                    # _actions: (num_agents, timesteps, batch_size)
                    actions_onehot = jax.nn.one_hot(
                        jnp.transpose(_actions, (1, 2, 0)), 
                        n_actions
                    )  # (timesteps, batch_size, num_agents, n_actions)
                    actions_onehot = actions_onehot.reshape(
                        actions_onehot.shape[0], 
                        actions_onehot.shape[1], 
                        -1
                    )  # (timesteps, batch_size, num_agents * n_actions)

                    # Compute Q_tot using QPLEX mixer (dueling: V + A)
                    # For current state-action pairs (use main mixer)
                    def mix_qplex_v(agent_qs, state):
                        """Compute value component V_tot"""
                        agent_qs = agent_qs[None, ...]  # (1, n_agents)
                        state = state[None, ...]  # (1, state_dim)
                        return mixer.apply(
                            mixer_p,
                            agent_qs,
                            state,
                            None,
                            None,
                            True
                        ).squeeze(0) 
                    
                    def mix_qplex_a(agent_qs, state, actions, max_q):
                        """Compute advantage component A_tot"""
                        agent_qs = agent_qs[None, ...]  # (1, n_agents)
                        state = state[None, ...]  # (1, state_dim)
                        actions = actions[None, ...]  # (1, n_agents * n_actions)
                        max_q = max_q[None, ...]  # (1, n_agents)
                        return mixer.apply(
                            mixer_p,
                            agent_qs,
                            state,
                            actions,
                            max_q,
                            False
                        ).squeeze(0) 
                    
                    # For target computation (use target mixer)
                    def mix_qplex_target(agent_qs, state, actions, max_q):
                        agent_qs = agent_qs[None, ...]
                        state = state[None, ...]
                        actions = actions[None, ...]
                        max_q = max_q[None, ...]
                        return mixer.apply(
                            train_state.target_mixer_params,
                            agent_qs,
                            state,
                            actions,
                            max_q,
                            False
                        ).squeeze(0)
                    
                    # For value computation (target)
                    def mix_qplex_v_target(agent_qs, state):
                        agent_qs = agent_qs[None, ...]
                        state = state[None, ...]
                        return mixer.apply(
                            train_state.target_mixer_params,
                            agent_qs,
                            state,
                            None,
                            None,
                            True
                        ).squeeze(0)

                    # DUELING: Compute V_tot and A_tot separately, then combine
                    # Value component
                    q_v_chosen = jax.vmap(jax.vmap(mix_qplex_v, in_axes=(0, 0)), in_axes=(0, 0))(
                        chosen_action_q_vals[:-1],  # (timesteps-1, batch_size, num_agents)
                        _states[:-1]  # (timesteps-1, batch_size, state_dim)
                    )  # (timesteps-1, batch_size, 1)
                    
                    # Advantage component
                    q_a_chosen = jax.vmap(jax.vmap(mix_qplex_a, in_axes=(0, 0, 0, 0)), in_axes=(0, 0, 0, 0))(
                        chosen_action_q_vals[:-1],  # (timesteps-1, batch_size, num_agents)
                        _states[:-1],  # (timesteps-1, batch_size, state_dim)
                        actions_onehot[:-1],  # (timesteps-1, batch_size, num_agents*n_actions)
                        max_q_i[:-1]  # (timesteps-1, batch_size, num_agents)
                    )  # (timesteps-1, batch_size, 1)
                    
                    # DUELING HAPPENS HERE: Q_tot = V_tot + A_tot
                    q_tot_chosen = q_v_chosen + q_a_chosen  # (timesteps-1, batch_size, 1)

                    # Mix target Q-values using target mixer
                    # For double Q-learning, use the actions selected by main network (max_action_indices)
                    # max_action_indices: (num_agents, timesteps, batch_size)
                    target_max_actions = jnp.transpose(max_action_indices[:, 1:], (1, 2, 0))  # (timesteps-1, batch_size, num_agents)
                    target_actions_onehot = jax.nn.one_hot(
                        target_max_actions,
                        n_actions
                    )  # (timesteps-1, batch_size, num_agents, n_actions)
                    target_actions_onehot = target_actions_onehot.reshape(
                        target_actions_onehot.shape[0],
                        target_actions_onehot.shape[1],
                        -1
                    )  # (timesteps-1, batch_size, num_agents*n_actions)

                    # Compute target: V_tot + A_tot (using target mixer)
                    target_v = jax.vmap(jax.vmap(mix_qplex_v_target, in_axes=(0, 0)), in_axes=(0, 0))(
                        q_next[1:],  # (timesteps-1, batch_size, num_agents)
                        _states[1:]  # (timesteps-1, batch_size, state_dim)
                    )  # (timesteps-1, batch_size, 1)

                    target_adv = jax.vmap(jax.vmap(mix_qplex_target, in_axes=(0, 0, 0, 0)), in_axes=(0, 0, 0, 0))(
                        q_next[1:],  # (timesteps-1, batch_size, num_agents)
                        _states[1:],  # (timesteps-1, batch_size, state_dim)
                        target_actions_onehot,  # (timesteps-1, batch_size, num_agents*n_actions)
                        max_q_i[1:]  # (timesteps-1, batch_size, num_agents)
                    )  # (timesteps-1, batch_size, 1)

                    target_q_tot = target_v + target_adv  # (timesteps-1, batch_size, 1)

                    # Compute TD target
                    # rewards_tot = jnp.sum(
                    #     jnp.stack([_rewards[agent][:-1] for agent in env.agents], axis=0),
                    #     axis=0
                    # )  # (timesteps-1, batch_size)
                    dones_tot = minibatch.dones["__all__"][:-1]  # (timesteps-1, batch_size)

                    target = (minibatch.rewards["__all__"][..., np.newaxis][:-1]
                        + (1 - dones_tot[..., np.newaxis]) * config["GAMMA"] * target_q_tot
                    )  # (timesteps-1, batch_size, 1)

                    # TD error
                    td_error = q_tot_chosen - jax.lax.stop_gradient(target)
                    loss = jnp.mean(td_error ** 2)

                    return loss, q_tot_chosen.mean()

                (loss, qvals), grads = jax.value_and_grad(_loss_fn, has_aux=True, argnums=(0, 1))(
                    train_state.params, mixer_params
                )
                
                # Apply gradients
                train_state = train_state.apply_gradients(grads=grads[0])
                
                # Apply mixer gradients with optimizer
                mixer_updates, mixer_opt_state = tx_mixer.update(grads[1], mixer_opt_state, mixer_params)
                mixer_params = optax.apply_updates(mixer_params, mixer_updates)
                
                train_state = train_state.replace(
                    grad_steps=train_state.grad_steps + 1,
                )
                return (train_state, mixer_params, mixer_opt_state, rng), (loss, qvals)

            rng, _rng = jax.random.split(rng)
            is_learn_time = (
                buffer.can_sample(buffer_state)
            ) & (  # enough experience in buffer
                train_state.timesteps > config["LEARNING_STARTS"]
            )
            (train_state, mixer_params, mixer_opt_state, rng), (loss, qvals) = jax.lax.cond(
                is_learn_time,
                lambda carry: jax.lax.scan(
                    _learn_phase, carry, None, config["NUM_EPOCHS"]
                ),
                lambda carry: (
                    carry,
                    (
                        jnp.zeros(config["NUM_EPOCHS"]),
                        jnp.zeros(config["NUM_EPOCHS"]),
                    ),
                ),  # do nothing
                (train_state, mixer_params, mixer_opt_state, _rng),
            )

            # update target network and mixer
            train_state = jax.lax.cond(
                train_state.n_updates % config["TARGET_UPDATE_INTERVAL"] == 0,
                lambda ts: ts.replace(
                    target_network_params=optax.incremental_update(
                        ts.params,
                        ts.target_network_params,
                        config["TAU"],
                    ),
                    target_mixer_params=optax.incremental_update(
                        mixer_params,
                        ts.target_mixer_params,
                        config["TAU"],
                    )
                ),
                lambda ts: ts,
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
            }
            metrics.update(jax.tree.map(lambda x: x.mean(), infos))
            if config.get("LOG_AGENTS_SEPARATELY", False):
                for i, a in enumerate(env.agents):
                    m = jax.tree.map(
                        lambda x: x[..., i].mean(),
                        infos,
                    )
                    m = {k + f"_{a}": v for k, v in m.items()}
                    metrics.update(m)

            # update the test metrics
            if config.get("TEST_DURING_TRAINING", True):
                rng, _rng = jax.random.split(rng)
                test_state = jax.lax.cond(
                    train_state.n_updates
                    % int(config["NUM_UPDATES"] * config["TEST_INTERVAL"])
                    == 0,
                    lambda _: get_greedy_metrics(_rng, train_state, mixer_params),
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

            runner_state = (train_state, mixer_params, mixer_opt_state, train_state.target_mixer_params, buffer_state, test_state, rng)

            return runner_state, None

        def get_greedy_metrics(rng, train_state, mixer_params):
            """Help function to test greedy policy during training"""
            if not config.get("TEST_DURING_TRAINING", True):
                return None
            params = train_state.params
            def _greedy_env_step(step_state, unused):
                params, mixer_p, env_state, last_obs, last_dones, hstate, rng = step_state
                rng, key_s = jax.random.split(rng)
                _obs = batchify(last_obs)[:, np.newaxis]
                _dones = batchify(last_dones)[:, np.newaxis]
                hstate, q_vals = jax.vmap(network.apply, in_axes=(None, 0, 0, 0))(
                    params,
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
                step_state = (params, mixer_p, env_state, obs, dones, hstate, rng)
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
                params,
                mixer_params,
                env_state,
                init_obs,
                init_dones,
                hstate,
                _rng,
            )
            step_state, (rewards, dones, infos) = jax.lax.scan(
                _greedy_env_step, step_state, None, config["TEST_NUM_STEPS"]
            )
            if config.get("LOG_AGENTS_SEPARATELY", False):
                metrics = {}
                for i, a in enumerate(env.agents):
                    m = jax.tree.map(
                        lambda x: jnp.nanmean(
                            jnp.where(
                                infos["returned_episode"][..., i],
                                x[..., i],
                                jnp.nan,
                            )
                        ),
                        infos,
                    )
                    m = {k + f"_{a}": v for k, v in m.items()}
                    metrics.update(m)
            else:
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
        test_state = get_greedy_metrics(_rng, train_state, mixer_params)

        # train
        rng, _rng = jax.random.split(rng)
        runner_state = (train_state, mixer_params, mixer_opt_state, train_state.target_mixer_params, buffer_state, test_state, _rng)

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

    alg_name = config.get("ALG_NAME", "qplex_rnn")
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
    alg_name = default_config.get("ALG_NAME", "qplex_rnn") 
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

