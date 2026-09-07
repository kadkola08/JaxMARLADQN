import os
import copy
import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
from typing import Any

import chex
import optax
import flax.linen as nn
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState
from gymnax.wrappers.purerl import LogWrapper
import hydra
from omegaconf import OmegaConf
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
        return nn.GRUCell(hidden_size, parent=None).initialize_carry(
            jax.random.PRNGKey(0), (*batch_size, hidden_size)
        )


class RNNQNetwork(nn.Module):
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


class HyperNetwork(nn.Module):
    """HyperNetwork for generating weights of QMix' mixing network."""
    hidden_dim: int
    output_dim: int
    init_scale: float

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(
            self.hidden_dim,
            kernel_init=orthogonal(self.init_scale),
            bias_init=constant(0.0),
        )(x)
        x = nn.relu(x)
        x = nn.Dense(
            self.output_dim,
            kernel_init=orthogonal(self.init_scale),
            bias_init=constant(0.0),
        )(x)
        return x


class MixingNetwork(nn.Module):
    """Mixing network for projecting individual agent Q-values into Q_tot."""
    embedding_dim: int
    hypernet_hidden_dim: int
    init_scale: float

    @nn.compact
    def __call__(self, q_vals, states):
        n_agents, time_steps, batch_size = q_vals.shape
        q_vals = jnp.transpose(q_vals, (1, 2, 0))

        w_1 = HyperNetwork(
            hidden_dim=self.hypernet_hidden_dim,
            output_dim=self.embedding_dim * n_agents,
            init_scale=self.init_scale,
        )(states)
        b_1 = nn.Dense(
            self.embedding_dim,
            kernel_init=orthogonal(self.init_scale),
            bias_init=constant(0.0),
        )(states)
        w_2 = HyperNetwork(
            hidden_dim=self.hypernet_hidden_dim,
            output_dim=self.embedding_dim,
            init_scale=self.init_scale,
        )(states)
        b_2 = HyperNetwork(
            hidden_dim=self.embedding_dim, output_dim=1, init_scale=self.init_scale
        )(states)

        # monotonicity constraint via abs()
        w_1 = jnp.abs(w_1.reshape(time_steps, batch_size, n_agents, self.embedding_dim))
        b_1 = b_1.reshape(time_steps, batch_size, 1, self.embedding_dim)
        w_2 = jnp.abs(w_2.reshape(time_steps, batch_size, self.embedding_dim, 1))
        b_2 = b_2.reshape(time_steps, batch_size, 1, 1)

        hidden = nn.elu(jnp.matmul(q_vals[:, :, None, :], w_1) + b_1)
        q_tot = jnp.matmul(hidden, w_2) + b_2

        return q_tot.squeeze()


class RNDTargetNetwork(nn.Module):
    """Fixed random target network g(·) for RND. Never trained."""
    hidden_dim: int
    output_dim: int
    
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.hidden_dim, kernel_init=orthogonal(1.0))(x)
        x = nn.relu(x)
        x = nn.Dense(self.hidden_dim, kernel_init=orthogonal(1.0))(x)
        x = nn.relu(x)
        x = nn.Dense(self.output_dim, kernel_init=orthogonal(1.0))(x)
        return x


class RNDPredictorNetwork(nn.Module):
    """Predictor network ĝ(·; θ_predictor) for RND. Trained to predict target output."""
    hidden_dim: int
    output_dim: int
    
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.hidden_dim, kernel_init=orthogonal(1.0))(x)
        x = nn.relu(x)
        x = nn.Dense(self.hidden_dim, kernel_init=orthogonal(1.0))(x)
        x = nn.relu(x)
        x = nn.Dense(self.hidden_dim, kernel_init=orthogonal(1.0))(x)
        x = nn.relu(x)
        x = nn.Dense(self.output_dim, kernel_init=orthogonal(1.0))(x)
        return x


class RunningMeanStd:
    """Running mean and std for observation and reward normalization."""
    
    @staticmethod
    def init(shape):
        return {
            'mean': jnp.zeros(shape),
            'var': jnp.ones(shape),
            'count': jnp.array(1e-4),
        }
    
    @staticmethod
    def update(state, x):
        """Update running statistics with new batch of data."""
        batch_mean = jnp.mean(x, axis=0)
        batch_var = jnp.var(x, axis=0)
        batch_count = x.shape[0]
        
        delta = batch_mean - state['mean']
        tot_count = state['count'] + batch_count
        
        new_mean = state['mean'] + delta * batch_count / tot_count
        m_a = state['var'] * state['count']
        m_b = batch_var * batch_count
        M2 = m_a + m_b + jnp.square(delta) * state['count'] * batch_count / tot_count
        new_var = M2 / tot_count
        
        return {
            'mean': new_mean,
            'var': new_var,
            'count': tot_count,
        }
    
    @staticmethod
    def normalize(state, x, clip_val=5.0):
        """Normalize observations using running statistics."""
        return jnp.clip(
            (x - state['mean']) / jnp.sqrt(state['var'] + 1e-8),
            -clip_val,
            clip_val
        )

@chex.dataclass(frozen=True)
class Timestep:
    obs: dict
    actions: dict
    rewards: dict
    dones: dict
    avail_actions: dict


class CustomTrainState(TrainState):
    target_network_params: Any
    timesteps: int = 0
    n_updates: int = 0
    grad_steps: int = 0


class RNDTrainState(TrainState):
    """Separate train state for RND predictor."""
    pass

def make_train(config, env):

    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )

    # RND config defaults
    rnd_hidden_dim = config.get("RND_HIDDEN_DIM", 256)
    rnd_output_dim = config.get("RND_OUTPUT_DIM", 64)
    rnd_lr = config.get("RND_LR", 1e-4)
    intrinsic_reward_coef_start = config.get("INTRINSIC_REWARD_COEF_START", 0.5) 
    intrinsic_reward_coef_end = config.get("INTRINSIC_REWARD_COEF_END", 0.005) 
    rnd_update_proportion = config.get("RND_UPDATE_PROPORTION", 0.05)

    eps_scheduler = optax.linear_schedule(
        init_value=config["EPS_START"],
        end_value=config["EPS_FINISH"],
        transition_steps=config["EPS_DECAY"] * config["NUM_UPDATES"],
    )

    rnd_scheduler = optax.linear_schedule(
        init_value=intrinsic_reward_coef_start,
        end_value=intrinsic_reward_coef_end,
        transition_steps=rnd_update_proportion * config["NUM_UPDATES"],
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

        chosed_actions = jnp.where(
            jax.random.uniform(rng_e, greedy_actions.shape) < eps,
            random_actions,
            greedy_actions,
        )
        return chosed_actions

    def batchify(x: dict):
        return jnp.stack([x[agent] for agent in env.agents], axis=0)

    def unbatchify(x: jnp.ndarray):
        return {agent: x[i] for i, agent in enumerate(env.agents)}

    def train(rng):

        original_seed = rng[0]
        rng, _rng = jax.random.split(rng)
        wrapped_env = CTRolloutManager(env, batch_size=config["NUM_ENVS"])
        test_env = CTRolloutManager(env, batch_size=config["TEST_NUM_ENVS"])

        def _env_sample_step(env_state, unused):
            rng, key_a, key_s = jax.random.split(jax.random.PRNGKey(0), 3)
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
                obs=obs, actions=actions, rewards=rewards,
                dones=dones, avail_actions=avail_actions,
            )
            return env_state, timestep

        _, _env_state = wrapped_env.batch_reset(rng)
        _, sample_traj = jax.lax.scan(
            _env_sample_step, _env_state, None, config["NUM_STEPS"]
        )
        sample_traj_unbatched = jax.tree.map(lambda x: x[:, 0], sample_traj)

        # INIT Q-NETWORK AND MIXER
        network = RNNQNetwork(
            action_dim=wrapped_env.max_action_space,
            hidden_dim=config["HIDDEN_SIZE"],
        )
        mixer = MixingNetwork(
            config["MIXER_EMBEDDING_DIM"],
            config["MIXER_HYPERNET_HIDDEN_DIM"],
            config["MIXER_INIT_SCALE"],
        )

        # INIT RND NETWORKS
        state_size = sample_traj.obs["__all__"].shape[-1]
        rnd_target = RNDTargetNetwork(hidden_dim=rnd_hidden_dim, output_dim=rnd_output_dim)
        rnd_predictor = RNDPredictorNetwork(hidden_dim=rnd_hidden_dim, output_dim=rnd_output_dim)

        def create_agent(rng):
            init_x = (
                jnp.zeros((1, 1, wrapped_env.obs_size)),
                jnp.zeros((1, 1)),
            )
            init_hs = ScannedRNN.initialize_carry(config["HIDDEN_SIZE"], 1)
            agent_params = network.init(rng, init_hs, *init_x)

            rng, _rng = jax.random.split(rng)
            init_x = jnp.zeros((len(env.agents), 1, 1))
            init_state = jnp.zeros((1, 1, state_size))
            mixer_params = mixer.init(_rng, init_x, init_state)

            network_params = {'agent': agent_params, 'mixer': mixer_params}

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

        def create_rnd(rng):
            rng, rng_target, rng_pred = jax.random.split(rng, 3)
            
            dummy_state = jnp.zeros((1, state_size))
            target_params = rnd_target.init(rng_target, dummy_state)
            predictor_params = rnd_predictor.init(rng_pred, dummy_state)
            
            rnd_tx = optax.adam(learning_rate=rnd_lr)
            rnd_train_state = RNDTrainState.create(
                apply_fn=rnd_predictor.apply,
                params=predictor_params,
                tx=rnd_tx,
            )
            
            obs_rms = RunningMeanStd.init(state_size)
            reward_rms = RunningMeanStd.init(())
            
            return target_params, rnd_train_state, obs_rms, reward_rms

        rng, _rng = jax.random.split(rng)
        rnd_target_params, rnd_train_state, obs_rms, reward_rms = create_rnd(_rng)

        buffer = fbx.make_trajectory_buffer(
            max_length_time_axis=config["BUFFER_SIZE"] // config["NUM_ENVS"],
            min_length_time_axis=config["BUFFER_BATCH_SIZE"],
            sample_batch_size=config["BUFFER_BATCH_SIZE"],
            add_batch_size=config["NUM_ENVS"],
            sample_sequence_length=1,
            period=1,
        )
        buffer_state = buffer.init(sample_traj_unbatched)

        def compute_intrinsic_reward(rnd_target_params, rnd_predictor_params, next_states, obs_rms):
            
            target_features = rnd_target.apply(rnd_target_params, next_states)
            predicted_features = rnd_predictor.apply(rnd_predictor_params, next_states)
            
            intrinsic_reward = jnp.sum(jnp.square(predicted_features - target_features), axis=-1)
            
            return intrinsic_reward

        def _update_step(runner_state, unused):

            train_state, rnd_train_state, rnd_target_params, obs_rms, reward_rms, buffer_state, test_state, rng = runner_state

            # SAMPLE PHASE
            def _step_env(carry, _):
                hs, last_obs, last_dones, env_state, rng = carry
                rng, rng_a, rng_s = jax.random.split(rng, 3)

                _obs = batchify(last_obs)[:, np.newaxis]
                _dones = batchify(last_dones)[:, np.newaxis]

                new_hs, q_vals = jax.vmap(network.apply, in_axes=(None, 0, 0, 0))(
                    train_state.params['agent'], hs, _obs, _dones,
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
                
                timestep = Timestep(
                    obs=last_obs,
                    actions=actions,
                    rewards=jax.tree.map(lambda x: config.get("REW_SCALE", 1) * x, rewards),
                    dones=last_dones,
                    avail_actions=avail_actions,
                )
                return (new_hs, new_obs, dones, new_env_state, rng), (timestep, infos)

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
                _step_env, (*expl_state, _rng), None, config["NUM_STEPS"],
            )

            train_state = train_state.replace(
                timesteps=train_state.timesteps + config["NUM_STEPS"] * config["NUM_ENVS"]
            )

            # Update observation running statistics for RND normalization
            # Use all states for normalization (both current and next states will be normalized)
            global_states = timesteps.obs["__all__"]  # (time_steps, num_envs, state_dim)
            flat_states = global_states.reshape(-1, state_size)
            obs_rms = RunningMeanStd.update(obs_rms, flat_states)

            # BUFFER UPDATE
            buffer_traj_batch = jax.tree.map(
                lambda x: jnp.swapaxes(x, 0, 1)[:, np.newaxis],
                timesteps,
            )
            buffer_state = buffer.add(buffer_state, buffer_traj_batch)

            # NETWORKS UPDATE
            def _learn_phase(carry, _):
                train_state, rnd_train_state, obs_rms, reward_rms, rng = carry
                rng, _rng = jax.random.split(rng)
                minibatch = buffer.sample(buffer_state, _rng).experience
                minibatch = jax.tree.map(
                    lambda x: jnp.swapaxes(x[:, 0], 0, 1),
                    minibatch,
                )

                init_hs = ScannedRNN.initialize_carry(
                    config["HIDDEN_SIZE"], len(env.agents), config["BUFFER_BATCH_SIZE"],
                )
                _obs = batchify(minibatch.obs)
                _dones = batchify(minibatch.dones)
                _actions = batchify(minibatch.actions)
                _avail_actions = batchify(minibatch.avail_actions)

                _, q_next_target = jax.vmap(network.apply, in_axes=(None, 0, 0, 0))(
                    train_state.target_network_params['agent'], init_hs, _obs, _dones,
                )

                # Compute intrinsic rewards on NEXT states (s_{t+1})
                # Following Algorithm 1, line 15: r_int,t uses s_{t+1}
                # We need to shift states to get next states
                global_states = minibatch.obs["__all__"]  # (time_steps, batch_size, state_dim)
                # next_states[t] = states[t+1], pad last with zeros (terminal)
                next_states = jnp.concatenate([
                    global_states[1:],  # s_1, s_2, ..., s_T
                    jnp.zeros_like(global_states[:1])  # padding for terminal
                ], axis=0)

                flat_next_states = next_states.reshape(-1, state_size)
                intrinsic_rewards_flat = compute_intrinsic_reward(
                    rnd_target_params, rnd_train_state.params, flat_next_states, obs_rms
                )
                intrinsic_rewards = intrinsic_rewards_flat.reshape(global_states.shape[:-1])
                
                # Update intrinsic reward running stats and normalize
                # Only update stats on non-terminal transitions
                reward_rms = RunningMeanStd.update(reward_rms, intrinsic_rewards_flat)
                intrinsic_rewards_normalized = intrinsic_rewards / jnp.sqrt(reward_rms['var'] + 1e-8)
                
                # Mask out intrinsic rewards for terminal states (where done=True)
                # Since r_int,t is based on s_{t+1}, we zero it out when s_{t+1} is terminal/invalid
                terminal_mask = 1.0 - minibatch.dones["__all__"]
                intrinsic_rewards_normalized = intrinsic_rewards_normalized * terminal_mask

                # Combined rewards: r_total,t = r_t + β * r_int,t (Algorithm 1, line 16)
                intrinsic_reward_coef = rnd_scheduler(train_state.n_updates)
                combined_rewards = (
                    minibatch.rewards["__all__"] + 
                    intrinsic_reward_coef * intrinsic_rewards_normalized
                )

                def _loss_fn(params):
                    _, q_vals = jax.vmap(network.apply, in_axes=(None, 0, 0, 0))(
                        params['agent'], init_hs, _obs, _dones,
                    )

                    chosen_action_q_vals = jnp.take_along_axis(
                        q_vals, _actions[..., np.newaxis], axis=-1,
                    ).squeeze(-1)

                    unavailable_actions = 1 - _avail_actions
                    valid_q_vals = q_vals - (unavailable_actions * 1e10)

                    q_next = jnp.take_along_axis(
                        q_next_target,
                        jnp.argmax(valid_q_vals, axis=-1)[..., np.newaxis],
                        axis=-1,
                    ).squeeze(-1)

                    qmix_next = mixer.apply(
                        train_state.target_network_params['mixer'], 
                        q_next, minibatch.obs["__all__"]
                    )
                    
                    # Use combined rewards (extrinsic + intrinsic) for TD target
                    qmix_target = (
                        combined_rewards[:-1]
                        + (1 - minibatch.dones["__all__"][:-1])
                        * config["GAMMA"]
                        * qmix_next[1:]
                    )

                    qmix = mixer.apply(params['mixer'], chosen_action_q_vals, minibatch.obs["__all__"])[:-1]
                    loss = jnp.mean((qmix - jax.lax.stop_gradient(qmix_target)) ** 2)

                    return loss, (chosen_action_q_vals.mean(), qmix.mean(), intrinsic_rewards.mean())

                (loss, aux), grads = jax.value_and_grad(_loss_fn, has_aux=True)(train_state.params)
                qvals, qmix, int_rew = aux
                train_state = train_state.apply_gradients(grads=grads)
                train_state = train_state.replace(grad_steps=train_state.grad_steps + 1)

                def _rnd_loss_fn(predictor_params):
                    target_features = jax.lax.stop_gradient(
                        rnd_target.apply(rnd_target_params, next_states)
                    )
                    predicted_features = rnd_predictor.apply(predictor_params, next_states)
                    
                    per_sample_loss = jnp.sum(jnp.square(predicted_features - target_features), axis=-1)
                    
                    loss = jnp.sum(per_sample_loss)
                    return loss

                rnd_loss, rnd_grads = jax.value_and_grad(_rnd_loss_fn)(rnd_train_state.params)
                rnd_train_state = rnd_train_state.apply_gradients(grads=rnd_grads)

                return (train_state, rnd_train_state, obs_rms, reward_rms, rng), (loss, qvals, qmix, int_rew, rnd_loss)

            rng, _rng = jax.random.split(rng)
            is_learn_time = (
                buffer.can_sample(buffer_state)
            ) & (
                train_state.timesteps > config["LEARNING_STARTS"]
            )
            
            (train_state, rnd_train_state, obs_rms, reward_rms, rng), (loss, qvals, qmix, int_rew, rnd_loss) = jax.lax.cond(
                is_learn_time,
                lambda ts, rnd_ts, obs_r, rew_r, rng: jax.lax.scan(
                    _learn_phase, (ts, rnd_ts, obs_r, rew_r, rng), None, config["NUM_EPOCHS"]
                ),
                lambda ts, rnd_ts, obs_r, rew_r, rng: (
                    (ts, rnd_ts, obs_r, rew_r, rng),
                    (
                        jnp.zeros(config["NUM_EPOCHS"]),
                        jnp.zeros(config["NUM_EPOCHS"]),
                        jnp.zeros(config["NUM_EPOCHS"]),
                        jnp.zeros(config["NUM_EPOCHS"]),
                        jnp.zeros(config["NUM_EPOCHS"]),
                    ),
                ),
                train_state, rnd_train_state, obs_rms, reward_rms, _rng,
            )

            # Update target network
            train_state = jax.lax.cond(
                train_state.n_updates % config["TARGET_UPDATE_INTERVAL"] == 0,
                lambda ts: ts.replace(
                    target_network_params=optax.incremental_update(
                        ts.params, ts.target_network_params, config["TAU"],
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
                "q_tot_vals": qmix.mean(),
                "intrinsic_reward": int_rew.mean(),
                "rnd_loss": rnd_loss.mean(),
            }
            metrics.update(jax.tree.map(lambda x: x.mean(), infos))

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
                    if config.get('WANDB_LOG_ALL_SEEDS', False):
                        metrics.update({f"rng{int(original_seed)}/{k}": v for k, v in metrics.items()})
                    wandb.log(metrics)
                jax.debug.callback(callback, metrics, original_seed)

            runner_state = (train_state, rnd_train_state, rnd_target_params, obs_rms, reward_rms, buffer_state, test_state, rng)
            return runner_state, None

        def get_greedy_metrics(rng, train_state):
            if not config.get("TEST_DURING_TRAINING", True):
                return None
            
            params = train_state.params['agent']
            
            def _greedy_env_step(step_state, unused):
                params, env_state, last_obs, last_dones, hstate, rng = step_state
                rng, key_s = jax.random.split(rng)
                _obs = batchify(last_obs)[:, np.newaxis]
                _dones = batchify(last_dones)[:, np.newaxis]
                hstate, q_vals = jax.vmap(network.apply, in_axes=(None, 0, 0, 0))(
                    params, hstate, _obs, _dones,
                )
                q_vals = q_vals.squeeze(axis=1)
                valid_actions = test_env.get_valid_actions(env_state.env_state)
                actions = get_greedy_actions(q_vals, batchify(valid_actions))
                actions = unbatchify(actions)
                obs, env_state, rewards, dones, infos = test_env.batch_step(key_s, env_state, actions)
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
            step_state = (params, env_state, init_obs, init_dones, hstate, _rng)
            step_state, (rewards, dones, infos) = jax.lax.scan(
                _greedy_env_step, step_state, None, config["TEST_NUM_STEPS"]
            )
            metrics = jax.tree.map(
                lambda x: jnp.nanmean(jnp.where(infos["returned_episode"], x, jnp.nan)),
                infos,
            )
            return metrics

        rng, _rng = jax.random.split(rng)
        test_state = get_greedy_metrics(_rng, train_state)

        rng, _rng = jax.random.split(rng)
        runner_state = (train_state, rnd_train_state, rnd_target_params, obs_rms, reward_rms, buffer_state, test_state, _rng)

        runner_state, metrics = jax.lax.scan(
            _update_step, runner_state, None, config["NUM_UPDATES"]
        )

        return {"runner_state": runner_state, "metrics": metrics}

    return train


def env_from_config(config):
    env_name = config["ENV_NAME"]
    if "smax" in env_name.lower():
        config["ENV_KWARGS"]["scenario"] = map_name_to_scenario(config["MAP_NAME"])
        env_name = f"{config['ENV_NAME']}_{config['MAP_NAME']}"
        env = make(config["ENV_NAME"], **config["ENV_KWARGS"])
        env = SMAXLogWrapper(env)
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

    alg_name = config.get("ALG_NAME", "qmix_rnd")
    env, env_name = env_from_config(copy.deepcopy(config))

    wandb.init(
        entity=config["ENTITY"],
        project=config["PROJECT"],
        tags=[alg_name.upper(), env_name.upper(), f"jax_{jax.__version__}"],
        name=f"{alg_name}_{env_name}",
        config=config,
        mode=config["WANDB_MODE"],
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


@hydra.main(version_base=None, config_path="./config", config_name="config")
def main(config):
    config = OmegaConf.to_container(config)
    print("Config:\n", OmegaConf.to_yaml(config))
    single_run(config)


if __name__ == "__main__":
    main()