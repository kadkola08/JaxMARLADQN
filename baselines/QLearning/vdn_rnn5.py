Cimport os
import copy
import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
from typing import NamedTuple, Dict, Union, Any, List

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

class RNNQNetwork(nn.Module):
    # Individual agent network, no parameter sharing
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

@chex.dataclass(frozen=True)
class Timestep:
    obs: dict
    actions: dict
    rewards: dict
    dones: dict
    avail_actions: dict

class MultiAgentTrainState(TrainState):
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

        networks = {}
        for agent in env.agents:
            networks[agent] = RNNQNetwork(
                action_dim=wrapped_env.max_action_space,
                hidden_dim=config["HIDDEN_SIZE"],
            )

        def create_agents(rng):
            agent_states = {}
            for i, agent in enumerate(env.agents):
                rng, _rng = jax.random.split(rng)

                init_x = (
                    jnp.zeros((1, 1, wrapped_env.obs_size)),  # (time_step, batch_size, obs_size)
                    jnp.zeros((1, 1)),  # (time_step, batch size)
                )

                init_hs = ScannedRNN.initialize_carry(config["HIDDEN_SIZE"], 1)  # (batch_size, hidden_dim)
                network_params = networks[agent].init(_rng, init_hs, *init_x)

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

                agent_states[agent] = MultiAgentTrainState.create(
                    apply_fn=networks[agent].apply,
                    params=network_params,
                    target_network_params=network_params,
                    tx=tx,
                )

            return agent_states

        rng, _rng = jax.random.split(rng)
        agent_train_states = create_agents(_rng)

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
        )

        buffer = fbx.make_trajectory_buffer(
            max_length_time_axis=config["BUFFER_SIZE"] // config["NUM_ENVS"],
            min_length_time_axis=config["BUFFER_BATCH_SIZE"],
            sample_batch_size=config["BUFFER_BATCH_SIZE"],
            add_batch_size=config["NUM_ENVS"],
            sample_sequence_length=1,
            period=1,
        )

        buffer_state = buffer.init(sample_traj_unbatched)

        def _update_step(runner_state, unused):
            agent_train_states, buffer_state, test_state, rng = runner_state

            def _step_env(carry, _):
                agent_hs, last_obs, last_dones, env_state, rng = carry
                rng, rng_a, rng_s = jax.random.split(rng, 3)

                q_vals = {}
                new_hs = {}

                for i, agent in enumerate(env.agents):
                    _obs = last_obs[agent][np.newaxis, :, :]
                    _dones = last_dones[agent][np.newaxis, :]

                    hs, q_val = networks[agent].apply(
                        agent_train_states[agent].params,
                        agent_hs[i],
                        _obs,
                        _dones,
                    )

                    new_hs[agent] = hs
                    q_vals[agent] = q_val.squeeze(0)
                
                new_hs = batchify(new_hs)
                q_vals = batchify(q_vals)

                avail_actions = wrapped_env.get_valid_actions(env_state.env_state)
                
                eps = eps_scheduler(agent_train_states[env.agents[0]].n_updates)
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
                    rewards=jax.tree.map(lambda x: config.get("REW_SCALE", 1)*x, rewards),
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

            init_hs = {
                agent: ScannedRNN.initialize_carry(config["HIDDEN_SIZE"], config["NUM_ENVS"])
                for agent in env.agents
            }
            init_hs = batchify(init_hs)

            expl_state = (init_hs, init_obs, init_dones, env_state)
            rng, _rng = jax.random.split(rng)

            _, (timesteps, infos) = jax.lax.scan(
                _step_env,
                (*expl_state, _rng),
                None,
                config["NUM_STEPS"],
            )

            for agent in env.agents:
                agent_train_states[agent] = agent_train_states[agent].replace(
                    timesteps=agent_train_states[agent].timesteps
                    + config["NUM_STEPS"] * config["NUM_ENVS"]
                )


            # BUFFER UPDATE
            buffer_traj_batch = jax.tree.map(
                lambda x: jnp.swapaxes(x, 0, 1)[
                    :, np.newaxis
                ],  # put the batch dim first and add a dummy sequence dim
                timesteps,
            )  # (num_envs, 1, time_steps, ...)
            buffer_state = buffer.add(buffer_state, buffer_traj_batch)

            def _learn_phase(carry, _):
                agent_train_states, rng = carry
                rng, _rng = jax.random.split(rng)
                minibatch = buffer.sample(buffer_state, _rng).experience
                minibatch = jax.tree.map(
                    lambda x: jnp.swapaxes(
                        x[:, 0], 0, 1
                    ),  # remove the dummy sequence dim (1) and swap batch and temporal dims
                    minibatch,
                )  # (max_time_steps, batch_size, ...

                q_next_target_dict = {}
                for agent_idx, agent in enumerate(env.agents):
                    init_hs = ScannedRNN.initialize_carry(
                        config["HIDDEN_SIZE"],
                        config["BUFFER_BATCH_SIZE"],
                    )
                    _obs = minibatch.obs[agent]# [np.newaxis, :, :]
                    _dones = minibatch.dones[agent]# [np.newaxis, :]

                    _, q_next = networks[agent].apply(
                        agent_train_states[agent].target_network_params,
                        init_hs,
                        _obs,
                        _dones,
                    )

                    q_next_target_dict[agent] = q_next # remove added dimension


                updated_agent_states = {}
                loss_values = {}
                qval_values = {}

                for agent_idx, agent in enumerate(env.agents):
                    init_hs = ScannedRNN.initialize_carry(
                        config["HIDDEN_SIZE"],
                        config["BUFFER_BATCH_SIZE"],
                    )

                    _obs = minibatch.obs[agent]# [:, :, np.newaxis]
                    _dones = minibatch.dones[agent]# [:, :, np.newaxis]
                    _actions = minibatch.actions[agent]
                    _avail_actions = minibatch.avail_actions[agent]

                    def _loss_fn(params):
                         ##############################
                        # init_hs = ScannedRNN.initialize_carry(
                        #     config["HIDDEN_SIZE"],
                        #     config["BUFFER_BATCH_SIZE"],
                        # )
                        # q_current = {}
                        # for other_agent in env.agents:
                        #     _obs_ = minibatch.obs[other_agent]
                        #     _dones_ = minibatch.dones[other_agent]
                        #     _actions_ = minibatch.actions[other_agent]
                        #     _avail_action_ = minibatch.avail_actions[other_agent] 
                        #     _, q_c = networks[other_agent].apply(
                        #         agent_train_states[other_agent].params,
                        #         init_hs,
                        #         _obs,
                        #         _dones,
                        #     )
                        #     breakpoint()
                        #     q_c = jnp.take_along_axis(
                        #         q_c,
                        #         _actions_[..., np.newaxis],
                        #         axis=-1,
                        #     )
                        #     breakpoint()
                        #     q_current[other_agent] = q_c
                        # q_current = batchify(q_current)
                        # breakpoint()
                        # chosen_action_q_vals = jnp.sum(q_current, axis=0)
                        ##############################
                        _, q_vals = networks[agent].apply(
                            params,
                            init_hs,
                            _obs,
                            _dones
                        )

                        q_vals = q_vals

                        chosen_action_q_vals = jnp.take_along_axis(
                            q_vals,
                            _actions[..., np.newaxis],
                            axis=-1,
                        )
                        breakpoint()

                        # unavailable_actions = 1 - _avail_actions
                        # valid_q_vals = q_vals - (unavailable_actions * 1e10)

                        # q_next = jnp.take_along_axis(
                        #     q_next_target_dict[agent],
                        #     jnp.argmax(valid_q_vals, axis=-1)[..., np.newaxis],
                        #     axis=-1,
                        # ).squeeze(-1)

                        next_max_q_vals = {}
                        for other_agent in env.agents:
                            agent_avail_actions = minibatch.avail_actions[other_agent]
                            
                            unavail_mask = 1 - agent_avail_actions
                            masked_q_vals = q_next_target_dict[other_agent] - (unavail_mask * 1e10)

                            max_q = jnp.max(masked_q_vals, axis=-1)
                            next_max_q_vals[other_agent] = max_q
                        
                        breakpoint()
                        q_next_sum = jnp.sum(batchify(next_max_q_vals), axis=0)

                        vdn_target = (
                            minibatch.rewards["__all__"][:-1] +
                            (1 - minibatch.dones["__all__"][:-1]) *
                            config["GAMMA"] * q_next_sum[1:]
                        )
                        breakpoint()

                        loss = jnp.mean(
                            (chosen_action_q_vals[:-1].squeeze(-1) - jax.lax.stop_gradient(vdn_target)) ** 2
                        )

                        return loss, chosen_action_q_vals.mean()

                    (loss, qvals), grads = jax.value_and_grad(_loss_fn, has_aux=True)(
                        agent_train_states[agent].params
                    )

                    updated_state = agent_train_states[agent].apply_gradients(grads=grads)
                    updated_state = updated_state.replace(grad_steps=updated_state.grad_steps + 1,)

                    updated_agent_states[agent] = updated_state
                    loss_values[agent] = loss
                    qval_values[agent] = qvals

                avg_loss = jnp.mean(batchify(loss_values))
                avg_qvals = jnp.mean(batchify(qval_values))

                return (updated_agent_states, rng), (avg_loss, avg_qvals)

            rng, _rng = jax.random.split(rng)
            is_learn_time = (
                buffer.can_sample(buffer_state)
            ) & (  # enough experience in buffer
                agent_train_states[env.agents[0]].timesteps > config["LEARNING_STARTS"]
            )
            (agent_train_states, rng), (loss, qvals) = jax.lax.cond(
                is_learn_time,
                lambda agent_train_states, rng: jax.lax.scan(
                    _learn_phase, (agent_train_states, rng), None, config["NUM_EPOCHS"]
                ),
                lambda agent_train_states, rng: (
                    (agent_train_states, rng),
                    (
                        jnp.zeros(config["NUM_EPOCHS"]),
                        jnp.zeros(config["NUM_EPOCHS"]),
                    ),
                ),  # do nothing
                agent_train_states,
                _rng,
            )

            for agent in env.agents:
                agent_train_states[agent] = jax.lax.cond(
                    agent_train_states[agent].n_updates % config["TARGET_UPDATE_INTERVAL"] == 0,
                    lambda state: state.replace(
                        target_network_params=optax.incremental_update(
                            state.params,
                            state.target_network_params,
                            config["TAU"],
                        )
                    ),
                    lambda state: state,
                    operand=agent_train_states[agent],
                )

            for agent in env.agents:
                agent_train_states[agent] = agent_train_states[agent].replace(
                    n_updates=agent_train_states[agent].n_updates + 1
                )

            metrics = {
                "env_step": agent_train_states[env.agents[0]].timesteps,
                "update_steps": agent_train_states[env.agents[0]].n_updates,
                "grad_steps": agent_train_states[env.agents[0]].grad_steps,
                "loss": loss.mean(),
                "qvals": qvals.mean(),
            }
            metrics.update(jax.tree.map(lambda x: x.mean(), infos))

            if config.get("TEST_DURING_TRAINING", True):
                rng, _rng = jax.random.split(rng)
                test_state = jax.lax.cond(
                    agent_train_states[env.agents[0]].n_updates
                    % int(config["NUM_UPDATES"] * config["TEST_INTERVAL"])
                    == 0,
                    lambda _: get_greedy_metrics(_rng, agent_train_states),
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

            runner_state = (agent_train_states, buffer_state, test_state, rng)

            return runner_state, None

        def get_greedy_metrics(rng, agent_train_states):
            """Help function to test greedy policy during training"""
            if not config.get("TEST_DURING_TRAINING", True):
                return None

            def _greedy_env_step(step_state, unused):
                agent_train_states, env_state, last_obs, last_dones, agent_hstates, rng = step_state
                rng, key_s = jax.random.split(rng)

                q_vals = {}
                new_hs = {}

                for i, agent in enumerate(env.agents):
                    _obs = last_obs[agent][np.newaxis, :, :]
                    _dones = last_dones[agent][np.newaxis, : ]

                    hs, q_val = networks[agent].apply(
                        agent_train_states[agent].params,
                        agent_hstates[i],
                        _obs,
                        _dones,
                    )

                    new_hs[agent] = hs
                    q_vals[agent] = q_val.squeeze(0)

                new_agent_hstates = batchify(new_hs)
                q_vals = batchify(q_vals)

                valid_actions = test_env.get_valid_actions(env_state.env_state)
                valid_actions_stacked = batchify(valid_actions)

                actions = get_greedy_actions(q_vals, valid_actions_stacked)
                actions = unbatchify(actions)

                obs, env_state, rewards, dones, infos = test_env.batch_step(
                    key_s, env_state, actions
                )

                step_state = (agent_train_states, env_state, obs, dones, new_agent_hstates, rng)
                return step_state, (rewards, dones, infos)

            rng, _rng = jax.random.split(rng)
            init_obs, env_state = test_env.batch_reset(_rng)
            init_dones = {
                agent: jnp.zeros((config["TEST_NUM_ENVS"]), dtype=bool)
                for agent in env.agents + ["__all__"]
            }

            init_hs = {
                agent : ScannedRNN.initialize_carry(config["HIDDEN_SIZE"], config["TEST_NUM_ENVS"])
                for agent in env.agents
            }
            init_hs = batchify(init_hs)

            rng, _rng = jax.random.split(rng)
            step_state = (
                agent_train_states,
                env_state,
                init_obs,
                init_dones,
                init_hs,
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
                        jnp.nan
                    )
                ),
                infos,
            )

            return metrics

        rng, _rng = jax.random.split(rng)
        test_state = get_greedy_metrics(_rng, agent_train_states)

        rng, _rng = jax.random.split(rng)
        runner_state = (agent_train_states, buffer_state, test_state, _rng)

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

    alg_name = config.get("ALG_NAME", "ind_vdn_rnn")  # Changed to indicate independent networks
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

        model_states = outs["runner_state"][0]
        save_dir = os.path.join(config["SAVE_PATH"], env_name)
        os.makedirs(save_dir, exist_ok=True)
        OmegaConf.save(
            config,
            os.path.join(
                save_dir, f'{alg_name}_{env_name}_seed{config["SEED"]}_config.yaml'
            ),
        )

        for i, rng in enumerate(rngs):
            # Save each agent's parameters separately
            for agent in env.agents:
                agent_params = jax.tree.map(lambda x: x[i][agent], model_states)
                save_path = os.path.join(
                    save_dir,
                    f'{alg_name}_{env_name}_{agent}_seed{config["SEED"]}_vmap{i}.safetensors',
                )
                save_params(agent_params.params, save_path)


def tune(default_config):
    """Hyperparameter sweep with wandb."""

    default_config = {**default_config, **default_config["alg"]}  # merge the alg config with the main config
    env_name = default_config["ENV_NAME"]
    alg_name = default_config.get("ALG_NAME", "ind_vdn_rnn")  # Changed to indicate independent networks
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
