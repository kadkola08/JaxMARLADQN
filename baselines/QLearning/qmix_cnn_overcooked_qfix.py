"""
Specific to this implementation: CNN network and Reward Shaping Annealing as per Overcooked paper.
"""

import copy
import os
from typing import Any

import chex
import flashbax as fbx
import flax.linen as nn
import hydra
import jax
import jax.core
import jax.numpy as jnp
import numpy as np
import optax
from flax.training.train_state import TrainState
from omegaconf import OmegaConf

import wandb
from jaxmarl import make
from jaxmarl.environments.overcooked import overcooked_layouts
from jaxmarl.environments.smax import map_name_to_scenario
from jaxmarl.wrappers.baselines import (CTRolloutManager, LogWrapper,
                                        MPELogWrapper, SMAXLogWrapper)

from flax.linen.initializers import orthogonal, constant


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


class CNNOvercooked(nn.Module):
    """CNN module for overcooked"""

    activation: str = "relu"

    @nn.compact
    def __call__(self, x):
        # x.shape == (*B, H, W, C)
        activation = nn.relu if self.activation == "relu" else nn.tanh

        x = nn.Conv(
            features=32,
            kernel_size=(5, 5),
        )(x)
        x = activation(x)

        x = nn.Conv(
            features=32,
            kernel_size=(3, 3),
        )(x)
        x = activation(x)

        # x = nn.Conv(
        #     features=32,
        #     kernel_size=(3, 3),
        # )(x)
        # x = activation(x)

        x = x.reshape(*x.shape[:-3], -1)  # Flatten

        # x = nn.Dense(features=64)(x)
        # x = activation(x)

        return x


class QMIX_Overcooked(nn.Module):
    """
    Mixing network for projecting individual utilities into joint qvalues.
    Follows the original QMix implementation.
    """

    embedding_dim: int
    hypernet_hidden_dim: int
    init_scale: float

    state_module: nn.Module | None = None

    @nn.compact
    def __call__(self, individual_qvalues: jax.Array, states):
        individual_qvalues= jnp.expand_dims(individual_qvalues, axis=1)
        states = jnp.expand_dims(states, axis=0)

        # individual_qvalues.shape == (N, T, B)
        N, T, B = individual_qvalues.shape

        if self.state_module is not None:
            states = self.state_module(states)

        w1 = MLP(
            features=[N * self.embedding_dim],
            # features=[self.hypernet_hidden_dim, N * self.embedding_dim],
            # hidden_dim=self.hypernet_hidden_dim,
            # output_dim=N * self.embedding_dim,
            init_scale=self.init_scale,
        )(states)
        # w1.shape == (T, B, N*D)
        w1 = w1.reshape(T, B, N, self.embedding_dim)
        # w1.shape == (T, B, N, D)
        w1 = jnp.abs(w1)

        b1 = MLP(
            # hidden_dim=self.embedding_dim,
            # # output_dim=self.embedding_dim,
            # output_dim=1,
            # features=[self.embedding_dim, self.embedding_dim],
            # features=[self.embedding_dim, 1],
            features=[1],
            init_scale=self.init_scale,
        )(states)

        return jnp.einsum("ntb,tbnd->tb", individual_qvalues, w1) + b1.squeeze(-1)


class CNN(nn.Module):
    activation: str = "relu"

    @nn.compact
    def __call__(self, x):
        # x.shape == (B, H, W, C)
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh
        x = nn.Conv(
            features=32,
            kernel_size=(5, 5),
        )(x)
        x = activation(x)
        x = nn.Conv(
            features=32,
            kernel_size=(3, 3),
        )(x)
        x = activation(x)
        x = nn.Conv(
            features=32,
            kernel_size=(3, 3),
        )(x)
        x = activation(x)
        x = x.reshape((x.shape[0], -1))  # Flatten

        x = nn.Dense(features=64)(x)
        x = activation(x)

        return x


class QNetwork(nn.Module):
    action_dim: int
    hidden_size: int = 64
    debug_normalize_qi: bool = False

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        # TODO x.shape == ()
        embedding = CNN()(x)
        x = nn.Dense(self.hidden_size)(embedding)
        x = nn.relu(x)
        x = nn.Dense(self.action_dim)(x)

        if self.debug_normalize_qi:
            xmax = jax.scipy.special.logsumexp(x, axis=-1, keepdims=True)
            xmin = -jax.scipy.special.logsumexp(-x, axis=-1, keepdims=True)
            x = (x - xmin) / (xmax - xmin)

        return x


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
        config["EPS_START"],
        config["EPS_FINISH"],
        config["EPS_DECAY"] * config["NUM_UPDATES"],
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

        # INIT NETWORK AND OPTIMIZER
        network = QNetwork(
            action_dim=wrapped_env.max_action_space,
            hidden_size=config["HIDDEN_SIZE"],
            debug_normalize_qi=config.get("DEBUG_NORMALIZE_QI", False),
        )

        state_module = CNNOvercooked()
        fixee = QMIX_Overcooked(
            config['MIXER_EMBEDDING_DIM'],
            config['MIXER_HYPERNET_HIDDEN_DIM'],
            config['MIXER_INIT_SCALE'],
            state_module=state_module
        )

        def create_agent(rng):
            init_x = jnp.zeros((1, *env.observation_space().shape))
            agent_params = network.init(rng, init_x)

            # init mixer
            rng, _rng = jax.random.split(rng)
            state_shape = env.observation_space().shape
            init_state = jnp.zeros((1,) + state_shape)
            init_qvalues = jnp.zeros((len(env.agents), 1))
            fixee_params = fixee.init(
                _rng,
                init_qvalues,
                init_state
            )
            network_params = {"agent" : agent_params, "fixee" : fixee_params}

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

        # INIT BUFFER
        buffer = fbx.make_flat_buffer(
            max_length=int(config["BUFFER_SIZE"]),
            min_length=int(config["BUFFER_BATCH_SIZE"]),
            sample_batch_size=int(config["BUFFER_BATCH_SIZE"]),
            add_sequences=False,
            add_batch_size=int(config["NUM_ENVS"] * config["NUM_STEPS"]),
        )
        buffer = buffer.replace(
            init=jax.jit(buffer.init),
            add=jax.jit(buffer.add, donate_argnums=0),
            sample=jax.jit(buffer.sample),
            can_sample=jax.jit(buffer.can_sample),
        )

        _obs, _env_state = wrapped_env.batch_reset(_rng)
        _actions = {
            agent: wrapped_env.batch_sample(_rng, agent) for agent in env.agents
        }
        _obs, _, _rewards, _dones, _infos = wrapped_env.batch_step(
            _rng, _env_state, _actions
        )
        _avail_actions = wrapped_env.get_valid_actions(_env_state.env_state)
        _timestep = Timestep(
            obs=_obs,
            actions=_actions,
            avail_actions=_avail_actions,
            rewards=_rewards,
            dones=_dones,
        )
        _timestep_unbatched = jax.tree.map(
            lambda x: x[0], _timestep
        )  # remove the NUM_ENV dim
        buffer_state = buffer.init(_timestep_unbatched)


        # TRAINING LOOP
        def _update_step(runner_state, unused):
            train_state, buffer_state, expl_state, test_state, rng = runner_state

            # SAMPLE PHASE
            def _step_env(carry, _):
                last_obs, env_state, rng = carry
                rng, rng_a, rng_s = jax.random.split(rng, 3)
                q_vals = jax.vmap(network.apply, in_axes=(None, 0))(
                    train_state.params["agent"],
                    batchify(last_obs),  # (num_agents, num_envs, num_actions)
                )  # (num_agents, num_envs, num_actions)

                # explore
                avail_actions = wrapped_env.get_valid_actions(env_state.env_state)

                eps = eps_scheduler(train_state.n_updates)
                _rngs = jax.random.split(rng_a, env.num_agents)
                new_action = jax.vmap(eps_greedy_exploration, in_axes=(0, 0, None, 0))(
                    _rngs, q_vals, eps, batchify(avail_actions)
                )
                actions = unbatchify(new_action)

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
                    avail_actions=avail_actions,
                    rewards=rewards,
                    dones=dones,
                )
                return (new_obs, new_env_state, rng), (timestep, infos)

            # step the env
            rng, _rng = jax.random.split(rng)
            carry, (timesteps, infos) = jax.lax.scan(
                _step_env,
                (*expl_state, _rng),
                None,
                config["NUM_STEPS"],
            )
            expl_state = carry[:2]

            train_state = train_state.replace(
                timesteps=train_state.timesteps
                + config["NUM_STEPS"] * config["NUM_ENVS"]
            )  # update timesteps count

            # BUFFER UPDATE
            timesteps = jax.tree.map(
                lambda x: x.reshape(-1, *x.shape[2:]), timesteps
            )  # (num_envs*num_steps, ...)
            buffer_state = buffer.add(buffer_state, timesteps)

            # NETWORKS UPDATE
            def _learn_phase(carry, _):
                train_state, rng = carry
                rng, _rng = jax.random.split(rng)
                minibatch = buffer.sample(buffer_state, _rng).experience

                target_individual_qvalues = jax.vmap(network.apply, in_axes=(None, 0))(
                    train_state.target_network_params["agent"],
                    batchify(minibatch.second.obs),
                )

                # target_individual_qvalues.shape == (N, B, A)
                assert isinstance(target_individual_qvalues, jax.Array)
                target_individual_vvalues = target_individual_qvalues.max(axis=-1)
                # target_individual_vvalues.shape == (N, B)

                # NOTE: no need to use stop_gradient on the target here:
                # the target qvalues are computed using the target model/parameters only

                actions = batchify(minibatch.first.actions)
                # actions.shape == (N, B)
                observations = batchify(minibatch.first.obs)
                # actions.shape == (N, B, DO)

                def _loss_fn(params):
                    individual_qvalues = jax.vmap(network.apply, in_axes=(None, 0))(
                        params["agent"],
                        observations,
                    )
                    # individual_qvlues.shape == (N, B, A)
                    assert isinstance(individual_qvalues, jax.Array)

                    chosen_qvalues = jnp.take_along_axis(
                        individual_qvalues,
                        actions[..., jnp.newaxis],
                        axis=-1,
                    ).squeeze(-1)
                    # chosen_qvalues.shape == (N, B)

                    # state extractin is heavility environment dependent
                    # so better to make sure everything is as expected
                    # check_observations(minibatch.second.obs)
                    # NOTE: NVM, it's apparently really hard to check on concrete values

                    target_states = minibatch.second.obs["agent_0"]

                    target_qvalues = fixee.apply(
                        train_state.target_network_params['fixee'],
                        target_individual_vvalues,
                        target_states
                    )
                    target_qvalues = target_qvalues.squeeze(0)
                    # target_vvalues.shape == (B,)

                    target_qvalues = (
                        minibatch.first.rewards["__all__"]
                        + (1 - minibatch.first.dones["__all__"])
                        * config["GAMMA"]
                        * target_qvalues
                    )
                    # target_qvalues == (B,)

                    # each agent already sees the whole state
                    # plus this info has structure
                    # states = minibatch.first.obs["__all__"]
                    states = minibatch.first.obs["agent_0"]

                    qfix_qvalues = fixee.apply(
                        params["fixee"],
                        chosen_qvalues,
                        states
                    )
                    qfix_qvalues = qfix_qvalues.squeeze(0)

                    target_qvalues = jax.lax.stop_gradient(target_qvalues)
                    loss = jnp.mean((qfix_qvalues - target_qvalues) ** 2)

                    aux = chosen_qvalues # , w.mean(-1), b
                    return loss, aux

                (loss, aux), grads = jax.value_and_grad(_loss_fn, has_aux=True)(
                    train_state.params
                )
                train_state = train_state.apply_gradients(grads=grads)
                train_state = train_state.replace(
                    grad_steps=train_state.grad_steps + 1,
                )
                return (train_state, rng), (loss, aux)

            rng, _rng = jax.random.split(rng)
            is_learn_time = (
                buffer.can_sample(buffer_state)
            ) & (  # enough experience in buffer
                train_state.timesteps > config["LEARNING_STARTS"]
            )
            jnp.stack
            (train_state, rng), (loss, aux) = jax.lax.cond(
                is_learn_time,
                lambda train_state, rng: jax.lax.scan(
                    _learn_phase, (train_state, rng), None, config["NUM_EPOCHS"]
                ),
                lambda train_state, rng: (
                    (train_state, rng),
                    (
                        jnp.zeros(config["NUM_EPOCHS"]),
                        jnp.zeros(
                            (
                                config["NUM_EPOCHS"],
                                env.num_agents,
                                config["BUFFER_BATCH_SIZE"],
                            )
                        ),
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
            qvals = aux
            metrics_to_histogram = {"qvals": qvals,} #  "ws": ws, "bs": bs}
            metrics = {
                "env_step": train_state.timesteps,
                "update_steps": train_state.n_updates,
                "grad_steps": train_state.grad_steps,
                "loss": loss.mean(),
                "qvals": qvals.mean(),
                # "ws": ws.mean(),
                # "bs": bs.mean(),
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

                def callback(metrics, metrics_to_histogram, original_seed):
                    metrics.update(
                        {
                            f"histograms/{k}": wandb.Histogram(np.array(v).ravel())
                            for k, v in metrics_to_histogram.items()
                        }
                    )
                    if config.get("WANDB_LOG_ALL_SEEDS", False):
                        metrics.update(
                            {
                                f"rng{int(original_seed)}/{k}": v
                                for k, v in metrics.items()
                            }
                        )
                    wandb.log(metrics, step=metrics["update_steps"])

                jax.debug.callback(
                    callback, metrics, metrics_to_histogram, original_seed
                )

            runner_state = (train_state, buffer_state, expl_state, test_state, rng)

            return runner_state, None

        def get_greedy_metrics(rng, train_state):
            if not config.get("TEST_DURING_TRAINING", True):
                return None
            """Help function to test greedy policy during training"""

            def _greedy_env_step(step_state, unused):
                last_obs, env_state, rng = step_state
                rng, rng_a, rng_s = jax.random.split(rng, 3)
                q_vals = jax.vmap(network.apply, in_axes=(None, 0))(
                    train_state.params["agent"],
                    batchify(last_obs),  # (num_agents, num_envs, num_actions)
                )  # (num_agents, num_envs, num_actions)
                actions = jnp.argmax(q_vals, axis=-1)
                actions = unbatchify(actions)
                new_obs, new_env_state, rewards, dones, infos = test_env.batch_step(
                    rng_s, env_state, actions
                )
                step_state = (new_obs, new_env_state, rng)
                return step_state, (rewards, dones, infos)

            rng, _rng = jax.random.split(rng)
            init_obs, env_state = test_env.batch_reset(_rng)
            rng, _rng = jax.random.split(rng)
            step_state, (rewards, dones, infos) = jax.lax.scan(
                _greedy_env_step,
                (init_obs, env_state, _rng),
                None,
                config["TEST_NUM_STEPS"],
            )
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
        obs, env_state = wrapped_env.batch_reset(_rng)
        expl_state = (obs, env_state)

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

    alg_name = config["ALG_NAME"]
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

    default_config = {
        **default_config,
        **default_config["alg"],
    }  # merge the alg config with the main config
    env_name = default_config["ENV_NAME"]
    alg_name = config["ALG_NAME"]
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
