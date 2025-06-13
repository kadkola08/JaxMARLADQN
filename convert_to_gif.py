import jax
import jax.numpy as jnp
from jaxmarl.environments.smax import SMAX
from jaxmarl.viz.visualizer import SMAXVisualizer
from jaxmarl.environments.smax import map_name_to_scenario
from jaxmarl import make
from jaxmarl.wrappers.baselines import (
    SMAXLogWrapper,
    MPELogWrapper,
    LogWrapper,
    CTRolloutManager,
    )
import numpy as np
import pickle

if __name__ == "__main__":
    state_seq_dir = "exps/state_seqs"
    # maps = ["2s3z", "3s5z_vs_3s6z", "5m_vs_6m", "3s5z", "6h_vs_8z", "smacv2_5_units", "10m_vs_11m", "smacv2_10_units"]
    maps = ["2s3z", "3s5z_vs_3s6z", "5m_vs_6m", "3s5z", "6h_vs_8z", "smacv2_5_units", "10m_vs_11m", "smacv2_10_units"]
    map_name = "2s3z"
    env_name = "HeuristicEnemySMAX"
    alg_name = "vanqmix_rnn"
    # seeds = [10, 11, 12, 13, 14]
    seeds = [15, 16, 17, 18, 19]
    seed = 10
    vmap_index = 0

    for map_name in maps:
        for seed in seeds:
            state_list_path = f"{state_seq_dir}/smax_{map_name}_{alg_name}_{seed}_state_list.pkl"
            with open(state_list_path, "rb") as file:
                state_list = pickle.load(file)
            
            scenario = map_name_to_scenario(map_name)
            env_kwargs = {
                "scenario" : scenario,
                "walls_cause_death" : True,
                "see_enemy_actions" : True,
                "attack_mode" : "closest"
            }
            # _env = make(env_name, **env_kwargs)
            env = make(env_name, **env_kwargs)
            env = CTRolloutManager(env, batch_size=1)

            viz = SMAXVisualizer(env, state_list)  
            viz.animate(save_fname=f'exps/media/smax_{map_name}_{alg_name}_{seed}_animation_1.gif', view=True)
