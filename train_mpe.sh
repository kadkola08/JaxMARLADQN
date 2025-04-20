#!/bin/bash

# python baselines/QLearning/vdn_rnn.py +alg=ql_rnn_mpe alg.ENV_NAME=MPE_simple_spread_v3 # +alg.ENV_KWARGS.num_agents=5 
# python baselines/QLearning/qmix_rnn.py +alg=ql_rnn_mpe alg.ENV_NAME=MPE_simple_spread_v3 # +alg.ENV_KWARGS.num_agents=5
# python baselines/QLearning/qmix_rnn2.py +alg=ql_rnn_mpe alg.ENV_NAME=MPE_simple_spread_v3 # +alg.ENV_KWARGS.num_agents=5
# python baselines/QLearning/qmix_rnn22.py +alg=ql_rnn_mpe alg.ENV_NAME=MPE_simple_spread_v3 # +alg.ENV_KWARGS.num_agents=5


# python baselines/QLearning/vdn_rnn.py +alg=ql_rnn_mpe alg.ENV_NAME=MPE_simple_spread_v3 +alg.ENV_KWARGS.num_agents=5 +alg.ENV_KWARGS.num_landmarks=5
# python baselines/QLearning/qmix_rnn.py +alg=ql_rnn_mpe alg.ENV_NAME=MPE_simple_spread_v3 +alg.ENV_KWARGS.num_agents=5 +alg.ENV_KWARGS.num_landmarks=5
# python baselines/QLearning/qmix_rnn2.py +alg=ql_rnn_mpe alg.ENV_NAME=MPE_simple_spread_v3 +alg.ENV_KWARGS.num_agents=5 +alg.ENV_KWARGS.num_landmarks=5

# python baselines/QLearning/vdn_rnn.py +alg=ql_rnn_mpe alg.ENV_NAME=MPE_simple_speaker_listener_v4
# python baselines/QLearning/qmix_rnn.py +alg=ql_rnn_mpe alg.ENV_NAME=MPE_simple_speaker_listener_v4
# python baselines/QLearning/qmix_rnn2.py +alg=ql_rnn_mpe alg.ENV_NAME=MPE_simple_speaker_listener_v4

python baselines/QLearning/vdn_rnn.py +alg=ql_rnn_mpe alg.ENV_NAME=MPE_simple_reference_v3 # +alg.ENV_KWARGS.num_agents=5 
python baselines/QLearning/qmix_rnn.py +alg=ql_rnn_mpe alg.ENV_NAME=MPE_simple_reference_v3 # +alg.ENV_KWARGS.num_agents=5
python baselines/QLearning/qmix_rnn2.py +alg=ql_rnn_mpe alg.ENV_NAME=MPE_simple_reference_v3 # +alg.ENV_KWARGS.num_agents=5
python baselines/QLearning/qmix_rnn22.py +alg=ql_rnn_mpe alg.ENV_NAME=MPE_simple_reference_v3 # +alg.ENV_KWARGS.num_agents=5
