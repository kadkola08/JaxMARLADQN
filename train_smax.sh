device="$1"
# CUDA_VISIBLE_DEVICES=$device python baselines/QLearning/adqn_rnn32.py +alg=ql_rnn_smax alg.LR=0.0001 ++PROJECT=ADQN_test alg.MAP_NAME=smacv2_5_units++SEED=1 alg.LR=0.00008
# CUDA_VISIBLE_DEVICES=$device python baselines/QLearning/adqn_rnn32.py +alg=ql_rnn_smax alg.LR=0.0001 ++PROJECT=ADQN_test alg.MAP_NAME=smacv2_5_units ++SEED=2 alg.LR=0.00008
# CUDA_VISIBLE_DEVICES=$device python baselines/QLearning/adqn_rnn32.py +alg=ql_rnn_smax alg.LR=0.0001 ++PROJECT=ADQN_test alg.MAP_NAME=smacv2_5_units ++SEED=3 alg.LR=0.00008
# CUDA_VISIBLE_DEVICES=$device python baselines/QLearning/adqn_rnn32.py +alg=ql_rnn_smax alg.LR=0.0001 ++PROJECT=ADQN_test alg.MAP_NAME=smacv2_5_units ++SEED=4 alg.LR=0.00008

# CUDA_VISIBLE_DEVICES=0 python baselines/QLearning/adqn_rnn322.py +alg=ql_rnn_smax alg.MIXER_EMBEDDING_DIM=512 alg.LR=0.0005 ++PROJECT=ADQN_test alg.MAP_NAME=6h_vs_8z ++SEED=0
# CUDA_VISIBLE_DEVICES=0 python baselines/QLearning/adqn_rnn322.py +alg=ql_rnn_smax alg.MIXER_EMBEDDING_DIM=512 alg.LR=0.0005 ++PROJECT=ADQN_test alg.MAP_NAME=6h_vs_8z ++SEED=1
# CUDA_VISIBLE_DEVICES=0 python baselines/QLearning/adqn_rnn322.py +alg=ql_rnn_smax alg.MIXER_EMBEDDING_DIM=512 alg.LR=0.0005 ++PROJECT=ADQN_test alg.MAP_NAME=6h_vs_8z ++SEED=2
# CUDA_VISIBLE_DEVICES=0 python baselines/QLearning/adqn_rnn322.py +alg=ql_rnn_smax alg.MIXER_EMBEDDING_DIM=512 alg.LR=0.0005 ++PROJECT=ADQN_test alg.MAP_NAME=6h_vs_8z ++SEED=3
# CUDA_VISIBLE_DEVICES=0 python baselines/QLearning/adqn_rnn322.py +alg=ql_rnn_smax alg.MIXER_EMBEDDING_DIM=512 alg.LR=0.0005 ++PROJECT=ADQN_test alg.MAP_NAME=6h_vs_8z ++SEED=4

python baselines/QLearning/qplex_rnn2.py +alg=ql_rnn_smax ++SEED=1 alg.MAP_NAME=2s3z alg.NUM_STEPS=100 alg.NUM_ENVS=8
python baselines/QLearning/qplex_rnn2.py +alg=ql_rnn_smax ++SEED=1 alg.MAP_NAME=3s_vs_5z alg.NUM_STEPS=100 alg.NUM_ENVS=8
python baselines/QLearning/qplex_rnn2.py +alg=ql_rnn_smax ++SEED=1 alg.MAP_NAME=3s5z alg.NUM_STEPS=100 alg.NUM_ENVS=8
python baselines/QLearning/qplex_rnn2.py +alg=ql_rnn_smax ++SEED=1 alg.MAP_NAME=3s5z_vs_3s6z alg.NUM_STEPS=100 alg.NUM_ENVS=8
python baselines/QLearning/qplex_rnn2.py +alg=ql_rnn_smax ++SEED=1 alg.MAP_NAME=5m_vs_6m alg.NUM_STEPS=100 alg.NUM_ENVS=8
python baselines/QLearning/qplex_rnn2.py +alg=ql_rnn_smax ++SEED=1 alg.MAP_NAME=6h_vs_8z alg.NUM_STEPS=100 alg.NUM_ENVS=8
python baselines/QLearning/qplex_rnn2.py +alg=ql_rnn_smax ++SEED=1 alg.MAP_NAME=smacv2_5_units alg.NUM_STEPS=100 alg.NUM_ENVS=8
python baselines/QLearning/qplex_rnn2.py +alg=ql_rnn_smax ++SEED=1 alg.MAP_NAME=smacv2_10_units alg.NUM_STEPS=100 alg.NUM_ENVS=8