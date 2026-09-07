# CUDA_VISIBLE_DEVICES=1 python baselines/QLearning/adqn_rnn322.py +alg=ql_rnn_smax alg.HIDDEN_SIZE=128 alg.LR=0.0002 ++PROJECT=ADQN_test alg.MAP_NAME=5m_vs_6m ++SEED=1
# CUDA_VISIBLE_DEVICES=1 python baselines/QLearning/adqn_rnn322.py +alg=ql_rnn_smax alg.HIDDEN_SIZE=128 alg.LR=0.0002 ++PROJECT=ADQN_test alg.MAP_NAME=6h_vs_8z ++SEED=1
# CUDA_VISIBLE_DEVICES=1 python baselines/QLearning/adqn_rnn322.py +alg=ql_rnn_smax alg.HIDDEN_SIZE=128 alg.LR=0.0002 ++PROJECT=ADQN_test alg.MAP_NAME=3s5z ++SEED=1

CUDA_VISIBLE_DEVICES=1 python baselines/QLearning/adqn_rnn322.py +alg=ql_rnn_smax alg.MIXER_EMBEDDING_DIM=512 alg.LR=0.001 alg.LR_LINEAR_DECAY=True ++PROJECT=ADQN_test alg.MAP_NAME=6h_vs_8z ++SEED=0
CUDA_VISIBLE_DEVICES=1 python baselines/QLearning/adqn_rnn322.py +alg=ql_rnn_smax alg.MIXER_EMBEDDING_DIM=512 alg.LR=0.001 alg.LR_LINEAR_DECAY=True ++PROJECT=ADQN_test alg.MAP_NAME=6h_vs_8z ++SEED=1
CUDA_VISIBLE_DEVICES=1 python baselines/QLearning/adqn_rnn322.py +alg=ql_rnn_smax alg.MIXER_EMBEDDING_DIM=512 alg.LR=0.001 alg.LR_LINEAR_DECAY=True ++PROJECT=ADQN_test alg.MAP_NAME=6h_vs_8z ++SEED=2
CUDA_VISIBLE_DEVICES=1 python baselines/QLearning/adqn_rnn322.py +alg=ql_rnn_smax alg.MIXER_EMBEDDING_DIM=512 alg.LR=0.001 alg.LR_LINEAR_DECAY=True ++PROJECT=ADQN_test alg.MAP_NAME=6h_vs_8z ++SEED=3
CUDA_VISIBLE_DEVICES=1 python baselines/QLearning/adqn_rnn322.py +alg=ql_rnn_smax alg.MIXER_EMBEDDING_DIM=512 alg.LR=0.001 alg.LR_LINEAR_DECAY=True ++PROJECT=ADQN_test alg.MAP_NAME=6h_vs_8z ++SEED=4