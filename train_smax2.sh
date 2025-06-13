#!/bin/bash

# python baselines/QLearning/qmix_rnn32.py +alg=ql_rnn_smax alg.MAP_NAME=10m_vs_11m alg.BUFFER_SIZE=1250
# python baselines/QLearning/qmix_rnn32.py +alg=ql_rnn_smax alg.MAP_NAME=smacv2_5_units alg.BUFFER_SIZE=2500
# python baselines/QLearning/qmix_rnn32.py +alg=ql_rnn_smax alg.MAP_NAME=6h_vs_8z
# python baselines/QLearning/qmix_rnn32.py +alg=ql_rnn_smax alg.BUFFER_SIZE=2500 alg.MAP_NAME=3s5z_vs_3s6z
# python baselines/QLearning/qmix_rnn32.py +alg=ql_rnn_smax alg.BUFFER_SIZE=5000

# python baselines/QLearning/qmix_rnn32.py +alg=ql_rnn_smax alg.MAP_NAME=5m_vs_6m
# python baselines/QLearning/vdn_rnn.py +alg=ql_rnn_smax alg.MAP_NAME=5m_vs_6m
# python baselines/QLearning/qmix_rnn22.py +alg=ql_rnn_smax alg.MAP_NAME=5m_vs_6m

# python baselines/QLearning/qmix_rnn32.py +alg=ql_rnn_smax alg.MAP_NAME=3s5z_vs_3s6z ++SEED=0 alg.BUFFER_SIZE=2500
# python baselines/QLearning/qmix_rnn32.py +alg=ql_rnn_smax alg.MAP_NAME=3s5z_vs_3s6z ++SEED=2 alg.BUFFER_SIZE=2500
# python baselines/QLearning/qmix_rnn32.py +alg=ql_rnn_smax alg.MAP_NAME=3s_vs_5z ++SEED=2

# python baselines/QLearning/qmix_rnn22.py +alg=ql_rnn_smax alg.MAP_NAME=3s_vs_5z ++SEED=1
# python baselines/QLearning/qmix_rnn22.py +alg=ql_rnn_smax alg.MAP_NAME=3s_vs_5z ++SEED=2 # alg.BUFFER_SIZE=1250
# python baselines/QLearning/qmix_rnn.py +alg=ql_rnn_smax alg.MAP_NAME=smacv2_10_units ++SEED=0 alg.BUFFER_SIZE=1250
# python baselines/QLearning/vdn_rnn.py +alg=ql_rnn_smax alg.MAP_NAME=smacv2_10_units ++SEED=0 alg.BUFFER_SIZE=1250

# python baselines/QLearning/qmix_rnn22.py +alg=ql_rnn_smax alg.MAP_NAME=2s3z ++SEED=0
# python baselines/QLearning/qmix_rnn22.py +alg=ql_rnn_smax alg.MAP_NAME=2s3z ++SEED=1
# python baselines/QLearning/qmix_rnn22.py +alg=ql_rnn_smax alg.MAP_NAME=2s3z ++SEED=2

# python baselines/QLearning/qmix_rnn.py +alg=ql_rnn_smax alg.MAP_NAME=2s3z ++SEED=0
# python baselines/QLearning/qmix_rnn.py +alg=ql_rnn_smax alg.MAP_NAME=2s3z ++SEED=1
# python baselines/QLearning/qmix_rnn.py +alg=ql_rnn_smax alg.MAP_NAME=2s3z ++SEED=2

# python baselines/QLearning/vdn_rnn.py +alg=ql_rnn_smax alg.MAP_NAME=2s3z ++SEED=0
# python baselines/QLearning/vdn_rnn.py +alg=ql_rnn_smax alg.MAP_NAME=2s3z ++SEED=1
python baselines/QLearning/vdn_rnn.py +alg=ql_rnn_smax alg.MAP_NAME=2s3z ++SEED=2

