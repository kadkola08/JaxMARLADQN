#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=16G
#SBATCH --time=08:00:00

module load singularity/3.10.3
singularity run --nv -B /work:/work -B /scratch:/scratch jaxmarl_latest.sif /bin/bash


# python baselines/QLearning/vdn_rnn.py +alg=ql_rnn_smax alg.MAP_NAME=10m_vs_11m
# python baselines/QLearning/vdn_rnn.py +alg=ql_rnn_smax alg.MAP_NAME=smacv2_5_units
# python baselines/QLearning/vdn_rnn.py +alg=ql_rnn_smax alg.MAP_NAME=6h_vs_8z
# python baselines/QLearning/vdn_rnn.py +alg=ql_rnn_smax alg.MAP_NAME=3s5z_vs_3s6z
# python baselines/QLearning/vdn_rnn.py +alg=ql_rnn_smax

# python baselines/QLearning/qmix_rnn.py +alg=ql_rnn_smax alg.MAP_NAME=10m_vs_11m
# python baselines/QLearning/qmix_rnn.py +alg=ql_rnn_smax alg.MAP_NAME=smacv2_5_units
# python baselines/QLearning/qmix_rnn.py +alg=ql_rnn_smax alg.MAP_NAME=6h_vs_8z
# python baselines/QLearning/qmix_rnn.py +alg=ql_rnn_smax alg.MAP_NAME=3s5z_vs_3s6z
# python baselines/QLearning/qmix_rnn.py +alg=ql_rnn_smax

# python baselines/QLearning/qmix_rnn2.py +alg=ql_rnn_smax alg.MAP_NAME=10m_vs_11m
# python baselines/QLearning/qmix_rnn2.py +alg=ql_rnn_smax alg.MAP_NAME=smacv2_5_units
# python baselines/QLearning/qmix_rnn2.py +alg=ql_rnn_smax alg.MAP_NAME=6h_vs_8z
# python baselines/QLearning/qmix_rnn2.py +alg=ql_rnn_smax alg.MAP_NAME=3s5z_vs_3s6z
# python baselines/QLearning/qmix_rnn2.py +alg=ql_rnn_smax

# python baselines/QLearning/qmix_rnn22.py +alg=ql_rnn_smax alg.MAP_NAME=10m_vs_11m
# python baselines/QLearning/qmix_rnn22.py +alg=ql_rnn_smax alg.MAP_NAME=smacv2_5_units
# python baselines/QLearning/qmix_rnn22.py +alg=ql_rnn_smax alg.MAP_NAME=6h_vs_8z
# python baselines/QLearning/qmix_rnn22.py +alg=ql_rnn_smax alg.MAP_NAME=3s5z_vs_3s6z
# python baselines/QLearning/qmix_rnn22.py +alg=ql_rnn_smax

# python baselines/QLearning/qmix_rnn32.py +alg=ql_rnn_smax alg.MAP_NAME=10m_vs_11m alg.BUFFER_SIZE=1250
# python baselines/QLearning/qmix_rnn32.py +alg=ql_rnn_smax alg.MAP_NAME=smacv2_5_units alg.BUFFER_SIZE=2500
# python baselines/QLearning/qmix_rnn32.py +alg=ql_rnn_smax alg.MAP_NAME=6h_vs_8z
# python baselines/QLearning/qmix_rnn32.py +alg=ql_rnn_smax alg.BUFFER_SIZE=2500 alg.MAP_NAME=3s5z_vs_3s6z
# python baselines/QLearning/qmix_rnn32.py +alg=ql_rnn_smax alg.BUFFER_SIZE=5000
python baselines/QLearning/iql_rnn.py +alg=ql_rnn_smax alg.MAP_NAME=3s5z_vs_3s6z