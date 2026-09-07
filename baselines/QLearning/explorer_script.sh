#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:v100-sxm2:1
#SBATCH --mem=32Gb
#SBATCH --time=08:00:00
#SBATCH --job-name=adqn_oveercooked
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=kadkolag08@gmail.com
#SBATCH --output=/home/kadkol.a/logs/qmix_%j.out
#SBATCH --error=/home/kadkol.a/logs/qmix_%j.err

# Usage:
#   sbatch explorer_script.sh <script_file>
#     <script_file> : python file in baselines/QLearning/ to run
#                     (e.g. adqn_cnn_rnn_overcooked.py, iql_cnn_rnn_overcooked2.py)
#
# Layouts are listed inside the container script below; comment out the ones
# you don't want to run.

# Load singularity module
module load singularity/3.10.3

# Script file to run (located in baselines/QLearning/) from CLI
SCRIPT_FILE="${1:?Usage: sbatch explorer_script.sh <script_file>}"
SCRIPT_FILE="$(basename "$SCRIPT_FILE")"   # keep just the filename

# Create a temporary script to run inside the container
cat > /tmp/run_inside_container_$$.sh << 'EOF'
#!/bin/bash
cd /home/kadkol.a/JaxMARLADQN
export PYTHONPATH=/home/kadkol.a/JaxMARLADQN:$PYTHONPATH

# Args passed in from the launcher
SCRIPT_FILE="$1"

# Layouts to run -- comment out any you don't want.
LAYOUTS=(
    # Adapted layouts
    "cramped_room_v2"
    "asymm_advantages_recipes_center"
    # "asymm_advantages_recipes_right"
    # "asymm_advantages_recipes_left"
    "two_rooms"
    # Other layouts
    # "two_rooms_both"
    # "long_room"
    "fun_coordination"
    # "more_fun_coordination"
    "fun_symmetries"
    # "fun_symmetries_plates"
    # "fun_symmetries1"
    # "overcookedv2_demo"
)

for MAP_NAME in "${LAYOUTS[@]}"; do
for timesteps in 5e6; do
for hid_mult in 1; do
for num_layers in 1; do
for seed in 40 41 42; do
for lr in 0.00007; do
    for num_steps in 8; do
        for hidden_size in 128; do
            # Calculate reward shaping horizon as half of total timesteps
            rew_horizon=$(echo $timesteps | awk '{print $1/2}')

            python baselines/QLearning/${SCRIPT_FILE} \
                +alg=ql_cnn_rnn_overcooked2.yaml \
                alg.ENV_KWARGS.layout=${MAP_NAME} \
                alg.ENV_KWARGS.agent_view_size=2 \
                alg.TOTAL_TIMESTEPS=$timesteps \
                alg.NUM_ENVS=32 \
                alg.NUM_STEPS=$num_steps \
                alg.LR=$lr \
                alg.LR_LINEAR_DECAY=False \
                alg.HIDDEN_SIZE=$hidden_size \
                alg.REW_SHAPING_HORIZON=$rew_horizon \
                alg.NUM_EPOCHS=4 \
                alg.HIDDEN_MULTIPLIER=$hid_mult \
                alg.NUM_ENCODING_LAYERS=$num_layers \
                ++SEED=$seed
        done
    done
done
done
done
done
done
done
EOF

# Make the temporary script executable
chmod +x /tmp/run_inside_container_$$.sh

# Run the singularity container with the script
singularity exec --nv \
    -B /work:/work \
    -B /scratch:/scratch \
    -B /home/kadkol.a:/home/kadkol.a \
    jaxmarl_latest.sif \
    /tmp/run_inside_container_$$.sh "$SCRIPT_FILE"

# Clean up
rm /tmp/run_inside_container_$$.sh
