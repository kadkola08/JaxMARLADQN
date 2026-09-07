#!/usr/bin/env bash
#
# Run IQL, VDN and QMIX on all "adapted" and "other" overcooked_v2 layouts.
#
# Each algorithm has its own entry-point script and they share the
# ql_cnn_rnn_overcooked alg config. Layout is selected via a Hydra override
# (alg.ENV_KWARGS.layout=<name>); ENV_NAME=overcooked_v2 is set in that config.
#
# Usage:
#   ./run_iql_vdn_qmix_layouts.sh              # run everything (seeds 0 1 2)
#   SEEDS="0 1 2 3 4" ./run_iql_vdn_qmix_layouts.sh   # override seeds
#   EXTRA="alg.TOTAL_TIMESTEPS=1e6" ./run_iql_vdn_qmix_layouts.sh
#
set -euo pipefail

# Run from the directory containing this script (baselines/QLearning).
cd "$(dirname "$0")"

SEEDS="${SEEDS:-0 1 2}"   # space-separated list of seeds
read -r -a SEED_LIST <<< "${SEEDS}"
ALG_CONFIG="${ALG_CONFIG:-ql_cnn_rnn_overcooked2}"
EXTRA="${EXTRA:-}"   # any extra Hydra overrides, space-separated

# algorithm name -> entry-point script
declare -A ALGOS=(
  # [iql]="iql_cnn_rnn_overcooked32.py"
  [vdn]="vdn_cnn_rnn_overcooked.py"
  [qmix]="qmix_cnn_rnn_overcooked.py"
)

# Adapted layouts (multi-ingredient + recipe indicator)
ADAPTED_LAYOUTS=(
  cramped_room_v2
  # asymm_advantages_recipes_center
  # asymm_advantages_recipes_right
  # asymm_advantages_recipes_left
  # two_rooms
)

# Other layouts
OTHER_LAYOUTS=(
  # two_rooms_both
  long_room
  fun_coordination
  # more_fun_coordination
  fun_symmetries
  # fun_symmetries_plates
  # fun_symmetries1
  # overcookedv2_demo
)

LAYOUTS=("${ADAPTED_LAYOUTS[@]}" "${OTHER_LAYOUTS[@]}")

total=$(( ${#ALGOS[@]} * ${#LAYOUTS[@]} * ${#SEED_LIST[@]} ))
count=0
failures=()

echo "Running ${#ALGOS[@]} algorithms x ${#LAYOUTS[@]} layouts x ${#SEED_LIST[@]} seeds = ${total} runs (SEEDS=${SEEDS})"

for algo in "${!ALGOS[@]}"; do
  script="${ALGOS[$algo]}"
  for layout in "${LAYOUTS[@]}"; do
    for seed in "${SEED_LIST[@]}"; do
      count=$(( count + 1 ))
      echo ""
      echo "=============================================================="
      echo "[${count}/${total}] ${algo}  |  layout=${layout}  |  seed=${seed}"
      echo "=============================================================="
      if python "${script}" \
          "+alg=${ALG_CONFIG}" \
          "alg.ENV_KWARGS.layout=${layout}" \
          "++SEED=${seed}" \
          ${EXTRA}; then
        echo "OK: ${algo} / ${layout} / seed ${seed}"
      else
        echo "FAILED: ${algo} / ${layout} / seed ${seed}"
        failures+=("${algo}/${layout}/seed${seed}")
      fi
    done
  done
done

echo ""
echo "=============================================================="
if [ ${#failures[@]} -eq 0 ]; then
  echo "All ${total} runs completed successfully."
else
  echo "${#failures[@]}/${total} runs FAILED:"
  printf '  - %s\n' "${failures[@]}"
  exit 1
fi
