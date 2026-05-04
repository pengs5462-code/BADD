#!/usr/bin/env bash
set -euo pipefail

OUT_ROOT=${OUT_ROOT:-paper_experiments_reviewer1}
SEEDS=${SEEDS:-"0 1 2"}
MODES=${MODES:-"baseline badd random_zero_mean shuffled_residual sign_flipped"}
CONFIGS=${CONFIGS:-"configs/cifar100_res32_shufv2.yaml configs/tiny_imagenet_res18_mbv2.yaml configs/imagenet100_res18_mbv2.yaml configs/cub200_res18_mbv2.yaml"}

for cfg in ${CONFIGS}; do
  for mode in ${MODES}; do
    for seed in ${SEEDS}; do
      echo "[run] config=${cfg} mode=${mode} seed=${seed} out=${OUT_ROOT}"
      python -m baddlab.train \
        --config "${cfg}" \
        --mode "${mode}" \
        --seed "${seed}" \
        --output-root "${OUT_ROOT}"
    done
  done
done

python tools/aggregate_results.py --root "${OUT_ROOT}"
python tools/analyze_frequency.py --root "${OUT_ROOT}" || true
python tools/analyze_gradient_share.py --root "${OUT_ROOT}" || true
python tools/analyze_cumulative_effect.py --root "${OUT_ROOT}" || true
