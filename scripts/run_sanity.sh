#!/usr/bin/env bash
set -euo pipefail

python -m baddlab.train \
  --config configs/cifar100_res32_shufv2.yaml \
  --mode baseline \
  --seed 0 \
  --epochs 1 \
  --output-root debug_runs

python tools/aggregate_results.py --root debug_runs
