# Revision Reproducibility Guide

This document explains how the public code corresponds to the revised manuscript experiments.

## Main experimental groups

| Manuscript / response item | Public file(s) | Purpose |
|---|---|---|
| CIFAR-100 heterogeneous setting | `configs/cifar100_res32_shufv2.yaml` | ResNet-32 / ShuffleNetV2 online mutual learning setting |
| Tiny-ImageNet generalization | `configs/tiny_imagenet_res18_mbv2.yaml` | Added complex low-resolution ImageNet-style benchmark |
| ImageNet-100 generalization | `configs/imagenet100_res18_mbv2.yaml` | Added ImageNet-derived benchmark |
| CUB-200-2011 fine-grained recognition | `configs/cub200_res18_mbv2.yaml` | Added fine-grained benchmark |
| BADD and controls | `baddlab/losses.py` | Implements `baseline`, `badd`, `random_zero_mean`, `shuffled_residual`, `sign_flipped` |
| Multi-seed training | `baddlab/train.py`, `scripts/run_reviewer1_core.sh` | Runs seeds 0/1/2 under matched configs |
| Mean/std + paired wins | `tools/aggregate_results.py` | Aggregates `summary.json` files |
| Frequency evidence | `tools/analyze_frequency.py` | Computes low-frequency energy ratio from saved traces |
| Gradient-share evidence | `tools/analyze_gradient_share.py` | Measures top-residual KL-gradient share shifts |
| Cumulative effect | `tools/analyze_cumulative_effect.py` | Tracks accumulated KL/gradient redistribution |
| MSP residual ranking | `tools/msp_reliability_eval.py` | Groups samples by centered residual and reports teacher quality indicators |

## Recommended run order

1. Prepare datasets.
2. Run baseline and BADD for seeds 0, 1, 2.
3. Run perturbation controls for the same seeds.
4. Aggregate results.
5. Run diagnostics from the saved `epoch_metrics.csv`, `filter_gradient_trace.csv`, and checkpoint files.

## Example: CIFAR-100 controls

```bash
for mode in baseline badd random_zero_mean shuffled_residual sign_flipped; do
  for seed in 0 1 2; do
    python -m baddlab.train \
      --config configs/cifar100_res32_shufv2.yaml \
      --mode ${mode} \
      --seed ${seed} \
      --output-root paper_experiments_reviewer1
  done
done

python tools/aggregate_results.py --root paper_experiments_reviewer1
python tools/analyze_frequency.py --root paper_experiments_reviewer1
python tools/analyze_gradient_share.py --root paper_experiments_reviewer1
python tools/analyze_cumulative_effect.py --root paper_experiments_reviewer1
```

## Notes on public release

Do not commit datasets, checkpoint files, private reviewer correspondence, or manuscript files to the public repository. The public repository should contain code, configuration, and reproducibility documentation only.
