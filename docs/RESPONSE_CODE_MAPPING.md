# Response-to-Code Mapping

This file is intended to make the repository update consistent with the revised manuscript and response letter.

## Reviewer 1: filtering-inspired interpretation

- Code: `baddlab/losses.py`
- Diagnostics: `tools/analyze_frequency.py`, `tools/analyze_gradient_share.py`
- Saved traces: `filter_gradient_trace.csv`
- Explanation: BADD is implemented as a batch-centered residual weighting method. The frequency script evaluates low-frequency energy ratios; the gradient-share script evaluates whether weighted KL shifts the gradient budget toward high-residual samples.

## Reviewer 1: larger/more complex datasets

- Code: `baddlab/datasets.py`, `baddlab/models.py`
- Configs:
  - `configs/tiny_imagenet_res18_mbv2.yaml`
  - `configs/imagenet100_res18_mbv2.yaml`
  - `configs/cub200_res18_mbv2.yaml`
- Explanation: These configs support the added Tiny-ImageNet, ImageNet-100, and CUB-200-2011 evaluations.

## Reviewer 1: random seed variance and small perturbations

- Code: `baddlab/losses.py`, `baddlab/train.py`
- Controls:
  - `random_zero_mean`
  - `shuffled_residual`
  - `sign_flipped`
- Aggregation: `tools/aggregate_results.py`
- Explanation: All controls use the same training entry and can reuse the same alpha, warm-up, and clipping settings.

## Reviewer 1: MSP overconfidence concern

- Method wording: README emphasizes that raw MSP is not treated as calibrated reliability.
- Diagnostic: `tools/msp_reliability_eval.py`
- Explanation: The script groups samples by centered MSP residuals and reports teacher top-1 accuracy, teacher CE, and teacher true-label confidence.

## Reviewer 2: reproducibility and public code visibility

- README: updated with method, data preparation, running commands, outputs, and diagnostics.
- Scripts: `scripts/run_sanity.sh`, `scripts/run_reviewer1_core.sh`
- Docs: `docs/REVISION_REPRODUCIBILITY.md`, `docs/GITHUB_UPLOAD_GUIDE.md`
