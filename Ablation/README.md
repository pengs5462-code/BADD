# A Study: Alpha Sensitivity

This folder contains controlled studies and ablation scripts that are intentionally separated from the main training pipeline.

## Alpha Study (V17-family)
We sweep the alpha parameter that controls the magnitude of batch-mean centered confidence reweighting:

- `conf_i = max softmax(teacher_logits)_i`
- `mu_B = mean(conf_i)`
- `w_i = 1 + alpha * (conf_i - mu_B)`
- clamp to `[0.8, 1.2]`
- warmup for the first 10 epochs

Outputs:
- Per-alpha epoch logs: `log_alpha_{alpha}.csv`
- Summary table: `summary_accuracy_vs_alpha.csv`