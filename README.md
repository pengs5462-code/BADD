# BADD: Batch-Adaptive Dynamic Distillation for Online Deep Mutual Learning

This repository contains the official implementation of **Batch-Adaptive Dynamic Distillation (BADD)**, a sample-wise reweighting strategy for **online Deep Mutual Learning (DML)**.

BADD uses the **mini-batch mean teacher confidence as a moving reference**, and assigns a **micro-perturbation** weight to each sample to emphasize reliable distillation signals while keeping the overall distillation strength stable across batches.

> **Core idea:** reweight distillation targets **within each mini-batch** by comparing per-sample teacher confidence to the batch mean, with safety clamping and warmup.

---



## Method Overview

In online DML, two peer networks are trained jointly using supervised loss and mutual distillation.  
BADD modifies the distillation term by applying **sample-wise weights** computed from teacher confidence:

- Compute teacher confidence per sample (e.g., maximum softmax probability):  
  $$c_i = \max_k p_T(k \mid x_i)$$

- Use the mini-batch mean as a moving reference:  
  $$\mu_B = \frac{1}{M}\sum_{i=1}^M c_i$$

- Apply batch-adaptive micro-perturbation weighting:  
  $$w_i = 1 + \alpha (c_i - \mu_B)$$

- Safety clamp (recommended range):  
  $$w_i \in [0.8, 1.2]$$

- Warmup for early epochs to stabilize optimization.

This yields a weighted distillation objective:  
$$\mathcal{L} = \mathcal{L}_{CE} + \mathbb{E}_i \left[ w_i \cdot \mathrm{KL}\left(p_S(\cdot|x_i) \,\|\, p_T(\cdot|x_i)\right)\right]$$


---

## Highlights

- **Batch-adaptive reference:** uses mini-batch mean confidence to normalize the reweighting baseline.
- **Micro-perturbation weights:** keeps weights close to 1.0 to avoid training instability.
- **Architecture-agnostic:** supports both **homogeneous** and **heterogeneous** peer settings.
- **Modular research code:** clean separation of models, distillation strategies, training engine, and studies.

---

## Repository Structure

- `src/models/`  
  Peer architectures (e.g., **ResNet32-CIFAR** and **ShuffleNetV2**)

- `src/distill/`  
  Distillation implementations:
  - `strategies.py`: weight-based strategies (BADD variants + ADM-style gating)
  - `loss.py`: unified distillation loss entry (weight strategies + KDCL/OKDDip/ODKD)
  - `dkd.py`: DKD implementation used by ODKD baseline

- `train.py`  
  Main training entry for online DML (supports multiple modes and peer settings)

- `a_study/`  
  Controlled studies and ablations (kept separate from the main pipeline)
  - `alpha_study.py`: alpha sensitivity study for BADD/V17-family weighting

- `tools/`  
  Utilities for log parsing/plotting (optional extensions)

- `experiments/`  
  Local outputs (CSV logs, checkpoints). This directory should be ignored by git.

---

## Implemented Modes (Research Variants and Baselines)

### Weight-based DML (BADD-family and related)
- `baseline` (plain DML with uniform weights)
- `dynamic_v17_11` (recommended BADD-style, batch-mean centered weighting with clamp)
- `dynamic_v18` (adaptive-alpha variant)
- `adm` (asymmetric gating based on teacher-student target gap)

### Non-weight baselines (alternative distillation forms)
- `kdcl` (ensemble-target KL)
- `okddip` (entropy-weighted ensemble distillation)
- `odkd` (DKD-style decoupled distillation)

> The study scripts under `a_study/` provide parameter sensitivity analysis and controlled ablations.

---

## Running (Minimal)

### Environment
Install dependencies:
```bash
pip install -r requirements.txt
