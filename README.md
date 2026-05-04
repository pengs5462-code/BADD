# BADD: Sample-wise Reallocation of KL Divergence Supervision for Online Mutual Learning

This repository contains the official implementation and reproducibility materials for **BADD**, a lightweight sample-wise KL-divergence supervision reallocation method for online mutual learning.

BADD uses the mini-batch mean teacher confidence as a moving reference and reallocates the KL supervision budget according to centered confidence residuals. The method keeps the batch-average distillation strength close to the original DML objective while shifting relatively more KL supervision toward samples with higher within-batch teacher-confidence residuals.

> Paper title: **BADD: Sample-wise Reallocation of KL Divergence Supervision for Online Mutual Learning**

---

## 1. Method Overview

In online mutual learning, two peer networks are optimized jointly using supervised cross-entropy and bidirectional KL distillation.

For a teacher peer, BADD computes the maximum softmax probability confidence for each sample:

$$
c_i = \max_k p_T(k \mid x_i).
$$

The mini-batch mean is used as a local moving reference:

$$
\mu_B = \frac{1}{M}\sum_{i=1}^{M} c_i .
$$

The centered confidence residual is:

$$
r_i = c_i - \mu_B .
$$

The sample-wise KL weight is computed as:

$$
w_i = 1 + \alpha r_i ,
$$

followed by warm-up, safety clipping, and mean-one renormalization.

The weighted objective is:

$$
\mathcal{L}
=
\mathcal{L}_{CE}
+
\mathbb{E}_i
\left[
w_i
\,
\mathrm{KL}
\left(
p_S(\cdot\mid x_i),
p_T(\cdot\mid x_i)
\right)
\right].
$$

BADD does **not** treat raw MSP as a calibrated probability of correctness. MSP is used after batch centering as a relative within-batch ranking signal.

---

## 2. What This Repository Provides

This repository provides a reproducibility pipeline for the main experiments:

- multi-dataset configurations for **CIFAR-100**, **Tiny-ImageNet**, **ImageNet-100**, and **CUB-200-2011**;
- homogeneous and heterogeneous online mutual learning support through one training entry point;
- DML baseline, BADD, and perturbation-control variants under the same loss interface;
- magnitude-matched control modes: `random_zero_mean`, `shuffled_residual`, and `sign_flipped`;
- frequency-domain diagnostics for confidence drift and centered residual signals;
- gradient-share diagnostics for KL-gradient budget reallocation;
- cumulative KL/gradient redistribution analysis;
- MSP-residual ranking diagnostics from saved checkpoints;
- aggregation scripts for mean/std and paired BADD-vs-DML comparisons.

---

## 3. Repository Structure

~~~text
BADD/
├── README.md
├── requirements.txt
├── requirements_extra.txt
├── configs/
│   ├── default.yaml
│   ├── cifar100_res32_shufv2.yaml
│   ├── tiny_imagenet_res18_mbv2.yaml
│   ├── imagenet100_res18_mbv2.yaml
│   ├── cub200_res18_mbv2.yaml
│   └── imagenet100_synsets_template.txt
├── baddlab/
│   ├── __init__.py
│   ├── datasets.py
│   ├── losses.py
│   ├── models.py
│   ├── train.py
│   └── utils.py
├── tools/
│   ├── aggregate_results.py
│   ├── analyze_frequency.py
│   ├── analyze_gradient_share.py
│   ├── analyze_cumulative_effect.py
│   ├── msp_reliability_eval.py
│   ├── make_imagenet100_subset.py
│   └── prepare_tiny_imagenet.py
├── scripts/
│   ├── run_sanity.sh
│   └── run_core_experiments.sh
├── docs/
│   └── REPRODUCIBILITY.md
└── src/
    └── earlier modular implementation
~~~

The `baddlab/`, `tools/`, `scripts/`, and `docs/` folders provide a compact public pipeline for reproducing the main experiments and diagnostics.

---

## 4. Environment

Install the dependencies:

~~~bash
pip install -r requirements.txt
pip install -r requirements_extra.txt
~~~

A typical environment uses PyTorch, torchvision, numpy, pandas, PyYAML, Pillow, tqdm, and matplotlib.

---

## 5. Data Preparation

### CIFAR-100

CIFAR-100 can be downloaded automatically by torchvision when `download: true` is used in `configs/cifar100_res32_shufv2.yaml`.

### Tiny-ImageNet

Expected structure after extraction:

~~~text
data/tiny-imagenet-200/
├── train/<class>/*.JPEG
└── val/images/*.JPEG
~~~

Prepare the official validation folder into ImageFolder format:

~~~bash
python tools/prepare_tiny_imagenet.py --root data/tiny-imagenet-200
~~~

### ImageNet-100

Expected structure:

~~~text
data/imagenet100/
├── train/<class>/*.JPEG
└── val/<class>/*.JPEG
~~~

If constructing ImageNet-100 from an ImageNet-style source, replace `configs/imagenet100_synsets_template.txt` with the exact 100 synset IDs used in your split, save it as `configs/imagenet100_synsets.txt`, and run:

~~~bash
python tools/make_imagenet100_subset.py \
  --imagenet-root /path/to/imagenet \
  --synsets configs/imagenet100_synsets.txt \
  --out data/imagenet100
~~~

Use `--copy` if symbolic links are not suitable for your filesystem.

### CUB-200-2011

Both structures are supported.

Raw official format:

~~~text
data/CUB_200_2011/
├── images.txt
├── image_class_labels.txt
├── train_test_split.txt
└── images/<class>/<image>.jpg
~~~

ImageFolder split:

~~~text
data/CUB_200_2011/
├── train/<class>/*.jpg
└── val/<class>/*.jpg
~~~

---

## 6. Running Experiments

Use the main entry point from the repository root:

~~~bash
python -m baddlab.train \
  --config <config.yaml> \
  --mode <mode> \
  --seed <seed> \
  --output-root paper_experiments
~~~

### CIFAR-100 heterogeneous setting

~~~bash
python -m baddlab.train \
  --config configs/cifar100_res32_shufv2.yaml \
  --mode badd \
  --seed 0 \
  --output-root paper_experiments
~~~

### Added datasets

~~~bash
python -m baddlab.train --config configs/tiny_imagenet_res18_mbv2.yaml --mode badd --seed 0 --output-root paper_experiments
python -m baddlab.train --config configs/imagenet100_res18_mbv2.yaml --mode badd --seed 0 --output-root paper_experiments
python -m baddlab.train --config configs/cub200_res18_mbv2.yaml --mode badd --seed 0 --output-root paper_experiments
~~~

### Baseline and perturbation controls

Supported modes:

~~~text
baseline
badd
badd_zscore
random_zero_mean
shuffled_residual
sign_flipped
absolute_msp
~~~

Example:

~~~bash
for mode in baseline badd random_zero_mean shuffled_residual sign_flipped; do
  for seed in 0 1 2; do
    python -m baddlab.train \
      --config configs/cifar100_res32_shufv2.yaml \
      --mode ${mode} \
      --seed ${seed} \
      --output-root paper_experiments
  done
done
~~~

You may also run the prepared script:

~~~bash
bash scripts/run_core_experiments.sh
~~~

---

## 7. Aggregation and Diagnostics

Aggregate multi-seed summaries:

~~~bash
python tools/aggregate_results.py --root paper_experiments
~~~

Frequency-domain diagnostic:

~~~bash
python tools/analyze_frequency.py --root paper_experiments
~~~

Gradient-share diagnostic:

~~~bash
python tools/analyze_gradient_share.py --root paper_experiments
~~~

Cumulative redistribution diagnostic:

~~~bash
python tools/analyze_cumulative_effect.py --root paper_experiments
~~~

MSP-residual ranking diagnostic from a saved run:

~~~bash
python tools/msp_reliability_eval.py \
  --run-config paper_experiments/cifar100/shufflenetv2_cifar_vs_resnet32_cifar/baseline/seed_0/run_config.json \
  --checkpoint paper_experiments/cifar100/shufflenetv2_cifar_vs_resnet32_cifar/baseline/seed_0/latest.pt \
  --device cuda
~~~

---

## 8. Notes

- Do not commit datasets, checkpoints, or large experiment outputs.
- Use `paper_experiments/` or another ignored output directory for generated results.
- The file `configs/imagenet100_synsets_template.txt` is a template; replace it with the actual synset list used in your ImageNet-100 split.

---

## Citation

~~~bibtex
@article{sun2026badd,
  title={BADD: Sample-wise Reallocation of KL Divergence Supervision for Online Mutual Learning},
  author={Sun, Peng and Zhong, Yuanhong},
  journal={IEEE Signal Processing Letters},
  year={2026}
}
~~~
