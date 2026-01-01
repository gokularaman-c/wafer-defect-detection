# Wafer Defect Detection (WaferNet) — Mask-Aware DenseNet for Wafer Map Patterns

Automated wafer-map defect pattern classification using a pragmatic data pipeline + a compact DenseNet variant (“WaferNet”) with **channel attention (ECA)** and **mask-aware dual pooling (GAP ∥ GMP)**.

This repo is designed to be **reproducible** and **deployment-friendly**: fixed input size, explicit wafer geometry masking, class-imbalance handling, and leakage-aware evaluation.

---

## What this project does

Given a wafer map (die-grid) from the **WM-811K / LSWMD** dataset, we:

* Clean/standardize labels into **9 classes**:
  `Center, Donut, Edge-Loc, Edge-Ring, Loc, Random, Scratch, Near-Full, None`
* Convert each sample into a **2-channel tensor** of shape **[2, 96, 96]**:

  * **Channel 0 (wafer map):** values `{0,1,2} → {0.0, 0.5, 1.0}`

    * `0` = background / no-die
    * `1` = pass
    * `2` = fail
  * **Channel 1 (mask):** binary wafer geometry mask `{0,1}` (valid die sites vs background)
* Train **WaferNet** to output a **9-logit** vector per wafer; apply softmax for probabilities.
* Optionally generate **Grad-CAM** explanations constrained to the wafer region.

---

## Why the “mask channel” exists (and why you should care)

Wafer maps have:

* variable shapes/sizes (different die grids),
* lots of “empty” background,
* defects whose geometry matters.

Instead of warping wafers via resizing/cropping, we **standardize to 96×96** and provide an explicit **mask** so the model can ignore padded/background regions during pooling and explanation. This is a central design choice in WaferNet.

---

## Model: WaferNet (high-level)

**Backbone:** DenseNet-121 (trained from scratch; first conv modified to accept 2 channels).
**Attention:** Efficient Channel Attention (ECA) after the final DenseNet feature map BN+ReLU.
**Head:** Mask-aware pooling on downsampled mask:

* masked Global Average Pooling (GAP)
* masked Global Max Pooling (GMP)
* concatenate → **2048-d**, then Linear → **9 logits**

This keeps the model lightweight (~7M params) and robust to background/padding.

---

## Training recipe (default)

Typical training settings used in the project:

* Optimizer: **AdamW** (lr `3e-4`, weight decay `1e-4`)
* Batch size: `64`
* Epochs: `30`
* Loss: **Logit-Adjusted Cross Entropy** (helps long-tail class imbalance)

Hardware support:

* CUDA (NVIDIA) or Apple Silicon **MPS**, with CPU fallback.

---

## Data curation & imbalance strategy (the “make it not lie to you” section)

### 1) Labeled-only extraction

Only use labeled wafers (unlabeled removed).

### 2) Fixed resolution at 96×96

* If `H ≤ 96` and `W ≤ 96`: **zero-pad to 96×96** (no warping).
* If larger than 96 in either dimension: **exclude** (to avoid resizing artifacts).

### 3) Leakage-aware splitting (recommended)

Wafers are associated with manufacturing **lots**; random splitting can leak lot-specific artifacts.

* Prefer **lot-disjoint splits** (no lot appears in both train and test), while keeping class ratios.
* If lot metadata is missing, fall back to standard stratified split.

### 4) Training-set rebalancing (only on train split)

* Cap **None** to ~25,000 samples
* Up-sample minority classes to ~3,000 using **label-preserving transforms**

  * 90° rotations and flips

Validation/test are left untouched to avoid “cheating by augmentation”.

---

## Suggested repository layout

> If your current code structure differs, adapt these names—this layout is the clean “canonical” version.

```
.
├── data/
│   ├── raw/               # LSWMD.pkl (or extracted files)
│   ├── processed/         # cached tensors, manifests, splits
│   └── manifests/         # split indices, lot ids, seeds
├── configs/
│   └── wafernet.yaml
├── src/
│   ├── data/
│   │   ├── make_splits.py
│   │   ├── preprocess.py
│   │   └── dataset.py
│   ├── models/
│   │   ├── wafernet.py
│   │   └── eca.py
│   ├── train.py
│   ├── eval.py
│   └── explain.py         # Grad-CAM utilities
├── scripts/
│   ├── download_data.md
│   └── run_experiments.sh
├── outputs/
│   ├── checkpoints/
│   ├── metrics/
│   └── gradcam/
├── requirements.txt
└── README.md
```

---

## Setup

### 1) Environment

Use Python 3.10+ (recommended).

Example (venv):

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

### 2) Key dependencies (typical)

* torch, torchvision
* numpy, pandas
* scikit-learn (metrics/splits)
* matplotlib (plots)
* opencv-python or pillow (image ops)
* tqdm, pyyaml
* (optional) streamlit/gradio for a demo UI

---

## Data: getting WM-811K / LSWMD

A common distribution is the Kaggle dataset containing `LSWMD.pkl`.

**Place it here:**

```
data/raw/LSWMD.pkl
```

---

## Run pipeline (example commands)

### 1) Preprocess → build 2-channel 96×96 tensors + manifests

```bash
python -m src.data.preprocess \
  --input data/raw/LSWMD.pkl \
  --output data/processed \
  --size 96
```

### 2) Create splits (lot-disjoint if lot_id exists)

```bash
python -m src.data.make_splits \
  --processed data/processed \
  --out data/manifests \
  --split 0.70 0.15 0.15 \
  --seed 42 \
  --lot_disjoint true
```

### 3) Train

```bash
python -m src.train \
  --config configs/wafernet.yaml \
  --manifests data/manifests \
  --out outputs \
  --seed 42
```

### 4) Evaluate (macro-F1, confusion matrix, etc.)

```bash
python -m src.eval \
  --ckpt outputs/checkpoints/best.pt \
  --manifests data/manifests \
  --out outputs/metrics
```

### 5) Grad-CAM (mask-constrained)

```bash
python -m src.explain \
  --ckpt outputs/checkpoints/best.pt \
  --sample_id <ID> \
  --out outputs/gradcam
```

---

## Reproducibility checklist (non-negotiable if you want believable results)

This project’s philosophy: **if you can’t reproduce it, it didn’t happen**.

Minimum requirements:

* Fix seeds across `random`, `numpy`, and `torch`
* Use deterministic settings where possible
* Save:

  * exact train/val/test indices
  * lot IDs (if available)
  * augmentation recipe
  * class caps / target counts
  * library versions + git commit hash

---
