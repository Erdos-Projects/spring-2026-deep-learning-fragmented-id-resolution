# Project Methodology: NCVoters Duplicate Detection

This document describes the implemented methodology in this repository after the reproducibility and apples-to-apples comparison updates.

## 1. Objective

Given two voter records, predict whether they refer to the same individual.

This is treated as supervised binary classification over labeled pairs:

- duplicate pair (`label=1`)
- non-duplicate pair (`label=0`)

## 2. Data and Labels

Inputs:

- `data/raw/ncvoters.tsv`
- `data/raw/ncvoters_DPL.tsv` (positives)
- `data/raw/ncvoters_NDPL.tsv` (negatives)

Prepared artifacts:

- `data/processed/ncvoters_prepared.tsv`
- `data/processed/ncvoters_attribute_stats.tsv`

Current footprint in this repo:

- records: 14,183
- duplicate pairs: 9,819
- non-duplicate pairs: 98,142

## 3. Preprocessing and Feature Sets

Preprocessing is handled by `scripts/ncvoters_preprocess_and_eda.py`:

- lowercase normalization
- special-character cleanup
- whitespace normalization

Two attribute presets are used in experiments:

- `baseline`: `first_name,last_name,house_num,street_name,zip_code`
- `extended`: baseline + `age,sex,race_desc,ethnic_desc`

## 4. Split Strategy and Leakage Control

The default split strategy is `id_disjoint`, which prevents the same record ID from appearing across train/validation/test.

Deterministic split artifacts are saved to:

- `data/processed/splits/{strategy}_seed{seed}.tsv`

The canonical committed split is:

- `data/processed/splits/id_disjoint_seed42.tsv`

This makes results reproducible and prevents entity-level leakage.

## 5. Shared Blocking for Candidate Generation

To reduce pair-comparison cost, both model families can use the same blocking stage (`src/blocking.py`).

Default blocking used in end-to-end comparisons:

- keys: `first_name,age`
- mode: `any` (pair passes if any blocking key matches)

For the current test split in the apples-to-apples runs:

- positive pass rate: `1.0`
- negative pass rate: `0.03495`
- candidate pairs after blocking: `303` of `2291`

## 6. Model Families

### A. Deep model: Siamese neural classifier

Implemented in `src/model.py`, trained via `src/train.py`.
Supported encoders:

- `bilstm` (default)
- `charcnn`

Pipeline:

1. serialize selected attributes for each record
2. character embedding + encoder (`bilstm` or `charcnn`)
3. absolute difference between encoded pair vectors
4. MLP head outputs duplicate probability

Training details:

- weighted BCE option for class imbalance
- early stopping by validation metric (default monitor: PR-AUC)
- threshold selected on validation (default mode: F1)

### B. Non-deep baseline: TF-IDF distance scorer

Implemented in `src/baseline_tfidf.py`.

Pipeline:

1. serialize records
2. TF-IDF char n-gram embedding
3. cosine distance between record vectors
4. distance threshold selected on validation (default: F1)

This baseline can run with the same split and blocking settings as the Siamese model.

## 7. Evaluation Protocol

The project reports:

- accuracy
- precision
- recall
- F1
- ROC-AUC
- PR-AUC
- confusion matrix

Evaluation supports two scenarios:

- `pair_scoring`: no blocking for either method
- `blocked_pipeline`: shared blocking for both methods

This separates pure scoring quality from realistic candidate-generation conditions.

## 8. Hard-Example Mining and Weighted Training

The repository now mines hard positives and hard negatives directly from the labeled pair files instead of relying first on synthetic label generation.

Definitions:

- hard positives: true duplicate pairs from `DPL` that look unusually different
- hard negatives: true non-duplicate pairs from `NDPL` that look unusually similar

Implementation:

- mining logic: `src/hard_examples.py`
- artifact generation: `src/mine_hard_examples.py`
- weighting-aware training: `src/train.py`
- tuning sweep: `src/tune_hard_weighting.py`

The mining procedure computes pair-level difficulty features such as:

- surname changes / surname expansion
- address changes
- name similarity
- address similarity
- strong exact matches on household and demographic fields

The current real-data hard-example artifact is:

- `data/processed/hard_examples_real/labeled_hard_examples_real.tsv`

Current counts:

- hard examples: `3744`
- hard positives: `1950`
- hard negatives: `1794`

These hard examples are then used as training weights, not as a replacement for the original labeled distribution.

## 9. Apples-to-Apples Comparison Design

`src/run_comparison_matrix.py` runs a full comparison matrix across:

- methods: `siamese`, `tfidf`
- attribute sets: `baseline`, `extended`
- scenarios: `pair_scoring`, `blocked_pipeline`

Outputs:

- `models/comparisons/apples_to_apples/comparison_summary.tsv`
- `models/comparisons/apples_to_apples/comparison_summary.json`

## 10. Current Results Snapshot (March 4, 2026)

From `models/comparisons/apples_to_apples/comparison_summary.json`:

- `pair_scoring + baseline`
  - Siamese F1: `0.9492`
  - TF-IDF F1: `0.9156`
- `pair_scoring + extended`
  - Siamese F1: `0.9806`
  - TF-IDF F1: `0.9176`
- `blocked_pipeline + baseline`
  - Siamese F1: `0.9407`
  - TF-IDF F1: `0.9313`
- `blocked_pipeline + extended`
  - Siamese F1: `0.9849`
  - TF-IDF F1: `0.9630`

Conclusion from the controlled matrix:

- TF-IDF is a strong baseline.
- Shared blocking is effective for scale.
- The Siamese model outperforms TF-IDF when compared under matched split, attributes, and blocking conditions.

## 11. Hard-Weighted Siamese Tuning Snapshot (March 18, 2026)

After mining hard examples from real `DPL/NDPL` pairs, we tuned separate hard-positive and hard-negative weighting scales for the BiLSTM Siamese model.

The sweep output is:

- `models/experiments/hard_weight_tuning_bilstm_blended/tuning_summary.tsv`

The deployment-selected checkpoint is:

- `models/experiments/hard_weight_tuning_bilstm_blended/both_pos0.50_neg0.25_blended_score/best_model.pth`

Reason for selection:

- it improved overall test F1 over the previous Siamese checkpoint
- it improved hard-subset F1 substantially over TF-IDF and modestly over the earlier Siamese
- it achieved perfect rejection on the mined hard negatives in the test split
- it preserved easy-subset quality

Comparison on the blocked extended-attribute test split:

- TF-IDF baseline
  - F1: `0.9630`
  - hard-subset F1: `0.8793`
- previous Siamese BiLSTM
  - F1: `0.9849`
  - hard-subset F1: `0.9677`
- tuned Siamese BiLSTM (`hard_positive_weight_scale=0.50`, `hard_negative_weight_scale=0.25`)
  - F1: `0.9891`
  - hard-subset F1: `0.9833`
  - hard-positive recall: `0.9672`
  - hard-negative rejection: `1.0000`

## 12. Deployment Default

The deployment app currently uses:

- TF-IDF as the non-deep benchmark
- the tuned BiLSTM Siamese checkpoint above as the default deep model

This makes the product demo reflect the current best validated tradeoff between easy-case accuracy and hard-case robustness.

## 13. Next Iteration Priorities

1. repeat the same hard-example tuning process for `charcnn`
2. compare tuned `charcnn` against tuned `bilstm`
3. better calibration and threshold selection for deployment constraints
