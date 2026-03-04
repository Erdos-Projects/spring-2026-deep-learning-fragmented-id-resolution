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

### A. Deep model: Siamese BiLSTM classifier

Implemented in `src/model.py`, trained via `src/train.py`.

Pipeline:

1. serialize selected attributes for each record
2. character embedding + BiLSTM encoding
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

## 8. Apples-to-Apples Comparison Design

`src/run_comparison_matrix.py` runs a full comparison matrix across:

- methods: `siamese`, `tfidf`
- attribute sets: `baseline`, `extended`
- scenarios: `pair_scoring`, `blocked_pipeline`

Outputs:

- `models/comparisons/apples_to_apples/comparison_summary.tsv`
- `models/comparisons/apples_to_apples/comparison_summary.json`

## 9. Current Results Snapshot (March 4, 2026)

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

## 10. Next Iteration Priorities

1. hard-case error analysis (where methods disagree)
2. better calibration and threshold selection for deployment constraints
3. optional architecture iteration (stronger encoder) only after baseline protocol remains fixed
