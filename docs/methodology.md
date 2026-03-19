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

The original deployment-selected checkpoint was:

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

## 12. Encoder Comparison Under the Same Hard-Example Sweep

We repeated the same hard-example weighting sweep for `charcnn`.

Artifacts:

- `models/experiments/hard_weight_tuning_charcnn_blended/tuning_summary.tsv`
- `models/experiments/encoder_hard_weight_comparison.tsv`

Best overall CharCNN run:

- `models/experiments/hard_weight_tuning_charcnn_blended/both_pos1.00_neg1.00_blended_score`
  - F1: `0.9766`
  - PR-AUC: `0.9946`
  - hard-subset F1: `0.9683`
  - easy-subset F1: `0.9797`

Best hard-subset CharCNN run:

- `models/experiments/hard_weight_tuning_charcnn_blended/both_pos0.50_neg0.50_blended_score`
  - F1: `0.9662`
  - hard-subset F1: `0.9913`
  - easy-subset F1: `0.9385`

Conclusion:

- CharCNN can slightly outperform the tuned BiLSTM on the mined hard subset in one setting.
- But it loses too much overall and easy-subset quality to be the better deployment default.
- The tuned BiLSTM remains the better balanced model.

## 13. Deployment Default

The deployment app currently uses:

- TF-IDF as the non-deep benchmark
- the tuned BiLSTM checkpoint:
  - `models/experiments/hard_weight_tuning_extended_midl_sex_aware_bilstm_blended/both_pos0.50_neg0.75_blended_score/best_model.pth`

Reason:

- it adds `midl_name`
- it uses sex-aware pair features instead of a brittle hard rule
- it fixes the obvious surname-expansion duplicate misses that appeared in deployment review
- it still keeps `hard-negative rejection = 1.0` on the mined hard test subset

Key metrics for this deployment default:

- F1: `0.9935`
- PR-AUC: `0.9998`
- hard-subset F1: `0.9833`
- easy-subset F1: `0.9971`
- hard-positive recall: `0.9672`
- hard-negative rejection: `1.0000`

This makes the product demo reflect the current best validated tradeoff between easy-case accuracy, difficult-case behavior, and real review-case quality.

## 14. Next Iteration Priorities

1. improve deployment calibration and threshold selection for review-vs-auto-merge operating points
2. analyze disagreement cases between TF-IDF, tuned BiLSTM, and CharCNN to identify systematic failure modes
3. test whether adding or down-weighting volatile fields improves robustness on surname changes and demographic conflicts

## 15. Middle-Name and Sex-Aware Pair-Feature Follow-Up

Review cases from the deployment UI exposed a specific failure mode: some true duplicates shared the same first name, last name, address, ZIP, age, and sex, but differed in surname expansion or middle-name presentation. Examples included cases like:

- `emma yarbaugh` vs `emma perkins yarbaugh`
- `yajaira diaz` vs `yajaira diaz kumar`
- `alejandro garcia` vs `alejandro paniagua garcia`

To probe that failure mode without hard-coding deterministic business rules, we tested two additions:

1. `extended_midl`
   - keep `race_desc` and `ethnic_desc`
   - add `midl_name` to the serialized record encoder input
2. `sex_aware_name`
   - add explicit pair features such as:
     - same middle name
     - both male / both female
     - male same-first-name with different last names
     - female same-first-name with different last names
     - the same patterns conditioned on same age

Artifacts:

- `models/experiments/extended_midl_bilstm_blended/best_metrics.json`
- `models/experiments/extended_midl_sex_aware_bilstm_blended/best_metrics.json`

Results on the same blocked hard-weighted setting:

- current tuned deployment BiLSTM
  - F1: `0.9891`
  - hard-subset F1: `0.9833`
  - hard-negative rejection: `1.0000`
- `extended_midl`
  - F1: `0.9847`
  - hard-subset F1: `0.9756`
  - hard-negative rejection: `0.9565`
- `extended_midl + sex_aware_name`
  - F1: `0.9892`
  - hard-subset F1: `0.9677`
  - hard-negative rejection: `0.9348`

Interpretation:

- adding `midl_name` clearly helps some surname-expansion duplicate cases
- the sex-aware pair features also suppress some suspicious same-first-name male collisions
- however, the new feature set currently gives up too much hard-negative protection to replace the existing deployment checkpoint

So the follow-up is promising, but it is not yet the new deployment default.
