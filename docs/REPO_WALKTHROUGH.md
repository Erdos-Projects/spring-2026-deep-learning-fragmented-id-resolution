# Repository Walkthrough

## Purpose

This repository studies duplicate detection / entity resolution on the NCVoters benchmark.

The core task is:

- input: two voter records
- output: predict whether they refer to the same real person

The project currently contains:

1. raw-data preprocessing
2. exploratory data analysis (attribute-level and pair-level)
3. a reproducible Siamese pipeline (BiLSTM or CharCNN encoder)
4. a TF-IDF embedding-distance baseline
5. saved split and experiment artifacts

## High-Level Pipeline

### 1. Raw data

The main files are:

- `data/raw/ncvoters.tsv`
- `data/raw/ncvoters_DPL.tsv`
- `data/raw/ncvoters_NDPL.tsv`

Definitions:

- `DPL`: labeled duplicate pairs
- `NDPL`: labeled non-duplicate pairs

### 2. Preprocessing

Raw records are normalized into:

- `data/processed/ncvoters_prepared.tsv`

Attribute-level EDA is written to:

- `data/processed/ncvoters_attribute_stats.tsv`

### 3. Split generation

The project now uses deterministic pair splits. The key split artifact is:

- `data/processed/splits/id_disjoint_seed42.tsv`

This split is important because the same record IDs do not appear across train/val/test.

### 4. Modeling

There are two modeling paths:

- deep model: Siamese neural classifier (BiLSTM or CharCNN encoder)
- baseline: TF-IDF embedding-distance scorer

### 5. Evaluation artifacts

Saved outputs are committed under:

- `models/runs/baseline_id_disjoint/`
- `models/baselines/tfidf_id_disjoint/`

## Directory Map

### `scripts/`

Contains preprocessing and exploratory analysis scripts.

### `src/`

Contains reusable utilities and model pipelines.

### `tests/`

Contains smoke tests and unit tests for utilities, splits, models, and the baseline.

### `docs/`

Contains methodology, dataset notes, EDA insights, and this walkthrough.

### `data/`

Contains raw inputs, processed tables, and deterministic split artifacts.

### `models/`

Contains saved checkpoints and metrics from committed experiments.

## Script-by-Script Walkthrough

## `scripts/ncvoters_preprocess_and_eda.py`

Role:

- main preprocessing entrypoint

What it does:

- reads `ncvoters.tsv`
- lowercases text
- removes special characters
- normalizes whitespace
- optionally keeps only the NCVoters attribute subset
- writes cleaned TSV
- computes attribute-level completeness and uniqueness statistics

Outputs:

- `data/processed/ncvoters_prepared.tsv`
- `data/processed/ncvoters_attribute_stats.tsv`

Use when:

- rebuilding prepared data
- re-running basic attribute EDA

## `scripts/pair_eda.py`

Role:

- quick inspection of labeled duplicate/non-duplicate pairs

What it does:

- loads DPL/NDPL pair files and prepared records via CLI paths
- validates required pair columns (`id1`, `id2`)
- reports missing IDs in pair files
- merges pair IDs back to the prepared record table
- computes Levenshtein distances for names
- prints deterministic sampled duplicate and non-duplicate pairs
- reports basic overlap counts

Use when:

- inspecting how hard the matching task is
- sanity-checking pair files

## `scripts/extended_eda.py`

Role:

- more strategic EDA for modeling and blocking

What it does:

- loads data and pair files from CLI paths
- evaluates blocking key match rates for DPL and NDPL
- reports `first_name OR age` and `first_name AND age` rates when available
- summarizes how attributes change across duplicate and non-duplicate pairs
- estimates serialized record lengths from configurable columns

Use when:

- selecting blocking keys
- choosing attributes
- estimating input length constraints

## `src/data_utils.py`

Role:

- shared utility layer used by both model families

What it does:

- defines baseline and extended attribute sets
- loads prepared data
- loads DPL/NDPL pair files
- checks for duplicate canonical pairs
- counts missing IDs
- sets random seeds

This is the lowest-level shared module and should stay model-agnostic.

## `src/splits.py`

Role:

- split creation and validation

What it does:

- builds `pair_random` splits
- builds `id_disjoint` splits
- saves and loads split artifacts
- checks overlap and class counts

Important property:

- `id_disjoint` prevents the same record from appearing in multiple splits

## `src/blocking.py`

Role:

- shared blocking / candidate-generation utilities

What it does:

- parses blocking keys
- computes whether a pair survives the block
- summarizes positive/negative pass rates

Important usage:

- both the Siamese pipeline and the TF-IDF baseline can now use the same blocking stage

## `src/metrics.py`

Role:

- shared evaluation logic

What it computes:

- accuracy
- precision
- recall
- F1
- ROC-AUC
- PR-AUC
- confusion matrix

It also supports validation-based threshold selection.

## `src/dataset.py`

Role:

- PyTorch dataset for the Siamese model

What it does:

- loads prepared records
- concatenates selected attributes into one text string
- builds a character vocabulary
- converts text into fixed-length character sequences
- returns `(x1, x2, label)` tensors for model training

Notes:

- this is a character-level representation
- input length is capped by `max_len`

## `src/model.py`

Role:

- Siamese neural network definition

Architecture:

- character embedding layer
- encoder options:
  - bidirectional LSTM (`bilstm`)
  - character CNN (`charcnn`)
- max pooling over sequence outputs
- absolute-difference comparison
- MLP classifier head

Important clarification:

- this is not a metric-learning embedding model in the strict sense
- it is a Siamese classifier that consumes two encoded records and predicts a label

## `src/train.py`

Role:

- main training entrypoint for the Siamese model

What it does:

- loads prepared data and labeled pairs
- builds or loads a split artifact
- constructs train/val/test datasets
- trains the Siamese model
- supports weighted loss for imbalance
- selects a threshold on validation
- early-stops based on a chosen validation metric
- can apply shared blocking before model scoring
- saves checkpoint and metrics artifacts

Outputs include:

- checkpoint
- metrics history
- best metrics summary
- threshold sweep
- confusion matrix

## `src/evaluate.py`

Role:

- standalone evaluation entrypoint for saved Siamese checkpoints

What it does:

- reloads model and vocabulary
- scores a requested split
- applies checkpoint blocking settings unless overridden
- uses either checkpoint threshold or a newly tuned threshold
- writes JSON/TSV evaluation artifacts

Use when:

- re-evaluating a trained model
- generating external metrics files

## `src/baseline_tfidf.py`

Role:

- fair non-deep baseline

What it does:

- serializes each record independently
- builds TF-IDF embeddings
- computes cosine distance for each pair
- optionally applies the shared blocking stage
- tunes a distance threshold on validation
- writes baseline artifacts

Important detail:

- in the committed run, blocking is enabled with `first_name OR age`

## `src/run_ablations.py`

Role:

- convenience runner for Siamese experiments

What it does:

- runs the Siamese pipeline for baseline attributes
- runs the Siamese pipeline for extended attributes
- summarizes which one performs better

Use when:

- comparing attribute subsets for the Siamese model

## `src/hard_examples.py`

Role:

- mines real hard positives and hard negatives from the labeled pair files

What it does:

- scores pair difficulty from name, address, and attribute overlap
- identifies hard positives from `DPL`
- identifies hard negatives from `NDPL`
- prepares per-pair weights for training and hard-subset reporting

Use when:

- building hard-example artifacts from real labels
- biasing Siamese training toward informative edge cases

## `src/mine_hard_examples.py`

Role:

- CLI wrapper for hard-example artifact generation

What it does:

- loads prepared records and labeled pair files
- runs the miner in `src/hard_examples.py`
- writes TSV and summary artifacts under `data/processed/hard_examples_real/`

## `src/tune_hard_weighting.py`

Role:

- sweep runner for hard-positive and hard-negative weighting scales

What it does:

- launches repeated Siamese training runs
- varies the hard-example weight scales
- writes per-run outputs and a compact tuning summary table

## `src/run_comparison_matrix.py`

Role:

- convenience runner for apples-to-apples comparisons

What it does:

- runs Siamese and TF-IDF under the same split
- compares baseline and extended attributes
- compares unblocked pair scoring and blocked pipeline scenarios
- writes a summary table for all runs

## Tests

### `tests/test_data_utils.py`

- pair integrity checks
- missing-ID handling

### `tests/test_splits.py`

- split correctness
- ID overlap prevention

### `tests/test_model_and_metrics.py`

- model I/O checks
- metric correctness

### `tests/test_smoke_train_eval.py`

- end-to-end Siamese smoke test

### `tests/test_baseline_tfidf.py`

- TF-IDF blocking and threshold logic
- TF-IDF smoke test

## Current Saved Experiments

### Siamese run

Directory:

- `models/runs/baseline_id_disjoint/`

Configuration summary:

- split: `id_disjoint`
- attributes: `first_name,last_name,house_num,street_name,zip_code`
- max length: `80`
- model: character-level Siamese with selectable encoder (`bilstm` or `charcnn`)

### TF-IDF baseline run

Directory:

- `models/baselines/tfidf_id_disjoint/`

Configuration summary:

- split: same `id_disjoint` artifact
- attributes: extended attribute set
- analyzer: char n-grams `2..4`
- blocking: `first_name OR age`

### Apples-to-apples matrix run

Directory:

- `models/comparisons/apples_to_apples/`

Contains:

- `comparison_summary.tsv`
- `comparison_summary.json`
- per-run outputs under `siamese/` and `tfidf/` for each scenario and attribute set

### Hard-example tuning runs

Directories:

- `models/experiments/hard_weight_tuning_bilstm_blended/`
- `models/experiments/hard_weight_tuning_charcnn_blended/`

Contains:

- per-run checkpoints and metrics for each weighting configuration
- one `tuning_summary.tsv` per encoder
- `models/experiments/encoder_hard_weight_comparison.tsv` as the compact encoder comparison

## Important Caveats

### 1. Older single-run artifacts are not fully apples-to-apples

The originally committed TF-IDF baseline uses:

- extended attributes
- blocking

The originally committed Siamese run uses:

- only the baseline attribute subset
- no blocking

So those historical artifacts are informative, but not yet a perfectly fair head-to-head model comparison.

Use `models/comparisons/apples_to_apples/` for controlled comparisons.

### 2. There is no evidence of split leakage in the committed `id_disjoint` run

The committed split has:

- `train_val_id_overlap = 0`
- `train_test_id_overlap = 0`
- `val_test_id_overlap = 0`

So the baseline beating the Siamese model is not explained by obvious ID leakage.

### 3. Blocking support is now symmetric in the comparison pipeline

`src/run_comparison_matrix.py` can run both methods with:

- no blocking (`pair_scoring`)
- shared blocking (`blocked_pipeline`)

Both still operate on labeled pair files for evaluation, so this is pair-level benchmark evaluation, not full all-vs-all production candidate generation.

### 4. Controlled matrix results currently favor the Siamese model

From `models/comparisons/apples_to_apples/comparison_summary.json`:

- Siamese outperforms TF-IDF on test F1 in all four matched settings
- strongest run is `siamese + extended + blocked_pipeline` with test F1 `0.9849`

So the current project claim should be based on the controlled matrix, not on older unmatched artifact comparisons.

### 5. The tuned BiLSTM remains the best deployment default

The completed hard-example sweeps show:

- tuned BiLSTM improves both overall and hard-subset performance over the earlier Siamese checkpoint
- CharCNN can push the mined hard subset slightly higher in one setting
- but CharCNN gives up too much overall and easy-case quality to replace BiLSTM as the deployment default
- the latest deployment default now uses `extended_midl` plus `sex_aware_name` pair features because it fixes the surname-expansion review cases while keeping perfect mined hard-negative rejection

Relevant artifacts:

- `models/experiments/hard_weight_tuning_bilstm_blended/tuning_summary.tsv`
- `models/experiments/hard_weight_tuning_extended_midl_sex_aware_bilstm_blended/tuning_summary.tsv`
- `models/experiments/hard_weight_tuning_charcnn_blended/tuning_summary.tsv`
- `models/experiments/encoder_hard_weight_comparison.tsv`

## Recommended Next Review Questions

1. Which deployment threshold should be used for review-vs-auto-merge behavior?
2. Where do tuned BiLSTM and TF-IDF disagree most on real clerical edge cases?
3. Should volatile fields like `race_desc` and `ethnic_desc` be down-weighted or removed?
4. Should hard-example mining be expanded with manually reviewed disagreement cases?
