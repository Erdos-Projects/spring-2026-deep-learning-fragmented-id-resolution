# Repository Walkthrough

## Purpose

This repository studies duplicate detection / entity resolution on the NCVoters benchmark.

The core task is:

- input: two voter records
- output: predict whether they refer to the same real person

The project currently contains:

1. raw-data preprocessing
2. exploratory data analysis (attribute-level and pair-level)
3. a reproducible Siamese LSTM pipeline
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

- deep model: Siamese BiLSTM classifier
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

- merges pair IDs back to the prepared record table
- computes Levenshtein distances for names
- prints example duplicate and non-duplicate pairs
- reports basic overlap counts

Use when:

- inspecting how hard the matching task is
- sanity-checking pair files

## `scripts/extended_eda.py`

Role:

- more strategic EDA for modeling and blocking

What it does:

- evaluates blocking recall for candidate keys
- summarizes how attributes change across duplicate and non-duplicate pairs
- estimates serialized record lengths

Use when:

- selecting blocking keys
- choosing attributes
- estimating input length constraints

## `scripts/print_names.py`

Role:

- one-off debugging helper

What it does:

- compares two hardcoded voter IDs field-by-field

Notes:

- this is not part of the production workflow
- it appears to contain duplicated logic and should be treated as a scratch script

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
- bidirectional LSTM encoder
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
- model: character-level Siamese BiLSTM

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

## Recommended Next Review Questions

1. Which operating point should be used for deployment: max F1, precision-targeted, or recall-targeted?
2. Where do Siamese and TF-IDF disagree most (hard positives vs hard negatives)?
3. Should we freeze blocking as `first_name OR age`, or evaluate alternative block keys for robustness?
4. Should we tune Siamese hyperparameters further now that extended-attribute runs are strong?
5. Do we want to keep `scripts/print_names.py` as-is, or clean/remove it?
