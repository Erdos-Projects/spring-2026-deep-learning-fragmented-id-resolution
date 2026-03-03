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
- optionally applies blocking
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

## Important Caveats

### 1. The saved comparison is not fully apples-to-apples

The committed TF-IDF baseline uses:

- extended attributes
- blocking

The committed Siamese run uses:

- only the baseline attribute subset
- no blocking

So the current comparison is informative, but not yet a perfectly fair head-to-head model comparison.

### 2. There is no evidence of split leakage in the committed `id_disjoint` run

The committed split has:

- `train_val_id_overlap = 0`
- `train_test_id_overlap = 0`
- `val_test_id_overlap = 0`

So the baseline beating the Siamese model is not explained by obvious ID leakage.

### 3. Blocking is not used symmetrically

The current saved TF-IDF baseline does apply blocking during scoring.

The current saved Siamese workflow does not.

Both pipelines still evaluate on the labeled pair files rather than generating candidates from the full dataset, so blocking is acting as a scoring gate inside the baseline, not as a full candidate-generation stage used by both methods.

### 4. The deep model currently lags the baseline

This is a project-level research result, not a code-organization issue.

It likely means:

- the baseline is very strong for this dataset
- the Siamese configuration is still underpowered or mismatched to the task
- the current comparison needs to be made more fair before drawing conclusions

## Recommended Next Review Questions

1. Should the Siamese model be rerun with the same extended attributes as the baseline?
2. Should blocking be removed from the baseline for a pure embedding-distance comparison?
3. Should a candidate-generation block be applied to both methods consistently?
4. Is `max_len=80` limiting the Siamese model too much for extended attributes?
5. Do we want to keep `scripts/print_names.py` as-is, or clean/remove it?

