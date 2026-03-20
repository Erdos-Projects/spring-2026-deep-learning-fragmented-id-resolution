# NCVoters Duplicate Detection (Siamese + TF-IDF Baselines)

This repository trains and evaluates a supervised duplicate-record classifier on the NCVoters benchmark.

## Project scope

- Input data:
  - `data/raw/ncvoters.tsv`
  - `data/raw/ncvoters_DPL.tsv` (duplicate pairs, label=1)
  - `data/raw/ncvoters_NDPL.tsv` (non-duplicate pairs, label=0)
- Prepared table: `data/processed/ncvoters_prepared.tsv`
- Model: character-level Siamese encoder (`bilstm` or `charcnn`) + MLP classifier
- Task: binary classification of record pairs (`duplicate` vs `non-duplicate`)

## Current dataset footprint (in this repo)

- Prepared records: 14,183
- Duplicate pairs (DPL): 9,819
- Non-duplicate pairs (NDPL): 98,142
- Class ratio (NDPL:DPL): 10:1

## Reproducibility defaults

- Default random seed: `42`
- Default split strategy: `id_disjoint`
- Split artifacts are deterministic for a fixed seed and saved under `data/processed/splits/`

## Environment

```bash
conda env create -f environment.yml
conda activate fragmented-id
```

## 1) Preprocess + attribute EDA

```bash
python scripts/ncvoters_preprocess_and_eda.py ^
  --input data/raw/ncvoters.tsv ^
  --prepared-output data/processed/ncvoters_prepared.tsv ^
  --stats-output data/processed/ncvoters_attribute_stats.tsv ^
  --keep-only-config-columns
```

Optional pair-level EDA:

```bash
python scripts/pair_eda.py
python scripts/extended_eda.py
```

## 2) Train (with split artifact generation)

Default training run (id-disjoint split + early stopping + threshold selection on validation):

```bash
python src/train.py ^
  --data-path data/processed/ncvoters_prepared.tsv ^
  --dpl-path data/raw/ncvoters_DPL.tsv ^
  --ndpl-path data/raw/ncvoters_NDPL.tsv ^
  --split-strategy id_disjoint ^
  --seed 42 ^
  --batch-size 64 ^
  --epochs 15 ^
  --lr 0.001 ^
  --max-len 80 ^
  --attribute-set baseline ^
  --use-weighted-loss ^
  --monitor-metric pr_auc ^
  --threshold-mode f1 ^
  --run-dir models/runs/baseline_id_disjoint
```

Use extended attributes:

```bash
python src/train.py --attribute-set extended --run-dir models/runs/extended_id_disjoint
```

Use CharCNN encoder instead of BiLSTM:

```bash
python src/train.py ^
  --encoder charcnn ^
  --cnn-channels 64 ^
  --cnn-kernel-sizes 3,4,5 ^
  --classifier-hidden-dim 64 ^
  --run-dir models/runs/charcnn_id_disjoint
```

Use shared blocking for the Siamese pipeline:

```bash
python src/train.py ^
  --split-path data/processed/splits/id_disjoint_seed42.tsv ^
  --attribute-set extended ^
  --max-len 128 ^
  --blocking-keys first_name,age ^
  --blocking-mode any ^
  --run-dir models/runs/extended_id_disjoint_blocked
```

Run both baseline and extended as an ablation and keep the best by monitor metric:

```bash
python src/run_ablations.py ^
  --split-strategy id_disjoint ^
  --seed 42 ^
  --monitor-metric pr_auc ^
  --run-root models/runs/ablations
```

Use pair-random split (baseline comparison):

```bash
python src/train.py --split-strategy pair_random --run-dir models/runs/pair_random_baseline
```

### Train outputs

`--run-dir` contains:

- `best_model.pth` (checkpoint with model weights + vocab + config + selected threshold)
- `metrics_history.tsv` (epoch-by-epoch train/val metrics)
- `best_metrics.json` (best epoch summary and test metrics)
- `val_threshold_sweep.tsv` (validation threshold sweep)
- `test_confusion_matrix.tsv`
- `split_summary.json`

Split artifacts:

- `data/processed/splits/{strategy}_seed{seed}.tsv` unless overridden
- Columns: `pair_id, id1, id2, label, split`

## 3) Evaluate checkpoint on a split artifact

Evaluate the trained checkpoint on test split using the checkpoint threshold:

```bash
python src/evaluate.py ^
  --model-path models/runs/baseline_id_disjoint/best_model.pth ^
  --split-path data/processed/splits/id_disjoint_seed42.tsv ^
  --split-name test ^
  --threshold-mode checkpoint ^
  --output-dir models/eval ^
  --output-prefix baseline
```

Tune threshold directly on evaluated split (for analysis only):

```bash
python src/evaluate.py ^
  --model-path models/runs/baseline_id_disjoint/best_model.pth ^
  --split-path data/processed/splits/id_disjoint_seed42.tsv ^
  --split-name val ^
  --threshold-mode f1 ^
  --output-dir models/eval ^
  --output-prefix baseline_val_tuned
```

### Evaluation outputs

- `{prefix}_{split}_metrics.json`
- `{prefix}_{split}_metrics.tsv`
- `{prefix}_{split}_confusion_matrix.tsv`
- `{prefix}_{split}_threshold_sweep.tsv` (when threshold tuning is enabled)

Metrics reported:

- Accuracy
- Precision
- Recall
- F1
- ROC-AUC
- PR-AUC
- Confusion matrix (`tn`, `fp`, `fn`, `tp`)

## TF-IDF Distance Baseline

This is the non-deep baseline for fair comparison against the Siamese model:

1. Serialize each record independently
2. Embed records with TF-IDF n-grams
3. Optionally apply blocking on candidate pairs
4. Compute cosine distance between record embeddings
5. Predict duplicate when distance is at or below a validation-tuned threshold

Run it on the same split artifact used by the Siamese model:

```bash
python src/baseline_tfidf.py ^
  --data-path data/processed/ncvoters_prepared.tsv ^
  --dpl-path data/raw/ncvoters_DPL.tsv ^
  --ndpl-path data/raw/ncvoters_NDPL.tsv ^
  --split-path data/processed/splits/id_disjoint_seed42.tsv ^
  --attribute-set extended ^
  --analyzer char ^
  --ngram-min 2 ^
  --ngram-max 4 ^
  --blocking-keys first_name,age ^
  --blocking-mode any ^
  --threshold-selection f1 ^
  --run-dir models/baselines/tfidf_id_disjoint
```

Disable blocking if you want a pure embedding-distance baseline:

```bash
python src/baseline_tfidf.py --split-path data/processed/splits/id_disjoint_seed42.tsv --blocking-keys none
```

Baseline outputs:

- `baseline_metrics.json`
- `baseline_metrics.tsv`
- `val_distance_threshold_sweep.tsv`
- `test_confusion_matrix.tsv`

Suggested comparison:

- Run the TF-IDF baseline and Siamese model on the same `id_disjoint` split
- Compare both:
  - pure pair scoring (`--blocking-keys none`)
  - blocked end-to-end pipeline (`--blocking-keys first_name,age`)
- Compare Precision / Recall / F1 / PR-AUC on the test split
- Inspect hard positives and hard negatives where TF-IDF fails but Siamese succeeds

## Hard-Example Mining and Weighted Siamese Training

This repo now supports mining hard positives and hard negatives directly from the real labeled pair files, then using those mined examples to bias Siamese training.

Why:

- `DPL` already contains true positives that are unusually difficult
- `NDPL` already contains true negatives that are unusually similar
- weighting those real hard cases is more defensible than inventing synthetic labels first

Mine hard examples:

```bash
python src/mine_hard_examples.py ^
  --data-path data/processed/ncvoters_prepared.tsv ^
  --dpl-path data/raw/ncvoters_DPL.tsv ^
  --ndpl-path data/raw/ncvoters_NDPL.tsv ^
  --output-dir data/processed/hard_examples_real ^
  --output-prefix labeled_hard_examples_real
```

Train a hard-weighted Siamese model:

```bash
python src/train.py ^
  --data-path data/processed/ncvoters_prepared.tsv ^
  --dpl-path data/raw/ncvoters_DPL.tsv ^
  --ndpl-path data/raw/ncvoters_NDPL.tsv ^
  --split-path data/processed/splits/id_disjoint_seed42.tsv ^
  --attribute-set extended ^
  --blocking-keys first_name,age ^
  --blocking-mode any ^
  --max-len 128 ^
  --hard-example-path data/processed/hard_examples_real/labeled_hard_examples_real.tsv ^
  --hard-weighting both ^
  --hard-positive-weight-scale 0.50 ^
  --hard-negative-weight-scale 0.25 ^
  --monitor-metric blended_score ^
  --blended-overall-weight 0.5 ^
  --blended-hard-positive-weight 0.25 ^
  --blended-hard-negative-weight 0.25 ^
  --run-dir models/experiments/hard_weight_tuning_bilstm_blended/both_pos0.50_neg0.25_blended_score
```

Run the small tuning sweep:

```bash
python src/tune_hard_weighting.py ^
  --hard-example-path data/processed/hard_examples_real/labeled_hard_examples_real.tsv ^
  --attribute-set extended ^
  --blocking-keys first_name,age ^
  --blocking-mode any ^
  --encoder bilstm ^
  --max-len 128 ^
  --epochs 15 ^
  --batch-size 64 ^
  --lr 0.001 ^
  --hard-weighting-modes both ^
  --hard-positive-scales 0.25,0.5,1.0 ^
  --hard-negative-scales 0.25,0.5,1.0 ^
  --monitor-metric blended_score ^
  --run-root models/experiments/hard_weight_tuning_bilstm_blended
```

Run the same sweep for CharCNN:

```bash
python src/tune_hard_weighting.py ^
  --hard-example-path data/processed/hard_examples_real/labeled_hard_examples_real.tsv ^
  --attribute-set extended ^
  --blocking-keys first_name,age ^
  --blocking-mode any ^
  --encoder charcnn ^
  --max-len 128 ^
  --epochs 15 ^
  --batch-size 64 ^
  --lr 0.001 ^
  --hard-weighting-modes both ^
  --hard-positive-scales 0.25,0.5,1.0 ^
  --hard-negative-scales 0.25,0.5,1.0 ^
  --monitor-metric blended_score ^
  --run-root models/experiments/hard_weight_tuning_charcnn_blended
```

## Apples-to-Apples Comparison Matrix

To compare both model families fairly across:

- baseline vs extended attributes
- unblocked pair scoring vs blocked pipeline evaluation

run:

```bash
python src/run_comparison_matrix.py ^
  --split-path data/processed/splits/id_disjoint_seed42.tsv ^
  --siamese-encoders bilstm,charcnn ^
  --blocking-keys first_name,age ^
  --blocking-mode any ^
  --baseline-max-len 80 ^
  --extended-max-len 128 ^
  --run-root models/comparisons/apples_to_apples
```

This produces:

- `models/comparisons/apples_to_apples/comparison_summary.tsv`
- `models/comparisons/apples_to_apples/comparison_summary.json`

Scenarios:

- `pair_scoring`: no blocking for either model
- `blocked_pipeline`: shared blocking for both models

When `--siamese-encoders bilstm,charcnn` is used, outputs are written under:

- `models/comparisons/.../siamese_bilstm/...`
- `models/comparisons/.../siamese_charcnn/...`

## Human-Readable Comparison Summary

To convert `comparison_summary.json` into an easy-to-read report:

```bash
python src/summarize_comparison.py ^
  --input-path models/comparisons/apples_to_apples/comparison_summary.json ^
  --output-report models/comparisons/apples_to_apples/comparison_summary_report.txt ^
  --output-winners-tsv models/comparisons/apples_to_apples/comparison_winners.tsv
```

This prints:

- overall best run
- winner per scenario and attribute set
- average metrics by method
- metric deltas versus TF-IDF (if TF-IDF is present)

## Latest Comparison Snapshot (March 4, 2026)

Source:

- `models/comparisons/apples_to_apples/comparison_summary.json`

Key test-set F1 results:

- `pair_scoring + baseline attrs`
  - Siamese: `0.9492`
  - TF-IDF: `0.9156`
- `pair_scoring + extended attrs`
  - Siamese: `0.9806`
  - TF-IDF: `0.9176`
- `blocked_pipeline + baseline attrs`
  - Siamese: `0.9407`
  - TF-IDF: `0.9313`
- `blocked_pipeline + extended attrs`
  - Siamese: `0.9849`
  - TF-IDF: `0.9630`

Blocking summary for `first_name OR age` on the same test split:

- positive pass rate: `1.0`
- negative pass rate: `0.03495`
- candidate pairs: `303 / 2291`

## Latest Tuned Deployment Snapshot (March 18, 2026)

Hard-example mining artifact:

- `data/processed/hard_examples_real/labeled_hard_examples_real.tsv`

Best tuned deployment checkpoint:

- `models/experiments/hard_weight_tuning_extended_midl_sex_aware_bilstm_blended/both_pos0.50_neg0.75_blended_score/best_model.pth`

This tuned BiLSTM with `extended_midl` and `sex_aware_name` pair features is the current default Siamese model used by the deployment app.

Encoder comparison summary:

- `models/experiments/encoder_hard_weight_comparison.tsv`

Test-set comparison on the blocked extended-attribute setting:

- TF-IDF baseline
  - F1: `0.9630`
  - PR-AUC: `0.9945`
  - hard-subset F1: `0.8793`
  - hard-positive recall: `0.8361`
  - hard-negative rejection: `0.9130`
- previous Siamese BiLSTM
  - F1: `0.9849`
  - PR-AUC: `0.9994`
  - hard-subset F1: `0.9677`
  - hard-positive recall: `0.9836`
  - hard-negative rejection: `0.9348`
- tuned Siamese BiLSTM (`hard_positive_weight_scale=0.50`, `hard_negative_weight_scale=0.25`)
  - F1: `0.9891`
  - PR-AUC: `0.9994`
  - hard-subset F1: `0.9833`
  - easy-subset F1: `0.9912`
  - hard-positive recall: `0.9672`
  - hard-negative rejection: `1.0000`
- best overall tuned Siamese CharCNN (`hard_positive_weight_scale=1.00`, `hard_negative_weight_scale=1.00`)
  - F1: `0.9766`
  - PR-AUC: `0.9946`
  - hard-subset F1: `0.9683`
  - easy-subset F1: `0.9797`
- best hard-subset tuned Siamese CharCNN (`hard_positive_weight_scale=0.50`, `hard_negative_weight_scale=0.50`)
  - F1: `0.9662`
  - PR-AUC: `0.9960`
  - hard-subset F1: `0.9913`
  - easy-subset F1: `0.9385`

Interpretation:

- TF-IDF remains the operational non-deep benchmark.
- The tuned Siamese model now improves both overall F1 and hard-subset F1 relative to the previous Siamese checkpoint.
- CharCNN can push the mined hard subset slightly higher in one setting, but it gives up too much overall and easy-case quality.
- The chosen BiLSTM deployment checkpoint now uses `midl_name` plus sex-aware pair features and is the best tradeoff we found between easy-case quality, difficult-case behavior, and the real disagreement cases surfaced in the UI.

Current deployment-default metrics:

- F1: `0.9935`
- PR-AUC: `0.9998`
- hard-subset F1: `0.9833`
- easy-subset F1: `0.9971`
- hard-positive recall: `0.9672`
- hard-negative rejection: `1.0000`

## Middle-Name and Sex-Aware Follow-Up (March 18, 2026)

We also tested a follow-up feature expansion motivated by review cases in the deployment UI:

- `extended_midl`
  - adds `midl_name` while keeping `race_desc` and `ethnic_desc`
- `sex_aware_name`
  - adds structured pair features such as:
    - same middle name
    - both male / both female
    - male same-first-name but different-last-name
    - female same-first-name but different-last-name

Artifacts:

- `models/experiments/extended_midl_bilstm_blended/best_metrics.json`
- `models/experiments/extended_midl_sex_aware_bilstm_blended/best_metrics.json`

Results:

- current deployed tuned BiLSTM
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

- `midl_name` and the sex-aware pair features fix several obvious duplicate false negatives involving surname expansion.
- The sex-aware model also suppresses some suspicious same-first-name male collisions.
- But the current tuned deployment checkpoint still has the better hard-negative safety margin, so it remains the default deployment model for now.

## Tests

```bash
pytest -q
```

Coverage includes:

- Pair integrity checks (required columns, missing IDs, duplicate canonical pair detection)
- Split correctness (id-disjoint leakage checks and ratio checks)
- Model I/O shape checks
- Metric correctness vs scikit-learn
- End-to-end smoke run (1 epoch train + evaluate, artifact assertions)

## Deployment API

This repo also includes a lightweight FastAPI serving layer for demoing the project as a product.

Supported flows:

1. Upload a CSV/TSV dataset and find likely duplicates inside it.
2. Load a dataset once and check a new incoming record against it.

The user-facing app now behaves like a product flow rather than an experiment dashboard:

- Step 1: load a dataset
- Step 2: choose a task
- Step 3A: run a full duplicate scan
- Step 3B: check one incoming record
- the app uses the deployed Siamese model by default and hides backend-facing model controls from the main UI

The review-oriented result panels include:

- disagreement sections are labeled as `Review recommended`
- disagreement cards show exact matching fields and exact differing fields
- record previews include `midl_name` when present so surname-expansion and middle-name cases are visible in the UI
- preview sections are capped at `10` rows each for presentation clarity
- export buttons provide a duplicate CSV and a human-review CSV for the current run
- a persisted human-review queue lets a reviewer mark pairs as accepted duplicate, rejected duplicate, or uncertain
- reviewers can add optional notes and switch the queue between `Pending only` and `Show all`
- review decisions are stored in a SQLite database under the runtime directory and surfaced back in the UI

Start the API:

```bash
uvicorn src.api:app --reload --host 0.0.0.0 --port 8000
```

User-facing web app:

- `http://127.0.0.1:8000/`

Interactive docs:

- `http://127.0.0.1:8000/docs`

Core endpoints:

- `POST /dataset/upload`
- `POST /dataset/load`
- `GET /dataset/summary`
- `DELETE /dataset`
- `POST /duplicates/find`
- `POST /duplicates/check-entry`

Detailed usage examples:

- `docs/DEPLOYMENT.md`

Portable demo bundle:

- `demo/`
- this is a self-contained snapshot of the product UI, API, model artifacts, and default dataset
- it is meant to be copied into `main` without merging the entire `Deployment` branch structure
- refresh it after deployment/UI changes with:

```bash
python scripts/sync_demo_bundle.py
```

One-command local demo:

```bash
docker compose up --build
```
