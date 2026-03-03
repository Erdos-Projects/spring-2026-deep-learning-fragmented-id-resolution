# NCVoters Duplicate Detection (Siamese LSTM Baseline)

This repository trains and evaluates a supervised duplicate-record classifier on the NCVoters benchmark.

## Project scope

- Input data:
  - `data/raw/ncvoters.tsv`
  - `data/raw/ncvoters_DPL.tsv` (duplicate pairs, label=1)
  - `data/raw/ncvoters_NDPL.tsv` (non-duplicate pairs, label=0)
- Prepared table: `data/processed/ncvoters_prepared.tsv`
- Model: character-level Siamese BiLSTM + MLP classifier
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
- Compare Precision / Recall / F1 / PR-AUC on the test split
- Inspect hard positives and hard negatives where TF-IDF fails but Siamese succeeds

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
