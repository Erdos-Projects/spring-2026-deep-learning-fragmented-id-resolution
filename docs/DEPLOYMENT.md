# Deployment Guide

This branch adds a simple serving layer around the trained models so the project can be demonstrated as a product.

## Supported product flows

1. Upload a CSV/TSV dataset and ask the service to find likely duplicate pairs and duplicate clusters.
2. Upload a dataset once, then submit one new record at a time and ask whether it already exists in the loaded dataset.

The API supports the saved models already present in the repo:

- `siamese_bilstm`
- `siamese_charcnn`
- `tfidf`

By default, the service uses:

- TF-IDF from the saved blocked-pipeline extended-attribute baseline artifacts
- the tuned BiLSTM Siamese checkpoint from:
  - `models/experiments/hard_weight_tuning_bilstm_blended/both_pos0.50_neg0.25_blended_score/best_model.pth`

That tuned BiLSTM is the current recommended deployment model because it improved both overall F1 and hard-subset F1 over the previous Siamese checkpoint while staying well ahead of the TF-IDF baseline.

We also ran the same hard-example sweep for `siamese_charcnn`:

- `models/experiments/hard_weight_tuning_charcnn_blended/tuning_summary.tsv`
- `models/experiments/encoder_hard_weight_comparison.tsv`

CharCNN remained available in the app as a comparison model, but it did not beat the tuned BiLSTM on the overall deployment tradeoff.

## Local run

Create the environment if needed:

```powershell
conda env create -f environment.yml
conda activate fragmented-id
```

Start the API:

```powershell
uvicorn src.api:app --reload --host 0.0.0.0 --port 8000
```

Open the user-facing app:

- `http://127.0.0.1:8000/`

Open the interactive docs:

- `http://127.0.0.1:8000/docs`

## Load a dataset

Upload a local TSV/CSV:

```powershell
curl -X POST "http://127.0.0.1:8000/dataset/upload" `
  -H "accept: application/json" `
  -H "Content-Type: multipart/form-data" `
  -F "file=@data/processed/ncvoters_prepared.tsv"
```

Or load a server-side file path:

```powershell
curl -X POST "http://127.0.0.1:8000/dataset/load" `
  -H "Content-Type: application/json" `
  -d "{\"path\":\"data/processed/ncvoters_prepared.tsv\"}"
```

Check the currently loaded dataset:

```powershell
curl "http://127.0.0.1:8000/dataset/summary"
```

## Find duplicates inside the loaded dataset

This runs candidate generation through blocking and then scores candidate pairs with the selected model.

```powershell
curl -X POST "http://127.0.0.1:8000/duplicates/find" `
  -H "Content-Type: application/json" `
  -d "{
    \"model_name\": \"siamese_bilstm\",
    \"blocking_keys\": [\"first_name\", \"age\"],
    \"blocking_mode\": \"any\",
    \"top_k\": 25
  }"
```

The response includes:

- `candidate_pair_count`
- `predicted_duplicate_pair_count`
- `duplicate_pairs`
- `duplicate_clusters`

For full-database search, the app defaults to:

- blocking keys: `first_name`, `last_name`, `zip_code`
- blocking mode: `all`

Those defaults are intentionally stricter than the model training-time blocking because they keep candidate generation manageable on the full NCVoters table.

## Check one new entry against the loaded dataset

This is the clerk-facing flow.

```powershell
curl -X POST "http://127.0.0.1:8000/duplicates/check-entry" `
  -H "Content-Type: application/json" `
  -d "{
    \"model_name\": \"siamese_bilstm\",
    \"entry\": {
      \"first_name\": \"john\",
      \"last_name\": \"smith\",
      \"house_num\": \"12\",
      \"street_name\": \"oak street\",
      \"zip_code\": \"27514\",
      \"age\": \"44\",
      \"sex\": \"m\",
      \"race_desc\": \"white\",
      \"ethnic_desc\": \"not hispanic or latino\"
    },
    \"blocking_keys\": [\"first_name\", \"age\"],
    \"blocking_mode\": \"any\",
    \"top_k\": 10
  }"
```

The response includes:

- `duplicate_exists`
- `candidate_count`
- `matches`

Each match reports the model score, duplicate decision, and the matched existing record excerpt.

## Runtime behavior

- Uploaded datasets are normalized with the same lowercase / special-character cleanup used in preprocessing.
- If the dataset has no `id` column, runtime IDs are generated automatically.
- Missing model attributes are added as empty columns so the scorer can still run.
- Blocking is shared across all models at inference time.
- TF-IDF is re-fit on the currently loaded dataset each time a new dataset is loaded.
- Siamese models reuse the saved vocabulary and checkpoint threshold from training.

## Environment variables

- `FRAGMENTED_ID_RUNTIME_DIR`
  - where runtime dataset state is written
- `FRAGMENTED_ID_DEFAULT_DATASET`
  - optional path to auto-load on startup

Example:

```powershell
$env:FRAGMENTED_ID_DEFAULT_DATASET = "data/processed/ncvoters_prepared.tsv"
uvicorn src.api:app --host 0.0.0.0 --port 8000
```

## Docker

Build:

```powershell
docker build -t fragmented-id-api .
```

Run:

```powershell
docker run -p 8000:8000 fragmented-id-api
```

## Practical caveats

- This is a demo-serving layer, not a production persistence stack.
- The current runtime dataset is kept in memory and overwritten when a new dataset is loaded.
- Full all-pairs search can explode combinatorially; blocking is strongly recommended.
- Candidate generation ignores blank blocking values to avoid degenerate giant blocks.
