# Deployment Guide

This branch adds a serving layer around the trained models so the project can be demonstrated as a product.

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
  - `models/experiments/hard_weight_tuning_extended_midl_sex_aware_bilstm_blended/both_pos0.50_neg0.75_blended_score/best_model.pth`

That tuned BiLSTM is the current recommended deployment model because it:

- adds `midl_name` to the encoded attribute set
- uses explicit sex-aware pair features for name comparison
- fixes the surname-expansion duplicate cases that were surfacing in the deployment review UI
- preserves the strong hard-negative rejection of the earlier deployment checkpoint

We also ran the same hard-example sweep for `siamese_charcnn`:

- `models/experiments/hard_weight_tuning_charcnn_blended/tuning_summary.tsv`
- `models/experiments/encoder_hard_weight_comparison.tsv`

CharCNN remained available through the API as a comparison model, but it did not beat the tuned BiLSTM on the overall deployment tradeoff.

The user-facing web app is intentionally simpler than the API:

- it defaults to the deployed Siamese product model
- it does not expose model-selection controls to a naive user
- it guides the user through three steps:
  1. load a dataset
  2. choose a task
  3. run either a full duplicate scan or a single-record check

This deployment now also includes a persisted human-review workflow:

- reviewers can mark a near-threshold pair as:
  - accepted duplicate
  - rejected duplicate
  - uncertain
- reviewers can add an optional note explaining the decision
- those decisions are stored in a SQLite database under the runtime directory
- the app shows a review queue, review counters, recent saved decisions, and a `Pending only / Show all` queue filter for the current dataset

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

This runs candidate generation through blocking and then scores candidate pairs with the deployed product model.

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

The UI does not just dump raw pairs anymore. It now emphasizes:

- high-confidence duplicates
- a human-review queue that combines near-threshold accepted and rejected cases
- persisted review decisions with optional notes
- cluster summaries instead of every cluster
- optional `Review recommended` disagreement cases between models

For disagreement cases, each card shows:

- record previews including `midl_name`
- exact matching fields
- exact differing fields

The visible preview sections are capped at `10` rows each for presentation clarity, but the export buttons provide:

- a duplicate CSV for the full predicted duplicate queue from the current run
- a human-review CSV for near-threshold cases on either side of the threshold

Review-decision endpoints:

- `GET /reviews`
- `POST /reviews`

Each saved review decision is stored with:

- canonical pair ids
- dataset source
- decision label
- model score context
- record snapshots for auditability

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

The product UI includes a `midl_name` field in this form because the current deployment checkpoint uses middle name as part of the learned representation.

## Runtime behavior

- Uploaded datasets are normalized with the same lowercase / special-character cleanup used in preprocessing.
- If the dataset has no `id` column, runtime IDs are generated automatically.
- Missing model attributes are added as empty columns so the scorer can still run.
- Blocking is shared across all models at inference time.
- TF-IDF is re-fit on the currently loaded dataset each time a new dataset is loaded.
- Siamese models reuse the saved vocabulary and checkpoint threshold from training.
- Some Siamese checkpoints can also consume explicit pair features from the checkpoint config, such as sex-aware name-comparison signals.

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

For a one-command local demo with persisted runtime state:

```powershell
docker compose up --build
```

This starts the app and:

- exposes it on `http://127.0.0.1:8000/`
- auto-loads `data/processed/ncvoters_prepared.tsv`
- persists the review database and runtime dataset state in a named Docker volume

## Portable demo bundle

This branch now includes a self-contained `demo/` folder.

Why:

- `main` can continue to hold notebooks and exploratory work
- `demo/` acts as the clean showroom version of the product
- the demo can be copied into `main` without merging the entire `Deployment` branch structure

What is inside `demo/`:

- the deployment API
- the product UI
- the SQLite-backed review workflow
- the deployed Siamese checkpoint
- the TF-IDF comparison config used for optional disagreement analysis
- the prepared NCVoters dataset used for default preload
- its own `Dockerfile` and `docker-compose.yml`

To refresh `demo/` from the current Deployment branch state:

```powershell
python scripts/sync_demo_bundle.py
```

Then the demo can be run independently:

```powershell
cd demo
docker compose up --build
```

Or without Docker:

```powershell
cd demo
uvicorn src.api:app --reload --host 0.0.0.0 --port 8000
```

If the UI changes later in `Deployment`, rerun the sync script and update only the `demo/` folder in `main`.

You can still use the plain Docker commands if you prefer:

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
