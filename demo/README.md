# Demo Bundle

This folder is a self-contained product demo bundle for the duplicate-detection UI.

It is designed so the rest of the repository can keep notebooks, experiments, and exploratory code while this folder stays focused on the runnable demo.

## What is inside

- `src/`: API, deployment service, review store, and UI
- `models/`: the deployed Siamese checkpoint plus the TF-IDF comparison config used for optional disagreement analysis
- `data/processed/ncvoters_prepared.tsv`: the default preloaded demo dataset
- `Dockerfile` and `docker-compose.yml`: one-command local demo

## Run locally with Docker

```powershell
cd demo
docker compose up --build
```

Then open:

- `http://127.0.0.1:8000/`

The app will:

- start the web UI
- auto-load the bundled NCVoters prepared dataset
- persist human-review decisions in a Docker volume

## Run locally without Docker

Create an environment with the required packages, then from `demo/` run:

```powershell
uvicorn src.api:app --reload --host 0.0.0.0 --port 8000
```

## Keeping this folder updated

This folder is generated from the current Deployment branch runtime files.

To refresh it after UI or deployment changes:

```powershell
python scripts/sync_demo_bundle.py
```

That command copies the current deployment app, assets, default dataset, and required model artifacts into `demo/`.
