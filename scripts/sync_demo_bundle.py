from pathlib import Path
import re
import shutil


REPO_ROOT = Path(__file__).resolve().parents[1]
DEMO_ROOT = REPO_ROOT / "demo"


COPY_MAP = {
    REPO_ROOT / "src" / "__init__.py": DEMO_ROOT / "src" / "__init__.py",
    REPO_ROOT / "src" / "api.py": DEMO_ROOT / "src" / "api.py",
    REPO_ROOT / "src" / "data_utils.py": DEMO_ROOT / "src" / "data_utils.py",
    REPO_ROOT / "src" / "dataset.py": DEMO_ROOT / "src" / "dataset.py",
    REPO_ROOT / "src" / "deployment.py": DEMO_ROOT / "src" / "deployment.py",
    REPO_ROOT / "src" / "model.py": DEMO_ROOT / "src" / "model.py",
    REPO_ROOT / "src" / "review_store.py": DEMO_ROOT / "src" / "review_store.py",
    REPO_ROOT / "src" / "ui" / "app.js": DEMO_ROOT / "src" / "ui" / "app.js",
    REPO_ROOT / "src" / "ui" / "index.html": DEMO_ROOT / "src" / "ui" / "index.html",
    REPO_ROOT / "src" / "ui" / "styles.css": DEMO_ROOT / "src" / "ui" / "styles.css",
    REPO_ROOT / "data" / "processed" / "ncvoters_prepared.tsv": DEMO_ROOT / "data" / "processed" / "ncvoters_prepared.tsv",
    REPO_ROOT
    / "models"
    / "experiments"
    / "hard_weight_tuning_extended_midl_sex_aware_bilstm_blended"
    / "both_pos0.50_neg0.75_blended_score"
    / "best_model.pth": DEMO_ROOT / "models" / "siamese_bilstm" / "best_model.pth",
    REPO_ROOT
    / "models"
    / "comparisons"
    / "apples_to_apples_with_charcnn"
    / "tfidf"
    / "blocked_pipeline"
    / "extended"
    / "baseline_metrics.json": DEMO_ROOT / "models" / "tfidf" / "baseline_metrics.json",
}


REQUIREMENTS_TEXT = """fastapi
uvicorn[standard]
python-multipart
pandas
numpy
scikit-learn
torch
"""


DOCKERFILE_TEXT = """FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app

EXPOSE 8000

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
"""


DOCKER_COMPOSE_TEXT = """services:
  app:
    build: .
    ports:
      - "8000:8000"
    environment:
      FRAGMENTED_ID_RUNTIME_DIR: /app/data/runtime
      FRAGMENTED_ID_DEFAULT_DATASET: /app/data/processed/ncvoters_prepared.tsv
    volumes:
      - fragmented_id_runtime:/app/data/runtime
    restart: unless-stopped

volumes:
  fragmented_id_runtime:
"""


DOCKERIGNORE_TEXT = """__pycache__/
*.pyc
data/runtime/
"""


GITIGNORE_TEXT = """data/runtime/
"""


README_TEXT = """# Demo Bundle

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
"""


DEPLOYMENT_DEFAULT_SPECS = """DEFAULT_MODEL_SPECS = {
    "siamese_bilstm": {
        "kind": "siamese",
        "path": "models/siamese_bilstm/best_model.pth",
    },
    "tfidf": {
        "kind": "tfidf",
        "path": "models/tfidf/baseline_metrics.json",
    },
}
"""


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def copy_file(source: Path, destination: Path) -> None:
    if not source.exists():
        raise FileNotFoundError(f"Required demo source file not found: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def patch_demo_deployment() -> None:
    deployment_path = DEMO_ROOT / "src" / "deployment.py"
    text = deployment_path.read_text(encoding="utf-8")
    text = re.sub(
        r"DEFAULT_MODEL_SPECS = \{.*?\n\}\nDEFAULT_COLUMN_ALIASES = \{",
        f"{DEPLOYMENT_DEFAULT_SPECS}\nDEFAULT_COLUMN_ALIASES = {{",
        text,
        flags=re.S,
    )
    deployment_path.write_text(text, encoding="utf-8", newline="\n")


def main() -> None:
    for source, destination in COPY_MAP.items():
        copy_file(source, destination)

    write_text(DEMO_ROOT / "requirements.txt", REQUIREMENTS_TEXT)
    write_text(DEMO_ROOT / "Dockerfile", DOCKERFILE_TEXT)
    write_text(DEMO_ROOT / "docker-compose.yml", DOCKER_COMPOSE_TEXT)
    write_text(DEMO_ROOT / ".dockerignore", DOCKERIGNORE_TEXT)
    write_text(DEMO_ROOT / ".gitignore", GITIGNORE_TEXT)
    write_text(DEMO_ROOT / "README.md", README_TEXT)
    write_text(DEMO_ROOT / "data" / "runtime" / ".gitkeep", "")

    patch_demo_deployment()

    print(f"Demo bundle refreshed at {DEMO_ROOT}")


if __name__ == "__main__":
    main()
