# **Fragmented ID Resolution:** _Using Deep Learning to Match Noisy Identity Records_

This repository is for the Fragmented ID Resolution project as part of the fulfillment of [The Erdős Institute](https://www.erdosinstitute.org/) Deep Learning Bootcamp for Spring 2026.

**Team members:** [Noimot Bakare Ayoub](https://github.com/NoimotBAyoub), [Pedro Fontanarrosa](https://github.com/Fontanapink), [Arpith Shanbhag](https://github.com/anunknownpleasure), [Dharineesh Somisetty](https://github.com/Dharineesh-Somisetty)

**Presentation Deliverables**
- [Presentation slides PDF](presentation/Fragmented_ID_Resolution_Erdos_SP26.pdf)
- [Executive Summary](presentation/Executive_Summary-Fragmented_ID_Resolution.pdf)

# Contents

1. [Introduction](#introduction)
2. [Dataset](#dataset)
3. [Methods and Models](#methods-and-models)
4. [Results](#results)
5. [Key Performance Indicators](#key-performance-indicators-kpis)
6. [Challenges](#challenges)
7. [Deployment](#deployment)
8. [Files](#files)


## Introduction

Identity data in real-world systems are noisy and fragmented. Small inconsistencies — typos, nicknames, surname changes, ZIP variations — can cause a single person to appear as multiple records, creating major risks across industries:

- **Financial Services:** Vulnerability to synthetic identity fraud and account takeover.
- **Healthcare:** Patient mismatches, duplicated charts, and potential clinical risk.
- **Government & Public Sector:** Duplicate citizen records, complicating benefits, taxation, and verification.

Traditional matching methods break down in these high-noise environments, motivating the need for a more robust approach. This project proposes a deep learning framework for fragmented identity resolution: a **Siamese neural network** that learns an adaptive similarity function over identity record pairs, significantly outperforming both rule-based and embedding-only baselines — especially on hard cases where duplicates vary across multiple fields simultaneously.

Our best model achieves **99.4% F1 overall** and **98.3% F1 on hard cases**, with a **+25 point improvement** over the deep learning baseline on hard-case F1 (0.71 → 0.96).


## Dataset

We utilize the [Hasso Plattner Institute (HPI)](https://hpi.de/naumann/projects/repeatability/datasets/ncvoters-dataset.html) North Carolina State Board of Election (NCSBE) voter registration dataset, a standard benchmark for duplicate detection research.

**Dataset summary:**

| Statistic | Count |
|:---|:---:|
| Voter records | 14,183 |
| Duplicate pairs (labeled) | 9,891 |
| Non-duplicate pairs (labeled) | 98,142 |
| Class ratio (non-dup : dup) | ~10 : 1 |

Each voter record contains attributes including first name, last name, middle name, age, sex, race, ethnicity, house number, street name, street type, and ZIP code.

**Data preparation:**

- Augmented data to reflect real-world variations (typos, nicknames, surname changes, ZIP variations)
- Constructed **hard negatives** to capture confusable identities (different people who look similar)
- Group-based **entity-disjoint train / validation / test split** to prevent data leakage during evaluation


## Methods and Models

### Why Simple Thresholding Fails

1. **Duplicates vary across fields** — the same pair can be "close" in one attribute and "far" in another (e.g., same address but different last name due to marriage)
2. **Similarity scores overlap** — cosine similarity distributions for duplicates and non-duplicates overlap significantly, causing single-threshold approaches to misclassify borderline cases

### Models

We developed and compared three model families:

<div align="center">

| Model | Approach | Strengths | Limitations |
|:---|:---|:---|:---|
| **Model 1:** Logistic Regression + XGBoost | Manually engineered features (string distances, exact matches) | Solves typos, nicknames, ZIP variance | Struggles with last name differences |
| **Model 2:** CNN Embedding + Cosine Similarity | Converts records into embeddings, compares via cosine similarity | Captures some variations | Poor separation on hard cases |
| **Model 3:** Siamese Network + Learned Weights | Shared encoder learns both agreement and difference signals; adaptive similarity function | Resolves fragmentation across variations; distinguishes similar but different individuals | Requires more training data and compute |

</div>

### Siamese Network Architecture

The Siamese network is the core contribution of this project:

1. Both records are encoded through a **shared network** (BiLSTM or CharCNN encoder)
2. The model learns a **similarity function** from how record embeddings agree and differ
3. An **adaptive similarity function** improves hard-case detection by assigning higher weight to meaningful differences
4. The loss function uses binary cross-entropy: `P(Similar) * log(similarity) + P(Different) * log(1 - similarity)`

**Key design choices:**
- Character-level encoding to handle typos and abbreviations
- Absolute difference between pair vectors as input to the classifier head
- Weighted loss to handle 10:1 class imbalance
- Hard-example mining and weighting to focus learning on difficult cases
- Entity-disjoint splits to prevent leakage

### Hard-Example Mining

We mine hard positives (true duplicates that look very different) and hard negatives (different people who look very similar) directly from the labeled data. These hard examples are used as training weights to bias the model toward learning difficult cases without distorting the original label distribution.

Current hard-example counts:
- Hard positives: 1,950
- Hard negatives: 1,794


## Results

The Siamese model significantly improves performance on hard cases compared to both the rule-based and embedding-only baselines.

<div align="center">

**Table 1:** Test-set performance across all models

| Model | Precision | | Recall | | F1 | |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| | **Overall** | **Hard** | **Overall** | **Hard** | **Overall** | **Hard** |
| XGBoost | .97 | .86 | .97 | .82 | .97 | .84 |
| Embedding + Cosine (DL Baseline) | .94 | .65 | .91 | .78 | .92 | .71 |
| **Siamese + Weights** | **.99** | **.93** | **.99** | **.99** | **.99** | **.96** |

</div>

**Hard-case F1 improvement: +25 points** (0.71 → 0.96) over the deep learning baseline.

### Detailed Comparison on Blocked Extended-Attribute Setting

<div align="center">

**Table 2:** Test-set metrics with blocking and extended attributes

| Model | F1 | PR-AUC | Hard-subset F1 | Hard-positive Recall | Hard-negative Rejection |
|:---:|:---:|:---:|:---:|:---:|:---:|
| TF-IDF baseline | 0.9630 | 0.9945 | 0.8793 | 0.8361 | 0.9130 |
| Siamese BiLSTM (initial) | 0.9849 | 0.9994 | 0.9677 | 0.9836 | 0.9348 |
| **Siamese BiLSTM (tuned)** | **0.9891** | **0.9994** | **0.9833** | **0.9672** | **1.0000** |

</div>

The tuned Siamese model achieves perfect rejection on mined hard negatives while maintaining high overall quality.

### Deployment-Default Model Metrics

The final deployed model incorporates middle-name features and sex-aware pair features:

<div align="center">

| Metric | Result |
|:---|:---:|
| F1 | 0.9935 |
| PR-AUC | 0.9998 |
| Hard-subset F1 | 0.9833 |
| Easy-subset F1 | 0.9971 |
| Hard-positive Recall | 0.9672 |
| Hard-negative Rejection | 1.0000 |

</div>


## Key Performance Indicators (KPIs)

### Model Performance KPIs

Our primary KPIs focus on performance where it matters most — on hard cases that traditional methods miss:

| KPI | Target | Achieved |
|:---|:---:|:---:|
| Overall F1 | > 0.95 | **0.9935** |
| Hard-case F1 | > 0.90 | **0.9833** |
| Hard-positive Recall | > 0.95 | **0.9672** |
| Hard-negative Rejection | > 0.95 | **1.0000** |
| PR-AUC | > 0.99 | **0.9998** |

### Business KPIs

The blocking stage reduces candidate pair comparisons by ~87% while maintaining 100% positive pass-through:

| Metric | Result | Interpretation |
|:---|:---:|:---|
| Positive pass rate | 100% | No true duplicates missed by blocking |
| Negative pass rate | 3.5% | 96.5% of non-duplicate pairs eliminated early |
| Candidate reduction | 303 / 2,291 | ~87% fewer comparisons needed |


## Challenges

Developing a fragmented identity resolution model presents several challenges:

1. **Feature Variation Across Fields:** Duplicates can match on some attributes (e.g., address) while differing on others (e.g., last name after marriage). No single similarity threshold captures all cases.

2. **Hard Negatives:** Different individuals who share similar attributes (same first name, same street, close age) create confusable pairs that push false positive rates up.

3. **Class Imbalance:** Non-duplicate pairs outnumber duplicates ~10:1, making it difficult for models to learn the minority class.

4. **Surname Expansion and Nickname Variation:** Real-world name changes (e.g., "Liz Miller-Davis" vs. "Elisabeth Millar") require the model to look beyond character-level similarity.

5. **Entity-Disjoint Evaluation:** Standard random splits can leak entity information across train/test. We enforce entity-disjoint splits to get honest generalization estimates.


## Deployment

### LinkID — Identity Matching Simplified

The project includes **LinkID**, a web application for production-style identity resolution:

1. **Upload** voter registration data (CSV/TSV)
2. **Detect and rank** likely duplicates using the deployed Siamese model
3. **Review** high-confidence matches and borderline cases close to the threshold
4. **Export** duplicate lists and human-review queues

**Features:**
- Bulk duplicate scan and single-record clerk check modes
- Disagreement sections labeled as "Review recommended" with exact matching/differing fields shown
- Human-review queue with accept/reject/uncertain decisions and optional notes
- SQLite-backed review persistence
- Export buttons for duplicate CSV and human-review CSV

**Quick start:**

```bash
# Using Docker (recommended)
docker compose up --build

# Or manually
conda env create -f environment.yml
conda activate fragmented-id
uvicorn src.api:app --reload --host 0.0.0.0 --port 8000
```

- Web app: `http://127.0.0.1:8000/`
- API docs: `http://127.0.0.1:8000/docs`

See [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) for detailed API usage.


## Files

### Core Source Code

- [`src/model.py`](src/model.py): Siamese network architecture (BiLSTM and CharCNN encoders)
- [`src/train.py`](src/train.py): Training pipeline with hard-example weighting and early stopping
- [`src/evaluate.py`](src/evaluate.py): Evaluation and threshold tuning
- [`src/baseline_tfidf.py`](src/baseline_tfidf.py): TF-IDF distance baseline
- [`src/blocking.py`](src/blocking.py): Shared blocking for candidate generation
- [`src/hard_examples.py`](src/hard_examples.py): Hard-example mining logic
- [`src/api.py`](src/api.py): FastAPI deployment serving layer
- [`src/deployment.py`](src/deployment.py): Deployment inference pipeline
- [`src/review_store.py`](src/review_store.py): SQLite-backed human review storage

### Scripts

- [`scripts/ncvoters_preprocess_and_eda.py`](scripts/ncvoters_preprocess_and_eda.py): Data preprocessing and attribute EDA
- [`scripts/pair_eda.py`](scripts/pair_eda.py): Pair-level exploratory analysis
- [`scripts/extended_eda.py`](scripts/extended_eda.py): Extended EDA
- [`scripts/sync_demo_bundle.py`](scripts/sync_demo_bundle.py): Sync portable demo bundle

### Configuration and Setup

- [`environment.yml`](environment.yml): Conda environment specification
- [`requirements-deploy.txt`](requirements-deploy.txt): Deployment dependencies
- [`Dockerfile`](Dockerfile): Container build for deployment
- [`docker-compose.yml`](docker-compose.yml): One-command local demo

### Documentation

- [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md): Detailed deployment and API usage guide
- [`docs/methodology.md`](docs/methodology.md): Full project methodology
- [`docs/EDA_INSIGHTS.md`](docs/EDA_INSIGHTS.md): Exploratory data analysis insights
- [`docs/REPO_WALKTHROUGH.md`](docs/REPO_WALKTHROUGH.md): Repository structure walkthrough

### Tests

```bash
pytest -q
```

Coverage includes pair integrity checks, split correctness (entity-disjoint leakage), model I/O shape checks, metric correctness vs. scikit-learn, and end-to-end smoke tests.
