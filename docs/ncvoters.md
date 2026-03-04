# NCVoters Dataset Notes

This document summarizes the NCVoters benchmark files used in this repository.

Source reference:

- https://hpi.de/naumann/projects/repeatability/datasets/ncvoters-dataset.html

## 1. Files Used in This Repo

- `data/raw/ncvoters.tsv`
  - main record table
  - each row is one voter record
- `data/raw/ncvoters_DPL.tsv`
  - labeled duplicate pairs (`label=1`)
  - each row is a pair `(id1, id2)`
- `data/raw/ncvoters_NDPL.tsv`
  - labeled non-duplicate pairs (`label=0`)
  - each row is a pair `(id1, id2)`

Prepared artifact:

- `data/processed/ncvoters_prepared.tsv`
  - normalized version of `ncvoters.tsv`

## 2. Current Footprint (in this repo)

- records in prepared table: `14,183`
- duplicate pairs (DPL): `9,819`
- non-duplicate pairs (NDPL): `98,142`
- NDPL:DPL ratio: `10:1`

Important clarification:

- DPL and NDPL are pair files, not standalone record-object counts.

## 3. How the Dataset Is Used Here

This project is a supervised pair-classification benchmark:

- input: two records
- output: duplicate vs non-duplicate

The same labeled pair data is used for:

- Siamese BiLSTM training/evaluation (`src/train.py`, `src/evaluate.py`)
- TF-IDF distance baseline (`src/baseline_tfidf.py`)
- apples-to-apples comparison matrix (`src/run_comparison_matrix.py`)

## 4. Split Policy

Default split strategy:

- `id_disjoint` (prevents the same record ID from appearing across train/val/test)

Deterministic split artifact path pattern:

- `data/processed/splits/{strategy}_seed{seed}.tsv`

Canonical committed split:

- `data/processed/splits/id_disjoint_seed42.tsv`

## 5. Evaluation Context

Two evaluation scenarios are supported:

- `pair_scoring`: no blocking
- `blocked_pipeline`: shared blocking for both model families

Default blocking used in comparison runs:

- keys: `first_name,age`
- mode: `any`

This keeps evaluation fair across Siamese and TF-IDF methods.
