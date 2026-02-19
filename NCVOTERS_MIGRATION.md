# NCVoters reuse guide (preprocessing + EDA)

This repository now includes a standalone script:

- `ncvoters_preprocess_and_eda.py`

It reuses logic from:

- `datasets/preparation/global_preparations.py` (cleaning)
- `datasets/dataset_statistics.py` (attribute-level EDA)
- `resources/reference.conf` (`ncvoters` selected attributes)

## What you need

Minimum (for cleaning + EDA only):

- `ncvoters.tsv`

Needed only for supervised duplicate-detection training/evaluation:

- `ncvoters_DPL.tsv`
- `ncvoters_NDPL.tsv`

## Run

```bash
python ncvoters_preprocess_and_eda.py ^
  --input ncvoters.tsv ^
  --prepared-output ncvoters_prepared.tsv ^
  --stats-output ncvoters_attribute_stats.tsv ^
  --keep-only-config-columns
```

## Outputs

- `ncvoters_prepared.tsv`: normalized values (special chars removed, whitespace normalized, lowercase)
- `ncvoters_attribute_stats.tsv`: per-attribute metrics
  - `uniqueness`
  - `completeness`
  - `counter_total`
  - `counter_valid`
  - `counter_invalid`

## If you later need pair-based duplicate detection

Then bring in these scripts as well:

- `datasets/import_datasets_to_relations.py`
- `steps/hymd/mdedup_calculate_similarities.py`

These scripts use `ncvoters_DPL.tsv` and `ncvoters_NDPL.tsv` to build pair labels and pairwise similarities.
