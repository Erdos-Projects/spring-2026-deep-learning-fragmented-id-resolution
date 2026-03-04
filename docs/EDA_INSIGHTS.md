# EDA and Pairing Insights: NCVoters Duplicate Detection

This document summarizes the EDA conclusions that directly affect model and evaluation design.

## 1. Attribute Signal Quality

High-signal identity anchors:

- `first_name`
- `last_name`
- `age`
- `sex`

Useful contextual fields (less stable but still informative):

- `house_num`
- `street_name`
- `zip_code`

Additional disambiguation fields:

- `race_desc`
- `ethnic_desc`

Practical implication:

- keep both baseline and extended attribute sets in experiments
- do not rely only on address fields, because duplicates can move or be reformatted

## 2. Blocking Implications

Blocking is used as candidate generation, not clustering.

Shared blocking currently used in the controlled pipeline:

- keys: `first_name,age`
- mode: `any`

Observed on the apples-to-apples test split:

- positive pass rate: `1.0`
- negative pass rate: `0.03495`
- candidates after blocking: `303 / 2291`

Practical implication:

- blocking drastically cuts pair volume while preserving recall on this split
- final claims should still report blocked and unblocked results separately

## 3. Input Length and Encoding

The Siamese model uses character-level encoding with `max_len`.

Practical implication:

- baseline attributes work well with smaller `max_len` values
- extended attributes should use a larger `max_len` (for example `128`) to avoid truncation penalties

## 4. Evaluation Design Requirements

To avoid misleading conclusions:

- use deterministic `id_disjoint` splits
- evaluate both model families on the same split artifact
- compare both scenarios:
  - `pair_scoring` (no blocking)
  - `blocked_pipeline` (shared blocking)

Metrics to always report:

- Precision
- Recall
- F1
- PR-AUC
- ROC-AUC

## 5. Current Outcome Snapshot

From `models/comparisons/apples_to_apples/comparison_summary.json`:

- Siamese outperforms TF-IDF on test F1 in all matched settings
- strongest run: `siamese + extended + blocked_pipeline`
  - test F1: `0.9849`
  - test PR-AUC: `0.9994`

Interpretation:

- TF-IDF remains a strong baseline
- with matched conditions, the deep model provides a measurable gain
