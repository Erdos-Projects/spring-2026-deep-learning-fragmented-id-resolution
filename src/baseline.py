"""
Classical supervised baseline for pairwise voter-record matching.

The baseline is intentionally simple:
  - build field-aware similarity features
  - fit a standard classifier (logistic regression by default)
  - score blocked candidate pairs and cluster above-threshold links

For this project's structured names/addresses/birth years, this is a stronger
and more stable first-line model than the current character-level Siamese CNN.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .blocking import block_and_generate_candidates
from .preprocessing import normalize_dataframe

# Minimum name-similarity threshold.  Any candidate pair whose
# max(first_name_ratio, last_name_ratio) falls below this is discarded
# *before* clustering, regardless of the classifier score.  This prevents
# completely-unrelated records (different first AND last name) from being
# linked just because they share the same birth year or address.
NAME_SIMILARITY_FLOOR = 0.30

PAIR_FIELDS = [
    "first_name",
    "middle_name",
    "last_name",
    "name_suffix",
    "street_address",
    "city",
    "zip5",
    "birth_year",
]

FEATURE_COLUMNS = [
    # --- Name similarity (fuzzy ratios subsume exact-match booleans) ---
    "first_name_ratio",
    "middle_name_ratio",
    "last_name_ratio",
    "full_name_ratio",
    # --- Address / location similarity ---
    "street_ratio",
    "street_token_set_ratio",
    "city_ratio",
    "full_record_ratio",
    # --- Exact-match signals (only those NOT redundant with a ratio) ---
    "name_suffix_eq",
    "zip5_eq",
    "house_number_eq",
    # --- Birth year ---
    "birth_year_eq",
    "birth_year_absdiff",
    "birth_year_missing",
    # --- Name-length safety ---
    "short_name_flag",
]


@dataclass
class PairClassifierBundle:
    """Serializable wrapper around a fitted classical pair classifier."""

    model: object
    model_type: str
    feature_columns: List[str]
    threshold: float = 0.85


class UnionFind:
    """Weighted union-find with path compression for clustering."""

    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, left: int, right: int) -> None:
        root_left = self.find(left)
        root_right = self.find(right)
        if root_left == root_right:
            return

        if self.rank[root_left] < self.rank[root_right]:
            root_left, root_right = root_right, root_left
        self.parent[root_right] = root_left
        if self.rank[root_left] == self.rank[root_right]:
            self.rank[root_left] += 1


def _read_csv_with_encoding_fallback(path: Union[str, Path], **kwargs) -> pd.DataFrame:
    """Read delimited text with a small set of common encoding fallbacks."""
    last_error: Optional[UnicodeDecodeError] = None
    for encoding in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
        try:
            return pd.read_csv(path, encoding=encoding, **kwargs)
        except UnicodeDecodeError as exc:
            last_error = exc

    if last_error is not None:
        raise last_error
    raise RuntimeError("Failed to read input file with the supported encodings.")


def _prepare_records(records_df: pd.DataFrame) -> pd.DataFrame:
    """Return records with a stable string `id` column and expected fields."""
    if "id" in records_df.columns:
        records = records_df.copy()
    else:
        records = records_df.reset_index().rename(columns={"index": "id"})

    records["id"] = records["id"].astype(str)
    for field in PAIR_FIELDS:
        if field not in records.columns:
            records[field] = ""

    return records[["id"] + PAIR_FIELDS].copy()


def _normalize_text_series(series: pd.Series) -> pd.Series:
    """Stringify and strip missing values for pairwise string features."""
    cleaned = series.fillna("").astype(str).str.strip()
    return cleaned.replace({"<NA>": "", "nan": "", "None": ""})


def _fuzz_ratio(left: Sequence[str], right: Sequence[str]) -> np.ndarray:
    return np.array([fuzz.ratio(a, b) / 100.0 for a, b in zip(left, right)], dtype=np.float32)


def _token_set_ratio(left: Sequence[str], right: Sequence[str]) -> np.ndarray:
    return np.array(
        [fuzz.token_set_ratio(a, b) / 100.0 for a, b in zip(left, right)],
        dtype=np.float32,
    )


def _house_numbers(addresses: pd.Series) -> pd.Series:
    return _normalize_text_series(addresses).str.extract(r"(\d+)", expand=False).fillna("")


def _safe_initials(series: pd.Series) -> pd.Series:
    return _normalize_text_series(series).str[:1]


def _feature_chunk(pairs_chunk: pd.DataFrame, records: pd.DataFrame) -> pd.DataFrame:
    records_left = records.add_suffix("_1")
    records_right = records.add_suffix("_2")

    merged = pairs_chunk.merge(records_left, left_on="id1", right_on="id_1", how="left")
    merged = merged.merge(records_right, left_on="id2", right_on="id_2", how="left")

    if merged["id_1"].isna().any() or merged["id_2"].isna().any():
        missing = merged[merged["id_1"].isna() | merged["id_2"].isna()][["id1", "id2"]]
        raise KeyError(f"Pair references missing record IDs:\n{missing.head(5).to_string(index=False)}")

    text_cols = [
        "first_name",
        "middle_name",
        "last_name",
        "name_suffix",
        "street_address",
        "city",
        "zip5",
    ]
    for col in text_cols:
        merged[f"{col}_1"] = _normalize_text_series(merged[f"{col}_1"])
        merged[f"{col}_2"] = _normalize_text_series(merged[f"{col}_2"])

    birth_year_1 = pd.to_numeric(merged["birth_year_1"], errors="coerce")
    birth_year_2 = pd.to_numeric(merged["birth_year_2"], errors="coerce")
    birth_year_missing = birth_year_1.isna() | birth_year_2.isna()
    birth_year_absdiff = (birth_year_1 - birth_year_2).abs().fillna(99.0).clip(0, 99)

    full_name_1 = (
        merged["first_name_1"] + " " + merged["middle_name_1"] + " "
        + merged["last_name_1"] + " " + merged["name_suffix_1"]
    ).str.strip()
    full_name_2 = (
        merged["first_name_2"] + " " + merged["middle_name_2"] + " "
        + merged["last_name_2"] + " " + merged["name_suffix_2"]
    ).str.strip()
    full_record_1 = (
        full_name_1 + " | " + merged["street_address_1"] + " | " + merged["city_1"]
        + " | " + merged["zip5_1"] + " | " + birth_year_1.fillna(-1).astype(int).astype(str)
    )
    full_record_2 = (
        full_name_2 + " | " + merged["street_address_2"] + " | " + merged["city_2"]
        + " | " + merged["zip5_2"] + " | " + birth_year_2.fillna(-1).astype(int).astype(str)
    )

    features = pd.DataFrame(index=merged.index)
    for col in text_cols:
        features[f"{col}_eq"] = (merged[f"{col}_1"] == merged[f"{col}_2"]).astype(np.float32)

    features["birth_year_eq"] = ((birth_year_1 == birth_year_2) & ~birth_year_missing).astype(np.float32)
    features["first_initial_eq"] = (_safe_initials(merged["first_name_1"]) == _safe_initials(merged["first_name_2"])).astype(np.float32)
    features["last_initial_eq"] = (_safe_initials(merged["last_name_1"]) == _safe_initials(merged["last_name_2"])).astype(np.float32)
    features["house_number_eq"] = (_house_numbers(merged["street_address_1"]) == _house_numbers(merged["street_address_2"])).astype(np.float32)
    features["zip3_eq"] = (merged["zip5_1"].str[:3] == merged["zip5_2"].str[:3]).astype(np.float32)
    features["birth_year_missing"] = birth_year_missing.astype(np.float32)
    features["birth_year_close_1"] = ((birth_year_absdiff <= 1) & ~birth_year_missing).astype(np.float32)
    features["birth_year_close_3"] = ((birth_year_absdiff <= 3) & ~birth_year_missing).astype(np.float32)
    features["birth_year_absdiff"] = birth_year_absdiff.astype(np.float32)

    features["first_name_ratio"] = _fuzz_ratio(merged["first_name_1"], merged["first_name_2"])
    features["middle_name_ratio"] = _fuzz_ratio(merged["middle_name_1"], merged["middle_name_2"])
    features["last_name_ratio"] = _fuzz_ratio(merged["last_name_1"], merged["last_name_2"])
    features["street_ratio"] = _fuzz_ratio(merged["street_address_1"], merged["street_address_2"])
    features["street_token_set_ratio"] = _token_set_ratio(merged["street_address_1"], merged["street_address_2"])
    features["city_ratio"] = _fuzz_ratio(merged["city_1"], merged["city_2"])
    features["full_name_ratio"] = _fuzz_ratio(full_name_1, full_name_2)
    features["full_record_ratio"] = _fuzz_ratio(full_record_1, full_record_2)

    # Flag pairs where either record has a very short last name (<=1 char).
    # These single-letter surnames (e.g. "A") produce unreliable similarity
    # scores and should be heavily penalised by the classifier.
    ln1_len = merged["last_name_1"].str.len()
    ln2_len = merged["last_name_2"].str.len()
    fn1_len = merged["first_name_1"].str.len()
    fn2_len = merged["first_name_2"].str.len()
    features["short_name_flag"] = (
        (ln1_len <= 1) | (ln2_len <= 1) | (fn1_len <= 1) | (fn2_len <= 1)
    ).astype(np.float32)

    return features[FEATURE_COLUMNS]


def build_pair_feature_frame(
    pairs_df: pd.DataFrame,
    records_df: pd.DataFrame,
    batch_size: int = 50_000,
) -> pd.DataFrame:
    """
    Build a feature matrix for record-id pairs.

    Args:
        pairs_df: DataFrame with columns `id1`, `id2` and optionally `label`
        records_df: normalized records with an `id` column
        batch_size: chunk size used to limit merge memory
    """
    if pairs_df.empty:
        return pd.DataFrame(columns=FEATURE_COLUMNS)

    pairs = pairs_df.copy()
    pairs["id1"] = pairs["id1"].astype(str)
    pairs["id2"] = pairs["id2"].astype(str)
    records = _prepare_records(records_df)

    chunks = []
    for start in range(0, len(pairs), batch_size):
        end = min(start + batch_size, len(pairs))
        chunks.append(_feature_chunk(pairs.iloc[start:end], records))
    return pd.concat(chunks, ignore_index=True)


def _build_estimator(
    model_type: str,
    random_state: int,
    max_iter: int,
):
    if model_type == "logreg":
        return Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=max_iter,
                        class_weight="balanced",
                        random_state=random_state,
                    ),
                ),
            ]
        )

    if model_type == "hgbt":
        return HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_depth=6,
            max_iter=250,
            random_state=random_state,
        )

    raise ValueError(f"Unsupported model_type: {model_type}")


def train_pair_classifier(
    train_pairs: pd.DataFrame,
    records_df: pd.DataFrame,
    model_type: str = "logreg",
    random_state: int = 42,
    max_iter: int = 1000,
) -> PairClassifierBundle:
    """Fit a classical pair classifier on the labeled training pairs."""
    features = build_pair_feature_frame(train_pairs, records_df)
    labels = train_pairs["label"].astype(int).to_numpy()

    estimator = _build_estimator(model_type=model_type, random_state=random_state, max_iter=max_iter)
    estimator.fit(features, labels)
    return PairClassifierBundle(
        model=estimator,
        model_type=model_type,
        feature_columns=list(FEATURE_COLUMNS),
    )


def predict_pair_probabilities(
    bundle: PairClassifierBundle,
    pairs_df: pd.DataFrame,
    records_df: pd.DataFrame,
    batch_size: int = 50_000,
) -> np.ndarray:
    """Predict match probabilities for explicit `id1`, `id2` pairs."""
    features = build_pair_feature_frame(pairs_df, records_df, batch_size=batch_size)
    model = bundle.model
    return model.predict_proba(features[bundle.feature_columns])[:, 1]


def evaluate_pair_classifier(
    bundle: PairClassifierBundle,
    eval_pairs: pd.DataFrame,
    records_df: pd.DataFrame,
    threshold: Optional[float] = None,
    batch_size: int = 50_000,
) -> Dict[str, float]:
    """Evaluate a fitted classical pair classifier."""
    probs = predict_pair_probabilities(bundle, eval_pairs, records_df, batch_size=batch_size)
    labels = eval_pairs["label"].astype(int).to_numpy()
    threshold = bundle.threshold if threshold is None else threshold
    preds = (probs >= threshold).astype(int)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary", zero_division=0
    )
    acc = accuracy_score(labels, preds)
    auc = roc_auc_score(labels, probs) if len(np.unique(labels)) > 1 else 0.0

    n_neg = (labels == 0).sum()
    n_pos = (labels == 1).sum()
    fp = ((preds == 1) & (labels == 0)).sum()
    fn = ((preds == 0) & (labels == 1)).sum()

    return {
        "accuracy": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "auc": float(auc),
        "fpr": float(fp / max(n_neg, 1)),
        "fnr": float(fn / max(n_pos, 1)),
    }


def save_pair_classifier(bundle: PairClassifierBundle, path: Union[str, Path]) -> None:
    """Persist a trained classical baseline with joblib."""
    payload = {
        "model": bundle.model,
        "model_type": bundle.model_type,
        "feature_columns": bundle.feature_columns,
        "threshold": bundle.threshold,
    }
    joblib.dump(payload, path)


def load_pair_classifier(path: Union[str, Path]) -> PairClassifierBundle:
    """Load a previously saved classical baseline."""
    payload = joblib.load(path)
    return PairClassifierBundle(
        model=payload["model"],
        model_type=payload["model_type"],
        feature_columns=list(payload["feature_columns"]),
        threshold=float(payload.get("threshold", 0.5)),
    )


def _compute_name_sim_for_candidates(
    candidate_pairs: Sequence[Tuple[int, int]],
    records_df: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (first_name_ratio, last_name_ratio) arrays for candidate pairs.

    Used to enforce the name-similarity floor *before* clustering so that
    pairs with completely different names are never linked regardless of
    classifier score.
    """
    records = _prepare_records(records_df).reset_index(drop=True)
    fn = _normalize_text_series(records["first_name"]).values
    ln = _normalize_text_series(records["last_name"]).values
    fn_ratios = np.array(
        [fuzz.ratio(fn[i], fn[j]) / 100.0 for i, j in candidate_pairs],
        dtype=np.float32,
    )
    ln_ratios = np.array(
        [fuzz.ratio(ln[i], ln[j]) / 100.0 for i, j in candidate_pairs],
        dtype=np.float32,
    )
    return fn_ratios, ln_ratios


def score_candidate_pairs(
    bundle: PairClassifierBundle,
    candidate_pairs: Sequence[Tuple[int, int]],
    records_df: pd.DataFrame,
    batch_size: int = 50_000,
) -> np.ndarray:
    """
    Score candidate row-index pairs using the fitted classical baseline.

    This is the inference path intended for large blocked candidate sets.
    """
    records = _prepare_records(records_df).reset_index(drop=True)
    record_ids = records["id"].tolist()
    all_scores: List[np.ndarray] = []

    for start in range(0, len(candidate_pairs), batch_size):
        end = min(start + batch_size, len(candidate_pairs))
        chunk = candidate_pairs[start:end]
        chunk_pairs = pd.DataFrame(
            {
                "id1": [record_ids[i] for i, _ in chunk],
                "id2": [record_ids[j] for _, j in chunk],
            }
        )
        all_scores.append(predict_pair_probabilities(bundle, chunk_pairs, records, batch_size=batch_size))

    if not all_scores:
        return np.array([], dtype=np.float32)
    return np.concatenate(all_scores).astype(np.float32)


def cluster_pairs(
    pairs: Sequence[Tuple[int, int]],
    scores: np.ndarray,
    n_records: int,
    threshold: float,
    records_df: Optional[pd.DataFrame] = None,
    name_sim_floor: float = NAME_SIMILARITY_FLOOR,
) -> Dict[int, List[int]]:
    """Cluster above-threshold candidate links into duplicate components.

    When *records_df* is provided, a name-similarity guard is applied:
    pairs whose max(first_name_ratio, last_name_ratio) falls below
    *name_sim_floor* are never linked, regardless of classifier score.
    This prevents completely-unrelated records from being merged just
    because they share demographics (same birth year / address).
    """
    # Pre-compute name-similarity mask when records are available
    name_floor_mask: Optional[np.ndarray] = None
    if records_df is not None:
        fn_ratios, ln_ratios = _compute_name_sim_for_candidates(pairs, records_df)
        name_max = np.maximum(fn_ratios, ln_ratios)
        name_floor_mask = name_max < name_sim_floor

    uf = UnionFind(n_records)
    for idx, ((left, right), score) in enumerate(zip(pairs, scores)):
        if score < threshold:
            continue
        if name_floor_mask is not None and name_floor_mask[idx]:
            continue
        uf.union(left, right)

    clusters: Dict[int, List[int]] = {}
    for idx in range(n_records):
        root = uf.find(idx)
        clusters.setdefault(root, []).append(idx)
    return {root: members for root, members in clusters.items() if len(members) >= 2}


def find_duplicates_with_baseline(
    input_path: Union[str, Path],
    model_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    threshold: Optional[float] = None,
    max_block_size: int = 500,
    max_pairs_per_block: int = 200,
    score_batch_size: int = 50_000,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    End-to-end duplicate detection with the classical baseline.

    The baseline uses blocking only, which makes it practical for large files
    such as Mecklenburg once candidate generation is tuned.
    """
    input_path = Path(input_path)
    sep = "\t" if input_path.suffix in (".tsv", ".txt") else ","
    raw_df = _read_csv_with_encoding_fallback(
        input_path,
        sep=sep,
        dtype=str,
        na_filter=False,
        on_bad_lines="skip",
    )
    raw_df = raw_df.loc[:, ~raw_df.columns.str.contains(r"^Unnamed")]
    raw_df.columns = [c.strip().lower() for c in raw_df.columns]

    records = normalize_dataframe(raw_df)
    bundle = load_pair_classifier(model_path)
    score_threshold = bundle.threshold if threshold is None else threshold

    candidates = block_and_generate_candidates(
        records,
        max_block_size=max_block_size,
        max_pairs_per_block=max_pairs_per_block,
    )
    scores = score_candidate_pairs(
        bundle,
        candidates,
        records,
        batch_size=score_batch_size,
    )

    # Name-similarity floor is applied inside cluster_pairs — pairs
    # whose max(first_name_ratio, last_name_ratio) < NAME_SIMILARITY_FLOOR
    # are never linked, preventing completely-unrelated records from
    # clustering just because they share demographics.
    clusters = cluster_pairs(
        candidates, scores, len(records),
        threshold=score_threshold,
        records_df=records,
    )

    cluster_rows = []
    for cluster_id, members in clusters.items():
        for row_idx in members:
            row = records.iloc[row_idx].to_dict()
            row["cluster_id"] = cluster_id
            cluster_rows.append(row)
    clusters_df = pd.DataFrame(cluster_rows) if cluster_rows else pd.DataFrame(
        columns=list(records.columns) + ["cluster_id"]
    )

    pair_rows = []
    for (left, right), score in zip(candidates, scores):
        if score >= score_threshold:
            pair_rows.append(
                {
                    "id1": str(records.iloc[left]["id"]),
                    "id2": str(records.iloc[right]["id"]),
                    "score": float(score),
                }
            )
    scored_pairs_df = pd.DataFrame(pair_rows) if pair_rows else pd.DataFrame(
        columns=["id1", "id2", "score"]
    )
    if not scored_pairs_df.empty:
        scored_pairs_df = scored_pairs_df.sort_values("score", ascending=False).reset_index(drop=True)

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        scored_pairs_df.to_csv(output_path, sep="\t", index=False)

    return clusters_df, scored_pairs_df
