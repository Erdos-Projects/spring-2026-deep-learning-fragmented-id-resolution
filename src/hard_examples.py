from __future__ import annotations

from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd

try:
    from .data_utils import load_labeled_pairs, load_prepared_data
except ImportError:
    from data_utils import load_labeled_pairs, load_prepared_data


STRONG_MATCH_COLUMNS = [
    "same_first_name",
    "same_last_name",
    "same_house_num",
    "same_street_name",
    "same_zip_code",
    "same_age",
    "same_sex",
]


def _safe_text(value) -> str:
    return str(value or "").strip().lower()


def _similarity(left: str, right: str) -> float:
    left = _safe_text(left)
    right = _safe_text(right)
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    return float(SequenceMatcher(None, left, right).ratio())


def _token_containment(left: str, right: str) -> bool:
    left_tokens = {token for token in _safe_text(left).split() if token}
    right_tokens = {token for token in _safe_text(right).split() if token}
    if not left_tokens or not right_tokens:
        return False
    smaller, larger = sorted([left_tokens, right_tokens], key=len)
    return smaller.issubset(larger) and smaller != larger


def _record_similarity(row: pd.Series) -> float:
    left = " ".join(
        [
            _safe_text(row.get("first_name_1")),
            _safe_text(row.get("last_name_1")),
            _safe_text(row.get("house_num_1")),
            _safe_text(row.get("street_name_1")),
            _safe_text(row.get("zip_code_1")),
            _safe_text(row.get("age_1")),
            _safe_text(row.get("sex_1")),
        ]
    ).strip()
    right = " ".join(
        [
            _safe_text(row.get("first_name_2")),
            _safe_text(row.get("last_name_2")),
            _safe_text(row.get("house_num_2")),
            _safe_text(row.get("street_name_2")),
            _safe_text(row.get("zip_code_2")),
            _safe_text(row.get("age_2")),
            _safe_text(row.get("sex_2")),
        ]
    ).strip()
    return _similarity(left, right)


def _enrich_pairs(pairs_df: pd.DataFrame, data_df: pd.DataFrame) -> pd.DataFrame:
    left_cols = {
        column: f"{column}_1"
        for column in [
            "id",
            "first_name",
            "midl_name",
            "last_name",
            "house_num",
            "street_name",
            "zip_code",
            "age",
            "sex",
            "race_desc",
            "ethnic_desc",
        ]
        if column in data_df.columns
    }
    right_cols = {column: f"{column}_2" for column in left_cols}

    enriched = pairs_df.copy()
    left_frame = data_df.rename(columns=left_cols)
    right_frame = data_df.rename(columns=right_cols)
    enriched = enriched.merge(left_frame, left_on="id1", right_on="id_1", how="left")
    enriched = enriched.merge(right_frame, left_on="id2", right_on="id_2", how="left")
    return enriched


def _compute_feature_columns(enriched: pd.DataFrame) -> pd.DataFrame:
    working = enriched.copy()

    field_pairs = [
        ("first_name", "same_first_name"),
        ("last_name", "same_last_name"),
        ("house_num", "same_house_num"),
        ("street_name", "same_street_name"),
        ("zip_code", "same_zip_code"),
        ("age", "same_age"),
        ("sex", "same_sex"),
        ("race_desc", "same_race_desc"),
        ("ethnic_desc", "same_ethnic_desc"),
        ("midl_name", "same_midl_name"),
    ]
    for field, output_col in field_pairs:
        left_col = f"{field}_1"
        right_col = f"{field}_2"
        if left_col in working.columns and right_col in working.columns:
            left_values = working[left_col].fillna("").astype(str).str.lower().str.strip()
            right_values = working[right_col].fillna("").astype(str).str.lower().str.strip()
            working[output_col] = (left_values == right_values) & left_values.ne("")
        else:
            working[output_col] = False

    working["first_name_similarity"] = working.apply(
        lambda row: _similarity(row.get("first_name_1"), row.get("first_name_2")), axis=1
    )
    working["last_name_similarity"] = working.apply(
        lambda row: _similarity(row.get("last_name_1"), row.get("last_name_2")), axis=1
    )
    working["full_name_similarity"] = working.apply(
        lambda row: _similarity(
            " ".join(
                [
                    _safe_text(row.get("first_name_1")),
                    _safe_text(row.get("midl_name_1")),
                    _safe_text(row.get("last_name_1")),
                ]
            ),
            " ".join(
                [
                    _safe_text(row.get("first_name_2")),
                    _safe_text(row.get("midl_name_2")),
                    _safe_text(row.get("last_name_2")),
                ]
            ),
        ),
        axis=1,
    )
    working["address_similarity"] = working.apply(
        lambda row: _similarity(
            " ".join([_safe_text(row.get("house_num_1")), _safe_text(row.get("street_name_1"))]),
            " ".join([_safe_text(row.get("house_num_2")), _safe_text(row.get("street_name_2"))]),
        ),
        axis=1,
    )
    working["record_similarity"] = working.apply(_record_similarity, axis=1)
    working["surname_token_containment"] = working.apply(
        lambda row: _token_containment(row.get("last_name_1"), row.get("last_name_2")), axis=1
    )
    working["strong_match_count"] = working[STRONG_MATCH_COLUMNS].astype(int).sum(axis=1)
    working["strong_diff_count"] = len(STRONG_MATCH_COLUMNS) - working["strong_match_count"]
    return working


def _hard_positive_profile(row: pd.Series) -> Dict[str, object]:
    score = 0.0
    reasons: List[str] = []

    if not bool(row["same_last_name"]):
        score += 2.0
        reasons.append("last_name_differs")
    if bool(row["surname_token_containment"]):
        score += 1.0
        reasons.append("surname_expansion_or_compound_name")
    if not bool(row["same_street_name"]) or not bool(row["same_house_num"]):
        score += 1.5
        reasons.append("address_changed")
    if not bool(row["same_zip_code"]):
        score += 1.0
        reasons.append("zip_changed")
    if not bool(row["same_age"]):
        score += 0.75
        reasons.append("age_changed")
    if not bool(row["same_sex"]):
        score += 0.5
        reasons.append("sex_changed")
    if float(row["full_name_similarity"]) < 0.88:
        score += 1.25
        reasons.append("full_name_similarity_low")
    if float(row["address_similarity"]) < 0.90:
        score += 1.0
        reasons.append("address_similarity_low")
    if float(row["record_similarity"]) < 0.84:
        score += 0.75
        reasons.append("overall_similarity_low")

    return {
        "difficulty_score": float(score),
        "difficulty_reasons": "|".join(reasons),
    }


def _hard_negative_profile(row: pd.Series) -> Dict[str, object]:
    score = 0.0
    reasons: List[str] = []

    if bool(row["same_first_name"]):
        score += 1.5
        reasons.append("same_first_name")
    if bool(row["same_last_name"]):
        score += 1.75
        reasons.append("same_last_name")
    elif float(row["last_name_similarity"]) >= 0.82:
        score += 1.0
        reasons.append("similar_last_name")
    if bool(row["same_house_num"]):
        score += 0.75
        reasons.append("same_house_number")
    if bool(row["same_street_name"]):
        score += 1.5
        reasons.append("same_street_name")
    elif float(row["address_similarity"]) >= 0.90:
        score += 1.0
        reasons.append("similar_address")
    if bool(row["same_zip_code"]):
        score += 0.75
        reasons.append("same_zip")
    if bool(row["same_age"]):
        score += 1.0
        reasons.append("same_age")
    if bool(row["same_sex"]):
        score += 0.25
        reasons.append("same_sex")
    if float(row["full_name_similarity"]) >= 0.90:
        score += 1.5
        reasons.append("high_name_similarity")
    if float(row["record_similarity"]) >= 0.88:
        score += 1.25
        reasons.append("high_overall_similarity")

    return {
        "difficulty_score": float(score),
        "difficulty_reasons": "|".join(reasons),
    }


def mine_hard_examples(
    data_path: str | Path,
    dpl_path: str | Path,
    ndpl_path: str | Path,
    hard_positive_threshold: float | None = None,
    hard_negative_threshold: float | None = None,
    hard_positive_quantile: float = 0.85,
    hard_negative_quantile: float = 0.99,
) -> pd.DataFrame:
    data_df = load_prepared_data(data_path)
    pairs_df = load_labeled_pairs(dpl_path, ndpl_path, fail_on_duplicate_pairs=True)
    enriched = _compute_feature_columns(_enrich_pairs(pairs_df, data_df))

    profiles = []
    for row in enriched.itertuples(index=False):
        row_series = pd.Series(row._asdict())
        if int(float(row_series["label"])) == 1:
            profile = _hard_positive_profile(row_series)
        else:
            profile = _hard_negative_profile(row_series)
        profiles.append(profile)

    profile_df = pd.DataFrame(profiles)
    result = pd.concat([enriched.reset_index(drop=True), profile_df], axis=1)
    pos_scores = result.loc[result["label"].astype(int) == 1, "difficulty_score"]
    neg_scores = result.loc[result["label"].astype(int) == 0, "difficulty_score"]
    if hard_positive_threshold is None:
        hard_positive_threshold = float(pos_scores.quantile(hard_positive_quantile)) if len(pos_scores) else float("inf")
    if hard_negative_threshold is None:
        hard_negative_threshold = float(neg_scores.quantile(hard_negative_quantile)) if len(neg_scores) else float("inf")

    result["hard_example"] = False
    result["hard_type"] = np.where(result["label"].astype(int) == 1, "easy_positive", "easy_negative")
    positive_mask = (result["label"].astype(int) == 1) & (result["difficulty_score"] >= float(hard_positive_threshold))
    negative_mask = (result["label"].astype(int) == 0) & (result["difficulty_score"] >= float(hard_negative_threshold))
    result.loc[positive_mask | negative_mask, "hard_example"] = True
    result.loc[positive_mask, "hard_type"] = "hard_positive"
    result.loc[negative_mask, "hard_type"] = "hard_negative"
    result["difficulty_bucket"] = np.where(result["hard_example"], result["hard_type"], "easy")

    pos_scale = max(float(hard_positive_threshold), 1e-6)
    neg_scale = max(float(hard_negative_threshold), 1e-6)
    result["suggested_weight"] = 1.0
    result.loc[positive_mask, "suggested_weight"] = 1.0 + np.minimum(
        2.5, result.loc[positive_mask, "difficulty_score"] / pos_scale
    )
    result.loc[negative_mask, "suggested_weight"] = 1.0 + np.minimum(
        2.5, result.loc[negative_mask, "difficulty_score"] / neg_scale
    )
    result["hard_positive_threshold"] = float(hard_positive_threshold)
    result["hard_negative_threshold"] = float(hard_negative_threshold)
    return result


def load_hard_example_artifact(path: str | Path) -> pd.DataFrame:
    artifact = pd.read_csv(path, sep="\t", dtype=str)
    if "pair_id" not in artifact.columns:
        raise ValueError("Hard-example artifact must include 'pair_id'.")

    artifact["pair_id"] = artifact["pair_id"].astype(int)
    for column in ["difficulty_score", "suggested_weight"]:
        if column in artifact.columns:
            artifact[column] = artifact[column].astype(float)
    if "hard_example" in artifact.columns:
        artifact["hard_example"] = artifact["hard_example"].astype(str).str.lower().isin(["1", "true", "yes"])
    return artifact


def attach_hard_example_metadata(
    pairs_df: pd.DataFrame,
    hard_example_df: pd.DataFrame | None,
    weight_scale: float = 1.0,
    hard_positive_weight_scale: float | None = None,
    hard_negative_weight_scale: float | None = None,
) -> pd.DataFrame:
    merged = pairs_df.copy()
    merged["pair_id"] = merged["pair_id"].astype(int)

    default_columns = {
        "difficulty_score": 0.0,
        "hard_example": False,
        "hard_type": "easy",
        "difficulty_bucket": "easy",
        "difficulty_reasons": "",
        "suggested_weight": 1.0,
    }

    if hard_example_df is None or hard_example_df.empty:
        for column, value in default_columns.items():
            merged[column] = value
        merged["sample_weight"] = 1.0
        return merged

    keep_columns = ["pair_id"] + [column for column in default_columns if column in hard_example_df.columns]
    hard_subset = hard_example_df[keep_columns].copy()
    merged = merged.merge(hard_subset, on="pair_id", how="left")

    for column, value in default_columns.items():
        if column not in merged.columns:
            merged[column] = value
        else:
            merged[column] = merged[column].fillna(value)

    positive_scale = float(weight_scale if hard_positive_weight_scale is None else hard_positive_weight_scale)
    negative_scale = float(weight_scale if hard_negative_weight_scale is None else hard_negative_weight_scale)
    merged["sample_weight"] = 1.0
    positive_mask = merged["hard_type"].astype(str).eq("hard_positive")
    negative_mask = merged["hard_type"].astype(str).eq("hard_negative")
    merged.loc[positive_mask, "sample_weight"] = 1.0 + (
        merged.loc[positive_mask, "suggested_weight"].astype(float) - 1.0
    ) * positive_scale
    merged.loc[negative_mask, "sample_weight"] = 1.0 + (
        merged.loc[negative_mask, "suggested_weight"].astype(float) - 1.0
    ) * negative_scale
    return merged


def _subset_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float, mask: np.ndarray) -> Dict[str, object]:
    try:
        from .metrics import compute_binary_metrics
    except ImportError:
        from metrics import compute_binary_metrics

    if int(mask.sum()) == 0:
        return {"count": 0}
    subset_true = np.asarray(y_true)[mask]
    subset_prob = np.asarray(y_prob)[mask]
    metrics = compute_binary_metrics(subset_true, subset_prob, threshold=threshold)
    metrics["count"] = int(mask.sum())
    return metrics


def summarize_hard_example_metrics(
    pairs_df: pd.DataFrame,
    y_true: Iterable[int],
    y_prob: Iterable[float],
    threshold: float,
) -> Dict[str, object]:
    y_true = np.asarray(list(y_true)).astype(int)
    y_prob = np.asarray(list(y_prob)).astype(float)
    y_pred = (y_prob >= threshold).astype(int)

    hard_example = pairs_df.get("hard_example", pd.Series(False, index=pairs_df.index)).astype(bool).to_numpy()
    hard_positive = pairs_df.get("hard_type", pd.Series("", index=pairs_df.index)).astype(str).eq("hard_positive").to_numpy()
    hard_negative = pairs_df.get("hard_type", pd.Series("", index=pairs_df.index)).astype(str).eq("hard_negative").to_numpy()
    easy_mask = ~hard_example

    hard_positive_recall = float((y_pred[hard_positive] == 1).mean()) if int(hard_positive.sum()) else float("nan")
    hard_negative_rejection_rate = (
        float((y_pred[hard_negative] == 0).mean()) if int(hard_negative.sum()) else float("nan")
    )

    summary = {
        "hard_example_count": int(hard_example.sum()),
        "hard_positive_count": int(hard_positive.sum()),
        "hard_negative_count": int(hard_negative.sum()),
        "easy_example_count": int(easy_mask.sum()),
        "hard_positive_recall": hard_positive_recall,
        "hard_negative_rejection_rate": hard_negative_rejection_rate,
        "hard_subset_metrics": _subset_metrics(y_true, y_prob, threshold, hard_example),
        "easy_subset_metrics": _subset_metrics(y_true, y_prob, threshold, easy_mask),
        "hard_positive_metrics": _subset_metrics(y_true, y_prob, threshold, hard_positive),
        "hard_negative_metrics": _subset_metrics(y_true, y_prob, threshold, hard_negative),
    }
    return summary


def summarize_hard_examples_for_report(hard_df: pd.DataFrame) -> Dict[str, object]:
    summary = {
        "total_pairs": int(len(hard_df)),
        "hard_example_count": int(hard_df["hard_example"].astype(bool).sum()),
        "hard_positive_count": int((hard_df["hard_type"] == "hard_positive").sum()),
        "hard_negative_count": int((hard_df["hard_type"] == "hard_negative").sum()),
    }
    reason_counter: Dict[str, int] = {}
    for raw_reasons in hard_df.loc[hard_df["hard_example"].astype(bool), "difficulty_reasons"].fillna(""):
        for reason in [item for item in str(raw_reasons).split("|") if item]:
            reason_counter[reason] = reason_counter.get(reason, 0) + 1
    summary["top_reasons"] = [
        {"reason": reason, "count": count}
        for reason, count in sorted(reason_counter.items(), key=lambda item: (-item[1], item[0]))[:10]
    ]
    return summary
