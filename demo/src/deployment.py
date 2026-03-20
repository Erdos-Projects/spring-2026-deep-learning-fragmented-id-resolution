import io
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union

import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer

try:
    from .dataset import Vocabulary, compute_pair_features_from_records
    from .model import SiameseNetwork
    from .review_store import ReviewStore, VALID_REVIEW_DECISIONS, canonical_pair
except ImportError:
    from dataset import Vocabulary, compute_pair_features_from_records
    from model import SiameseNetwork
    from review_store import ReviewStore, VALID_REVIEW_DECISIONS, canonical_pair


NORMALIZE_PATTERN = re.compile(r"[\W_]+", re.UNICODE)
DEFAULT_RUNTIME_DIR = Path("data/runtime")
DEFAULT_MODEL_SPECS = {
    "siamese_bilstm": {
        "kind": "siamese",
        "path": "models/siamese_bilstm/best_model.pth",
    },
    "tfidf": {
        "kind": "tfidf",
        "path": "models/tfidf/baseline_metrics.json",
    },
}

DEFAULT_COLUMN_ALIASES = {
    "firstname": "first_name",
    "first": "first_name",
    "lastname": "last_name",
    "last": "last_name",
    "house_number": "house_num",
    "street": "street_name",
    "zipcode": "zip_code",
    "zip": "zip_code",
    "gender": "sex",
    "race": "race_desc",
    "ethnicity": "ethnic_desc",
}
FIELD_LABELS = {
    "first_name": "First name",
    "midl_name": "Middle name",
    "last_name": "Last name",
    "house_num": "House number",
    "street_name": "Street name",
    "zip_code": "ZIP",
    "age": "Age",
    "sex": "Sex",
    "race_desc": "Race",
    "ethnic_desc": "Ethnicity",
}
PAIR_COMPARISON_FIELDS = [
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
REVIEW_EXPORT_LIMIT_PER_SIDE = 100


class DeploymentError(RuntimeError):
    pass


class DatasetNotLoadedError(DeploymentError):
    pass


class ModelUnavailableError(DeploymentError):
    pass


def normalize_value(value: Any) -> str:
    cleaned = NORMALIZE_PATTERN.sub(" ", str(value)).strip()
    cleaned = " ".join(token for token in cleaned.split(" ") if token)
    return cleaned.lower()


def canonicalize_columns(columns: Iterable[str]) -> List[str]:
    normalized = []
    for column in columns:
        key = normalize_value(column).replace(" ", "_")
        key = DEFAULT_COLUMN_ALIASES.get(key, key)
        normalized.append(key)
    return normalized


def prepare_runtime_dataset(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        raise ValueError("Uploaded dataset is empty.")

    runtime_df = df.copy()
    runtime_df.columns = canonicalize_columns(runtime_df.columns)
    runtime_df = runtime_df.loc[:, ~runtime_df.columns.str.contains(r"^unnamed")]
    runtime_df = runtime_df.fillna("")

    if "id" not in runtime_df.columns:
        runtime_df.insert(0, "id", [f"runtime_{idx:06d}" for idx in range(1, len(runtime_df) + 1)])
    else:
        runtime_df["id"] = runtime_df["id"].astype(str).str.strip()
        invalid_mask = runtime_df["id"].eq("") | runtime_df["id"].duplicated(keep=False)
        if invalid_mask.any():
            runtime_df.loc[invalid_mask, "id"] = [
                f"runtime_{idx:06d}" for idx in range(1, int(invalid_mask.sum()) + 1)
            ]

    for column in runtime_df.columns:
        if column == "id":
            runtime_df[column] = runtime_df[column].astype(str)
        else:
            runtime_df[column] = runtime_df[column].map(normalize_value)

    ordered_columns = ["id"] + [column for column in runtime_df.columns if column != "id"]
    return runtime_df[ordered_columns].reset_index(drop=True)


def read_uploaded_table(file_bytes: bytes, filename: str) -> pd.DataFrame:
    suffix = Path(filename or "uploaded.tsv").suffix.lower()
    sep = "," if suffix == ".csv" else "\t"
    return pd.read_csv(io.BytesIO(file_bytes), sep=sep, dtype=str).fillna("")


def serialize_record(record: Dict[str, Any], attributes: Sequence[str], tagged: bool = False) -> str:
    tokens = []
    for attr in attributes:
        value = str(record.get(attr, "")).strip()
        if tagged:
            tokens.append(f"{attr}:{value}")
        else:
            tokens.append(value)
    return " ".join(token for token in tokens if token).strip()


def _add_pairs_from_ids(
    record_ids: Sequence[str], pair_set: set[tuple[str, str]], max_candidate_pairs: int
) -> None:
    sorted_ids = sorted({str(record_id) for record_id in record_ids})
    for idx, left_id in enumerate(sorted_ids):
        for right_id in sorted_ids[idx + 1 :]:
            pair_set.add((left_id, right_id))
            if len(pair_set) > max_candidate_pairs:
                raise DeploymentError(
                    f"Candidate pair limit exceeded ({max_candidate_pairs}). Use stricter blocking or a higher limit."
                )


def generate_candidate_pairs(
    data_df: pd.DataFrame,
    blocking_keys: Sequence[str],
    blocking_mode: str,
    max_candidate_pairs: int,
) -> pd.DataFrame:
    if len(data_df) < 2:
        return pd.DataFrame(columns=["id1", "id2"])

    if not blocking_keys:
        total_pairs = len(data_df) * (len(data_df) - 1) // 2
        if total_pairs > max_candidate_pairs:
            raise DeploymentError(
                f"All-pairs candidate generation exceeds limit ({max_candidate_pairs}). "
                "Enable blocking or increase max_candidate_pairs."
            )
        rows = []
        ids = data_df["id"].astype(str).tolist()
        for idx, left_id in enumerate(ids):
            for right_id in ids[idx + 1 :]:
                rows.append({"id1": left_id, "id2": right_id})
        return pd.DataFrame(rows)

    missing_keys = [key for key in blocking_keys if key not in data_df.columns]
    if missing_keys:
        raise ValueError(f"Blocking keys not found in dataset: {', '.join(sorted(missing_keys))}")

    pair_set: set[tuple[str, str]] = set()
    if blocking_mode == "all":
        grouped = data_df.groupby(list(blocking_keys), dropna=False)["id"].apply(list)
        for key_values, group_ids in grouped.items():
            values = key_values if isinstance(key_values, tuple) else (key_values,)
            if all(str(value).strip() == "" for value in values):
                continue
            _add_pairs_from_ids(group_ids, pair_set, max_candidate_pairs)
    elif blocking_mode == "any":
        for key in blocking_keys:
            grouped = data_df.groupby(key, dropna=False)["id"].apply(list)
            for block_value, group_ids in grouped.items():
                if str(block_value).strip() == "":
                    continue
                _add_pairs_from_ids(group_ids, pair_set, max_candidate_pairs)
    else:
        raise ValueError(f"Unknown blocking_mode '{blocking_mode}'.")

    return pd.DataFrame(
        [{"id1": left_id, "id2": right_id} for left_id, right_id in sorted(pair_set)]
    )


def filter_entry_candidates(
    entry: Dict[str, Any],
    data_df: pd.DataFrame,
    blocking_keys: Sequence[str],
    blocking_mode: str,
    max_candidates: int,
) -> pd.DataFrame:
    if data_df.empty:
        return data_df.copy()

    if not blocking_keys:
        if len(data_df) > max_candidates:
            raise DeploymentError(
                f"Entry comparison requires {len(data_df)} candidates, above limit {max_candidates}. "
                "Enable blocking or increase max_candidates."
            )
        return data_df.copy()

    usable_keys = [key for key in blocking_keys if str(entry.get(key, "")).strip() != ""]
    if not usable_keys:
        if len(data_df) > max_candidates:
            raise DeploymentError(
                "Entry has no usable blocking values and the full dataset exceeds the candidate limit. "
                "Provide blocking fields or increase max_candidates."
            )
        return data_df.copy()

    if blocking_mode == "all":
        mask = np.ones(len(data_df), dtype=bool)
        for key in usable_keys:
            mask &= data_df[key].astype(str).eq(str(entry.get(key, "")))
    elif blocking_mode == "any":
        mask = np.zeros(len(data_df), dtype=bool)
        for key in usable_keys:
            mask |= data_df[key].astype(str).eq(str(entry.get(key, "")))
    else:
        raise ValueError(f"Unknown blocking_mode '{blocking_mode}'.")

    candidates = data_df.loc[mask].copy()
    if len(candidates) > max_candidates:
        raise DeploymentError(
            f"Entry comparison produced {len(candidates)} candidates, above limit {max_candidates}. "
            "Use stricter blocking or a lower-volume dataset."
        )
    return candidates


def _compute_clusters_and_membership(
    pairs_df: pd.DataFrame,
) -> tuple[List[Dict[str, Any]], Dict[str, str]]:
    if pairs_df.empty:
        return [], {}

    parent: Dict[str, str] = {}

    def find(node: str) -> str:
        parent.setdefault(node, node)
        if parent[node] != node:
            parent[node] = find(parent[node])
        return parent[node]

    def union(left: str, right: str) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for row in pairs_df.itertuples(index=False):
        union(str(row.id1), str(row.id2))

    clusters: Dict[str, set[str]] = {}
    for node in set(pairs_df["id1"].astype(str)).union(set(pairs_df["id2"].astype(str))):
        clusters.setdefault(find(node), set()).add(node)

    cluster_rows = [
        {
            "cluster_id": f"cluster_{idx:03d}",
            "record_ids": sorted(record_ids),
            "size": len(record_ids),
        }
        for idx, record_ids in enumerate(sorted(clusters.values(), key=lambda values: (-len(values), sorted(values))))
    ]
    cluster_membership = {}
    for cluster in cluster_rows:
        for record_id in cluster["record_ids"]:
            cluster_membership[record_id] = cluster["cluster_id"]
    return cluster_rows, cluster_membership


def build_duplicate_clusters(pairs_df: pd.DataFrame) -> List[Dict[str, Any]]:
    clusters, _ = _compute_clusters_and_membership(pairs_df)
    return clusters


def summarize_duplicate_clusters(
    pairs_df: pd.DataFrame,
    data_lookup: Dict[str, Dict[str, Any]],
    display_attributes: Sequence[str],
    top_k: int = 5,
) -> Dict[str, Any]:
    clusters, _ = _compute_clusters_and_membership(pairs_df)
    if not clusters:
        return {
            "total_cluster_count": 0,
            "average_cluster_size": 0.0,
            "largest_cluster_size": 0,
            "two_record_cluster_count": 0,
            "three_plus_record_cluster_count": 0,
            "top_clusters": [],
        }

    preferred_order = [
        "first_name",
        "last_name",
        "age",
        "sex",
        "house_num",
        "street_name",
        "zip_code",
        "race_desc",
        "ethnic_desc",
    ]
    ordered_attributes = [
        attr for attr in preferred_order if attr in display_attributes
    ] + [
        attr for attr in display_attributes if attr not in preferred_order
    ]

    enriched_clusters = []
    cluster_sizes = []
    for cluster in clusters:
        member_records = [
            {"id": record_id, **data_lookup.get(record_id, {})}
            for record_id in cluster["record_ids"]
        ]
        cluster_sizes.append(cluster["size"])

        characteristic_rows = []
        for attr in ordered_attributes:
            values = [str(record.get(attr, "")).strip() for record in member_records]
            values = [value for value in values if value]
            if not values:
                continue
            counts = pd.Series(values).value_counts()
            top_value = str(counts.index[0])
            support_count = int(counts.iloc[0])
            support_ratio = support_count / max(cluster["size"], 1)
            characteristic_rows.append(
                {
                    "attribute": attr,
                    "value": top_value,
                    "support_count": support_count,
                    "support_ratio": support_ratio,
                }
            )

        characteristic_rows.sort(
            key=lambda row: (-row["support_ratio"], -row["support_count"], ordered_attributes.index(row["attribute"]))
        )

        member_preview = []
        for record in member_records[:3]:
            preview_bits = [
                " ".join(filter(None, [record.get("first_name", ""), record.get("last_name", "")])).strip(),
                " ".join(filter(None, [record.get("house_num", ""), record.get("street_name", "")])).strip(),
                record.get("zip_code", ""),
                record.get("age", ""),
                record.get("sex", ""),
            ]
            preview_bits = [bit for bit in preview_bits if bit]
            member_preview.append(", ".join(preview_bits))

        enriched_clusters.append(
            {
                "cluster_id": cluster["cluster_id"],
                "size": cluster["size"],
                "shared_characteristics": characteristic_rows[:5],
                "member_preview": member_preview,
            }
        )

    enriched_clusters.sort(key=lambda cluster: (-cluster["size"], cluster["cluster_id"]))
    return {
        "total_cluster_count": int(len(clusters)),
        "average_cluster_size": float(np.mean(cluster_sizes)) if cluster_sizes else 0.0,
        "largest_cluster_size": int(max(cluster_sizes)) if cluster_sizes else 0,
        "two_record_cluster_count": int(sum(1 for size in cluster_sizes if size == 2)),
        "three_plus_record_cluster_count": int(sum(1 for size in cluster_sizes if size >= 3)),
        "top_clusters": enriched_clusters[:top_k],
    }


def _safe_value(record: Dict[str, Any], attribute: str) -> str:
    return str(record.get(attribute, "")).strip()


def _pair_signals(record1: Dict[str, Any], record2: Dict[str, Any]) -> List[str]:
    same_first = _safe_value(record1, "first_name") and _safe_value(record1, "first_name") == _safe_value(record2, "first_name")
    same_middle = _safe_value(record1, "midl_name") and _safe_value(record1, "midl_name") == _safe_value(record2, "midl_name")
    same_last = _safe_value(record1, "last_name") and _safe_value(record1, "last_name") == _safe_value(record2, "last_name")
    same_age = _safe_value(record1, "age") and _safe_value(record1, "age") == _safe_value(record2, "age")
    same_sex = _safe_value(record1, "sex") and _safe_value(record1, "sex") == _safe_value(record2, "sex")
    same_house = _safe_value(record1, "house_num") and _safe_value(record1, "house_num") == _safe_value(record2, "house_num")
    same_street = _safe_value(record1, "street_name") and _safe_value(record1, "street_name") == _safe_value(record2, "street_name")
    same_zip = _safe_value(record1, "zip_code") and _safe_value(record1, "zip_code") == _safe_value(record2, "zip_code")
    same_address = same_house and same_street

    signals = []
    if same_first and same_last:
        signals.append("same full name")
    elif same_last:
        signals.append("same last name")
    elif same_first:
        signals.append("same first name")
    if same_middle:
        signals.append("same middle name")

    if same_address:
        signals.append("same street address")
    elif any(
        _safe_value(record1, key) != _safe_value(record2, key)
        for key in ["house_num", "street_name"]
        if _safe_value(record1, key) or _safe_value(record2, key)
    ):
        signals.append("address differs")

    if same_zip:
        signals.append("same ZIP")
    elif _safe_value(record1, "zip_code") and _safe_value(record2, "zip_code"):
        signals.append("ZIP differs")

    if same_age:
        signals.append("same age")
    elif _safe_value(record1, "age") and _safe_value(record2, "age"):
        signals.append("age differs")

    if not same_sex and _safe_value(record1, "sex") and _safe_value(record2, "sex"):
        signals.append("sex differs")

    return signals[:5]


def _pair_field_comparison(record1: Dict[str, Any], record2: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    matching_fields = []
    differing_fields = []

    for attribute in PAIR_COMPARISON_FIELDS:
        left_value = _safe_value(record1, attribute)
        right_value = _safe_value(record2, attribute)
        if not left_value and not right_value:
            continue

        field_row = {
            "attribute": attribute,
            "label": FIELD_LABELS.get(attribute, attribute),
            "left": left_value,
            "right": right_value,
        }
        if left_value == right_value:
            if left_value:
                matching_fields.append({**field_row, "value": left_value})
        else:
            differing_fields.append(field_row)

    return {
        "matching_fields": matching_fields,
        "differing_fields": differing_fields,
    }


def _build_pair_payload(
    id1: str,
    id2: str,
    score: float,
    data_lookup: Dict[str, Dict[str, Any]],
    display_attributes: Sequence[str],
    cluster_id: Optional[str] = None,
    comparison_score: Optional[float] = None,
    comparison_model_name: Optional[str] = None,
    comparison_prediction: Optional[bool] = None,
) -> Dict[str, Any]:
    _, _, pair_key = canonical_pair(id1, id2)
    record1 = {"id": str(id1), **data_lookup[str(id1)]}
    record2 = {"id": str(id2), **data_lookup[str(id2)]}
    payload = {
        "pair_key": pair_key,
        "id1": str(id1),
        "id2": str(id2),
        "score": float(score),
        "cluster_id": cluster_id,
        "record1": _record_excerpt(record1, display_attributes),
        "record2": _record_excerpt(record2, display_attributes),
        "signals": _pair_signals(record1, record2),
        **_pair_field_comparison(record1, record2),
    }
    if comparison_score is not None:
        payload["comparison_score"] = float(comparison_score)
    if comparison_model_name is not None:
        payload["comparison_model_name"] = comparison_model_name
    if comparison_prediction is not None:
        payload["comparison_prediction"] = bool(comparison_prediction)
    return payload


def _attach_review_metadata(
    pair_payloads: List[Dict[str, Any]],
    review_lookup: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    for payload in pair_payloads:
        payload["review_state"] = review_lookup.get(str(payload.get("pair_key")))
    return pair_payloads


def _select_display_rows(
    results_df: pd.DataFrame,
    limit: int,
    ascending: bool = False,
    cluster_membership: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    if results_df.empty or limit <= 0:
        return results_df.iloc[0:0].copy()

    sorted_df = results_df.sort_values(["score", "id1", "id2"], ascending=[ascending, True, True]).copy()
    if not cluster_membership:
        return sorted_df.head(limit).copy()

    selected_rows = []
    selected_clusters = set()

    for row in sorted_df.itertuples(index=False):
        cluster_id = cluster_membership.get(str(row.id1)) or cluster_membership.get(str(row.id2))
        if cluster_id and cluster_id in selected_clusters:
            continue
        selected_rows.append({"id1": str(row.id1), "id2": str(row.id2)})
        if cluster_id:
            selected_clusters.add(cluster_id)
        if len(selected_rows) >= limit:
            break

    if len(selected_rows) < limit:
        already_selected = {(row["id1"], row["id2"]) for row in selected_rows}
        for row in sorted_df.itertuples(index=False):
            key = (str(row.id1), str(row.id2))
            if key in already_selected:
                continue
            selected_rows.append({"id1": key[0], "id2": key[1]})
            if len(selected_rows) >= limit:
                break

    selected_keys = {(row["id1"], row["id2"]) for row in selected_rows}
    selected_df = sorted_df[
        sorted_df.apply(lambda row: (str(row["id1"]), str(row["id2"])) in selected_keys, axis=1)
    ].copy()
    selected_df["__rank"] = selected_df.apply(
        lambda row: next(
            idx
            for idx, candidate in enumerate(selected_rows)
            if candidate["id1"] == str(row["id1"]) and candidate["id2"] == str(row["id2"])
        ),
        axis=1,
    )
    selected_df = selected_df.sort_values("__rank").drop(columns="__rank")
    return selected_df


def _rows_to_pair_payloads(
    rows_df: pd.DataFrame,
    data_lookup: Dict[str, Dict[str, Any]],
    display_attributes: Sequence[str],
    cluster_membership: Optional[Dict[str, str]] = None,
    comparison_model_name: Optional[str] = None,
) -> List[Dict[str, Any]]:
    payloads = []
    for row in rows_df.itertuples(index=False):
        cluster_id = None
        if cluster_membership:
            cluster_id = cluster_membership.get(str(row.id1)) or cluster_membership.get(str(row.id2))
        payloads.append(
            _build_pair_payload(
                id1=str(row.id1),
                id2=str(row.id2),
                score=float(row.score),
                data_lookup=data_lookup,
                display_attributes=display_attributes,
                cluster_id=cluster_id,
                comparison_score=getattr(row, "comparison_score", None),
                comparison_model_name=comparison_model_name,
                comparison_prediction=getattr(row, "comparison_is_duplicate", None),
            )
        )
    return payloads


def summarize_duplicate_patterns(
    duplicate_df: pd.DataFrame,
    data_lookup: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    if duplicate_df.empty:
        return []

    pattern_counts = {
        "Same full name": 0,
        "Same full name, different address": 0,
        "Same household, different name": 0,
        "Same ZIP, different address": 0,
        "Same age, different last name": 0,
        "Same last name + ZIP, different first name": 0,
    }

    for row in duplicate_df.itertuples(index=False):
        record1 = data_lookup[str(row.id1)]
        record2 = data_lookup[str(row.id2)]

        same_first = _safe_value(record1, "first_name") == _safe_value(record2, "first_name")
        same_last = _safe_value(record1, "last_name") == _safe_value(record2, "last_name")
        same_name = same_first and same_last and (_safe_value(record1, "first_name") or _safe_value(record1, "last_name"))
        same_age = _safe_value(record1, "age") and _safe_value(record1, "age") == _safe_value(record2, "age")
        same_household = (
            _safe_value(record1, "house_num")
            and _safe_value(record1, "house_num") == _safe_value(record2, "house_num")
            and _safe_value(record1, "street_name")
            and _safe_value(record1, "street_name") == _safe_value(record2, "street_name")
        )
        same_zip = _safe_value(record1, "zip_code") and _safe_value(record1, "zip_code") == _safe_value(record2, "zip_code")
        address_diff = any(
            _safe_value(record1, key) != _safe_value(record2, key)
            for key in ["house_num", "street_name", "zip_code"]
            if _safe_value(record1, key) or _safe_value(record2, key)
        )

        if same_name:
            pattern_counts["Same full name"] += 1
        if same_name and address_diff:
            pattern_counts["Same full name, different address"] += 1
        if same_household and not same_name:
            pattern_counts["Same household, different name"] += 1
        if same_zip and address_diff:
            pattern_counts["Same ZIP, different address"] += 1
        if same_age and _safe_value(record1, "last_name") != _safe_value(record2, "last_name"):
            pattern_counts["Same age, different last name"] += 1
        if (
            _safe_value(record1, "last_name")
            and _safe_value(record1, "last_name") == _safe_value(record2, "last_name")
            and same_zip
            and _safe_value(record1, "first_name") != _safe_value(record2, "first_name")
        ):
            pattern_counts["Same last name + ZIP, different first name"] += 1

    total = len(duplicate_df)
    rows = [
        {
            "label": label,
            "count": int(count),
            "share": float(count / total) if total else 0.0,
        }
        for label, count in pattern_counts.items()
        if count > 0
    ]
    rows.sort(key=lambda row: (-row["count"], row["label"]))
    return rows


def compute_score_summary(results_df: pd.DataFrame) -> Dict[str, Any]:
    duplicate_scores = results_df.loc[results_df["is_duplicate"], "score"].astype(float)
    rejected_scores = results_df.loc[~results_df["is_duplicate"], "score"].astype(float)
    acceptance_rate = float(len(duplicate_scores) / len(results_df)) if len(results_df) else 0.0

    def _summary(values: pd.Series) -> Dict[str, Optional[float]]:
        if values.empty:
            return {"max": None, "median": None, "min": None}
        return {
            "max": float(values.max()),
            "median": float(values.median()),
            "min": float(values.min()),
        }

    return {
        "acceptance_rate": acceptance_rate,
        "duplicates": _summary(duplicate_scores),
        "rejected": _summary(rejected_scores),
    }


def _score_candidate_pairs_with_model(
    model: Any,
    candidates_df: pd.DataFrame,
    data_lookup: Dict[str, Dict[str, Any]],
) -> np.ndarray:
    if isinstance(model, TfidfScorer):
        return model.score_existing_pairs(candidates_df)

    left_records = [{"id": str(record_id), **data_lookup[str(record_id)]} for record_id in candidates_df["id1"].astype(str)]
    right_records = [{"id": str(record_id), **data_lookup[str(record_id)]} for record_id in candidates_df["id2"].astype(str)]
    return model.score_pairs(left_records, right_records)


def summarize_model_disagreements(
    candidate_results_df: pd.DataFrame,
    comparison_scores: np.ndarray,
    comparison_threshold: float,
    comparison_model_name: str,
    data_lookup: Dict[str, Dict[str, Any]],
    display_attributes: Sequence[str],
    top_k: int,
) -> Dict[str, Any]:
    if candidate_results_df.empty:
        return {
            "available": True,
            "comparison_model_name": comparison_model_name,
            "comparison_threshold": comparison_threshold,
            "agreement_rate": 1.0,
            "selected_only_duplicate_count": 0,
            "comparison_only_duplicate_count": 0,
            "selected_only_duplicates": [],
            "comparison_only_duplicates": [],
        }

    comparison_df = candidate_results_df.copy()
    comparison_df["comparison_score"] = comparison_scores.astype(float)
    comparison_df["comparison_is_duplicate"] = comparison_df["comparison_score"] >= float(comparison_threshold)

    agreement_rate = float(
        (comparison_df["is_duplicate"].astype(bool) == comparison_df["comparison_is_duplicate"].astype(bool)).mean()
    )
    selected_only_df = comparison_df.loc[
        comparison_df["is_duplicate"].astype(bool) & ~comparison_df["comparison_is_duplicate"].astype(bool)
    ].copy().sort_values("score", ascending=False)
    comparison_only_df = comparison_df.loc[
        ~comparison_df["is_duplicate"].astype(bool) & comparison_df["comparison_is_duplicate"].astype(bool)
    ].copy().sort_values("comparison_score", ascending=False)

    return {
        "available": True,
        "comparison_model_name": comparison_model_name,
        "comparison_threshold": float(comparison_threshold),
        "agreement_rate": agreement_rate,
        "selected_only_duplicate_count": int(len(selected_only_df)),
        "comparison_only_duplicate_count": int(len(comparison_only_df)),
        "selected_only_duplicates": _rows_to_pair_payloads(
            selected_only_df.head(top_k),
            data_lookup=data_lookup,
            display_attributes=display_attributes,
            comparison_model_name=comparison_model_name,
        ),
        "comparison_only_duplicates": _rows_to_pair_payloads(
            comparison_only_df.head(top_k),
            data_lookup=data_lookup,
            display_attributes=display_attributes,
            comparison_model_name=comparison_model_name,
        ),
    }


def _record_excerpt(record: Dict[str, Any], attributes: Sequence[str]) -> Dict[str, Any]:
    payload = {"id": str(record.get("id", ""))}
    excerpt_attributes = list(dict.fromkeys(list(attributes) + PAIR_COMPARISON_FIELDS))
    for attr in excerpt_attributes:
        if attr in record:
            payload[attr] = record.get(attr, "")
    return payload


@dataclass
class RuntimeDataset:
    frame: pd.DataFrame
    source_name: str
    saved_path: Path

    @property
    def lookup(self) -> Dict[str, Dict[str, Any]]:
        return self.frame.set_index("id").to_dict("index")

    def summary(self) -> Dict[str, Any]:
        return {
            "source_name": self.source_name,
            "record_count": int(len(self.frame)),
            "columns": list(self.frame.columns),
            "saved_path": str(self.saved_path),
        }


class SiameseScorer:
    def __init__(self, name: str, checkpoint_path: Path, device: Optional[str] = None):
        self.name = name
        self.checkpoint_path = Path(checkpoint_path)
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Siamese checkpoint not found: {self.checkpoint_path}")

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        config = checkpoint["config"]
        self.attributes = list(config["attributes"])
        self.blocking_keys = list(config.get("blocking_keys", []))
        self.blocking_mode = config.get("blocking_mode", "any")
        self.max_len = int(config["max_len"])
        self.threshold = float(checkpoint["best_threshold"])
        self.vocab = Vocabulary.from_dict(checkpoint["vocab"])
        self.model = SiameseNetwork(
            vocab_size=self.vocab.vocab_size,
            embedding_dim=int(config["embedding_dim"]),
            hidden_dim=int(config["hidden_dim"]),
            encoder_type=config.get("encoder_type", "bilstm"),
            cnn_channels=int(config.get("cnn_channels", 64)),
            cnn_kernel_sizes=config.get("cnn_kernel_sizes", (3, 4, 5)),
            classifier_hidden_dim=int(config.get("classifier_hidden_dim", 64)),
            pair_feature_dim=int(config.get("pair_feature_dim", 0)),
        ).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        self.pair_feature_set = config.get("pair_feature_set", "none")
        self.metadata = {
            "kind": "siamese",
            "path": str(self.checkpoint_path),
            "attributes": self.attributes,
            "threshold": self.threshold,
            "blocking_keys": self.blocking_keys,
            "blocking_mode": self.blocking_mode,
            "encoder_type": config.get("encoder_type", "bilstm"),
            "max_len": self.max_len,
            "pair_feature_set": self.pair_feature_set,
        }

    def score_pairs(
        self,
        left_records: Sequence[Dict[str, Any]],
        right_records: Sequence[Dict[str, Any]],
        batch_size: int = 256,
    ) -> np.ndarray:
        if len(left_records) != len(right_records):
            raise ValueError("left_records and right_records must have the same length.")
        if not left_records:
            return np.zeros(0, dtype=float)

        chunks = []
        with torch.no_grad():
            for start in range(0, len(left_records), batch_size):
                left_batch = left_records[start : start + batch_size]
                right_batch = right_records[start : start + batch_size]
                x1 = [
                    self.vocab.numericalize(serialize_record(record, self.attributes), self.max_len)
                    for record in left_batch
                ]
                x2 = [
                    self.vocab.numericalize(serialize_record(record, self.attributes), self.max_len)
                    for record in right_batch
                ]
                x1_tensor = torch.tensor(x1, dtype=torch.long, device=self.device)
                x2_tensor = torch.tensor(x2, dtype=torch.long, device=self.device)
                pair_features = None
                if self.pair_feature_set != "none":
                    pair_features = torch.tensor(
                        [
                            compute_pair_features_from_records(left_record, right_record, self.pair_feature_set)
                            for left_record, right_record in zip(left_batch, right_batch)
                        ],
                        dtype=torch.float,
                        device=self.device,
                    )
                logits = self.model(x1_tensor, x2_tensor, pair_features=pair_features).squeeze(-1)
                chunks.append(torch.sigmoid(logits).detach().cpu().numpy())
        return np.concatenate(chunks).astype(float)


class TfidfScorer:
    def __init__(self, name: str, config_path: Path):
        self.name = name
        self.config_path = Path(config_path)
        if not self.config_path.exists():
            raise FileNotFoundError(f"TF-IDF baseline config not found: {self.config_path}")

        with open(self.config_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)

        vectorizer_config = payload["vectorizer"]
        self.attributes = list(payload["attributes"])
        self.blocking_keys = list(payload.get("blocking_keys", []))
        self.blocking_mode = payload.get("blocking_mode", "any")
        self.tagged_serialization = bool(payload.get("tagged_serialization", False))
        self.threshold = float(payload["selected_similarity_threshold"])
        self.vectorizer = TfidfVectorizer(
            analyzer=vectorizer_config["analyzer"],
            ngram_range=tuple(vectorizer_config["ngram_range"]),
            min_df=int(vectorizer_config["min_df"]),
            max_features=vectorizer_config["max_features"],
        )
        self.matrix = None
        self.record_to_row: Dict[str, int] = {}
        self.metadata = {
            "kind": "tfidf",
            "path": str(self.config_path),
            "attributes": self.attributes,
            "threshold": self.threshold,
            "blocking_keys": self.blocking_keys,
            "blocking_mode": self.blocking_mode,
            "tagged_serialization": self.tagged_serialization,
            "vectorizer": vectorizer_config,
        }

    def fit(self, data_df: pd.DataFrame) -> None:
        texts = data_df.apply(
            lambda row: serialize_record(row.to_dict(), self.attributes, tagged=self.tagged_serialization), axis=1
        )
        self.matrix = self.vectorizer.fit_transform(texts)
        self.record_to_row = {record_id: idx for idx, record_id in enumerate(data_df["id"].astype(str))}

    def score_existing_pairs(self, pairs_df: pd.DataFrame) -> np.ndarray:
        if self.matrix is None:
            raise DeploymentError("TF-IDF scorer has not been fit on a dataset yet.")
        if pairs_df.empty:
            return np.zeros(0, dtype=float)

        row_idx_1 = [self.record_to_row[str(record_id)] for record_id in pairs_df["id1"].astype(str)]
        row_idx_2 = [self.record_to_row[str(record_id)] for record_id in pairs_df["id2"].astype(str)]
        similarities = np.asarray(self.matrix[row_idx_1].multiply(self.matrix[row_idx_2]).sum(axis=1)).ravel()
        return similarities.astype(float)

    def score_entry_against_candidates(self, entry_record: Dict[str, Any], candidates_df: pd.DataFrame) -> np.ndarray:
        if self.matrix is None:
            raise DeploymentError("TF-IDF scorer has not been fit on a dataset yet.")
        if candidates_df.empty:
            return np.zeros(0, dtype=float)

        entry_text = serialize_record(entry_record, self.attributes, tagged=self.tagged_serialization)
        entry_vector = self.vectorizer.transform([entry_text])
        row_indices = [self.record_to_row[record_id] for record_id in candidates_df["id"].astype(str)]
        similarities = np.asarray(self.matrix[row_indices].dot(entry_vector.T).todense()).ravel()
        return similarities.astype(float)


class DuplicateDetectionService:
    def __init__(
        self,
        model_specs: Optional[Dict[str, Dict[str, str]]] = None,
        runtime_dir: Union[Path, str] = DEFAULT_RUNTIME_DIR,
        device: Optional[str] = None,
    ):
        self.runtime_dir = Path(runtime_dir)
        self.runtime_dir.mkdir(parents=True, exist_ok=True)
        self.dataset_dir = self.runtime_dir / "datasets"
        self.dataset_dir.mkdir(parents=True, exist_ok=True)
        self.review_store = ReviewStore(self.runtime_dir / "review_store.sqlite3")
        self.current_dataset: Optional[RuntimeDataset] = None
        self.models: Dict[str, Any] = {}
        self.device = device

        specs = model_specs or DEFAULT_MODEL_SPECS
        for model_name, spec in specs.items():
            model_path = Path(spec["path"])
            if not model_path.exists():
                continue
            if spec["kind"] == "siamese":
                self.models[model_name] = SiameseScorer(model_name, model_path, device=device)
            elif spec["kind"] == "tfidf":
                self.models[model_name] = TfidfScorer(model_name, model_path)
            else:
                raise ValueError(f"Unknown model kind '{spec['kind']}' for {model_name}")

    def list_models(self) -> List[Dict[str, Any]]:
        return [{"name": name, **model.metadata} for name, model in sorted(self.models.items())]

    def load_dataset_from_dataframe(self, df: pd.DataFrame, source_name: str) -> Dict[str, Any]:
        prepared = prepare_runtime_dataset(df)
        required_columns = {"id"}
        for model in self.models.values():
            required_columns.update(model.attributes)
            required_columns.update(model.blocking_keys)
        for column in sorted(required_columns):
            if column not in prepared.columns:
                prepared[column] = ""
        prepared = prepared[["id"] + [column for column in prepared.columns if column != "id"]]

        output_path = self.dataset_dir / "current_dataset.tsv"
        prepared.to_csv(output_path, sep="\t", index=False)
        self.current_dataset = RuntimeDataset(frame=prepared, source_name=source_name, saved_path=output_path)

        for model in self.models.values():
            if isinstance(model, TfidfScorer):
                model.fit(prepared)

        summary = self.current_dataset.summary()
        summary["available_models"] = [model["name"] for model in self.list_models()]
        return summary

    def load_dataset_from_file(self, file_bytes: bytes, filename: str) -> Dict[str, Any]:
        frame = read_uploaded_table(file_bytes, filename)
        return self.load_dataset_from_dataframe(frame, source_name=filename)

    def load_dataset_from_path(self, path: Union[str, Path]) -> Dict[str, Any]:
        data_path = Path(path)
        if not data_path.exists():
            raise FileNotFoundError(f"Dataset path does not exist: {data_path}")
        sep = "," if data_path.suffix.lower() == ".csv" else "\t"
        frame = pd.read_csv(data_path, sep=sep, dtype=str).fillna("")
        return self.load_dataset_from_dataframe(frame, source_name=str(data_path))

    def clear_dataset(self) -> None:
        self.current_dataset = None

    def dataset_summary(self) -> Dict[str, Any]:
        if self.current_dataset is None:
            raise DatasetNotLoadedError("No dataset has been loaded.")
        return self.current_dataset.summary()

    def review_summary(self) -> Dict[str, Any]:
        dataset_source = self.current_dataset.source_name if self.current_dataset is not None else None
        return self.review_store.summary(dataset_source=dataset_source)

    def list_review_decisions(self, limit: int = 20) -> List[Dict[str, Any]]:
        dataset_source = self.current_dataset.source_name if self.current_dataset is not None else None
        return self.review_store.list_decisions(dataset_source=dataset_source, limit=limit)

    def save_review_decision(
        self,
        *,
        id1: str,
        id2: str,
        decision: str,
        model_name: str,
        score: Optional[float] = None,
        comparison_model_name: Optional[str] = None,
        comparison_score: Optional[float] = None,
        source_section: Optional[str] = None,
        cluster_id: Optional[str] = None,
        notes: str = "",
    ) -> Dict[str, Any]:
        runtime_dataset = self._require_dataset()
        lookup = runtime_dataset.lookup
        left_id, right_id, _ = canonical_pair(id1, id2)
        if left_id not in lookup or right_id not in lookup:
            raise DeploymentError(f"Pair ids '{id1}' and '{id2}' were not found in the loaded dataset.")

        saved = self.review_store.save_decision(
            id1=left_id,
            id2=right_id,
            dataset_source=runtime_dataset.source_name,
            model_name=model_name,
            decision=decision,
            notes=notes,
            score=score,
            comparison_model_name=comparison_model_name,
            comparison_score=comparison_score,
            source_section=source_section,
            cluster_id=cluster_id,
            record1=_record_excerpt({"id": left_id, **lookup[left_id]}, PAIR_COMPARISON_FIELDS),
            record2=_record_excerpt({"id": right_id, **lookup[right_id]}, PAIR_COMPARISON_FIELDS),
        )
        return {
            "saved_review": saved,
            "review_summary": self.review_summary(),
            "recent_reviews": self.list_review_decisions(limit=10),
        }

    def _require_dataset(self) -> RuntimeDataset:
        if self.current_dataset is None:
            raise DatasetNotLoadedError("No dataset has been loaded.")
        return self.current_dataset

    def _resolve_model(self, model_name: str):
        if model_name not in self.models:
            raise ModelUnavailableError(f"Model '{model_name}' is not available.")
        return self.models[model_name]

    def _resolve_threshold(self, model: Any, threshold_override: Optional[float]) -> float:
        return float(model.threshold if threshold_override is None else threshold_override)

    def _choose_comparison_model_name(self, selected_model_name: str) -> Optional[str]:
        if selected_model_name != "tfidf" and "tfidf" in self.models:
            return "tfidf"
        if selected_model_name != "siamese_bilstm" and "siamese_bilstm" in self.models:
            return "siamese_bilstm"
        available_names = [name for name in self.models if name != selected_model_name]
        return available_names[0] if available_names else None

    def find_duplicates(
        self,
        model_name: str = "siamese_bilstm",
        blocking_keys: Optional[Sequence[str]] = None,
        blocking_mode: Optional[str] = None,
        threshold: Optional[float] = None,
        max_candidate_pairs: int = 250000,
        top_k: int = 10,
        include_disagreement_analysis: bool = False,
    ) -> Dict[str, Any]:
        runtime_dataset = self._require_dataset()
        model = self._resolve_model(model_name)
        blocking_keys = list(model.blocking_keys if blocking_keys is None else blocking_keys)
        blocking_mode = model.blocking_mode if blocking_mode is None else blocking_mode
        threshold_value = self._resolve_threshold(model, threshold)

        candidates_df = generate_candidate_pairs(
            runtime_dataset.frame,
            blocking_keys=blocking_keys,
            blocking_mode=blocking_mode,
            max_candidate_pairs=max_candidate_pairs,
        )

        lookup = runtime_dataset.lookup
        scores = _score_candidate_pairs_with_model(model, candidates_df, lookup)

        results_df = candidates_df.copy()
        results_df["score"] = scores
        results_df["is_duplicate"] = results_df["score"] >= threshold_value
        duplicate_df = results_df.loc[results_df["is_duplicate"]].copy().sort_values("score", ascending=False)
        rejected_df = results_df.loc[~results_df["is_duplicate"]].copy().sort_values("score", ascending=False)
        _, cluster_membership = _compute_clusters_and_membership(duplicate_df[["id1", "id2"]])
        cluster_summary = summarize_duplicate_clusters(
            duplicate_df[["id1", "id2"]],
            data_lookup=lookup,
            display_attributes=model.attributes,
            top_k=5,
        )

        high_confidence_df = _select_display_rows(
            duplicate_df,
            limit=top_k,
            ascending=False,
            cluster_membership=cluster_membership,
        )
        borderline_df = _select_display_rows(
            duplicate_df,
            limit=top_k,
            ascending=True,
            cluster_membership=cluster_membership,
        )
        near_miss_df = rejected_df.head(top_k).copy()

        display_attributes = model.attributes
        high_confidence_duplicates = _rows_to_pair_payloads(
            high_confidence_df,
            data_lookup=lookup,
            display_attributes=display_attributes,
            cluster_membership=cluster_membership,
        )
        borderline_duplicates = _rows_to_pair_payloads(
            borderline_df,
            data_lookup=lookup,
            display_attributes=display_attributes,
            cluster_membership=cluster_membership,
        )
        near_miss_non_duplicates = _rows_to_pair_payloads(
            near_miss_df,
            data_lookup=lookup,
            display_attributes=display_attributes,
        )
        export_duplicate_pairs = _rows_to_pair_payloads(
            duplicate_df,
            data_lookup=lookup,
            display_attributes=display_attributes,
            cluster_membership=cluster_membership,
        )
        export_review_pairs = _rows_to_pair_payloads(
            pd.concat(
                [
                    duplicate_df.sort_values("score", ascending=True).head(REVIEW_EXPORT_LIMIT_PER_SIDE),
                    rejected_df.sort_values("score", ascending=False).head(REVIEW_EXPORT_LIMIT_PER_SIDE),
                ],
                ignore_index=True,
            ),
            data_lookup=lookup,
            display_attributes=display_attributes,
            cluster_membership=cluster_membership,
        )
        representative_cases = {
            "highest_confidence_duplicate": high_confidence_duplicates[0] if high_confidence_duplicates else None,
            "most_borderline_duplicate": borderline_duplicates[0] if borderline_duplicates else None,
            "closest_rejected_pair": near_miss_non_duplicates[0] if near_miss_non_duplicates else None,
            "largest_cluster": cluster_summary["top_clusters"][0] if cluster_summary["top_clusters"] else None,
        }

        disagreement_analysis: Dict[str, Any]
        comparison_model_name = self._choose_comparison_model_name(model_name)
        comparison_candidate_limit = 50000
        if not include_disagreement_analysis:
            disagreement_analysis = {
                "requested": False,
                "available": False,
                "reason": "Disagreement analysis was not requested for this search.",
            }
        elif comparison_model_name is None:
            disagreement_analysis = {
                "requested": True,
                "available": False,
                "reason": "No comparison model is configured for this deployment.",
            }
        elif len(results_df) > comparison_candidate_limit:
            disagreement_analysis = {
                "requested": True,
                "available": False,
                "reason": (
                    f"Disagreement analysis skipped because {len(results_df)} candidate pairs exceed "
                    f"the comparison limit of {comparison_candidate_limit}."
                ),
            }
        else:
            comparison_model = self._resolve_model(comparison_model_name)
            comparison_scores = _score_candidate_pairs_with_model(comparison_model, candidates_df, lookup)
            disagreement_analysis = summarize_model_disagreements(
                candidate_results_df=results_df,
                comparison_scores=comparison_scores,
                comparison_threshold=float(comparison_model.threshold),
                comparison_model_name=comparison_model_name,
                data_lookup=lookup,
                display_attributes=list(dict.fromkeys(model.attributes + comparison_model.attributes)),
                top_k=min(top_k, 10),
            )
            disagreement_analysis["requested"] = True

        all_review_pairs = (
            high_confidence_duplicates
            + borderline_duplicates
            + near_miss_non_duplicates
            + export_duplicate_pairs
            + export_review_pairs
            + disagreement_analysis.get("selected_only_duplicates", [])
            + disagreement_analysis.get("comparison_only_duplicates", [])
        )
        review_lookup = self.review_store.get_decisions_for_pairs(
            [(payload["id1"], payload["id2"]) for payload in all_review_pairs]
        )
        high_confidence_duplicates = _attach_review_metadata(high_confidence_duplicates, review_lookup)
        borderline_duplicates = _attach_review_metadata(borderline_duplicates, review_lookup)
        near_miss_non_duplicates = _attach_review_metadata(near_miss_non_duplicates, review_lookup)
        export_duplicate_pairs = _attach_review_metadata(export_duplicate_pairs, review_lookup)
        export_review_pairs = _attach_review_metadata(export_review_pairs, review_lookup)
        review_queue = export_review_pairs[:top_k]
        review_summary = self.review_summary()
        recent_reviews = self.list_review_decisions(limit=10)
        disagreement_analysis["selected_only_duplicates"] = _attach_review_metadata(
            disagreement_analysis.get("selected_only_duplicates", []),
            review_lookup,
        )
        disagreement_analysis["comparison_only_duplicates"] = _attach_review_metadata(
            disagreement_analysis.get("comparison_only_duplicates", []),
            review_lookup,
        )

        return {
            "model_name": model_name,
            "threshold": threshold_value,
            "blocking_keys": blocking_keys,
            "blocking_mode": blocking_mode,
            "dataset": runtime_dataset.summary(),
            "candidate_pair_count": int(len(results_df)),
            "predicted_duplicate_pair_count": int(len(duplicate_df)),
            "duplicate_pairs": high_confidence_duplicates,
            "high_confidence_duplicates": high_confidence_duplicates,
            "borderline_duplicates": borderline_duplicates,
            "near_miss_non_duplicates": near_miss_non_duplicates,
            "score_summary": compute_score_summary(results_df),
            "duplicate_cluster_summary": cluster_summary,
            "duplicate_pattern_summary": summarize_duplicate_patterns(duplicate_df, lookup),
            "representative_cases": representative_cases,
            "disagreement_analysis": disagreement_analysis,
            "review_queue": review_queue,
            "review_summary": review_summary,
            "recent_reviews": recent_reviews,
            "exports": {
                "duplicates": export_duplicate_pairs,
                "review": export_review_pairs,
            },
        }

    def check_entry(
        self,
        entry: Dict[str, Any],
        model_name: str = "siamese_bilstm",
        blocking_keys: Optional[Sequence[str]] = None,
        blocking_mode: Optional[str] = None,
        threshold: Optional[float] = None,
        top_k: int = 10,
        max_candidates: int = 50000,
    ) -> Dict[str, Any]:
        runtime_dataset = self._require_dataset()
        model = self._resolve_model(model_name)
        blocking_keys = list(model.blocking_keys if blocking_keys is None else blocking_keys)
        blocking_mode = model.blocking_mode if blocking_mode is None else blocking_mode
        threshold_value = self._resolve_threshold(model, threshold)

        entry_df = prepare_runtime_dataset(pd.DataFrame([entry]))
        entry_record = entry_df.iloc[0].to_dict()

        candidates_df = filter_entry_candidates(
            entry=entry_record,
            data_df=runtime_dataset.frame,
            blocking_keys=blocking_keys,
            blocking_mode=blocking_mode,
            max_candidates=max_candidates,
        )

        lookup = runtime_dataset.lookup
        if isinstance(model, TfidfScorer):
            scores = model.score_entry_against_candidates(entry_record, candidates_df)
        else:
            left_records = [entry_record] * len(candidates_df)
            right_records = [{"id": str(record_id), **lookup[str(record_id)]} for record_id in candidates_df["id"].astype(str)]
            scores = model.score_pairs(left_records, right_records)

        results_df = candidates_df[["id"]].copy()
        results_df["score"] = scores
        results_df["is_duplicate"] = results_df["score"] >= threshold_value
        results_df = results_df.sort_values("score", ascending=False)
        top_matches_df = results_df.head(top_k)

        matches = [
            {
                "existing_id": str(row.id),
                "score": float(row.score),
                "is_duplicate": bool(row.is_duplicate),
                "existing_record": _record_excerpt({"id": str(row.id), **lookup[str(row.id)]}, model.attributes),
            }
            for row in top_matches_df.itertuples(index=False)
        ]

        return {
            "model_name": model_name,
            "threshold": threshold_value,
            "blocking_keys": blocking_keys,
            "blocking_mode": blocking_mode,
            "candidate_count": int(len(results_df)),
            "duplicate_exists": bool(results_df["is_duplicate"].any()) if len(results_df) else False,
            "entry": _record_excerpt(entry_record, model.attributes),
            "matches": matches,
        }


def build_default_service(
    runtime_dir: Union[Path, str] = DEFAULT_RUNTIME_DIR,
    device: Optional[str] = None,
) -> DuplicateDetectionService:
    return DuplicateDetectionService(runtime_dir=runtime_dir, device=device)
