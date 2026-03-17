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
    from .dataset import Vocabulary
    from .model import SiameseNetwork
except ImportError:
    from dataset import Vocabulary
    from model import SiameseNetwork


NORMALIZE_PATTERN = re.compile(r"[\W_]+", re.UNICODE)
DEFAULT_RUNTIME_DIR = Path("data/runtime")
DEFAULT_MODEL_SPECS = {
    "siamese_bilstm": {
        "kind": "siamese",
        "path": "models/comparisons/apples_to_apples_with_charcnn/siamese_bilstm/blocked_pipeline/extended/best_model.pth",
    },
    "siamese_charcnn": {
        "kind": "siamese",
        "path": "models/comparisons/apples_to_apples_with_charcnn/siamese_charcnn/blocked_pipeline/extended/best_model.pth",
    },
    "tfidf": {
        "kind": "tfidf",
        "path": "models/comparisons/apples_to_apples_with_charcnn/tfidf/blocked_pipeline/extended/baseline_metrics.json",
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


def build_duplicate_clusters(pairs_df: pd.DataFrame) -> List[Dict[str, Any]]:
    if pairs_df.empty:
        return []

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

    return [
        {
            "cluster_id": f"cluster_{idx:03d}",
            "record_ids": sorted(record_ids),
            "size": len(record_ids),
        }
        for idx, record_ids in enumerate(sorted(clusters.values(), key=lambda values: (-len(values), sorted(values))))
    ]


def _record_excerpt(record: Dict[str, Any], attributes: Sequence[str]) -> Dict[str, Any]:
    payload = {"id": str(record.get("id", ""))}
    for attr in attributes:
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
        ).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        self.metadata = {
            "kind": "siamese",
            "path": str(self.checkpoint_path),
            "attributes": self.attributes,
            "threshold": self.threshold,
            "blocking_keys": self.blocking_keys,
            "blocking_mode": self.blocking_mode,
            "encoder_type": config.get("encoder_type", "bilstm"),
            "max_len": self.max_len,
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
                logits = self.model(x1_tensor, x2_tensor).squeeze(-1)
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

    def find_duplicates(
        self,
        model_name: str = "siamese_bilstm",
        blocking_keys: Optional[Sequence[str]] = None,
        blocking_mode: Optional[str] = None,
        threshold: Optional[float] = None,
        max_candidate_pairs: int = 250000,
        top_k: int = 50,
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
        if isinstance(model, TfidfScorer):
            scores = model.score_existing_pairs(candidates_df)
        else:
            left_records = [{"id": str(record_id), **lookup[str(record_id)]} for record_id in candidates_df["id1"].astype(str)]
            right_records = [{"id": str(record_id), **lookup[str(record_id)]} for record_id in candidates_df["id2"].astype(str)]
            scores = model.score_pairs(left_records, right_records)

        results_df = candidates_df.copy()
        results_df["score"] = scores
        results_df["is_duplicate"] = results_df["score"] >= threshold_value
        duplicate_df = results_df.loc[results_df["is_duplicate"]].copy().sort_values("score", ascending=False)
        top_df = duplicate_df.head(top_k)

        duplicate_pairs = [
            {
                "id1": str(row.id1),
                "id2": str(row.id2),
                "score": float(row.score),
                "record1": _record_excerpt({"id": str(row.id1), **lookup[str(row.id1)]}, model.attributes),
                "record2": _record_excerpt({"id": str(row.id2), **lookup[str(row.id2)]}, model.attributes),
            }
            for row in top_df.itertuples(index=False)
        ]

        return {
            "model_name": model_name,
            "threshold": threshold_value,
            "blocking_keys": blocking_keys,
            "blocking_mode": blocking_mode,
            "dataset": runtime_dataset.summary(),
            "candidate_pair_count": int(len(results_df)),
            "predicted_duplicate_pair_count": int(len(duplicate_df)),
            "duplicate_pairs": duplicate_pairs,
            "duplicate_clusters": build_duplicate_clusters(duplicate_df[["id1", "id2"]]) if not duplicate_df.empty else [],
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
