import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


VALID_REVIEW_DECISIONS = {
    "accept_duplicate": "Accepted duplicate",
    "reject_duplicate": "Rejected duplicate",
    "uncertain": "Marked uncertain",
}


def canonical_pair(id1: str, id2: str) -> Tuple[str, str, str]:
    left, right = sorted((str(id1), str(id2)))
    return left, right, f"{left}::{right}"


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


@dataclass
class ReviewDecisionRecord:
    pair_key: str
    id1: str
    id2: str
    dataset_source: str
    model_name: str
    decision: str
    decision_label: str
    notes: str
    score: Optional[float]
    comparison_model_name: Optional[str]
    comparison_score: Optional[float]
    source_section: Optional[str]
    cluster_id: Optional[str]
    record1: Dict[str, Any]
    record2: Dict[str, Any]
    created_at: str
    updated_at: str

    def to_payload(self) -> Dict[str, Any]:
        return {
            "pair_key": self.pair_key,
            "id1": self.id1,
            "id2": self.id2,
            "dataset_source": self.dataset_source,
            "model_name": self.model_name,
            "decision": self.decision,
            "decision_label": self.decision_label,
            "notes": self.notes,
            "score": self.score,
            "comparison_model_name": self.comparison_model_name,
            "comparison_score": self.comparison_score,
            "source_section": self.source_section,
            "cluster_id": self.cluster_id,
            "record1": self.record1,
            "record2": self.record2,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


class ReviewStore:
    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS review_decisions (
                    pair_key TEXT PRIMARY KEY,
                    id1 TEXT NOT NULL,
                    id2 TEXT NOT NULL,
                    dataset_source TEXT NOT NULL,
                    model_name TEXT NOT NULL,
                    decision TEXT NOT NULL,
                    notes TEXT NOT NULL DEFAULT '',
                    score REAL,
                    comparison_model_name TEXT,
                    comparison_score REAL,
                    source_section TEXT,
                    cluster_id TEXT,
                    record1_json TEXT NOT NULL,
                    record2_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS review_events (
                    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    pair_key TEXT NOT NULL,
                    id1 TEXT NOT NULL,
                    id2 TEXT NOT NULL,
                    dataset_source TEXT NOT NULL,
                    model_name TEXT NOT NULL,
                    decision TEXT NOT NULL,
                    notes TEXT NOT NULL DEFAULT '',
                    score REAL,
                    comparison_model_name TEXT,
                    comparison_score REAL,
                    source_section TEXT,
                    cluster_id TEXT,
                    record1_json TEXT NOT NULL,
                    record2_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                """
            )

    def save_decision(
        self,
        *,
        id1: str,
        id2: str,
        dataset_source: str,
        model_name: str,
        decision: str,
        record1: Dict[str, Any],
        record2: Dict[str, Any],
        notes: str = "",
        score: Optional[float] = None,
        comparison_model_name: Optional[str] = None,
        comparison_score: Optional[float] = None,
        source_section: Optional[str] = None,
        cluster_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        if decision not in VALID_REVIEW_DECISIONS:
            raise ValueError(f"Unsupported review decision '{decision}'.")

        left_id, right_id, pair_key = canonical_pair(id1, id2)
        timestamp = _utcnow_iso()
        record1_json = json.dumps(record1, sort_keys=True)
        record2_json = json.dumps(record2, sort_keys=True)

        with self._connect() as connection:
            existing = connection.execute(
                "SELECT created_at FROM review_decisions WHERE pair_key = ?",
                (pair_key,),
            ).fetchone()
            created_at = existing["created_at"] if existing else timestamp
            connection.execute(
                """
                INSERT INTO review_decisions (
                    pair_key, id1, id2, dataset_source, model_name, decision, notes,
                    score, comparison_model_name, comparison_score, source_section, cluster_id,
                    record1_json, record2_json, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(pair_key) DO UPDATE SET
                    dataset_source = excluded.dataset_source,
                    model_name = excluded.model_name,
                    decision = excluded.decision,
                    notes = excluded.notes,
                    score = excluded.score,
                    comparison_model_name = excluded.comparison_model_name,
                    comparison_score = excluded.comparison_score,
                    source_section = excluded.source_section,
                    cluster_id = excluded.cluster_id,
                    record1_json = excluded.record1_json,
                    record2_json = excluded.record2_json,
                    updated_at = excluded.updated_at
                """,
                (
                    pair_key,
                    left_id,
                    right_id,
                    dataset_source,
                    model_name,
                    decision,
                    notes,
                    score,
                    comparison_model_name,
                    comparison_score,
                    source_section,
                    cluster_id,
                    record1_json,
                    record2_json,
                    created_at,
                    timestamp,
                ),
            )
            connection.execute(
                """
                INSERT INTO review_events (
                    pair_key, id1, id2, dataset_source, model_name, decision, notes,
                    score, comparison_model_name, comparison_score, source_section, cluster_id,
                    record1_json, record2_json, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    pair_key,
                    left_id,
                    right_id,
                    dataset_source,
                    model_name,
                    decision,
                    notes,
                    score,
                    comparison_model_name,
                    comparison_score,
                    source_section,
                    cluster_id,
                    record1_json,
                    record2_json,
                    timestamp,
                ),
            )
            connection.commit()

        return self.get_decision(pair_key=pair_key) or {}

    def get_decision(self, *, pair_key: Optional[str] = None, id1: Optional[str] = None, id2: Optional[str] = None) -> Optional[Dict[str, Any]]:
        resolved_pair_key = pair_key
        if resolved_pair_key is None:
            if id1 is None or id2 is None:
                raise ValueError("Provide either pair_key or both id1 and id2.")
            _, _, resolved_pair_key = canonical_pair(id1, id2)

        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM review_decisions WHERE pair_key = ?",
                (resolved_pair_key,),
            ).fetchone()
        return self._row_to_payload(row) if row else None

    def get_decisions_for_pairs(self, pair_ids: Sequence[Tuple[str, str]]) -> Dict[str, Dict[str, Any]]:
        pair_keys = []
        for id1, id2 in pair_ids:
            _, _, pair_key = canonical_pair(id1, id2)
            pair_keys.append(pair_key)
        if not pair_keys:
            return {}

        placeholders = ",".join("?" for _ in pair_keys)
        with self._connect() as connection:
            rows = connection.execute(
                f"SELECT * FROM review_decisions WHERE pair_key IN ({placeholders})",
                tuple(pair_keys),
            ).fetchall()
        return {row["pair_key"]: self._row_to_payload(row) for row in rows}

    def list_decisions(self, *, dataset_source: Optional[str] = None, limit: int = 25) -> List[Dict[str, Any]]:
        query = "SELECT * FROM review_decisions"
        params: List[Any] = []
        if dataset_source:
            query += " WHERE dataset_source = ?"
            params.append(dataset_source)
        query += " ORDER BY updated_at DESC LIMIT ?"
        params.append(int(limit))
        with self._connect() as connection:
            rows = connection.execute(query, tuple(params)).fetchall()
        return [self._row_to_payload(row) for row in rows]

    def summary(self, *, dataset_source: Optional[str] = None) -> Dict[str, Any]:
        filter_clause = ""
        params: List[Any] = []
        if dataset_source:
            filter_clause = " WHERE dataset_source = ?"
            params.append(dataset_source)

        with self._connect() as connection:
            total_row = connection.execute(
                f"SELECT COUNT(*) AS total FROM review_decisions{filter_clause}",
                tuple(params),
            ).fetchone()
            breakdown_rows = connection.execute(
                f"""
                SELECT decision, COUNT(*) AS count
                FROM review_decisions
                {filter_clause}
                GROUP BY decision
                """,
                tuple(params),
            ).fetchall()

        breakdown = {decision: 0 for decision in VALID_REVIEW_DECISIONS}
        for row in breakdown_rows:
            breakdown[str(row["decision"])] = int(row["count"])

        return {
            "total": int(total_row["total"]) if total_row else 0,
            "accepted_duplicate_count": breakdown["accept_duplicate"],
            "rejected_duplicate_count": breakdown["reject_duplicate"],
            "uncertain_count": breakdown["uncertain"],
            "counts_by_decision": breakdown,
        }

    def clear_decisions(self, *, dataset_source: Optional[str] = None) -> Dict[str, Any]:
        with self._connect() as connection:
            if dataset_source:
                decision_total_row = connection.execute(
                    "SELECT COUNT(*) AS total FROM review_decisions WHERE dataset_source = ?",
                    (dataset_source,),
                ).fetchone()
                event_total_row = connection.execute(
                    "SELECT COUNT(*) AS total FROM review_events WHERE dataset_source = ?",
                    (dataset_source,),
                ).fetchone()
                connection.execute(
                    "DELETE FROM review_decisions WHERE dataset_source = ?",
                    (dataset_source,),
                )
                connection.execute(
                    "DELETE FROM review_events WHERE dataset_source = ?",
                    (dataset_source,),
                )
            else:
                decision_total_row = connection.execute(
                    "SELECT COUNT(*) AS total FROM review_decisions"
                ).fetchone()
                event_total_row = connection.execute(
                    "SELECT COUNT(*) AS total FROM review_events"
                ).fetchone()
                connection.execute("DELETE FROM review_decisions")
                connection.execute("DELETE FROM review_events")
            connection.commit()

        return {
            "cleared_decisions": int(decision_total_row["total"]) if decision_total_row else 0,
            "cleared_events": int(event_total_row["total"]) if event_total_row else 0,
            "dataset_source": dataset_source,
        }

    def _row_to_payload(self, row: sqlite3.Row) -> Dict[str, Any]:
        if row is None:
            return {}
        decision = str(row["decision"])
        return ReviewDecisionRecord(
            pair_key=str(row["pair_key"]),
            id1=str(row["id1"]),
            id2=str(row["id2"]),
            dataset_source=str(row["dataset_source"]),
            model_name=str(row["model_name"]),
            decision=decision,
            decision_label=VALID_REVIEW_DECISIONS[decision],
            notes=str(row["notes"] or ""),
            score=None if row["score"] is None else float(row["score"]),
            comparison_model_name=None if row["comparison_model_name"] is None else str(row["comparison_model_name"]),
            comparison_score=None if row["comparison_score"] is None else float(row["comparison_score"]),
            source_section=None if row["source_section"] is None else str(row["source_section"]),
            cluster_id=None if row["cluster_id"] is None else str(row["cluster_id"]),
            record1=json.loads(row["record1_json"]),
            record2=json.loads(row["record2_json"]),
            created_at=str(row["created_at"]),
            updated_at=str(row["updated_at"]),
        ).to_payload()
