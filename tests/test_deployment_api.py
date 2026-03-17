from pathlib import Path
import shutil
import tempfile

import pandas as pd
from fastapi.testclient import TestClient

from src.api import create_app
from src.deployment import DuplicateDetectionService


REPO_ROOT = Path(__file__).resolve().parents[1]
TFIDF_CONFIG_PATH = REPO_ROOT / "models/comparisons/apples_to_apples_with_charcnn/tfidf/blocked_pipeline/extended/baseline_metrics.json"
SIAMESE_MODEL_PATH = REPO_ROOT / "models/comparisons/apples_to_apples_with_charcnn/siamese_bilstm/blocked_pipeline/extended/best_model.pth"


def _sample_runtime_frame():
    return pd.DataFrame(
        [
            {
                "id": "r1",
                "first_name": "John",
                "last_name": "Smith",
                "house_num": "12",
                "street_name": "Oak Street",
                "zip_code": "27514",
                "age": "44",
                "sex": "m",
                "race_desc": "white",
                "ethnic_desc": "not hispanic",
            },
            {
                "id": "r2",
                "first_name": "John",
                "last_name": "Smith",
                "house_num": "12",
                "street_name": "Oak St",
                "zip_code": "27514",
                "age": "44",
                "sex": "m",
                "race_desc": "white",
                "ethnic_desc": "not hispanic",
            },
            {
                "id": "r3",
                "first_name": "John",
                "last_name": "Smith",
                "house_num": "12",
                "street_name": "Oak Street",
                "zip_code": "27514",
                "age": "19",
                "sex": "m",
                "race_desc": "white",
                "ethnic_desc": "not hispanic",
            },
            {
                "id": "r4",
                "first_name": "Maria",
                "last_name": "Stone",
                "house_num": "88",
                "street_name": "Pine Avenue",
                "zip_code": "27516",
                "age": "31",
                "sex": "f",
                "race_desc": "white",
                "ethnic_desc": "not hispanic",
            },
        ]
    )


def _make_runtime_dir():
    root = REPO_ROOT / "data" / "runtime_test_artifacts"
    root.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix="deploy_test_", dir=root))


def test_api_upload_find_and_check_entry_with_tfidf():
    runtime_dir = _make_runtime_dir()
    service = DuplicateDetectionService(
        model_specs={"tfidf": {"kind": "tfidf", "path": str(TFIDF_CONFIG_PATH)}},
        runtime_dir=runtime_dir,
    )
    try:
        client = TestClient(create_app(service=service))

        root_response = client.get("/")
        assert root_response.status_code == 200
        assert "Fragmented-ID-Resolution for Voter Registration Records" in root_response.text

        state_response = client.get("/app/state")
        assert state_response.status_code == 200
        state_payload = state_response.json()
        assert state_payload["models"][0]["name"] == "tfidf"
        assert state_payload["defaults"]["duplicate_search"]["blocking_mode"] == "all"
        assert state_payload["defaults"]["duplicate_search"]["blocking_keys"] == [
            "first_name",
            "last_name",
            "zip_code",
        ]

        upload_frame = _sample_runtime_frame().rename(
            columns={
                "first_name": "FirstName",
                "last_name": "LastName",
                "house_num": "House Number",
                "street_name": "Street",
                "zip_code": "ZipCode",
                "race_desc": "Race",
                "ethnic_desc": "Ethnicity",
            }
        )
        upload_bytes = upload_frame.to_csv(sep="\t", index=False).encode("utf-8")

        upload_response = client.post(
            "/dataset/upload",
            files={"file": ("sample.tsv", upload_bytes, "text/tab-separated-values")},
        )
        assert upload_response.status_code == 200
        assert upload_response.json()["dataset"]["record_count"] == 4

        find_response = client.post(
            "/duplicates/find",
            json={
                "model_name": "tfidf",
                "blocking_keys": ["first_name", "age"],
                "blocking_mode": "any",
                "top_k": 10,
            },
        )
        assert find_response.status_code == 200
        payload = find_response.json()
        assert payload["candidate_pair_count"] >= 1
        assert payload["predicted_duplicate_pair_count"] >= 1
        assert any(
            {pair["id1"], pair["id2"]} == {"r1", "r2"} for pair in payload["duplicate_pairs"]
        )

        entry_response = client.post(
            "/duplicates/check-entry",
            json={
                "model_name": "tfidf",
                "entry": {
                    "first_name": "John",
                    "last_name": "Smith",
                    "house_num": "12",
                    "street_name": "Oak Street",
                    "zip_code": "27514",
                    "age": "44",
                },
                "blocking_keys": ["first_name", "age"],
                "blocking_mode": "any",
                "top_k": 5,
            },
        )
        assert entry_response.status_code == 200
        entry_payload = entry_response.json()
        assert entry_payload["duplicate_exists"] is True
        assert entry_payload["matches"][0]["existing_id"] in {"r1", "r2"}
    finally:
        shutil.rmtree(runtime_dir, ignore_errors=True)


def test_siamese_service_check_entry_smoke():
    runtime_dir = _make_runtime_dir()
    service = DuplicateDetectionService(
        model_specs={"siamese_bilstm": {"kind": "siamese", "path": str(SIAMESE_MODEL_PATH)}},
        runtime_dir=runtime_dir,
        device="cpu",
    )
    try:
        service.load_dataset_from_dataframe(_sample_runtime_frame(), source_name="in_memory")

        payload = service.check_entry(
            entry={
                "first_name": "John",
                "last_name": "Smith",
                "house_num": "12",
                "street_name": "Oak Street",
                "zip_code": "27514",
                "age": "44",
                "sex": "m",
                "race_desc": "white",
                "ethnic_desc": "not hispanic",
            },
            model_name="siamese_bilstm",
            blocking_keys=["first_name", "age"],
            blocking_mode="any",
            top_k=5,
        )

        assert payload["candidate_count"] >= 1
        assert payload["matches"]
        assert payload["matches"][0]["score"] >= 0.0
    finally:
        shutil.rmtree(runtime_dir, ignore_errors=True)
