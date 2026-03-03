import json
import subprocess
import sys

import numpy as np
import pandas as pd

from src.baseline_tfidf import compute_block_mask, parse_blocking_keys, select_distance_threshold


def _write_tsv(df, path):
    df.to_csv(path, sep="\t", index=False)


def test_parse_blocking_keys_and_block_mask():
    assert parse_blocking_keys("none") == []
    assert parse_blocking_keys("first_name, age") == ["first_name", "age"]

    data_df = pd.DataFrame(
        [
            {"id": "1", "first_name": "ann", "age": "20"},
            {"id": "2", "first_name": "ann", "age": "21"},
            {"id": "3", "first_name": "bob", "age": "20"},
        ]
    )
    pairs_df = pd.DataFrame(
        [
            {"pair_id": 0, "id1": "1", "id2": "2", "label": 1.0},
            {"pair_id": 1, "id1": "1", "id2": "3", "label": 0.0},
        ]
    )

    mask_any = compute_block_mask(pairs_df, data_df, ["first_name", "age"], "any")
    mask_all = compute_block_mask(pairs_df, data_df, ["first_name", "age"], "all")
    assert mask_any.tolist() == [True, True]
    assert mask_all.tolist() == [False, False]


def test_select_distance_threshold_prefers_best_f1():
    distances = np.array([0.05, 0.10, 0.20, 0.70, 0.80, 0.90])
    labels = np.array([1, 1, 1, 0, 0, 0])
    threshold, metrics, sweep_df = select_distance_threshold(distances, labels, selection_mode="f1")

    assert 0.20 <= threshold <= 0.70
    assert metrics["f1"] == 1.0
    assert not sweep_df.empty


def test_baseline_tfidf_smoke_run(tmp_path):
    prepared = pd.DataFrame(
        [
            {"id": "1", "first_name": "ann", "last_name": "lee", "house_num": "10", "street_name": "oak", "zip_code": "11111", "age": "20"},
            {"id": "2", "first_name": "anne", "last_name": "lee", "house_num": "10", "street_name": "oak", "zip_code": "11111", "age": "20"},
            {"id": "3", "first_name": "bob", "last_name": "ray", "house_num": "20", "street_name": "elm", "zip_code": "22222", "age": "40"},
            {"id": "4", "first_name": "bobby", "last_name": "ray", "house_num": "20", "street_name": "elm", "zip_code": "22222", "age": "40"},
            {"id": "5", "first_name": "cara", "last_name": "kim", "house_num": "30", "street_name": "maple", "zip_code": "33333", "age": "50"},
            {"id": "6", "first_name": "dora", "last_name": "king", "house_num": "31", "street_name": "cedar", "zip_code": "44444", "age": "51"},
        ]
    )
    dpl = pd.DataFrame([{"id1": "1", "id2": "2"}, {"id1": "3", "id2": "4"}])
    ndpl = pd.DataFrame(
        [
            {"id1": "1", "id2": "3"},
            {"id1": "1", "id2": "5"},
            {"id1": "2", "id2": "6"},
            {"id1": "3", "id2": "5"},
            {"id1": "4", "id2": "6"},
            {"id1": "5", "id2": "6"},
        ]
    )

    prepared_path = tmp_path / "prepared.tsv"
    dpl_path = tmp_path / "dpl.tsv"
    ndpl_path = tmp_path / "ndpl.tsv"
    run_dir = tmp_path / "baseline_run"
    split_path = tmp_path / "split.tsv"
    _write_tsv(prepared, prepared_path)
    _write_tsv(dpl, dpl_path)
    _write_tsv(ndpl, ndpl_path)

    cmd = [
        sys.executable,
        "src/baseline_tfidf.py",
        "--data-path",
        str(prepared_path),
        "--dpl-path",
        str(dpl_path),
        "--ndpl-path",
        str(ndpl_path),
        "--split-strategy",
        "pair_random",
        "--split-output",
        str(split_path),
        "--run-dir",
        str(run_dir),
        "--blocking-keys",
        "none",
        "--distance-grid-size",
        "31",
    ]
    subprocess.run(cmd, check=True)

    metrics_path = run_dir / "baseline_metrics.json"
    assert metrics_path.exists()

    with open(metrics_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    assert payload["model_type"] == "tfidf_cosine_distance_baseline"
    assert "test" in payload["metrics"]
    assert (run_dir / "baseline_metrics.tsv").exists()
    assert (run_dir / "test_confusion_matrix.tsv").exists()
