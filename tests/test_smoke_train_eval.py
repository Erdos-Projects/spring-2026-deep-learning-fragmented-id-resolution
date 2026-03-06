import json
import subprocess
import sys

import pandas as pd


def _write_tsv(df, path):
    df.to_csv(path, sep="\t", index=False)


def _build_tiny_dataset(tmp_path):
    prepared = pd.DataFrame(
        [
            {"id": "1", "first_name": "ann", "last_name": "lee", "house_num": "10", "street_name": "oak", "zip_code": "11111", "age": "20", "sex": "f", "race_desc": "w", "ethnic_desc": "n"},
            {"id": "2", "first_name": "anne", "last_name": "lee", "house_num": "10", "street_name": "oak", "zip_code": "11111", "age": "20", "sex": "f", "race_desc": "w", "ethnic_desc": "n"},
            {"id": "3", "first_name": "bob", "last_name": "ray", "house_num": "20", "street_name": "elm", "zip_code": "22222", "age": "40", "sex": "m", "race_desc": "b", "ethnic_desc": "n"},
            {"id": "4", "first_name": "bobby", "last_name": "ray", "house_num": "20", "street_name": "elm", "zip_code": "22222", "age": "40", "sex": "m", "race_desc": "b", "ethnic_desc": "n"},
            {"id": "5", "first_name": "cara", "last_name": "kim", "house_num": "30", "street_name": "maple", "zip_code": "33333", "age": "50", "sex": "f", "race_desc": "a", "ethnic_desc": "n"},
            {"id": "6", "first_name": "cora", "last_name": "king", "house_num": "31", "street_name": "maple", "zip_code": "33333", "age": "50", "sex": "f", "race_desc": "a", "ethnic_desc": "n"},
            {"id": "7", "first_name": "dan", "last_name": "zhu", "house_num": "40", "street_name": "pine", "zip_code": "44444", "age": "60", "sex": "m", "race_desc": "w", "ethnic_desc": "n"},
            {"id": "8", "first_name": "danny", "last_name": "zhu", "house_num": "40", "street_name": "pine", "zip_code": "44444", "age": "60", "sex": "m", "race_desc": "w", "ethnic_desc": "n"},
        ]
    )
    dpl = pd.DataFrame(
        [
            {"id1": "1", "id2": "2"},
            {"id1": "3", "id2": "4"},
            {"id1": "7", "id2": "8"},
            {"id1": "5", "id2": "6"},
        ]
    )
    ndpl = pd.DataFrame(
        [
            {"id1": "1", "id2": "3"},
            {"id1": "1", "id2": "5"},
            {"id1": "1", "id2": "7"},
            {"id1": "2", "id2": "4"},
            {"id1": "2", "id2": "6"},
            {"id1": "2", "id2": "8"},
            {"id1": "3", "id2": "5"},
            {"id1": "3", "id2": "7"},
            {"id1": "4", "id2": "6"},
            {"id1": "4", "id2": "8"},
            {"id1": "5", "id2": "7"},
            {"id1": "6", "id2": "8"},
        ]
    )
    prepared_path = tmp_path / "prepared.tsv"
    dpl_path = tmp_path / "dpl.tsv"
    ndpl_path = tmp_path / "ndpl.tsv"
    _write_tsv(prepared, prepared_path)
    _write_tsv(dpl, dpl_path)
    _write_tsv(ndpl, ndpl_path)
    return prepared_path, dpl_path, ndpl_path


def test_smoke_train_then_evaluate(tmp_path):
    prepared_path, dpl_path, ndpl_path = _build_tiny_dataset(tmp_path)

    run_dir = tmp_path / "run"
    split_path = tmp_path / "split.tsv"
    train_cmd = [
        sys.executable,
        "src/train.py",
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
        "--epochs",
        "1",
        "--batch-size",
        "4",
        "--run-dir",
        str(run_dir),
        "--disable-progress",
        "--seed",
        "7",
    ]
    subprocess.run(train_cmd, check=True)

    checkpoint_path = run_dir / "best_model.pth"
    metrics_json_path = run_dir / "best_metrics.json"
    history_path = run_dir / "metrics_history.tsv"
    assert checkpoint_path.exists()
    assert metrics_json_path.exists()
    assert history_path.exists()
    assert split_path.exists()

    with open(metrics_json_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    assert "test_metrics" in payload

    eval_out_dir = tmp_path / "eval"
    eval_cmd = [
        sys.executable,
        "src/evaluate.py",
        "--model-path",
        str(checkpoint_path),
        "--split-path",
        str(split_path),
        "--split-name",
        "test",
        "--output-dir",
        str(eval_out_dir),
        "--output-prefix",
        "smoke",
        "--disable-progress",
    ]
    subprocess.run(eval_cmd, check=True)

    assert (eval_out_dir / "smoke_test_metrics.json").exists()
    assert (eval_out_dir / "smoke_test_metrics.tsv").exists()
    assert (eval_out_dir / "smoke_test_confusion_matrix.tsv").exists()


def test_smoke_train_then_evaluate_with_shared_blocking(tmp_path):
    prepared_path, dpl_path, ndpl_path = _build_tiny_dataset(tmp_path)

    run_dir = tmp_path / "run_blocked"
    split_path = tmp_path / "split_blocked.tsv"
    train_cmd = [
        sys.executable,
        "src/train.py",
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
        "--epochs",
        "1",
        "--batch-size",
        "4",
        "--run-dir",
        str(run_dir),
        "--blocking-keys",
        "first_name,age",
        "--blocking-mode",
        "any",
        "--disable-progress",
        "--seed",
        "7",
    ]
    subprocess.run(train_cmd, check=True)

    checkpoint_path = run_dir / "best_model.pth"
    assert checkpoint_path.exists()

    with open(run_dir / "best_metrics.json", "r", encoding="utf-8") as f:
        payload = json.load(f)
    assert payload["blocking_keys"] == ["first_name", "age"]
    assert "blocking_summary" in payload

    eval_out_dir = tmp_path / "eval_blocked"
    eval_cmd = [
        sys.executable,
        "src/evaluate.py",
        "--model-path",
        str(checkpoint_path),
        "--split-path",
        str(split_path),
        "--split-name",
        "test",
        "--output-dir",
        str(eval_out_dir),
        "--output-prefix",
        "smoke_blocked",
        "--disable-progress",
    ]
    subprocess.run(eval_cmd, check=True)

    with open(eval_out_dir / "smoke_blocked_test_metrics.json", "r", encoding="utf-8") as f:
        eval_payload = json.load(f)
    assert eval_payload["blocking_keys"] == ["first_name", "age"]
    assert "blocking" in eval_payload["metrics"]


def test_smoke_train_then_evaluate_charcnn(tmp_path):
    prepared_path, dpl_path, ndpl_path = _build_tiny_dataset(tmp_path)

    run_dir = tmp_path / "run_charcnn"
    split_path = tmp_path / "split_charcnn.tsv"
    train_cmd = [
        sys.executable,
        "src/train.py",
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
        "--epochs",
        "1",
        "--batch-size",
        "4",
        "--encoder",
        "charcnn",
        "--cnn-channels",
        "16",
        "--cnn-kernel-sizes",
        "3,4,5",
        "--classifier-hidden-dim",
        "16",
        "--run-dir",
        str(run_dir),
        "--disable-progress",
        "--seed",
        "7",
    ]
    subprocess.run(train_cmd, check=True)

    checkpoint_path = run_dir / "best_model.pth"
    assert checkpoint_path.exists()

    with open(run_dir / "best_metrics.json", "r", encoding="utf-8") as f:
        payload = json.load(f)
    assert payload["best_epoch"] >= 1

    eval_out_dir = tmp_path / "eval_charcnn"
    eval_cmd = [
        sys.executable,
        "src/evaluate.py",
        "--model-path",
        str(checkpoint_path),
        "--split-path",
        str(split_path),
        "--split-name",
        "test",
        "--output-dir",
        str(eval_out_dir),
        "--output-prefix",
        "smoke_charcnn",
        "--disable-progress",
    ]
    subprocess.run(eval_cmd, check=True)

    assert (eval_out_dir / "smoke_charcnn_test_metrics.json").exists()
