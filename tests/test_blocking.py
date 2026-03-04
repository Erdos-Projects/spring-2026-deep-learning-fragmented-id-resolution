import numpy as np
import pandas as pd

from src.blocking import compute_block_mask, parse_blocking_keys, summarize_blocking


def test_parse_blocking_keys_supports_none_and_csv():
    assert parse_blocking_keys("none", default_keys=[]) == []
    assert parse_blocking_keys("first_name, age", default_keys=[]) == ["first_name", "age"]


def test_compute_block_mask_any_and_all():
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


def test_summarize_blocking_reports_positive_and_negative_pass_rates():
    labels = np.array([1, 1, 0, 0, 0])
    mask = np.array([True, False, True, False, False])
    summary = summarize_blocking(labels, mask)

    assert summary["candidate_pair_count"] == 2
    assert summary["blocked_pair_count"] == 3
    assert summary["positive_pair_count"] == 2
    assert summary["negative_pair_count"] == 3
    assert summary["positive_candidate_count"] == 1
    assert summary["negative_candidate_count"] == 1
    assert summary["positive_pass_rate"] == 0.5
    assert summary["negative_pass_rate"] == (1 / 3)
