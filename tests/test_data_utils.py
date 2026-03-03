import pandas as pd
import pytest

from src.data_utils import count_duplicate_canonical_pairs, count_missing_pair_ids, load_labeled_pairs
from src.dataset import NCVotersDataset


def _write_tsv(df, path):
    df.to_csv(path, sep="\t", index=False)


def test_load_labeled_pairs_raises_on_duplicate_canonical_pairs(tmp_path):
    dpl = pd.DataFrame({"id1": ["1"], "id2": ["2"]})
    ndpl = pd.DataFrame({"id1": ["2"], "id2": ["1"]})

    dpl_path = tmp_path / "dpl.tsv"
    ndpl_path = tmp_path / "ndpl.tsv"
    _write_tsv(dpl, dpl_path)
    _write_tsv(ndpl, ndpl_path)

    with pytest.raises(ValueError):
        load_labeled_pairs(dpl_path, ndpl_path, fail_on_duplicate_pairs=True)


def test_missing_ids_and_duplicate_count_helpers(tmp_path):
    dpl = pd.DataFrame({"id1": ["1", "3"], "id2": ["2", "9"]})
    ndpl = pd.DataFrame({"id1": ["4"], "id2": ["5"]})
    dpl_path = tmp_path / "dpl.tsv"
    ndpl_path = tmp_path / "ndpl.tsv"
    _write_tsv(dpl, dpl_path)
    _write_tsv(ndpl, ndpl_path)

    pairs = load_labeled_pairs(dpl_path, ndpl_path, fail_on_duplicate_pairs=True)
    assert count_duplicate_canonical_pairs(pairs) == 0

    missing_rows, missing_ids = count_missing_pair_ids(pairs, valid_ids={"1", "2", "3", "4", "5"})
    assert missing_rows == 1
    assert missing_ids == {"9"}


def test_dataset_logs_and_counts_missing_ids(tmp_path):
    prepared = pd.DataFrame(
        [
            {"id": "1", "first_name": "ann", "last_name": "lee", "house_num": "10", "street_name": "oak", "zip_code": "11111"},
            {"id": "2", "first_name": "bob", "last_name": "ray", "house_num": "20", "street_name": "elm", "zip_code": "22222"},
        ]
    )
    pairs = pd.DataFrame(
        [
            {"pair_id": 0, "id1": "1", "id2": "2", "label": 1.0},
            {"pair_id": 1, "id1": "1", "id2": "9", "label": 0.0},
        ]
    )
    prepared_path = tmp_path / "prepared.tsv"
    prepared.to_csv(prepared_path, sep="\t", index=False)

    ds = NCVotersDataset(
        data_path=prepared_path,
        pairs_df=pairs,
        max_len=20,
        attributes=["first_name", "last_name", "house_num", "street_name", "zip_code"],
        strict_missing_ids=False,
        drop_missing_ids=True,
    )
    assert ds.missing_pair_rows == 1
    assert ds.missing_pair_ids == {"9"}
    assert len(ds) == 1
