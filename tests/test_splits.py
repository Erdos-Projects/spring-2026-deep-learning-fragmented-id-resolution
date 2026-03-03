import numpy as np
import pandas as pd

from src.splits import build_splits, check_split_integrity


def _make_pairs(n_pairs=600, n_ids=120, seed=13):
    rng = np.random.RandomState(seed)
    ids = [str(i) for i in range(n_ids)]
    rows = []
    for pair_id in range(n_pairs):
        id1 = ids[rng.randint(0, n_ids)]
        id2 = ids[rng.randint(0, n_ids)]
        while id2 == id1:
            id2 = ids[rng.randint(0, n_ids)]
        label = float(rng.rand() < 0.2)
        rows.append({"pair_id": pair_id, "id1": id1, "id2": id2, "label": label})
    return pd.DataFrame(rows)


def test_id_disjoint_split_has_zero_id_overlap():
    pairs = _make_pairs()
    split_df = build_splits(pairs, strategy="id_disjoint", val_frac=0.15, test_frac=0.15, seed=42)
    summary = check_split_integrity(split_df)

    assert summary["train_pairs"] > 0
    assert summary["val_pairs"] > 0
    assert summary["test_pairs"] > 0
    assert summary["train_val_id_overlap"] == 0
    assert summary["train_test_id_overlap"] == 0
    assert summary["val_test_id_overlap"] == 0


def test_pair_random_split_preserves_label_ratio():
    pairs = _make_pairs()
    split_df = build_splits(pairs, strategy="pair_random", val_frac=0.15, test_frac=0.15, seed=42)

    overall_ratio = float((pairs["label"] == 1).mean())
    for split_name in ["train", "val", "test"]:
        section = split_df[split_df["split"] == split_name]
        split_ratio = float((section["label"] == 1).mean())
        assert abs(split_ratio - overall_ratio) < 0.05
