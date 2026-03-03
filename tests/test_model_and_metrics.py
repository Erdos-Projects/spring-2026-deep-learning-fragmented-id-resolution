import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, precision_score, recall_score

from src.dataset import NCVotersDataset
from src.metrics import compute_binary_metrics, select_threshold
from src.model import SiameseNetwork


def _write_tsv(df, path):
    df.to_csv(path, sep="\t", index=False)


def test_dataset_getitem_and_model_forward(tmp_path):
    prepared = pd.DataFrame(
        [
            {"id": "1", "first_name": "ann", "last_name": "lee", "house_num": "10", "street_name": "oak", "zip_code": "11111"},
            {"id": "2", "first_name": "anne", "last_name": "lee", "house_num": "10", "street_name": "oak", "zip_code": "11111"},
            {"id": "3", "first_name": "bob", "last_name": "ray", "house_num": "20", "street_name": "elm", "zip_code": "22222"},
        ]
    )
    pairs = pd.DataFrame(
        [
            {"pair_id": 0, "id1": "1", "id2": "2", "label": 1.0},
            {"pair_id": 1, "id1": "1", "id2": "3", "label": 0.0},
        ]
    )
    prepared_path = tmp_path / "prepared.tsv"
    _write_tsv(prepared, prepared_path)

    ds = NCVotersDataset(
        data_path=prepared_path,
        pairs_df=pairs,
        max_len=24,
        attributes=["first_name", "last_name", "house_num", "street_name", "zip_code"],
    )
    x1, x2, y = ds[0]
    assert x1.shape == (24,)
    assert x2.shape == (24,)
    assert x1.dtype == torch.long
    assert y.dtype == torch.float32

    model = SiameseNetwork(vocab_size=ds.vocab.vocab_size, embedding_dim=16, hidden_dim=8)
    logits = model(x1.unsqueeze(0), x2.unsqueeze(0))
    assert logits.shape == (1, 1)


def test_metrics_match_sklearn_and_threshold_is_deterministic():
    y_true = np.array([0, 0, 1, 1, 1, 0, 1, 0])
    y_prob = np.array([0.10, 0.20, 0.80, 0.70, 0.55, 0.40, 0.90, 0.05])
    threshold = 0.5
    y_pred = (y_prob >= threshold).astype(int)

    m = compute_binary_metrics(y_true, y_prob, threshold=threshold)
    assert abs(m["precision"] - precision_score(y_true, y_pred, zero_division=0)) < 1e-9
    assert abs(m["recall"] - recall_score(y_true, y_pred, zero_division=0)) < 1e-9
    assert abs(m["f1"] - f1_score(y_true, y_pred, zero_division=0)) < 1e-9

    t1, _, _ = select_threshold(y_true, y_prob, mode="f1")
    t2, _, _ = select_threshold(y_true, y_prob, mode="f1")
    assert t1 == t2
