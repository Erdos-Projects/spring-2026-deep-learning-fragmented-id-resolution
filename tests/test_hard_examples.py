import pandas as pd

from src.hard_examples import attach_hard_example_metadata, mine_hard_examples, summarize_hard_example_metrics


def _write_tsv(df, path):
    df.to_csv(path, sep="\t", index=False)


def test_mine_hard_examples_flags_realistic_hard_positive_and_negative(tmp_path):
    prepared = pd.DataFrame(
        [
            {
                "id": "1",
                "first_name": "helen",
                "midl_name": "delores",
                "last_name": "bullock",
                "house_num": "700",
                "street_name": "bloodworth",
                "zip_code": "27601",
                "age": "65",
                "sex": "female",
                "race_desc": "black",
                "ethnic_desc": "n",
            },
            {
                "id": "2",
                "first_name": "helen",
                "midl_name": "delores",
                "last_name": "brown bullock",
                "house_num": "700",
                "street_name": "bloodworth",
                "zip_code": "27601",
                "age": "65",
                "sex": "female",
                "race_desc": "asian",
                "ethnic_desc": "n",
            },
            {
                "id": "3",
                "first_name": "john",
                "midl_name": "",
                "last_name": "smith",
                "house_num": "12",
                "street_name": "oak",
                "zip_code": "11111",
                "age": "44",
                "sex": "m",
                "race_desc": "white",
                "ethnic_desc": "n",
            },
            {
                "id": "4",
                "first_name": "john",
                "midl_name": "",
                "last_name": "smyth",
                "house_num": "12",
                "street_name": "oak",
                "zip_code": "11111",
                "age": "44",
                "sex": "m",
                "race_desc": "white",
                "ethnic_desc": "n",
            },
            {
                "id": "5",
                "first_name": "amy",
                "midl_name": "",
                "last_name": "lee",
                "house_num": "30",
                "street_name": "pine",
                "zip_code": "22222",
                "age": "50",
                "sex": "f",
                "race_desc": "white",
                "ethnic_desc": "n",
            },
            {
                "id": "6",
                "first_name": "bob",
                "midl_name": "",
                "last_name": "ray",
                "house_num": "40",
                "street_name": "elm",
                "zip_code": "33333",
                "age": "30",
                "sex": "m",
                "race_desc": "black",
                "ethnic_desc": "n",
            },
        ]
    )
    dpl = pd.DataFrame([{"id1": "1", "id2": "2"}])
    ndpl = pd.DataFrame([{"id1": "3", "id2": "4"}, {"id1": "5", "id2": "6"}])

    prepared_path = tmp_path / "prepared.tsv"
    dpl_path = tmp_path / "dpl.tsv"
    ndpl_path = tmp_path / "ndpl.tsv"
    _write_tsv(prepared, prepared_path)
    _write_tsv(dpl, dpl_path)
    _write_tsv(ndpl, ndpl_path)

    hard_df = mine_hard_examples(prepared_path, dpl_path, ndpl_path)
    hard_types = {
        (row.id1, row.id2): row.hard_type
        for row in hard_df.itertuples(index=False)
    }
    assert hard_types[("1", "2")] == "hard_positive"
    assert hard_types[("3", "4")] == "hard_negative"

    attached = attach_hard_example_metadata(
        hard_df[["pair_id", "id1", "id2", "label"]],
        hard_df[["pair_id", "hard_example", "hard_type", "difficulty_bucket", "difficulty_reasons", "difficulty_score", "suggested_weight"]],
        weight_scale=1.5,
    )
    assert attached.loc[attached["id1"] == "1", "sample_weight"].iloc[0] > 1.0
    assert attached.loc[attached["id1"] == "3", "sample_weight"].iloc[0] > 1.0

    y_true = attached["label"].astype(int).to_numpy()
    y_prob = [0.95, 0.85, 0.05]
    metrics = summarize_hard_example_metrics(attached, y_true, y_prob, threshold=0.5)
    assert metrics["hard_positive_count"] == 1
    assert metrics["hard_negative_count"] == 1
    assert metrics["hard_positive_recall"] == 1.0
    assert metrics["hard_negative_rejection_rate"] == 0.0
