from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def build_pair_random_split(pairs_df, val_frac=0.15, test_frac=0.15, seed=42):
    if val_frac + test_frac >= 1.0:
        raise ValueError("val_frac + test_frac must be < 1.0")

    labels = pairs_df["label"].astype(float)
    label_counts = labels.value_counts()
    stratify_labels = labels if len(label_counts) > 1 and int(label_counts.min()) >= 2 else None
    train_df, temp_df = train_test_split(
        pairs_df,
        test_size=(val_frac + test_frac),
        random_state=seed,
        shuffle=True,
        stratify=stratify_labels,
    )

    if len(temp_df) == 0:
        raise ValueError("Validation/test slice is empty. Adjust split fractions.")

    rel_test = test_frac / (val_frac + test_frac)
    temp_labels = temp_df["label"].astype(float)
    temp_label_counts = temp_labels.value_counts()
    temp_stratify = (
        temp_labels if len(temp_label_counts) > 1 and int(temp_label_counts.min()) >= 2 else None
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=rel_test,
        random_state=seed,
        shuffle=True,
        stratify=temp_stratify,
    )

    return _concat_with_split(train_df, val_df, test_df)


def build_id_disjoint_split(pairs_df, val_frac=0.15, test_frac=0.15, seed=42, max_attempts=50):
    if val_frac + test_frac >= 1.0:
        raise ValueError("val_frac + test_frac must be < 1.0")

    ids = sorted(set(pairs_df["id1"].astype(str)).union(set(pairs_df["id2"].astype(str))))
    if len(ids) < 3:
        raise ValueError("Need at least 3 unique IDs for an id-disjoint train/val/test split.")

    best_result = None
    best_retained = -1

    for attempt in range(max_attempts):
        rng = np.random.RandomState(seed + attempt)
        shuffled_ids = np.array(ids)
        rng.shuffle(shuffled_ids)

        n_ids = len(shuffled_ids)
        n_test = int(round(n_ids * test_frac))
        n_val = int(round(n_ids * val_frac))
        n_train = n_ids - n_val - n_test

        if n_train <= 0 or n_val <= 0 or n_test <= 0:
            continue

        train_ids = set(shuffled_ids[:n_train])
        val_ids = set(shuffled_ids[n_train : n_train + n_val])
        test_ids = set(shuffled_ids[n_train + n_val :])

        split_map = {}
        for i in train_ids:
            split_map[i] = "train"
        for i in val_ids:
            split_map[i] = "val"
        for i in test_ids:
            split_map[i] = "test"

        candidate = pairs_df.copy()
        candidate["split"] = candidate.apply(
            lambda row: split_map[row["id1"]]
            if split_map.get(row["id1"]) == split_map.get(row["id2"])
            else "drop",
            axis=1,
        )
        candidate = candidate[candidate["split"] != "drop"].copy()

        if candidate.empty:
            continue

        counts = candidate["split"].value_counts()
        if any(s not in counts for s in ["train", "val", "test"]):
            continue

        retained = len(candidate)
        if retained > best_retained:
            best_retained = retained
            best_result = candidate

    if best_result is None:
        raise ValueError(
            "Could not build a non-empty id-disjoint split. "
            "Try different seed/fractions or use pair_random strategy."
        )

    return best_result[["pair_id", "id1", "id2", "label", "split"]].reset_index(drop=True)


def build_splits(pairs_df, strategy="id_disjoint", val_frac=0.15, test_frac=0.15, seed=42):
    required = {"pair_id", "id1", "id2", "label"}
    missing = required.difference(set(pairs_df.columns))
    if missing:
        missing_list = ", ".join(sorted(missing))
        raise ValueError(f"pairs_df missing required columns: {missing_list}")

    working = pairs_df.copy()
    working["id1"] = working["id1"].astype(str)
    working["id2"] = working["id2"].astype(str)
    working["label"] = working["label"].astype(float)

    if strategy == "pair_random":
        return build_pair_random_split(working, val_frac=val_frac, test_frac=test_frac, seed=seed)
    if strategy == "id_disjoint":
        return build_id_disjoint_split(working, val_frac=val_frac, test_frac=test_frac, seed=seed)
    raise ValueError(f"Unknown split strategy '{strategy}'.")


def check_split_integrity(split_df):
    required = {"pair_id", "id1", "id2", "label", "split"}
    missing = required.difference(set(split_df.columns))
    if missing:
        missing_list = ", ".join(sorted(missing))
        raise ValueError(f"Split artifact missing required columns: {missing_list}")

    summary = {}
    for split_name in ["train", "val", "test"]:
        section = split_df[split_df["split"] == split_name]
        pos = int((section["label"] == 1).sum())
        neg = int((section["label"] == 0).sum())
        summary[f"{split_name}_pairs"] = int(len(section))
        summary[f"{split_name}_pos"] = pos
        summary[f"{split_name}_neg"] = neg

    split_ids = {}
    for split_name in ["train", "val", "test"]:
        section = split_df[split_df["split"] == split_name]
        ids = set(section["id1"].astype(str)).union(set(section["id2"].astype(str)))
        split_ids[split_name] = ids
        summary[f"{split_name}_ids"] = len(ids)

    summary["train_val_id_overlap"] = len(split_ids["train"].intersection(split_ids["val"]))
    summary["train_test_id_overlap"] = len(split_ids["train"].intersection(split_ids["test"]))
    summary["val_test_id_overlap"] = len(split_ids["val"].intersection(split_ids["test"]))
    return summary


def save_split_artifact(split_df, output_path):
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    split_df.to_csv(path, sep="\t", index=False)
    return path


def load_split_artifact(split_path):
    split_df = pd.read_csv(split_path, sep="\t", dtype=str)
    split_df["pair_id"] = split_df["pair_id"].astype(int)
    split_df["label"] = split_df["label"].astype(float)
    return split_df


def _concat_with_split(train_df, val_df, test_df):
    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()
    train_df["split"] = "train"
    val_df["split"] = "val"
    test_df["split"] = "test"
    return pd.concat([train_df, val_df, test_df], ignore_index=True)[
        ["pair_id", "id1", "id2", "label", "split"]
    ]
