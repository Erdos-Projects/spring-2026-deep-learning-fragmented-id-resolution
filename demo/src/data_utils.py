import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch


DEFAULT_BASELINE_ATTRIBUTES = [
    "first_name",
    "last_name",
    "house_num",
    "street_name",
    "zip_code",
]

DEFAULT_EXTENDED_ATTRIBUTES = DEFAULT_BASELINE_ATTRIBUTES + [
    "age",
    "sex",
    "race_desc",
    "ethnic_desc",
]

DEFAULT_EXTENDED_MIDL_ATTRIBUTES = [
    "first_name",
    "midl_name",
    "last_name",
    "house_num",
    "street_name",
    "zip_code",
    "age",
    "sex",
    "race_desc",
    "ethnic_desc",
]

DEFAULT_EXTENDED_NO_DEMO_ATTRIBUTES = DEFAULT_BASELINE_ATTRIBUTES + [
    "age",
    "sex",
]

DEFAULT_EXTENDED_NO_DEMO_MIDL_ATTRIBUTES = [
    "first_name",
    "midl_name",
    "last_name",
    "house_num",
    "street_name",
    "zip_code",
    "age",
    "sex",
]

ATTRIBUTE_SET_DEFINITIONS = {
    "baseline": DEFAULT_BASELINE_ATTRIBUTES,
    "extended": DEFAULT_EXTENDED_ATTRIBUTES,
    "extended_midl": DEFAULT_EXTENDED_MIDL_ATTRIBUTES,
    "extended_no_demo": DEFAULT_EXTENDED_NO_DEMO_ATTRIBUTES,
    "extended_no_demo_midl": DEFAULT_EXTENDED_NO_DEMO_MIDL_ATTRIBUTES,
}

ATTRIBUTE_SET_NAMES = list(ATTRIBUTE_SET_DEFINITIONS.keys())


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_attributes(attribute_set="baseline", attributes_csv=None):
    if attributes_csv:
        attrs = [a.strip() for a in attributes_csv.split(",") if a.strip()]
        if not attrs:
            raise ValueError("attributes_csv was provided but no valid attributes were parsed.")
        return attrs

    if attribute_set in ATTRIBUTE_SET_DEFINITIONS:
        return list(ATTRIBUTE_SET_DEFINITIONS[attribute_set])

    raise ValueError(f"Unknown attribute_set '{attribute_set}'.")


def load_prepared_data(data_path):
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"Prepared data file not found: {path}")
    data_df = pd.read_csv(path, sep="\t", dtype=str).fillna("")
    if "id" not in data_df.columns:
        raise ValueError("Prepared data must include an 'id' column.")
    data_df["id"] = data_df["id"].astype(str)
    return data_df


def _read_pairs_with_label(path, label):
    pair_df = pd.read_csv(path, sep="\t", dtype=str)
    required = {"id1", "id2"}
    missing = required.difference(set(pair_df.columns))
    if missing:
        missing_list = ", ".join(sorted(missing))
        raise ValueError(f"Pair file {path} missing required columns: {missing_list}")
    pair_df = pair_df[["id1", "id2"]].copy()
    pair_df["id1"] = pair_df["id1"].astype(str)
    pair_df["id2"] = pair_df["id2"].astype(str)
    pair_df["label"] = float(label)
    return pair_df


def canonical_pair_series(pairs_df):
    return pairs_df.apply(
        lambda row: "||".join(sorted((str(row["id1"]), str(row["id2"])))),
        axis=1,
    )


def count_duplicate_canonical_pairs(pairs_df):
    canonical = canonical_pair_series(pairs_df)
    return int(canonical.duplicated().sum())


def load_labeled_pairs(dpl_path, ndpl_path, fail_on_duplicate_pairs=True):
    dpl = _read_pairs_with_label(dpl_path, label=1.0)
    ndpl = _read_pairs_with_label(ndpl_path, label=0.0)
    all_pairs = pd.concat([dpl, ndpl], ignore_index=True)
    all_pairs.insert(0, "pair_id", np.arange(len(all_pairs), dtype=int))

    duplicate_count = count_duplicate_canonical_pairs(all_pairs)
    if duplicate_count > 0 and fail_on_duplicate_pairs:
        raise ValueError(
            f"Found {duplicate_count} duplicated canonical pairs in combined DPL/NDPL files."
        )
    return all_pairs


def count_missing_pair_ids(pairs_df, valid_ids):
    valid_ids = set(valid_ids)
    missing_id1 = set(pairs_df.loc[~pairs_df["id1"].isin(valid_ids), "id1"].astype(str))
    missing_id2 = set(pairs_df.loc[~pairs_df["id2"].isin(valid_ids), "id2"].astype(str))
    missing_ids = missing_id1.union(missing_id2)
    missing_rows = int((~pairs_df["id1"].isin(valid_ids) | ~pairs_df["id2"].isin(valid_ids)).sum())
    return missing_rows, missing_ids
