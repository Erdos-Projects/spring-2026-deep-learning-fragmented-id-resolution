import numpy as np
import pandas as pd


DEFAULT_BLOCKING_KEYS = ["first_name", "age"]


def parse_blocking_keys(raw_value, default_keys=None):
    if default_keys is None:
        default_keys = DEFAULT_BLOCKING_KEYS
    if raw_value is None:
        return list(default_keys)
    value = raw_value.strip().lower()
    if value in {"", "none", "off"}:
        return []
    return [item.strip() for item in raw_value.split(",") if item.strip()]


def compute_block_mask(pairs_df, data_df, blocking_keys, blocking_mode):
    if not blocking_keys:
        return np.ones(len(pairs_df), dtype=bool)

    missing_keys = [key for key in blocking_keys if key not in data_df.columns]
    if missing_keys:
        missing_list = ", ".join(sorted(missing_keys))
        raise ValueError(f"Blocking keys not present in prepared data: {missing_list}")

    attr_df = data_df[["id"] + blocking_keys].copy()
    merged = pairs_df.merge(attr_df, left_on="id1", right_on="id", how="left")
    merged = merged.merge(attr_df, left_on="id2", right_on="id", how="left", suffixes=("_1", "_2"))

    comparisons = []
    for key in blocking_keys:
        left = merged[f"{key}_1"].fillna("").astype(str)
        right = merged[f"{key}_2"].fillna("").astype(str)
        comparisons.append((left == right).to_numpy())

    if blocking_mode == "all":
        mask = np.logical_and.reduce(comparisons)
    elif blocking_mode == "any":
        mask = np.logical_or.reduce(comparisons)
    else:
        raise ValueError(f"Unknown blocking mode '{blocking_mode}'.")
    return mask.astype(bool)


def summarize_blocking(labels, block_mask):
    labels = np.asarray(labels).astype(int)
    block_mask = np.asarray(block_mask).astype(bool)

    summary = {
        "candidate_pair_count": int(block_mask.sum()),
        "blocked_pair_count": int((~block_mask).sum()),
        "blocking_pass_rate": float(block_mask.mean()) if len(block_mask) else 0.0,
    }

    for label_value, prefix in [(1, "positive"), (0, "negative")]:
        label_mask = labels == label_value
        count = int(label_mask.sum())
        if count == 0:
            pass_rate = float("nan")
            pass_count = 0
        else:
            pass_count = int((block_mask & label_mask).sum())
            pass_rate = pass_count / count
        summary[f"{prefix}_pair_count"] = count
        summary[f"{prefix}_candidate_count"] = pass_count
        summary[f"{prefix}_pass_rate"] = pass_rate

    return summary
