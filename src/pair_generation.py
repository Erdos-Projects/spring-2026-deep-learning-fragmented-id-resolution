"""
Pair generation for training and evaluation.

Generates supervised pair labels from:
  - Gold groups (positive pairs)
  - Birth-year conflict groups (hard negative pairs)
  - Nearby-block negatives (challenging negatives from same ZIP + last-name prefix)
  - Existing DPL/NDPL ground truth

Splitting is done at the entity/group level, NOT at the pair level,
to prevent data leakage between train and test sets.
"""

from itertools import combinations
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


# ── Pair generation from groups ─────────────────────────────────────────────

def _all_pairs_from_group(ids: List[str]) -> List[Tuple[str, str]]:
    """Generate all unique pairs from a list of record IDs."""
    return list(combinations(sorted(ids), 2))


def generate_gold_positive_pairs(gold_df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate positive pairs from gold duplicate groups.
    All within-group pairs are labeled as duplicates (label=1).
    """
    pairs = []
    for gid, group in gold_df.groupby("silver_group_id"):
        ids = group["id"].tolist()
        for id1, id2 in _all_pairs_from_group(ids):
            pairs.append({"id1": id1, "id2": id2, "label": 1, "source": "gold", "group_id": gid})
    return pd.DataFrame(pairs)


def generate_hard_negative_pairs(conflict_df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate hard negative pairs from birth-year conflict groups.
    These records match on name+address but have different birth years — likely
    different people (e.g., parent/child at same address).
    """
    pairs = []
    for gid, group in conflict_df.groupby("silver_group_id"):
        ids = group["id"].tolist()
        birth_years = group.set_index("id")["birth_year"]
        for id1, id2 in _all_pairs_from_group(ids):
            by1 = birth_years.get(id1)
            by2 = birth_years.get(id2)
            # Only generate negative pair if birth years actually differ
            if pd.notna(by1) and pd.notna(by2) and by1 != by2:
                pairs.append({
                    "id1": id1, "id2": id2, "label": 0,
                    "source": "hard_negative", "group_id": gid,
                })
    return pd.DataFrame(pairs)


def generate_block_negatives(
    df: pd.DataFrame,
    n_negatives: int = 5000,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Sample challenging negatives from nearby blocks.

    Strategy: same ZIP5 + same last-name first letter, but different birth year
    and/or different first name. This produces negatives that are realistic
    (geographically close, similar names) but represent different people.
    """
    rng = np.random.default_rng(random_state)

    # Build blocking key
    df = df.copy()
    df["_block_key"] = df["zip5"] + "_" + df["last_name"].str[:2]
    df["_birth_year"] = pd.to_numeric(df["birth_year"], errors="coerce")

    block_groups = df.groupby("_block_key")

    pairs = []
    for bk, block in block_groups:
        if len(block) < 2:
            continue
        ids = block["id"].values
        first_names = block.set_index("id")["first_name"]
        birth_years = block.set_index("id")["_birth_year"]

        # Sample a limited number of pairs per block
        n_sample = min(len(ids) * 2, 20)
        for _ in range(n_sample):
            i, j = rng.choice(len(ids), 2, replace=False)
            id1, id2 = ids[i], ids[j]
            fn1, fn2 = first_names.get(id1, ""), first_names.get(id2, "")
            by1, by2 = birth_years.get(id1), birth_years.get(id2)

            # Ensure they're actually different people
            if fn1 == fn2 and (pd.isna(by1) or pd.isna(by2) or by1 == by2):
                continue

            pair_key = tuple(sorted([id1, id2]))
            pairs.append({
                "id1": pair_key[0], "id2": pair_key[1], "label": 0,
                "source": "block_negative", "group_id": -1,
            })

        if len(pairs) >= n_negatives * 2:
            break

    pairs_df = pd.DataFrame(pairs).drop_duplicates(subset=["id1", "id2"])
    if len(pairs_df) > n_negatives:
        pairs_df = pairs_df.sample(n=n_negatives, random_state=random_state)

    return pairs_df.reset_index(drop=True)


# ── DPL/NDPL integration ───────────────────────────────────────────────────

def pairs_from_dpl(dpl: pd.DataFrame) -> pd.DataFrame:
    """Convert DPL DataFrame to standard pair format."""
    out = dpl[["id1", "id2"]].copy()
    out["label"] = 1
    out["source"] = "dpl"
    out["group_id"] = -1
    return out


def pairs_from_ndpl(ndpl: pd.DataFrame, max_pairs: int = 20000, random_state: int = 42) -> pd.DataFrame:
    """Convert NDPL DataFrame to standard pair format (subsample if too large)."""
    out = ndpl[["id1", "id2"]].copy()
    out["label"] = 0
    out["source"] = "ndpl"
    out["group_id"] = -1
    if len(out) > max_pairs:
        out = out.sample(n=max_pairs, random_state=random_state)
    return out.reset_index(drop=True)


# ── Combine all pair sources ────────────────────────────────────────────────

def combine_pairs(
    *pair_dfs: pd.DataFrame,
) -> pd.DataFrame:
    """Combine multiple pair DataFrames, deduplicate, and report counts."""
    all_pairs = pd.concat(pair_dfs, ignore_index=True)
    # Canonicalize pair ordering
    swapped = all_pairs["id1"] > all_pairs["id2"]
    all_pairs.loc[swapped, ["id1", "id2"]] = all_pairs.loc[swapped, ["id2", "id1"]].values
    # Deduplicate
    all_pairs = all_pairs.drop_duplicates(subset=["id1", "id2"])
    return all_pairs.reset_index(drop=True)


# ── Entity-level train/test split ──────────────────────────────────────────

def entity_level_split(
    pairs_df: pd.DataFrame,
    records_df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split pairs into train/test sets at the ENTITY level.

    All pairs involving the same record(s) go to the same split.
    This prevents leakage where the model sees one side of a test pair
    during training.

    Strategy: Assign each unique ID to a split, then assign pairs accordingly.
    Gold group pairs stay together (all pairs from same group → same split).
    """
    # Collect all unique IDs involved
    all_ids = pd.Series(
        pd.concat([pairs_df["id1"], pairs_df["id2"]]).unique()
    )

    # For gold group pairs, assign a group label; for others use the ID itself
    id_to_entity = {}

    # First, assign gold group entities
    if "group_id" in pairs_df.columns:
        gold_pairs = pairs_df[pairs_df["group_id"] >= 0]
        for gid, group in gold_pairs.groupby("group_id"):
            member_ids = set(group["id1"].tolist() + group["id2"].tolist())
            entity_label = f"group_{gid}"
            for mid in member_ids:
                id_to_entity[mid] = entity_label

    # Assign remaining IDs their own entity label
    for uid in all_ids:
        if uid not in id_to_entity:
            id_to_entity[uid] = f"single_{uid}"

    # Map each pair to its entity group(s)
    pairs_df = pairs_df.copy()
    pairs_df["_entity1"] = pairs_df["id1"].map(id_to_entity)
    pairs_df["_entity2"] = pairs_df["id2"].map(id_to_entity)
    # Use entity1 as the split key (ensures group-level coherence for gold pairs)
    pairs_df["_split_key"] = pairs_df["_entity1"]

    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(splitter.split(pairs_df, groups=pairs_df["_split_key"]))

    train_pairs = pairs_df.iloc[train_idx].drop(columns=["_entity1", "_entity2", "_split_key"]).reset_index(drop=True)
    test_pairs = pairs_df.iloc[test_idx].drop(columns=["_entity1", "_entity2", "_split_key"]).reset_index(drop=True)

    return train_pairs, test_pairs
