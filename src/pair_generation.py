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

def generate_synthetic_hard_negatives(
    df: pd.DataFrame,
    n_negatives: int = 5000,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate synthetic confusing non-match pairs by mixing attributes
    between different real people.

    Strategies:
      1. **Same-last-name swap**: Two people with the same last name but
         different first names → create a synthetic record that copies one
         person's name onto another's address. The pair (original, synthetic)
         is a NON-match despite sharing a last name + address.
      2. **First-name collision**: Different people with the same first name,
         similar zip — teaches the model that first-name alone isn't enough.
      3. **Attribute-mix negatives**: Take person A's name fields and person B's
         address fields → pair this chimera with person A. Forces the model to
         verify ALL fields, not just names.

    Returns:
        pairs_df: DataFrame [id1, id2, label=0, source, group_id]
        synthetic_df: DataFrame of chimeric records with unique IDs
    """
    rng = np.random.default_rng(random_state)
    df = df.copy()

    name_fields = ["first_name", "middle_name", "last_name", "name_suffix"]
    addr_fields = ["street_address", "city", "zip5"]
    all_fields = name_fields + addr_fields + ["birth_year"]

    pairs = []
    synthetic_rows = []
    syn_counter = 0

    # ── Strategy 1: Same-last-name, different person ──
    last_name_groups = df.groupby("last_name")
    for ln, group in last_name_groups:
        if len(group) < 2 or ln.strip() == "":
            continue
        # Only use groups where first names differ (different people)
        unique_first = group["first_name"].nunique()
        if unique_first < 2:
            continue

        records = group.sample(n=min(len(group), 10), random_state=random_state).to_dict("records")
        for i in range(len(records)):
            for j in range(i + 1, len(records)):
                if records[i]["first_name"] == records[j]["first_name"]:
                    continue
                # Create chimera: person i's name + person j's address
                chimera = {}
                for f in all_fields:
                    chimera[f] = records[i].get(f, "")
                for f in addr_fields:
                    chimera[f] = records[j].get(f, "")
                chimera["id"] = f"syn_hardneg_{syn_counter}"
                chimera["is_synthetic"] = True
                chimera["perturbation_types"] = "name_addr_chimera"
                chimera["seed_id"] = records[i]["id"]
                synthetic_rows.append(chimera)

                # Pair chimera with person j (shares address but different person)
                pairs.append({
                    "id1": str(records[j]["id"]),
                    "id2": chimera["id"],
                    "label": 0,
                    "source": "synthetic_hard_negative",
                    "group_id": -1,
                })
                syn_counter += 1

                if syn_counter >= n_negatives:
                    break
            if syn_counter >= n_negatives:
                break
        if syn_counter >= n_negatives:
            break

    # ── Strategy 2: First-name collision across different last names ──
    remaining = n_negatives - syn_counter
    if remaining > 0:
        first_name_groups = df.groupby("first_name")
        for fn, group in first_name_groups:
            if len(group) < 2 or fn.strip() == "":
                continue
            unique_last = group["last_name"].nunique()
            if unique_last < 2:
                continue

            records = group.sample(n=min(len(group), 6), random_state=random_state).to_dict("records")
            for i in range(len(records)):
                for j in range(i + 1, len(records)):
                    if records[i]["last_name"] == records[j]["last_name"]:
                        continue
                    # Create chimera: person i's name + person j's address
                    chimera = {}
                    for f in all_fields:
                        chimera[f] = records[i].get(f, "")
                    for f in addr_fields:
                        chimera[f] = records[j].get(f, "")
                    chimera["id"] = f"syn_hardneg_{syn_counter}"
                    chimera["is_synthetic"] = True
                    chimera["perturbation_types"] = "firstname_collision"
                    chimera["seed_id"] = records[i]["id"]
                    synthetic_rows.append(chimera)

                    pairs.append({
                        "id1": str(records[j]["id"]),
                        "id2": chimera["id"],
                        "label": 0,
                        "source": "synthetic_hard_negative",
                        "group_id": -1,
                    })
                    syn_counter += 1

                    if syn_counter >= n_negatives:
                        break
                if syn_counter >= n_negatives:
                    break
            if syn_counter >= n_negatives:
                break

    # ── Strategy 3: Attribute-mix (random cross-pollination) ──
    remaining = n_negatives - syn_counter
    if remaining > 0:
        indices = rng.choice(len(df), size=(remaining, 2), replace=True)
        for idx_a, idx_b in indices:
            if idx_a == idx_b:
                continue
            row_a = df.iloc[idx_a]
            row_b = df.iloc[idx_b]
            if row_a["id"] == row_b["id"]:
                continue

            chimera = {}
            for f in name_fields:
                chimera[f] = str(row_a.get(f, ""))
            for f in addr_fields:
                chimera[f] = str(row_b.get(f, ""))
            chimera["birth_year"] = row_a.get("birth_year", "")
            chimera["id"] = f"syn_hardneg_{syn_counter}"
            chimera["is_synthetic"] = True
            chimera["perturbation_types"] = "attribute_mix"
            chimera["seed_id"] = str(row_a["id"])
            synthetic_rows.append(chimera)

            # This chimera shares an address with row_b but is a different person
            pairs.append({
                "id1": str(row_b["id"]),
                "id2": chimera["id"],
                "label": 0,
                "source": "synthetic_hard_negative",
                "group_id": -1,
            })
            syn_counter += 1

    pairs_df = pd.DataFrame(pairs).drop_duplicates(subset=["id1", "id2"]).reset_index(drop=True)
    synthetic_df = pd.DataFrame(synthetic_rows) if synthetic_rows else pd.DataFrame()

    return pairs_df, synthetic_df


def generate_address_change_pairs(
    gold_df: pd.DataFrame,
    records_df: pd.DataFrame,
    n_pairs: int = 3000,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate positive pairs simulating a person moving to a different address.

    Takes records from gold groups, clones each one with a randomly sampled
    address from the full record pool, and labels the (original, clone) pair
    as a match (label=1). This teaches the model that name + birth_year
    alone can confirm identity even when address differs.

    Returns:
        pairs_df: DataFrame with columns [id1, id2, label, source, group_id]
        synthetic_df: DataFrame of the cloned records (with new synthetic IDs)
    """
    rng = np.random.default_rng(random_state)

    gold_records = gold_df[gold_df["silver_group_id"] >= 0].copy()
    if gold_records.empty:
        empty_pairs = pd.DataFrame(columns=["id1", "id2", "label", "source", "group_id"])
        return empty_pairs, pd.DataFrame()

    n_seed = min(n_pairs, len(gold_records))
    seed_records = gold_records.sample(n=n_seed, random_state=random_state)

    # Build address pool — all non-empty, non-placeholder addresses in the dataset
    addr_cols = ["street_address", "city", "zip5"]
    address_pool = (
        records_df[addr_cols]
        .dropna()
        .drop_duplicates()
        .reset_index(drop=True)
    )
    address_pool = address_pool[address_pool["street_address"].str.strip() != ""].reset_index(drop=True)

    pairs = []
    synthetic_rows = []

    for i, (_, row) in enumerate(seed_records.iterrows()):
        # Sample an address that differs from the record's current one
        candidates = address_pool[address_pool["street_address"] != row.get("street_address", "")]
        if candidates.empty:
            continue
        addr_row = candidates.iloc[rng.integers(0, len(candidates))]

        syn_id = f"syn_addr_{i}"
        new_row = row.to_dict()
        new_row["id"] = syn_id
        new_row["street_address"] = addr_row["street_address"]
        new_row["city"] = addr_row["city"]
        new_row["zip5"] = addr_row["zip5"]
        new_row["is_synthetic"] = True
        new_row["perturbation_types"] = "address_swap"
        new_row["seed_id"] = row["id"]
        new_row["silver_group_id"] = -1

        synthetic_rows.append(new_row)
        pairs.append({
            "id1": str(row["id"]),
            "id2": syn_id,
            "label": 1,
            "source": "address_change",
            "group_id": -1,
        })

    synthetic_df = pd.DataFrame(synthetic_rows)
    pairs_df = pd.DataFrame(pairs)
    return pairs_df, synthetic_df


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
