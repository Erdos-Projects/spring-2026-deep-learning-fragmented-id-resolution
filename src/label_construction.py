"""
Construct silver and gold duplicate groups from normalized voter records.

Silver groups: exact match on (first, middle, last, suffix, street, city, zip5).
Gold groups:   silver groups validated by birth_year consistency.

Also integrates the existing DPL (duplicate pair list) ground truth when available.
"""

from typing import Tuple

import pandas as pd


# ── Signature construction ──────────────────────────────────────────────────

SIGNATURE_FIELDS = [
    "first_name", "middle_name", "last_name", "name_suffix",
    "street_address", "city", "zip5",
]


def build_signature(row: pd.Series) -> str:
    """Create a canonical signature string for deterministic grouping."""
    parts = []
    for field in SIGNATURE_FIELDS:
        val = str(row.get(field, "")).strip()
        parts.append(val)
    return "||".join(parts)


def _is_placeholder_address(street: str) -> bool:
    """Filter out empty or placeholder street addresses."""
    if not street or street.strip() == "":
        return True
    placeholders = {"GENERAL DELIVERY", "PO BOX", "P O BOX", "HOMELESS", "UNKNOWN"}
    upper = street.strip().upper()
    return any(upper.startswith(p) for p in placeholders) or len(upper) < 3


# ── Silver groups ───────────────────────────────────────────────────────────

def build_silver_groups(df: pd.DataFrame) -> pd.DataFrame:
    """
    Group records by exact signature. Returns the DataFrame with an added
    'silver_group_id' column (integer group label, -1 for singletons).

    Only groups with size > 1 AND non-placeholder addresses are kept.
    """
    df = df.copy()
    df["_signature"] = df.apply(build_signature, axis=1)

    # Filter out rows with placeholder addresses
    df["_valid_addr"] = ~df["street_address"].apply(_is_placeholder_address)

    # Build group IDs only for valid-address records
    valid = df[df["_valid_addr"]].copy()
    group_map = valid.groupby("_signature").ngroup()
    group_sizes = valid.groupby("_signature")["id"].transform("count")

    # Only keep groups with size > 1
    valid["silver_group_id"] = group_map.where(group_sizes > 1, -1)

    # Merge back
    df["silver_group_id"] = -1
    df.loc[valid.index, "silver_group_id"] = valid["silver_group_id"]

    # Clean up temp columns
    df.drop(columns=["_signature", "_valid_addr"], inplace=True)

    return df


# ── Gold groups (birth-year validated) ──────────────────────────────────────

def validate_gold_groups(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Validate silver groups using birth_year.

    Returns:
        gold_df: records in silver groups where ALL members share the same birth_year
        conflict_df: records in silver groups with conflicting birth years (hard negatives)
        stats: dict with counts {n_silver, n_gold, n_conflict, n_missing_by}
    """
    # Only look at silver groups (group_id >= 0)
    grouped = df[df["silver_group_id"] >= 0].copy()

    if grouped.empty:
        return (
            pd.DataFrame(columns=df.columns),
            pd.DataFrame(columns=df.columns),
            {"n_silver": 0, "n_gold": 0, "n_conflict": 0, "n_missing_by": 0},
        )

    # For each silver group, check birth_year consistency
    def _check_group(group):
        birth_years = group["birth_year"].dropna().unique()
        if len(birth_years) == 0:
            return "missing"
        elif len(birth_years) == 1:
            return "gold"
        else:
            return "conflict"

    group_status = (
        grouped.groupby("silver_group_id")
        .apply(_check_group, include_groups=False)
        .rename("group_status")
    )

    grouped = grouped.merge(
        group_status, left_on="silver_group_id", right_index=True, how="left"
    )

    gold_df = grouped[grouped["group_status"] == "gold"].drop(columns=["group_status"])
    conflict_df = grouped[grouped["group_status"] == "conflict"].drop(columns=["group_status"])
    missing_df = grouped[grouped["group_status"] == "missing"].drop(columns=["group_status"])

    n_silver = grouped["silver_group_id"].nunique()
    n_gold = gold_df["silver_group_id"].nunique()
    n_conflict = conflict_df["silver_group_id"].nunique()
    n_missing = missing_df["silver_group_id"].nunique()

    stats = {
        "n_silver_groups": n_silver,
        "n_gold_groups": n_gold,
        "n_conflict_groups": n_conflict,
        "n_missing_by_groups": n_missing,
        "n_gold_records": len(gold_df),
        "n_conflict_records": len(conflict_df),
    }

    return gold_df, conflict_df, stats


# ── Integration with existing DPL/NDPL ─────────────────────────────────────

def load_existing_pairs(
    dpl_path: str,
    ndpl_path: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load existing duplicate (DPL) and non-duplicate (NDPL) pair lists.

    Returns:
        dpl: DataFrame with columns [id1, id2] — positive pairs
        ndpl: DataFrame with columns [id1, id2, participation] — negative pairs
    """
    dpl = pd.read_csv(dpl_path, sep="\t", dtype=str)
    dpl.columns = [c.strip() for c in dpl.columns]

    ndpl = pd.read_csv(ndpl_path, sep="\t", dtype=str)
    ndpl.columns = [c.strip() for c in ndpl.columns]

    return dpl, ndpl
