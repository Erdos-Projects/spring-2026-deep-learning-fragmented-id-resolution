"""
Blocking strategies for scalable candidate generation.

At 962K records, brute-force all-pairs comparison (~460 billion pairs) is
infeasible.  Blocking partitions records into small buckets so that only
intra-bucket pairs are considered, reducing complexity from O(n²) to
O(n × average_block_size).

Multiple blocking keys are combined (union of candidates) to maximize
recall while keeping candidate set size manageable.
"""

from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd


# ── Phonetic helpers (Soundex) ──────────────────────────────────────────────

def _soundex(name: str) -> str:
    """Compute American Soundex code for a name string."""
    if not name:
        return ""
    name = name.upper().strip()
    if not name:
        return ""

    # Keep first letter
    first = name[0]

    # Mapping
    MAP = {
        "B": "1", "F": "1", "P": "1", "V": "1",
        "C": "2", "G": "2", "J": "2", "K": "2", "Q": "2",
        "S": "2", "X": "2", "Z": "2",
        "D": "3", "T": "3",
        "L": "4",
        "M": "5", "N": "5",
        "R": "6",
    }

    digits = []
    prev = MAP.get(first, "0")
    for ch in name[1:]:
        d = MAP.get(ch, "0")
        if d != "0" and d != prev:
            digits.append(d)
        prev = d if d != "0" else prev
        if len(digits) == 3:
            break

    code = first + "".join(digits)
    return code.ljust(4, "0")[:4]


# ── Blocking key generators ────────────────────────────────────────────────

def _key_zip_last3(row: pd.Series) -> Optional[str]:
    """Block on ZIP5 + first 3 chars of last name."""
    z = str(row.get("zip5", "")).strip()
    ln = str(row.get("last_name", "")).strip()[:3]
    if len(z) >= 3 and len(ln) >= 2:
        return f"ZL3_{z}_{ln}"
    return None


def _key_zip_soundex(row: pd.Series) -> Optional[str]:
    """Block on ZIP5 + Soundex of last name."""
    z = str(row.get("zip5", "")).strip()
    ln = str(row.get("last_name", "")).strip()
    if len(ln) < 2:          # single-char surnames produce overly generic keys
        return None
    sx = _soundex(ln)
    if len(z) >= 3 and sx:
        return f"ZSX_{z}_{sx}"
    return None


def _key_first3_last3_by(row: pd.Series) -> Optional[str]:
    """Block on first_name[:3] + last_name[:3] + birth_year."""
    fn = str(row.get("first_name", "")).strip()[:3]
    ln = str(row.get("last_name", "")).strip()[:3]
    by = str(row.get("birth_year", "")).strip()
    if len(fn) >= 2 and len(ln) >= 2 and by and by != "<NA>" and by != "nan":
        return f"FLB_{fn}_{ln}_{by}"
    return None


def _key_soundex_first_last(row: pd.Series) -> Optional[str]:
    """Block on Soundex(first_name) + Soundex(last_name)."""
    fn = str(row.get("first_name", "")).strip()
    ln = str(row.get("last_name", "")).strip()
    if len(fn) < 2 or len(ln) < 2:   # short names give overly generic soundex
        return None
    sf = _soundex(fn)
    sl = _soundex(ln)
    if sf and sl:
        return f"SFL_{sf}_{sl}"
    return None


def _key_address_token_by(row: pd.Series) -> Optional[str]:
    """Block on first numeric token of address + birth_year.

    Requires first_name and last_name to each be >= 2 characters so that
    records with stub names (e.g. last_name='A') do not end up in broad
    house-number+birth-year blocks alongside unrelated people.
    """
    fn = str(row.get("first_name", "")).strip()
    ln = str(row.get("last_name", "")).strip()
    if len(fn) < 2 or len(ln) < 2:
        return None
    addr = str(row.get("street_address", "")).strip()
    by = str(row.get("birth_year", "")).strip()
    if not addr or not by or by in ("<NA>", "nan"):
        return None
    # Extract first numeric token (house number)
    tokens = addr.split()
    house_num = ""
    for t in tokens:
        if t.isdigit():
            house_num = t
            break
    if house_num:
        return f"ABY_{house_num}_{by}"
    return None


def _key_birth_year_first3(row: pd.Series) -> Optional[str]:
    """Block on birth_year + first_name[:3].

    Catches changed-name pairs (e.g., maiden name to married name) who share
    birth year and first name prefix.  Does not depend on last_name or address.
    """
    fn = str(row.get("first_name", "")).strip()[:3]
    by = str(row.get("birth_year", "")).strip()
    if len(fn) >= 2 and by and by not in ("<NA>", "nan", "None", ""):
        return f"BYF3_{by}_{fn}"
    return None


def _key_last_name_soundex_by(row: pd.Series) -> Optional[str]:
    """Block on Soundex(last_name) + birth_year.

    Catches address movers who keep the same name.  Does not depend on ZIP
    or street_address, so it complements the zip-based keys.
    """
    ln = str(row.get("last_name", "")).strip()
    by = str(row.get("birth_year", "")).strip()
    if len(ln) < 2 or not by or by in ("<NA>", "nan", "None", ""):
        return None
    sx = _soundex(ln)
    if sx:
        return f"LSXBY_{sx}_{by}"
    return None


def _key_first_last_initial(row: pd.Series) -> Optional[str]:
    """Block on first_name[:1] + last_name + birth_year.

    Catches nickname variants (e.g., WILLIAM vs WILLY share 'W' initial when
    last_name and birth_year match).  The full last_name keeps blocks small.
    """
    fn = str(row.get("first_name", "")).strip()[:1]
    ln = str(row.get("last_name", "")).strip()
    by = str(row.get("birth_year", "")).strip()
    if fn and len(ln) >= 2 and by and by not in ("<NA>", "nan", "None", ""):
        return f"FiLBY_{fn}_{ln}_{by}"
    return None


# ── Main blocking interface ─────────────────────────────────────────────────

BLOCKING_FUNCTIONS = [
    _key_zip_last3,
    _key_zip_soundex,
    _key_first3_last3_by,
    _key_soundex_first_last,
    _key_address_token_by,
    _key_birth_year_first3,
    _key_last_name_soundex_by,
    _key_first_last_initial,
]


def build_block_index(
    df: pd.DataFrame,
    blocking_fns: Optional[List] = None,
    max_block_size: int = 500,
) -> Dict[str, List[int]]:
    """
    Build a blocking index mapping blocking keys to lists of DataFrame
    integer positions.

    Args:
        df: Normalized DataFrame (must have standard columns).
        blocking_fns: List of blocking key functions. Defaults to all built-in.
        max_block_size: Discard blocks larger than this (too generic to be useful).

    Returns:
        Dict mapping blocking_key → list of integer row positions.
    """
    if blocking_fns is None:
        blocking_fns = BLOCKING_FUNCTIONS

    blocks: Dict[str, List[int]] = defaultdict(list)

    for idx in range(len(df)):
        row = df.iloc[idx]
        for fn in blocking_fns:
            key = fn(row)
            if key is not None:
                blocks[key].append(idx)

    # Prune oversized blocks
    pruned = {k: v for k, v in blocks.items() if 1 < len(v) <= max_block_size}
    return pruned


def generate_candidate_pairs_from_blocks(
    block_index: Dict[str, List[int]],
    max_pairs_per_block: int = 200,
    rng: Optional[np.random.Generator] = None,
) -> Set[Tuple[int, int]]:
    """
    Generate candidate pairs from blocking index.

    For small blocks, all pairs are enumerated.  For larger blocks,
    pairs are randomly sampled to cap total work.

    Returns:
        Set of (row_idx_i, row_idx_j) pairs with i < j.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    candidates: Set[Tuple[int, int]] = set()

    for key, members in block_index.items():
        n = len(members)
        n_pairs = n * (n - 1) // 2

        if n_pairs <= max_pairs_per_block:
            # Enumerate all
            for i in range(n):
                for j in range(i + 1, n):
                    a, b = members[i], members[j]
                    candidates.add((min(a, b), max(a, b)))
        else:
            # Sample pairs
            seen = set()
            attempts = 0
            while len(seen) < max_pairs_per_block and attempts < max_pairs_per_block * 3:
                attempts += 1
                i, j = rng.choice(n, 2, replace=False)
                a, b = members[i], members[j]
                pair = (min(a, b), max(a, b))
                if pair not in seen:
                    seen.add(pair)
                    candidates.add(pair)

    return candidates


def block_and_generate_candidates(
    df: pd.DataFrame,
    max_block_size: int = 500,
    max_pairs_per_block: int = 200,
    random_state: int = 42,
) -> List[Tuple[int, int]]:
    """
    Convenience function: build blocks and return a deduplicated list of
    candidate pairs (as integer row positions).
    """
    rng = np.random.default_rng(random_state)
    block_index = build_block_index(df, max_block_size=max_block_size)

    n_blocks = len(block_index)
    block_sizes = [len(v) for v in block_index.values()]
    print(f"Blocking: {n_blocks:,} blocks, "
          f"mean size={np.mean(block_sizes):.1f}, "
          f"max size={max(block_sizes) if block_sizes else 0}")

    candidates = generate_candidate_pairs_from_blocks(
        block_index, max_pairs_per_block=max_pairs_per_block, rng=rng
    )
    print(f"Candidate pairs from blocking: {len(candidates):,}")
    return sorted(candidates)
