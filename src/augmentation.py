"""
Rule-based synthetic augmentation for fragmented identity simulation.

Instead of using an LLM (slow, unreliable), we apply controlled, reproducible
perturbations that simulate real-world data entry variation:

  - Name variations: nicknames, initials, typos, transpositions, hyphen changes
  - Address variations: suffix swaps, unit format changes, whitespace noise
  - Missingness: selective blanking of non-critical fields

Anti-overfitting safeguard: perturbation types are tagged so a held-out set
can be withheld from training and used only at test time.
"""

import random
import re
import string
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ── Nickname dictionary (common English first-name mappings) ────────────────
NICKNAME_MAP = {
    "WILLIAM": ["WILL", "BILL", "BILLY", "WILLY", "LIAM"],
    "JAMES": ["JIM", "JIMMY", "JAMIE"],
    "ROBERT": ["ROB", "BOB", "BOBBY", "ROBBIE"],
    "JOHN": ["JACK", "JOHNNY", "JON"],
    "RICHARD": ["RICK", "DICK", "RICH", "RICKY"],
    "THOMAS": ["TOM", "TOMMY"],
    "CHARLES": ["CHARLIE", "CHUCK", "CHAS"],
    "MICHAEL": ["MIKE", "MIKEY", "MICK"],
    "JOSEPH": ["JOE", "JOEY"],
    "DAVID": ["DAVE", "DAVEY"],
    "DANIEL": ["DAN", "DANNY"],
    "MATTHEW": ["MATT", "MATTY"],
    "CHRISTOPHER": ["CHRIS", "TOPHER"],
    "ANDREW": ["ANDY", "DREW"],
    "ELIZABETH": ["LIZ", "LIZZY", "BETH", "BETTY", "ELIZA"],
    "MARGARET": ["MAGGIE", "PEGGY", "MEG", "MARGE"],
    "JENNIFER": ["JEN", "JENNY"],
    "KATHERINE": ["KATE", "KATHY", "KAT", "KATIE"],
    "CATHERINE": ["KATE", "CATHY", "CAT", "KATIE"],
    "PATRICIA": ["PAT", "PATTY", "TRICIA"],
    "JESSICA": ["JESS", "JESSIE"],
    "REBECCA": ["BECKY", "BECCA"],
    "STEPHANIE": ["STEPH", "STEPHIE"],
    "NICHOLAS": ["NICK", "NICKY"],
    "BENJAMIN": ["BEN", "BENNY"],
    "ALEXANDER": ["ALEX", "XANDER"],
    "SAMANTHA": ["SAM", "SAMMY"],
    "JONATHAN": ["JON", "JONNY"],
    "TIMOTHY": ["TIM", "TIMMY"],
    "EDWARD": ["ED", "EDDIE", "TED"],
    "ANTHONY": ["TONY"],
    "STEPHEN": ["STEVE"],
    "STEVEN": ["STEVE"],
    "JOSHUA": ["JOSH"],
    "KENNETH": ["KEN", "KENNY"],
    "GREGORY": ["GREG"],
    "GERALD": ["JERRY"],
    "NATHAN": ["NATE"],
    "RAYMOND": ["RAY"],
    "LAWRENCE": ["LARRY"],
    "THEODORE": ["TED", "TEDDY"],
    "DOROTHY": ["DOT", "DOTTY"],
    "VIRGINIA": ["GINNY"],
    "FREDERICK": ["FRED", "FREDDY"],
    "DEBORAH": ["DEB", "DEBBIE"],
    "BARBARA": ["BARB"],
    "SUZANNE": ["SUE", "SUZY"],
    "SUSAN": ["SUE", "SUZY"],
    "DONNA": ["DONNIE"],
    "CAROLYN": ["CAROL"],
    "CHRISTINA": ["CHRIS", "TINA"],
    "CHRISTINE": ["CHRIS", "TINA"],
}

# Build reverse map for bidirectional lookup
REVERSE_NICKNAME = {}
for formal, nicks in NICKNAME_MAP.items():
    for nick in nicks:
        if nick not in REVERSE_NICKNAME:
            REVERSE_NICKNAME[nick] = formal

# ── Street suffix variations ────────────────────────────────────────────────
SUFFIX_VARIANTS = {
    "ST": ["STREET", "ST", "STR"],
    "DR": ["DRIVE", "DR", "DRV"],
    "RD": ["ROAD", "RD"],
    "AVE": ["AVENUE", "AVE", "AV"],
    "BLVD": ["BOULEVARD", "BLVD"],
    "LN": ["LANE", "LN"],
    "CT": ["COURT", "CT"],
    "CIR": ["CIRCLE", "CIR"],
    "PL": ["PLACE", "PL"],
    "TRL": ["TRAIL", "TRL"],
    "PKWY": ["PARKWAY", "PKWY"],
    "HWY": ["HIGHWAY", "HWY"],
}

# ── Unit format variations ──────────────────────────────────────────────────
UNIT_PREFIXES = ["APT", "UNIT", "STE", "#", "APARTMENT", "SUITE", "BLDG"]


# ── Perturbation registry (each function is tagged for train/test splitting) ──

class PerturbationType:
    """Tags for perturbation types to enable train/test held-out splitting."""
    NICKNAME = "nickname"
    INITIAL = "initial"
    DROP_MIDDLE = "drop_middle"
    TYPO_SWAP = "typo_swap"
    TYPO_INSERT = "typo_insert"
    TYPO_DELETE = "typo_delete"
    TYPO_REPLACE = "typo_replace"
    SUFFIX_SWAP = "suffix_swap"
    UNIT_FORMAT = "unit_format"
    DIRECTIONAL_EXPAND = "directional_expand"
    WHITESPACE_NOISE = "whitespace_noise"
    BLANK_MIDDLE = "blank_middle"
    BLANK_SUFFIX = "blank_suffix"
    BLANK_UNIT = "blank_unit"
    HYPHEN_CHANGE = "hyphen_change"

    # Default split: train uses these, test holds out the rest
    TRAIN_TYPES = {
        NICKNAME, INITIAL, DROP_MIDDLE, TYPO_SWAP, SUFFIX_SWAP,
        UNIT_FORMAT, BLANK_MIDDLE, WHITESPACE_NOISE,
    }
    TEST_ONLY_TYPES = {
        TYPO_INSERT, TYPO_DELETE, TYPO_REPLACE,
        DIRECTIONAL_EXPAND, BLANK_SUFFIX, BLANK_UNIT, HYPHEN_CHANGE,
    }


# ── Individual perturbation functions ───────────────────────────────────────

def _nickname_sub(name: str, rng: np.random.Generator) -> Tuple[str, str]:
    """Replace a first name with a known nickname or vice versa."""
    upper = name.upper()
    if upper in NICKNAME_MAP:
        return rng.choice(NICKNAME_MAP[upper]), PerturbationType.NICKNAME
    if upper in REVERSE_NICKNAME:
        # Go formal, or pick among other nicknames
        formal = REVERSE_NICKNAME[upper]
        options = [formal] + [n for n in NICKNAME_MAP.get(formal, []) if n != upper]
        return rng.choice(options), PerturbationType.NICKNAME
    return name, ""


def _to_initial(name: str) -> Tuple[str, str]:
    """Replace name with its initial."""
    if name and len(name) > 1:
        return name[0], PerturbationType.INITIAL
    return name, ""


def _typo_swap(name: str, rng: np.random.Generator) -> Tuple[str, str]:
    """Swap two adjacent characters."""
    if len(name) < 3:
        return name, ""
    # Pick a position (not first or last to avoid confusion)
    pos = rng.integers(1, len(name) - 1)
    chars = list(name)
    chars[pos], chars[pos - 1] = chars[pos - 1], chars[pos]
    return "".join(chars), PerturbationType.TYPO_SWAP


def _typo_insert(name: str, rng: np.random.Generator) -> Tuple[str, str]:
    """Insert a random character."""
    if len(name) < 2:
        return name, ""
    pos = rng.integers(1, len(name))
    char = rng.choice(list(string.ascii_uppercase))
    return name[:pos] + char + name[pos:], PerturbationType.TYPO_INSERT


def _typo_delete(name: str, rng: np.random.Generator) -> Tuple[str, str]:
    """Delete a random character (not the first)."""
    if len(name) < 3:
        return name, ""
    pos = rng.integers(1, len(name))
    return name[:pos] + name[pos + 1:], PerturbationType.TYPO_DELETE


def _typo_replace(name: str, rng: np.random.Generator) -> Tuple[str, str]:
    """Replace a random character with a nearby key or random letter."""
    if len(name) < 2:
        return name, ""
    pos = rng.integers(0, len(name))
    char = rng.choice(list(string.ascii_uppercase))
    return name[:pos] + char + name[pos + 1:], PerturbationType.TYPO_REPLACE


def _hyphen_change(name: str, rng: np.random.Generator) -> Tuple[str, str]:
    """Toggle hyphens in compound names (e.g., SMITH-JONES ↔ SMITH JONES)."""
    if "-" in name:
        return name.replace("-", " "), PerturbationType.HYPHEN_CHANGE
    if " " in name and len(name.split()) == 2:
        return name.replace(" ", "-"), PerturbationType.HYPHEN_CHANGE
    return name, ""


# ── Address perturbation ────────────────────────────────────────────────────

def _swap_street_suffix(address: str, rng: np.random.Generator) -> Tuple[str, str]:
    """Replace a street suffix with an alternative form."""
    tokens = address.split()
    for i, tok in enumerate(tokens):
        upper_tok = tok.upper()
        for canonical, variants in SUFFIX_VARIANTS.items():
            if upper_tok == canonical or upper_tok in [v.upper() for v in variants]:
                # Pick a different variant
                alt = [v for v in variants if v.upper() != upper_tok]
                if alt:
                    tokens[i] = rng.choice(alt)
                    return " ".join(tokens), PerturbationType.SUFFIX_SWAP
    return address, ""


def _change_unit_format(address: str, rng: np.random.Generator) -> Tuple[str, str]:
    """Change unit designator format (APT 2B → #2B → UNIT 2B)."""
    for prefix in UNIT_PREFIXES:
        pattern = re.compile(rf"\b{re.escape(prefix)}\s*(\w+)", re.IGNORECASE)
        m = pattern.search(address)
        if m:
            unit_val = m.group(1)
            new_prefix = rng.choice([p for p in UNIT_PREFIXES if p.upper() != prefix.upper()])
            if new_prefix == "#":
                replacement = f"#{unit_val}"
            else:
                replacement = f"{new_prefix} {unit_val}"
            new_addr = address[:m.start()] + replacement + address[m.end():]
            return new_addr.strip(), PerturbationType.UNIT_FORMAT
    return address, ""


def _expand_directional(address: str, rng: np.random.Generator) -> Tuple[str, str]:
    """Toggle directional abbreviations (N ↔ NORTH, etc.)."""
    dir_map = {"N": "NORTH", "S": "SOUTH", "E": "EAST", "W": "WEST",
               "NE": "NORTHEAST", "NW": "NORTHWEST", "SE": "SOUTHEAST", "SW": "SOUTHWEST"}
    rev_map = {v: k for k, v in dir_map.items()}

    tokens = address.split()
    for i, tok in enumerate(tokens):
        upper_tok = tok.upper()
        if upper_tok in dir_map:
            tokens[i] = dir_map[upper_tok]
            return " ".join(tokens), PerturbationType.DIRECTIONAL_EXPAND
        if upper_tok in rev_map:
            tokens[i] = rev_map[upper_tok]
            return " ".join(tokens), PerturbationType.DIRECTIONAL_EXPAND
    return address, ""


def _whitespace_noise(address: str, rng: np.random.Generator) -> Tuple[str, str]:
    """Add/remove extra whitespace."""
    # Insert double space at a random position
    tokens = address.split()
    if len(tokens) > 1:
        pos = rng.integers(0, len(tokens) - 1)
        tokens[pos] = tokens[pos] + " "  # extra space
        return " ".join(tokens), PerturbationType.WHITESPACE_NOISE
    return address, ""


# ── Main augmentation engine ───────────────────────────────────────────────

def augment_record(
    record: Dict[str, str],
    rng: np.random.Generator,
    allowed_types: Optional[set] = None,
    n_perturbations: int = 2,
) -> Tuple[Dict[str, str], List[str]]:
    """
    Apply n_perturbations random perturbations to a record.

    Args:
        record: dict with keys {first_name, middle_name, last_name, name_suffix,
                street_address, city, zip5, birth_year}
        rng: numpy random generator
        allowed_types: set of PerturbationType values to use (None = all)
        n_perturbations: target number of perturbations to apply

    Returns:
        (perturbed_record, list_of_perturbation_types_applied)
    """
    rec = dict(record)  # copy
    applied = []

    # Build candidate perturbation functions
    candidates = []

    # Name perturbations
    if rec.get("first_name"):
        candidates.append(("first_name", "nickname", _nickname_sub))
        candidates.append(("first_name", "typo_swap", _typo_swap))
        candidates.append(("first_name", "typo_insert", _typo_insert))
        candidates.append(("first_name", "typo_delete", _typo_delete))
        candidates.append(("first_name", "typo_replace", _typo_replace))
        candidates.append(("first_name", "hyphen_change", _hyphen_change))

    if rec.get("last_name"):
        candidates.append(("last_name", "typo_swap", _typo_swap))
        candidates.append(("last_name", "typo_insert", _typo_insert))
        candidates.append(("last_name", "typo_delete", _typo_delete))
        candidates.append(("last_name", "typo_replace", _typo_replace))
        candidates.append(("last_name", "hyphen_change", _hyphen_change))

    if rec.get("middle_name"):
        candidates.append(("middle_name", "initial", lambda n, r: _to_initial(n)))
        candidates.append(("middle_name", "drop_middle",
                           lambda n, r: ("", PerturbationType.DROP_MIDDLE)))

    # Address perturbations
    if rec.get("street_address"):
        candidates.append(("street_address", "suffix_swap", _swap_street_suffix))
        candidates.append(("street_address", "unit_format", _change_unit_format))
        candidates.append(("street_address", "directional_expand", _expand_directional))
        candidates.append(("street_address", "whitespace_noise", _whitespace_noise))

    # Missingness perturbations
    if rec.get("middle_name"):
        candidates.append(("middle_name", "blank_middle",
                           lambda n, r: ("", PerturbationType.BLANK_MIDDLE)))
    if rec.get("name_suffix"):
        candidates.append(("name_suffix", "blank_suffix",
                           lambda n, r: ("", PerturbationType.BLANK_SUFFIX)))

    # Filter by allowed types
    if allowed_types is not None:
        candidates = [(f, t, fn) for f, t, fn in candidates if t in allowed_types]

    if not candidates:
        return rec, applied

    # Shuffle and try to apply n_perturbations
    rng.shuffle(candidates)
    for field, ptype, fn in candidates:
        if len(applied) >= n_perturbations:
            break
        new_val, actual_type = fn(rec[field], rng)
        if actual_type and new_val != rec[field]:
            rec[field] = new_val
            applied.append(actual_type)

    return rec, applied


def augment_dataframe(
    df: pd.DataFrame,
    mode: str = "train",
    frac: float = 0.3,
    n_variants_per_record: int = 2,
    n_perturbations_per_variant: int = 2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate synthetic augmented records from a normalized DataFrame.

    Args:
        df: Normalized DataFrame with standard columns
        mode: "train" (use train perturbation types) or "test" (use all + held-out)
        frac: Fraction of records to augment
        n_variants_per_record: Number of variants to create per seed record
        n_perturbations_per_variant: Number of perturbations per variant
        random_state: For reproducibility

    Returns:
        (seed_df, synthetic_df) where synthetic_df has augmented records
    """
    rng = np.random.default_rng(random_state)

    if mode == "train":
        allowed = PerturbationType.TRAIN_TYPES
    elif mode == "test":
        # Test uses ALL types, including held-out ones
        allowed = PerturbationType.TRAIN_TYPES | PerturbationType.TEST_ONLY_TYPES
    elif mode == "test_only":
        # Stress test: ONLY held-out types
        allowed = PerturbationType.TEST_ONLY_TYPES
    else:
        allowed = None  # all

    seed_df = df.sample(frac=frac, random_state=random_state).copy()
    record_fields = ["first_name", "middle_name", "last_name", "name_suffix",
                     "street_address", "city", "zip5", "birth_year"]

    synthetic_rows = []
    for _, row in seed_df.iterrows():
        base_record = {f: str(row.get(f, "")) for f in record_fields}

        for _ in range(n_variants_per_record):
            perturbed, applied = augment_record(
                base_record, rng,
                allowed_types=allowed,
                n_perturbations=n_perturbations_per_variant,
            )
            new_row = row.to_dict()
            for f in record_fields:
                new_row[f] = perturbed[f]
            new_row["is_synthetic"] = True
            new_row["perturbation_types"] = ",".join(applied)
            new_row["seed_id"] = row["id"]
            synthetic_rows.append(new_row)

    synthetic_df = pd.DataFrame(synthetic_rows)

    # Mark seed records
    seed_df["is_synthetic"] = False
    seed_df["perturbation_types"] = ""
    seed_df["seed_id"] = seed_df["id"]

    return seed_df, synthetic_df


def generate_augmented_pairs(
    seed_df: pd.DataFrame,
    synthetic_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Generate positive pairs from synthetic augmentation.
    Each synthetic record is paired with its seed (label=1).
    """
    pairs = []
    for _, syn_row in synthetic_df.iterrows():
        pairs.append({
            "id1": str(syn_row["seed_id"]),
            "id2": str(syn_row.get("id", f"syn_{syn_row.name}")),
            "label": 1,
            "source": "synthetic",
            "group_id": -1,
        })
    return pd.DataFrame(pairs)
