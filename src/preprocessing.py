"""
Preprocessing and normalization for NCVoters identity resolution.

Standardizes names, addresses, and ZIP codes to produce stable representations
for deterministic grouping and downstream matching features.
"""

import re
from typing import Optional

import pandas as pd


# ── Street suffix canonicalization ──────────────────────────────────────────
STREET_SUFFIX_MAP = {
    "STREET": "ST", "STREETS": "ST",
    "DRIVE": "DR", "DRIVES": "DR",
    "ROAD": "RD", "ROADS": "RD",
    "AVENUE": "AVE", "AVENUES": "AVE",
    "BOULEVARD": "BLVD", "BOULEVARDS": "BLVD",
    "LANE": "LN", "LANES": "LN",
    "COURT": "CT", "COURTS": "CT",
    "CIRCLE": "CIR", "CIRCLES": "CIR",
    "PLACE": "PL", "PLACES": "PL",
    "TERRACE": "TER", "TERRACES": "TER",
    "TRAIL": "TRL", "TRAILS": "TRL",
    "WAY": "WAY", "WAYS": "WAY",
    "PARKWAY": "PKWY", "PARKWAYS": "PKWY",
    "HIGHWAY": "HWY", "HIGHWAYS": "HWY",
    "EXPRESSWAY": "EXPY",
    "FREEWAY": "FWY",
    "PIKE": "PIKE",
    "SQUARE": "SQ",
    "LOOP": "LOOP",
    "CROSSING": "XING",
    "POINT": "PT",
    "RIDGE": "RDG",
}

# ── Directional canonicalization ────────────────────────────────────────────
DIRECTIONAL_MAP = {
    "NORTH": "N", "SOUTH": "S", "EAST": "E", "WEST": "W",
    "NORTHEAST": "NE", "NORTHWEST": "NW",
    "SOUTHEAST": "SE", "SOUTHWEST": "SW",
}

# ── Unit designator canonicalization ────────────────────────────────────────
UNIT_DESIGNATOR_MAP = {
    "APARTMENT": "APT", "SUITE": "STE", "UNIT": "UNIT",
    "BUILDING": "BLDG", "FLOOR": "FL", "ROOM": "RM",
    "NUMBER": "#", "NO": "#", "NUM": "#",
}

# ── Regex patterns ──────────────────────────────────────────────────────────
_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)
_MULTI_SPACE_RE = re.compile(r"\s+")


# ─── Name normalization ────────────────────────────────────────────────────
def normalize_name(value: Optional[str]) -> str:
    """Uppercase, strip punctuation, collapse whitespace."""
    if value is None or (isinstance(value, float)):
        return ""
    s = str(value).strip().upper()
    s = _PUNCT_RE.sub(" ", s)
    s = _MULTI_SPACE_RE.sub(" ", s).strip()
    return s


def normalize_suffix(value: Optional[str]) -> str:
    """Normalize name suffixes (JR, SR, II, III, IV, etc.)."""
    s = normalize_name(value)
    # Common suffix normalization
    suffix_map = {
        "JUNIOR": "JR", "SENIOR": "SR",
        "SECOND": "II", "THIRD": "III", "FOURTH": "IV",
    }
    return suffix_map.get(s, s)


# ─── Address normalization ─────────────────────────────────────────────────
def _replace_suffix(token: str) -> str:
    return STREET_SUFFIX_MAP.get(token, token)


def _replace_directional(token: str) -> str:
    return DIRECTIONAL_MAP.get(token, token)


def _replace_unit_designator(token: str) -> str:
    return UNIT_DESIGNATOR_MAP.get(token, token)


def normalize_street_address(
    house_num: Optional[str] = None,
    street_dir: Optional[str] = None,
    street_name: Optional[str] = None,
    street_type_cd: Optional[str] = None,
    street_sufx_cd: Optional[str] = None,
    unit_designator: Optional[str] = None,
    unit_num: Optional[str] = None,
) -> str:
    """
    Build and normalize a full street address from decomposed fields.
    Normalizes suffixes, directionals, and unit markers.
    """
    parts = []

    # House number
    hn = normalize_name(house_num)
    if hn:
        parts.append(hn)

    # Street direction (pre-directional)
    sd = normalize_name(street_dir)
    if sd:
        parts.append(_replace_directional(sd))

    # Street name — normalize each token for embedded suffixes/directionals
    sn = normalize_name(street_name)
    if sn:
        tokens = sn.split()
        tokens = [_replace_directional(t) for t in tokens]
        tokens = [_replace_suffix(t) for t in tokens]
        parts.extend(tokens)

    # Street type (STREET, DRIVE, etc.)
    st = normalize_name(street_type_cd)
    if st:
        parts.append(_replace_suffix(st))

    # Street suffix (post-directional)
    ss = normalize_name(street_sufx_cd)
    if ss:
        parts.append(_replace_directional(ss))

    # Unit
    ud = normalize_name(unit_designator)
    un = normalize_name(unit_num)
    if ud:
        parts.append(_replace_unit_designator(ud))
    if un:
        # Strip leading # if we already have a unit designator
        if un.startswith("#") and ud:
            un = un[1:].strip()
        parts.append(un)

    return " ".join(parts)


def normalize_zip5(value: Optional[str]) -> str:
    """Extract first 5 digits of ZIP code."""
    if value is None or (isinstance(value, float)):
        return ""
    s = str(value).strip()
    # Extract digits only
    digits = re.sub(r"[^\d]", "", s)
    return digits[:5] if len(digits) >= 5 else digits


def derive_birth_year(age: Optional[str], snapshot_year: int = 2018) -> Optional[int]:
    """Derive birth year from age and snapshot year."""
    if age is None or str(age).strip() == "":
        return None
    try:
        a = int(float(str(age).strip()))
        if 0 < a < 150:
            return snapshot_year - a
        return None
    except (ValueError, TypeError):
        return None


# ─── Full record normalization ─────────────────────────────────────────────
def normalize_dataframe(df: pd.DataFrame, snapshot_year: int = 2018) -> pd.DataFrame:
    """
    Normalize an NCVoters DataFrame.

    Expects the raw column names from ncvoters.tsv:
        first_name, midl_name, last_name, name_sufx_cd,
        house_num, street_dir, street_name, street_type_cd,
        street_sufx_cd, unit_designator, unit_num,
        res_city_desc, zip_code, age

    Returns a DataFrame with standardized columns:
        first_name, middle_name, last_name, name_suffix,
        street_address, city, zip5, birth_year, id
    """
    out = pd.DataFrame()

    # Preserve ID
    if "id" in df.columns:
        out["id"] = df["id"].astype(str).str.strip()
    elif "ncid" in df.columns:
        out["id"] = df["ncid"].astype(str).str.strip()

    # Names
    out["first_name"] = df["first_name"].apply(normalize_name)
    out["middle_name"] = df["midl_name"].apply(normalize_name) if "midl_name" in df.columns else ""
    out["last_name"] = df["last_name"].apply(normalize_name)
    out["name_suffix"] = (
        df["name_sufx_cd"].apply(normalize_suffix)
        if "name_sufx_cd" in df.columns else ""
    )

    # Street address (compose from decomposed fields)
    out["street_address"] = df.apply(
        lambda r: normalize_street_address(
            house_num=r.get("house_num"),
            street_dir=r.get("street_dir"),
            street_name=r.get("street_name"),
            street_type_cd=r.get("street_type_cd"),
            street_sufx_cd=r.get("street_sufx_cd"),
            unit_designator=r.get("unit_designator"),
            unit_num=r.get("unit_num"),
        ),
        axis=1,
    )

    # City
    out["city"] = df["res_city_desc"].apply(normalize_name) if "res_city_desc" in df.columns else ""

    # ZIP5
    out["zip5"] = df["zip_code"].apply(normalize_zip5) if "zip_code" in df.columns else ""

    # Birth year
    if "age" in df.columns:
        out["birth_year"] = df["age"].apply(lambda a: derive_birth_year(a, snapshot_year))
    elif "birth_year" in df.columns:
        out["birth_year"] = pd.to_numeric(df["birth_year"], errors="coerce").astype("Int64")
    else:
        out["birth_year"] = pd.NA

    return out
