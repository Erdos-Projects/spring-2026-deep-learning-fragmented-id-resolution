import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd
from tqdm import tqdm


PATTERN = re.compile(r"[\W_]+", re.UNICODE)

# Default NCVoters attribute subset used in this project.
NCVOTERS_ATTRIBUTES = [
    "age", "area_cd", "birth_place", "cancellation_dt", "county_desc", "county_id", "ethnic_desc", "first_name",
    "house_num", "last_name", "midl_name", "party_desc", "phone_num", "race_desc", "reason_cd", "registr_dt",
    "res_city_desc", "sex", "state_cd", "status_cd", "street_name", "street_type_cd", "voter_status_desc",
    "voter_status_reason_desc", "zip_code"
]


def remove_special_characters(value: str) -> str:
    cleaned = PATTERN.sub(" ", value).strip()
    cleaned = " ".join(cleaned.split(" "))
    return cleaned


def normalize_value(value: str) -> str:
    return remove_special_characters(str(value)).lower()


def preprocess_file(input_path: Path, output_path: Path, keep_only_config_columns: bool) -> list[str]:
    df_data = pd.read_csv(
        input_path,
        sep="\t",
        quotechar="'",
        index_col=None,
        na_filter=False,
        dtype=str,
        encoding="utf-8"
    ).fillna("")

    df_data = df_data.loc[:, ~df_data.columns.str.contains(r"^Unnamed")]

    if keep_only_config_columns:
        keep = [column for column in NCVOTERS_ATTRIBUTES if column in df_data.columns]
        if "id" in df_data.columns and "id" not in keep:
            keep = ["id"] + keep
        if not keep:
            raise ValueError(
                "None of the expected NCVoters columns were found in the input file.")
        df_data = df_data[keep]

    data = df_data.to_dict(orient="index")
    header = sorted(list(set(df_data.columns.values)))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="\n", encoding="utf-8") as fout:
        dictout = csv.DictWriter(fout, fieldnames=header, delimiter="\t")
        dictout.writeheader()
        for _, row in tqdm(data.items(), desc="Preprocessing records", total=len(data)):
            for attr in df_data.columns.values:
                row[attr] = normalize_value(row[attr])
            dictout.writerow(row)

    return header


def calculate_metrics_for_attribute(data_file: Path, attribute: str, record_count: int) -> dict:
    frequencies = defaultdict(int)
    counter_valid = 0
    counter_invalid = 0

    with open(data_file, "rt", encoding="utf-8") as fin:
        csvin = csv.DictReader(fin, delimiter="\t", quoting=csv.QUOTE_NONE)
        for row in tqdm(csvin, desc=f"Attribute: {attribute}", total=record_count):
            value = row[attribute].strip()
            frequencies[value] += 1
            if value != "":
                counter_valid += 1
            else:
                counter_invalid += 1

    if counter_valid > 0:
        uniqueness = float(len(frequencies)) / counter_valid
        completeness = float(counter_valid) / (counter_valid + counter_invalid)
        counter_total = counter_valid + counter_invalid
    else:
        uniqueness = 0.0
        completeness = 0.0
        counter_total = 0

    return {
        "uniqueness": uniqueness,
        "completeness": completeness,
        "counter_total": counter_total,
        "counter_valid": counter_valid,
        "counter_invalid": counter_invalid,
    }


def run_attribute_eda(prepared_file: Path, stats_output_file: Path) -> None:
    with open(prepared_file, "rt", encoding="utf-8") as fin:
        csvin = csv.DictReader(fin, delimiter="\t", quoting=csv.QUOTE_NONE)
        attributes = csvin.fieldnames

    record_count = 0
    with open(prepared_file, "rt", encoding="utf-8") as fin:
        for _ in fin:
            record_count += 1

    metrics_per_attribute = {}
    for attribute in attributes:
        metrics_per_attribute[attribute] = calculate_metrics_for_attribute(
            prepared_file, attribute, record_count)

    rows = []
    metric_columns = ["uniqueness", "completeness",
                      "counter_total", "counter_valid", "counter_invalid"]
    for attribute in metrics_per_attribute:
        row = [attribute]
        for metric in metric_columns:
            row.append(metrics_per_attribute[attribute][metric])
        rows.append(tuple(row))

    df = pd.DataFrame.from_records(
        rows, columns=["attribute"] + metric_columns)
    stats_output_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(stats_output_file, sep="\t", index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess NCVoters TSV and generate attribute-level EDA statistics."
    )
    parser.add_argument("--input", default="data/raw/ncvoters.tsv", help="Path to ncvoters.tsv")
    parser.add_argument(
        "--prepared-output",
        default="data/processed/ncvoters_prepared.tsv",
        help="Path to write cleaned/preprocessed TSV"
    )
    parser.add_argument(
        "--stats-output",
        default="data/processed/ncvoters_attribute_stats.tsv",
        help="Path to write attribute EDA metrics TSV"
    )
    parser.add_argument(
        "--keep-only-config-columns",
        action="store_true",
        help="Keep only the NCVoters attribute subset used in the original project config"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    prepared_output = Path(args.prepared_output)
    stats_output = Path(args.stats_output)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file does not exist: {input_path}")

    preprocess_file(
        input_path=input_path,
        output_path=prepared_output,
        keep_only_config_columns=args.keep_only_config_columns,
    )
    run_attribute_eda(prepared_file=prepared_output,
                      stats_output_file=stats_output)

    print(f"Prepared file: {prepared_output}")
    print(f"EDA stats file: {stats_output}")
    print("DPL/NDPL files are only needed for supervised pair-based duplicate detection.")


if __name__ == "__main__":
    main()
