import argparse
import json
from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = {"scenario", "attribute_set", "method", "test_f1", "test_pr_auc"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a human-readable summary from comparison_summary.json/tsv."
    )
    parser.add_argument(
        "--input-path",
        default="models/comparisons/apples_to_apples/comparison_summary.json",
        help="Path to comparison summary file (.json or .tsv).",
    )
    parser.add_argument(
        "--primary-metric",
        default="test_f1",
        help="Primary metric for ranking winners (default: test_f1).",
    )
    parser.add_argument(
        "--secondary-metric",
        default="test_pr_auc",
        help="Secondary metric for tie-breaking (default: test_pr_auc).",
    )
    parser.add_argument(
        "--output-report",
        default=None,
        help="Optional path to save the text report. If omitted, report is only printed.",
    )
    parser.add_argument(
        "--output-winners-tsv",
        default=None,
        help="Optional path to save winner rows (one per scenario/attribute set) as TSV.",
    )
    return parser.parse_args()


def load_summary(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input file does not exist: {path}")

    if path.suffix.lower() == ".json":
        with open(path, "r", encoding="utf-8") as f:
            rows = json.load(f)
        df = pd.DataFrame(rows)
    elif path.suffix.lower() == ".tsv":
        df = pd.read_csv(path, sep="\t")
    else:
        raise ValueError("Unsupported input extension. Use .json or .tsv.")

    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(f"Input summary is missing required columns: {sorted(missing)}")
    return df


def pick_winners(df: pd.DataFrame, primary_metric: str, secondary_metric: str) -> pd.DataFrame:
    ranked = df.sort_values(
        ["scenario", "attribute_set", primary_metric, secondary_metric],
        ascending=[True, True, False, False],
    )
    winners = ranked.groupby(["scenario", "attribute_set"], as_index=False).head(1)
    return winners.reset_index(drop=True)


def fmt(value):
    if pd.isna(value):
        return "n/a"
    if isinstance(value, (int, float)):
        return f"{value:.4f}"
    return str(value)


def build_report(df: pd.DataFrame, winners: pd.DataFrame, primary_metric: str, secondary_metric: str) -> str:
    methods = sorted(df["method"].astype(str).unique().tolist())
    scenarios = sorted(df["scenario"].astype(str).unique().tolist())
    attrs = sorted(df["attribute_set"].astype(str).unique().tolist())

    overall_best = df.sort_values([primary_metric, secondary_metric], ascending=[False, False]).iloc[0]
    avg_by_method = (
        df.groupby("method", as_index=False)[[primary_metric, secondary_metric]]
        .mean()
        .sort_values([primary_metric, secondary_metric], ascending=[False, False])
    )

    lines = []
    lines.append("Comparison Summary Report")
    lines.append("")
    lines.append(f"- Runs analyzed: {len(df)}")
    lines.append(f"- Methods: {', '.join(methods)}")
    lines.append(f"- Scenarios: {', '.join(scenarios)}")
    lines.append(f"- Attribute sets: {', '.join(attrs)}")
    lines.append("")
    lines.append("Overall best run:")
    lines.append(
        f"- {overall_best['method']} | {overall_best['scenario']} | {overall_best['attribute_set']} | "
        f"{primary_metric}={fmt(overall_best[primary_metric])}, {secondary_metric}={fmt(overall_best[secondary_metric])}"
    )
    lines.append("")
    lines.append("Winners by scenario and attribute set:")
    for _, row in winners.iterrows():
        lines.append(
            f"- {row['scenario']} / {row['attribute_set']}: {row['method']} "
            f"({primary_metric}={fmt(row[primary_metric])}, {secondary_metric}={fmt(row[secondary_metric])})"
        )

    lines.append("")
    lines.append(f"Average by method ({primary_metric}, {secondary_metric}):")
    for _, row in avg_by_method.iterrows():
        lines.append(
            f"- {row['method']}: {primary_metric}={fmt(row[primary_metric])}, "
            f"{secondary_metric}={fmt(row[secondary_metric])}"
        )

    if "tfidf" in methods:
        lines.append("")
        lines.append("Delta vs TF-IDF by scenario/attribute set:")
        for scenario in scenarios:
            for attr in attrs:
                subset = df[(df["scenario"] == scenario) & (df["attribute_set"] == attr)]
                tfidf_rows = subset[subset["method"] == "tfidf"]
                if tfidf_rows.empty:
                    continue
                tfidf_primary = float(tfidf_rows.iloc[0][primary_metric])
                tfidf_secondary = float(tfidf_rows.iloc[0][secondary_metric])
                for _, row in subset.iterrows():
                    if row["method"] == "tfidf":
                        continue
                    delta_primary = float(row[primary_metric]) - tfidf_primary
                    delta_secondary = float(row[secondary_metric]) - tfidf_secondary
                    sign_p = "+" if delta_primary >= 0 else ""
                    sign_s = "+" if delta_secondary >= 0 else ""
                    lines.append(
                        f"- {scenario} / {attr} | {row['method']} vs tfidf: "
                        f"{primary_metric} {sign_p}{delta_primary:.4f}, "
                        f"{secondary_metric} {sign_s}{delta_secondary:.4f}"
                    )

    return "\n".join(lines) + "\n"


def main():
    args = parse_args()
    input_path = Path(args.input_path)
    df = load_summary(input_path)

    for metric in [args.primary_metric, args.secondary_metric]:
        if metric not in df.columns:
            raise ValueError(f"Metric '{metric}' is not present in input columns.")

    winners = pick_winners(df, args.primary_metric, args.secondary_metric)
    report = build_report(df, winners, args.primary_metric, args.secondary_metric)
    print(report, end="")

    if args.output_report:
        output_report = Path(args.output_report)
        output_report.parent.mkdir(parents=True, exist_ok=True)
        output_report.write_text(report, encoding="utf-8")
        print(f"Saved report: {output_report}")

    if args.output_winners_tsv:
        output_tsv = Path(args.output_winners_tsv)
        output_tsv.parent.mkdir(parents=True, exist_ok=True)
        winners.to_csv(output_tsv, sep="\t", index=False)
        print(f"Saved winners TSV: {output_tsv}")


if __name__ == "__main__":
    main()
