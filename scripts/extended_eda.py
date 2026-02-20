import pandas as pd
import numpy as np
from pathlib import Path

def analyze_blocking_recall(pairs_file, data_df, label):
    print(f"\n--- Blocking Recall Analysis ({label}) ---")
    try:
        pairs_df = pd.read_csv(pairs_file, sep='\t', dtype=str)
    except FileNotFoundError:
        print(f"File {pairs_file} not found.")
        return

    # Merge pairs with data
    merged = pairs_df.merge(data_df, left_on='id1', right_on='id')
    merged = merged.merge(data_df, left_on='id2', right_on='id', suffixes=('_1', '_2'))
    
    total_pairs = len(merged)
    if total_pairs == 0:
        print("No pairs merged.")
        return

    # Check potential blocking keys
    blocking_keys = ['zip_code', 'last_name', 'first_name', 'street_name', 'age']
    
    for key in blocking_keys:
        k1 = f"{key}_1"
        k2 = f"{key}_2"
        if k1 in merged.columns and k2 in merged.columns:
            matches = (merged[k1] == merged[k2]).sum()
            recall = matches / total_pairs
            print(f"Blocking on '{key}' matches: {matches}/{total_pairs} ({recall:.2%})")

def analyze_attribute_changes(pairs_file, data_df, label):
    print(f"\n--- Attribute Change Analysis ({label}) ---")
    try:
        pairs_df = pd.read_csv(pairs_file, sep='\t', dtype=str)
    except FileNotFoundError:
        print(f"File {pairs_file} not found.")
        return

    merged = pairs_df.merge(data_df, left_on='id1', right_on='id')
    merged = merged.merge(data_df, left_on='id2', right_on='id', suffixes=('_1', '_2'))
    
    attributes = [col[:-2] for col in merged.columns if col.endswith('_1') and col != 'id_1']
    
    results = []
    for attr in attributes:
        k1 = f"{attr}_1"
        k2 = f"{attr}_2"
        # Count how often they are the same vs different
        both_filled = (merged[k1] != "") & (merged[k2] != "")
        matches = (merged[k1] == merged[k2]) & both_filled
        differ = (merged[k1] != merged[k2]) & both_filled
        one_empty = (merged[k1] == "") ^ (merged[k2] == "")
        both_empty = (merged[k1] == "") & (merged[k2] == "")
        
        results.append({
            'attribute': attr,
            'match_pct': matches.sum() / len(merged),
            'diff_pct': differ.sum() / len(merged),
            'one_empty_pct': one_empty.sum() / len(merged),
            'both_empty_pct': both_empty.sum() / len(merged)
        })
    
    df_results = pd.DataFrame(results).sort_values('match_pct', ascending=False)
    print(df_results.to_string(index=False))

def analyze_string_lengths(data_df):
    print("\n--- String Length Analysis for Transformer ---")
    # Simulate serialization
    def serialize(row):
        return " ".join([f"{k}: {v}" for k, v in row.items() if v != ""])
    
    lengths = data_df.apply(lambda r: len(serialize(r)), axis=1)
    print(f"Mean serialized length: {lengths.mean():.1f}")
    print(f"95th percentile length: {lengths.quantile(0.95):.1f}")
    print(f"Max length: {lengths.max()}")

def main():
    try:
        data_df = pd.read_csv('data/processed/ncvoters_prepared.tsv', sep='\t', dtype=str).fillna("")
    except FileNotFoundError:
        print("Error: data/processed/ncvoters_prepared.tsv not found.")
        return
    
    # Blocking recall for DPL
    analyze_blocking_recall('data/raw/ncvoters_DPL.tsv', data_df, "DPL")
    
    # Attribute changes
    analyze_attribute_changes('data/raw/ncvoters_DPL.tsv', data_df, "DPL")
    analyze_attribute_changes('data/raw/ncvoters_NDPL.tsv', data_df, "NDPL")
    
    # String lengths
    analyze_string_lengths(data_df)

if __name__ == "__main__":
    main()
