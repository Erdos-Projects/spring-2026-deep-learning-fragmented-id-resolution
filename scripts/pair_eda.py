import pandas as pd
import numpy as np
from collections import Counter

def levenshtein(s1, s2):
    if len(s1) < len(s2):
        return levenshtein(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]

def analyze_pairs(pairs_file, data_df, label):
    print(f"\n--- Analyzing {label} Pairs ---")
    
    # Read pairs
    try:
        pairs_df = pd.read_csv(pairs_file, sep='\t')
    except FileNotFoundError:
        print(f"File {pairs_file} not found.")
        return set()

    print(f"Total pairs: {len(pairs_df)}")
    
    # Check for ID overlap
    ids_in_pairs = set(pairs_df['id1']).union(set(pairs_df['id2']))
    print(f"Unique IDs involved: {len(ids_in_pairs)}")
    
    # Join with data
    # We need to ensure IDs are treated as strings to match
    pairs_df['id1'] = pairs_df['id1'].astype(str)
    pairs_df['id2'] = pairs_df['id2'].astype(str)
    data_df['id'] = data_df['id'].astype(str)
    
    # Merge for id1
    merged = pairs_df.merge(data_df, left_on='id1', right_on='id', suffixes=('', '_dummy'))
    # Rename columns from id1 merge to have _1 suffix
    rename_dict_1 = {col: col + '_1' for col in data_df.columns if col != 'id'}
    merged = merged.rename(columns=rename_dict_1)
    
    # Merge for id2
    merged = merged.merge(data_df, left_on='id2', right_on='id', suffixes=('', '_dummy'))
    # Rename columns from id2 merge to have _2 suffix
    rename_dict_2 = {col: col + '_2' for col in data_df.columns if col != 'id'}
    merged = merged.rename(columns=rename_dict_2)
    
    # Calculate Levenshtein distances for First and Last Name
    if 'first_name_1' in merged.columns and 'first_name_2' in merged.columns:
        merged['first_name_dist'] = merged.apply(lambda row: levenshtein(str(row['first_name_1']), str(row['first_name_2'])), axis=1)
        print(f"Mean First Name Distance: {merged['first_name_dist'].mean():.2f}")
    
    if 'last_name_1' in merged.columns and 'last_name_2' in merged.columns:
        merged['last_name_dist'] = merged.apply(lambda row: levenshtein(str(row['last_name_1']), str(row['last_name_2'])), axis=1)
        print(f"Mean Last Name Distance: {merged['last_name_dist'].mean():.2f}")
    
    # Show examples
    print("\nSample Pairs:")
    if len(merged) > 0:
        sample = merged.sample(min(5, len(merged)))
        for _, row in sample.iterrows():
            print(f"Pair: {row['id1']} - {row['id2']}")
            print(f"  Name: {row.get('first_name_1', '')} {row.get('last_name_1', '')}  vs  {row.get('first_name_2', '')} {row.get('last_name_2', '')}")
            print(f"  Addr: {row.get('house_num_1', '')} {row.get('street_name_1', '')}  vs  {row.get('house_num_2', '')} {row.get('street_name_2', '')}")
            print(f"  Age : {row.get('age_1', '')} vs {row.get('age_2', '')}")
            print(f"  Sex : {row.get('sex_1', '')} vs {row.get('sex_2', '')}")
            print(f"  Race: {row.get('race_desc_1', '')} vs {row.get('race_desc_2', '')}")
            print(f"  Zip : {row.get('zip_code_1', '')} vs {row.get('zip_code_2', '')}")
            print("-" * 20)
    else:
        print("No pairs could be merged with data (check ID formats).")

    return ids_in_pairs

def main():
    print("Loading data...")
    # Load prepared data
    try:
        # Use dtype=str to preserve leading zeros in IDs if any
        data_df = pd.read_csv('data/processed/ncvoters_prepared.tsv', sep='\t', dtype=str).fillna("")
    except FileNotFoundError:
        print("Error: data/processed/ncvoters_prepared.tsv not found. Please run ncvoters_preprocess_and_eda.py first.")
        return

    # Analyze DPL
    dpl_ids = analyze_pairs('data/raw/ncvoters_DPL.tsv', data_df, "Duplicate (DPL)")
    
    # Analyze NDPL
    ndpl_ids = analyze_pairs('data/raw/ncvoters_NDPL.tsv', data_df, "Non-Duplicate (NDPL)")
    
    # Check overlap
    if dpl_ids and ndpl_ids:
        overlap = dpl_ids.intersection(ndpl_ids)
        print(f"\nIDs appearing in both DPL and NDPL: {len(overlap)}")

if __name__ == "__main__":
    main()