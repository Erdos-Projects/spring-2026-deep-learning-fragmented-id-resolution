# Import necessary libraries
import pandas as pd
import sys

try:
    # Load the dataset
    df = pd.read_csv('ncvoters.tsv', sep='\t', low_memory=False)

    # Get the two rows to compare
    row1 = df.loc[df['id'] == 511079].iloc[0]
    row2 = df.loc[df['id'] == 11401396].iloc[0]

    print(f"Differences between ID {row1['id']} and ID {row2['id']}:")
    for col in df.columns:
        val1 = row1[col]
        val2 = row2[col]
        # Check for inequality, treating NaNs as equal
        if val1 != val2 and not (pd.isna(val1) and pd.isna(val2)):
            print(f"Difference in '{col}': '{val1}' vs '{val2}'")
except FileNotFoundError:
    print("Error: The data file 'ncvoters.tsv' was not found or is not readable. "
          "Please ensure the file exists and is accessible, then try again.")
    sys.exit(1)
row2_df = df.loc[df['id'] == 11401396]
if row2_df.empty:
    raise ValueError("ID 11401396 not found in the dataset.")
row2 = row2_df.iloc[0]
print(f"Differences between ID {row1['id']} and ID {row2['id']}:")
for col in df.columns:
    val1 = row1[col]
    val2 = row2[col]
    # Check for inequality, treating NaNs as equal
    if val1 != val2 and not (pd.isna(val1) and pd.isna(val2)):
        print(f"Difference in '{col}': '{val1}' vs '{val2}'")
