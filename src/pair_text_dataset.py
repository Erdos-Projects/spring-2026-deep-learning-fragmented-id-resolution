"""
Dataset class for processing voter text pairs on-the-fly.
"""

import pandas as pd
import torch
from torch.utils.data import Dataset


class PairTextDataset(Dataset):
    """
    Dataset that processes voter text on-the-fly from the pairs CSV.
    Each row has person1's info and person2's info.
    """
    def __init__(self, data, char_to_idx, max_len=100, 
                 features=['first_name', 'midl_name', 'last_name', 'age', 'sex']):
        """
        Args:
            pairs_csv_path: CSV with columns like first_name_1, ..., first_name_2, ..., label
            char_to_idx: Character to index mapping
            max_len: Maximum sequence length
            features: Which features to use (they'll have _1 and _2 suffixes)
        """
        self.pairs = data
        self.char_to_idx = char_to_idx
        self.max_len = max_len
        self.features = features
        
        # Build feature column names for person 1 and person 2
        self.person1_cols = [f"{feat}_1" for feat in features]
        self.person2_cols = [f"{feat}_2" for feat in features]
        
        print(f"\nLoaded {len(self.pairs)} pairs from the data")
        print(f"  - Duplicates: {(self.pairs['label'] == 1).sum()}")
        print(f"  - Non-duplicates: {(self.pairs['label'] == 0).sum()}")
        print(f"  - Using features: {features}")
    
    def format_person(self, row, person_cols):
        """Concatenate person's attributes into a single string."""
        parts = []
        for col in person_cols:
            value = row[col]
            if pd.notna(value):
                parts.append(str(value))
        return ' '.join(parts).lower().strip()
    
    def tokenize_and_pad(self, text):
        """Convert text to padded sequence of character indices."""
        sequence = [self.char_to_idx.get(char, self.char_to_idx.get('<UNK>', 0)) 
                   for char in text if char in self.char_to_idx]
        sequence = sequence[:self.max_len]
        padding = [0] * (self.max_len - len(sequence))
        sequence.extend(padding)
        return sequence
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        row = self.pairs.iloc[idx]
        
        # Format both persons' text
        text1 = self.format_person(row, self.person1_cols)
        text2 = self.format_person(row, self.person2_cols)
        
        # Tokenize
        seq1 = self.tokenize_and_pad(text1)
        seq2 = self.tokenize_and_pad(text2)
        
        label = row['label']
        
        return (torch.tensor(seq1, dtype=torch.long),
                torch.tensor(seq2, dtype=torch.long),
                torch.tensor(label, dtype=torch.float32))
