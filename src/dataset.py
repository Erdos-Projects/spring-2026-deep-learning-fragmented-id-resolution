import pandas as pd
import torch
from torch.utils.data import Dataset
import numpy as np
from collections import Counter

class Vocabulary:
    def __init__(self, specials=["<PAD>", "<UNK>", "<SOS>", "<EOS>"]):
        self.specials = specials
        self.stoi = {s: i for i, s in enumerate(specials)}
        self.itos = {i: s for i, s in enumerate(specials)}
        self.vocab_size = len(specials)

    def build_vocabulary(self, sentence_list):
        frequencies = Counter()
        for sentence in sentence_list:
            frequencies.update(str(sentence))
        
        # Add characters to vocab
        for char, count in frequencies.items():
            if char not in self.stoi:
                self.stoi[char] = self.vocab_size
                self.itos[self.vocab_size] = char
                self.vocab_size += 1

    def numericalize(self, text, max_len=None):
        tokenized = [self.stoi.get(c, self.stoi["<UNK>"]) for c in str(text)]
        
        if max_len:
            if len(tokenized) > max_len:
                tokenized = tokenized[:max_len]
            else:
                tokenized = tokenized + [self.stoi["<PAD>"]] * (max_len - len(tokenized))
        
        return tokenized

class NCVotersDataset(Dataset):
    def __init__(self, data_path, dpl_path, ndpl_path, vocab=None, max_len=50, attributes=['first_name', 'last_name', 'house_num', 'street_name', 'zip_code']):
        """
        Args:
            data_path (str): Path to ncvoters_prepared.tsv
            dpl_path (str): Path to ncvoters_DPL.tsv (positives)
            ndpl_path (str): Path to ncvoters_NDPL.tsv (negatives)
            vocab (Vocabulary, optional): Pre-built vocabulary. If None, builds from data.
            max_len (int): Maximum sequence length for string attributes.
            attributes (list): List of column names to concatenate as the input feature.
        """
        self.data_df = pd.read_csv(data_path, sep='	', dtype=str).fillna("")
        # Ensure ID is index for fast lookup
        self.data_df['id'] = self.data_df['id'].astype(str)
        self.data_dict = self.data_df.set_index('id').to_dict('index')

        self.attributes = attributes
        self.max_len = max_len

        # Load Pairs
        dpl = pd.read_csv(dpl_path, sep='	', dtype=str)
        dpl['label'] = 1.0
        
        ndpl = pd.read_csv(ndpl_path, sep='	', dtype=str)
        ndpl['label'] = 0.0

        # Combine and shuffle
        self.pairs = pd.concat([dpl[['id1', 'id2', 'label']], ndpl[['id1', 'id2', 'label']]], ignore_index=True)
        self.pairs = self.pairs.sample(frac=1, random_state=42).reset_index(drop=True)

        # Build Vocab if not provided
        if vocab is None:
            self.vocab = Vocabulary()
            print("Building vocabulary from all selected attributes...")
            all_text = []
            for attr in attributes:
                all_text.extend(self.data_df[attr].tolist())
            self.vocab.build_vocabulary(all_text)
            print(f"Vocabulary built with {self.vocab.vocab_size} characters.")
        else:
            self.vocab = vocab

    def _get_record_string(self, record_id):
        if record_id not in self.data_dict:
            # Handle missing IDs gracefully (though ideally shouldn't happen)
            return ""
        
        record = self.data_dict[record_id]
        # Concatenate selected attributes with a space separator
        return " ".join([str(record.get(attr, "")) for attr in self.attributes])

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        row = self.pairs.iloc[idx]
        id1, id2, label = row['id1'], row['id2'], row['label']

        s1 = self._get_record_string(id1)
        s2 = self._get_record_string(id2)

        x1 = self.vocab.numericalize(s1, self.max_len)
        x2 = self.vocab.numericalize(s2, self.max_len)

        return torch.tensor(x1, dtype=torch.long), torch.tensor(x2, dtype=torch.long), torch.tensor(label, dtype=torch.float)

if __name__ == "__main__":
    # Simple test to verify the dataset works
    dataset = NCVotersDataset(
        data_path='data/processed/ncvoters_prepared.tsv',
        dpl_path='data/raw/ncvoters_DPL.tsv',
        ndpl_path='data/raw/ncvoters_NDPL.tsv',
        max_len=60
    )

    print(f"\nDataset size: {len(dataset)}")
    x1, x2, label = dataset[0]
    print(f"Sample 0:")
    print(f"  Input 1 shape: {x1.shape}")
    print(f"  Input 1 (first 10): {x1[:10]}")
    print(f"  Label: {label}")
