"""
PyTorch Dataset and DataLoader for record pair classification.

Each sample is a pair of records, represented as concatenated field strings,
with a binary label (1 = same person, 0 = different).
"""

from typing import Dict, List, Optional, Tuple

import pandas as pd
import torch
from torch.utils.data import Dataset


class RecordPairDataset(Dataset):
    """
    Dataset for Siamese / pairwise record matching.

    Each item returns:
        - text1: concatenated identity string for record 1
        - text2: concatenated identity string for record 2
        - label: 1 (match) or 0 (non-match)
    """

    def __init__(
        self,
        pairs_df: pd.DataFrame,
        records_df: pd.DataFrame,
        fields: Optional[List[str]] = None,
        char_vocab: Optional[Dict[str, int]] = None,
        max_len: int = 128,
    ):
        """
        Args:
            pairs_df: DataFrame with columns [id1, id2, label]
            records_df: DataFrame of normalized records, indexed or with 'id' column
            fields: identity fields to concatenate (default: name + address fields)
            char_vocab: character-to-index mapping; if None, built from records
            max_len: maximum character sequence length
        """
        self.pairs = pairs_df.reset_index(drop=True)
        self.max_len = max_len

        if fields is None:
            fields = [
                "first_name", "middle_name", "last_name", "name_suffix",
                "street_address", "city", "zip5",
            ]
        self.fields = fields

        # Build record lookup
        if "id" in records_df.columns:
            self.records = records_df.set_index("id")
        else:
            self.records = records_df

        # Build character vocabulary
        if char_vocab is None:
            self.char_vocab = self._build_vocab()
        else:
            self.char_vocab = char_vocab

    def _build_vocab(self) -> Dict[str, int]:
        """Build character vocabulary from all record fields."""
        chars = set()
        for field in self.fields:
            if field in self.records.columns:
                for val in self.records[field].astype(str):
                    chars.update(val)
        # Special tokens
        vocab = {"<PAD>": 0, "<UNK>": 1, "<SEP>": 2}
        for i, c in enumerate(sorted(chars), start=3):
            vocab[c] = i
        return vocab

    def _record_to_text(self, record_id: str) -> str:
        """Concatenate identity fields into a single string."""
        if record_id not in self.records.index:
            return ""
        row = self.records.loc[record_id]
        parts = []
        for f in self.fields:
            val = str(row.get(f, "")).strip()
            if val:
                parts.append(val)
        return " | ".join(parts)

    def _encode(self, text: str) -> torch.LongTensor:
        """Encode a text string as a padded character index tensor."""
        indices = []
        for ch in text[:self.max_len]:
            indices.append(self.char_vocab.get(ch, self.char_vocab["<UNK>"]))
        # Pad
        while len(indices) < self.max_len:
            indices.append(self.char_vocab["<PAD>"])
        return torch.LongTensor(indices)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.pairs.iloc[idx]
        text1 = self._record_to_text(str(row["id1"]))
        text2 = self._record_to_text(str(row["id2"]))

        return {
            "input1": self._encode(text1),
            "input2": self._encode(text2),
            "label": torch.FloatTensor([float(row["label"])]),
        }

    @property
    def vocab_size(self) -> int:
        return len(self.char_vocab)
