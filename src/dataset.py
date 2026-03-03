"""
PyTorch Dataset and DataLoader for record pair classification.

Each sample is a pair of records, represented as concatenated field strings,
with a binary label (1 = same person, 0 = different).

Also provides utility methods for encoding raw records at inference time
(for duplicate detection on unseen data).
"""

from typing import Dict, List, Optional, Tuple

import pandas as pd
import torch
from torch.utils.data import Dataset


# Default identity fields used across the pipeline
DEFAULT_FIELDS = [
    "first_name", "middle_name", "last_name", "name_suffix",
    "street_address", "city", "zip5", "birth_year",
]


class RecordPairDataset(Dataset):
    """
    Dataset for Siamese / pairwise record matching.

    Each item returns:
        - input1: character-encoded identity string for record 1
        - input2: character-encoded identity string for record 2
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
            fields = list(DEFAULT_FIELDS)
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
        """
        Build character vocabulary from all record fields.
        
        Includes both uppercase and lowercase characters plus common
        punctuation to handle real-world data that may not be pre-normalized.
        """
        chars = set()
        for field in self.fields:
            if field in self.records.columns:
                for val in self.records[field].astype(str):
                    chars.update(val.upper())  # normalize to uppercase for vocab
                    chars.update(val)          # also include original chars
        # Add common chars that might appear in unseen data
        import string
        chars.update(string.ascii_uppercase)
        chars.update(string.ascii_lowercase)
        chars.update(string.digits)
        chars.update(" |-./,#&'")

        # Special tokens
        vocab = {"<PAD>": 0, "<UNK>": 1, "<SEP>": 2}
        for i, c in enumerate(sorted(chars), start=3):
            if c not in vocab:
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
                parts.append(val.upper())  # normalize to uppercase
        return " | ".join(parts)

    @staticmethod
    def record_dict_to_text(
        record: Dict[str, str],
        fields: Optional[List[str]] = None,
    ) -> str:
        """
        Convert a raw record dict to a text string for encoding.
        Useful for inference on unseen data.
        """
        if fields is None:
            fields = list(DEFAULT_FIELDS)
        parts = []
        for f in fields:
            val = str(record.get(f, "")).strip()
            if val:
                parts.append(val.upper())
        return " | ".join(parts)

    def _encode(self, text: str) -> torch.LongTensor:
        """Encode a text string as a padded character index tensor."""
        # Uppercase for consistent encoding
        text = text.upper()
        indices = []
        for ch in text[:self.max_len]:
            indices.append(self.char_vocab.get(ch, self.char_vocab["<UNK>"]))
        # Pad
        while len(indices) < self.max_len:
            indices.append(self.char_vocab["<PAD>"])
        return torch.LongTensor(indices)

    def encode_text(self, text: str) -> torch.LongTensor:
        """Public encode method for inference use."""
        return self._encode(text)

    def encode_record(self, record: Dict[str, str]) -> torch.LongTensor:
        """Encode a raw record dict (for inference on unseen data)."""
        text = self.record_dict_to_text(record, self.fields)
        return self._encode(text)

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
