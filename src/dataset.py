from collections import Counter

import pandas as pd
import torch
from torch.utils.data import Dataset

try:
    from .data_utils import DEFAULT_BASELINE_ATTRIBUTES, load_labeled_pairs, load_prepared_data
except ImportError:
    from data_utils import DEFAULT_BASELINE_ATTRIBUTES, load_labeled_pairs, load_prepared_data


class Vocabulary:
    def __init__(self, specials=None):
        if specials is None:
            specials = ["<PAD>", "<UNK>", "<SOS>", "<EOS>"]
        self.specials = list(specials)
        self.stoi = {s: i for i, s in enumerate(self.specials)}
        self.itos = {i: s for i, s in enumerate(self.specials)}
        self.vocab_size = len(self.specials)

    def build_vocabulary(self, sentence_list):
        frequencies = Counter()
        for sentence in sentence_list:
            frequencies.update(str(sentence))

        for char in frequencies:
            if char not in self.stoi:
                self.stoi[char] = self.vocab_size
                self.itos[self.vocab_size] = char
                self.vocab_size += 1

    def numericalize(self, text, max_len=None):
        tokenized = [self.stoi.get(c, self.stoi["<UNK>"]) for c in str(text)]
        if max_len is not None:
            if len(tokenized) > max_len:
                tokenized = tokenized[:max_len]
            else:
                tokenized = tokenized + [self.stoi["<PAD>"]] * (max_len - len(tokenized))
        return tokenized

    def to_dict(self):
        return {"specials": self.specials, "stoi": self.stoi}

    @classmethod
    def from_dict(cls, payload):
        vocab = cls(specials=payload["specials"])
        vocab.stoi = {k: int(v) for k, v in payload["stoi"].items()}
        vocab.itos = {v: k for k, v in vocab.stoi.items()}
        vocab.vocab_size = len(vocab.stoi)
        return vocab


class NCVotersDataset(Dataset):
    def __init__(
        self,
        data_path,
        pairs_df=None,
        dpl_path=None,
        ndpl_path=None,
        vocab=None,
        max_len=80,
        attributes=None,
        strict_missing_ids=False,
        drop_missing_ids=True,
    ):
        self.data_df = load_prepared_data(data_path)
        self.data_dict = self.data_df.set_index("id").to_dict("index")

        if attributes is None:
            attributes = list(DEFAULT_BASELINE_ATTRIBUTES)
        self.attributes = list(attributes)
        self.max_len = max_len

        for attr in self.attributes:
            if attr not in self.data_df.columns:
                raise ValueError(f"Requested attribute '{attr}' not found in prepared data.")

        if pairs_df is None:
            if not dpl_path or not ndpl_path:
                raise ValueError("Either pairs_df or both dpl_path/ndpl_path must be provided.")
            pairs_df = load_labeled_pairs(dpl_path, ndpl_path)
        else:
            pairs_df = pairs_df.copy()

        for column in ["id1", "id2"]:
            pairs_df[column] = pairs_df[column].astype(str)
        pairs_df["label"] = pairs_df["label"].astype(float)

        valid_ids = set(self.data_df["id"].astype(str))
        valid_pair_mask = pairs_df["id1"].isin(valid_ids) & pairs_df["id2"].isin(valid_ids)
        self.missing_pair_rows = int((~valid_pair_mask).sum())
        self.missing_pair_ids = set()
        if self.missing_pair_rows > 0:
            missing_left = set(pairs_df.loc[~pairs_df["id1"].isin(valid_ids), "id1"].astype(str))
            missing_right = set(pairs_df.loc[~pairs_df["id2"].isin(valid_ids), "id2"].astype(str))
            self.missing_pair_ids = missing_left.union(missing_right)
            print(
                f"Warning: {self.missing_pair_rows} pair rows reference missing IDs. "
                f"Unique missing IDs: {len(self.missing_pair_ids)}."
            )
            if strict_missing_ids:
                raise ValueError("Pair file contains IDs absent from prepared data.")

        if drop_missing_ids:
            pairs_df = pairs_df.loc[valid_pair_mask].copy()

        self.pairs = pairs_df.reset_index(drop=True)

        if vocab is None:
            self.vocab = Vocabulary()
            all_text = []
            for attr in self.attributes:
                all_text.extend(self.data_df[attr].tolist())
            self.vocab.build_vocabulary(all_text)
            print(f"Vocabulary built with {self.vocab.vocab_size} characters.")
        else:
            self.vocab = vocab

    def _get_record_string(self, record_id):
        record = self.data_dict.get(record_id)
        if record is None:
            return ""
        return " ".join([str(record.get(attr, "")) for attr in self.attributes])

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        row = self.pairs.iloc[idx]
        s1 = self._get_record_string(row["id1"])
        s2 = self._get_record_string(row["id2"])

        x1 = self.vocab.numericalize(s1, self.max_len)
        x2 = self.vocab.numericalize(s2, self.max_len)

        return (
            torch.tensor(x1, dtype=torch.long),
            torch.tensor(x2, dtype=torch.long),
            torch.tensor(row["label"], dtype=torch.float),
        )
