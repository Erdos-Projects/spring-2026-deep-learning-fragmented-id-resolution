from collections import Counter

import pandas as pd
import torch
from torch.utils.data import Dataset

try:
    from .data_utils import DEFAULT_BASELINE_ATTRIBUTES, load_labeled_pairs, load_prepared_data
except ImportError:
    from data_utils import DEFAULT_BASELINE_ATTRIBUTES, load_labeled_pairs, load_prepared_data


PAIR_FEATURE_SET_DEFINITIONS = {
    "none": [],
    "sex_aware_name": [
        "same_first_name",
        "same_last_name",
        "same_midl_name",
        "same_age",
        "same_sex",
        "same_house_num",
        "same_street_name",
        "same_zip_code",
        "both_male",
        "both_female",
        "male_same_first_diff_last",
        "female_same_first_diff_last",
        "male_same_first_diff_last_same_age",
        "female_same_first_diff_last_same_age",
    ],
}
PAIR_FEATURE_SET_NAMES = list(PAIR_FEATURE_SET_DEFINITIONS.keys())


def get_pair_feature_names(pair_feature_set="none"):
    if pair_feature_set not in PAIR_FEATURE_SET_DEFINITIONS:
        raise ValueError(f"Unknown pair_feature_set '{pair_feature_set}'.")
    return list(PAIR_FEATURE_SET_DEFINITIONS[pair_feature_set])


def _normalized_value(record, attribute):
    return str(record.get(attribute, "")).strip().lower()


def _same_nonempty(record1, record2, attribute):
    left = _normalized_value(record1, attribute)
    right = _normalized_value(record2, attribute)
    return bool(left and right and left == right)


def _is_male(record):
    return _normalized_value(record, "sex") in {"m", "male"}


def _is_female(record):
    return _normalized_value(record, "sex") in {"f", "female"}


def compute_pair_features_from_records(record1, record2, pair_feature_set="none"):
    if pair_feature_set == "none":
        return []
    if pair_feature_set != "sex_aware_name":
        raise ValueError(f"Unknown pair_feature_set '{pair_feature_set}'.")

    same_first = _same_nonempty(record1, record2, "first_name")
    same_last = _same_nonempty(record1, record2, "last_name")
    same_midl = _same_nonempty(record1, record2, "midl_name")
    same_age = _same_nonempty(record1, record2, "age")
    same_sex = _same_nonempty(record1, record2, "sex")
    same_house = _same_nonempty(record1, record2, "house_num")
    same_street = _same_nonempty(record1, record2, "street_name")
    same_zip = _same_nonempty(record1, record2, "zip_code")
    last_name_diff = bool(
        _normalized_value(record1, "last_name")
        and _normalized_value(record2, "last_name")
        and _normalized_value(record1, "last_name") != _normalized_value(record2, "last_name")
    )
    both_male = _is_male(record1) and _is_male(record2)
    both_female = _is_female(record1) and _is_female(record2)

    features = [
        same_first,
        same_last,
        same_midl,
        same_age,
        same_sex,
        same_house,
        same_street,
        same_zip,
        both_male,
        both_female,
        both_male and same_first and last_name_diff,
        both_female and same_first and last_name_diff,
        both_male and same_first and last_name_diff and same_age,
        both_female and same_first and last_name_diff and same_age,
    ]
    return [float(value) for value in features]


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
        sample_weight_column=None,
        pair_feature_set="none",
    ):
        self.data_df = load_prepared_data(data_path)
        self.data_dict = self.data_df.set_index("id").to_dict("index")

        if attributes is None:
            attributes = list(DEFAULT_BASELINE_ATTRIBUTES)
        self.attributes = list(attributes)
        self.max_len = max_len
        self.sample_weight_column = sample_weight_column
        self.pair_feature_set = pair_feature_set
        self.pair_feature_names = get_pair_feature_names(pair_feature_set)
        self.pair_feature_dim = len(self.pair_feature_names)

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

        if self.sample_weight_column is not None and self.sample_weight_column not in self.pairs.columns:
            raise ValueError(
                f"sample_weight_column '{self.sample_weight_column}' not found in pair dataframe."
            )

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

    def _get_pair_features(self, id1, id2):
        if self.pair_feature_set == "none":
            return []
        record1 = self.data_dict.get(str(id1), {})
        record2 = self.data_dict.get(str(id2), {})
        return compute_pair_features_from_records(record1, record2, self.pair_feature_set)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        row = self.pairs.iloc[idx]
        s1 = self._get_record_string(row["id1"])
        s2 = self._get_record_string(row["id2"])

        x1 = self.vocab.numericalize(s1, self.max_len)
        x2 = self.vocab.numericalize(s2, self.max_len)
        pair_features = None
        if self.pair_feature_dim > 0:
            pair_features = torch.tensor(self._get_pair_features(row["id1"], row["id2"]), dtype=torch.float)

        if self.sample_weight_column is not None and pair_features is not None:
            return (
                torch.tensor(x1, dtype=torch.long),
                torch.tensor(x2, dtype=torch.long),
                torch.tensor(row["label"], dtype=torch.float),
                pair_features,
                torch.tensor(float(row[self.sample_weight_column]), dtype=torch.float),
            )
        if self.sample_weight_column is not None:
            return (
                torch.tensor(x1, dtype=torch.long),
                torch.tensor(x2, dtype=torch.long),
                torch.tensor(row["label"], dtype=torch.float),
                torch.tensor(float(row[self.sample_weight_column]), dtype=torch.float),
            )
        if pair_features is not None:
            return (
                torch.tensor(x1, dtype=torch.long),
                torch.tensor(x2, dtype=torch.long),
                torch.tensor(row["label"], dtype=torch.float),
                pair_features,
            )
        return (
            torch.tensor(x1, dtype=torch.long),
            torch.tensor(x2, dtype=torch.long),
            torch.tensor(row["label"], dtype=torch.float),
        )
