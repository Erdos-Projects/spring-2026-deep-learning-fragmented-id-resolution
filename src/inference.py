"""
Inference pipeline for duplicate detection on unseen data.

Provides a DuplicateDetector class that wraps:
  1. Preprocessing / normalization of raw input records
  2. Encoding via trained char-CNN encoder
  3. Candidate retrieval via cosine similarity (ANN-compatible)
  4. Pairwise scoring with the Siamese matching network
  5. Ranked match results with calibrated confidence scores

Usage:
    detector = DuplicateDetector.from_checkpoint("siamese_model.pt", device)
    results = detector.find_duplicates(new_records_df, existing_records_df, top_k=10)
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from .dataset import DEFAULT_FIELDS, RecordPairDataset
from .model import SiameseMatchingNetwork
from .preprocessing import normalize_dataframe, normalize_name, normalize_street_address, normalize_zip5


class DuplicateDetector:
    """
    End-to-end duplicate detection on unseen, unaugmented data.
    
    Wraps the trained Siamese char-CNN model and provides methods for:
      - Embedding records into the learned vector space
      - Finding candidate duplicate pairs via nearest-neighbor search
      - Scoring specific record pairs with calibrated probabilities
    """

    def __init__(
        self,
        model: SiameseMatchingNetwork,
        char_vocab: Dict[str, int],
        device: torch.device,
        fields: Optional[List[str]] = None,
        max_len: int = 128,
    ):
        self.model = model.to(device)
        self.model.eval()
        self.char_vocab = char_vocab
        self.device = device
        self.fields = fields or list(DEFAULT_FIELDS)
        self.max_len = max_len

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        device: torch.device,
        **model_kwargs,
    ) -> "DuplicateDetector":
        """
        Load a DuplicateDetector from a saved checkpoint.
        
        Args:
            checkpoint_path: Path to .pt file saved by the training pipeline
            device: torch device to load model onto
            **model_kwargs: Override model hyperparameters if needed
        """
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        char_vocab = ckpt["char_vocab"]

        # Default model config (matches pipeline defaults)
        config = {
            "vocab_size": len(char_vocab),
            "embed_dim": 32,
            "num_filters": 128,
            "kernel_sizes": (3, 4, 5),
            "encoder_output_dim": 128,
            "classifier_hidden_dim": 64,
            "dropout": 0.3,
        }
        config.update(model_kwargs)

        model = SiameseMatchingNetwork(**config)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        return cls(model=model, char_vocab=char_vocab, device=device)

    def _encode_text(self, text: str) -> torch.LongTensor:
        """Encode text string to character index tensor."""
        text = text.upper()
        indices = []
        for ch in text[:self.max_len]:
            indices.append(self.char_vocab.get(ch, self.char_vocab.get("<UNK>", 1)))
        while len(indices) < self.max_len:
            indices.append(self.char_vocab.get("<PAD>", 0))
        return torch.LongTensor(indices)

    def _record_to_text(self, record: Dict[str, str]) -> str:
        """Convert a record dict to concatenated text."""
        parts = []
        for f in self.fields:
            val = str(record.get(f, "")).strip()
            if val:
                parts.append(val.upper())
        return " | ".join(parts)

    def normalize_records(
        self,
        df: pd.DataFrame,
        snapshot_year: int = 2018,
    ) -> pd.DataFrame:
        """
        Normalize raw records for model consumption.
        
        Handles both pre-normalized data (has 'first_name', 'street_address')
        and raw NCVoters format (has 'house_num', 'street_name', etc.).
        """
        # Check if already normalized
        normalized_cols = {"first_name", "last_name", "street_address", "zip5"}
        raw_cols = {"house_num", "street_name", "street_type_cd"}

        if normalized_cols.issubset(set(df.columns)):
            # Already normalized — just ensure string types and uppercase
            out = df.copy()
            for col in self.fields:
                if col in out.columns:
                    out[col] = out[col].astype(str).str.strip().str.upper()
                    out.loc[out[col] == "NAN", col] = ""
                    out.loc[out[col] == "NONE", col] = ""
            return out
        elif raw_cols.issubset(set(df.columns)):
            # Raw NCVoters format — full normalization
            return normalize_dataframe(df, snapshot_year=snapshot_year)
        else:
            # Best-effort: uppercase all available fields
            out = df.copy()
            for col in self.fields:
                if col in out.columns:
                    out[col] = out[col].astype(str).str.strip().str.upper()
                    out.loc[out[col] == "NAN", col] = ""
                    out.loc[out[col] == "NONE", col] = ""
            return out

    @torch.no_grad()
    def embed_records(
        self,
        records: pd.DataFrame,
        batch_size: int = 256,
    ) -> torch.Tensor:
        """
        Embed all records using the trained encoder.
        
        Args:
            records: DataFrame with identity fields (must have 'id' column)
            batch_size: Batch size for encoding
            
        Returns:
            Tensor of shape (n_records, embedding_dim)
        """
        self.model.eval()
        texts = []
        for _, row in records.iterrows():
            record = row.to_dict()
            texts.append(self._record_to_text(record))

        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_tensors = torch.stack([
                self._encode_text(t) for t in batch_texts
            ]).to(self.device)
            emb = self.model.encode(batch_tensors)
            embeddings.append(emb.cpu())

        return torch.cat(embeddings, dim=0)

    @torch.no_grad()
    def score_pairs(
        self,
        pairs: List[Tuple[Dict[str, str], Dict[str, str]]],
        batch_size: int = 256,
    ) -> np.ndarray:
        """
        Score a list of record pairs.
        
        Args:
            pairs: List of (record1_dict, record2_dict) tuples
            batch_size: Batch size for scoring
            
        Returns:
            Array of match probabilities, shape (n_pairs,)
        """
        self.model.eval()
        all_probs = []

        for i in range(0, len(pairs), batch_size):
            batch = pairs[i:i + batch_size]
            input1 = torch.stack([
                self._encode_text(self._record_to_text(r1))
                for r1, r2 in batch
            ]).to(self.device)
            input2 = torch.stack([
                self._encode_text(self._record_to_text(r2))
                for r1, r2 in batch
            ]).to(self.device)

            probs = self.model.predict_proba(input1, input2).cpu().numpy().flatten()
            all_probs.extend(probs)

        return np.array(all_probs)

    def find_duplicates(
        self,
        query_records: pd.DataFrame,
        database_records: pd.DataFrame,
        top_k: int = 10,
        threshold: float = 0.5,
        batch_size: int = 256,
        normalize: bool = True,
    ) -> pd.DataFrame:
        """
        Find potential duplicates for query records in a database.
        
        This is the main entry point for duplicate detection on unseen data.
        It performs:
          1. Normalization (if needed)
          2. Embedding of all records
          3. Cosine-similarity candidate retrieval (top-K)
          4. Pairwise Siamese scoring of candidates
          5. Threshold filtering and ranking
        
        Args:
            query_records: New records to check for duplicates
            database_records: Existing records to match against
            top_k: Number of nearest neighbors to retrieve per query
            threshold: Minimum match probability to include in results
            batch_size: Batch size for encoding/scoring
            normalize: Whether to normalize input records
            
        Returns:
            DataFrame with columns:
                [query_id, match_id, similarity, match_probability, 
                 query_name, match_name, query_address, match_address]
            sorted by match_probability descending
        """
        # Normalize if needed
        if normalize:
            query_norm = self.normalize_records(query_records)
            db_norm = self.normalize_records(database_records)
        else:
            query_norm = query_records
            db_norm = database_records

        # Embed
        print(f"Embedding {len(query_norm)} query records...")
        query_emb = self.embed_records(query_norm, batch_size=batch_size)
        print(f"Embedding {len(db_norm)} database records...")
        db_emb = self.embed_records(db_norm, batch_size=batch_size)

        # Cosine similarity for candidate retrieval
        query_normed = query_emb / query_emb.norm(dim=1, keepdim=True).clamp(min=1e-8)
        db_normed = db_emb / db_emb.norm(dim=1, keepdim=True).clamp(min=1e-8)
        sim_matrix = query_normed @ db_normed.T  # (n_query, n_db)

        # Get top-K candidates per query
        actual_k = min(top_k, sim_matrix.shape[1])
        top_sims, top_indices = sim_matrix.topk(actual_k, dim=1)

        query_ids = query_norm["id"].tolist() if "id" in query_norm.columns else list(range(len(query_norm)))
        db_ids = db_norm["id"].tolist() if "id" in db_norm.columns else list(range(len(db_norm)))

        # Score candidates with Siamese network
        results = []
        candidate_pairs = []
        pair_metadata = []

        for qi in range(len(query_norm)):
            query_record = query_norm.iloc[qi].to_dict()
            for ki in range(actual_k):
                di = top_indices[qi, ki].item()
                cos_sim = top_sims[qi, ki].item()

                # Skip self-matches
                qid = query_ids[qi]
                did = db_ids[di]
                if str(qid) == str(did):
                    continue

                db_record = db_norm.iloc[di].to_dict()
                candidate_pairs.append((query_record, db_record))
                pair_metadata.append({
                    "query_id": qid,
                    "match_id": did,
                    "cosine_similarity": cos_sim,
                    "query_name": f"{query_record.get('first_name', '')} {query_record.get('last_name', '')}".strip(),
                    "match_name": f"{db_record.get('first_name', '')} {db_record.get('last_name', '')}".strip(),
                    "query_address": query_record.get("street_address", ""),
                    "match_address": db_record.get("street_address", ""),
                })

        if not candidate_pairs:
            return pd.DataFrame(columns=[
                "query_id", "match_id", "cosine_similarity", "match_probability",
                "query_name", "match_name", "query_address", "match_address",
            ])

        # Batch score all candidates
        print(f"Scoring {len(candidate_pairs)} candidate pairs...")
        probs = self.score_pairs(candidate_pairs, batch_size=batch_size)

        for i, meta in enumerate(pair_metadata):
            meta["match_probability"] = float(probs[i])

        results_df = pd.DataFrame(pair_metadata)

        # Filter by threshold and sort
        results_df = results_df[results_df["match_probability"] >= threshold]
        results_df = results_df.sort_values("match_probability", ascending=False).reset_index(drop=True)

        return results_df

    def score_single_pair(
        self,
        record1: Dict[str, str],
        record2: Dict[str, str],
    ) -> float:
        """
        Score a single pair of records. Returns match probability [0, 1].
        
        Useful for interactive / Streamlit use cases.
        """
        probs = self.score_pairs([(record1, record2)])
        return float(probs[0])
