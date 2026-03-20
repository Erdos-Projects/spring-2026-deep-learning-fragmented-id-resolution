"""
Hard Negative Weighting Module (Cosine Similarity-based)

Standalone module that computes sample weights for hard negatives.
Use this AFTER loading data, BEFORE training.

Flow: 
    data load → compute_hard_negative_weights() → training → evaluation
"""

import pandas as pd
import numpy as np


def compute_char_ngram_similarity(text1: str, text2: str, n: int = 3) -> float:
    """
    Compute character n-gram Jaccard similarity.
    
    Args:
        text1, text2: Strings to compare
        n: N-gram size (default 3)
    
    Returns:
        Similarity score in [0, 1]
    """
    if not text1 or not text2:
        return 0.0
    
    text1 = str(text1).lower().strip()
    text2 = str(text2).lower().strip()
    
    if text1 == text2:
        return 1.0
    
    # Generate character n-grams
    def get_ngrams(text, n):
        if len(text) < n:
            return {text}
        return {text[i:i+n] for i in range(len(text) - n + 1)}
    
    ngrams1 = get_ngrams(text1, n)
    ngrams2 = get_ngrams(text2, n)
    
    if not ngrams1 or not ngrams2:
        return 0.0
    
    # Jaccard similarity
    intersection = len(ngrams1 & ngrams2)
    union = len(ngrams1 | ngrams2)
    
    return intersection / union if union > 0 else 0.0


def detect_hard_negatives(data: pd.DataFrame, similarity_threshold: float = 0.7) -> np.ndarray:
    """
    Detect hard negative samples using cosine similarity.
    
    Hard negatives are non-duplicate pairs that look deceptively similar:
    - Same name, different age (family members)
    - Same last name + address (household members)
    - Near-match names at same location (typos vs different people)
    - etc. (10 total buckets)
    
    Args:
        data: DataFrame with columns like first_name_1, first_name_2, label, etc.
        similarity_threshold: Threshold for "near match" (default 0.7 = 70% similar)
    
    Returns:
        Boolean numpy array where True = hard negative
    """
    # Only check non-duplicates (label=0)
    non_dup_mask = data['label'] == 0
    is_hard = np.zeros(len(data), dtype=bool)
    
    if non_dup_mask.sum() == 0:
        print("  No non-duplicates to check.")
        return is_hard
    
    non_dup_df = data[non_dup_mask].copy()
    
    # Helper functions for exact matching
    def exact_match(col):
        c1 = f"{col}_1"
        c2 = f"{col}_2"
        return (non_dup_df[c1] == non_dup_df[c2]) & (non_dup_df[c1] != "") & (non_dup_df[c2] != "")
    
    def exact_diff(col):
        c1 = f"{col}_1"
        c2 = f"{col}_2"
        return (non_dup_df[c1] != non_dup_df[c2]) & (non_dup_df[c1] != "") & (non_dup_df[c2] != "")
    
    # Exact matches
    first_eq = exact_match("first_name")
    last_eq = exact_match("last_name")
    age_eq = exact_match("age")
    age_diff = exact_diff("age")
    sex_diff = exact_diff("sex")
    
    # Address columns (if available)
    has_address = all(col in non_dup_df.columns for col in 
                      ['house_num_1', 'house_num_2', 'street_name_1', 
                       'street_name_2', 'zip_code_1', 'zip_code_2'])
    
    if has_address:
        house_eq = exact_match("house_num")
        street_eq = exact_match("street_name")
        zip_eq = exact_match("zip_code")
        address_eq = house_eq & street_eq & zip_eq
    else:
        address_eq = pd.Series([False] * len(non_dup_df), index=non_dup_df.index)
        zip_eq = pd.Series([False] * len(non_dup_df), index=non_dup_df.index)
    
    # Compute cosine similarity for names
    print(f"  Computing name similarities (n={len(non_dup_df)} non-duplicates)...")
    
    first_sim = pd.Series([
        compute_char_ngram_similarity(str(a), str(b))
        for a, b in zip(non_dup_df["first_name_1"], non_dup_df["first_name_2"])
    ], index=non_dup_df.index)
    
    last_sim = pd.Series([
        compute_char_ngram_similarity(str(a), str(b))
        for a, b in zip(non_dup_df["last_name_1"], non_dup_df["last_name_2"])
    ], index=non_dup_df.index)
    
    # Near matches (similar but not exact)
    first_near = (first_sim >= similarity_threshold) & ~first_eq & \
                 (non_dup_df["first_name_1"] != "") & (non_dup_df["first_name_2"] != "")
    last_near = (last_sim >= similarity_threshold) & ~last_eq & \
                (non_dup_df["last_name_1"] != "") & (non_dup_df["last_name_2"] != "")
    
    # Define hard negative buckets
    buckets = {
        "same_first_last_diff_age": first_eq & last_eq & age_diff,
        "same_last_and_address": last_eq & address_eq,
        "same_first_and_age_diff_last": first_eq & age_eq & exact_diff("last_name"),
        "same_address_diff_age_or_sex": address_eq & (age_diff | sex_diff),
        "near_first_same_last_address": first_near & last_eq & address_eq,
        "same_full_name_address": first_eq & last_eq & address_eq,
        "same_full_name_zip_diff_address": first_eq & last_eq & zip_eq & ~address_eq,
        "same_first_last_zip_diff_age": first_eq & last_eq & zip_eq & age_diff,
        "same_first_age_zip_diff_last": first_eq & age_eq & zip_eq & exact_diff("last_name"),
        "near_name_same_zip": first_near & last_near & zip_eq,
    }
    
    # Combine all buckets
    combined_mask = pd.Series([False] * len(non_dup_df), index=non_dup_df.index)
    
    print(f"\n  Hard Negative Buckets:")
    for bucket_name, bucket_mask in buckets.items():
        count = bucket_mask.sum()
        if count > 0:
            print(f"    {bucket_name}: {count}")
        combined_mask |= bucket_mask
    
    # Map back to original DataFrame indices
    is_hard[non_dup_mask] = combined_mask.values
    
    return is_hard


def compute_hard_negative_weights(data: pd.DataFrame, 
                                   hard_negative_weight: float = 2.0,
                                   similarity_threshold: float = 0.7,
                                   return_mask: bool = False) -> np.ndarray:
    """
    Compute sample weights for hard negative training.
    
    This is the main function you'll use in your training pipeline:
    
    Example:
        # Load data
        train_df = pd.read_csv('data/train_pairs_aug.csv')
        
        # Compute weights (SEPARATE STEP)
        weights = compute_hard_negative_weights(
            train_df, 
            hard_negative_weight=2.5,  # Hyperparameter to tune!
            similarity_threshold=0.7
        )
        
        # Now use weights in training
        # (You'll pass this to your training loop)
    
    Args:
        data: DataFrame with pair data
        hard_negative_weight: Weight multiplier for hard negatives (HYPERPARAMETER)
        similarity_threshold: Cosine similarity threshold for "near match" (default 0.7)
        return_mask: If True, also return boolean mask of hard negatives
    
    Returns:
        weights: numpy array of sample weights (1.0 for regular, hard_negative_weight for hard)
        (optional) is_hard_negative: boolean mask if return_mask=True
    """
    print("\n" + "="*60)
    print("COMPUTING HARD NEGATIVE WEIGHTS")
    print("="*60)
    print(f"Hard negative weight: {hard_negative_weight}x")
    print(f"Similarity threshold: {similarity_threshold}")
    
    # Detect hard negatives
    is_hard_negative = detect_hard_negatives(data, similarity_threshold)
    
    # Compute weights
    weights = np.ones(len(data), dtype=np.float32)
    weights[is_hard_negative] = hard_negative_weight
    
    # Print summary
    total = len(data)
    n_duplicates = (data['label'] == 1).sum()
    n_non_duplicates = (data['label'] == 0).sum()
    n_hard_negatives = is_hard_negative.sum()
    
    print(f"\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Total samples: {total:,}")
    print(f"  Duplicates (label=1): {n_duplicates:,} ({n_duplicates/total*100:.1f}%)")
    print(f"  Non-duplicates (label=0): {n_non_duplicates:,} ({n_non_duplicates/total*100:.1f}%)")
    print(f"  Hard negatives: {n_hard_negatives:,} ({n_hard_negatives/total*100:.1f}%)")
    print(f"\nWeight distribution:")
    print(f"  Regular samples: {(weights == 1.0).sum():,} with weight 1.0x")
    print(f"  Hard negatives: {(weights == hard_negative_weight).sum():,} with weight {hard_negative_weight}x")
    print(f"="*60 + "\n")
    
    if return_mask:
        return weights, is_hard_negative
    return weights


def save_hard_negative_analysis(data: pd.DataFrame, is_hard_negative: np.ndarray, 
                                 output_path: str = "hard_negative_analysis.csv"):
    """
    Save hard negative samples to CSV for manual inspection.
    
    Args:
        data: DataFrame with pair data
        is_hard_negative: Boolean mask from compute_hard_negative_weights()
        output_path: Where to save the CSV
    """
    hard_neg_df = data[is_hard_negative].copy()
    hard_neg_df.to_csv(output_path, index=False)
    print(f"Saved {len(hard_neg_df)} hard negative samples to '{output_path}'")
    return hard_neg_df


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == '__main__':
    """
    Example: How to use this module in your training pipeline
    """
    import pandas as pd
    
    # 1. Load data (your existing code)
    print("Loading data...")
    train_df = pd.read_csv('data/train_pairs_aug.csv')
    
    # 2. Compute weights (NEW STEP - SEPARATE FROM TRAINING)
    weights = compute_hard_negative_weights(
        data=train_df,
        hard_negative_weight=2.0,  # HYPERPARAMETER - try 1.5, 2.0, 3.0, etc.
        similarity_threshold=0.7    # HYPERPARAMETER - try 0.6, 0.7, 0.8
    )
    
    # Output:
    # ============================================================
    # COMPUTING HARD NEGATIVE WEIGHTS
    # ============================================================
    # Hard negative weight: 2.0x
    # Similarity threshold: 0.7
    #   Computing name similarities (n=45000 non-duplicates)...
    # 
    #   Hard Negative Buckets:
    #     same_first_and_age_diff_last: 850
    #     same_address_diff_age_or_sex: 340
    #     ...
    # 
    # ============================================================
    # SUMMARY
    # ============================================================
    # Total samples: 50,000
    #   Duplicates (label=1): 5,000 (10.0%)
    #   Non-duplicates (label=0): 45,000 (90.0%)
    #   Hard negatives: 1,625 (3.2%)
    # 
    # Weight distribution:
    #   Regular samples: 48,375 with weight 1.0x
    #   Hard negatives: 1,625 with weight 2.0x
    # ============================================================
    
    # 3. Now use weights in your training loop
    # (You'll modify train_epoch to accept and apply these weights)
    print(f"\nWeights shape: {weights.shape}")
    print(f"Weights dtype: {weights.dtype}")
    print(f"Ready to pass to training loop!")
    
    # Optional: Save hard negatives for inspection
    weights_with_mask, is_hard_negative = compute_hard_negative_weights(
        train_df, 
        hard_negative_weight=2.0,
        return_mask=True
    )
    save_hard_negative_analysis(train_df, is_hard_negative, "hard_negatives.csv")
