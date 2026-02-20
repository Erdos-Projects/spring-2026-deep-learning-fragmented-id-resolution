# EDA and Preprocessing Insights: NCVoters Duplicate Detection

This document outlines the data preprocessing requirements and strategies for the North Carolina Voters duplicate resolution project.

## 1. Inclusion and Exclusion Criteria
Analysis of confirmed duplicate pairs (DPL) reveals that identity is preserved across record updates, but location is often not.

### **Inclusion Criteria (High Signal)**
*   **Identity Anchors:** `first_name`, `last_name`, `age`, and `sex`.
    *   *Insight:* `age` and `first_name` match in **>97%** of duplicate records. These are the most stable features for deep learning.
*   **Contextual Features:** `zip_code`, `street_name`, and `house_num`.
    *   *Insight:* These only match in **~10%** of duplicates. Including them allows the model to learn that a change in address does *not* necessarily imply a different person (migration patterns).
*   **Demographics:** `race_desc` and `ethnic_desc`. These provide additional disambiguation with high consistency (>64% match).

### **Exclusion Criteria (Noisy/Administrative)**
*   **Administrative Metadata:** `registr_dt`, `cancellation_dt`, `reason_cd`, and `status_cd`. 
    *   *Rationale:* These are artifacts of the database management system. Including them risks the model learning "snapshot" patterns rather than human identity, leading to poor generalization on future data.
*   **Sparsity Constraints:** `phone_num` and `area_cd`.
    *   *Rationale:* These attributes are missing in **~60%** of the dataset. While potentially useful, their sparsity creates a "missing data" bias that can destabilize Transformer training.

---

## 2. Transformation and Preprocessing Requirements
Before the data enters the Siamese Transformer model, the following transformations are necessary:

*   **Tagged Serialization:** Instead of raw concatenation, records should be serialized with attribute tags (e.g., `fn: [FIRST] ln: [LAST]`). This explicitly helps the Attention mechanism distinguish between different semantic fields.
*   **Null-Value Tokenization:** Empty strings should be replaced with a reserved `[NULL]` token. This prevents the model from collapsing field boundaries when data is missing.
*   **Normalization:**
    *   **Phonetic Encoding:** Implementing a Soundex or Metaphone layer for names can help the model handle "Hard Positives" (e.g., "Jon" vs "John").
    *   **Address Standardization:** Mapping suffixes (Street -> St, Avenue -> Ave) is critical given the high variance in address matching found during EDA.
*   **Subword Tokenization:** We recommend moving from character-level encoding to **Subword (BPE/WordPiece) Tokenization**. The mean serialized record length is **487 characters**, which is too long for efficient character-level RNNs but fits perfectly within the 512-token window of a Transformer.

---

## 3. Data Splitting and Overfitting Mitigation

### **Proposed Split**
*   **Train:** 70% (~75,000 pairs)
*   **Validation:** 15% (~16,000 pairs)
*   **Test:** 15% (~16,000 pairs)

### **Overfitting Risks and Solutions**
Deep Learning models for Entity Resolution overfit easily when the training set contains only "Easy Negatives."
*   **Hard Negative Mining:** Our EDA shows that the current `ncvoters_NDPL.tsv` contains pairs with high Levenshtein distances (>6.0). To ensure the model doesn't overfit, we must generate "Hard Negatives"—records that share a last name and age but have different first names (e.g., siblings or twins).
*   **Cluster-Based Splitting:** We will ensure that no individual record ID appears in both the training and testing sets. This prevents the model from "memorizing" specific individuals and forces it to learn the *logic* of similarity.
*   **Class Balancing:** The current ratio of Non-Duplicates to Duplicates is **10:1**. We will use **Triplet Loss** or weighted cross-entropy to prevent the model from defaulting to a "Never a Duplicate" prediction.

---

## 4. Computational Efficiency: Blocking
To avoid the $O(N^2)$ complexity of comparing every record, our EDA identifies a viable blocking strategy:
*   **Primary Block:** `first_name` or `age`.
*   *Justification:* Blocking by `first_name` retains **97.8%** of true duplicates while reducing the comparison space by several orders of magnitude. Traditional geographic blocking (by Zip Code) would fail, losing **85%** of the target duplicates.
