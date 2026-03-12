# Deep Learning Fragmented ID Resolution

This repository contains code for resolving fragmented IDs using a Character-level Convolutional Neural Network (CharCNN) based embedding model.

## Project Structure

- `src/model.py`: Defines the `CharCNNEncoder` neural network architecture.
- `src/model_utils.py`: Contains utility functions for tokenization, embedding generation, cosine similarity calculation, and evaluation of embeddings.
- `data/splits/Naive/`: Contains the datasets (`train_voters.csv`, `val_voters.csv`, `test_voters.csv`) and duplicate/non-duplicate pairs (`naive_dup_val.csv`, etc.) used for training and evaluation.

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    !git clone https://github.com/Erdos-Projects/spring-2026-deep-learning-fragmented-id-resolution
    %cd spring-2026-deep-learning-fragmented-id-resolution
    ```
2.  **Install dependencies:**
    This project primarily uses `torch`, `pandas`, `numpy`, and `scikit-learn`. Ensure these are installed in your environment.
    ```bash
    !pip install torch pandas numpy scikit-learn
    ```

## Usage

### 1. Data Loading and Preprocessing

The data is loaded from the `data/splits/Naive/` directory. The voter datasets are preprocessed to create a `formatted_name` column by concatenating relevant identity features.

```python
import pandas as pd
import os

# Change directory to the repository root
%cd /content/spring-2026-deep-learning-fragmented-id-resolution

# Load data (example for validation split)
val_voters = pd.read_csv('data/splits/Naive/val_voters.csv')
dup_val = pd.read_csv('data/splits/Naive/naive_dup_val.csv')
ndup_val = pd.read_csv('data/splits/Naive/naive_ndup_val.csv')

def prepare_voter_strings(df, features):
    df_cols = df[features].fillna('')
    df['formatted_name'] = df_cols.astype(str).agg(' '.join, axis=1)
    df['formatted_name'] = df['formatted_name'].str.lower().str.replace(r'\\s+', ' ', regex=True).str.strip()
    return df

features_to_encode = ['first_name', 'midl_name', 'last_name', 'age', 'sex']
val_voters = prepare_voter_strings(val_voters, features_to_encode)
```

### 2. Initializing the Model and Utilities

The `CharCNNEncoder` model and utility functions for embedding generation and evaluation are defined in `src/model.py` and `src/model_utils.py`.

```python
import torch
from src.model_utils import CharCNNEncoder, get_embeddings, evaluate_embeddings, char_to_idx, vocab_size, device

# Initialize the model
model = CharCNNEncoder(
    vocab_size=vocab_size,
    embed_dim=32,
    num_filters=128,
    kernel_sizes=(3, 4, 5),
    dropout=0.3,
    output_dim=128
).to(device)
model.eval()
```

### 3. Generating Embeddings

Use the `get_embeddings` function from `model_utils.py` to generate embeddings for your dataset.

```python
val_emb_df_new = get_embeddings(val_voters, model, char_map=char_to_idx)
print(f"Validation embeddings DataFrame shape: {val_emb_df_new.shape}")
```

### 4. Evaluating Embeddings

Use the `evaluate_embeddings` function to assess the performance of the generated embeddings on duplicate and non-duplicate pairs.

```python
evaluate_embeddings(dup_val, ndup_val, val_emb_df_new, thresholds=[0.97, 0.98, 0.99])
```

## Model Details

The `CharCNNEncoder` is a character-level convolutional neural network designed to generate fixed-size embeddings from input text (e.g., formatted names). It uses:
- An embedding layer to convert characters to dense vectors.
- Multiple 1D convolutional layers with different kernel sizes to capture local patterns.
- Max-pooling to get fixed-size representations from each convolutional filter.
- A concatenation of pooled features, followed by a dropout layer and a linear layer to produce the final embedding.
- Layer Normalization for stable training.

## Evaluation Metrics

The `evaluate_embeddings` function provides a `classification_report` and `confusion_matrix` for various cosine similarity thresholds, allowing you to gauge the model's ability to distinguish between duplicate and non-duplicate records.
