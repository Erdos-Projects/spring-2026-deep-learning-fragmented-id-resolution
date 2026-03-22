import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report, confusion_matrix
import os
import importlib.util

# Define paths for dynamic import
#repo_root = '/content/spring-2026-deep-learning-fragmented-id-resolution'
#model_path = os.path.join(repo_root, 'src/model.py')

# Load the module directly from the file path
#spec = importlib.util.spec_from_file_location('model', model_path)
#model_module = importlib.util.module_from_spec(spec)
#spec.loader.exec_module(model_module)

# Extract the class
#CharCNNEncoder = model_module.CharCNNEncoder

# Device configuration
device = torch.device('cpu')  # Use CPU by default for consistency with models

# Vocabulary mapping
chars = 'abcdefghijklmnopqrstuvwxyz0123456789 '
char_to_idx = {char: i + 1 for i, char in enumerate(chars)}
vocab_size = len(char_to_idx) + 1

def tokenize_and_pad(text, mapping=char_to_idx, max_len=100):
    sequence = [mapping[char] for char in str(text).lower() if char in mapping] # Convert to lower case here
    sequence = sequence[:max_len]
    padding_length = max_len - len(sequence)
    sequence.extend([0] * padding_length)
    return sequence

def get_embeddings(df, model, char_map=char_to_idx, max_len=100, batch_size=128):
    sequences = []
    for _, row in df.iterrows():
        # Format using all columns except 'id'
        formatted_name = ' '.join([str(row[col]) for col in df.columns if col != 'id'])
        sequences.append(tokenize_and_pad(formatted_name, char_map, max_len))
    
    dataset = TensorDataset(torch.LongTensor(sequences))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model.eval()
    embeddings_list = []
    with torch.no_grad():
        for batch in loader:
            outputs = model(batch[0].to(device))
            embeddings_list.append(outputs.cpu().numpy())

    embeddings_array = np.concatenate(embeddings_list, axis=0)
    res_df = pd.DataFrame({'id': df['id'].values})
    enc_cols = [f'enc_{i}' for i in range(embeddings_array.shape[1])]
    enc_df = pd.DataFrame(embeddings_array, columns=enc_cols)
    return pd.concat([res_df, enc_df], axis=1)

def cosine_sim(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def evaluate_embeddings(dup_pairs_df, ndup_pairs_df, embeddings_df, thresholds=[0.95, 0.96, 0.97, 0.98, 0.99]):
    emb_id_map = {idx: i for i, idx in enumerate(embeddings_df['id'].values)}
    embeddings_array = embeddings_df.drop(columns=['id']).values

    def get_pair_sims_internal(pairs_df, embeddings, id_map):
        sims = []
        for _, row in pairs_df.iterrows():
            id1, id2 = row['id1'], row['id2']
            if id1 in id_map and id2 in id_map:
                v1 = embeddings[id_map[id1]]
                v2 = embeddings[id_map[id2]]
                sims.append(cosine_sim(v1, v2))
        return sims

    dup_sims = get_pair_sims_internal(dup_pairs_df, embeddings_array, emb_id_map)
    ndup_sims = get_pair_sims_internal(ndup_pairs_df, embeddings_array, emb_id_map)

    y_true = [1] * len(dup_sims) + [0] * len(ndup_sims)
    y_sims = dup_sims + ndup_sims

    print(f"Pairs fully present in embeddings:")
    print(f"- Duplicates: {len(dup_sims)}")
    print(f"- Non-duplicates: {len(ndup_sims)}")
    print(f"Total pairs for evaluation: {len(y_true)}\n")

    for threshold in thresholds:
        y_pred = [1 if sim >= threshold else 0 for sim in y_sims]

        print(f"--- Evaluation Performance (Threshold: {threshold}) ---")
        print(classification_report(y_true, y_pred, target_names=['Non-Duplicate', 'Duplicate']))

        cm = confusion_matrix(y_true, y_pred)
        print("Confusion Matrix:")
        print(f"TP: {cm[1,1]} | FN: {cm[1,0]}")
        print(f"FP: {cm[0,1]} | TN: {cm[0,0]}")
        print("-" * 30)
