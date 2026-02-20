# Project Methodology: Deep Learning for Entity Resolution

## 1. Data Acquisition and Exploratory Data Analysis (EDA)
The project utilizes the **North Carolina Voters Registry** as the primary dataset. Before model development, we will perform a comprehensive EDA to establish a baseline for data quality and distribution.

* **Data Profiling:** We will analyze the dataset for cardinality (unique values), null-value density, and attribute distribution (e.g., distribution of zip codes or birth years).
* **Exact Duplicate Detection:** A baseline "naive" approach will be executed to identify and remove pre-existing exact duplicates (100% string matches) to ensure our training subset is distinct.
* **Subset Selection:** We will isolate a "clean" subset of approximately 10,000–50,000 unique records. This subset will serve as the **Anchor** data for our synthetic generation pipeline, effectively acting as the ground truth ($Y_{true}$) for the self-supervised learning task.

## 2. Synthetic Data Generation Strategy
To overcome the "Cold Start" problem (the lack of labeled duplicate pairs), we will implement a self-supervised learning approach using **Data Augmentation**. We will generate a synthetic training dataset containing labeled pairs of records.

### A. Noise Injection Models (Generating Positives)
We will define a set of stochastic functions to simulate real-world data entry errors. For a given anchor record $A$, we generate a positive duplicate $A'$ using:

* **Keyboard Distance Noise:** Simulating "fat-finger" typos based on QWERTY keyboard adjacency (e.g., replacing 's' with 'a' or 'd').
* **Phonetic Noise:** Replacing names with phonetically similar but strictly distinct spellings (e.g., "Catherine" $\rightarrow$ "Katherine").
* **OCR/Visual Noise:** Simulating errors common in scanned documents (e.g., '1' $\rightarrow$ 'I', '0' $\rightarrow$ 'O').
* **Token Corruption:** Randomly dropping middle names, swapping first/last names, or truncating street suffixes (e.g., "Street" $\rightarrow$ "St").

### B. Hard Sample Mining
To ensure the model learns semantic similarity rather than simple string matching, we will explicitly engineer "Hard" samples:

* **Hard Positives:** Pairs that refer to the same entity but look significantly different (e.g., high edit distance due to abbreviations and nicknames).
* **Hard Negatives:** Pairs that represent *different* entities but look highly similar.
    * *Example:* Same Name and Address, but different Date of Birth (simulating twins or clerical errors).
    * *Example:* Same Last Name and Address, but different First Name (simulating family members living together).

The final training set will consist of triplets or pairs: `(Anchor, Positive, Label=1)` and `(Anchor, Negative, Label=0)`.

## 3. Model Architecture
We will employ a **Deep Learning approach based on Metric Learning**.

![Placeholder: Diagram of Siamese Network Architecture]

* **Architecture:** We will utilize a **Siamese Network** (or Bi-Encoder) architecture. This setup uses two identical subnetworks with shared weights to process the two records independently.
* **Encoder Backbone:** The subnetworks will be powered by a Transformer-based model (e.g., **DistilBERT** or **RoBERTa**). Transformers are chosen over RNNs/LSTMs for their superior ability to capture global context and semantic relationships within the text (e.g., understanding that "Bill" and "William" are related).
* **Embedding Space:** The model will output high-dimensional vector embeddings for each record. The objective is to map true duplicates close together in vector space and non-duplicates far apart.

## 4. Training and Optimization
* **Input Representation:** Records will be serialized into a single string (e.g., `[CLS] First Last [SEP] Address [SEP] Zip [SEP]`).
* **Loss Function:** We will use **Contrastive Loss** or **Triplet Loss**. These functions penalize the model when the distance between an Anchor and a Positive is large, or when the distance between an Anchor and a Negative is small.
* **Validation:** During training, performance will be monitored on a held-out synthetic validation set to prevent overfitting to the noise generators.

## 5. Evaluation Strategy (The "Gold Standard")
Evaluating solely on synthetic data risks "learning the noise generator" rather than solving the real problem.

* **The Gold Set:** We will manually label a small, representative sample of the *real, unaltered* North Carolina dataset (approx. 100–200 pairs) to create a "Gold Standard" test set.
* **Metrics:** The model will be evaluated on this Gold Set using **Precision** (how many predicted duplicates are actually duplicates), **Recall** (how many real duplicates did we find), and **F1-Score**.

## 6. Inference Pipeline (Real-World Application)
Once trained, the model will be applied to the remaining unlabeled dataset.

![Placeholder: Diagram of Record Linkage Pipeline with Blocking]

* **Blocking (Candidate Generation):** To avoid the computational expense of comparing every record against every other ($N^2$ complexity), we will implement a Blocking pass. We will only run the Deep Learning model on pairs that share a coarse attribute (e.g., matching Zip Code or Soundex of Last Name).
* **Pairwise Classification:** The trained model will predict a similarity score for the candidate pairs.
* **Thresholding:** Pairs with a similarity score above a determined threshold (e.g., $>0.85$) will be flagged as duplicates.