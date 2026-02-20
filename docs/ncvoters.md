# NCVoters Dataset Summary
This document provides a summary of the NCVoters dataset, its components, and its intended use for duplicate detection research, obtained from: https://hpi.de/naumann/projects/repeatability/datasets/ncvoters-dataset.html
## Overview
The **NCVoters dataset** is a sampled subset of the North Carolina Voter Registration data (specifically the `VR_Snapshot_20181106` snapshot). It was curated by the [HPI Information Systems Group](https://hpi.de/naumann/projects/repeatability/datasets/ncvoters-dataset.html) for research in data cleaning and duplicate detection.

## Dataset Components

### 1. Main Dataset
* **Size:** 14,183 objects (records).
* **Format:** Tab-Separated Values (TSV).
* **Preparation:** Data has been normalized with lower-casing and the removal of special characters.
* **Characteristics:** The sampling technique was designed to reduce the size of the original massive registration list while maintaining the original ratios of duplicate clusters.

### 2. Duplicates (Ground Truth)
* **Size:** 9,819 objects.
* **Description:** This file contains the records from the main dataset that are known to be duplicates. 
* **Use Case:** These serve as the "Positive" matches for testing entity resolution and linking algorithms.

### 3. Non-duplicates (Control Group)
* **Size:** 98,142 pairs.
* **Description:** A large collection of record pairs generated through a [systematic approach](https://hpi.de/naumann/projects/repeatability/datasets/ncvoters-dataset.html) to ensure they are distinct individuals.
* **Use Case:** These serve as the "Negative" matches to test the precision of an algorithm and ensure it does not incorrectly link different people.

## Purpose & Methodology
The dataset is primarily used to benchmark the efficiency and accuracy of **duplicate detection** algorithms. By providing a fixed set of confirmed matches (Duplicates) and confirmed non-matches (Non-duplicates), researchers can calculate standard metrics like Precision, Recall, and F1-score.

**Source:** [HPI NCVoters Project Page](https://hpi.de/naumann/projects/repeatability/datasets/ncvoters-dataset.html)