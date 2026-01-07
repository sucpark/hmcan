# Data Preparation Guide

## Overview

HMCAN supports various document classification datasets.

---

## Supported Datasets

### Quick Reference

| Dataset | Classes | Size | Doc Length | Task |
|---------|---------|------|------------|------|
| **Yelp** | 5 | 650K | Medium | Sentiment |
| **IMDB** | 2 | 50K | Medium | Sentiment |
| **AG News** | 4 | 120K | Short | Topic |
| **DBpedia** | 14 | 630K | Short | Topic |
| **Yahoo Answers** | 10 | 1.4M | Medium | Q&A |
| **20 Newsgroups** | 20 | 20K | Medium | Topic |

### Detailed Information

#### Yelp Review Full
- **HuggingFace**: `yelp_review_full`
- **Classes**: 1-5 star ratings
- **Use case**: Sentiment analysis benchmark
- **Features**: Various review lengths, balanced classes

#### IMDB
- **HuggingFace**: `imdb`
- **Classes**: Positive / Negative
- **Use case**: Binary sentiment classification
- **Features**: Movie reviews, long texts

#### AG News
- **HuggingFace**: `ag_news`
- **Classes**: World, Sports, Business, Sci/Tech
- **Use case**: News topic classification
- **Features**: Short news articles

#### DBpedia
- **HuggingFace**: `dbpedia_14`
- **Classes**: 14 Wikipedia categories
- **Use case**: Large-scale multi-class classification
- **Features**: Structured short descriptions

#### Yahoo Answers
- **HuggingFace**: `yahoo_answers_topics`
- **Classes**: 10 topics (Science, Health, Sports, ...)
- **Use case**: Q&A topic classification
- **Features**: Question + answer combined

#### 20 Newsgroups
- **HuggingFace**: `SetFit/20_newsgroups`
- **Classes**: 20 newsgroups
- **Use case**: Multi-class classification
- **Features**: Classic NLP benchmark

---

## Usage in Python

### Quick Load

```python
from hmcan.data import quick_load, list_datasets

# List available datasets
list_datasets()

# Load dataset
train, test, num_classes, labels = quick_load("imdb", max_samples=10000)
print(f"Classes: {num_classes}")
print(f"Labels: {labels}")
```

### Detailed Load

```python
from hmcan.data import load_classification_dataset, get_class_labels

# Load dataset
train_data, test_data, info = load_classification_dataset(
    name="ag_news",
    max_samples=10000,
    seed=42
)

# Check info
print(f"Dataset: {info.name}")
print(f"Classes: {info.num_classes}")
print(f"Train: {len(train_data)}, Test: {len(test_data)}")

# Class labels
labels = get_class_labels("ag_news")
print(f"Labels: {labels}")
```

### Usage in Colab

```python
from datasets import load_dataset

# Load directly from HuggingFace
dataset = load_dataset("imdb")
train = dataset["train"]
test = dataset["test"]
```

---

## Yelp Data (Legacy Method)

Preparing Yelp data for original HMCAN model:

Data Pipeline:
1. Download Yelp reviews (Hugging Face)
2. Download GloVe embeddings
3. Text preprocessing and tokenization
4. Build vocabulary
5. Save data

## Quick Start

```bash
# Download with default settings (10,000 samples)
python scripts/download_data.py
```

## Detailed Options

```bash
python scripts/download_data.py \
    --data-dir data \           # Data save path
    --max-samples 10000 \       # Maximum samples
    --embedding-dim 50 \        # Embedding dimension (50, 100, 200, 300)
    --min-freq 5 \              # Minimum word frequency
    --max-vocab 50000           # Maximum vocabulary size
```

### Option Description

| Option | Default | Description |
|--------|---------|-------------|
| `--data-dir` | `data` | Data storage directory |
| `--max-samples` | `10000` | Maximum number of reviews to download |
| `--embedding-dim` | `50` | GloVe embedding dimension |
| `--min-freq` | `5` | Minimum word frequency for vocabulary |
| `--max-vocab` | `50000` | Maximum vocabulary size |

## Generated Files

```
data/
├── embeddings/
│   └── glove.6B.50d.txt       # Original GloVe file (~160MB)
└── processed/
    ├── word2idx.json          # Vocabulary (word → index)
    ├── embeddings_50d.npz     # Extracted embeddings for vocabulary
    └── yelp_processed.npz     # Preprocessed document data
```

## Data Format

### word2idx.json
```json
{
    "<PAD>": 0,
    "<UNK>": 1,
    "the": 2,
    "a": 3,
    ...
}
```

### yelp_processed.npz
Each document is stored in the following format:
```python
{
    "document": [[1, 23, 45, ...], [67, 89, ...], ...],  # Word indices per sentence
    "label": 3  # 0-4 (star rating 1-5)
}
```

## Preprocessing Steps

1. **Sentence splitting**: NLTK `sent_tokenize`
2. **Word tokenization**: NLTK `word_tokenize`
3. **Lowercasing**: All text
4. **Punctuation removal**: Remove special characters
5. **Empty sentence filtering**: Minimum 1 word required
6. **OOV handling**: Words not in vocabulary → `<UNK>`

## Dataset Statistics

Based on default settings (10,000 samples):
- Train: 8,000 samples (80%)
- Validation: 1,000 samples (10%)
- Test: 1,000 samples (10%)
- Vocabulary size: ~15,000-20,000 words
- Average sentences/document: ~8-10
- Average words/sentence: ~15-20

## Custom Data Usage

### 1. Prepare Data Format

```python
# Prepare document list
documents = [
    [[1, 2, 3], [4, 5, 6, 7]],  # Document 1: 2 sentences
    [[8, 9], [10, 11, 12]],     # Document 2: 2 sentences
    ...
]
labels = [0, 3, ...]  # 0-4 class labels

# Save as npz
import numpy as np
data = [{"document": doc, "label": label} for doc, label in zip(documents, labels)]
np.savez_compressed("data/processed/custom_data.npz", data=np.array(data, dtype=object))
```

### 2. Create Vocabulary

```python
import json

word2idx = {"<PAD>": 0, "<UNK>": 1, "word1": 2, ...}
with open("data/processed/word2idx.json", "w") as f:
    json.dump(word2idx, f)
```

### 3. Prepare Embeddings (Optional)

```python
import numpy as np

vocab_size = len(word2idx)
embedding_dim = 50
embeddings = np.random.randn(vocab_size, embedding_dim).astype(np.float32)
np.savez_compressed("data/processed/embeddings_50d.npz", embeddings=embeddings)
```

## Memory Considerations

| Sample Count | Approximate Memory |
|--------------|-------------------|
| 10,000 | ~500 MB |
| 50,000 | ~2 GB |
| 100,000 | ~4 GB |
| Full (650K) | ~25 GB |

For large datasets, use `--max-samples` to limit or process in chunks.
