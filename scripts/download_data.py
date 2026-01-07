#!/usr/bin/env python3
"""
Download and prepare the Yelp review dataset for training.

This script:
1. Downloads the Yelp review dataset from Hugging Face
2. Preprocesses the text (tokenization, vocabulary building)
3. Downloads GloVe embeddings
4. Saves processed data and embeddings
"""

from __future__ import annotations

import json
import zipfile
from collections import Counter
from pathlib import Path
from typing import Optional

import numpy as np
import requests
from tqdm import tqdm

# Optional: use datasets library if available
try:
    from datasets import load_dataset
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False

try:
    import nltk
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)
    from nltk.tokenize import sent_tokenize, word_tokenize
    HAS_NLTK = True
except ImportError:
    HAS_NLTK = False


def download_glove(data_dir: Path, dim: int = 50) -> Path:
    """
    Download GloVe embeddings.

    Args:
        data_dir: Directory to save embeddings
        dim: Embedding dimension (50, 100, 200, 300)

    Returns:
        Path to the embeddings file
    """
    embeddings_dir = data_dir / "embeddings"
    embeddings_dir.mkdir(parents=True, exist_ok=True)

    glove_file = embeddings_dir / f"glove.6B.{dim}d.txt"

    if glove_file.exists():
        print(f"GloVe embeddings already exist at {glove_file}")
        return glove_file

    # Download GloVe
    url = "http://nlp.stanford.edu/data/glove.6B.zip"
    zip_path = embeddings_dir / "glove.6B.zip"

    if not zip_path.exists():
        print(f"Downloading GloVe embeddings from {url}...")
        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()
        total_size = int(response.headers.get("content-length", 0))

        with (
            open(zip_path, "wb") as f,
            tqdm(total=total_size, unit="B", unit_scale=True, desc="Downloading") as pbar,
        ):
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                pbar.update(len(chunk))

    # Extract
    print("Extracting GloVe embeddings...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(embeddings_dir)

    # Cleanup zip
    zip_path.unlink()

    return glove_file


def load_glove_embeddings(
    glove_path: Path,
    word2idx: dict[str, int],
    embedding_dim: int = 50,
) -> np.ndarray:
    """
    Load GloVe embeddings for vocabulary.

    Args:
        glove_path: Path to GloVe file
        word2idx: Word to index mapping
        embedding_dim: Embedding dimension

    Returns:
        Embedding matrix of shape (vocab_size, embedding_dim)
    """
    vocab_size = len(word2idx)
    embeddings = np.random.uniform(-0.25, 0.25, (vocab_size, embedding_dim))
    embeddings[0] = 0  # Padding token

    found = 0
    print(f"Loading GloVe embeddings from {glove_path}...")

    with open(glove_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Loading embeddings"):
            parts = line.strip().split()
            word = parts[0]
            if word in word2idx:
                idx = word2idx[word]
                embeddings[idx] = np.array(parts[1:], dtype=np.float32)
                found += 1

    print(f"Found {found}/{vocab_size} words in GloVe ({100*found/vocab_size:.1f}%)")
    return embeddings


def preprocess_text(text: str) -> list[list[str]]:
    """
    Preprocess text into list of sentences, each a list of words.

    Args:
        text: Raw text

    Returns:
        List of sentences, each a list of lowercase words
    """
    import string

    # Remove punctuation (keeping sentence structure)
    sentences = sent_tokenize(text.lower())

    processed = []
    for sent in sentences:
        # Remove punctuation
        sent = sent.translate(str.maketrans("", "", string.punctuation))
        words = word_tokenize(sent)
        if len(words) > 0:
            processed.append(words)

    return processed


def build_vocabulary(
    texts: list[str],
    min_freq: int = 5,
    max_vocab: int = 50000,
) -> tuple[dict[str, int], Counter]:
    """
    Build vocabulary from texts.

    Args:
        texts: List of raw texts
        min_freq: Minimum word frequency
        max_vocab: Maximum vocabulary size

    Returns:
        Tuple of (word2idx dict, word counts)
    """
    word_counts: Counter = Counter()

    print("Building vocabulary...")
    for text in tqdm(texts, desc="Counting words"):
        sentences = preprocess_text(text)
        for sent in sentences:
            word_counts.update(sent)

    # Filter by frequency and limit size
    filtered_words = [
        word for word, count in word_counts.most_common(max_vocab)
        if count >= min_freq
    ]

    # Build word2idx (0 is padding, 1 is unknown)
    word2idx = {"<PAD>": 0, "<UNK>": 1}
    for idx, word in enumerate(filtered_words, start=2):
        word2idx[word] = idx

    print(f"Vocabulary size: {len(word2idx)} (from {len(word_counts)} unique words)")
    return word2idx, word_counts


def download_yelp_dataset(data_dir: Path, max_samples: Optional[int] = 10000) -> tuple[list[str], list[int]]:
    """
    Download Yelp review dataset.

    Args:
        data_dir: Directory to save data
        max_samples: Maximum number of samples (None for all)

    Returns:
        Tuple of (texts, labels)
    """
    if not HAS_DATASETS:
        raise ImportError("Please install datasets: pip install datasets")

    print("Loading Yelp dataset from Hugging Face...")
    dataset = load_dataset("yelp_review_full", split="train")

    texts = []
    labels = []

    # Shuffle and limit samples
    indices = np.random.permutation(len(dataset))
    if max_samples is not None:
        indices = indices[:max_samples]

    for idx in tqdm(indices, desc="Loading samples"):
        item = dataset[int(idx)]
        texts.append(item["text"])
        labels.append(item["label"])  # 0-4 (5 classes)

    return texts, labels


def prepare_data(
    data_dir: str = "data",
    max_samples: int = 10000,
    embedding_dim: int = 50,
    min_freq: int = 5,
    max_vocab: int = 50000,
) -> None:
    """
    Main function to prepare all data.

    Args:
        data_dir: Base data directory
        max_samples: Maximum samples to use
        embedding_dim: Word embedding dimension
        min_freq: Minimum word frequency for vocabulary
        max_vocab: Maximum vocabulary size
    """
    if not HAS_NLTK:
        raise ImportError("Please install nltk: pip install nltk")

    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    processed_dir = data_dir / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    # Download and load dataset
    texts, labels = download_yelp_dataset(data_dir, max_samples)

    # Build vocabulary
    word2idx, _ = build_vocabulary(texts, min_freq, max_vocab)

    # Save vocabulary
    vocab_path = processed_dir / "word2idx.json"
    with open(vocab_path, "w") as f:
        json.dump(word2idx, f)
    print(f"Saved vocabulary to {vocab_path}")

    # Download and load GloVe embeddings
    glove_path = download_glove(data_dir, embedding_dim)
    embeddings = load_glove_embeddings(glove_path, word2idx, embedding_dim)

    # Save embeddings
    embeddings_path = processed_dir / f"embeddings_{embedding_dim}d.npz"
    np.savez_compressed(embeddings_path, embeddings=embeddings)
    print(f"Saved embeddings to {embeddings_path}")

    # Process and save data
    print("Processing documents...")
    processed_data = []
    for text, label in tqdm(zip(texts, labels), total=len(texts), desc="Processing"):
        sentences = preprocess_text(text)
        # Convert to indices
        doc_indices = []
        for sent in sentences:
            sent_indices = [word2idx.get(w, word2idx["<UNK>"]) for w in sent]
            if len(sent_indices) > 0:
                doc_indices.append(sent_indices)
        if len(doc_indices) > 0:
            processed_data.append({
                "document": doc_indices,
                "label": label,
            })

    # Save processed data
    data_path = processed_dir / "yelp_processed.npz"
    np.savez_compressed(
        data_path,
        data=np.array(processed_data, dtype=object),
    )
    print(f"Saved {len(processed_data)} processed documents to {data_path}")

    print("\nData preparation complete!")
    print(f"  Vocabulary size: {len(word2idx)}")
    print(f"  Embedding dim: {embedding_dim}")
    print(f"  Total samples: {len(processed_data)}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download and prepare Yelp dataset")
    parser.add_argument("--data-dir", type=str, default="data", help="Data directory")
    parser.add_argument("--max-samples", type=int, default=10000, help="Max samples")
    parser.add_argument("--embedding-dim", type=int, default=50, help="Embedding dimension")
    parser.add_argument("--min-freq", type=int, default=5, help="Min word frequency")
    parser.add_argument("--max-vocab", type=int, default=50000, help="Max vocabulary size")

    args = parser.parse_args()

    prepare_data(
        data_dir=args.data_dir,
        max_samples=args.max_samples,
        embedding_dim=args.embedding_dim,
        min_freq=args.min_freq,
        max_vocab=args.max_vocab,
    )
