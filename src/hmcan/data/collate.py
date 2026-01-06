"""Custom collate functions for hierarchical document batching."""

from typing import Any
import torch


def hierarchical_collate_fn(batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    """
    Collate function for hierarchical documents.

    Since documents have variable numbers of sentences and words,
    this function processes one document at a time (original paper approach).

    For batch_size > 1, use batched_hierarchical_collate_fn instead.

    Args:
        batch: List of document dictionaries from dataset

    Returns:
        Dictionary with:
            - document: Tensor of shape (num_sentences, max_words)
            - sentence_lengths: Tensor of shape (num_sentences,)
            - label: Scalar tensor
            - num_sentences: Scalar tensor
    """
    # Process single document
    item = batch[0]
    document = item["document"]
    sentence_lengths = item["sentence_lengths"]

    # Pad sentences to max length within this document
    max_words = max(sentence_lengths) if sentence_lengths else 1
    num_sentences = len(document)

    # Create padded tensor
    padded_document = torch.zeros(num_sentences, max_words, dtype=torch.long)
    for i, sentence in enumerate(document):
        padded_document[i, : len(sentence)] = torch.tensor(sentence, dtype=torch.long)

    return {
        "document": padded_document,  # (num_sentences, max_words)
        "sentence_lengths": torch.tensor(sentence_lengths, dtype=torch.long),
        "label": torch.tensor(item["label"], dtype=torch.long),
        "num_sentences": torch.tensor(num_sentences, dtype=torch.long),
    }


def batched_hierarchical_collate_fn(
    batch: list[dict[str, Any]],
) -> dict[str, torch.Tensor]:
    """
    Collate function for batched hierarchical documents.

    Pads all documents to (batch_size, max_sentences, max_words).
    More efficient for training but uses more memory.

    Args:
        batch: List of document dictionaries from dataset

    Returns:
        Dictionary with:
            - documents: Tensor of shape (batch_size, max_sentences, max_words)
            - sentence_lengths: Tensor of shape (batch_size, max_sentences)
            - labels: Tensor of shape (batch_size,)
            - num_sentences: Tensor of shape (batch_size,)
    """
    batch_size = len(batch)

    # Find max dimensions across batch
    max_sentences = max(item["num_sentences"] for item in batch)
    max_words = max(
        max(item["sentence_lengths"]) if item["sentence_lengths"] else 1
        for item in batch
    )

    # Initialize padded tensors
    documents = torch.zeros(batch_size, max_sentences, max_words, dtype=torch.long)
    sentence_lengths = torch.zeros(batch_size, max_sentences, dtype=torch.long)
    labels = torch.zeros(batch_size, dtype=torch.long)
    num_sentences = torch.zeros(batch_size, dtype=torch.long)

    for i, item in enumerate(batch):
        doc = item["document"]
        for j, sentence in enumerate(doc):
            documents[i, j, : len(sentence)] = torch.tensor(sentence, dtype=torch.long)
            sentence_lengths[i, j] = len(sentence)
        labels[i] = item["label"]
        num_sentences[i] = item["num_sentences"]

    return {
        "documents": documents,
        "sentence_lengths": sentence_lengths,
        "labels": labels,
        "num_sentences": num_sentences,
    }


def create_attention_mask(
    sentence_lengths: torch.Tensor,
    max_len: int,
) -> torch.Tensor:
    """
    Create attention mask for variable-length sequences.

    Args:
        sentence_lengths: Tensor of actual lengths (batch,) or (num_sentences,)
        max_len: Maximum sequence length

    Returns:
        Boolean mask tensor where True = valid position
    """
    batch_size = sentence_lengths.size(0)
    mask = torch.arange(max_len, device=sentence_lengths.device).expand(
        batch_size, max_len
    ) < sentence_lengths.unsqueeze(1)
    return mask
