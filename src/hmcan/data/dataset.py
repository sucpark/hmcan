"""PyTorch Dataset for hierarchical document classification."""

from typing import Any, Optional
import torch
from torch.utils.data import Dataset


class HierarchicalDocumentDataset(Dataset):
    """
    PyTorch Dataset for hierarchical document classification.

    Each document is represented as a list of sentences,
    where each sentence is a list of word indices.

    Args:
        documents: List of documents, each is list[list[int]] (sentences of word indices)
        labels: List of integer labels
        max_sentences: Optional maximum sentences per document
        max_words: Optional maximum words per sentence
    """

    def __init__(
        self,
        documents: list[list[list[int]]],
        labels: list[int],
        max_sentences: Optional[int] = None,
        max_words: Optional[int] = None,
    ) -> None:
        assert len(documents) == len(labels), "Documents and labels must have same length"

        self.documents = documents
        self.labels = labels
        self.max_sentences = max_sentences
        self.max_words = max_words

    def __len__(self) -> int:
        """Return number of documents."""
        return len(self.documents)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """
        Get a single document.

        Args:
            idx: Document index

        Returns:
            Dictionary with:
                - document: List of sentences (list of word indices)
                - sentence_lengths: List of word counts per sentence
                - label: Integer label
                - num_sentences: Number of sentences
        """
        document = self.documents[idx]
        label = self.labels[idx]

        # Truncate sentences if needed
        if self.max_sentences is not None:
            document = document[: self.max_sentences]

        # Truncate words in each sentence if needed
        if self.max_words is not None:
            document = [sent[: self.max_words] for sent in document]

        # Compute sentence lengths (before padding)
        sentence_lengths = [len(sent) for sent in document]

        return {
            "document": document,
            "sentence_lengths": sentence_lengths,
            "label": label,
            "num_sentences": len(document),
        }

    @classmethod
    def from_processed_data(
        cls,
        data: list[dict[str, Any]],
        max_sentences: Optional[int] = None,
        max_words: Optional[int] = None,
    ) -> "HierarchicalDocumentDataset":
        """
        Create dataset from processed data.

        Args:
            data: List of dicts with 'document' and 'label' keys
            max_sentences: Maximum sentences per document
            max_words: Maximum words per sentence

        Returns:
            Dataset instance
        """
        documents = [item["document"] for item in data]
        labels = [item["label"] for item in data]
        return cls(documents, labels, max_sentences, max_words)

    def get_stats(self) -> dict[str, Any]:
        """
        Get dataset statistics.

        Returns:
            Dictionary with statistics
        """
        all_sent_lengths = []
        all_doc_lengths = []

        for doc in self.documents:
            all_doc_lengths.append(len(doc))
            for sent in doc:
                all_sent_lengths.append(len(sent))

        return {
            "num_documents": len(self.documents),
            "avg_sentences_per_doc": sum(all_doc_lengths) / len(all_doc_lengths),
            "max_sentences_per_doc": max(all_doc_lengths),
            "avg_words_per_sentence": sum(all_sent_lengths) / len(all_sent_lengths),
            "max_words_per_sentence": max(all_sent_lengths),
        }
