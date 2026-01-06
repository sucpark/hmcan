"""Base model class for hierarchical document classification."""

from abc import ABC, abstractmethod
from typing import Dict, Optional, Any
import torch
import torch.nn as nn


class BaseHierarchicalModel(nn.Module, ABC):
    """
    Abstract base class for hierarchical document classification models.

    All models (HAN, HCAN, HMCAN) inherit from this class.
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        num_classes: int,
        pretrained_embeddings: Optional[torch.Tensor] = None,
        dropout: float = 0.1,
    ) -> None:
        """
        Initialize base model.

        Args:
            vocab_size: Size of vocabulary
            embedding_dim: Word embedding dimension
            num_classes: Number of output classes
            pretrained_embeddings: Optional pretrained embedding matrix
            dropout: Dropout probability
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.dropout_rate = dropout
        self._pretrained_embeddings = pretrained_embeddings

    @abstractmethod
    def forward(
        self,
        document: torch.Tensor,
        sentence_lengths: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            document: Word indices of shape (num_sentences, max_words)
            sentence_lengths: Words per sentence of shape (num_sentences,)

        Returns:
            Dict with keys:
                - 'logits': Classification logits of shape (1, num_classes)
                - 'word_attention': Optional word attention weights
                - 'sentence_attention': Optional sentence attention weights
        """
        pass

    @abstractmethod
    def get_document_embedding(
        self,
        document: torch.Tensor,
        sentence_lengths: torch.Tensor,
    ) -> torch.Tensor:
        """
        Get document representation before classification.

        Args:
            document: Word indices
            sentence_lengths: Words per sentence

        Returns:
            Document embedding tensor
        """
        pass

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def count_all_parameters(self) -> int:
        """Count all parameters (including frozen)."""
        return sum(p.numel() for p in self.parameters())

    def get_config(self) -> Dict[str, Any]:
        """Get model configuration."""
        return {
            "vocab_size": self.vocab_size,
            "embedding_dim": self.embedding_dim,
            "num_classes": self.num_classes,
            "dropout": self.dropout_rate,
        }
