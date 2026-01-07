"""Embedding layers for hierarchical models."""

from typing import Optional
import torch
import torch.nn as nn


class DualEmbedding(nn.Module):
    """
    Dual word embeddings for HMCAN.

    Combines:
        1. Pre-trained embeddings (frozen by default)
        2. Learnable embeddings (Xavier initialized)

    Output is the element-wise sum of both embeddings.
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        pretrained_embeddings: Optional[torch.Tensor] = None,
        freeze_pretrained: bool = True,
        padding_idx: int = 0,
    ) -> None:
        """
        Initialize dual embeddings.

        Args:
            vocab_size: Size of vocabulary
            embedding_dim: Embedding dimension
            pretrained_embeddings: Optional pretrained embedding matrix
            freeze_pretrained: Whether to freeze pretrained embeddings
            padding_idx: Index of padding token
        """
        super().__init__()

        # Pre-trained embedding (frozen by default)
        self.pretrained = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        if pretrained_embeddings is not None:
            with torch.no_grad():
                self.pretrained.weight.copy_(pretrained_embeddings)
        if freeze_pretrained:
            self.pretrained.weight.requires_grad = False

        # Learnable embedding (Xavier initialized)
        self.learnable = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        nn.init.xavier_normal_(self.learnable.weight)
        # Reset padding to zero
        with torch.no_grad():
            self.learnable.weight[padding_idx].zero_()

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            input_ids: Token indices of shape (batch, seq_len) or (seq_len,)

        Returns:
            Embeddings of shape (*input_shape, embedding_dim)
        """
        pretrained_embeds = self.pretrained(input_ids)
        learnable_embeds = self.learnable(input_ids)
        return pretrained_embeds + learnable_embeds


class PositionalEmbedding(nn.Module):
    """
    Learnable positional embeddings for HCAN.

    Unlike sinusoidal positional encodings (Transformer),
    these are learned during training.
    """

    def __init__(
        self,
        max_position: int,
        embedding_dim: int,
    ) -> None:
        """
        Initialize positional embeddings.

        Args:
            max_position: Maximum sequence length
            embedding_dim: Embedding dimension
        """
        super().__init__()
        self.embedding = nn.Embedding(max_position, embedding_dim)
        nn.init.xavier_normal_(self.embedding.weight)

    def forward(self, seq_len: int) -> torch.Tensor:
        """
        Get positional embeddings.

        Args:
            seq_len: Sequence length

        Returns:
            Positional embeddings of shape (1, seq_len, embedding_dim)
        """
        positions = torch.arange(seq_len, device=self.embedding.weight.device)
        return self.embedding(positions).unsqueeze(0)

    def forward_batch(self, batch_size: int, seq_len: int) -> torch.Tensor:
        """
        Get positional embeddings for a batch.

        Args:
            batch_size: Batch size
            seq_len: Sequence length

        Returns:
            Positional embeddings of shape (batch_size, seq_len, embedding_dim)
        """
        return self.forward(seq_len).expand(batch_size, -1, -1)


class StandardEmbedding(nn.Module):
    """
    Standard word embedding with optional pretrained initialization.

    Used by HAN model.
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        pretrained_embeddings: Optional[torch.Tensor] = None,
        freeze: bool = False,
        padding_idx: int = 0,
    ) -> None:
        """
        Initialize standard embedding.

        Args:
            vocab_size: Size of vocabulary
            embedding_dim: Embedding dimension
            pretrained_embeddings: Optional pretrained embedding matrix
            freeze: Whether to freeze embeddings
            padding_idx: Index of padding token
        """
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)

        if pretrained_embeddings is not None:
            with torch.no_grad():
                self.embedding.weight.copy_(pretrained_embeddings)

        if freeze:
            self.embedding.weight.requires_grad = False

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            input_ids: Token indices

        Returns:
            Embeddings
        """
        return self.embedding(input_ids)
