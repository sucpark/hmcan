"""
Hierarchical Cascaded Attention Network (HCAN) implementation.

HCAN extends HAN with:
    - Positional embeddings
    - Multi-head self-attention with cascaded structure
    - Target attention for aggregation
"""

from typing import Dict, Optional, Any
import torch
import torch.nn as nn

from .base import BaseHierarchicalModel
from .layers.embeddings import StandardEmbedding, PositionalEmbedding
from .layers.attention import CascadedMultiHeadAttention, TargetAttention


class HCAN(BaseHierarchicalModel):
    """
    Hierarchical Cascaded Attention Network for document classification.

    Architecture:
        Word Embedding + Positional -> Cascaded Multi-Head Self-Attention
        -> Target Attention -> Sentence Embedding + Positional
        -> Cascaded Multi-Head Self-Attention -> Target Attention -> Classifier

    Key features:
        - Learnable positional embeddings at both word and sentence levels
        - Cascaded multi-head attention (two branches with element-wise multiply)
        - Target attention for aggregation
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 50,
        attention_dim: int = 50,
        num_heads: int = 5,
        max_words: int = 200,
        max_sentences: int = 50,
        num_classes: int = 5,
        pretrained_embeddings: Optional[torch.Tensor] = None,
        dropout: float = 0.1,
        activation: str = "elu",
    ) -> None:
        """
        Initialize HCAN model.

        Args:
            vocab_size: Size of vocabulary
            embedding_dim: Word embedding dimension
            attention_dim: Attention dimension
            num_heads: Number of attention heads
            max_words: Maximum words per sentence
            max_sentences: Maximum sentences per document
            num_classes: Number of output classes
            pretrained_embeddings: Optional pretrained embedding matrix
            dropout: Dropout probability
            activation: Activation function (elu, relu)
        """
        super().__init__(vocab_size, embedding_dim, num_classes, pretrained_embeddings, dropout)

        self.attention_dim = attention_dim
        self.num_heads = num_heads
        self.max_words = max_words
        self.max_sentences = max_sentences

        # Word embedding
        self.word_embedding = StandardEmbedding(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            pretrained_embeddings=pretrained_embeddings,
            freeze=False,
            padding_idx=0,
        )

        # Positional embeddings
        self.word_position = PositionalEmbedding(max_words, attention_dim)
        self.sentence_position = PositionalEmbedding(max_sentences, attention_dim)

        # Projection if embedding_dim != attention_dim
        if embedding_dim != attention_dim:
            self.embed_projection = nn.Linear(embedding_dim, attention_dim)
        else:
            self.embed_projection = None

        # Word-level cascaded attention
        self.word_attention = CascadedMultiHeadAttention(
            attention_dim=attention_dim,
            num_heads=num_heads,
            dropout=dropout,
            activation=activation,
        )
        self.word_target_attention = TargetAttention(
            attention_dim=attention_dim,
            dropout=dropout,
        )

        # Sentence-level cascaded attention
        self.sentence_attention = CascadedMultiHeadAttention(
            attention_dim=attention_dim,
            num_heads=num_heads,
            dropout=dropout,
            activation=activation,
        )
        self.sentence_target_attention = TargetAttention(
            attention_dim=attention_dim,
            dropout=dropout,
        )

        # Classifier
        self.classifier = nn.Linear(attention_dim, num_classes)
        self.dropout = nn.Dropout(dropout)

        # Initialize
        nn.init.xavier_normal_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def forward(
        self,
        document: torch.Tensor,
        sentence_lengths: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            document: Word indices (num_sentences, max_words)
            sentence_lengths: Words per sentence (num_sentences,)

        Returns:
            Dict with logits and attention weights
        """
        num_sentences, max_words = document.shape

        # Word embedding
        word_embeds = self.word_embedding(document)

        # Project if needed
        if self.embed_projection is not None:
            word_embeds = self.embed_projection(word_embeds)

        # Add positional embedding
        word_embeds = self.dropout(
            word_embeds + self.word_position(max_words)
        )

        # Word self-attention (cascaded)
        word_outputs = self.word_attention(word_embeds)

        # Word target attention -> sentence embeddings
        # (num_sentences, attention_dim)
        sent_embeds, word_attn = self.word_target_attention(word_outputs)

        # Add batch and positional for sentence level
        # (1, num_sentences, attention_dim)
        sent_embeds = sent_embeds.unsqueeze(0)
        sent_embeds = self.dropout(
            sent_embeds + self.sentence_position(num_sentences)
        )

        # Sentence self-attention (cascaded)
        sent_outputs = self.sentence_attention(sent_embeds)

        # Sentence target attention -> document embedding
        # (1, attention_dim)
        doc_embed, sent_attn = self.sentence_target_attention(sent_outputs, expand_target=False)

        # Classification
        logits = self.classifier(self.dropout(doc_embed))

        return {
            "logits": logits,
            "word_attention": word_attn,
            "sentence_attention": sent_attn,
        }

    def get_document_embedding(
        self,
        document: torch.Tensor,
        sentence_lengths: torch.Tensor,
    ) -> torch.Tensor:
        """Get document representation before classifier."""
        with torch.no_grad():
            num_sentences, max_words = document.shape

            word_embeds = self.word_embedding(document)
            if self.embed_projection is not None:
                word_embeds = self.embed_projection(word_embeds)
            word_embeds = word_embeds + self.word_position(max_words)

            word_outputs = self.word_attention(word_embeds)
            sent_embeds, _ = self.word_target_attention(word_outputs)

            sent_embeds = sent_embeds.unsqueeze(0)
            sent_embeds = sent_embeds + self.sentence_position(num_sentences)

            sent_outputs = self.sentence_attention(sent_embeds)
            doc_embed, _ = self.sentence_target_attention(sent_outputs, expand_target=False)

        return doc_embed

    def get_config(self) -> Dict[str, Any]:
        """Get model configuration."""
        config = super().get_config()
        config.update({
            "attention_dim": self.attention_dim,
            "num_heads": self.num_heads,
            "max_words": self.max_words,
            "max_sentences": self.max_sentences,
        })
        return config
