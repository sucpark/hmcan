"""
Hierarchical Multichannel CNN-based Attention Network (HMCAN) implementation.

HMCAN is the main model with:
    - Dual embeddings (pretrained + learnable) as multichannel input
    - Conv1D Q,K,V projections
    - Self-attention with residual connections and layer normalization
    - Target attention for hierarchical aggregation
"""

from typing import Dict, Optional, Any
import torch
import torch.nn as nn

from .base import BaseHierarchicalModel
from .layers.embeddings import DualEmbedding
from .layers.encoder import HMCANEncoderBlock


class HMCAN(BaseHierarchicalModel):
    """
    Hierarchical Multichannel CNN-based Attention Network for document classification.

    Architecture:
        Dual Word Embedding (pretrained frozen + learnable) as multichannel input
        -> Conv1D(Q,K,V) -> Self-Attention -> Residual + LayerNorm
        -> Dense -> Residual + LayerNorm -> Target Attention (Tw)
        -> Conv1D(Q,K,V) -> Self-Attention -> Residual + LayerNorm
        -> Dense -> Residual + LayerNorm -> Target Attention (Ts)
        -> Dense Classifier

    Key features:
        - Multichannel embeddings: pretrained (frozen) + learnable, summed together
        - Conv1D projections for Q, K, V (kernel=3, same padding, ReLU)
        - Scaled dot-product self-attention
        - Residual connections + Layer normalization
        - Learnable target vectors (Tw, Ts) for aggregation
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 50,
        attention_dim: int = 50,
        num_classes: int = 5,
        pretrained_embeddings: Optional[torch.Tensor] = None,
        freeze_pretrained: bool = True,
        conv_kernel_size: int = 3,
        dropout: float = 0.1,
    ) -> None:
        """
        Initialize HMCAN model.

        Args:
            vocab_size: Size of vocabulary
            embedding_dim: Word embedding dimension
            attention_dim: Attention dimension
            num_classes: Number of output classes
            pretrained_embeddings: Optional pretrained embedding matrix
            freeze_pretrained: Whether to freeze pretrained embeddings
            conv_kernel_size: Convolution kernel size
            dropout: Dropout probability
        """
        super().__init__(vocab_size, embedding_dim, num_classes, pretrained_embeddings, dropout)

        self.attention_dim = attention_dim
        self.conv_kernel_size = conv_kernel_size

        # Dual word embeddings (pretrained + learnable)
        self.embedding = DualEmbedding(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            pretrained_embeddings=pretrained_embeddings,
            freeze_pretrained=freeze_pretrained,
            padding_idx=0,
        )

        # Word-level encoder
        self.word_encoder = HMCANEncoderBlock(
            input_dim=embedding_dim,
            attention_dim=attention_dim,
            conv_kernel_size=conv_kernel_size,
            dropout=dropout,
        )

        # Word target attention (learnable target vector Tw)
        self.word_target = nn.Parameter(torch.randn(1, 1, attention_dim))
        nn.init.kaiming_normal_(self.word_target)

        # Sentence-level encoder
        self.sentence_encoder = HMCANEncoderBlock(
            input_dim=attention_dim,
            attention_dim=attention_dim,
            conv_kernel_size=conv_kernel_size,
            dropout=dropout,
        )

        # Sentence target attention (learnable target vector Ts)
        self.sentence_target = nn.Parameter(torch.randn(1, 1, attention_dim))
        nn.init.kaiming_normal_(self.sentence_target)

        # Classification head
        self.classifier = nn.Linear(attention_dim, num_classes)
        self.dropout = nn.Dropout(dropout)

        # Initialize classifier
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
        num_sentences = document.size(0)

        # 1. Dual word embedding
        # (num_sents, max_words) -> (num_sents, max_words, embed_dim)
        word_embeds = self.dropout(self.embedding(document))

        # 2. Word-level encoding
        # (num_sents, max_words, attention_dim)
        word_encoded = self.word_encoder(word_embeds)

        # 3. Word target attention -> sentence embeddings
        # Expand target: (1, 1, attention_dim) -> (num_sents, 1, attention_dim)
        Tw = self.word_target.expand(num_sentences, -1, -1)

        # Attention scores: (num_sents, 1, attention_dim) @ (num_sents, attention_dim, max_words)
        # -> (num_sents, 1, max_words)
        word_attn_scores = torch.bmm(Tw, word_encoded.transpose(1, 2))
        word_attn_scores = word_attn_scores / (self.attention_dim**0.5)
        word_attn_weights = self.dropout(torch.softmax(word_attn_scores, dim=-1))

        # Weighted sum: (num_sents, 1, max_words) @ (num_sents, max_words, attention_dim)
        # -> (num_sents, 1, attention_dim)
        sent_embeds = torch.bmm(word_attn_weights, word_encoded)

        # Reshape for sentence level: (num_sents, 1, dim) -> (1, num_sents, dim)
        sent_embeds = sent_embeds.transpose(0, 1)

        # 4. Sentence-level encoding
        # (1, num_sents, attention_dim)
        sent_encoded = self.sentence_encoder(sent_embeds)

        # 5. Sentence target attention -> document embedding
        # (1, 1, attention_dim) @ (1, attention_dim, num_sents) -> (1, 1, num_sents)
        sent_attn_scores = torch.bmm(self.sentence_target, sent_encoded.transpose(1, 2))
        sent_attn_scores = sent_attn_scores / (self.attention_dim**0.5)
        sent_attn_weights = self.dropout(torch.softmax(sent_attn_scores, dim=-1))

        # Weighted sum: (1, 1, num_sents) @ (1, num_sents, attention_dim)
        # -> (1, 1, attention_dim) -> (1, attention_dim)
        doc_embed = torch.bmm(sent_attn_weights, sent_encoded).squeeze(1)

        # 6. Classification
        logits = self.classifier(self.dropout(doc_embed))

        return {
            "logits": logits,
            "word_attention": word_attn_weights.squeeze(1),  # (num_sents, max_words)
            "sentence_attention": sent_attn_weights.squeeze(0),  # (1, num_sents)
        }

    def get_document_embedding(
        self,
        document: torch.Tensor,
        sentence_lengths: torch.Tensor,
    ) -> torch.Tensor:
        """Get document representation before classifier."""
        with torch.no_grad():
            num_sentences = document.size(0)

            word_embeds = self.embedding(document)
            word_encoded = self.word_encoder(word_embeds)

            Tw = self.word_target.expand(num_sentences, -1, -1)
            word_attn = torch.softmax(
                torch.bmm(Tw, word_encoded.transpose(1, 2)) / (self.attention_dim**0.5),
                dim=-1,
            )
            sent_embeds = torch.bmm(word_attn, word_encoded).transpose(0, 1)

            sent_encoded = self.sentence_encoder(sent_embeds)
            sent_attn = torch.softmax(
                torch.bmm(self.sentence_target, sent_encoded.transpose(1, 2))
                / (self.attention_dim**0.5),
                dim=-1,
            )
            doc_embed = torch.bmm(sent_attn, sent_encoded).squeeze(1)

        return doc_embed

    def get_config(self) -> Dict[str, Any]:
        """Get model configuration."""
        config = super().get_config()
        config.update({
            "attention_dim": self.attention_dim,
            "conv_kernel_size": self.conv_kernel_size,
        })
        return config
