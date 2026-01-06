"""
Hierarchical Attention Network (HAN) implementation.

Reference: Yang et al., "Hierarchical Attention Networks for Document Classification"
"""

from typing import Dict, Optional, Any
import torch
import torch.nn as nn

from .base import BaseHierarchicalModel
from .layers.embeddings import StandardEmbedding
from .layers.attention import AdditiveAttention


class HAN(BaseHierarchicalModel):
    """
    Hierarchical Attention Network for document classification.

    Architecture:
        Word Embedding -> BiGRU -> Word Attention -> Sentence Embedding
        -> BiGRU -> Sentence Attention -> Document Embedding -> Classifier

    This is the baseline model that HCAN and HMCAN improve upon.
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 50,
        hidden_dim: int = 50,
        attention_dim: int = 50,
        num_classes: int = 5,
        pretrained_embeddings: Optional[torch.Tensor] = None,
        freeze_embeddings: bool = False,
        dropout: float = 0.1,
        bidirectional: bool = True,
    ) -> None:
        """
        Initialize HAN model.

        Args:
            vocab_size: Size of vocabulary
            embedding_dim: Word embedding dimension
            hidden_dim: GRU hidden dimension
            attention_dim: Attention context dimension
            num_classes: Number of output classes
            pretrained_embeddings: Optional pretrained embedding matrix
            freeze_embeddings: Whether to freeze word embeddings
            dropout: Dropout probability
            bidirectional: Whether to use bidirectional GRU
        """
        super().__init__(vocab_size, embedding_dim, num_classes, pretrained_embeddings, dropout)

        self.hidden_dim = hidden_dim
        self.attention_dim = attention_dim
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        # Word embedding
        self.word_embedding = StandardEmbedding(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            pretrained_embeddings=pretrained_embeddings,
            freeze=freeze_embeddings,
            padding_idx=0,
        )

        # Word-level encoder
        self.word_gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            bidirectional=bidirectional,
            batch_first=True,
        )
        self.word_attention = AdditiveAttention(
            input_dim=hidden_dim * self.num_directions,
            attention_dim=attention_dim,
        )

        # Sentence-level encoder
        self.sentence_gru = nn.GRU(
            input_size=hidden_dim * self.num_directions,
            hidden_size=hidden_dim,
            bidirectional=bidirectional,
            batch_first=True,
        )
        self.sentence_attention = AdditiveAttention(
            input_dim=hidden_dim * self.num_directions,
            attention_dim=attention_dim,
        )

        # Classifier
        self.classifier = nn.Linear(hidden_dim * self.num_directions, num_classes)
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

        # Word embedding: (num_sents, max_words) -> (num_sents, max_words, embed_dim)
        word_embeds = self.dropout(self.word_embedding(document))

        # Word-level GRU encoding
        # (num_sents, max_words, hidden*2)
        word_outputs, _ = self.word_gru(word_embeds)

        # Word attention -> sentence embeddings
        # (num_sents, hidden*2)
        sentence_embeds, word_attn = self.word_attention(word_outputs, sentence_lengths)

        # Add batch dimension for sentence-level: (1, num_sents, hidden*2)
        sentence_embeds = sentence_embeds.unsqueeze(0)
        num_sents_tensor = torch.tensor(
            [num_sentences], device=document.device, dtype=torch.long
        )

        # Sentence-level GRU encoding
        # (1, num_sents, hidden*2)
        sentence_outputs, _ = self.sentence_gru(sentence_embeds)

        # Sentence attention -> document embedding
        # (1, hidden*2)
        doc_embed, sent_attn = self.sentence_attention(sentence_outputs, num_sents_tensor)

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
            num_sentences = document.size(0)

            word_embeds = self.word_embedding(document)
            word_outputs, _ = self.word_gru(word_embeds)
            sentence_embeds, _ = self.word_attention(word_outputs, sentence_lengths)

            sentence_embeds = sentence_embeds.unsqueeze(0)
            num_sents_tensor = torch.tensor(
                [num_sentences], device=document.device, dtype=torch.long
            )

            sentence_outputs, _ = self.sentence_gru(sentence_embeds)
            doc_embed, _ = self.sentence_attention(sentence_outputs, num_sents_tensor)

        return doc_embed

    def get_config(self) -> Dict[str, Any]:
        """Get model configuration."""
        config = super().get_config()
        config.update({
            "hidden_dim": self.hidden_dim,
            "attention_dim": self.attention_dim,
            "bidirectional": self.bidirectional,
        })
        return config
