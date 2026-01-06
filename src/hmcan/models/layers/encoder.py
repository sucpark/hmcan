"""Encoder blocks for hierarchical models."""

import torch
import torch.nn as nn

from .attention import ScaledDotProductAttention


class HMCANEncoderBlock(nn.Module):
    """
    HMCAN Encoder Block.

    Structure:
        Input -> Conv1D(Q,K,V) -> Self-Attention -> Residual + LayerNorm
              -> Dense -> Residual + LayerNorm -> Output

    This follows the original HMCAN paper architecture.
    """

    def __init__(
        self,
        input_dim: int,
        attention_dim: int,
        conv_kernel_size: int = 3,
        dropout: float = 0.1,
    ) -> None:
        """
        Initialize encoder block.

        Args:
            input_dim: Input dimension
            attention_dim: Attention/output dimension
            conv_kernel_size: Convolution kernel size
            dropout: Dropout probability
        """
        super().__init__()

        self.attention_dim = attention_dim

        # Conv1D projections for Q, K, V with 'same' padding
        padding = conv_kernel_size // 2
        self.conv_q = nn.Conv1d(input_dim, attention_dim, conv_kernel_size, padding=padding)
        self.conv_k = nn.Conv1d(input_dim, attention_dim, conv_kernel_size, padding=padding)
        self.conv_v = nn.Conv1d(input_dim, attention_dim, conv_kernel_size, padding=padding)

        # Scaled dot-product attention
        self.attention = ScaledDotProductAttention(dropout)

        # Layer normalizations
        self.layer_norm1 = nn.LayerNorm(attention_dim)
        self.layer_norm2 = nn.LayerNorm(attention_dim)

        # Feed-forward projection
        self.dense = nn.Linear(attention_dim, attention_dim)

        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()

        # Projection for residual if dimensions differ
        if input_dim != attention_dim:
            self.input_projection = nn.Linear(input_dim, attention_dim)
        else:
            self.input_projection = None

        # Initialize with He initialization
        for conv in [self.conv_q, self.conv_k, self.conv_v]:
            nn.init.kaiming_normal_(conv.weight)
            if conv.bias is not None:
                nn.init.zeros_(conv.bias)
        nn.init.xavier_normal_(self.dense.weight)
        nn.init.zeros_(self.dense.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor (batch, seq_len, input_dim)

        Returns:
            Output tensor (batch, seq_len, attention_dim)
        """
        # Transpose for Conv1d: (batch, seq_len, dim) -> (batch, dim, seq_len)
        x_t = x.transpose(1, 2)

        # Conv1D projections with ReLU
        Q = self.activation(self.conv_q(x_t)).transpose(1, 2)  # (batch, seq_len, attention_dim)
        K = self.activation(self.conv_k(x_t)).transpose(1, 2)
        V = self.activation(self.conv_v(x_t)).transpose(1, 2)

        # Self-attention
        attn_output, _ = self.attention(Q, K, V)

        # First residual connection
        if self.input_projection is not None:
            residual = self.input_projection(x)
        else:
            residual = x
        output = self.layer_norm1(attn_output + residual)

        # Feed-forward with second residual
        ff_output = self.dense(output)
        output = self.layer_norm2(ff_output + output)

        return output


class HCANEncoderBlock(nn.Module):
    """
    HCAN Encoder Block with cascaded multi-head attention.

    Structure:
        Input + Position -> Cascaded Multi-Head Attention -> Output
    """

    def __init__(
        self,
        attention_dim: int,
        num_heads: int = 5,
        dropout: float = 0.1,
        activation: str = "elu",
    ) -> None:
        """
        Initialize encoder block.

        Args:
            attention_dim: Attention dimension
            num_heads: Number of attention heads
            dropout: Dropout probability
            activation: Activation function
        """
        super().__init__()

        from .attention import CascadedMultiHeadAttention

        self.attention = CascadedMultiHeadAttention(
            attention_dim=attention_dim,
            num_heads=num_heads,
            dropout=dropout,
            activation=activation,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor (batch, seq_len, attention_dim)

        Returns:
            Output tensor (batch, seq_len, attention_dim)
        """
        return self.attention(x)
