"""Attention mechanism implementations."""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class AdditiveAttention(nn.Module):
    """
    Additive (Bahdanau) attention for HAN.

    Computes:
        u = tanh(W @ h + b)
        score = exp(w @ u)
        alpha = score / sum(scores)  (with masking)
        output = sum(alpha * h)
    """

    def __init__(
        self,
        input_dim: int,
        attention_dim: int,
    ) -> None:
        """
        Initialize additive attention.

        Args:
            input_dim: Input hidden dimension
            attention_dim: Attention context dimension
        """
        super().__init__()
        self.W = nn.Linear(input_dim, attention_dim)
        self.w = nn.Linear(attention_dim, 1, bias=False)

        # Xavier initialization
        nn.init.xavier_normal_(self.W.weight)
        nn.init.xavier_normal_(self.w.weight)

    def forward(
        self,
        hidden_states: torch.Tensor,
        lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            hidden_states: (batch, seq_len, hidden_dim)
            lengths: (batch,) actual lengths for masking

        Returns:
            context: (batch, hidden_dim) weighted sum
            attention_weights: (batch, seq_len)
        """
        # (batch, seq_len, attention_dim)
        u = torch.tanh(self.W(hidden_states))

        # (batch, seq_len, 1) -> (batch, seq_len)
        scores = self.w(u).squeeze(-1)

        # Create mask for padding positions
        max_len = hidden_states.size(1)
        mask = torch.arange(max_len, device=lengths.device).expand(
            len(lengths), max_len
        ) < lengths.unsqueeze(1)

        # Mask padding positions with -inf before softmax
        scores = scores.masked_fill(~mask, float("-inf"))

        # Softmax over sequence
        attention_weights = F.softmax(scores, dim=-1)

        # Handle all-masked case (set to uniform)
        attention_weights = attention_weights.masked_fill(
            attention_weights.isnan(), 0.0
        )

        # Weighted sum: (batch, 1, seq_len) @ (batch, seq_len, hidden)
        context = torch.bmm(attention_weights.unsqueeze(1), hidden_states).squeeze(1)

        return context, attention_weights


class ScaledDotProductAttention(nn.Module):
    """
    Scaled dot-product attention for HMCAN.

    Computes: softmax(Q @ K^T / sqrt(d_k)) @ V
    """

    def __init__(self, dropout: float = 0.1) -> None:
        """
        Initialize attention.

        Args:
            dropout: Dropout probability
        """
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            Q: Query tensor (batch, seq_len, d_k)
            K: Key tensor (batch, seq_len, d_k)
            V: Value tensor (batch, seq_len, d_v)
            mask: Optional attention mask

        Returns:
            output: (batch, seq_len, d_v)
            attention_weights: (batch, seq_len, seq_len)
        """
        d_k = K.size(-1)

        # (batch, seq_len, seq_len)
        scores = torch.bmm(Q, K.transpose(1, 2)) / (d_k**0.5)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attention_weights = self.dropout(F.softmax(scores, dim=-1))
        output = torch.bmm(attention_weights, V)

        return output, attention_weights


class TargetAttention(nn.Module):
    """
    Target attention for aggregation in HMCAN.

    Uses a learnable target vector to compute attention over a sequence,
    aggregating it to a single vector.

    Used for:
        - Words -> Sentence representation
        - Sentences -> Document representation
    """

    def __init__(
        self,
        attention_dim: int,
        dropout: float = 0.1,
    ) -> None:
        """
        Initialize target attention.

        Args:
            attention_dim: Dimension of attention
            dropout: Dropout probability
        """
        super().__init__()
        self.attention_dim = attention_dim
        self.target = nn.Parameter(torch.randn(1, 1, attention_dim))
        self.dropout = nn.Dropout(dropout)

        # He initialization for target vector
        nn.init.kaiming_normal_(self.target)

    def forward(
        self,
        hidden_states: torch.Tensor,
        expand_target: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            hidden_states: (batch, seq_len, attention_dim)
            expand_target: Whether to expand target to batch size

        Returns:
            output: (batch, attention_dim)
            attention_weights: (batch, seq_len)
        """
        batch_size = hidden_states.size(0)

        if expand_target:
            target = self.target.expand(batch_size, -1, -1)
        else:
            target = self.target

        # (batch, 1, attention_dim) @ (batch, attention_dim, seq_len)
        # -> (batch, 1, seq_len)
        scores = torch.bmm(target, hidden_states.transpose(1, 2))
        scores = scores / (self.attention_dim**0.5)
        attention_weights = self.dropout(F.softmax(scores, dim=-1))

        # (batch, 1, seq_len) @ (batch, seq_len, attention_dim)
        # -> (batch, 1, attention_dim) -> (batch, attention_dim)
        output = torch.bmm(attention_weights, hidden_states).squeeze(1)

        return output, attention_weights.squeeze(1)


class CascadedMultiHeadAttention(nn.Module):
    """
    Cascaded multi-head attention for HCAN.

    Uses two parallel attention branches with different activations,
    then combines them via element-wise multiplication.

    Branch 1: ELU activation for Q, K, V
    Branch 2: ELU for Q, K; Tanh for V
    """

    def __init__(
        self,
        attention_dim: int,
        num_heads: int = 5,
        dropout: float = 0.1,
        activation: str = "elu",
    ) -> None:
        """
        Initialize cascaded attention.

        Args:
            attention_dim: Dimension of attention
            num_heads: Number of attention heads
            dropout: Dropout probability
            activation: Activation function (elu, relu)
        """
        super().__init__()
        self.attention_dim = attention_dim
        self.num_heads = num_heads
        self.head_dim = attention_dim // num_heads

        assert attention_dim % num_heads == 0, "attention_dim must be divisible by num_heads"

        # Branch 1: Q, K, V with same activation
        self.conv_q1 = nn.Conv1d(attention_dim, attention_dim, kernel_size=3, padding=1)
        self.conv_k1 = nn.Conv1d(attention_dim, attention_dim, kernel_size=3, padding=1)
        self.conv_v1 = nn.Conv1d(attention_dim, attention_dim, kernel_size=3, padding=1)

        # Branch 2: Q, K with ELU, V with Tanh
        self.conv_q2 = nn.Conv1d(attention_dim, attention_dim, kernel_size=3, padding=1)
        self.conv_k2 = nn.Conv1d(attention_dim, attention_dim, kernel_size=3, padding=1)
        self.conv_v2 = nn.Conv1d(attention_dim, attention_dim, kernel_size=3, padding=1)

        # Activations
        if activation == "elu":
            self.activation = nn.ELU()
        else:
            self.activation = nn.ReLU()

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(attention_dim)

        # Initialize
        for conv in [self.conv_q1, self.conv_k1, self.conv_v1,
                     self.conv_q2, self.conv_k2, self.conv_v2]:
            nn.init.kaiming_normal_(conv.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor (batch, seq_len, attention_dim)

        Returns:
            Output tensor (batch, seq_len, attention_dim)
        """
        batch_size, seq_len, _ = x.shape

        # Transpose for Conv1d: (batch, dim, seq_len)
        x_t = x.transpose(1, 2)

        # Branch 1
        Q1 = self.activation(self.conv_q1(x_t)).transpose(1, 2)
        K1 = self.activation(self.conv_k1(x_t)).transpose(1, 2)
        V1 = self.activation(self.conv_v1(x_t)).transpose(1, 2)

        # Branch 2
        Q2 = self.activation(self.conv_q2(x_t)).transpose(1, 2)
        K2 = self.activation(self.conv_k2(x_t)).transpose(1, 2)
        V2 = torch.tanh(self.conv_v2(x_t)).transpose(1, 2)

        # Multi-head attention for branch 1
        output1 = self._multi_head_attention(Q1, K1, V1, batch_size, seq_len)

        # Multi-head attention for branch 2
        output2 = self._multi_head_attention(Q2, K2, V2, batch_size, seq_len)

        # Element-wise multiplication
        output = output1 * output2

        # Layer norm
        output = self.layer_norm(output)

        return output

    def _multi_head_attention(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        batch_size: int,
        seq_len: int,
    ) -> torch.Tensor:
        """Compute multi-head attention."""
        # Reshape for multi-head: (batch, seq_len, num_heads, head_dim)
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim)

        # Transpose: (batch, num_heads, seq_len, head_dim)
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        # Attention: (batch, num_heads, seq_len, seq_len)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim**0.5)
        attn_weights = self.dropout(F.softmax(scores, dim=-1))

        # Output: (batch, num_heads, seq_len, head_dim)
        output = torch.matmul(attn_weights, V)

        # Reshape back: (batch, seq_len, attention_dim)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)

        return output
