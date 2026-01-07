# Model Architecture

## Overview

This project implements three attention-based models for hierarchical document classification:

| Model | Description | Key Features |
|-------|-------------|--------------|
| HAN | Hierarchical Attention Network | BiGRU + Additive Attention |
| HCAN | Hierarchical Cascaded Attention Network | Multi-head Self-Attention + Positional |
| HMCAN | Hierarchical Multichannel CNN-based Attention Network | Dual Embedding (Multichannel) + Conv1D + Target Attention |

## Common Structure

All models follow a 2-level hierarchical structure:

```
Document
    ↓
┌───────────────────────────────────┐
│        Word-Level Encoder         │
│   Words → Sentence Embedding      │
└───────────────────────────────────┘
    ↓
┌───────────────────────────────────┐
│      Sentence-Level Encoder       │
│   Sentences → Document Embedding  │
└───────────────────────────────────┘
    ↓
┌───────────────────────────────────┐
│          Classification           │
│   Document Embedding → 5-class    │
└───────────────────────────────────┘
```

---

## HAN (Hierarchical Attention Network)

### Architecture

```
Word Embedding
    ↓
Bidirectional GRU (Word)
    ↓
Additive Attention (Word)
    ↓
Sentence Embedding
    ↓
Bidirectional GRU (Sentence)
    ↓
Additive Attention (Sentence)
    ↓
Dense Classifier
```

### Key Components

#### 1. Bidirectional GRU
```python
BiGRU(input_dim=50, hidden_dim=50, bidirectional=True)
# Output: hidden_dim * 2 = 100
```

#### 2. Additive (Bahdanau) Attention
```
u = tanh(W @ h + b)           # Context transformation
score = exp(w^T @ u)          # Attention score
alpha = score / sum(scores)   # Normalized weights (with masking)
output = sum(alpha * h)       # Weighted sum
```

### Parameters
- `embedding_dim`: 50
- `hidden_dim`: 50
- `attention_dim`: 50
- `bidirectional`: True
- **Total parameters**: ~137K

### Usage

```python
from hmcan.models import HAN

model = HAN(
    vocab_size=20000,
    embedding_dim=50,
    hidden_dim=50,
    attention_dim=50,
    num_classes=5,
    dropout=0.1,
)
```

---

## HCAN (Hierarchical Cascaded Attention Network)

### Architecture

```
Word Embedding + Positional Embedding
    ↓
Cascaded Multi-Head Self-Attention (Word)
    ↓
Target Attention (Word → Sentence)
    ↓
Sentence Embedding + Positional Embedding
    ↓
Cascaded Multi-Head Self-Attention (Sentence)
    ↓
Target Attention (Sentence → Document)
    ↓
Dense Classifier
```

### Key Components

#### 1. Learnable Positional Embedding
```python
PositionalEmbedding(max_position=200, embedding_dim=50)
# Learnable position embeddings (different from Transformer's sinusoidal)
```

#### 2. Cascaded Multi-Head Attention
Two parallel attention branches with element-wise multiply:

```
Branch 1: Q1, K1, V1 = Conv1D(x) with ELU activation
Branch 2: Q2, K2, V2 = Conv1D(x) with ELU (Q,K), Tanh (V)

output1 = softmax(Q1 @ K1^T / sqrt(d)) @ V1
output2 = softmax(Q2 @ K2^T / sqrt(d)) @ V2

output = LayerNorm(output1 * output2)  # Element-wise multiply
```

#### 3. Target Attention
Aggregate sequence to single vector using learnable target:

```python
target = Parameter(shape=[1, 1, attention_dim])
weights = softmax(target @ hidden^T / sqrt(d))
output = weights @ hidden
```

### Parameters
- `embedding_dim`: 50
- `attention_dim`: 50
- `num_heads`: 5
- `max_words`: 200
- `max_sentences`: 50
- **Total parameters**: ~154K

### Usage

```python
from hmcan.models import HCAN

model = HCAN(
    vocab_size=20000,
    embedding_dim=50,
    attention_dim=50,
    num_heads=5,
    num_classes=5,
    dropout=0.1,
    activation="elu",
)
```

---

## HMCAN (Hierarchical Multichannel CNN-based Attention Network)

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Dual Word Embedding                      │
│              Pretrained (frozen) + Learnable                 │
│                    output = E1 + E2                          │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                   Word-Level Encoder Block                   │
│                                                              │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ Conv1D(Q), Conv1D(K), Conv1D(V)  [kernel=3, ReLU]   │    │
│  └─────────────────────────────────────────────────────┘    │
│                          ↓                                   │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ Scaled Dot-Product Self-Attention                    │    │
│  │ Attention = softmax(Q @ K^T / sqrt(d)) @ V           │    │
│  └─────────────────────────────────────────────────────┘    │
│                          ↓                                   │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ Residual + LayerNorm                                 │    │
│  └─────────────────────────────────────────────────────┘    │
│                          ↓                                   │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ Dense + Residual + LayerNorm                         │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│              Word Target Attention (Tw)                      │
│                                                              │
│     weights = softmax(Tw @ encoded^T / sqrt(d))             │
│     sentence_embed = weights @ encoded                       │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│          Sentence-Level Encoder Block (same structure)       │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│           Sentence Target Attention (Ts)                     │
│                                                              │
│     weights = softmax(Ts @ encoded^T / sqrt(d))             │
│     document_embed = weights @ encoded                       │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                    Dense Classifier                          │
│                  (attention_dim → 5)                         │
└─────────────────────────────────────────────────────────────┘
```

### Key Components

#### 1. Dual Embedding
Sum of pretrained and learnable embeddings:

```python
class DualEmbedding:
    pretrained = Embedding(freeze=True)   # GloVe (frozen)
    learnable = Embedding(xavier_init)    # Learnable

    def forward(x):
        return pretrained(x) + learnable(x)
```

**Benefits**: Pretrained semantic information + task-specific adaptation

#### 2. Conv1D Q, K, V Projections
Capture local context using Conv1D instead of Linear:

```python
Conv1D(in_channels=embed_dim, out_channels=attention_dim, kernel_size=3, padding=1)
# kernel_size=3: Learn local patterns across 3 tokens
```

#### 3. Target Attention
Aggregate variable-length sequence to fixed size using learnable "query" vector:

```python
# Tw: Target for sentence generation
Tw = Parameter([1, 1, attention_dim])

# For each sentence
for each sentence:
    weights = softmax(Tw @ word_encoded^T / sqrt(d))
    sentence_embed = weights @ word_encoded
```

### Parameters
- `embedding_dim`: 50
- `attention_dim`: 50
- `conv_kernel_size`: 3
- `freeze_pretrained`: True
- **Total parameters**: ~101K (trainable)

### Usage

```python
from hmcan.models import HMCAN

model = HMCAN(
    vocab_size=20000,
    embedding_dim=50,
    attention_dim=50,
    num_classes=5,
    conv_kernel_size=3,
    freeze_pretrained=True,
    dropout=0.1,
)
```

---

## Model Comparison

### Parameter Count

| Model | Trainable | Total |
|-------|-----------|-------|
| HAN | 136,905 | 136,905 |
| HCAN | 153,655 | 153,655 |
| HMCAN | 101,155 | ~1.1M (with embeddings) |

### Computational Complexity

| Model | Word-Level | Sentence-Level |
|-------|------------|----------------|
| HAN | O(n × d) (GRU) | O(m × d) (GRU) |
| HCAN | O(n² × d) (Self-Attn) | O(m² × d) (Self-Attn) |
| HMCAN | O(n² × d) (Self-Attn) | O(m² × d) (Self-Attn) |

*n: number of words, m: number of sentences, d: dimension*

### Feature Comparison

| Feature | HAN | HCAN | HMCAN |
|---------|-----|------|-------|
| Embedding | Single | Single | Dual (frozen + learnable) |
| Word Encoder | BiGRU | Multi-head Self-Attn | Conv1D Self-Attn |
| Attention | Additive | Cascaded (2-branch) | Scaled Dot-Product |
| Position Info | Implicit (RNN) | Learnable | None |
| Normalization | - | LayerNorm | LayerNorm |
| Residual | - | After attention | After attention + FFN |

---

## Model Selection Guide

### Choose HAN when:
- Interpretability is important (clear attention weights)
- Sequential information matters
- Computational resources are limited

### Choose HCAN when:
- You want to explicitly learn positional information
- Multiple attention perspectives are needed

### Choose HMCAN when:
- Best performance is required
- You want to leverage pretrained embeddings with task adaptation
- Both local patterns (Conv1D) and global patterns (Self-Attn) are important

---

## Custom Model Creation

```python
from hmcan.models import BaseHierarchicalModel
import torch.nn as nn

class MyModel(BaseHierarchicalModel):
    def __init__(self, vocab_size, embedding_dim, num_classes, **kwargs):
        super().__init__(vocab_size, embedding_dim, num_classes)
        # Define custom layers

    def forward(self, document, sentence_lengths):
        # Custom forward pass
        return {"logits": logits, "word_attention": None, "sentence_attention": None}

    def get_document_embedding(self, document, sentence_lengths):
        # Return document embedding
        pass

# Register in registry
from hmcan.models.registry import register_model
register_model("mymodel", MyModel)
```
