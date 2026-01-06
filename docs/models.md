# 모델 아키텍처

## 개요

이 프로젝트는 계층적 문서 분류를 위한 세 가지 어텐션 기반 모델을 구현합니다:

| 모델 | 설명 | 핵심 특징 |
|------|------|----------|
| HAN | Hierarchical Attention Network | BiGRU + Additive Attention |
| HCAN | Hierarchical Cascaded Attention Network | Multi-head Self-Attention + Positional |
| HMCAN | Hierarchical Multichannel CNN-based Attention Network | Dual Embedding (Multichannel) + Conv1D + Target Attention |

## 공통 구조

모든 모델은 2-level 계층 구조를 따릅니다:

```
문서 (Document)
    ↓
┌───────────────────────────────────┐
│        Word-Level Encoder         │
│   단어 → 문장 임베딩으로 집계      │
└───────────────────────────────────┘
    ↓
┌───────────────────────────────────┐
│      Sentence-Level Encoder       │
│   문장 → 문서 임베딩으로 집계      │
└───────────────────────────────────┘
    ↓
┌───────────────────────────────────┐
│          Classification           │
│     문서 임베딩 → 5-class 분류     │
└───────────────────────────────────┘
```

---

## HAN (Hierarchical Attention Network)

### 아키텍처

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

### 핵심 구성요소

#### 1. Bidirectional GRU
```python
BiGRU(input_dim=50, hidden_dim=50, bidirectional=True)
# 출력: hidden_dim * 2 = 100
```

#### 2. Additive (Bahdanau) Attention
```
u = tanh(W @ h + b)           # Context transformation
score = exp(w^T @ u)          # Attention score
alpha = score / sum(scores)   # Normalized weights (with masking)
output = sum(alpha * h)       # Weighted sum
```

### 파라미터
- `embedding_dim`: 50
- `hidden_dim`: 50
- `attention_dim`: 50
- `bidirectional`: True
- **총 파라미터**: ~137K

### 사용법

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

### 아키텍처

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

### 핵심 구성요소

#### 1. Learnable Positional Embedding
```python
PositionalEmbedding(max_position=200, embedding_dim=50)
# 학습 가능한 위치 임베딩 (Transformer의 sinusoidal과 다름)
```

#### 2. Cascaded Multi-Head Attention
두 개의 병렬 어텐션 브랜치를 element-wise multiply:

```
Branch 1: Q1, K1, V1 = Conv1D(x) with ELU activation
Branch 2: Q2, K2, V2 = Conv1D(x) with ELU (Q,K), Tanh (V)

output1 = softmax(Q1 @ K1^T / sqrt(d)) @ V1
output2 = softmax(Q2 @ K2^T / sqrt(d)) @ V2

output = LayerNorm(output1 * output2)  # Element-wise multiply
```

#### 3. Target Attention
학습 가능한 타겟 벡터로 시퀀스를 단일 벡터로 집계:

```python
target = Parameter(shape=[1, 1, attention_dim])
weights = softmax(target @ hidden^T / sqrt(d))
output = weights @ hidden
```

### 파라미터
- `embedding_dim`: 50
- `attention_dim`: 50
- `num_heads`: 5
- `max_words`: 200
- `max_sentences`: 50
- **총 파라미터**: ~154K

### 사용법

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

### 아키텍처

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
│          Sentence-Level Encoder Block (동일 구조)            │
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

### 핵심 구성요소

#### 1. Dual Embedding
사전학습 임베딩과 학습 가능 임베딩을 합산:

```python
class DualEmbedding:
    pretrained = Embedding(freeze=True)   # GloVe (고정)
    learnable = Embedding(xavier_init)    # 학습 가능

    def forward(x):
        return pretrained(x) + learnable(x)
```

**이점**: 사전학습된 의미 정보 + 태스크별 적응

#### 2. Conv1D Q, K, V Projections
Linear 대신 Conv1D로 지역 컨텍스트 캡처:

```python
Conv1D(in_channels=embed_dim, out_channels=attention_dim, kernel_size=3, padding=1)
# kernel_size=3: 3개 토큰의 지역 패턴 학습
```

#### 3. Target Attention
학습 가능한 "질의" 벡터로 가변 길이 시퀀스를 고정 크기로 집계:

```python
# Tw: 문장 생성용 타겟
Tw = Parameter([1, 1, attention_dim])

# 각 문장에 대해
for each sentence:
    weights = softmax(Tw @ word_encoded^T / sqrt(d))
    sentence_embed = weights @ word_encoded
```

### 파라미터
- `embedding_dim`: 50
- `attention_dim`: 50
- `conv_kernel_size`: 3
- `freeze_pretrained`: True
- **총 파라미터**: ~101K (학습 가능)

### 사용법

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

## 모델 비교

### 파라미터 수

| 모델 | 학습 가능 | 전체 |
|------|----------|------|
| HAN | 136,905 | 136,905 |
| HCAN | 153,655 | 153,655 |
| HMCAN | 101,155 | ~1.1M (임베딩 포함) |

### 계산 복잡도

| 모델 | Word-Level | Sentence-Level |
|------|------------|----------------|
| HAN | O(n × d) (GRU) | O(m × d) (GRU) |
| HCAN | O(n² × d) (Self-Attn) | O(m² × d) (Self-Attn) |
| HMCAN | O(n² × d) (Self-Attn) | O(m² × d) (Self-Attn) |

*n: 단어 수, m: 문장 수, d: 차원*

### 특징 비교

| 특징 | HAN | HCAN | HMCAN |
|------|-----|------|-------|
| 임베딩 | Single | Single | Dual (frozen + learnable) |
| Word Encoder | BiGRU | Multi-head Self-Attn | Conv1D Self-Attn |
| Attention | Additive | Cascaded (2-branch) | Scaled Dot-Product |
| Position Info | Implicit (RNN) | Learnable | None |
| Normalization | - | LayerNorm | LayerNorm |
| Residual | - | After attention | After attention + FFN |

---

## 모델 선택 가이드

### HAN 선택 시
- 해석 가능성이 중요할 때 (명확한 attention weights)
- 순차적 정보가 중요할 때
- 계산 리소스가 제한적일 때

### HCAN 선택 시
- 위치 정보를 명시적으로 학습하고 싶을 때
- 다양한 관점의 attention이 필요할 때

### HMCAN 선택 시
- 최고 성능이 필요할 때
- 사전학습 임베딩을 활용하면서 태스크 적응이 필요할 때
- 지역 패턴(Conv1D)과 전역 패턴(Self-Attn) 모두 중요할 때

---

## 커스텀 모델 생성

```python
from hmcan.models import BaseHierarchicalModel
import torch.nn as nn

class MyModel(BaseHierarchicalModel):
    def __init__(self, vocab_size, embedding_dim, num_classes, **kwargs):
        super().__init__(vocab_size, embedding_dim, num_classes)
        # 커스텀 레이어 정의

    def forward(self, document, sentence_lengths):
        # 커스텀 forward pass
        return {"logits": logits, "word_attention": None, "sentence_attention": None}

    def get_document_embedding(self, document, sentence_lengths):
        # 문서 임베딩 반환
        pass

# 레지스트리에 등록
from hmcan.models.registry import register_model
register_model("mymodel", MyModel)
```
