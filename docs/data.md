# 데이터 준비 가이드

## 개요

HMCAN은 다양한 문서 분류 데이터셋을 지원합니다.

---

## 지원 데이터셋

### 빠른 참조

| 데이터셋 | 클래스 | 크기 | 문서 길이 | 태스크 |
|----------|--------|------|----------|--------|
| **Yelp** | 5 | 650K | 중간 | 감성 분류 |
| **IMDB** | 2 | 50K | 중간 | 감성 분류 |
| **AG News** | 4 | 120K | 짧음 | 토픽 분류 |
| **DBpedia** | 14 | 630K | 짧음 | 토픽 분류 |
| **Yahoo Answers** | 10 | 1.4M | 중간 | Q&A 분류 |
| **20 Newsgroups** | 20 | 20K | 중간 | 토픽 분류 |

### 상세 정보

#### Yelp Review Full
- **HuggingFace**: `yelp_review_full`
- **클래스**: 1-5 star ratings
- **용도**: 감성 분석 기준 데이터셋
- **특징**: 다양한 길이의 리뷰, 균형 잡힌 클래스

#### IMDB
- **HuggingFace**: `imdb`
- **클래스**: Positive / Negative
- **용도**: 이진 감성 분류
- **특징**: 영화 리뷰, 긴 텍스트

#### AG News
- **HuggingFace**: `ag_news`
- **클래스**: World, Sports, Business, Sci/Tech
- **용도**: 뉴스 토픽 분류
- **특징**: 짧은 뉴스 기사

#### DBpedia
- **HuggingFace**: `dbpedia_14`
- **클래스**: 14개 위키피디아 카테고리
- **용도**: 대규모 다중 클래스 분류
- **특징**: 구조화된 짧은 설명

#### Yahoo Answers
- **HuggingFace**: `yahoo_answers_topics`
- **클래스**: 10개 토픽 (Science, Health, Sports, ...)
- **용도**: Q&A 토픽 분류
- **특징**: 질문 + 답변 결합

#### 20 Newsgroups
- **HuggingFace**: `SetFit/20_newsgroups`
- **클래스**: 20개 뉴스그룹
- **용도**: 다중 클래스 분류
- **특징**: 클래식 NLP 벤치마크

---

## Python에서 사용

### 빠른 로드

```python
from hmcan.data import quick_load, list_datasets

# 사용 가능한 데이터셋 확인
list_datasets()

# 데이터셋 로드
train, test, num_classes, labels = quick_load("imdb", max_samples=10000)
print(f"Classes: {num_classes}")
print(f"Labels: {labels}")
```

### 상세 로드

```python
from hmcan.data import load_classification_dataset, get_class_labels

# 데이터셋 로드
train_data, test_data, info = load_classification_dataset(
    name="ag_news",
    max_samples=10000,
    seed=42
)

# 정보 확인
print(f"Dataset: {info.name}")
print(f"Classes: {info.num_classes}")
print(f"Train: {len(train_data)}, Test: {len(test_data)}")

# 클래스 레이블
labels = get_class_labels("ag_news")
print(f"Labels: {labels}")
```

### Colab에서 사용

```python
from datasets import load_dataset

# 직접 HuggingFace에서 로드
dataset = load_dataset("imdb")
train = dataset["train"]
test = dataset["test"]
```

---

## Yelp 데이터 (기존 방식)

HMCAN 원본 모델용 Yelp 데이터 준비:

데이터 파이프라인:
1. Yelp 리뷰 다운로드 (Hugging Face)
2. GloVe 임베딩 다운로드
3. 텍스트 전처리 및 토큰화
4. 어휘 사전 생성
5. 데이터 저장

## 빠른 시작

```bash
# 기본 설정으로 데이터 다운로드 (10,000 샘플)
python scripts/download_data.py
```

## 상세 옵션

```bash
python scripts/download_data.py \
    --data-dir data \           # 데이터 저장 경로
    --max-samples 10000 \       # 최대 샘플 수
    --embedding-dim 50 \        # 임베딩 차원 (50, 100, 200, 300)
    --min-freq 5 \              # 최소 단어 빈도
    --max-vocab 50000           # 최대 어휘 크기
```

### 옵션 설명

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--data-dir` | `data` | 데이터 저장 디렉토리 |
| `--max-samples` | `10000` | 다운로드할 최대 리뷰 수 |
| `--embedding-dim` | `50` | GloVe 임베딩 차원 |
| `--min-freq` | `5` | 어휘에 포함할 최소 단어 빈도 |
| `--max-vocab` | `50000` | 최대 어휘 사전 크기 |

## 생성되는 파일

```
data/
├── embeddings/
│   └── glove.6B.50d.txt       # GloVe 원본 파일 (~160MB)
└── processed/
    ├── word2idx.json          # 어휘 사전 (word → index)
    ├── embeddings_50d.npz     # 어휘에 맞게 추출된 임베딩
    └── yelp_processed.npz     # 전처리된 문서 데이터
```

## 데이터 형식

### word2idx.json
```json
{
    "<PAD>": 0,
    "<UNK>": 1,
    "the": 2,
    "a": 3,
    ...
}
```

### yelp_processed.npz
각 문서는 다음 형식으로 저장:
```python
{
    "document": [[1, 23, 45, ...], [67, 89, ...], ...],  # 문장별 단어 인덱스
    "label": 3  # 0-4 (별점 1-5)
}
```

## 전처리 과정

1. **문장 분리**: NLTK `sent_tokenize`
2. **단어 토큰화**: NLTK `word_tokenize`
3. **소문자 변환**: 모든 텍스트
4. **구두점 제거**: 특수문자 제거
5. **빈 문장 필터링**: 최소 1개 단어 필요
6. **OOV 처리**: 어휘에 없는 단어 → `<UNK>`

## 데이터셋 통계

기본 설정 (10,000 샘플) 기준:
- 학습: 8,000 샘플 (80%)
- 검증: 1,000 샘플 (10%)
- 테스트: 1,000 샘플 (10%)
- 어휘 크기: ~15,000-20,000 단어
- 평균 문장 수/문서: ~8-10
- 평균 단어 수/문장: ~15-20

## 커스텀 데이터 사용

### 1. 데이터 형식 맞추기

```python
# 문서 리스트 준비
documents = [
    [[1, 2, 3], [4, 5, 6, 7]],  # 문서 1: 2개 문장
    [[8, 9], [10, 11, 12]],     # 문서 2: 2개 문장
    ...
]
labels = [0, 3, ...]  # 0-4 클래스 레이블

# npz로 저장
import numpy as np
data = [{"document": doc, "label": label} for doc, label in zip(documents, labels)]
np.savez_compressed("data/processed/custom_data.npz", data=np.array(data, dtype=object))
```

### 2. 어휘 사전 생성

```python
import json

word2idx = {"<PAD>": 0, "<UNK>": 1, "word1": 2, ...}
with open("data/processed/word2idx.json", "w") as f:
    json.dump(word2idx, f)
```

### 3. 임베딩 준비 (선택)

```python
import numpy as np

vocab_size = len(word2idx)
embedding_dim = 50
embeddings = np.random.randn(vocab_size, embedding_dim).astype(np.float32)
np.savez_compressed("data/processed/embeddings_50d.npz", embeddings=embeddings)
```

## 메모리 고려사항

| 샘플 수 | 대략적 메모리 |
|---------|---------------|
| 10,000 | ~500 MB |
| 50,000 | ~2 GB |
| 100,000 | ~4 GB |
| 전체 (650K) | ~25 GB |

큰 데이터셋의 경우 `--max-samples`로 제한하거나 청크 단위 처리 권장.
