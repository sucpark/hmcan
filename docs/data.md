# 데이터 준비 가이드

## 개요

HMCAN은 Yelp 리뷰 데이터셋을 사용하여 5-class 감성 분류를 수행합니다.

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
