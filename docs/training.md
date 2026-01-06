# 학습 가이드

## 빠른 시작

```bash
# HMCAN 학습 (기본)
python -m hmcan train --config configs/hmcan.yaml

# HAN 학습 (베이스라인)
python -m hmcan train --config configs/han.yaml

# HCAN 학습
python -m hmcan train --config configs/hcan.yaml
```

## 학습 옵션

```bash
python -m hmcan train \
    --config configs/hmcan.yaml \  # 설정 파일 (필수)
    --resume outputs/exp/checkpoints/checkpoint_epoch_010.pt \  # 체크포인트에서 재개
    --seed 42 \                    # 랜덤 시드 오버라이드
    --device cuda                  # 디바이스 오버라이드
```

## 설정 파일 구조

`configs/hmcan.yaml` 예시:

```yaml
# 모델 설정
model:
  name: hmcan                 # 모델 종류: han, hcan, hmcan
  vocab_size: 50000           # 어휘 크기 (자동 설정됨)
  embedding_dim: 50           # 임베딩 차원
  attention_dim: 50           # 어텐션 차원
  num_classes: 5              # 출력 클래스 수
  dropout: 0.1                # 드롭아웃 확률
  conv_kernel_size: 3         # Conv1D 커널 크기 (HMCAN)
  freeze_pretrained: true     # 사전학습 임베딩 고정

# 학습 설정
training:
  num_epochs: 30              # 에폭 수
  learning_rate: 2.0e-5       # 학습률
  weight_decay: 0.0           # 가중치 감쇠
  beta1: 0.9                  # Adam beta1
  beta2: 0.99                 # Adam beta2
  max_grad_norm: 1.0          # 그래디언트 클리핑
  early_stopping: true        # 조기 종료
  patience: 5                 # 조기 종료 인내심

# 데이터 설정
data:
  data_dir: data
  vocab_path: data/processed/word2idx.json
  embeddings_path: data/processed/embeddings_50d.npz
  train_ratio: 0.8
  val_ratio: 0.1
  test_ratio: 0.1
  max_samples: null           # null = 전체 데이터

# 일반 설정
seed: 14
device: auto                  # auto, cpu, cuda, mps
output_dir: outputs
experiment_name: hmcan_yelp
use_tensorboard: true
```

## 학습 과정

### 1. 초기화
- 랜덤 시드 설정
- 디바이스 선택 (GPU/MPS/CPU)
- 데이터 로딩 및 분할
- 모델 생성

### 2. 에폭 루프
```
for epoch in epochs:
    # 학습
    for document in train_loader:
        forward → loss → backward → update

    # 검증
    for document in val_loader:
        forward → metrics

    # 체크포인트
    if best_accuracy:
        save_model()

    # 조기 종료 체크
    if no_improvement >= patience:
        break
```

### 3. 출력 파일

```
outputs/{experiment_name}/
├── config.yaml              # 사용된 설정
├── checkpoints/
│   ├── best_model.pt        # 최고 성능 모델
│   ├── checkpoint_epoch_001.pt
│   ├── checkpoint_epoch_002.pt
│   └── ...
└── logs/                    # TensorBoard 로그
    └── events.out.tfevents.*
```

## TensorBoard 모니터링

```bash
# TensorBoard 실행
tensorboard --logdir outputs/hmcan_yelp/logs

# 브라우저에서 http://localhost:6006 접속
```

모니터링 가능한 메트릭:
- `train/loss`: 학습 손실
- `train/accuracy`: 학습 정확도
- `val/loss`: 검증 손실
- `val/accuracy`: 검증 정확도

## 하이퍼파라미터 튜닝

### 학습률

```yaml
# 권장 범위: 1e-5 ~ 1e-4
training:
  learning_rate: 2.0e-5  # 기본값 (원본 논문)
```

### 드롭아웃

```yaml
# 과적합 시 증가, 언더피팅 시 감소
model:
  dropout: 0.1   # 기본값
  # dropout: 0.2  # 과적합 방지
  # dropout: 0.05 # 더 많은 용량
```

### 임베딩 차원

```yaml
# 더 큰 차원 = 더 많은 표현력, 더 많은 메모리
model:
  embedding_dim: 50   # 기본값
  # embedding_dim: 100  # 더 풍부한 표현
  # embedding_dim: 300  # GloVe 최대
```

## 체크포인트 관리

### 학습 재개

```bash
# 마지막 체크포인트에서 재개
python -m hmcan train \
    --config configs/hmcan.yaml \
    --resume outputs/hmcan_yelp/checkpoints/checkpoint_epoch_015.pt
```

### 최고 모델 로드

```python
import torch
from hmcan.models import HMCAN

model = HMCAN(vocab_size=20000)
ckpt = torch.load("outputs/hmcan_yelp/checkpoints/best_model.pt")
model.load_state_dict(ckpt["model_state_dict"])
```

## 학습 팁

### 1. 작은 데이터로 테스트

```yaml
data:
  max_samples: 1000  # 먼저 1000개로 테스트
```

### 2. 그래디언트 클리핑

```yaml
training:
  max_grad_norm: 1.0  # 그래디언트 폭발 방지
```

### 3. 조기 종료 활용

```yaml
training:
  early_stopping: true
  patience: 5  # 5 에폭 동안 개선 없으면 종료
```

### 4. 시드 고정

```yaml
seed: 14  # 재현 가능한 결과
```

## 예상 성능

10,000 샘플, 30 에폭 기준:

| 모델 | 테스트 정확도 | 학습 시간 (M1 Mac) |
|------|---------------|-------------------|
| HAN | ~60.5% | ~30분 |
| HCAN | ~58.6% | ~20분 |
| HMCAN | ~61.7% | ~20분 |

*GPU 사용 시 5-10배 빠름*
