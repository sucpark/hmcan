# 설치 가이드

## 요구사항

- Python >= 3.11
- PyTorch >= 2.0
- macOS, Linux, Windows (CUDA 지원)

## 설치 방법

### 1. uv 사용 (권장)

[uv](https://github.com/astral-sh/uv)는 빠른 Python 패키지 관리자입니다.

```bash
# uv 설치 (아직 없는 경우)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 저장소 클론
git clone https://github.com/sucpark/hmcan.git
cd hmcan

# 의존성 설치 및 가상환경 생성
uv sync

# 가상환경 활성화
source .venv/bin/activate
```

### 2. pip 사용

```bash
# 저장소 클론
git clone https://github.com/sucpark/hmcan.git
cd hmcan

# 가상환경 생성
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 패키지 설치
pip install -e .

# 개발 의존성 포함 설치
pip install -e ".[dev]"
```

### 3. conda 사용

```bash
# 저장소 클론
git clone https://github.com/sucpark/hmcan.git
cd hmcan

# conda 환경 생성
conda create -n hmcan python=3.11
conda activate hmcan

# PyTorch 설치 (CUDA 버전에 맞게)
conda install pytorch torchvision -c pytorch

# 패키지 설치
pip install -e .
```

## 설치 확인

```bash
# 버전 확인
python -m hmcan --version

# 모델 목록 확인
python -m hmcan models

# 모듈 테스트
python -c "from hmcan.models import HAN, HCAN, HMCAN; print('OK')"
```

## Weights & Biases 설정 (선택)

실험 추적을 위해 [Weights & Biases](https://wandb.ai)를 사용할 수 있습니다.

### 1. 계정 생성

[https://wandb.ai/site](https://wandb.ai/site)에서 무료 계정을 생성합니다.

### 2. API 키 확인

로그인 후 [https://wandb.ai/authorize](https://wandb.ai/authorize)에서 API 키를 복사합니다.

### 3. 로그인

```bash
wandb login
# 프롬프트가 나타나면 API 키 붙여넣기
```

또는 환경 변수로 설정:

```bash
export WANDB_API_KEY="your-api-key-here"
```

### 4. 설정 파일에서 활성화

```yaml
# configs/hmcan.yaml
use_wandb: true
```

### 확인

```bash
# wandb 연결 테스트
python -c "import wandb; wandb.login(); print('OK')"
```

> **참고**: wandb 없이도 TensorBoard로 학습을 모니터링할 수 있습니다.

## GPU 설정

### CUDA (NVIDIA GPU)

```bash
# CUDA 버전 확인
nvidia-smi

# PyTorch CUDA 확인
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### MPS (Apple Silicon)

```bash
# MPS 확인 (M1/M2/M3 Mac)
python -c "import torch; print(f'MPS available: {torch.backends.mps.is_available()}')"
```

## 문제 해결

### NLTK 데이터 오류

```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"
```

### PyTorch 버전 충돌

```bash
# 기존 PyTorch 제거 후 재설치
pip uninstall torch torchvision
pip install torch torchvision
```

### 메모리 부족

`configs/*.yaml`에서 다음 설정 조정:
- `data.max_samples`: 샘플 수 줄이기
- `model.embedding_dim`: 임베딩 차원 줄이기
