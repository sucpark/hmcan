# HMCAN 문서

Hierarchical Multi-head Cascaded Attention Network for Document Classification

## 목차

1. [설치 가이드](installation.md)
   - 요구사항
   - uv / pip / conda 설치
   - GPU 설정
   - 문제 해결

2. [데이터 준비](data.md)
   - 데이터 다운로드
   - 전처리 과정
   - 커스텀 데이터 사용
   - 데이터 형식

3. [학습 가이드](training.md)
   - 학습 실행
   - 설정 파일
   - TensorBoard 모니터링
   - 하이퍼파라미터 튜닝
   - 체크포인트 관리

4. [모델 아키텍처](models.md)
   - HAN (Hierarchical Attention Network)
   - HCAN (Hierarchical Cascaded Attention Network)
   - HMCAN (Hierarchical Multi-head Cascaded Attention Network)
   - 모델 비교
   - 커스텀 모델 생성

5. [평가 및 추론](evaluation.md)
   - 모델 평가
   - 단일/배치 추론
   - Attention 시각화
   - 성능 메트릭
   - 모델 내보내기 (TorchScript, ONNX)

## 빠른 시작

```bash
# 1. 설치
git clone https://github.com/sucpark/hmcan.git
cd hmcan
uv sync  # 또는 pip install -e .

# 2. 데이터 준비
python scripts/download_data.py

# 3. 학습
python -m hmcan train --config configs/hmcan.yaml

# 4. 평가
python -m hmcan evaluate --checkpoint outputs/hmcan_yelp/checkpoints/best_model.pt
```

## CLI 명령어

| 명령어 | 설명 |
|--------|------|
| `python -m hmcan train` | 모델 학습 |
| `python -m hmcan evaluate` | 모델 평가 |
| `python -m hmcan download` | 데이터 다운로드 |
| `python -m hmcan models` | 사용 가능한 모델 목록 |

## 프로젝트 구조

```
hmcan/
├── configs/              # 설정 파일
├── data/                 # 데이터
├── docs/                 # 문서
├── notebooks/            # 원본 Jupyter 노트북
├── outputs/              # 학습 결과
├── scripts/              # 유틸리티 스크립트
├── src/hmcan/            # 소스 코드
│   ├── config.py
│   ├── data/
│   ├── models/
│   ├── training/
│   └── utils/
└── tests/                # 테스트
```

## 참고 문헌

- **HAN**: Yang et al., "Hierarchical Attention Networks for Document Classification" (NAACL 2016)
- **Transformer**: Vaswani et al., "Attention Is All You Need" (NeurIPS 2017)
- **GloVe**: Pennington et al., "GloVe: Global Vectors for Word Representation" (EMNLP 2014)

## 라이선스

MIT License
