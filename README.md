# HMCAN - Hierarchical Multi-head Cascaded Attention Network

PyTorch implementation of hierarchical attention models for document classification.

## Models

This repository includes three hierarchical attention models:

| Model | Description | Key Features |
|-------|-------------|--------------|
| **HAN** | Hierarchical Attention Network | BiGRU + Additive Attention |
| **HCAN** | Hierarchical Cascaded Attention Network | Multi-head Self-Attention + Positional Embeddings |
| **HMCAN** | Hierarchical Multi-head Cascaded Attention Network | Dual Embeddings + Conv1D Q,K,V + Target Attention |

## Architecture

### HMCAN (Main Model)

```
Document
    ↓
┌─────────────────────────────────────────────────────────────┐
│  Word Embedding (Pretrained + Learnable, summed)            │
└─────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────┐
│  Word-Level Encoder                                         │
│  Conv1D(Q,K,V) → Self-Attention → Residual + LayerNorm      │
│  → Dense → Residual + LayerNorm                             │
└─────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────┐
│  Word Target Attention (Tw) → Sentence Embeddings           │
└─────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────┐
│  Sentence-Level Encoder (same structure as word-level)      │
└─────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────┐
│  Sentence Target Attention (Ts) → Document Embedding        │
└─────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────┐
│  Dense Classifier → 5-class Softmax                         │
└─────────────────────────────────────────────────────────────┘
```

## Installation

```bash
# Clone repository
git clone https://github.com/sucpark/hmcan.git
cd hmcan

# Install with uv (recommended)
uv sync

# Or install with pip
pip install -e .
```

## Quick Start

### 1. Download Data

```bash
# Download Yelp dataset and GloVe embeddings
python scripts/download_data.py --max-samples 10000
```

### 2. Train Model

```bash
# Train HMCAN (main model)
python -m hmcan train --config configs/hmcan.yaml

# Train HAN (baseline)
python -m hmcan train --config configs/han.yaml

# Train HCAN
python -m hmcan train --config configs/hcan.yaml
```

### 3. Evaluate

```bash
# Evaluate on test set
python -m hmcan evaluate --checkpoint outputs/hmcan_yelp/checkpoints/best_model.pt
```

## Project Structure

```
hmcan/
├── configs/                 # YAML configuration files
│   ├── han.yaml
│   ├── hcan.yaml
│   └── hmcan.yaml
├── data/                    # Data directory
│   ├── processed/           # Preprocessed data
│   └── embeddings/          # Word embeddings
├── src/hmcan/               # Source code
│   ├── config.py            # Configuration dataclasses
│   ├── data/                # Data loading
│   │   ├── dataset.py       # PyTorch Dataset
│   │   ├── datamodule.py    # Data module
│   │   └── preprocessing.py # Text preprocessing
│   ├── models/              # Model implementations
│   │   ├── han.py           # HAN model
│   │   ├── hcan.py          # HCAN model
│   │   ├── hmcan.py         # HMCAN model
│   │   └── layers/          # Attention & embedding layers
│   ├── training/            # Training infrastructure
│   │   ├── trainer.py       # Training loop
│   │   ├── callbacks.py     # Early stopping, checkpointing
│   │   └── metrics.py       # Accuracy tracking
│   └── utils/               # Utilities
├── scripts/                 # Utility scripts
│   └── download_data.py     # Data download script
├── notebooks/               # Original Jupyter notebooks
└── outputs/                 # Training outputs
```

## Configuration

All hyperparameters are configured via YAML files. Key parameters:

```yaml
model:
  name: hmcan              # Model type: han, hcan, hmcan
  embedding_dim: 50        # Word embedding dimension
  attention_dim: 50        # Attention dimension
  dropout: 0.1             # Dropout probability

training:
  num_epochs: 30           # Training epochs
  learning_rate: 2.0e-5    # Adam learning rate
  early_stopping: true     # Enable early stopping
  patience: 5              # Early stopping patience

data:
  train_ratio: 0.8         # Train split ratio
  val_ratio: 0.1           # Validation split ratio
  test_ratio: 0.1          # Test split ratio
```

## Results

Expected performance on Yelp reviews (10K samples):

| Model | Test Accuracy |
|-------|---------------|
| HAN   | ~60.5%        |
| HCAN  | ~58.6%        |
| HMCAN | ~61.7%        |

## Documentation

상세 문서는 [docs/](docs/) 폴더를 참조하세요:

- [설치 가이드](docs/installation.md) - 설치 방법, GPU 설정, 문제 해결
- [데이터 준비](docs/data.md) - 데이터 다운로드, 전처리, 커스텀 데이터
- [학습 가이드](docs/training.md) - 학습 실행, 설정, 하이퍼파라미터 튜닝
- [모델 아키텍처](docs/models.md) - HAN, HCAN, HMCAN 상세 설명
- [평가 및 추론](docs/evaluation.md) - 테스트, 추론, Attention 시각화

## Requirements

- Python >= 3.11
- PyTorch >= 2.0
- NLTK
- scikit-learn
- tensorboard

## License

MIT License

## Citation

If you use this code, please cite:

```bibtex
@misc{hmcan2024,
  author = {sucpark},
  title = {HMCAN: Hierarchical Multi-head Cascaded Attention Network},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/sucpark/hmcan}
}
```
