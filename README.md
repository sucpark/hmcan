# HMCAN - Hierarchical Multichannel CNN-based Attention Network

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Weights & Biases](https://img.shields.io/badge/Weights_%26_Biases-FFCC33?logo=weightsandbiases&logoColor=black)](https://wandb.ai/)

PyTorch implementation of hierarchical attention models for document classification.

## Models

This repository includes three hierarchical attention models:

| Model | Description | Key Features |
|-------|-------------|--------------|
| **HAN** | Hierarchical Attention Network | BiGRU + Additive Attention |
| **HCAN** | Hierarchical Cascaded Attention Network | Multi-head Self-Attention + Positional Embeddings |
| **HMCAN** | Hierarchical Multichannel CNN-based Attention Network | Dual Embeddings (Multichannel) + Conv1D Q,K,V + Target Attention |

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

## Supported Datasets

| Dataset | Classes | Size | Task |
|---------|---------|------|------|
| **Yelp** | 5 | 650K | Sentiment (1-5 stars) |
| **IMDB** | 2 | 50K | Sentiment (pos/neg) |
| **AG News** | 4 | 120K | Topic (World, Sports, ...) |
| **DBpedia** | 14 | 630K | Topic (Wikipedia) |
| **Yahoo Answers** | 10 | 1.4M | Q&A Topic |
| **20 Newsgroups** | 20 | 20K | Topic (Classic) |

All datasets available via HuggingFace. See [docs/data.md](docs/data.md) for details.

## Google Colab Notebooks

Train models without local GPU:

| Notebook | Description |
|----------|-------------|
| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sucpark/hmcan/blob/main/notebooks/train_hmcan_colab.ipynb) | Phase 1: HAN, HCAN, HMCAN |
| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sucpark/hmcan/blob/main/notebooks/train_bert_colab.ipynb) | Phase 2: BERT (Multi-dataset) |
| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sucpark/hmcan/blob/main/notebooks/train_longformer_colab.ipynb) | Phase 3: Longformer, BigBird |

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

# Logging
use_tensorboard: true      # Enable TensorBoard logging
use_wandb: false           # Enable Weights & Biases logging
```

## Experiment Tracking

### TensorBoard

```bash
# View training logs
tensorboard --logdir outputs/hmcan_yelp/logs
```

### Weights & Biases

Enable W&B logging in your config:

```yaml
use_wandb: true
```

Or set via environment:

```bash
# Login to wandb (first time only)
wandb login

# Train with wandb enabled
python -m hmcan train --config configs/hmcan.yaml
```

W&B features:
- Real-time metrics visualization
- Hyperparameter tracking
- Model artifact versioning
- Experiment comparison

## Results

Expected performance on Yelp reviews (10K samples):

| Model | Test Accuracy |
|-------|---------------|
| HAN   | ~60.5%        |
| HCAN  | ~58.6%        |
| HMCAN | ~61.7%        |

### Attention Visualization

<!-- TODO: Add attention visualization images after training -->
<!--
![Sentence Attention](docs/images/sentence_attention.png)
![Word Attention](docs/images/word_attention.png)
-->

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
- TensorBoard
- Weights & Biases (wandb)

## License

MIT License

## Citation

If you use this code, please cite:

```bibtex
@misc{hmcan2024,
  author = {sucpark},
  title = {HMCAN: Hierarchical Multichannel CNN-based Attention Network},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/sucpark/hmcan}
}
```
