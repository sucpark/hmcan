# Training Guide

## Quick Start

```bash
# Train HMCAN (default)
python -m hmcan train --config configs/hmcan.yaml

# Train HAN (baseline)
python -m hmcan train --config configs/han.yaml

# Train HCAN
python -m hmcan train --config configs/hcan.yaml
```

## Training Options

```bash
python -m hmcan train \
    --config configs/hmcan.yaml \  # Config file (required)
    --resume outputs/exp/checkpoints/checkpoint_epoch_010.pt \  # Resume from checkpoint
    --seed 42 \                    # Override random seed
    --device cuda                  # Override device
```

## Configuration File Structure

Example `configs/hmcan.yaml`:

```yaml
# Model settings
model:
  name: hmcan                 # Model type: han, hcan, hmcan
  vocab_size: 50000           # Vocabulary size (auto-set)
  embedding_dim: 50           # Embedding dimension
  attention_dim: 50           # Attention dimension
  num_classes: 5              # Number of output classes
  dropout: 0.1                # Dropout probability
  conv_kernel_size: 3         # Conv1D kernel size (HMCAN)
  freeze_pretrained: true     # Freeze pretrained embeddings

# Training settings
training:
  num_epochs: 30              # Number of epochs
  learning_rate: 2.0e-5       # Learning rate
  weight_decay: 0.0           # Weight decay
  beta1: 0.9                  # Adam beta1
  beta2: 0.99                 # Adam beta2
  max_grad_norm: 1.0          # Gradient clipping
  early_stopping: true        # Early stopping
  patience: 5                 # Early stopping patience

# Data settings
data:
  data_dir: data
  vocab_path: data/processed/word2idx.json
  embeddings_path: data/processed/embeddings_50d.npz
  train_ratio: 0.8
  val_ratio: 0.1
  test_ratio: 0.1
  max_samples: null           # null = all data

# General settings
seed: 14
device: auto                  # auto, cpu, cuda, mps
output_dir: outputs
experiment_name: hmcan_yelp

# Logging
use_tensorboard: true         # Use TensorBoard
use_wandb: false              # Use Weights & Biases
```

## Training Process

### 1. Initialization
- Set random seed
- Select device (GPU/MPS/CPU)
- Load and split data
- Create model

### 2. Epoch Loop
```
for epoch in epochs:
    # Training
    for document in train_loader:
        forward → loss → backward → update

    # Validation
    for document in val_loader:
        forward → metrics

    # Checkpoint
    if best_accuracy:
        save_model()

    # Early stopping check
    if no_improvement >= patience:
        break
```

### 3. Output Files

```
outputs/{experiment_name}/
├── config.yaml              # Used configuration
├── checkpoints/
│   ├── best_model.pt        # Best performing model
│   ├── checkpoint_epoch_001.pt
│   ├── checkpoint_epoch_002.pt
│   └── ...
└── logs/                    # TensorBoard logs
    └── events.out.tfevents.*
```

## TensorBoard Monitoring

```bash
# Run TensorBoard
tensorboard --logdir outputs/hmcan_yelp/logs

# Open http://localhost:6006 in browser
```

Available metrics:
- `train/loss`: Training loss
- `train/accuracy`: Training accuracy
- `val/loss`: Validation loss
- `val/accuracy`: Validation accuracy

## Weights & Biases Monitoring

### Setup

```yaml
# configs/hmcan.yaml
use_wandb: true
```

### First-time Login

```bash
wandb login
# Enter API key (from https://wandb.ai/authorize)
```

### Run Training

```bash
python -m hmcan train --config configs/hmcan.yaml
```

### Features

| Feature | Description |
|---------|-------------|
| **Real-time dashboard** | Monitor training progress on web |
| **Hyperparameter tracking** | Auto-save all settings |
| **Experiment comparison** | Compare multiple experiments |
| **Model artifacts** | Auto-save best_model.pt |
| **Team sharing** | Share results via link |

### Logged Metrics

```
train/loss          # Training loss
train/accuracy      # Training accuracy
val/loss            # Validation loss
val/accuracy        # Validation accuracy
learning_rate       # Learning rate
epoch               # Epoch
```

### Offline Mode

Train without internet and sync later:

```bash
# Train in offline mode
WANDB_MODE=offline python -m hmcan train --config configs/hmcan.yaml

# Sync later
wandb sync outputs/hmcan_yelp/wandb/offline-run-*
```

## Hyperparameter Tuning

### Learning Rate

```yaml
# Recommended range: 1e-5 ~ 1e-4
training:
  learning_rate: 2.0e-5  # Default (from original paper)
```

### Dropout

```yaml
# Increase if overfitting, decrease if underfitting
model:
  dropout: 0.1   # Default
  # dropout: 0.2  # Prevent overfitting
  # dropout: 0.05 # More capacity
```

### Embedding Dimension

```yaml
# Larger dimension = more expressiveness, more memory
model:
  embedding_dim: 50   # Default
  # embedding_dim: 100  # Richer representation
  # embedding_dim: 300  # GloVe maximum
```

## Checkpoint Management

### Resume Training

```bash
# Resume from last checkpoint
python -m hmcan train \
    --config configs/hmcan.yaml \
    --resume outputs/hmcan_yelp/checkpoints/checkpoint_epoch_015.pt
```

### Load Best Model

```python
import torch
from hmcan.models import HMCAN

model = HMCAN(vocab_size=20000)
ckpt = torch.load("outputs/hmcan_yelp/checkpoints/best_model.pt")
model.load_state_dict(ckpt["model_state_dict"])
```

## Training Tips

### 1. Test with Small Data

```yaml
data:
  max_samples: 1000  # Test with 1000 samples first
```

### 2. Gradient Clipping

```yaml
training:
  max_grad_norm: 1.0  # Prevent gradient explosion
```

### 3. Use Early Stopping

```yaml
training:
  early_stopping: true
  patience: 5  # Stop if no improvement for 5 epochs
```

### 4. Fix Seed

```yaml
seed: 14  # Reproducible results
```

## Expected Performance

Based on 10,000 samples, 30 epochs:

| Model | Test Accuracy | Training Time (M1 Mac) |
|-------|---------------|------------------------|
| HAN | ~60.5% | ~30 min |
| HCAN | ~58.6% | ~20 min |
| HMCAN | ~61.7% | ~20 min |

*5-10x faster with GPU*
