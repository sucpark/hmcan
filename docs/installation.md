# Installation Guide

## Requirements

- Python >= 3.11
- PyTorch >= 2.0
- macOS, Linux, Windows (with CUDA support)

## Installation Methods

### 1. Using uv (Recommended)

[uv](https://github.com/astral-sh/uv) is a fast Python package manager.

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone repository
git clone https://github.com/sucpark/hmcan.git
cd hmcan

# Install dependencies and create virtual environment
uv sync

# Activate virtual environment
source .venv/bin/activate
```

### 2. Using pip

```bash
# Clone repository
git clone https://github.com/sucpark/hmcan.git
cd hmcan

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install package
pip install -e .

# Install with dev dependencies
pip install -e ".[dev]"
```

### 3. Using conda

```bash
# Clone repository
git clone https://github.com/sucpark/hmcan.git
cd hmcan

# Create conda environment
conda create -n hmcan python=3.11
conda activate hmcan

# Install PyTorch (match your CUDA version)
conda install pytorch torchvision -c pytorch

# Install package
pip install -e .
```

## Verify Installation

```bash
# Check version
python -m hmcan --version

# List available models
python -m hmcan models

# Test module imports
python -c "from hmcan.models import HAN, HCAN, HMCAN; print('OK')"
```

## Weights & Biases Setup (Optional)

You can use [Weights & Biases](https://wandb.ai) for experiment tracking.

### 1. Create Account

Create a free account at [https://wandb.ai/site](https://wandb.ai/site).

### 2. Get API Key

After logging in, copy your API key from [https://wandb.ai/authorize](https://wandb.ai/authorize).

### 3. Login

```bash
wandb login
# Paste API key when prompted
```

Or set via environment variable:

```bash
export WANDB_API_KEY="your-api-key-here"
```

### 4. Enable in Config

```yaml
# configs/hmcan.yaml
use_wandb: true
```

### Verify

```bash
# Test wandb connection
python -c "import wandb; wandb.login(); print('OK')"
```

> **Note**: You can still monitor training with TensorBoard without wandb.

## GPU Setup

### CUDA (NVIDIA GPU)

```bash
# Check CUDA version
nvidia-smi

# Verify PyTorch CUDA
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### MPS (Apple Silicon)

```bash
# Check MPS (M1/M2/M3 Mac)
python -c "import torch; print(f'MPS available: {torch.backends.mps.is_available()}')"
```

## Troubleshooting

### NLTK Data Error

```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"
```

### PyTorch Version Conflict

```bash
# Uninstall and reinstall PyTorch
pip uninstall torch torchvision
pip install torch torchvision
```

### Out of Memory

Adjust the following settings in `configs/*.yaml`:
- `data.max_samples`: Reduce sample count
- `model.embedding_dim`: Reduce embedding dimension
