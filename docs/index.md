# HMCAN Documentation

Hierarchical Multichannel CNN-based Attention Network for Document Classification

## Table of Contents

1. [Installation Guide](installation.md)
   - Requirements
   - uv / pip / conda installation
   - GPU setup
   - Troubleshooting

2. [Data Preparation](data.md)
   - Data download
   - Preprocessing
   - Custom data usage
   - Data format

3. [Training Guide](training.md)
   - Running training
   - Configuration files
   - TensorBoard monitoring
   - Hyperparameter tuning
   - Checkpoint management

4. [Model Architecture](models.md)
   - HAN (Hierarchical Attention Network)
   - HCAN (Hierarchical Cascaded Attention Network)
   - HMCAN (Hierarchical Multichannel CNN-based Attention Network)
   - Model comparison
   - Custom model creation

5. [Evaluation & Inference](evaluation.md)
   - Model evaluation
   - Single/batch inference
   - Attention visualization
   - Performance metrics
   - Model export (TorchScript, ONNX)

## Quick Start

```bash
# 1. Install
git clone https://github.com/sucpark/hmcan.git
cd hmcan
uv sync  # or pip install -e .

# 2. Prepare data
python scripts/download_data.py

# 3. Train
python -m hmcan train --config configs/hmcan.yaml

# 4. Evaluate
python -m hmcan evaluate --checkpoint outputs/hmcan_yelp/checkpoints/best_model.pt
```

## CLI Commands

| Command | Description |
|---------|-------------|
| `python -m hmcan train` | Train model |
| `python -m hmcan evaluate` | Evaluate model |
| `python -m hmcan download` | Download data |
| `python -m hmcan models` | List available models |

## Project Structure

```
hmcan/
├── configs/              # Configuration files
├── data/                 # Data
├── docs/                 # Documentation
├── notebooks/            # Jupyter notebooks
├── outputs/              # Training outputs
├── scripts/              # Utility scripts
├── src/hmcan/            # Source code
│   ├── config.py
│   ├── data/
│   ├── models/
│   ├── training/
│   └── utils/
└── tests/                # Tests
```

## References

- **HAN**: Yang et al., "Hierarchical Attention Networks for Document Classification" (NAACL 2016)
- **Transformer**: Vaswani et al., "Attention Is All You Need" (NeurIPS 2017)
- **GloVe**: Pennington et al., "GloVe: Global Vectors for Word Representation" (EMNLP 2014)

## License

MIT License
