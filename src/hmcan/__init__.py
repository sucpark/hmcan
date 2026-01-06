"""
HMCAN - Hierarchical Multichannel CNN-based Attention Network

A PyTorch implementation of hierarchical attention models for document classification.
Includes HAN, HCAN, and HMCAN architectures.
"""

__version__ = "0.1.0"
__author__ = "sucpark"

from .config import Config, ModelConfig, TrainingConfig, DataConfig

__all__ = [
    "Config",
    "ModelConfig",
    "TrainingConfig",
    "DataConfig",
    "__version__",
]
