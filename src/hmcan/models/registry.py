"""Model registry for creating models by name."""

from typing import Dict, Type, Optional, Any
import torch

from .base import BaseHierarchicalModel
from .han import HAN
from .hcan import HCAN
from .hmcan import HMCAN


# Model registry
MODEL_REGISTRY: Dict[str, Type[BaseHierarchicalModel]] = {
    "han": HAN,
    "hcan": HCAN,
    "hmcan": HMCAN,
}


def create_model(
    name: str,
    vocab_size: int,
    pretrained_embeddings: Optional[torch.Tensor] = None,
    **kwargs: Any,
) -> BaseHierarchicalModel:
    """
    Create a model by name.

    Args:
        name: Model name ('han', 'hcan', 'hmcan')
        vocab_size: Size of vocabulary
        pretrained_embeddings: Optional pretrained embedding matrix
        **kwargs: Additional model-specific arguments

    Returns:
        Model instance

    Raises:
        ValueError: If model name is not recognized
    """
    name = name.lower()

    if name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model: {name}. "
            f"Available models: {list(MODEL_REGISTRY.keys())}"
        )

    model_class = MODEL_REGISTRY[name]

    # Filter kwargs to only include valid arguments for the model
    # This handles when config has parameters for other models
    import inspect
    valid_params = set(inspect.signature(model_class.__init__).parameters.keys())
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_params}

    return model_class(
        vocab_size=vocab_size,
        pretrained_embeddings=pretrained_embeddings,
        **filtered_kwargs,
    )


def get_available_models() -> list[str]:
    """Get list of available model names."""
    return list(MODEL_REGISTRY.keys())


def register_model(name: str, model_class: Type[BaseHierarchicalModel]) -> None:
    """
    Register a new model.

    Args:
        name: Model name
        model_class: Model class
    """
    MODEL_REGISTRY[name.lower()] = model_class
