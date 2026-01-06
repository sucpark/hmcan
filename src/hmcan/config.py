"""Configuration management using dataclasses."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Any
import yaml


@dataclass
class ModelConfig:
    """Model architecture configuration."""

    name: str = "hmcan"  # han, hcan, hmcan
    vocab_size: int = 50000
    embedding_dim: int = 50
    attention_dim: int = 50
    num_classes: int = 5
    dropout: float = 0.1

    # HMCAN-specific
    conv_kernel_size: int = 3
    freeze_pretrained: bool = True

    # HCAN-specific
    num_heads: int = 5
    max_words: int = 200
    max_sentences: int = 50
    activation: str = "relu"  # relu, elu, tanh

    # HAN-specific
    hidden_dim: int = 50
    bidirectional: bool = True


@dataclass
class TrainingConfig:
    """Training configuration."""

    num_epochs: int = 30
    learning_rate: float = 2e-5
    weight_decay: float = 0.0

    # Adam parameters
    beta1: float = 0.9
    beta2: float = 0.99
    eps: float = 1e-8

    # Scheduler
    scheduler: Optional[str] = None  # None, 'cosine', 'linear', 'step'
    warmup_steps: int = 0

    # Regularization
    max_grad_norm: Optional[float] = 1.0

    # Early stopping
    early_stopping: bool = True
    patience: int = 5

    # Checkpointing
    save_every_n_epochs: int = 1
    keep_last_n_checkpoints: int = 3


@dataclass
class DataConfig:
    """Data configuration."""

    data_dir: str = "data"
    vocab_path: Optional[str] = None
    embeddings_path: Optional[str] = None

    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1

    max_samples: Optional[int] = None  # Limit samples for debugging
    num_workers: int = 0
    pin_memory: bool = True


@dataclass
class Config:
    """Complete configuration."""

    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)

    # General
    seed: int = 14
    device: str = "auto"  # auto, cpu, cuda, mps
    output_dir: str = "outputs"
    experiment_name: str = "hmcan_experiment"

    # Logging
    use_tensorboard: bool = True
    use_wandb: bool = False
    wandb_project: Optional[str] = None
    log_every_n_steps: int = 100

    @classmethod
    def from_yaml(cls, path: Path | str) -> "Config":
        """Load configuration from YAML file."""
        path = Path(path)
        with open(path) as f:
            data = yaml.safe_load(f)

        return cls._from_dict(data)

    @classmethod
    def _from_dict(cls, data: dict[str, Any]) -> "Config":
        """Create Config from dictionary."""
        model_data = data.pop("model", {})
        training_data = data.pop("training", {})
        data_config_data = data.pop("data", {})

        return cls(
            model=ModelConfig(**model_data),
            training=TrainingConfig(**training_data),
            data=DataConfig(**data_config_data),
            **data,
        )

    def to_yaml(self, path: Path | str) -> None:
        """Save configuration to YAML file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = self._to_dict()
        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    def _to_dict(self) -> dict[str, Any]:
        """Convert Config to dictionary."""
        from dataclasses import asdict
        return {
            "model": asdict(self.model),
            "training": asdict(self.training),
            "data": asdict(self.data),
            "seed": self.seed,
            "device": self.device,
            "output_dir": self.output_dir,
            "experiment_name": self.experiment_name,
            "use_tensorboard": self.use_tensorboard,
            "use_wandb": self.use_wandb,
            "wandb_project": self.wandb_project,
            "log_every_n_steps": self.log_every_n_steps,
        }

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        assert self.model.name in ["han", "hcan", "hmcan"], \
            f"Unknown model: {self.model.name}"
        assert self.data.train_ratio + self.data.val_ratio + self.data.test_ratio == 1.0, \
            "Train/val/test ratios must sum to 1.0"
        assert self.model.dropout >= 0.0 and self.model.dropout < 1.0, \
            "Dropout must be in [0, 1)"
