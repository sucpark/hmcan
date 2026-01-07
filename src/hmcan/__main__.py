"""CLI entry point for HMCAN."""

from pathlib import Path
from typing import Optional

import click
import torch
import torch.nn as nn

from .config import Config
from .data import YelpDataModule
from .models import create_model
from .training import Trainer, EarlyStopping, ModelCheckpoint, TensorBoardLogger, WandbLogger
from .utils import set_seed, get_device, CheckpointManager


@click.group()
@click.version_option(version="0.1.0")
def cli():
    """HMCAN - Hierarchical Multichannel CNN-based Attention Network for Document Classification."""
    pass


@cli.command()
@click.option(
    "--config", "-c",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to config YAML file",
)
@click.option(
    "--resume",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Resume from checkpoint",
)
@click.option("--seed", type=int, default=None, help="Override random seed")
@click.option("--device", type=str, default=None, help="Override device (cpu, cuda, mps)")
def train(config: Path, resume: Optional[Path], seed: Optional[int], device: Optional[str]):
    """Train a hierarchical attention model."""
    # Load config
    cfg = Config.from_yaml(config)
    if seed is not None:
        cfg.seed = seed
    if device is not None:
        cfg.device = device

    click.echo(f"Training {cfg.model.name.upper()} model")
    click.echo(f"Config: {config}")

    # Set seed for reproducibility
    set_seed(cfg.seed)

    # Setup device
    device_obj = get_device(cfg.device)
    click.echo(f"Device: {device_obj}")

    # Setup data
    data_module = YelpDataModule(
        data_dir=Path(cfg.data.data_dir),
        vocab_path=Path(cfg.data.vocab_path) if cfg.data.vocab_path else None,
        embeddings_path=Path(cfg.data.embeddings_path) if cfg.data.embeddings_path else None,
        train_ratio=cfg.data.train_ratio,
        val_ratio=cfg.data.val_ratio,
        test_ratio=cfg.data.test_ratio,
        random_seed=cfg.seed,
        num_workers=cfg.data.num_workers,
        max_samples=cfg.data.max_samples,
    )
    data_module.setup()

    # Get vocab size from data
    vocab_size = len(data_module.vocabulary) if data_module.vocabulary else cfg.model.vocab_size

    # Create model
    model = create_model(
        name=cfg.model.name,
        vocab_size=vocab_size,
        pretrained_embeddings=data_module.pretrained_embeddings,
        embedding_dim=cfg.model.embedding_dim,
        attention_dim=cfg.model.attention_dim,
        num_classes=cfg.model.num_classes,
        dropout=cfg.model.dropout,
        hidden_dim=cfg.model.hidden_dim,
        num_heads=cfg.model.num_heads,
        max_words=cfg.model.max_words,
        max_sentences=cfg.model.max_sentences,
        conv_kernel_size=cfg.model.conv_kernel_size,
        freeze_pretrained=cfg.model.freeze_pretrained,
    )

    click.echo(f"Model parameters: {model.count_parameters():,} trainable, {model.count_all_parameters():,} total")

    # Setup optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.training.learning_rate,
        betas=(cfg.training.beta1, cfg.training.beta2),
        eps=cfg.training.eps,
        weight_decay=cfg.training.weight_decay,
    )

    # Loss function
    criterion = nn.CrossEntropyLoss()

    # Setup callbacks
    callbacks = []

    output_dir = Path(cfg.output_dir) / cfg.experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)

    if cfg.training.early_stopping:
        callbacks.append(EarlyStopping(patience=cfg.training.patience))

    if cfg.use_tensorboard:
        callbacks.append(TensorBoardLogger(output_dir / "logs"))

    if cfg.use_wandb:
        wandb_config = {
            "model": cfg.model.name,
            "embedding_dim": cfg.model.embedding_dim,
            "attention_dim": cfg.model.attention_dim,
            "dropout": cfg.model.dropout,
            "learning_rate": cfg.training.learning_rate,
            "num_epochs": cfg.training.num_epochs,
            "seed": cfg.seed,
        }
        callbacks.append(WandbLogger(
            project="hmcan",
            name=cfg.experiment_name,
            config=wandb_config,
            tags=[cfg.model.name],
        ))

    # Setup trainer
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device_obj,
        callbacks=callbacks,
        checkpoint_dir=output_dir / "checkpoints",
        max_grad_norm=cfg.training.max_grad_norm,
        num_classes=cfg.model.num_classes,
    )

    # Resume from checkpoint if specified
    start_epoch = 0
    if resume:
        info = trainer.load_checkpoint(resume)
        start_epoch = info.get("epoch", 0) + 1
        click.echo(f"Resumed from epoch {start_epoch}")

    # Train
    history = trainer.fit(
        train_loader=data_module.train_dataloader(),
        val_loader=data_module.val_dataloader(),
        num_epochs=cfg.training.num_epochs,
    )

    # Save final config
    cfg.to_yaml(output_dir / "config.yaml")

    click.echo(f"\nTraining complete!")
    click.echo(f"Best validation accuracy: {max(history['val_acc'])*100:.2f}%")
    click.echo(f"Results saved to: {output_dir}")


@cli.command()
@click.option(
    "--checkpoint", "-c",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to model checkpoint",
)
@click.option(
    "--config",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Path to config YAML (optional, will look for config.yaml in checkpoint dir)",
)
@click.option("--device", type=str, default="auto", help="Device (cpu, cuda, mps)")
def evaluate(checkpoint: Path, config: Optional[Path], device: str):
    """Evaluate a trained model on test set."""
    # Find config
    if config is None:
        config = checkpoint.parent.parent / "config.yaml"
        if not config.exists():
            raise click.ClickException(f"Config not found at {config}. Please specify --config")

    cfg = Config.from_yaml(config)
    device_obj = get_device(device)

    click.echo(f"Evaluating {cfg.model.name.upper()} model")
    click.echo(f"Checkpoint: {checkpoint}")

    # Setup data
    data_module = YelpDataModule(
        data_dir=Path(cfg.data.data_dir),
        vocab_path=Path(cfg.data.vocab_path) if cfg.data.vocab_path else None,
        embeddings_path=Path(cfg.data.embeddings_path) if cfg.data.embeddings_path else None,
        random_seed=cfg.seed,
    )
    data_module.setup()

    vocab_size = len(data_module.vocabulary) if data_module.vocabulary else cfg.model.vocab_size

    # Create model
    model = create_model(
        name=cfg.model.name,
        vocab_size=vocab_size,
        pretrained_embeddings=data_module.pretrained_embeddings,
        embedding_dim=cfg.model.embedding_dim,
        attention_dim=cfg.model.attention_dim,
        num_classes=cfg.model.num_classes,
        dropout=cfg.model.dropout,
        hidden_dim=cfg.model.hidden_dim,
        num_heads=cfg.model.num_heads,
        conv_kernel_size=cfg.model.conv_kernel_size,
    )

    # Load checkpoint
    ckpt = torch.load(checkpoint, map_location=device_obj, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device_obj)

    # Evaluate
    criterion = nn.CrossEntropyLoss()
    trainer = Trainer(
        model=model,
        optimizer=torch.optim.Adam(model.parameters()),  # Dummy optimizer
        criterion=criterion,
        device=device_obj,
        num_classes=cfg.model.num_classes,
    )

    metrics = trainer.evaluate(data_module.test_dataloader())

    click.echo(f"\nTest Results:")
    click.echo(f"  Loss: {metrics['loss']:.4f}")
    click.echo(f"  Accuracy: {metrics['accuracy']*100:.2f}%")


@cli.command()
@click.option("--data-dir", type=str, default="data", help="Data directory")
@click.option("--max-samples", type=int, default=10000, help="Maximum samples")
@click.option("--embedding-dim", type=int, default=50, help="Embedding dimension")
def download(data_dir: str, max_samples: int, embedding_dim: int):
    """Download and prepare the Yelp dataset."""
    import subprocess
    import sys

    # Find script in common locations
    candidates = [
        Path(__file__).parent.parent.parent.parent / "scripts" / "download_data.py",
        Path.cwd() / "scripts" / "download_data.py",
    ]

    script_path = None
    for candidate in candidates:
        if candidate.exists():
            script_path = candidate
            break

    if script_path is None:
        raise click.ClickException(
            "download_data.py script not found. Please run from the project root directory."
        )

    result = subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--data-dir", data_dir,
            "--max-samples", str(max_samples),
            "--embedding-dim", str(embedding_dim),
        ],
        check=False,
    )

    if result.returncode != 0:
        raise click.ClickException(f"Data download failed with code {result.returncode}")


@cli.command()
def models():
    """List available models."""
    from .models import get_available_models
    click.echo("Available models:")
    for name in get_available_models():
        click.echo(f"  - {name}")


if __name__ == "__main__":
    cli()
