"""Checkpoint management utilities."""

from pathlib import Path
from typing import Any, Optional
import torch
import torch.nn as nn
from torch.optim import Optimizer


class CheckpointManager:
    """
    Manages saving and loading of model checkpoints.

    Features:
        - Save best model based on validation metric
        - Keep only last N checkpoints
        - Resume training from checkpoint
    """

    def __init__(
        self,
        checkpoint_dir: Path | str,
        keep_last_n: int = 3,
    ) -> None:
        """
        Initialize checkpoint manager.

        Args:
            checkpoint_dir: Directory to save checkpoints
            keep_last_n: Number of recent checkpoints to keep
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.keep_last_n = keep_last_n
        self._checkpoints: list[Path] = []

    def save(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        epoch: int,
        metrics: dict[str, float],
        filename: Optional[str] = None,
    ) -> Path:
        """
        Save a checkpoint.

        Args:
            model: The model to save
            optimizer: The optimizer state
            epoch: Current epoch number
            metrics: Dictionary of metrics
            filename: Optional custom filename

        Returns:
            Path to the saved checkpoint
        """
        if filename is None:
            filename = f"checkpoint_epoch_{epoch:03d}.pt"

        checkpoint_path = self.checkpoint_dir / filename

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
        }

        torch.save(checkpoint, checkpoint_path)
        self._checkpoints.append(checkpoint_path)

        # Remove old checkpoints
        self._cleanup_old_checkpoints()

        return checkpoint_path

    def save_best(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        epoch: int,
        metrics: dict[str, float],
    ) -> Path:
        """
        Save the best model checkpoint.

        Args:
            model: The model to save
            optimizer: The optimizer state
            epoch: Current epoch number
            metrics: Dictionary of metrics

        Returns:
            Path to the saved checkpoint
        """
        return self.save(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            metrics=metrics,
            filename="best_model.pt",
        )

    def load(
        self,
        checkpoint_path: Path | str,
        model: nn.Module,
        optimizer: Optional[Optimizer] = None,
        device: Optional[torch.device] = None,
    ) -> dict[str, Any]:
        """
        Load a checkpoint.

        Args:
            checkpoint_path: Path to the checkpoint file
            model: The model to load weights into
            optimizer: Optional optimizer to load state into
            device: Device to load the checkpoint to

        Returns:
            Dictionary with checkpoint info (epoch, metrics)
        """
        checkpoint_path = Path(checkpoint_path)

        if device is None:
            checkpoint = torch.load(checkpoint_path, weights_only=False)
        else:
            checkpoint = torch.load(
                checkpoint_path, map_location=device, weights_only=False
            )

        model.load_state_dict(checkpoint["model_state_dict"])

        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        return {
            "epoch": checkpoint.get("epoch", 0),
            "metrics": checkpoint.get("metrics", {}),
        }

    def load_best(
        self,
        model: nn.Module,
        optimizer: Optional[Optimizer] = None,
        device: Optional[torch.device] = None,
    ) -> dict[str, Any]:
        """
        Load the best model checkpoint.

        Args:
            model: The model to load weights into
            optimizer: Optional optimizer to load state into
            device: Device to load the checkpoint to

        Returns:
            Dictionary with checkpoint info
        """
        best_path = self.checkpoint_dir / "best_model.pt"
        if not best_path.exists():
            raise FileNotFoundError(f"Best model not found at {best_path}")

        return self.load(best_path, model, optimizer, device)

    def _cleanup_old_checkpoints(self) -> None:
        """Remove old checkpoints, keeping only the last N."""
        # Don't delete best_model.pt
        regular_checkpoints = [
            p for p in self._checkpoints
            if p.name != "best_model.pt"
        ]

        while len(regular_checkpoints) > self.keep_last_n:
            old_checkpoint = regular_checkpoints.pop(0)
            if old_checkpoint.exists():
                old_checkpoint.unlink()
            if old_checkpoint in self._checkpoints:
                self._checkpoints.remove(old_checkpoint)

    def get_latest_checkpoint(self) -> Optional[Path]:
        """Get the path to the latest checkpoint."""
        checkpoints = sorted(
            self.checkpoint_dir.glob("checkpoint_epoch_*.pt"),
            key=lambda p: p.stat().st_mtime,
        )
        return checkpoints[-1] if checkpoints else None
