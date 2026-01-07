"""Training loop implementation."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

if TYPE_CHECKING:
    from torch.optim.lr_scheduler import LRScheduler

from ..models.base import BaseHierarchicalModel
from ..utils.checkpoint import CheckpointManager
from .callbacks import Callback, CallbackList
from .metrics import MetricsTracker


class Trainer:
    """
    Training loop for hierarchical document classification models.

    Features:
        - Epoch-based training with progress bars
        - Validation at end of each epoch
        - Early stopping
        - Model checkpointing
        - Logging to TensorBoard
        - Reproducible training
    """

    def __init__(
        self,
        model: BaseHierarchicalModel,
        optimizer: Optimizer,
        criterion: nn.Module,
        device: torch.device,
        scheduler: Optional[LRScheduler] = None,
        callbacks: Optional[List[Callback]] = None,
        checkpoint_dir: Optional[Path | str] = None,
        max_grad_norm: Optional[float] = None,
        num_classes: int = 5,
    ) -> None:
        """
        Initialize trainer.

        Args:
            model: Model to train
            optimizer: Optimizer
            criterion: Loss function
            device: Device to train on
            scheduler: Optional learning rate scheduler
            callbacks: List of callbacks
            checkpoint_dir: Directory for checkpoints
            max_grad_norm: Maximum gradient norm for clipping
            num_classes: Number of output classes
        """
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.scheduler = scheduler
        self.callbacks = CallbackList(callbacks or [])
        self.max_grad_norm = max_grad_norm
        self.num_classes = num_classes

        # Checkpoint manager
        if checkpoint_dir:
            self.checkpoint_manager = CheckpointManager(Path(checkpoint_dir))
        else:
            self.checkpoint_manager = None

        # Metrics
        self.metrics_tracker = MetricsTracker(num_classes)

        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_metric = 0.0

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int,
        val_metric: str = "accuracy",
    ) -> Dict[str, List[float]]:
        """
        Train the model.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of epochs to train
            val_metric: Metric to use for model selection

        Returns:
            Training history with metrics per epoch
        """
        history = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
        }

        self.callbacks.on_train_begin(self)

        for epoch in range(num_epochs):
            self.current_epoch = epoch
            self.callbacks.on_epoch_begin(epoch, self)

            # Training phase
            train_metrics = self._train_epoch(train_loader)
            history["train_loss"].append(train_metrics["loss"])
            history["train_acc"].append(train_metrics["accuracy"])

            # Validation phase
            val_metrics = self._validate(val_loader)
            history["val_loss"].append(val_metrics["loss"])
            history["val_acc"].append(val_metrics["accuracy"])

            # Learning rate scheduling
            if self.scheduler is not None:
                self.scheduler.step()

            # Model selection (save best)
            val_score = val_metrics.get(val_metric, 0)
            if val_score > self.best_val_metric:
                self.best_val_metric = val_score
                if self.checkpoint_manager:
                    self.checkpoint_manager.save_best(
                        self.model, self.optimizer, epoch, val_metrics
                    )

            # Epoch logging
            lr = self.optimizer.param_groups[0]["lr"]
            print(
                f"Epoch {epoch+1}/{num_epochs} - "
                f"Train Loss: {train_metrics['loss']:.4f}, "
                f"Train Acc: {train_metrics['accuracy']*100:.2f}%, "
                f"Val Loss: {val_metrics['loss']:.4f}, "
                f"Val Acc: {val_metrics['accuracy']*100:.2f}%, "
                f"LR: {lr:.2e}"
            )

            self.callbacks.on_epoch_end(epoch, train_metrics, val_metrics, self)

            # Early stopping check
            if self.callbacks.should_stop:
                print(f"Early stopping at epoch {epoch + 1}")
                break

        self.callbacks.on_train_end(self)

        return history

    def _train_epoch(self, loader: DataLoader) -> Dict[str, float]:
        """Run one training epoch."""
        self.model.train()
        self.metrics_tracker.reset()

        progress_bar = tqdm(
            loader,
            desc=f"Epoch {self.current_epoch + 1} [Train]",
            leave=False,
        )

        for batch in progress_bar:
            self.callbacks.on_batch_begin(self.global_step, self)

            # Move data to device
            document = batch["document"].to(self.device)
            sentence_lengths = batch["sentence_lengths"].to(self.device)
            labels = batch["label"].to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(document, sentence_lengths)
            logits = outputs["logits"]

            # Compute loss
            # Handle shape differences (logits might be (1, num_classes))
            if logits.dim() == 2 and labels.dim() == 0:
                labels = labels.unsqueeze(0)
            loss = self.criterion(logits, labels)

            # Backward pass
            loss.backward()

            # Gradient clipping
            if self.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.max_grad_norm
                )

            self.optimizer.step()

            # Update metrics
            predictions = torch.argmax(logits, dim=-1)
            self.metrics_tracker.update(
                loss=loss.item(),
                predictions=predictions,
                labels=labels,
            )

            # Update progress bar
            progress_bar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "acc": f"{self.metrics_tracker.get_accuracy()*100:.2f}%",
            })

            self.callbacks.on_batch_end(self.global_step, loss.item(), self)
            self.global_step += 1

        return self.metrics_tracker.compute()

    @torch.no_grad()
    def _validate(self, loader: DataLoader) -> Dict[str, float]:
        """Run validation."""
        self.model.eval()
        self.metrics_tracker.reset()

        progress_bar = tqdm(
            loader,
            desc=f"Epoch {self.current_epoch + 1} [Val]",
            leave=False,
        )

        for batch in progress_bar:
            document = batch["document"].to(self.device)
            sentence_lengths = batch["sentence_lengths"].to(self.device)
            labels = batch["label"].to(self.device)

            outputs = self.model(document, sentence_lengths)
            logits = outputs["logits"]

            if logits.dim() == 2 and labels.dim() == 0:
                labels = labels.unsqueeze(0)
            loss = self.criterion(logits, labels)

            predictions = torch.argmax(logits, dim=-1)
            self.metrics_tracker.update(
                loss=loss.item(),
                predictions=predictions,
                labels=labels,
            )

        return self.metrics_tracker.compute()

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> Dict[str, float]:
        """
        Evaluate model on a dataset.

        Args:
            loader: Data loader

        Returns:
            Dictionary with evaluation metrics
        """
        return self._validate(loader)

    def load_checkpoint(self, checkpoint_path: Path | str) -> Dict:
        """
        Load a checkpoint.

        Args:
            checkpoint_path: Path to checkpoint

        Returns:
            Checkpoint info (epoch, metrics)
        """
        if self.checkpoint_manager is None:
            self.checkpoint_manager = CheckpointManager(
                Path(checkpoint_path).parent
            )

        return self.checkpoint_manager.load(
            checkpoint_path,
            self.model,
            self.optimizer,
            self.device,
        )
