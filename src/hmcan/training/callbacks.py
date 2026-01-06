"""Training callbacks."""

from abc import ABC
from pathlib import Path
from typing import Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .trainer import Trainer


class Callback(ABC):
    """Base callback class."""

    def on_train_begin(self, trainer: "Trainer") -> None:
        """Called at the beginning of training."""
        pass

    def on_train_end(self, trainer: "Trainer") -> None:
        """Called at the end of training."""
        pass

    def on_epoch_begin(self, epoch: int, trainer: "Trainer") -> None:
        """Called at the beginning of each epoch."""
        pass

    def on_epoch_end(
        self,
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float],
        trainer: "Trainer",
    ) -> None:
        """Called at the end of each epoch."""
        pass

    def on_batch_begin(self, step: int, trainer: "Trainer") -> None:
        """Called at the beginning of each batch."""
        pass

    def on_batch_end(self, step: int, loss: float, trainer: "Trainer") -> None:
        """Called at the end of each batch."""
        pass


class CallbackList:
    """Container for multiple callbacks."""

    def __init__(self, callbacks: Optional[List[Callback]] = None) -> None:
        """
        Initialize callback list.

        Args:
            callbacks: List of callbacks
        """
        self.callbacks = callbacks or []
        self.should_stop = False

    def on_train_begin(self, trainer: "Trainer") -> None:
        """Called at the beginning of training."""
        for cb in self.callbacks:
            cb.on_train_begin(trainer)

    def on_train_end(self, trainer: "Trainer") -> None:
        """Called at the end of training."""
        for cb in self.callbacks:
            cb.on_train_end(trainer)

    def on_epoch_begin(self, epoch: int, trainer: "Trainer") -> None:
        """Called at the beginning of each epoch."""
        for cb in self.callbacks:
            cb.on_epoch_begin(epoch, trainer)

    def on_epoch_end(
        self,
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float],
        trainer: "Trainer",
    ) -> None:
        """Called at the end of each epoch."""
        for cb in self.callbacks:
            cb.on_epoch_end(epoch, train_metrics, val_metrics, trainer)
            if hasattr(cb, "should_stop") and cb.should_stop:
                self.should_stop = True

    def on_batch_begin(self, step: int, trainer: "Trainer") -> None:
        """Called at the beginning of each batch."""
        for cb in self.callbacks:
            cb.on_batch_begin(step, trainer)

    def on_batch_end(self, step: int, loss: float, trainer: "Trainer") -> None:
        """Called at the end of each batch."""
        for cb in self.callbacks:
            cb.on_batch_end(step, loss, trainer)


class EarlyStopping(Callback):
    """
    Early stopping based on validation metric.

    Stops training when the monitored metric stops improving.
    """

    def __init__(
        self,
        patience: int = 5,
        metric: str = "accuracy",
        mode: str = "max",
        min_delta: float = 0.0,
    ) -> None:
        """
        Initialize early stopping.

        Args:
            patience: Number of epochs to wait for improvement
            metric: Metric to monitor
            mode: 'max' if higher is better, 'min' if lower is better
            min_delta: Minimum change to qualify as improvement
        """
        self.patience = patience
        self.metric = metric
        self.mode = mode
        self.min_delta = min_delta
        self.best_value: Optional[float] = None
        self.counter = 0
        self.should_stop = False

    def on_train_begin(self, trainer: "Trainer") -> None:
        """Reset state at training start."""
        self.best_value = None
        self.counter = 0
        self.should_stop = False

    def on_epoch_end(
        self,
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float],
        trainer: "Trainer",
    ) -> None:
        """Check for improvement."""
        current = val_metrics.get(self.metric)
        if current is None:
            return

        if self.best_value is None:
            self.best_value = current
        elif self._is_improvement(current):
            self.best_value = current
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                print(f"Early stopping triggered after {epoch + 1} epochs")

    def _is_improvement(self, current: float) -> bool:
        """Check if current value is an improvement."""
        if self.mode == "max":
            return current > self.best_value + self.min_delta
        return current < self.best_value - self.min_delta


class ModelCheckpoint(Callback):
    """
    Save model checkpoints during training.

    Saves the best model and optionally periodic checkpoints.
    """

    def __init__(
        self,
        checkpoint_dir: Path | str,
        metric: str = "accuracy",
        mode: str = "max",
        save_every_n_epochs: int = 1,
    ) -> None:
        """
        Initialize checkpoint callback.

        Args:
            checkpoint_dir: Directory to save checkpoints
            metric: Metric to monitor for best model
            mode: 'max' if higher is better, 'min' if lower is better
            save_every_n_epochs: Save checkpoint every N epochs
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.metric = metric
        self.mode = mode
        self.save_every_n_epochs = save_every_n_epochs
        self.best_value: Optional[float] = None

    def on_epoch_end(
        self,
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float],
        trainer: "Trainer",
    ) -> None:
        """Save checkpoints."""
        current = val_metrics.get(self.metric)

        # Save periodic checkpoint
        if (epoch + 1) % self.save_every_n_epochs == 0:
            if trainer.checkpoint_manager:
                trainer.checkpoint_manager.save(
                    trainer.model,
                    trainer.optimizer,
                    epoch,
                    val_metrics,
                )

        # Save best model
        if current is not None:
            if self.best_value is None or self._is_better(current):
                self.best_value = current
                if trainer.checkpoint_manager:
                    trainer.checkpoint_manager.save_best(
                        trainer.model,
                        trainer.optimizer,
                        epoch,
                        val_metrics,
                    )
                    print(f"Saved best model with {self.metric}={current:.4f}")

    def _is_better(self, current: float) -> bool:
        """Check if current is better than best."""
        if self.mode == "max":
            return current > self.best_value
        return current < self.best_value


class TensorBoardLogger(Callback):
    """Log metrics to TensorBoard."""

    def __init__(self, log_dir: Path | str) -> None:
        """
        Initialize TensorBoard logger.

        Args:
            log_dir: Directory for TensorBoard logs
        """
        from torch.utils.tensorboard import SummaryWriter

        self.log_dir = Path(log_dir)
        self.writer = SummaryWriter(log_dir=str(self.log_dir))

    def on_epoch_end(
        self,
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float],
        trainer: "Trainer",
    ) -> None:
        """Log epoch metrics."""
        for key, value in train_metrics.items():
            self.writer.add_scalar(f"train/{key}", value, epoch)
        for key, value in val_metrics.items():
            self.writer.add_scalar(f"val/{key}", value, epoch)

    def on_train_end(self, trainer: "Trainer") -> None:
        """Close writer."""
        self.writer.close()
