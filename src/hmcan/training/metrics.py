"""Training metrics tracking."""

from typing import Dict, List, Optional
import torch


class MetricsTracker:
    """
    Track training and evaluation metrics.

    Tracks:
        - Loss (average)
        - Accuracy
        - Per-class accuracy (optional)
    """

    def __init__(self, num_classes: Optional[int] = None) -> None:
        """
        Initialize metrics tracker.

        Args:
            num_classes: Number of classes for per-class metrics
        """
        self.num_classes = num_classes
        self.reset()

    def reset(self) -> None:
        """Reset all metrics."""
        self.total_loss = 0.0
        self.total_correct = 0
        self.total_samples = 0
        self.all_predictions: List[int] = []
        self.all_labels: List[int] = []

    def update(
        self,
        loss: float,
        predictions: torch.Tensor,
        labels: torch.Tensor,
    ) -> None:
        """
        Update metrics with batch results.

        Args:
            loss: Batch loss value
            predictions: Predicted class indices
            labels: True class indices
        """
        batch_size = labels.numel()

        self.total_loss += loss * batch_size
        self.total_samples += batch_size

        # Handle different shapes
        if predictions.dim() > 1:
            predictions = predictions.view(-1)
        if labels.dim() > 1:
            labels = labels.view(-1)

        correct = (predictions == labels).sum().item()
        self.total_correct += correct

        self.all_predictions.extend(predictions.cpu().tolist())
        self.all_labels.extend(labels.cpu().tolist())

    def get_loss(self) -> float:
        """Get average loss."""
        if self.total_samples == 0:
            return 0.0
        return self.total_loss / self.total_samples

    def get_accuracy(self) -> float:
        """Get accuracy."""
        if self.total_samples == 0:
            return 0.0
        return self.total_correct / self.total_samples

    def compute(self) -> Dict[str, float]:
        """
        Compute all metrics.

        Returns:
            Dictionary with metrics
        """
        metrics = {
            "loss": self.get_loss(),
            "accuracy": self.get_accuracy(),
        }

        # Per-class accuracy if num_classes is set
        if self.num_classes is not None:
            per_class_correct = [0] * self.num_classes
            per_class_total = [0] * self.num_classes

            for pred, label in zip(self.all_predictions, self.all_labels):
                per_class_total[label] += 1
                if pred == label:
                    per_class_correct[label] += 1

            for i in range(self.num_classes):
                if per_class_total[i] > 0:
                    metrics[f"accuracy_class_{i}"] = (
                        per_class_correct[i] / per_class_total[i]
                    )

        return metrics

    def get_confusion_matrix(self) -> Optional[torch.Tensor]:
        """
        Get confusion matrix.

        Returns:
            Confusion matrix tensor or None if no predictions
        """
        if self.num_classes is None or len(self.all_predictions) == 0:
            return None

        matrix = torch.zeros(self.num_classes, self.num_classes, dtype=torch.long)

        for pred, label in zip(self.all_predictions, self.all_labels):
            matrix[label, pred] += 1

        return matrix
