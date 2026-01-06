"""Data module for loading and managing datasets."""

from pathlib import Path
from typing import Optional
import json

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from .dataset import HierarchicalDocumentDataset
from .collate import hierarchical_collate_fn
from .vocabulary import Vocabulary


class YelpDataModule:
    """
    Data module for Yelp review dataset.

    Handles:
        - Loading processed data
        - Loading vocabulary and embeddings
        - Train/val/test splitting
        - Creating DataLoaders
    """

    def __init__(
        self,
        data_dir: Path | str,
        vocab_path: Optional[Path | str] = None,
        embeddings_path: Optional[Path | str] = None,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        random_seed: int = 14,
        num_workers: int = 0,
        max_samples: Optional[int] = None,
    ) -> None:
        """
        Initialize data module.

        Args:
            data_dir: Directory containing processed data
            vocab_path: Path to vocabulary JSON file
            embeddings_path: Path to embeddings NPZ file
            train_ratio: Training set ratio
            val_ratio: Validation set ratio
            test_ratio: Test set ratio
            random_seed: Random seed for splitting
            num_workers: DataLoader workers
            max_samples: Limit samples for debugging
        """
        self.data_dir = Path(data_dir)
        self.vocab_path = Path(vocab_path) if vocab_path else None
        self.embeddings_path = Path(embeddings_path) if embeddings_path else None
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.random_seed = random_seed
        self.num_workers = num_workers
        self.max_samples = max_samples

        self.vocabulary: Optional[Vocabulary] = None
        self.pretrained_embeddings: Optional[torch.Tensor] = None

        self.train_dataset: Optional[HierarchicalDocumentDataset] = None
        self.val_dataset: Optional[HierarchicalDocumentDataset] = None
        self.test_dataset: Optional[HierarchicalDocumentDataset] = None

    def setup(self) -> None:
        """Load and prepare data."""
        # Load vocabulary
        if self.vocab_path and self.vocab_path.exists():
            self.vocabulary = Vocabulary.load(self.vocab_path)
            print(f"Loaded vocabulary with {len(self.vocabulary)} words")
        else:
            # Try default path
            default_vocab = self.data_dir / "processed" / "word2idx.json"
            if default_vocab.exists():
                self.vocabulary = Vocabulary.load(default_vocab)
                print(f"Loaded vocabulary with {len(self.vocabulary)} words")

        # Load pretrained embeddings
        if self.embeddings_path and self.embeddings_path.exists():
            npz = np.load(self.embeddings_path)
            self.pretrained_embeddings = torch.tensor(
                npz["embeddings"], dtype=torch.float32
            )
            print(f"Loaded embeddings with shape {self.pretrained_embeddings.shape}")
        else:
            # Try default path
            default_emb = self.data_dir / "processed" / "embeddings_50d.npz"
            if default_emb.exists():
                npz = np.load(default_emb)
                self.pretrained_embeddings = torch.tensor(
                    npz["embeddings"], dtype=torch.float32
                )
                print(f"Loaded embeddings with shape {self.pretrained_embeddings.shape}")

        # Load processed data
        data_path = self.data_dir / "processed" / "yelp_processed.npz"
        if not data_path.exists():
            raise FileNotFoundError(
                f"Processed data not found at {data_path}. "
                "Run scripts/download_data.py first."
            )

        npz = np.load(data_path, allow_pickle=True)
        all_data = list(npz["data"])

        # Limit samples if specified
        if self.max_samples is not None:
            all_data = all_data[: self.max_samples]

        # Extract documents and labels
        documents = [item["document"] for item in all_data]
        labels = [item["label"] for item in all_data]

        # Split data
        X_train, X_temp, y_train, y_temp = train_test_split(
            documents,
            labels,
            test_size=(self.val_ratio + self.test_ratio),
            random_state=self.random_seed,
            stratify=labels,
        )

        relative_test = self.test_ratio / (self.val_ratio + self.test_ratio)
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp,
            y_temp,
            test_size=relative_test,
            random_state=self.random_seed,
            stratify=y_temp,
        )

        # Create datasets
        self.train_dataset = HierarchicalDocumentDataset(X_train, y_train)
        self.val_dataset = HierarchicalDocumentDataset(X_val, y_val)
        self.test_dataset = HierarchicalDocumentDataset(X_test, y_test)

        print(f"Train: {len(self.train_dataset)}, Val: {len(self.val_dataset)}, Test: {len(self.test_dataset)}")

    def train_dataloader(self) -> DataLoader:
        """Get training DataLoader."""
        if self.train_dataset is None:
            raise RuntimeError("Call setup() first")
        return DataLoader(
            self.train_dataset,
            batch_size=1,  # Process one document at a time
            shuffle=True,
            collate_fn=hierarchical_collate_fn,
            num_workers=self.num_workers,
        )

    def val_dataloader(self) -> DataLoader:
        """Get validation DataLoader."""
        if self.val_dataset is None:
            raise RuntimeError("Call setup() first")
        return DataLoader(
            self.val_dataset,
            batch_size=1,
            shuffle=False,
            collate_fn=hierarchical_collate_fn,
            num_workers=self.num_workers,
        )

    def test_dataloader(self) -> DataLoader:
        """Get test DataLoader."""
        if self.test_dataset is None:
            raise RuntimeError("Call setup() first")
        return DataLoader(
            self.test_dataset,
            batch_size=1,
            shuffle=False,
            collate_fn=hierarchical_collate_fn,
            num_workers=self.num_workers,
        )
