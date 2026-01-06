"""Vocabulary management."""

import json
from pathlib import Path
from typing import Optional


class Vocabulary:
    """
    Vocabulary class for word-to-index mapping.

    Special tokens:
        - <PAD>: Padding token (index 0)
        - <UNK>: Unknown token (index 1)
    """

    PAD_TOKEN = "<PAD>"
    UNK_TOKEN = "<UNK>"
    PAD_IDX = 0
    UNK_IDX = 1

    def __init__(self, word2idx: Optional[dict[str, int]] = None) -> None:
        """
        Initialize vocabulary.

        Args:
            word2idx: Optional pre-built word to index mapping
        """
        if word2idx is None:
            self.word2idx = {self.PAD_TOKEN: self.PAD_IDX, self.UNK_TOKEN: self.UNK_IDX}
        else:
            self.word2idx = word2idx

        self.idx2word = {idx: word for word, idx in self.word2idx.items()}

    def __len__(self) -> int:
        """Return vocabulary size."""
        return len(self.word2idx)

    def __contains__(self, word: str) -> bool:
        """Check if word is in vocabulary."""
        return word in self.word2idx

    @property
    def size(self) -> int:
        """Return vocabulary size."""
        return len(self.word2idx)

    def get_index(self, word: str) -> int:
        """
        Get index for a word.

        Args:
            word: The word to look up

        Returns:
            Index of the word, or UNK_IDX if not found
        """
        return self.word2idx.get(word, self.UNK_IDX)

    def get_word(self, idx: int) -> str:
        """
        Get word for an index.

        Args:
            idx: The index to look up

        Returns:
            Word at the index, or UNK_TOKEN if not found
        """
        return self.idx2word.get(idx, self.UNK_TOKEN)

    def encode(self, words: list[str]) -> list[int]:
        """
        Encode a list of words to indices.

        Args:
            words: List of words

        Returns:
            List of indices
        """
        return [self.get_index(word) for word in words]

    def decode(self, indices: list[int]) -> list[str]:
        """
        Decode a list of indices to words.

        Args:
            indices: List of indices

        Returns:
            List of words
        """
        return [self.get_word(idx) for idx in indices]

    def save(self, path: Path | str) -> None:
        """
        Save vocabulary to JSON file.

        Args:
            path: Path to save file
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.word2idx, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: Path | str) -> "Vocabulary":
        """
        Load vocabulary from JSON file.

        Args:
            path: Path to vocabulary file

        Returns:
            Vocabulary instance
        """
        with open(path, "r", encoding="utf-8") as f:
            word2idx = json.load(f)
        return cls(word2idx)

    @classmethod
    def from_dict(cls, word2idx: dict[str, int]) -> "Vocabulary":
        """
        Create vocabulary from dictionary.

        Args:
            word2idx: Word to index mapping

        Returns:
            Vocabulary instance
        """
        return cls(word2idx)
