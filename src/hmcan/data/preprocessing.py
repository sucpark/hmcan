"""Text preprocessing utilities."""

import string
from typing import Optional

try:
    import nltk
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)
    from nltk.tokenize import sent_tokenize, word_tokenize
    HAS_NLTK = True
except ImportError:
    HAS_NLTK = False

from .vocabulary import Vocabulary


class DocumentPreprocessor:
    """
    Preprocessor for hierarchical document classification.

    Converts raw text to tokenized and indexed format:
        Document -> List of Sentences -> List of Word Indices
    """

    def __init__(
        self,
        lowercase: bool = True,
        remove_punctuation: bool = True,
        min_sentence_length: int = 1,
    ) -> None:
        """
        Initialize preprocessor.

        Args:
            lowercase: Convert text to lowercase
            remove_punctuation: Remove punctuation
            min_sentence_length: Minimum words per sentence
        """
        if not HAS_NLTK:
            raise ImportError("NLTK is required. Install with: pip install nltk")

        self.lowercase = lowercase
        self.remove_punctuation = remove_punctuation
        self.min_sentence_length = min_sentence_length
        self.punctuation_table = str.maketrans("", "", string.punctuation)

    def tokenize_document(self, text: str) -> list[list[str]]:
        """
        Tokenize document into sentences and words.

        Args:
            text: Raw document text

        Returns:
            List of sentences, each a list of words
        """
        if self.lowercase:
            text = text.lower()

        # Sentence tokenization
        sentences = sent_tokenize(text)

        processed_sentences = []
        for sent in sentences:
            # Remove punctuation
            if self.remove_punctuation:
                sent = sent.translate(self.punctuation_table)

            # Word tokenization
            words = word_tokenize(sent)

            # Filter short sentences
            if len(words) >= self.min_sentence_length:
                processed_sentences.append(words)

        return processed_sentences

    def process_document(
        self,
        text: str,
        vocabulary: Optional[Vocabulary] = None,
    ) -> list[list[int]]:
        """
        Process document to list of word indices.

        Args:
            text: Raw document text
            vocabulary: Vocabulary for word-to-index mapping

        Returns:
            List of sentences as word index lists
        """
        sentences = self.tokenize_document(text)

        if vocabulary is None:
            raise ValueError("Vocabulary is required for indexing")

        indexed_sentences = []
        for sent in sentences:
            indices = vocabulary.encode(sent)
            # Filter out all-unknown sentences
            if any(idx != Vocabulary.UNK_IDX for idx in indices):
                indexed_sentences.append(indices)

        return indexed_sentences

    def batch_process(
        self,
        texts: list[str],
        vocabulary: Vocabulary,
    ) -> list[list[list[int]]]:
        """
        Process multiple documents.

        Args:
            texts: List of raw texts
            vocabulary: Vocabulary for indexing

        Returns:
            List of processed documents
        """
        return [
            self.process_document(text, vocabulary)
            for text in texts
        ]
