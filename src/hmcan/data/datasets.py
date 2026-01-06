"""
Multi-dataset support for document classification.

Supports loading various datasets from HuggingFace:
- Yelp Review Full (5-class sentiment)
- IMDB (2-class sentiment)
- AG News (4-class topic)
- DBpedia (14-class topic)
- Yahoo Answers (10-class topic)
- 20 Newsgroups (20-class topic)
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Callable
from datasets import load_dataset, Dataset


@dataclass
class DatasetInfo:
    """Dataset metadata."""
    name: str
    hf_name: str  # HuggingFace dataset name
    num_classes: int
    text_column: str
    label_column: str
    description: str
    avg_length: str  # short, medium, long
    subset: Optional[str] = None

    def __str__(self):
        return f"{self.name}: {self.num_classes}-class, {self.avg_length} documents"


# Dataset registry
DATASETS: Dict[str, DatasetInfo] = {
    "yelp": DatasetInfo(
        name="Yelp Review Full",
        hf_name="yelp_review_full",
        num_classes=5,
        text_column="text",
        label_column="label",
        description="Restaurant reviews with 1-5 star ratings",
        avg_length="medium",
    ),
    "imdb": DatasetInfo(
        name="IMDB",
        hf_name="imdb",
        num_classes=2,
        text_column="text",
        label_column="label",
        description="Movie reviews (positive/negative)",
        avg_length="medium",
    ),
    "ag_news": DatasetInfo(
        name="AG News",
        hf_name="ag_news",
        num_classes=4,
        text_column="text",
        label_column="label",
        description="News articles (World, Sports, Business, Tech)",
        avg_length="short",
    ),
    "dbpedia": DatasetInfo(
        name="DBpedia",
        hf_name="dbpedia_14",
        num_classes=14,
        text_column="content",
        label_column="label",
        description="Wikipedia articles by category",
        avg_length="short",
    ),
    "yahoo": DatasetInfo(
        name="Yahoo Answers",
        hf_name="yahoo_answers_topics",
        num_classes=10,
        text_column="question_content",
        label_column="topic",
        description="Q&A topics",
        avg_length="medium",
    ),
    "newsgroups": DatasetInfo(
        name="20 Newsgroups",
        hf_name="SetFit/20_newsgroups",
        num_classes=20,
        text_column="text",
        label_column="label",
        description="Newsgroup messages by topic",
        avg_length="medium",
    ),
}


# Class labels for each dataset
CLASS_LABELS: Dict[str, List[str]] = {
    "yelp": ["1 star", "2 stars", "3 stars", "4 stars", "5 stars"],
    "imdb": ["Negative", "Positive"],
    "ag_news": ["World", "Sports", "Business", "Sci/Tech"],
    "dbpedia": [
        "Company", "EducationalInstitution", "Artist", "Athlete",
        "OfficeHolder", "MeanOfTransportation", "Building", "NaturalPlace",
        "Village", "Animal", "Plant", "Album", "Film", "WrittenWork"
    ],
    "yahoo": [
        "Society & Culture", "Science & Mathematics", "Health",
        "Education & Reference", "Computers & Internet", "Sports",
        "Business & Finance", "Entertainment & Music",
        "Family & Relationships", "Politics & Government"
    ],
    "newsgroups": [
        "alt.atheism", "comp.graphics", "comp.os.ms-windows.misc",
        "comp.sys.ibm.pc.hardware", "comp.sys.mac.hardware", "comp.windows.x",
        "misc.forsale", "rec.autos", "rec.motorcycles", "rec.sport.baseball",
        "rec.sport.hockey", "sci.crypt", "sci.electronics", "sci.med",
        "sci.space", "soc.religion.christian", "talk.politics.guns",
        "talk.politics.mideast", "talk.politics.misc", "talk.religion.misc"
    ],
}


def list_datasets() -> None:
    """Print available datasets."""
    print("=" * 60)
    print("Available Datasets")
    print("=" * 60)
    print(f"{'Name':<12} {'Classes':>8} {'Length':<8} {'Description'}")
    print("-" * 60)
    for key, info in DATASETS.items():
        print(f"{key:<12} {info.num_classes:>8} {info.avg_length:<8} {info.description}")
    print("=" * 60)


def get_dataset_info(name: str) -> DatasetInfo:
    """Get dataset info by name."""
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset: {name}. Available: {list(DATASETS.keys())}")
    return DATASETS[name]


def load_classification_dataset(
    name: str,
    max_samples: Optional[int] = None,
    seed: int = 42,
) -> Tuple[Dataset, Dataset, DatasetInfo]:
    """
    Load a classification dataset from HuggingFace.

    Args:
        name: Dataset name (yelp, imdb, ag_news, dbpedia, yahoo, newsgroups)
        max_samples: Maximum samples to use (None for all)
        seed: Random seed for shuffling

    Returns:
        Tuple of (train_dataset, test_dataset, dataset_info)
    """
    info = get_dataset_info(name)

    print(f"Loading {info.name}...")

    # Load from HuggingFace
    if info.subset:
        dataset = load_dataset(info.hf_name, info.subset)
    else:
        dataset = load_dataset(info.hf_name)

    # Get train and test splits
    train_data = dataset["train"]

    # Some datasets have different test split names
    if "test" in dataset:
        test_data = dataset["test"]
    elif "validation" in dataset:
        test_data = dataset["validation"]
    else:
        # Split train if no test set
        split = train_data.train_test_split(test_size=0.1, seed=seed)
        train_data = split["train"]
        test_data = split["test"]

    # Shuffle and limit samples
    train_data = train_data.shuffle(seed=seed)
    test_data = test_data.shuffle(seed=seed)

    if max_samples:
        train_data = train_data.select(range(min(max_samples, len(train_data))))
        test_data = test_data.select(range(min(max_samples // 10, len(test_data))))

    # Normalize column names
    def normalize_columns(example):
        text = example.get(info.text_column, "")

        # Handle Yahoo's multi-field format
        if name == "yahoo":
            title = example.get("question_title", "")
            content = example.get("question_content", "")
            answer = example.get("best_answer", "")
            text = f"{title} {content} {answer}".strip()

        label = example.get(info.label_column, 0)

        return {"text": text, "label": label}

    train_data = train_data.map(normalize_columns)
    test_data = test_data.map(normalize_columns)

    print(f"  Train: {len(train_data)} samples")
    print(f"  Test: {len(test_data)} samples")
    print(f"  Classes: {info.num_classes}")

    return train_data, test_data, info


def get_class_labels(name: str) -> List[str]:
    """Get class labels for a dataset."""
    if name not in CLASS_LABELS:
        info = get_dataset_info(name)
        return [f"Class {i}" for i in range(info.num_classes)]
    return CLASS_LABELS[name]


def analyze_dataset(name: str, max_samples: int = 1000) -> Dict:
    """
    Analyze a dataset's characteristics.

    Args:
        name: Dataset name
        max_samples: Samples to analyze

    Returns:
        Dictionary with analysis results
    """
    train_data, test_data, info = load_classification_dataset(name, max_samples)

    # Analyze text lengths
    lengths = [len(x["text"].split()) for x in train_data]

    analysis = {
        "name": info.name,
        "num_classes": info.num_classes,
        "train_size": len(train_data),
        "test_size": len(test_data),
        "avg_words": sum(lengths) / len(lengths),
        "min_words": min(lengths),
        "max_words": max(lengths),
        "median_words": sorted(lengths)[len(lengths) // 2],
    }

    # Class distribution
    from collections import Counter
    label_counts = Counter(x["label"] for x in train_data)
    analysis["class_distribution"] = dict(label_counts)

    return analysis


def print_dataset_analysis(name: str, max_samples: int = 1000) -> None:
    """Print dataset analysis."""
    analysis = analyze_dataset(name, max_samples)

    print(f"\n{'=' * 50}")
    print(f"Dataset: {analysis['name']}")
    print(f"{'=' * 50}")
    print(f"Classes: {analysis['num_classes']}")
    print(f"Train samples: {analysis['train_size']}")
    print(f"Test samples: {analysis['test_size']}")
    print(f"\nText Length (words):")
    print(f"  Mean: {analysis['avg_words']:.1f}")
    print(f"  Min: {analysis['min_words']}")
    print(f"  Max: {analysis['max_words']}")
    print(f"  Median: {analysis['median_words']}")
    print(f"\nClass Distribution:")
    labels = get_class_labels(name)
    for label_id, count in sorted(analysis["class_distribution"].items()):
        label_name = labels[label_id] if label_id < len(labels) else f"Class {label_id}"
        print(f"  {label_name}: {count}")
    print(f"{'=' * 50}")


# Convenience function for quick loading
def quick_load(
    name: str = "yelp",
    max_samples: int = 10000,
) -> Tuple[Dataset, Dataset, int, List[str]]:
    """
    Quick load a dataset with common settings.

    Returns:
        Tuple of (train_data, test_data, num_classes, class_labels)
    """
    train, test, info = load_classification_dataset(name, max_samples)
    labels = get_class_labels(name)
    return train, test, info.num_classes, labels


if __name__ == "__main__":
    # Demo: list all datasets
    list_datasets()

    # Demo: analyze each dataset
    for name in DATASETS.keys():
        try:
            print_dataset_analysis(name, max_samples=500)
        except Exception as e:
            print(f"Error loading {name}: {e}")
