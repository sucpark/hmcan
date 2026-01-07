# Evaluation & Inference Guide

## Model Evaluation

### Basic Evaluation

```bash
# Evaluate on test set
python -m hmcan evaluate \
    --checkpoint outputs/hmcan_yelp/checkpoints/best_model.pt \
    --config outputs/hmcan_yelp/config.yaml
```

### Evaluation Options

```bash
python -m hmcan evaluate \
    --checkpoint <checkpoint path> \  # Required
    --config <config file path> \     # Optional (auto-detect)
    --device cuda                     # Specify device
```

### Example Output

```
Evaluating HMCAN model
Checkpoint: outputs/hmcan_yelp/checkpoints/best_model.pt

Test Results:
  Loss: 1.2345
  Accuracy: 61.70%
```

## Evaluation in Python

### Single Model Evaluation

```python
import torch
from hmcan.models import HMCAN
from hmcan.data import YelpDataModule
from hmcan.training import Trainer

# Load data
data_module = YelpDataModule(data_dir="data")
data_module.setup()

# Load model
model = HMCAN(vocab_size=len(data_module.vocabulary), num_classes=5)
checkpoint = torch.load("outputs/hmcan_yelp/checkpoints/best_model.pt")
model.load_state_dict(checkpoint["model_state_dict"])

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()

# Evaluate
trainer = Trainer(
    model=model,
    optimizer=torch.optim.Adam(model.parameters()),  # dummy
    criterion=torch.nn.CrossEntropyLoss(),
    device=device,
)

metrics = trainer.evaluate(data_module.test_dataloader())
print(f"Test Accuracy: {metrics['accuracy'] * 100:.2f}%")
```

### Compare Multiple Models

```python
from hmcan.models import create_model

models_to_compare = ["han", "hcan", "hmcan"]
results = {}

for model_name in models_to_compare:
    # Create and load model
    model = create_model(
        model_name,
        vocab_size=len(data_module.vocabulary),
        pretrained_embeddings=data_module.pretrained_embeddings,
    )

    checkpoint_path = f"outputs/{model_name}_yelp/checkpoints/best_model.pt"
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)

    # Evaluate
    trainer = Trainer(model=model, ...)
    metrics = trainer.evaluate(data_module.test_dataloader())
    results[model_name] = metrics["accuracy"]

# Print results
for name, acc in results.items():
    print(f"{name.upper()}: {acc * 100:.2f}%")
```

---

## Inference

### Single Document Inference

```python
import torch
from hmcan.models import HMCAN
from hmcan.data import Vocabulary, DocumentPreprocessor

# Load model
model = HMCAN(vocab_size=20000, num_classes=5)
checkpoint = torch.load("outputs/hmcan_yelp/checkpoints/best_model.pt")
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# Load vocabulary
vocab = Vocabulary.load("data/processed/word2idx.json")

# Preprocessor
preprocessor = DocumentPreprocessor()

# Inference function
def predict(text: str, model, vocab, preprocessor, device="cpu"):
    """Predict single text"""
    model = model.to(device)

    # Preprocess
    doc_indices = preprocessor.process_document(text, vocab)

    if len(doc_indices) == 0:
        return None, None

    # Convert to tensor
    max_words = max(len(sent) for sent in doc_indices)
    num_sentences = len(doc_indices)

    document = torch.zeros(num_sentences, max_words, dtype=torch.long)
    sentence_lengths = []

    for i, sent in enumerate(doc_indices):
        document[i, :len(sent)] = torch.tensor(sent)
        sentence_lengths.append(len(sent))

    document = document.to(device)
    sentence_lengths = torch.tensor(sentence_lengths, device=device)

    # Inference
    with torch.no_grad():
        outputs = model(document, sentence_lengths)
        logits = outputs["logits"]
        probs = torch.softmax(logits, dim=-1)
        pred_class = torch.argmax(probs, dim=-1).item()

    return pred_class, probs.cpu().numpy()

# Example usage
text = "This restaurant was amazing! The food was delicious and service was excellent."
pred, probs = predict(text, model, vocab, preprocessor)
print(f"Predicted class: {pred + 1} stars")  # 0-4 → 1-5
print(f"Probabilities: {probs}")
```

### Batch Inference

```python
def predict_batch(texts: list[str], model, vocab, preprocessor, device="cpu"):
    """Batch prediction for multiple texts"""
    results = []

    for text in texts:
        pred, probs = predict(text, model, vocab, preprocessor, device)
        results.append({
            "text": text[:100] + "..." if len(text) > 100 else text,
            "prediction": pred + 1 if pred is not None else None,
            "confidence": float(probs.max()) if probs is not None else None,
        })

    return results

# Example usage
texts = [
    "Terrible experience. Food was cold and waiter was rude.",
    "Pretty good, nothing special but decent food.",
    "Best restaurant in town! Highly recommend!",
]

results = predict_batch(texts, model, vocab, preprocessor)
for r in results:
    print(f"{r['prediction']} stars ({r['confidence']:.2%}): {r['text']}")
```

### Attention Visualization

```python
import matplotlib.pyplot as plt
import numpy as np

def visualize_attention(text, model, vocab, preprocessor, device="cpu"):
    """Visualize attention weights"""
    model = model.to(device)

    # Preprocess
    sentences = preprocessor.tokenize_document(text)
    doc_indices = preprocessor.process_document(text, vocab)

    if len(doc_indices) == 0:
        return

    # Convert to tensor
    max_words = max(len(sent) for sent in doc_indices)
    num_sentences = len(doc_indices)

    document = torch.zeros(num_sentences, max_words, dtype=torch.long)
    sentence_lengths = []

    for i, sent in enumerate(doc_indices):
        document[i, :len(sent)] = torch.tensor(sent)
        sentence_lengths.append(len(sent))

    document = document.to(device)
    sentence_lengths_tensor = torch.tensor(sentence_lengths, device=device)

    # Inference
    with torch.no_grad():
        outputs = model(document, sentence_lengths_tensor)
        word_attn = outputs.get("word_attention")
        sent_attn = outputs.get("sentence_attention")

    # Visualize
    if sent_attn is not None:
        sent_attn = sent_attn.cpu().numpy().flatten()

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.bar(range(len(sent_attn)), sent_attn)
        ax.set_xlabel("Sentence Index")
        ax.set_ylabel("Attention Weight")
        ax.set_title("Sentence-level Attention")
        plt.tight_layout()
        plt.savefig("sentence_attention.png")
        plt.show()

    return word_attn, sent_attn

# Example usage
text = "The appetizers were okay. Main course was fantastic! Dessert was too sweet."
word_attn, sent_attn = visualize_attention(text, model, vocab, preprocessor)
```

---

## Performance Metrics

### Additional Metrics Beyond Accuracy

```python
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

def detailed_evaluation(model, data_loader, device):
    """Detailed evaluation metrics"""
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in data_loader:
            document = batch["document"].to(device)
            sentence_lengths = batch["sentence_lengths"].to(device)
            labels = batch["label"]

            outputs = model(document, sentence_lengths)
            preds = torch.argmax(outputs["logits"], dim=-1)

            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.numpy().tolist())

    # Classification Report
    print("Classification Report:")
    print(classification_report(
        all_labels, all_preds,
        target_names=["1 star", "2 stars", "3 stars", "4 stars", "5 stars"]
    ))

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    print("\nConfusion Matrix:")
    print(cm)

    return all_preds, all_labels

# Example usage
preds, labels = detailed_evaluation(model, data_module.test_dataloader(), device)
```

### Example Output

```
Classification Report:
              precision    recall  f1-score   support

      1 star       0.58      0.62      0.60       200
     2 stars       0.45      0.42      0.43       200
     3 stars       0.52      0.48      0.50       200
     4 stars       0.55      0.58      0.56       200
     5 stars       0.72      0.75      0.73       200

    accuracy                           0.57      1000
   macro avg       0.56      0.57      0.56      1000
weighted avg       0.56      0.57      0.56      1000

Confusion Matrix:
[[124  32  22  14   8]
 [ 38  84  42  26  10]
 [ 20  36  96  32  16]
 [ 12  22  28 116  22]
 [  8   6  12  24 150]]
```

---

## Model Export

### TorchScript Export

```python
# Load model
model = HMCAN(vocab_size=20000, num_classes=5)
model.load_state_dict(torch.load("best_model.pt")["model_state_dict"])
model.eval()

# Convert to TorchScript (trace method)
example_doc = torch.randint(0, 1000, (5, 20))
example_lens = torch.tensor([20, 15, 18, 12, 10])

# Note: script method recommended for dynamic input sizes
traced_model = torch.jit.trace(model, (example_doc, example_lens))
traced_model.save("hmcan_traced.pt")

# Load and use
loaded_model = torch.jit.load("hmcan_traced.pt")
output = loaded_model(example_doc, example_lens)
```

### ONNX Export

```python
import torch.onnx

model.eval()
example_doc = torch.randint(0, 1000, (5, 20))
example_lens = torch.tensor([20, 15, 18, 12, 10])

torch.onnx.export(
    model,
    (example_doc, example_lens),
    "hmcan.onnx",
    input_names=["document", "sentence_lengths"],
    output_names=["logits"],
    dynamic_axes={
        "document": {0: "num_sentences", 1: "max_words"},
        "sentence_lengths": {0: "num_sentences"},
    },
)
```

---

## Troubleshooting

### Out of Memory

```python
# Reduce batch size (default is 1, so not an issue)
# Truncate long documents
max_sentences = 30
max_words = 50
```

### Slow Inference

```python
# Use GPU
device = torch.device("cuda")
model = model.to(device)

# Batch processing
# Padding required for processing multiple documents at once
```

### Attention Returns None

Some model configurations may not return attention weights.
Check the model's `forward()` method.
