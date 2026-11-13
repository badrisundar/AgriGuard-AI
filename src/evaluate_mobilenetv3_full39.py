import torch
import numpy as np
from torchvision import models
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from data_loader_full39 import test_loader, DEVICE, test_dataset

# =========================
# Configuration
# =========================

MODEL_PATH = "models/mobilenetv3_full39_best.pth"

NUM_CLASSES = 39

# =========================
# Load checkpoint
# =========================

print("=" * 70)
print("EVALUATING FINAL 39-CLASS MOBILENETV3 LARGE")
print("=" * 70)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
    weights_only=True
)

model = models.mobilenet_v3_large(weights=None)

model.classifier[3] = torch.nn.Linear(
    model.classifier[3].in_features,
    NUM_CLASSES
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model = model.to(DEVICE)
model.eval()

print(f"Device: {DEVICE}")
print(f"Test images: {len(test_dataset)}")
print(f"Classes: {NUM_CLASSES}")
print(
    f"Best validation accuracy: "
    f"{checkpoint['validation_accuracy'] * 100:.2f}%"
)

# =========================
# Evaluation
# =========================

all_predictions = []
all_labels = []

correct = 0
total = 0

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs = model(images)

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        total += labels.size(0)

        correct += (
            predictions == labels
        ).sum().item()

        all_predictions.extend(
            predictions.cpu().numpy()
        )

        all_labels.extend(
            labels.cpu().numpy()
        )

# =========================
# Metrics
# =========================

accuracy = accuracy_score(
    all_labels,
    all_predictions
)

precision = precision_score(
    all_labels,
    all_predictions,
    average="weighted",
    zero_division=0
)

recall = recall_score(
    all_labels,
    all_predictions,
    average="weighted",
    zero_division=0
)

f1 = f1_score(
    all_labels,
    all_predictions,
    average="weighted",
    zero_division=0
)

# =========================
# Print overall results
# =========================

print()
print("=" * 70)
print("OVERALL TEST RESULTS")
print("=" * 70)

print(f"Accuracy : {accuracy * 100:.2f}%")
print(f"Precision: {precision * 100:.2f}%")
print(f"Recall   : {recall * 100:.2f}%")
print(f"F1 Score : {f1 * 100:.2f}%")

# =========================
# Classification report
# =========================

print()
print("=" * 70)
print("PER-CLASS RESULTS")
print("=" * 70)

class_names = test_dataset.classes

report = classification_report(
    all_labels,
    all_predictions,
    target_names=class_names,
    digits=4,
    zero_division=0
)

print(report)

# =========================
# Confusion Matrix
# =========================

cm = confusion_matrix(
    all_labels,
    all_predictions
)

print()
print("=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)

print(cm)

# =========================
# Save results
# =========================

results_path = (
    "results/mobilenetv3_full39_test_results.txt"
)

with open(results_path, "w") as f:

    f.write(
        "FINAL 39-CLASS MOBILENETV3 LARGE "
        "TEST RESULTS\n"
    )

    f.write("=" * 70 + "\n")

    f.write(
        f"Best Validation Accuracy: "
        f"{checkpoint['validation_accuracy'] * 100:.2f}%\n"
    )

    f.write(
        f"Test Images: {len(test_dataset)}\n"
    )

    f.write(
        f"Accuracy: {accuracy * 100:.4f}%\n"
    )

    f.write(
        f"Precision: {precision * 100:.4f}%\n"
    )

    f.write(
        f"Recall: {recall * 100:.4f}%\n"
    )

    f.write(
        f"F1 Score: {f1 * 100:.4f}%\n"
    )

    f.write("\n\nPER-CLASS RESULTS\n")
    f.write("=" * 70 + "\n")
    f.write(report)

    f.write("\n\nCONFUSION MATRIX\n")
    f.write("=" * 70 + "\n")
    f.write(str(cm))

print()
print("=" * 70)
print("RESULTS SAVED")
print("=" * 70)

print(results_path)