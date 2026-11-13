from pathlib import Path

import torch
import torch.nn as nn

from torchvision import models

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

from data_loader import test_loader, DEVICE


# --------------------------------------------------
# 1. Paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_ROOT / "models" / "resnet50_best.pth"

RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 2. Load checkpoint
# --------------------------------------------------

print("\nLoading best ResNet50 model...")

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
    weights_only=True
)

num_classes = checkpoint["num_classes"]

model = models.resnet50(weights=None)

model.fc = nn.Linear(
    model.fc.in_features,
    num_classes
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model = model.to(DEVICE)

model.eval()

print("Model loaded successfully.")
print("Model trained for epoch:", checkpoint["epoch"])
print("Best validation accuracy:",
      f"{checkpoint['validation_accuracy']:.4f}")


# --------------------------------------------------
# 3. Get class names
# --------------------------------------------------

class_names = test_loader.dataset.classes


# --------------------------------------------------
# 4. Test evaluation
# --------------------------------------------------

all_predictions = []
all_labels = []

print("\nEvaluating on test set...")
print("=" * 70)

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs = model(images)

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        all_predictions.extend(
            predictions.cpu().numpy()
        )

        all_labels.extend(
            labels.cpu().numpy()
        )


# --------------------------------------------------
# 5. Calculate metrics
# --------------------------------------------------

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


# --------------------------------------------------
# 6. Print overall results
# --------------------------------------------------

print("\nOverall Test Results")
print("=" * 70)

print(f"Accuracy : {accuracy:.4f} ({accuracy * 100:.2f}%)")
print(f"Precision: {precision:.4f} ({precision * 100:.2f}%)")
print(f"Recall   : {recall:.4f} ({recall * 100:.2f}%)")
print(f"F1-score : {f1:.4f} ({f1 * 100:.2f}%)")


# --------------------------------------------------
# 7. Classification report
# --------------------------------------------------

print("\nPer-Class Classification Report")
print("=" * 70)

report = classification_report(
    all_labels,
    all_predictions,
    target_names=class_names,
    digits=4,
    zero_division=0
)

print(report)


# --------------------------------------------------
# 8. Confusion matrix
# --------------------------------------------------

cm = confusion_matrix(
    all_labels,
    all_predictions
)

print("\nConfusion Matrix")
print("=" * 70)

print(cm)


# --------------------------------------------------
# 9. Save metrics to text file
# --------------------------------------------------

results_file = RESULTS_DIR / "resnet50_test_results.txt"

with open(results_file, "w") as file:

    file.write("AgriGuard AI - ResNet50 Test Results\n")
    file.write("=" * 70 + "\n\n")

    file.write(
        f"Best validation accuracy: "
        f"{checkpoint['validation_accuracy']:.4f}\n"
    )

    file.write(
        f"Test Accuracy: "
        f"{accuracy:.4f}\n"
    )

    file.write(
        f"Test Precision: "
        f"{precision:.4f}\n"
    )

    file.write(
        f"Test Recall: "
        f"{recall:.4f}\n"
    )

    file.write(
        f"Test F1-score: "
        f"{f1:.4f}\n\n"
    )

    file.write(
        "Per-Class Classification Report\n"
    )

    file.write("=" * 70 + "\n")

    file.write(report)

    file.write("\n\nConfusion Matrix\n")
    file.write("=" * 70 + "\n")

    file.write(str(cm))


print("\nResults saved to:")
print(results_file)