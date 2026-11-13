from pathlib import Path

import torch
import torch.nn as nn

from torchvision import models
from torchvision.models import EfficientNet_B0_Weights

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

from data_loader import (
    test_loader,
    DEVICE
)


# --------------------------------------------------
# 1. Project paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "efficientnet_b0_best.pth"
)

RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_PATH = (
    RESULTS_DIR
    / "efficientnet_b0_test_results.txt"
)


# --------------------------------------------------
# 2. Configuration
# --------------------------------------------------

NUM_CLASSES = 13


# --------------------------------------------------
# 3. Class names
# --------------------------------------------------

CLASS_NAMES = [
    "Corn___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn___Common_rust",
    "Corn___Northern_Leaf_Blight",
    "Corn___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___healthy"
]


# --------------------------------------------------
# 4. Load pretrained EfficientNet-B0
# --------------------------------------------------

print("\nLoading best EfficientNet-B0 model...")

weights = EfficientNet_B0_Weights.DEFAULT

model = models.efficientnet_b0(
    weights=weights
)


# Replace final classification layer
model.classifier[-1] = nn.Linear(
    model.classifier[-1].in_features,
    NUM_CLASSES
)


# --------------------------------------------------
# 5. Load trained checkpoint
# --------------------------------------------------

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model = model.to(DEVICE)

model.eval()


print("Model loaded successfully.")

print(
    f"Model trained for epoch: "
    f"{checkpoint['epoch']}"
)

print(
    f"Best validation accuracy: "
    f"{checkpoint['validation_accuracy']:.4f}"
)


# --------------------------------------------------
# 6. Evaluate on test set
# --------------------------------------------------

print("\nEvaluating on test set...")
print("=" * 70)

all_predictions = []
all_labels = []


with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs = model(images)

        _, predictions = torch.max(
            outputs,
            1
        )

        all_predictions.extend(
            predictions.cpu().numpy()
        )

        all_labels.extend(
            labels.cpu().numpy()
        )


# --------------------------------------------------
# 7. Calculate overall metrics
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
# 8. Classification report
# --------------------------------------------------

report = classification_report(
    all_labels,
    all_predictions,
    target_names=CLASS_NAMES,
    digits=4,
    zero_division=0
)


# --------------------------------------------------
# 9. Confusion matrix
# --------------------------------------------------

cm = confusion_matrix(
    all_labels,
    all_predictions
)


# --------------------------------------------------
# 10. Print overall results
# --------------------------------------------------

print("\nOverall Test Results")
print("=" * 70)

print(
    f"Accuracy : {accuracy:.4f} "
    f"({accuracy * 100:.2f}%)"
)

print(
    f"Precision: {precision:.4f} "
    f"({precision * 100:.2f}%)"
)

print(
    f"Recall   : {recall:.4f} "
    f"({recall * 100:.2f}%)"
)

print(
    f"F1-score : {f1:.4f} "
    f"({f1 * 100:.2f}%)"
)


# --------------------------------------------------
# 11. Print classification report
# --------------------------------------------------

print("\nPer-Class Classification Report")
print("=" * 70)

print(report)


# --------------------------------------------------
# 12. Print confusion matrix
# --------------------------------------------------

print("\nConfusion Matrix")
print("=" * 70)

print(cm)


# --------------------------------------------------
# 13. Save results
# --------------------------------------------------

with open(
    RESULTS_PATH,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "EfficientNet-B0 Test Evaluation\n"
    )

    file.write(
        "=" * 70 + "\n\n"
    )

    file.write(
        "Model: EfficientNet-B0\n"
    )

    file.write(
        f"Number of classes: {NUM_CLASSES}\n"
    )

    file.write(
        f"Test images: {len(all_labels)}\n"
    )

    file.write(
        f"Best training epoch: "
        f"{checkpoint['epoch']}\n"
    )

    file.write(
        f"Best validation accuracy: "
        f"{checkpoint['validation_accuracy']:.4f}\n\n"
    )

    file.write(
        "Overall Test Results\n"
    )

    file.write(
        "=" * 70 + "\n"
    )

    file.write(
        f"Accuracy : {accuracy:.4f} "
        f"({accuracy * 100:.2f}%)\n"
    )

    file.write(
        f"Precision: {precision:.4f} "
        f"({precision * 100:.2f}%)\n"
    )

    file.write(
        f"Recall   : {recall:.4f} "
        f"({recall * 100:.2f}%)\n"
    )

    file.write(
        f"F1-score : {f1:.4f} "
        f"({f1 * 100:.2f}%)\n\n"
    )

    file.write(
        "Per-Class Classification Report\n"
    )

    file.write(
        "=" * 70 + "\n"
    )

    file.write(report)

    file.write(
        "\n\nConfusion Matrix\n"
    )

    file.write(
        "=" * 70 + "\n"
    )

    file.write(str(cm))


# --------------------------------------------------
# 14. Final message
# --------------------------------------------------

print(
    f"\nResults saved to:\n"
    f"{RESULTS_PATH}"
)