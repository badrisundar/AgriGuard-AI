from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


# ==================================================
# PATHS
# ==================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_DIR = (
    PROJECT_ROOT
    / "dataset"
    / "processed"
    / "leaf_validator"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "leaf_validator_mobilenetv3_small_best.pth"
)

RESULTS_DIR = PROJECT_ROOT / "results"

RESULTS_DIR.mkdir(exist_ok=True)


# ==================================================
# DEVICE
# ==================================================

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")


# ==================================================
# TRANSFORM
# ==================================================

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[
            0.485,
            0.456,
            0.406
        ],
        std=[
            0.229,
            0.224,
            0.225
        ]
    )
])


# ==================================================
# TEST DATASET
# ==================================================

test_dataset = datasets.ImageFolder(
    DATASET_DIR / "test",
    transform=test_transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=16,
    shuffle=False,
    num_workers=0
)

print("\nClasses:")
print(test_dataset.classes)

print(
    f"Test images: {len(test_dataset)}"
)


# ==================================================
# LOAD MODEL
# ==================================================

model = models.mobilenet_v3_small(
    weights=None
)

model.classifier[3] = nn.Linear(
    model.classifier[3].in_features,
    2
)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=True
    )
)

model = model.to(device)
model.eval()

print("\nModel loaded successfully.")


# ==================================================
# PREDICTION
# ==================================================

all_labels = []
all_predictions = []

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)

        outputs = model(images)

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        all_labels.extend(
            labels.numpy()
        )

        all_predictions.extend(
            predictions.cpu().numpy()
        )


# ==================================================
# METRICS
# ==================================================

accuracy = accuracy_score(
    all_labels,
    all_predictions
)

precision = precision_score(
    all_labels,
    all_predictions,
    average="binary",
    pos_label=test_dataset.class_to_idx["leaf"],
    zero_division=0
)

recall = recall_score(
    all_labels,
    all_predictions,
    average="binary",
    pos_label=test_dataset.class_to_idx["leaf"],
    zero_division=0
)

f1 = f1_score(
    all_labels,
    all_predictions,
    average="binary",
    pos_label=test_dataset.class_to_idx["leaf"],
    zero_division=0
)

cm = confusion_matrix(
    all_labels,
    all_predictions
)


# ==================================================
# RESULTS
# ==================================================

print("\n========================================")
print("Leaf Validator Test Results")
print("========================================")

print(
    f"Accuracy:  {accuracy * 100:.2f}%"
)

print(
    f"Precision: {precision * 100:.2f}%"
)

print(
    f"Recall:    {recall * 100:.2f}%"
)

print(
    f"F1 Score:  {f1 * 100:.2f}%"
)

print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")

print(
    classification_report(
        all_labels,
        all_predictions,
        target_names=test_dataset.classes,
        zero_division=0
    )
)


# ==================================================
# SAVE RESULTS
# ==================================================

result_path = (
    RESULTS_DIR
    / "leaf_validator_test_results.txt"
)

with open(result_path, "w") as file:

    file.write(
        "Leaf Validator - MobileNetV3 Small\n"
    )

    file.write(
        "Test Dataset: CIFAR-10 negatives + "
        "PlantVillage leaf positives\n\n"
    )

    file.write(
        f"Test Images: {len(test_dataset)}\n"
    )

    file.write(
        f"Accuracy: {accuracy * 100:.2f}%\n"
    )

    file.write(
        f"Precision: {precision * 100:.2f}%\n"
    )

    file.write(
        f"Recall: {recall * 100:.2f}%\n"
    )

    file.write(
        f"F1 Score: {f1 * 100:.2f}%\n\n"
    )

    file.write(
        "Confusion Matrix:\n"
    )

    file.write(
        str(cm)
    )

print(
    f"\nResults saved to:\n{result_path}"
)