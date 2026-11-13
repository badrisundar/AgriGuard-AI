from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)
import numpy as np


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = (
    PROJECT_ROOT
    / "dataset"
    / "processed"
    / "plantdoc_adaptation"
    / "test"
)

MODEL_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

RESULTS_DIR.mkdir(exist_ok=True)


# ============================================================
# Device
# ============================================================

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"\nUsing device: {device}")


# ============================================================
# Transform
# No augmentation during evaluation
# ============================================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# Dataset
# ============================================================

dataset = datasets.ImageFolder(
    DATA_DIR,
    transform=transform
)

loader = DataLoader(
    dataset,
    batch_size=16,
    shuffle=False,
    num_workers=0
)

class_names = dataset.classes

# Original models were trained with 13 classes
MODEL_NUM_CLASSES = 13

# PlantDoc held-out evaluation has 7 classes
EVAL_NUM_CLASSES = len(class_names)


print(f"Model classes: {MODEL_NUM_CLASSES}")
print(f"Evaluation classes: {EVAL_NUM_CLASSES}")
print(f"Total held-out PlantDoc images: {len(dataset)}")

print("\nEvaluation classes:")
for i, class_name in enumerate(class_names):
    print(f"{i}: {class_name}")


# ============================================================
# PlantDoc index -> Original model index
# ============================================================

PLANTDOC_TO_MODEL_INDEX = {
    0: 0,    # Corn Cercospora
    1: 1,    # Corn Common Rust
    2: 4,    # Grape Black Rot
    3: 8,    # Tomato Bacterial Spot
    4: 9,    # Tomato Early Blight
    5: 10,   # Tomato Late Blight
    6: 11    # Tomato Leaf Mold
}


# ============================================================
# Model creation
# ============================================================

def create_resnet50():

    model = models.resnet50(weights=None)

    model.fc = nn.Linear(
        model.fc.in_features,
        MODEL_NUM_CLASSES
    )

    return model


def create_mobilenetv3():

    model = models.mobilenet_v3_large(weights=None)

    model.classifier[-1] = nn.Linear(
        model.classifier[-1].in_features,
        MODEL_NUM_CLASSES
    )

    return model


def create_efficientnet():

    model = models.efficientnet_b0(weights=None)

    model.classifier[-1] = nn.Linear(
        model.classifier[-1].in_features,
        MODEL_NUM_CLASSES
    )

    return model


# ============================================================
# Evaluation
# ============================================================

def evaluate_model(model, model_name, checkpoint_path):

    print("\n" + "=" * 70)
    print(f"Evaluating {model_name}")
    print("=" * 70)

    # --------------------------------------------------------
    # Load checkpoint
    # --------------------------------------------------------

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model = model.to(device)
    model.eval()

    # --------------------------------------------------------
    # Matched model classes
    # --------------------------------------------------------

    matched_model_indices = torch.tensor(
        list(PLANTDOC_TO_MODEL_INDEX.values()),
        device=device
    )

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    all_predictions = []
    all_labels = []

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(device)

            outputs = model(images)

            # Only compare the seven classes represented
            # in the PlantDoc evaluation dataset.
            matched_outputs = outputs[
                :,
                matched_model_indices
            ]

            predictions = torch.argmax(
                matched_outputs,
                dim=1
            )

            all_predictions.extend(
                predictions.cpu().numpy()
            )

            all_labels.extend(
                labels.numpy()
            )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    accuracy = accuracy_score(
        all_labels,
        all_predictions
    )

    report = classification_report(
        all_labels,
        all_predictions,
        target_names=class_names,
        digits=4,
        zero_division=0
    )

    cm = confusion_matrix(
        all_labels,
        all_predictions
    )

    # --------------------------------------------------------
    # Display
    # --------------------------------------------------------

    print(
        f"\n{model_name} "
        f"PlantDoc TEST Accuracy: "
        f"{accuracy * 100:.2f}%"
    )

    print("\nClassification Report:")
    print(report)

    print("Confusion Matrix:")
    print(cm)

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    result_file = (
        RESULTS_DIR
        / f"{model_name.lower().replace(' ', '_')}_"
          "plantdoc_test_baseline.txt"
    )

    with open(result_file, "w") as f:

        f.write(
            f"{model_name} - PlantDoc Held-Out Baseline\n"
        )

        f.write("=" * 70 + "\n\n")

        f.write(
            "Training dataset: "
            "PlantVillage-derived experiment_1\n"
        )

        f.write(
            "Evaluation dataset: "
            "PlantDoc original test split\n"
        )

        f.write(
            "PlantDoc fine-tuning: NO\n"
        )

        f.write(
            f"Original model classes: "
            f"{MODEL_NUM_CLASSES}\n"
        )

        f.write(
            f"Evaluation classes: "
            f"{EVAL_NUM_CLASSES}\n"
        )

        f.write(
            f"Held-out test images: "
            f"{len(dataset)}\n\n"
        )

        f.write(
            f"Accuracy: "
            f"{accuracy * 100:.2f}%\n\n"
        )

        f.write(
            "Classification Report:\n"
        )

        f.write(report)

        f.write(
            "\nConfusion Matrix:\n"
        )

        f.write(
            np.array2string(cm)
        )

    print(
        f"\nResults saved to:\n{result_file}"
    )

    return accuracy


# ============================================================
# Evaluate models
# ============================================================

results = {}


results["ResNet50"] = evaluate_model(
    create_resnet50(),
    "ResNet50",
    MODEL_DIR / "resnet50_best.pth"
)


results["MobileNetV3"] = evaluate_model(
    create_mobilenetv3(),
    "MobileNetV3",
    MODEL_DIR / "mobilenetv3_best.pth"
)


results["EfficientNet-B0"] = evaluate_model(
    create_efficientnet(),
    "EfficientNet-B0",
    MODEL_DIR / "efficientnet_b0_best.pth"
)


# ============================================================
# Final comparison
# ============================================================

print("\n" + "=" * 70)
print("PLANTDOC HELD-OUT BASELINE")
print("=" * 70)

for model_name, accuracy in results.items():

    print(
        f"{model_name:<20}"
        f"{accuracy * 100:.2f}%"
    )

print("=" * 70)