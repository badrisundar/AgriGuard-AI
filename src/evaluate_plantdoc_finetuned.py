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

TEST_DIR = (
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
    TEST_DIR,
    transform=transform
)

loader = DataLoader(
    dataset,
    batch_size=16,
    shuffle=False,
    num_workers=0
)

class_names = dataset.classes
NUM_CLASSES = len(class_names)

print(f"Number of classes: {NUM_CLASSES}")
print(f"Held-out PlantDoc test images: {len(dataset)}")

print("\nClasses:")
for i, name in enumerate(class_names):
    print(f"{i}: {name}")


# ============================================================
# Model creation
# ============================================================

def create_resnet50():

    model = models.resnet50(weights=None)

    model.fc = nn.Linear(
        model.fc.in_features,
        NUM_CLASSES
    )

    return model


def create_mobilenetv3():

    model = models.mobilenet_v3_large(weights=None)

    model.classifier[-1] = nn.Linear(
        model.classifier[-1].in_features,
        NUM_CLASSES
    )

    return model


def create_efficientnet():

    model = models.efficientnet_b0(weights=None)

    model.classifier[-1] = nn.Linear(
        model.classifier[-1].in_features,
        NUM_CLASSES
    )

    return model


# ============================================================
# Evaluation
# ============================================================

def evaluate_model(model, model_name, checkpoint_name):

    print("\n" + "=" * 70)
    print(f"Evaluating fine-tuned {model_name}")
    print("=" * 70)

    checkpoint_path = MODEL_DIR / checkpoint_name

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

    all_predictions = []
    all_labels = []

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(device)

            outputs = model(images)

            predictions = torch.argmax(
                outputs,
                dim=1
            )

            all_predictions.extend(
                predictions.cpu().numpy()
            )

            all_labels.extend(
                labels.numpy()
            )

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

    print(
        f"\n{model_name} Fine-tuned PlantDoc "
        f"TEST Accuracy: {accuracy * 100:.2f}%"
    )

    print("\nClassification Report:")
    print(report)

    print("Confusion Matrix:")
    print(cm)

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    result_file = (
        RESULTS_DIR
        / f"{model_name.lower().replace(' ', '_')}"
          "_plantdoc_finetuned_test_results.txt"
    )

    with open(result_file, "w") as f:

        f.write(
            f"{model_name} - Fine-tuned PlantDoc Test\n"
        )

        f.write("=" * 70 + "\n\n")

        f.write(
            "Initial training: PlantVillage experiment_1\n"
        )

        f.write(
            "Adaptation training: PlantDoc train split\n"
        )

        f.write(
            "Final evaluation: PlantDoc original test split\n"
        )

        f.write(
            "PlantDoc test images used during training: NO\n"
        )

        f.write(
            f"Test images: {len(dataset)}\n\n"
        )

        f.write(
            f"Accuracy: {accuracy * 100:.2f}%\n\n"
        )

        f.write("Classification Report:\n")
        f.write(report)

        f.write("\nConfusion Matrix:\n")
        f.write(np.array2string(cm))

    print(
        f"\nResults saved to:\n{result_file}"
    )

    return accuracy


# ============================================================
# Evaluate all fine-tuned models
# ============================================================

results = {}

results["ResNet50"] = evaluate_model(
    create_resnet50(),
    "ResNet50",
    "resnet50_plantdoc_finetuned_best.pth"
)

results["MobileNetV3"] = evaluate_model(
    create_mobilenetv3(),
    "MobileNetV3",
    "mobilenetv3_plantdoc_finetuned_best.pth"
)

results["EfficientNet-B0"] = evaluate_model(
    create_efficientnet(),
    "EfficientNet-B0",
    "efficientnet-b0_plantdoc_finetuned_best.pth"
)


# ============================================================
# Final comparison
# ============================================================

print("\n" + "=" * 70)
print("FINAL PLANTDOC ADAPTATION COMPARISON")
print("=" * 70)

for model_name, accuracy in results.items():

    print(
        f"{model_name:<20}"
        f"{accuracy * 100:.2f}%"
    )

print("=" * 70)