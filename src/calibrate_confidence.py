import os
import torch
import torch.nn as nn
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader


# ============================================================
# Configuration
# ============================================================

DATA_DIR = "dataset/processed/plantdoc_finetune/validation"
CHECKPOINT = "models/mobilenetv3_plantdoc_finetuned_best.pth"

BATCH_SIZE = 16

device = torch.device(
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)

print(f"Using device: {device}")


# ============================================================
# Validation transform
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
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

print(f"Validation images: {len(dataset)}")
print(f"Classes: {len(dataset.classes)}")


# ============================================================
# MobileNetV3
# ============================================================

model = models.mobilenet_v3_large(weights=None)

model.classifier[3] = nn.Linear(
    model.classifier[3].in_features,
    7
)

checkpoint = torch.load(
    CHECKPOINT,
    map_location="cpu",
    weights_only=True
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model = model.to(device)
model.eval()


# ============================================================
# Predictions
# ============================================================

all_labels = []
all_predictions = []
all_confidences = []

with torch.no_grad():

    for images, labels in loader:

        images = images.to(device)

        outputs = model(images)

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        confidence, predictions = torch.max(
            probabilities,
            dim=1
        )

        all_labels.extend(labels.numpy())
        all_predictions.extend(
            predictions.cpu().numpy()
        )
        all_confidences.extend(
            confidence.cpu().numpy()
        )


all_labels = torch.tensor(all_labels)
all_predictions = torch.tensor(all_predictions)
all_confidences = torch.tensor(all_confidences)

correct = all_predictions == all_labels


# ============================================================
# Basic statistics
# ============================================================

print("\n" + "=" * 70)
print("VALIDATION CONFIDENCE ANALYSIS")
print("=" * 70)

accuracy = correct.float().mean().item()

print(f"Validation Accuracy: {accuracy * 100:.2f}%")

print(
    f"Average Confidence: "
    f"{all_confidences.mean().item() * 100:.2f}%"
)

print(
    f"Correct Prediction Confidence: "
    f"{all_confidences[correct].mean().item() * 100:.2f}%"
)

print(
    f"Incorrect Prediction Confidence: "
    f"{all_confidences[~correct].mean().item() * 100:.2f}%"
)


# ============================================================
# Threshold analysis
# ============================================================

thresholds = [
    0.40,
    0.45,
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
    0.75,
    0.80,
    0.85,
    0.90
]

print("\n" + "=" * 70)
print("VALIDATION THRESHOLD ANALYSIS")
print("=" * 70)

print(
    f"{'Threshold':<12}"
    f"{'Accepted':<12}"
    f"{'Uncertain':<12}"
    f"{'Coverage':<12}"
    f"{'Accuracy':<12}"
    f"{'Wrong':<10}"
)

print("-" * 70)

results = []

for threshold in thresholds:

    accepted = all_confidences >= threshold

    accepted_count = accepted.sum().item()
    uncertain_count = (~accepted).sum().item()

    if accepted_count > 0:

        accepted_correct = correct[accepted]

        accepted_accuracy = (
            accepted_correct.float().mean().item()
        )

        wrong = (
            (~correct) & accepted
        ).sum().item()

    else:

        accepted_accuracy = 0.0
        wrong = 0

    coverage = accepted_count / len(dataset)

    print(
        f"{threshold:<12.2f}"
        f"{accepted_count:<12}"
        f"{uncertain_count:<12}"
        f"{coverage * 100:<12.2f}"
        f"{accepted_accuracy * 100:<12.2f}"
        f"{wrong:<10}"
    )

    results.append({
        "threshold": threshold,
        "accepted": accepted_count,
        "uncertain": uncertain_count,
        "coverage": coverage,
        "accuracy": accepted_accuracy,
        "wrong": wrong
    })


# ============================================================
# Select candidate threshold
#
# Target:
#   - accepted accuracy >= 80%
#   - maximize coverage
# ============================================================

valid_candidates = [
    r for r in results
    if r["accuracy"] >= 0.80
    and r["accepted"] > 0
]

if valid_candidates:

    best = max(
        valid_candidates,
        key=lambda x: x["coverage"]
    )

    selected_threshold = best["threshold"]

    print("\n" + "=" * 70)
    print("SELECTED PRELIMINARY THRESHOLD")
    print("=" * 70)

    print(
        f"Threshold: {selected_threshold:.2f}"
    )

    print(
        f"Validation Accuracy of Accepted: "
        f"{best['accuracy'] * 100:.2f}%"
    )

    print(
        f"Coverage: "
        f"{best['coverage'] * 100:.2f}%"
    )

else:

    selected_threshold = None

    print("\nNo threshold achieved 80% accepted accuracy.")


# ============================================================
# Save results
# ============================================================

os.makedirs("results", exist_ok=True)

output_file = (
    "results/mobilenetv3_confidence_calibration.txt"
)

with open(output_file, "w") as f:

    f.write(
        "AgriGuard AI - Confidence Calibration\n"
    )

    f.write("=" * 60 + "\n\n")

    f.write("Model: MobileNetV3 Large\n")
    f.write("Dataset: PlantDoc Validation Set\n")
    f.write(
        f"Validation Images: {len(dataset)}\n\n"
    )

    f.write(
        f"Validation Accuracy: "
        f"{accuracy * 100:.2f}%\n"
    )

    f.write(
        f"Average Confidence: "
        f"{all_confidences.mean().item() * 100:.2f}%\n"
    )

    f.write("\nThreshold Results\n")
    f.write("-" * 60 + "\n")

    for r in results:

        f.write(
            f"Threshold={r['threshold']:.2f}, "
            f"Accepted={r['accepted']}, "
            f"Uncertain={r['uncertain']}, "
            f"Coverage={r['coverage'] * 100:.2f}%, "
            f"Accuracy={r['accuracy'] * 100:.2f}%, "
            f"Wrong={r['wrong']}\n"
        )

    f.write("\n")

    if selected_threshold is not None:

        f.write(
            f"Selected preliminary threshold: "
            f"{selected_threshold:.2f}\n"
        )

    else:

        f.write(
            "No threshold achieved the target "
            "accepted accuracy of 80%.\n"
        )

print("\nResults saved to:")
print(output_file)