import os
import torch
import torch.nn as nn
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score


# ============================================================
# Configuration
# ============================================================

DATA_DIR = "dataset/processed/plantdoc_adaptation/test"
CHECKPOINT = "models/mobilenetv3_plantdoc_finetuned_best.pth"

BATCH_SIZE = 16

device = torch.device(
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)

print(f"Using device: {device}")


# ============================================================
# Transform
# Same preprocessing used for validation/test
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

print(f"Number of test images: {len(dataset)}")
print(f"Number of classes: {len(dataset.classes)}")

print("\nClasses:")
for i, cls in enumerate(dataset.classes):
    print(f"{i}: {cls}")


# ============================================================
# MobileNetV3 Large
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
# Collect predictions and confidence
# ============================================================

all_labels = []
all_predictions = []
all_confidences = []

with torch.no_grad():

    for images, labels in loader:

        images = images.to(device)

        outputs = model(images)

        probabilities = torch.softmax(outputs, dim=1)

        confidence, predictions = torch.max(
            probabilities,
            dim=1
        )

        all_labels.extend(labels.numpy())
        all_predictions.extend(predictions.cpu().numpy())
        all_confidences.extend(confidence.cpu().numpy())


# ============================================================
# Overall results
# ============================================================

all_labels = torch.tensor(all_labels)
all_predictions = torch.tensor(all_predictions)
all_confidences = torch.tensor(all_confidences)

correct = all_predictions == all_labels

overall_accuracy = correct.float().mean().item()

print("\n" + "=" * 70)
print("OVERALL CONFIDENCE ANALYSIS")
print("=" * 70)

print(f"Overall Accuracy: {overall_accuracy * 100:.2f}%")

print(
    f"Average Confidence: "
    f"{all_confidences.mean().item() * 100:.2f}%"
)

print(
    f"Correct Prediction Average Confidence: "
    f"{all_confidences[correct].mean().item() * 100:.2f}%"
)

print(
    f"Incorrect Prediction Average Confidence: "
    f"{all_confidences[~correct].mean().item() * 100:.2f}%"
)


# ============================================================
# Threshold analysis
# ============================================================

thresholds = [
    0.50,
    0.60,
    0.70,
    0.80,
    0.90
]

results = []

print("\n" + "=" * 70)
print("CONFIDENCE THRESHOLD ANALYSIS")
print("=" * 70)

print(
    f"{'Threshold':<12}"
    f"{'Accepted':<12}"
    f"{'Uncertain':<12}"
    f"{'Coverage':<12}"
    f"{'Accuracy':<12}"
    f"{'Wrong Accepted':<15}"
)

print("-" * 70)

for threshold in thresholds:

    accepted = all_confidences >= threshold

    accepted_count = accepted.sum().item()
    uncertain_count = (~accepted).sum().item()

    if accepted_count > 0:

        accepted_correct = correct[accepted]

        accepted_accuracy = (
            accepted_correct.float().mean().item()
        )

        wrong_accepted = (
            (~correct) & accepted
        ).sum().item()

    else:

        accepted_accuracy = 0.0
        wrong_accepted = 0

    coverage = accepted_count / len(dataset)

    print(
        f"{threshold:<12.2f}"
        f"{accepted_count:<12}"
        f"{uncertain_count:<12}"
        f"{coverage * 100:<12.2f}"
        f"{accepted_accuracy * 100:<12.2f}"
        f"{wrong_accepted:<15}"
    )

    results.append({
        "threshold": threshold,
        "accepted": accepted_count,
        "uncertain": uncertain_count,
        "coverage": coverage,
        "accuracy": accepted_accuracy,
        "wrong_accepted": wrong_accepted
    })


# ============================================================
# Correct vs incorrect confidence distribution
# ============================================================

print("\n" + "=" * 70)
print("CONFIDENCE DISTRIBUTION")
print("=" * 70)

confidence_ranges = [
    (0.0, 0.50),
    (0.50, 0.60),
    (0.60, 0.70),
    (0.70, 0.80),
    (0.80, 0.90),
    (0.90, 1.01)
]

print(
    f"{'Range':<15}"
    f"{'Correct':<12}"
    f"{'Incorrect':<12}"
    f"{'Total':<12}"
)

print("-" * 50)

for low, high in confidence_ranges:

    mask = (
        (all_confidences >= low) &
        (all_confidences < high)
    )

    correct_count = (
        mask & correct
    ).sum().item()

    incorrect_count = (
        mask & ~correct
    ).sum().item()

    total = mask.sum().item()

    print(
        f"{low:.2f}-{min(high, 1.0):.2f}"
        f"{'':<8}"
        f"{correct_count:<12}"
        f"{incorrect_count:<12}"
        f"{total:<12}"
    )


# ============================================================
# Save results
# ============================================================

os.makedirs("results", exist_ok=True)

output_file = (
    "results/mobilenetv3_confidence_analysis.txt"
)

with open(output_file, "w") as f:

    f.write("AgriGuard AI - Confidence-Aware Prediction Analysis\n")
    f.write("=" * 60 + "\n\n")

    f.write("Model: MobileNetV3 Large\n")
    f.write("Dataset: Held-out PlantDoc Test Set\n")
    f.write(f"Test Images: {len(dataset)}\n\n")

    f.write(
        f"Overall Accuracy: "
        f"{overall_accuracy * 100:.2f}%\n"
    )

    f.write(
        f"Average Confidence: "
        f"{all_confidences.mean().item() * 100:.2f}%\n"
    )

    f.write(
        f"Correct Prediction Average Confidence: "
        f"{all_confidences[correct].mean().item() * 100:.2f}%\n"
    )

    f.write(
        f"Incorrect Prediction Average Confidence: "
        f"{all_confidences[~correct].mean().item() * 100:.2f}%\n\n"
    )

    f.write("Threshold Analysis\n")
    f.write("-" * 60 + "\n")

    for r in results:

        f.write(
            f"Threshold: {r['threshold']:.2f} | "
            f"Accepted: {r['accepted']} | "
            f"Uncertain: {r['uncertain']} | "
            f"Coverage: {r['coverage'] * 100:.2f}% | "
            f"Accuracy: {r['accuracy'] * 100:.2f}% | "
            f"Wrong Accepted: {r['wrong_accepted']}\n"
        )

print("\nResults saved to:")
print(output_file)