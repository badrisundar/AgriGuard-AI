import os
import torch
import torch.nn as nn
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader


# ============================================================
# Configuration
# ============================================================

DATA_DIR = "dataset/processed/plantdoc_adaptation/test"

CHECKPOINT = (
    "models/mobilenetv3_plantdoc_finetuned_best.pth"
)

# LOCKED FROM VALIDATION SET
THRESHOLD = 0.65

BATCH_SIZE = 16


# ============================================================
# Device
# ============================================================

device = torch.device(
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)

print(f"Using device: {device}")


# ============================================================
# Test transform
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
# Held-out PlantDoc test dataset
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

print(f"Test images: {len(dataset)}")
print(f"Number of classes: {len(dataset.classes)}")

print("\nClasses:")
for i, cls in enumerate(dataset.classes):
    print(f"{i}: {cls}")


# ============================================================
# MobileNetV3 Large
# ============================================================

model = models.mobilenet_v3_large(
    weights=None
)

model.classifier[3] = nn.Linear(
    model.classifier[3].in_features,
    7
)


# ============================================================
# Load fine-tuned checkpoint
# ============================================================

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
# Prediction
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

        all_labels.extend(
            labels.numpy()
        )

        all_predictions.extend(
            predictions.cpu().numpy()
        )

        all_confidences.extend(
            confidence.cpu().numpy()
        )


all_labels = torch.tensor(all_labels)
all_predictions = torch.tensor(all_predictions)
all_confidences = torch.tensor(all_confidences)


# ============================================================
# Overall prediction accuracy
# ============================================================

correct = (
    all_predictions == all_labels
)

overall_accuracy = (
    correct.float().mean().item()
)


# ============================================================
# Apply LOCKED threshold
# ============================================================

accepted = (
    all_confidences >= THRESHOLD
)

uncertain = ~accepted

accepted_count = accepted.sum().item()
uncertain_count = uncertain.sum().item()

coverage = (
    accepted_count / len(dataset)
)


# ============================================================
# Accuracy among accepted predictions
# ============================================================

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


# ============================================================
# Errors rejected by uncertainty mechanism
# ============================================================

incorrect_total = (
    (~correct).sum().item()
)

incorrect_rejected = (
    (~correct) & uncertain
).sum().item()

if incorrect_total > 0:

    error_rejection_rate = (
        incorrect_rejected / incorrect_total
    )

else:

    error_rejection_rate = 0.0


# ============================================================
# Results
# ============================================================

print("\n" + "=" * 70)
print("FINAL LOCKED CONFIDENCE EVALUATION")
print("=" * 70)

print(f"Model: MobileNetV3 Large")
print(f"Dataset: Held-out PlantDoc Test Set")
print(f"Test Images: {len(dataset)}")

print("\nLocked threshold:")
print(f"Threshold = {THRESHOLD:.2f}")

print("\nOverall:")
print(
    f"Overall Accuracy: "
    f"{overall_accuracy * 100:.2f}%"
)

print(
    f"Average Confidence: "
    f"{all_confidences.mean().item() * 100:.2f}%"
)

print("\nConfidence-aware results:")

print(
    f"Accepted Predictions: "
    f"{accepted_count}"
)

print(
    f"Uncertain Predictions: "
    f"{uncertain_count}"
)

print(
    f"Coverage: "
    f"{coverage * 100:.2f}%"
)

print(
    f"Accuracy of Accepted Predictions: "
    f"{accepted_accuracy * 100:.2f}%"
)

print(
    f"Wrong Accepted Predictions: "
    f"{wrong_accepted}"
)

print(
    f"Incorrect Predictions Rejected: "
    f"{incorrect_rejected}"
)

print(
    f"Total Incorrect Predictions: "
    f"{incorrect_total}"
)

print(
    f"Error Rejection Rate: "
    f"{error_rejection_rate * 100:.2f}%"
)


# ============================================================
# Final system interpretation
# ============================================================

print("\n" + "=" * 70)
print("SYSTEM INTERPRETATION")
print("=" * 70)

print(
    f"Images with confidence >= {THRESHOLD:.2f}: "
    f"AUTOMATIC PREDICTION"
)

print(
    f"Images with confidence < {THRESHOLD:.2f}: "
    f"UNCERTAIN / EXPERT REVIEW"
)


# ============================================================
# Save results
# ============================================================

os.makedirs(
    "results",
    exist_ok=True
)

output_file = (
    "results/"
    "mobilenetv3_final_confidence_test.txt"
)

with open(output_file, "w") as f:

    f.write(
        "AgriGuard AI - Final Confidence-Aware Evaluation\n"
    )

    f.write("=" * 60 + "\n\n")

    f.write(
        "Model: MobileNetV3 Large\n"
    )

    f.write(
        "Dataset: Held-out PlantDoc Test Set\n"
    )

    f.write(
        f"Test Images: {len(dataset)}\n\n"
    )

    f.write(
        f"Locked Threshold: {THRESHOLD:.2f}\n\n"
    )

    f.write(
        f"Overall Accuracy: "
        f"{overall_accuracy * 100:.2f}%\n"
    )

    f.write(
        f"Average Confidence: "
        f"{all_confidences.mean().item() * 100:.2f}%\n"
    )

    f.write(
        f"Accepted Predictions: "
        f"{accepted_count}\n"
    )

    f.write(
        f"Uncertain Predictions: "
        f"{uncertain_count}\n"
    )

    f.write(
        f"Coverage: "
        f"{coverage * 100:.2f}%\n"
    )

    f.write(
        f"Accuracy of Accepted Predictions: "
        f"{accepted_accuracy * 100:.2f}%\n"
    )

    f.write(
        f"Wrong Accepted Predictions: "
        f"{wrong_accepted}\n"
    )

    f.write(
        f"Incorrect Predictions Rejected: "
        f"{incorrect_rejected}\n"
    )

    f.write(
        f"Total Incorrect Predictions: "
        f"{incorrect_total}\n"
    )

    f.write(
        f"Error Rejection Rate: "
        f"{error_rejection_rate * 100:.2f}%\n"
    )

    f.write("\nSystem Rule:\n")

    f.write(
        f"Confidence >= {THRESHOLD:.2f}: "
        "Automatic prediction\n"
    )

    f.write(
        f"Confidence < {THRESHOLD:.2f}: "
        "Uncertain / expert review\n"
    )


print("\nResults saved to:")
print(output_file)