import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import numpy as np


# =========================
# Paths
# =========================
MODEL_PATH = "models/mobilenetv3_full39_plantdoc_adapted_best.pth"
VAL_DIR = "dataset/processed/plantdoc_full39_adaptation/val"
RESULT_PATH = "results/full39_plantdoc_confidence_calibration.txt"

DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)

IMAGE_SIZE = 224
BATCH_SIZE = 16


# =========================
# 39 PlantVillage classes
# =========================
CLASS_NAMES = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Background_without_leaves",
    "Blueberry___healthy",
    "Cherry___Powdery_mildew",
    "Cherry___healthy",
    "Corn___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn___Common_rust",
    "Corn___Northern_Leaf_Blight",
    "Corn___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch",
    "Strawberry___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy"
]

CLASS_TO_IDX = {
    name: idx for idx, name in enumerate(CLASS_NAMES)
}


# =========================
# Dataset
# =========================
class PlantDocValidationDataset(Dataset):

    def __init__(self, root_dir):
        self.samples = []

        for folder in sorted(os.listdir(root_dir)):
            folder_path = os.path.join(root_dir, folder)

            if not os.path.isdir(folder_path):
                continue

            if folder not in CLASS_TO_IDX:
                continue

            label = CLASS_TO_IDX[folder]

            for filename in sorted(os.listdir(folder_path)):
                path = os.path.join(folder_path, filename)

                try:
                    Image.open(path).verify()
                    self.samples.append((path, label))
                except Exception:
                    pass

        self.transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]

        image = Image.open(path).convert("RGB")
        image = self.transform(image)

        return image, label


# =========================
# Load model
# =========================
print("Device:", DEVICE)

model = models.mobilenet_v3_large(weights=None)

model.classifier[3] = nn.Linear(
    model.classifier[3].in_features,
    39
)

checkpoint = torch.load(
    MODEL_PATH,
    map_location="cpu"
)

model.load_state_dict(checkpoint["model_state_dict"])
model = model.to(DEVICE)
model.eval()


# =========================
# Load validation data
# =========================
dataset = PlantDocValidationDataset(VAL_DIR)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

print("Validation images:", len(dataset))


# =========================
# Collect predictions
# =========================
all_confidences = []
all_correct = []

with torch.no_grad():

    for images, labels in loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs = model(images)

        probabilities = torch.softmax(outputs, dim=1)

        confidence, predictions = torch.max(
            probabilities,
            dim=1
        )

        correct = predictions == labels

        all_confidences.extend(
            confidence.cpu().numpy()
        )

        all_correct.extend(
            correct.cpu().numpy()
        )


all_confidences = np.array(all_confidences)
all_correct = np.array(all_correct)


# =========================
# Overall validation results
# =========================
overall_accuracy = all_correct.mean() * 100
average_confidence = all_confidences.mean() * 100

print()
print("========================================")
print("CONFIDENCE CALIBRATION")
print("========================================")
print(f"Validation images: {len(dataset)}")
print(f"Overall accuracy: {overall_accuracy:.2f}%")
print(f"Average confidence: {average_confidence:.2f}%")
print()


# =========================
# Threshold evaluation
# =========================
thresholds = np.arange(0.50, 0.96, 0.05)

results = []

print(
    f"{'Threshold':<12}"
    f"{'Accepted':<12}"
    f"{'Coverage':<12}"
    f"{'Accepted Acc.':<16}"
    f"{'Rejected':<12}"
    f"{'Error Reject.':<16}"
)

print("-" * 80)

for threshold in thresholds:

    accepted = all_confidences >= threshold

    accepted_count = accepted.sum()
    rejected_count = (~accepted).sum()

    coverage = (
        accepted_count / len(dataset) * 100
    )

    if accepted_count > 0:
        accepted_accuracy = (
            all_correct[accepted].mean() * 100
        )
    else:
        accepted_accuracy = 0.0

    total_errors = (~all_correct).sum()

    rejected_errors = (
        (~all_correct) & (~accepted)
    ).sum()

    if total_errors > 0:
        error_rejection = (
            rejected_errors / total_errors * 100
        )
    else:
        error_rejection = 0.0

    results.append({
        "threshold": threshold,
        "accepted": accepted_count,
        "coverage": coverage,
        "accepted_accuracy": accepted_accuracy,
        "rejected": rejected_count,
        "error_rejection": error_rejection
    })

    print(
        f"{threshold:<12.2f}"
        f"{accepted_count:<12}"
        f"{coverage:<12.2f}"
        f"{accepted_accuracy:<16.2f}"
        f"{rejected_count:<12}"
        f"{error_rejection:<16.2f}"
    )


# =========================
# Preliminary threshold selection
# Criterion:
# accepted accuracy >= 80%
# among those, choose maximum coverage
# =========================
valid_results = [
    r for r in results
    if r["accepted_accuracy"] >= 80.0
    and r["accepted"] > 0
]

print()

if valid_results:

    selected = max(
        valid_results,
        key=lambda x: x["coverage"]
    )

    print("Preliminary selected threshold:")
    print(f"Threshold: {selected['threshold']:.2f}")
    print(f"Coverage: {selected['coverage']:.2f}%")
    print(
        f"Accepted accuracy: "
        f"{selected['accepted_accuracy']:.2f}%"
    )
    print(
        f"Error rejection: "
        f"{selected['error_rejection']:.2f}%"
    )

else:

    selected = None

    print(
        "No threshold achieved "
        "80% accepted accuracy."
    )


# =========================
# Save results
# =========================
os.makedirs("results", exist_ok=True)

with open(RESULT_PATH, "w") as f:

    f.write("AgriGuard AI - Full 39-Class PlantDoc Confidence Calibration\n")
    f.write("=" * 70 + "\n\n")

    f.write(f"Validation images: {len(dataset)}\n")
    f.write(f"Overall accuracy: {overall_accuracy:.2f}%\n")
    f.write(f"Average confidence: {average_confidence:.2f}%\n\n")

    f.write(
        "Threshold | Accepted | Coverage | "
        "Accepted Accuracy | Rejected | Error Rejection\n"
    )

    f.write("-" * 90 + "\n")

    for r in results:

        f.write(
            f"{r['threshold']:.2f} | "
            f"{r['accepted']} | "
            f"{r['coverage']:.2f}% | "
            f"{r['accepted_accuracy']:.2f}% | "
            f"{r['rejected']} | "
            f"{r['error_rejection']:.2f}%\n"
        )

    f.write("\n")

    if selected:

        f.write(
            f"Preliminary selected threshold: "
            f"{selected['threshold']:.2f}\n"
        )

        f.write(
            f"Selected coverage: "
            f"{selected['coverage']:.2f}%\n"
        )

        f.write(
            f"Selected accepted accuracy: "
            f"{selected['accepted_accuracy']:.2f}%\n"
        )

        f.write(
            f"Selected error rejection: "
            f"{selected['error_rejection']:.2f}%\n"
        )

print()
print("Saved:", RESULT_PATH)