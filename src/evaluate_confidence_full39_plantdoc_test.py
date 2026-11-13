import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import numpy as np


MODEL_PATH = "models/mobilenetv3_full39_plantdoc_adapted_best.pth"
TEST_DIR = "dataset/processed/plantdoc_full39_eval/test"
RESULT_PATH = "results/full39_plantdoc_confidence_test.txt"

DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)

IMAGE_SIZE = 224
BATCH_SIZE = 16

# Locked threshold selected using validation set only
THRESHOLD = 0.75


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


class PlantDocTestDataset(Dataset):

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

        return image, label, path


print("Device:", DEVICE)

# =========================
# Load model
# =========================

model = models.mobilenet_v3_large(weights=None)

model.classifier[3] = nn.Linear(
    model.classifier[3].in_features,
    39
)

checkpoint = torch.load(
    MODEL_PATH,
    map_location="cpu"
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model = model.to(DEVICE)
model.eval()


# =========================
# Load locked test set
# =========================

dataset = PlantDocTestDataset(TEST_DIR)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

print("Test images:", len(dataset))
print("Threshold:", THRESHOLD)


# =========================
# Evaluate
# =========================

all_confidences = []
all_correct = []
all_predictions = []
all_labels = []
all_paths = []

with torch.no_grad():

    for images, labels, paths in loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs = model(images)

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

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

        all_predictions.extend(
            predictions.cpu().numpy()
        )

        all_labels.extend(
            labels.cpu().numpy()
        )

        all_paths.extend(paths)


all_confidences = np.array(all_confidences)
all_correct = np.array(all_correct)
all_predictions = np.array(all_predictions)
all_labels = np.array(all_labels)


# =========================
# Overall performance
# =========================

overall_accuracy = (
    all_correct.mean() * 100
)

accepted = all_confidences >= THRESHOLD
rejected = ~accepted

accepted_count = accepted.sum()
rejected_count = rejected.sum()

accepted_accuracy = (
    all_correct[accepted].mean() * 100
    if accepted_count > 0
    else 0
)

coverage = (
    accepted_count / len(dataset) * 100
)

total_errors = (~all_correct).sum()

rejected_errors = (
    (~all_correct) & rejected
).sum()

error_rejection = (
    rejected_errors / total_errors * 100
    if total_errors > 0
    else 0
)


# =========================
# Print results
# =========================

print()
print("========================================")
print("LOCKED TEST CONFIDENCE EVALUATION")
print("========================================")

print(f"Test images: {len(dataset)}")
print(f"Overall accuracy: {overall_accuracy:.2f}%")
print()

print(f"Threshold: {THRESHOLD:.2f}")
print(f"Accepted: {accepted_count}")
print(f"Rejected: {rejected_count}")
print(f"Coverage: {coverage:.2f}%")
print(f"Accepted accuracy: {accepted_accuracy:.2f}%")
print(f"Total errors: {total_errors}")
print(f"Rejected errors: {rejected_errors}")
print(f"Error rejection: {error_rejection:.2f}%")

# =========================
# Accepted errors
# =========================

accepted_error_indices = np.where(
    accepted & (~all_correct)
)[0]

print()
print("Accepted wrong predictions:")
print(len(accepted_error_indices))

for index in accepted_error_indices:

    print(
        f"\nImage: {all_paths[index]}"
    )

    print(
        f"Actual: "
        f"{CLASS_NAMES[all_labels[index]]}"
    )

    print(
        f"Predicted: "
        f"{CLASS_NAMES[all_predictions[index]]}"
    )

    print(
        f"Confidence: "
        f"{all_confidences[index] * 100:.2f}%"
    )


# =========================
# Save results
# =========================

os.makedirs("results", exist_ok=True)

with open(RESULT_PATH, "w") as f:

    f.write(
        "AgriGuard AI - Full 39-Class PlantDoc "
        "Locked Test Confidence Evaluation\n"
    )

    f.write("=" * 70 + "\n\n")

    f.write(
        f"Test images: {len(dataset)}\n"
    )

    f.write(
        f"Threshold: {THRESHOLD:.2f}\n"
    )

    f.write(
        f"Overall accuracy: "
        f"{overall_accuracy:.2f}%\n"
    )

    f.write(
        f"Accepted: {accepted_count}\n"
    )

    f.write(
        f"Rejected: {rejected_count}\n"
    )

    f.write(
        f"Coverage: {coverage:.2f}%\n"
    )

    f.write(
        f"Accepted accuracy: "
        f"{accepted_accuracy:.2f}%\n"
    )

    f.write(
        f"Total errors: {total_errors}\n"
    )

    f.write(
        f"Rejected errors: {rejected_errors}\n"
    )

    f.write(
        f"Error rejection: "
        f"{error_rejection:.2f}%\n"
    )

    f.write("\nAccepted wrong predictions:\n")

    for index in accepted_error_indices:

        f.write(
            f"\nImage: {all_paths[index]}\n"
        )

        f.write(
            f"Actual: "
            f"{CLASS_NAMES[all_labels[index]]}\n"
        )

        f.write(
            f"Predicted: "
            f"{CLASS_NAMES[all_predictions[index]]}\n"
        )

        f.write(
            f"Confidence: "
            f"{all_confidences[index] * 100:.2f}%\n"
        )

print()
print("Saved:", RESULT_PATH)