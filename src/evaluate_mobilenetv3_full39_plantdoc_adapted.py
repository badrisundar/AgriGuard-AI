import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


# ============================================================
# Configuration
# ============================================================

TEST_DIR = "dataset/processed/plantdoc_full39_eval/test"

MODEL_PATH = (
    "models/"
    "mobilenetv3_full39_plantdoc_adapted_best.pth"
)

OUTPUT_FILE = (
    "results/"
    "mobilenetv3_full39_plantdoc_adapted_test_results.txt"
)

BATCH_SIZE = 16
NUM_CLASSES = 39


# ============================================================
# PlantVillage 39 classes
# ============================================================

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

CLASS_TO_INDEX = {
    name: index
    for index, name in enumerate(CLASS_NAMES)
}


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

class PlantDocTestDataset(Dataset):

    def __init__(self, root_dir):

        self.samples = []

        for folder_name in sorted(
            os.listdir(root_dir)
        ):

            folder_path = os.path.join(
                root_dir,
                folder_name
            )

            if not os.path.isdir(folder_path):
                continue

            # The test folders already contain
            # the original PlantVillage class names.
            if folder_name not in CLASS_TO_INDEX:

                print(
                    "Skipping unknown folder:",
                    folder_name
                )

                continue

            label = CLASS_TO_INDEX[
                folder_name
            ]

            for filename in os.listdir(
                folder_path
            ):

                filepath = os.path.join(
                    folder_path,
                    filename
                )

                if not os.path.isfile(filepath):
                    continue

                try:

                    with Image.open(filepath) as img:
                        img.verify()

                    self.samples.append(
                        (filepath, label)
                    )

                except Exception:

                    print(
                        "Skipping invalid:",
                        filepath
                    )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):

        filepath, label = self.samples[index]

        image = Image.open(
            filepath
        ).convert("RGB")

        image = transform(image)

        return image, label


# ============================================================
# Device
# ============================================================

if torch.backends.mps.is_available():

    device = torch.device("mps")

elif torch.cuda.is_available():

    device = torch.device("cuda")

else:

    device = torch.device("cpu")

print("Using device:", device)


# ============================================================
# Dataset
# ============================================================

test_dataset = PlantDocTestDataset(
    TEST_DIR
)

print(
    "PlantDoc locked test images:",
    len(test_dataset)
)

if len(test_dataset) == 0:

    raise RuntimeError(
        "No test images found. "
        "Check the test directory structure."
    )


test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# ============================================================
# Load adapted MobileNetV3
# ============================================================

model = models.mobilenet_v3_large(
    weights=None
)

model.classifier[3] = nn.Linear(
    model.classifier[3].in_features,
    NUM_CLASSES
)

checkpoint = torch.load(
    MODEL_PATH,
    map_location="cpu"
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model = model.to(device)

model.eval()


# ============================================================
# Evaluation
# ============================================================

all_labels = []
all_predictions = []

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)

        outputs = model(images)

        predictions = (
            outputs
            .argmax(dim=1)
            .cpu()
            .numpy()
        )

        all_predictions.extend(
            predictions
        )

        all_labels.extend(
            labels.numpy()
        )


# ============================================================
# Metrics
# ============================================================

accuracy = accuracy_score(
    all_labels,
    all_predictions
)

precision = precision_score(
    all_labels,
    all_predictions,
    average="macro",
    zero_division=0
)

recall = recall_score(
    all_labels,
    all_predictions,
    average="macro",
    zero_division=0
)

f1 = f1_score(
    all_labels,
    all_predictions,
    average="macro",
    zero_division=0
)


print()
print("=" * 70)
print("PlantDoc Adapted MobileNetV3")
print("LOCKED TEST RESULTS")
print("=" * 70)

print(
    f"Test Images: {len(test_dataset)}"
)

print(
    f"Accuracy: {accuracy * 100:.2f}%"
)

print(
    f"Macro Precision: {precision * 100:.2f}%"
)

print(
    f"Macro Recall: {recall * 100:.2f}%"
)

print(
    f"Macro F1: {f1 * 100:.2f}%"
)


# ============================================================
# Classification report
# ============================================================

present_labels = sorted(
    set(all_labels)
)

target_names = [
    CLASS_NAMES[index]
    for index in present_labels
]

report = classification_report(
    all_labels,
    all_predictions,
    labels=present_labels,
    target_names=target_names,
    zero_division=0
)

print()
print("Classification Report")
print("=" * 70)
print(report)


# ============================================================
# Confusion Matrix
# ============================================================

cm = confusion_matrix(
    all_labels,
    all_predictions,
    labels=present_labels
)

print()
print("Confusion Matrix")
print("=" * 70)
print(cm)


# ============================================================
# Save results
# ============================================================

os.makedirs(
    "results",
    exist_ok=True
)

with open(
    OUTPUT_FILE,
    "w"
) as f:

    f.write(
        "PlantDoc Adapted MobileNetV3 "
        "Locked Test Evaluation\n"
    )

    f.write("=" * 70 + "\n")

    f.write(
        f"Test Images: {len(test_dataset)}\n"
    )

    f.write(
        f"Accuracy: {accuracy * 100:.2f}%\n"
    )

    f.write(
        f"Macro Precision: "
        f"{precision * 100:.2f}%\n"
    )

    f.write(
        f"Macro Recall: "
        f"{recall * 100:.2f}%\n"
    )

    f.write(
        f"Macro F1: "
        f"{f1 * 100:.2f}%\n\n"
    )

    f.write(
        "Classification Report\n"
    )

    f.write("=" * 70 + "\n")

    f.write(report)

    f.write(
        "\n\nConfusion Matrix\n"
    )

    f.write("=" * 70 + "\n")

    f.write(str(cm))


print()
print(
    "Results saved to:",
    OUTPUT_FILE
)