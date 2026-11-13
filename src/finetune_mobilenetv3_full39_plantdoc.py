import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from tqdm import tqdm


# ============================================================
# Configuration
# ============================================================

DATA_DIR = "dataset/processed/plantdoc_full39_adaptation"
MODEL_PATH = "models/mobilenetv3_full39_best.pth"
SAVE_PATH = "models/mobilenetv3_full39_plantdoc_adapted_best.pth"

BATCH_SIZE = 16
EPOCHS = 10
LR = 0.0001
NUM_CLASSES = 39
SEED = 42


# ============================================================
# Original 39 PlantVillage classes
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
# Reproducibility
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


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
# Transforms
# ============================================================

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


val_transform = transforms.Compose([
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

class PlantDocAdaptationDataset(Dataset):

    def __init__(self, root_dir, transform=None):

        self.root_dir = root_dir
        self.transform = transform
        self.samples = []

        for folder_name in sorted(os.listdir(root_dir)):

            folder_path = os.path.join(
                root_dir,
                folder_name
            )

            if not os.path.isdir(folder_path):
                continue

            # Folder names are already original PlantVillage names
            if folder_name not in CLASS_TO_INDEX:
                print(
                    "WARNING: Unknown folder skipped:",
                    folder_name
                )
                continue

            class_index = CLASS_TO_INDEX[folder_name]

            for filename in os.listdir(folder_path):

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
                        (filepath, class_index)
                    )

                except Exception:
                    print(
                        "Skipping invalid image:",
                        filepath
                    )

        print(
            f"Loaded {len(self.samples)} images from {root_dir}"
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):

        filepath, label = self.samples[index]

        image = Image.open(
            filepath
        ).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


# ============================================================
# Datasets
# ============================================================

train_dataset = PlantDocAdaptationDataset(
    os.path.join(DATA_DIR, "train"),
    train_transform
)

val_dataset = PlantDocAdaptationDataset(
    os.path.join(DATA_DIR, "val"),
    val_transform
)

print()
print("Train images:", len(train_dataset))
print("Validation images:", len(val_dataset))


# ============================================================
# Verify dataset labels
# ============================================================

train_labels = sorted(
    set(label for _, label in train_dataset.samples)
)

val_labels = sorted(
    set(label for _, label in val_dataset.samples)
)

print()
print("Classes represented in training:")

for index in train_labels:
    print(
        f"{index}: {CLASS_NAMES[index]}"
    )

print()
print("Classes represented in validation:")

for index in val_labels:
    print(
        f"{index}: {CLASS_NAMES[index]}"
    )


# ============================================================
# DataLoaders
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# ============================================================
# Load existing 39-class MobileNetV3
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


# ============================================================
# Freeze feature extractor
# ============================================================

for param in model.features.parameters():
    param.requires_grad = False

for param in model.classifier.parameters():
    param.requires_grad = True


# ============================================================
# Loss + optimizer
# ============================================================

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(
    model.classifier.parameters(),
    lr=LR
)


# ============================================================
# Training
# ============================================================

best_val_accuracy = 0.0

for epoch in range(EPOCHS):

    model.train()

    # Feature extractor remains frozen
    for param in model.features.parameters():
        param.requires_grad = False

    running_loss = 0.0
    correct = 0
    total = 0

    progress = tqdm(
        train_loader,
        desc=f"Epoch {epoch + 1}/{EPOCHS}"
    )

    for images, labels in progress:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()

        running_loss += (
            loss.item() * images.size(0)
        )

        predictions = outputs.argmax(dim=1)

        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)

        progress.set_postfix(
            loss=f"{loss.item():.4f}",
            acc=f"{100 * correct / total:.2f}%"
        )

    train_loss = running_loss / total
    train_accuracy = correct / total


    # ========================================================
    # Validation
    # ========================================================

    model.eval()

    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            val_loss += (
                loss.item() * images.size(0)
            )

            predictions = outputs.argmax(dim=1)

            val_correct += (
                predictions == labels
            ).sum().item()

            val_total += labels.size(0)

    val_loss /= val_total
    val_accuracy = val_correct / val_total


    # ========================================================
    # Epoch results
    # ========================================================

    print()
    print("=" * 60)
    print(f"Epoch {epoch + 1}/{EPOCHS}")
    print(
        f"Train Loss: {train_loss:.4f}"
    )
    print(
        f"Train Accuracy: "
        f"{train_accuracy * 100:.2f}%"
    )
    print(
        f"Val Loss: {val_loss:.4f}"
    )
    print(
        f"Val Accuracy: "
        f"{val_accuracy * 100:.2f}%"
    )
    print("=" * 60)


    # ========================================================
    # Save best model
    # ========================================================

    if val_accuracy > best_val_accuracy:

        best_val_accuracy = val_accuracy

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "validation_accuracy": best_val_accuracy,
                "epoch": epoch + 1,
                "num_classes": NUM_CLASSES,
                "class_names": CLASS_NAMES
            },
            SAVE_PATH
        )

        print(
            f"Best model saved: "
            f"{best_val_accuracy * 100:.2f}%"
        )


print()
print("=" * 60)
print("Training completed.")
print(
    f"Best validation accuracy: "
    f"{best_val_accuracy * 100:.2f}%"
)
print(
    "Saved model:",
    SAVE_PATH
)
print("=" * 60)