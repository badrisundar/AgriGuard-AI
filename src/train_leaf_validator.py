from pathlib import Path
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from tqdm import tqdm


# ==================================================
# CONFIGURATION
# ==================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_DIR = (
    PROJECT_ROOT
    / "dataset"
    / "processed"
    / "leaf_validator"
)

MODEL_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

MODEL_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

BATCH_SIZE = 16
EPOCHS = 5
LEARNING_RATE = 0.0001
SEED = 42

IMAGE_SIZE = 224


# ==================================================
# REPRODUCIBILITY
# ==================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


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
# TRANSFORMS
# ==================================================

train_transform = transforms.Compose([
    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.RandomHorizontalFlip(),

    transforms.RandomRotation(10),

    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2
    ),

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


validation_transform = transforms.Compose([
    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

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
# DATASETS
# ==================================================

train_dataset = datasets.ImageFolder(
    DATASET_DIR / "train",
    transform=train_transform
)

validation_dataset = datasets.ImageFolder(
    DATASET_DIR / "validation",
    transform=validation_transform
)


print("\nClasses:")
print(train_dataset.classes)

print(
    f"\nTraining images: "
    f"{len(train_dataset)}"
)

print(
    f"Validation images: "
    f"{len(validation_dataset)}"
)


# ==================================================
# DATA LOADERS
# ==================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

validation_loader = DataLoader(
    validation_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# ==================================================
# MODEL
# ==================================================

print("\nLoading MobileNetV3 Small...")

model = models.mobilenet_v3_small(
    weights=models.MobileNet_V3_Small_Weights.DEFAULT
)

# Replace classifier
model.classifier[3] = nn.Linear(
    model.classifier[3].in_features,
    2
)

model = model.to(device)


# ==================================================
# LOSS + OPTIMIZER
# ==================================================

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE
)


# ==================================================
# TRAINING
# ==================================================

best_validation_accuracy = 0.0

train_losses = []
validation_losses = []

train_accuracies = []
validation_accuracies = []


for epoch in range(EPOCHS):

    print(
        f"\nEpoch "
        f"{epoch + 1}/{EPOCHS}"
    )

    # ------------------------------------------------
    # TRAIN
    # ------------------------------------------------

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    progress_bar = tqdm(
        train_loader,
        desc="Training"
    )

    for images, labels in progress_bar:

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

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        correct += (
            (predictions == labels)
            .sum()
            .item()
        )

        total += labels.size(0)

        progress_bar.set_postfix(
            loss=loss.item()
        )

    train_loss = (
        running_loss / total
    )

    train_accuracy = (
        correct / total
    )


    # ------------------------------------------------
    # VALIDATION
    # ------------------------------------------------

    model.eval()

    validation_running_loss = 0.0
    validation_correct = 0
    validation_total = 0

    with torch.no_grad():

        for images, labels in validation_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            validation_running_loss += (
                loss.item()
                * images.size(0)
            )

            predictions = torch.argmax(
                outputs,
                dim=1
            )

            validation_correct += (
                (predictions == labels)
                .sum()
                .item()
            )

            validation_total += labels.size(0)

    validation_loss = (
        validation_running_loss
        / validation_total
    )

    validation_accuracy = (
        validation_correct
        / validation_total
    )


    # ------------------------------------------------
    # SAVE METRICS
    # ------------------------------------------------

    train_losses.append(
        train_loss
    )

    validation_losses.append(
        validation_loss
    )

    train_accuracies.append(
        train_accuracy
    )

    validation_accuracies.append(
        validation_accuracy
    )


    print(
        f"Train Loss: "
        f"{train_loss:.4f}"
    )

    print(
        f"Train Accuracy: "
        f"{train_accuracy * 100:.2f}%"
    )

    print(
        f"Validation Loss: "
        f"{validation_loss:.4f}"
    )

    print(
        f"Validation Accuracy: "
        f"{validation_accuracy * 100:.2f}%"
    )


    # ------------------------------------------------
    # BEST CHECKPOINT
    # ------------------------------------------------

    if validation_accuracy > best_validation_accuracy:

        best_validation_accuracy = (
            validation_accuracy
        )

        checkpoint_path = (
            MODEL_DIR
            / "leaf_validator_mobilenetv3_small_best.pth"
        )

        torch.save(
            model.state_dict(),
            checkpoint_path
        )

        print(
            "✓ Best model saved."
        )


# ==================================================
# SAVE TRAINING HISTORY
# ==================================================

history_path = (
    RESULTS_DIR
    / "leaf_validator_training_history.txt"
)

with open(
    history_path,
    "w"
) as file:

    file.write(
        "Leaf Validator - MobileNetV3 Small\n"
    )

    file.write(
        f"Best Validation Accuracy: "
        f"{best_validation_accuracy * 100:.2f}%\n\n"
    )

    for i in range(EPOCHS):

        file.write(
            f"Epoch {i + 1}\n"
        )

        file.write(
            f"Train Loss: "
            f"{train_losses[i]:.4f}\n"
        )

        file.write(
            f"Train Accuracy: "
            f"{train_accuracies[i] * 100:.2f}%\n"
        )

        file.write(
            f"Validation Loss: "
            f"{validation_losses[i]:.4f}\n"
        )

        file.write(
            f"Validation Accuracy: "
            f"{validation_accuracies[i] * 100:.2f}%\n\n"
        )


print(
    "\n========================================"
)

print(
    "Leaf Validator Training Complete"
)

print(
    "========================================"
)

print(
    f"Best Validation Accuracy: "
    f"{best_validation_accuracy * 100:.2f}%"
)

print(
    f"Model saved to:\n"
    f"{MODEL_DIR / 'leaf_validator_mobilenetv3_small_best.pth'}"
)