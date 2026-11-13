import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


# -----------------------------
# Reproducibility
# -----------------------------
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.backends.mps.is_available():
    torch.mps.manual_seed(SEED)


# -----------------------------
# Paths
# -----------------------------
DATA_DIR = "dataset/processed/leaf_validator_v3"
MODEL_DIR = "models"
RESULTS_DIR = "results"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


# -----------------------------
# Device
# -----------------------------
device = torch.device(
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)

print("Device:", device)


# -----------------------------
# Transforms
# -----------------------------
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

eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# -----------------------------
# Datasets
# -----------------------------
train_dataset = datasets.ImageFolder(
    os.path.join(DATA_DIR, "train"),
    transform=train_transform
)

val_dataset = datasets.ImageFolder(
    os.path.join(DATA_DIR, "validation"),
    transform=eval_transform
)

print("Classes:", train_dataset.classes)
print("Training images:", len(train_dataset))
print("Validation images:", len(val_dataset))


# -----------------------------
# DataLoaders
# -----------------------------
train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=16,
    shuffle=False,
    num_workers=0
)


# -----------------------------
# MobileNetV3 Small
# -----------------------------
weights = models.MobileNet_V3_Small_Weights.DEFAULT

model = models.mobilenet_v3_small(
    weights=weights
)

model.classifier[3] = nn.Linear(
    model.classifier[3].in_features,
    2
)

model = model.to(device)


# -----------------------------
# Loss + optimizer
# -----------------------------
criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=0.0001
)


# -----------------------------
# Training
# -----------------------------
epochs = 5

best_val_accuracy = 0.0

history = []


for epoch in range(epochs):

    # =========================
    # TRAIN
    # =========================
    model.train()

    train_correct = 0
    train_total = 0
    train_loss = 0.0

    for images, labels in train_loader:

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

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        train_correct += (
            predictions == labels
        ).sum().item()

        train_total += labels.size(0)

        train_loss += loss.item()

    train_accuracy = (
        train_correct / train_total
    ) * 100

    avg_train_loss = (
        train_loss / len(train_loader)
    )


    # =========================
    # VALIDATION
    # =========================
    model.eval()

    val_correct = 0
    val_total = 0
    val_loss = 0.0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            val_loss += loss.item()

            predictions = torch.argmax(
                outputs,
                dim=1
            )

            val_correct += (
                predictions == labels
            ).sum().item()

            val_total += labels.size(0)

    val_accuracy = (
        val_correct / val_total
    ) * 100

    avg_val_loss = (
        val_loss / len(val_loader)
    )


    print(
        f"Epoch {epoch + 1}/{epochs} | "
        f"Train Loss: {avg_train_loss:.4f} | "
        f"Train Acc: {train_accuracy:.2f}% | "
        f"Val Loss: {avg_val_loss:.4f} | "
        f"Val Acc: {val_accuracy:.2f}%"
    )


    history.append({
        "epoch": epoch + 1,
        "train_loss": avg_train_loss,
        "train_accuracy": train_accuracy,
        "val_loss": avg_val_loss,
        "val_accuracy": val_accuracy
    })


    # =========================
    # SAVE BEST MODEL
    # =========================
    if val_accuracy > best_val_accuracy:

        best_val_accuracy = val_accuracy

        model_path = os.path.join(
            MODEL_DIR,
            "leaf_validator_v3_best.pth"
        )

        torch.save(
            model.state_dict(),
            model_path
        )

        print(
            f"Saved best model: {model_path}"
        )


# -----------------------------
# Save history
# -----------------------------
history_path = os.path.join(
    RESULTS_DIR,
    "leaf_validator_v3_training_history.txt"
)

with open(history_path, "w") as f:

    for item in history:

        f.write(
            f"Epoch {item['epoch']} | "
            f"Train Loss: {item['train_loss']:.4f} | "
            f"Train Accuracy: "
            f"{item['train_accuracy']:.2f}% | "
            f"Validation Loss: "
            f"{item['val_loss']:.4f} | "
            f"Validation Accuracy: "
            f"{item['val_accuracy']:.2f}%\n"
        )

    f.write(
        f"\nBest Validation Accuracy: "
        f"{best_val_accuracy:.2f}%\n"
    )


print("\nTraining complete.")

print(
    f"Best Validation Accuracy: "
    f"{best_val_accuracy:.2f}%"
)

print(
    "Model saved to: "
    "models/leaf_validator_v3_best.pth"
)