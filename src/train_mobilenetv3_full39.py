import random
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torchvision import models
from tqdm import tqdm

from data_loader_full39 import (
    train_loader,
    val_loader,
    DEVICE,
    train_dataset
)

# =========================
# Reproducibility
# =========================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# =========================
# Configuration
# =========================

NUM_CLASSES = 39
EPOCHS = 5
LEARNING_RATE = 0.0001

MODEL_PATH = "models/mobilenetv3_full39_best.pth"

# =========================
# Model
# =========================

print("=" * 70)
print("TRAINING FINAL 39-CLASS MOBILENETV3 LARGE")
print("=" * 70)

print(f"Device: {DEVICE}")
print(f"Classes: {NUM_CLASSES}")
print(f"Epochs: {EPOCHS}")
print(f"Learning rate: {LEARNING_RATE}")

model = models.mobilenet_v3_large(
    weights=models.MobileNet_V3_Large_Weights.DEFAULT
)

# Replace classifier
model.classifier[3] = nn.Linear(
    model.classifier[3].in_features,
    NUM_CLASSES
)

model = model.to(DEVICE)

# =========================
# Loss and optimizer
# =========================

criterion = nn.CrossEntropyLoss()

optimizer = AdamW(
    model.parameters(),
    lr=LEARNING_RATE
)

# =========================
# Training
# =========================

best_val_accuracy = 0.0

train_losses = []
val_losses = []

train_accuracies = []
val_accuracies = []

for epoch in range(EPOCHS):

    # ---------------------
    # Training
    # ---------------------

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    progress_bar = tqdm(
        train_loader,
        desc=f"Epoch {epoch + 1}/{EPOCHS} [Train]"
    )

    for images, labels in progress_bar:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item() * images.size(0)

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        progress_bar.set_postfix(
            loss=loss.item()
        )

    train_loss = running_loss / total
    train_accuracy = correct / total

    # ---------------------
    # Validation
    # ---------------------

    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)

            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)

            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_loss = running_loss / total
    val_accuracy = correct / total

    # ---------------------
    # Store metrics
    # ---------------------

    train_losses.append(train_loss)
    val_losses.append(val_loss)

    train_accuracies.append(train_accuracy)
    val_accuracies.append(val_accuracy)

    print()
    print(
        f"Epoch {epoch + 1}/{EPOCHS} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Train Acc: {train_accuracy * 100:.2f}% | "
        f"Val Loss: {val_loss:.4f} | "
        f"Val Acc: {val_accuracy * 100:.2f}%"
    )

    # ---------------------
    # Save best model
    # ---------------------

    if val_accuracy > best_val_accuracy:

        best_val_accuracy = val_accuracy

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "validation_accuracy": best_val_accuracy,
                "epoch": epoch + 1,
                "num_classes": NUM_CLASSES,
                "class_names": train_dataset.classes
            },
            MODEL_PATH
        )

        print(
            f"✓ Best model saved "
            f"(Val Acc: {best_val_accuracy * 100:.2f}%)"
        )

# =========================
# Final summary
# =========================

print()
print("=" * 70)
print("TRAINING COMPLETE")
print("=" * 70)

print(
    f"Best Validation Accuracy: "
    f"{best_val_accuracy * 100:.2f}%"
)

print(f"Model saved to: {MODEL_PATH}")