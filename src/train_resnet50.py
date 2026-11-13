from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from torchvision import models
from torchvision.models import ResNet50_Weights

from tqdm import tqdm

from data_loader import (
    train_loader,
    validation_loader,
    DEVICE
)


# --------------------------------------------------
# 1. Reproducibility
# --------------------------------------------------

torch.manual_seed(42)


# --------------------------------------------------
# 2. Project paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_DIR = PROJECT_ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

BEST_MODEL_PATH = MODEL_DIR / "resnet50_best.pth"

LOSS_CURVE_PATH = RESULTS_DIR / "resnet50_loss_curve.png"
ACCURACY_CURVE_PATH = RESULTS_DIR / "resnet50_accuracy_curve.png"


# --------------------------------------------------
# 3. Configuration
# --------------------------------------------------

NUM_CLASSES = 13
NUM_EPOCHS = 5
LEARNING_RATE = 0.0001


# --------------------------------------------------
# 4. Load pretrained ResNet50
# --------------------------------------------------

print("\nLoading pretrained ResNet50...")

weights = ResNet50_Weights.DEFAULT

model = models.resnet50(weights=weights)


# Replace final classification layer
model.fc = nn.Linear(
    model.fc.in_features,
    NUM_CLASSES
)


# Move model to Apple GPU / CPU
model = model.to(DEVICE)


print("Model loaded successfully.")
print("Device:", DEVICE)


# --------------------------------------------------
# 5. Loss function and optimizer
# --------------------------------------------------

criterion = nn.CrossEntropyLoss()

optimizer = optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE
)


# --------------------------------------------------
# 6. Training history
# --------------------------------------------------

train_losses = []
train_accuracies = []

validation_losses = []
validation_accuracies = []


# --------------------------------------------------
# 7. Training function
# --------------------------------------------------

def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer
):

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    progress_bar = tqdm(
        loader,
        desc="Training",
        leave=False
    )

    for images, labels in progress_bar:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        # Clear previous gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)

        # Calculate loss
        loss = criterion(
            outputs,
            labels
        )

        # Backpropagation
        loss.backward()

        # Update weights
        optimizer.step()

        # Statistics
        running_loss += (
            loss.item() * images.size(0)
        )

        _, predictions = torch.max(
            outputs,
            1
        )

        total += labels.size(0)

        correct += (
            predictions == labels
        ).sum().item()

        progress_bar.set_postfix(
            loss=f"{loss.item():.4f}"
        )

    epoch_loss = running_loss / total

    epoch_accuracy = correct / total

    return epoch_loss, epoch_accuracy


# --------------------------------------------------
# 8. Validation function
# --------------------------------------------------

def validate(
    model,
    loader,
    criterion
):

    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():

        progress_bar = tqdm(
            loader,
            desc="Validation",
            leave=False
        )

        for images, labels in progress_bar:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            # Forward pass
            outputs = model(images)

            # Calculate loss
            loss = criterion(
                outputs,
                labels
            )

            running_loss += (
                loss.item() * images.size(0)
            )

            _, predictions = torch.max(
                outputs,
                1
            )

            total += labels.size(0)

            correct += (
                predictions == labels
            ).sum().item()

    epoch_loss = running_loss / total

    epoch_accuracy = correct / total

    return epoch_loss, epoch_accuracy


# --------------------------------------------------
# 9. Training loop
# --------------------------------------------------

best_validation_accuracy = 0.0

print("\nStarting training...")
print("=" * 70)

for epoch in range(NUM_EPOCHS):

    print(
        f"\nEpoch {epoch + 1}/{NUM_EPOCHS}"
    )

    # ------------------------------
    # Training
    # ------------------------------

    train_loss, train_accuracy = train_one_epoch(
        model,
        train_loader,
        criterion,
        optimizer
    )

    # ------------------------------
    # Validation
    # ------------------------------

    validation_loss, validation_accuracy = validate(
        model,
        validation_loader,
        criterion
    )

    # ------------------------------
    # Save history
    # ------------------------------

    train_losses.append(train_loss)
    train_accuracies.append(train_accuracy)

    validation_losses.append(validation_loss)
    validation_accuracies.append(validation_accuracy)

    # ------------------------------
    # Print results
    # ------------------------------

    print(
        f"Train Loss: {train_loss:.4f} | "
        f"Train Accuracy: {train_accuracy:.4f}"
    )

    print(
        f"Validation Loss: {validation_loss:.4f} | "
        f"Validation Accuracy: {validation_accuracy:.4f}"
    )

    # ------------------------------
    # Save best model
    # ------------------------------

    if validation_accuracy > best_validation_accuracy:

        best_validation_accuracy = validation_accuracy

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "validation_accuracy": validation_accuracy,
                "epoch": epoch + 1,
                "num_classes": NUM_CLASSES,
            },
            BEST_MODEL_PATH
        )

        print(
            f"✓ Best model saved: "
            f"{BEST_MODEL_PATH}"
        )


# --------------------------------------------------
# 10. Final summary
# --------------------------------------------------

print("\n" + "=" * 70)

print("Training completed.")

print(
    f"Best validation accuracy: "
    f"{best_validation_accuracy:.4f}"
)

print(
    f"Best model saved at:\n"
    f"{BEST_MODEL_PATH}"
)


# --------------------------------------------------
# 11. Generate Loss Curve
# --------------------------------------------------

epochs = range(1, NUM_EPOCHS + 1)

plt.figure(figsize=(8, 5))

plt.plot(
    epochs,
    train_losses,
    marker="o",
    label="Training Loss"
)

plt.plot(
    epochs,
    validation_losses,
    marker="o",
    label="Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.title("ResNet50 Training and Validation Loss")

plt.xticks(epochs)

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.savefig(
    LOSS_CURVE_PATH,
    dpi=300
)

plt.close()

print(
    f"✓ Loss curve saved at:\n"
    f"{LOSS_CURVE_PATH}"
)


# --------------------------------------------------
# 12. Generate Accuracy Curve
# --------------------------------------------------

plt.figure(figsize=(8, 5))

plt.plot(
    epochs,
    train_accuracies,
    marker="o",
    label="Training Accuracy"
)

plt.plot(
    epochs,
    validation_accuracies,
    marker="o",
    label="Validation Accuracy"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy")

plt.title("ResNet50 Training and Validation Accuracy")

plt.xticks(epochs)

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.savefig(
    ACCURACY_CURVE_PATH,
    dpi=300
)

plt.close()

print(
    f"✓ Accuracy curve saved at:\n"
    f"{ACCURACY_CURVE_PATH}"
)


# --------------------------------------------------
# 13. Print complete training history
# --------------------------------------------------

print("\nTraining History")
print("=" * 70)

for i in range(NUM_EPOCHS):

    print(
        f"Epoch {i + 1}: "
        f"Train Loss = {train_losses[i]:.4f}, "
        f"Train Acc = {train_accuracies[i]:.4f}, "
        f"Val Loss = {validation_losses[i]:.4f}, "
        f"Val Acc = {validation_accuracies[i]:.4f}"
    )

print("=" * 70)

print("\nAll training results generated successfully.")