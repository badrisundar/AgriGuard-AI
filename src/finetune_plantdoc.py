from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from tqdm import tqdm


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = (
    PROJECT_ROOT
    / "dataset"
    / "processed"
    / "plantdoc_finetune"
)

MODEL_DIR = PROJECT_ROOT / "models"

MODEL_DIR.mkdir(exist_ok=True)


# ============================================================
# Configuration
# ============================================================

BATCH_SIZE = 16
EPOCHS = 10
LEARNING_RATE = 0.0001
SEED = 42

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

print(f"\nUsing device: {device}")


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


eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# Datasets
# ============================================================

train_dataset = datasets.ImageFolder(
    DATA_DIR / "train",
    transform=train_transform
)

val_dataset = datasets.ImageFolder(
    DATA_DIR / "validation",
    transform=eval_transform
)


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


CLASS_NAMES = train_dataset.classes
NUM_CLASSES = len(CLASS_NAMES)

print(f"Number of classes: {NUM_CLASSES}")
print(f"Training images: {len(train_dataset)}")
print(f"Validation images: {len(val_dataset)}")

print("\nClasses:")
for i, class_name in enumerate(CLASS_NAMES):
    print(f"{i}: {class_name}")


# ============================================================
# Model builders
# ============================================================

def load_resnet50():

    model = models.resnet50(weights=None)

    checkpoint = torch.load(
        MODEL_DIR / "resnet50_best.pth",
        map_location="cpu",
        weights_only=True
    )

    original_state = checkpoint["model_state_dict"]

    # Load all original weights except the 13-class FC layer
    filtered_state = {
        key: value
        for key, value in original_state.items()
        if not key.startswith("fc.")
    }

    model.load_state_dict(
        filtered_state,
        strict=False
    )

    model.fc = nn.Linear(
        model.fc.in_features,
        NUM_CLASSES
    )

    return model


def load_mobilenetv3():

    model = models.mobilenet_v3_large(weights=None)

    checkpoint = torch.load(
        MODEL_DIR / "mobilenetv3_best.pth",
        map_location="cpu",
        weights_only=True
    )

    original_state = checkpoint["model_state_dict"]

    filtered_state = {
        key: value
        for key, value in original_state.items()
        if not key.startswith("classifier.3.")
    }

    model.load_state_dict(
        filtered_state,
        strict=False
    )

    model.classifier[-1] = nn.Linear(
        model.classifier[-1].in_features,
        NUM_CLASSES
    )

    return model


def load_efficientnet():

    model = models.efficientnet_b0(weights=None)

    checkpoint = torch.load(
        MODEL_DIR / "efficientnet_b0_best.pth",
        map_location="cpu",
        weights_only=True
    )

    original_state = checkpoint["model_state_dict"]

    filtered_state = {
        key: value
        for key, value in original_state.items()
        if not key.startswith("classifier.1.")
    }

    model.load_state_dict(
        filtered_state,
        strict=False
    )

    model.classifier[-1] = nn.Linear(
        model.classifier[-1].in_features,
        NUM_CLASSES
    )

    return model


# ============================================================
# Freeze feature extractor
# ============================================================

def freeze_features(model, model_name):

    if model_name == "ResNet50":

        for parameter in model.parameters():
            parameter.requires_grad = False

        for parameter in model.fc.parameters():
            parameter.requires_grad = True


    elif model_name == "MobileNetV3":

        for parameter in model.parameters():
            parameter.requires_grad = False

        for parameter in model.classifier[-1].parameters():
            parameter.requires_grad = True


    elif model_name == "EfficientNet-B0":

        for parameter in model.parameters():
            parameter.requires_grad = False

        for parameter in model.classifier[-1].parameters():
            parameter.requires_grad = True


# ============================================================
# Training function
# ============================================================

def train_model(model, model_name):

    print("\n" + "=" * 70)
    print(f"Fine-tuning {model_name}")
    print("=" * 70)

    freeze_features(
        model,
        model_name
    )

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        filter(
            lambda parameter: parameter.requires_grad,
            model.parameters()
        ),
        lr=LEARNING_RATE
    )

    best_val_accuracy = 0.0
    best_epoch = 0

    for epoch in range(EPOCHS):

        # ----------------------------------------------------
        # Training
        # ----------------------------------------------------

        model.train()

        # Keep frozen feature extractor in eval mode
        # where applicable to stabilize BatchNorm.
        if model_name == "ResNet50":
            model.fc.train()

        elif model_name in [
            "MobileNetV3",
            "EfficientNet-B0"
        ]:
            model.classifier[-1].train()

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
                loss.item()
                * images.size(0)
            )

            predictions = torch.argmax(
                outputs,
                dim=1
            )

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

        train_loss = running_loss / total
        train_accuracy = correct / total


        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        model.eval()

        val_correct = 0
        val_total = 0

        with torch.no_grad():

            for images, labels in val_loader:

                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)

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
        )

        print(
            f"\nEpoch {epoch + 1}/{EPOCHS}"
        )

        print(
            f"Train Loss: {train_loss:.4f}"
        )

        print(
            f"Train Accuracy: "
            f"{train_accuracy * 100:.2f}%"
        )

        print(
            f"Validation Accuracy: "
            f"{val_accuracy * 100:.2f}%"
        )


        # ----------------------------------------------------
        # Save best model
        # ----------------------------------------------------

        if val_accuracy > best_val_accuracy:

            best_val_accuracy = val_accuracy
            best_epoch = epoch + 1

            checkpoint_path = (
                MODEL_DIR
                / f"{model_name.lower().replace(' ', '_')}"
                  "_plantdoc_finetuned_best.pth"
            )

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "validation_accuracy": val_accuracy,
                    "epoch": epoch + 1,
                    "num_classes": NUM_CLASSES,
                    "class_names": CLASS_NAMES
                },
                checkpoint_path
            )

            print(
                f"Best model saved: "
                f"{checkpoint_path}"
            )


    print("\n" + "-" * 70)

    print(
        f"{model_name} complete"
    )

    print(
        f"Best validation accuracy: "
        f"{best_val_accuracy * 100:.2f}%"
    )

    print(
        f"Best epoch: {best_epoch}"
    )

    print("-" * 70)

    return best_val_accuracy


# ============================================================
# Train all three models
# ============================================================

results = {}


results["ResNet50"] = train_model(
    load_resnet50(),
    "ResNet50"
)


results["MobileNetV3"] = train_model(
    load_mobilenetv3(),
    "MobileNetV3"
)


results["EfficientNet-B0"] = train_model(
    load_efficientnet(),
    "EfficientNet-B0"
)


# ============================================================
# Final summary
# ============================================================

print("\n" + "=" * 70)
print("PLANTDOC FINE-TUNING SUMMARY")
print("=" * 70)

for model_name, accuracy in results.items():

    print(
        f"{model_name:<20}"
        f"{accuracy * 100:.2f}%"
    )

print("=" * 70)