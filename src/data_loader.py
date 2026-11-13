from pathlib import Path

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


# --------------------------------------------------
# 1. Project paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_ROOT = (
    PROJECT_ROOT
    / "dataset"
    / "processed"
    / "experiment_1"
)


# --------------------------------------------------
# 2. Device
# --------------------------------------------------

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")


print("Using device:", DEVICE)


# --------------------------------------------------
# 3. Image transformations
# --------------------------------------------------

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


# Validation and test remain unchanged
evaluation_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# --------------------------------------------------
# 4. Load datasets
# --------------------------------------------------

train_dataset = datasets.ImageFolder(
    DATASET_ROOT / "train",
    transform=train_transform
)

validation_dataset = datasets.ImageFolder(
    DATASET_ROOT / "validation",
    transform=evaluation_transform
)

test_dataset = datasets.ImageFolder(
    DATASET_ROOT / "test",
    transform=evaluation_transform
)


# --------------------------------------------------
# 5. Create DataLoaders
# --------------------------------------------------

BATCH_SIZE = 16

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

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# --------------------------------------------------
# 6. Display dataset information
# --------------------------------------------------

print("\nDataset information")
print("=" * 50)

print("Number of classes:", len(train_dataset.classes))
print("Training images:", len(train_dataset))
print("Validation images:", len(validation_dataset))
print("Test images:", len(test_dataset))

print("\nClasses:")
for index, class_name in enumerate(train_dataset.classes):
    print(f"{index}: {class_name}")


# --------------------------------------------------
# 7. Test one batch
# --------------------------------------------------

images, labels = next(iter(train_loader))

print("\nOne batch test")
print("=" * 50)

print("Image batch shape:", images.shape)
print("Label batch shape:", labels.shape)

print("Image data type:", images.dtype)
print("Label data type:", labels.dtype)

print("Minimum pixel value:", images.min().item())
print("Maximum pixel value:", images.max().item())


# --------------------------------------------------
# 8. Move batch to device
# --------------------------------------------------

images = images.to(DEVICE)
labels = labels.to(DEVICE)

print("\nBatch successfully moved to:", DEVICE)
print("Images device:", images.device)
print("Labels device:", labels.device)