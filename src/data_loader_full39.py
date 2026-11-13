import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# =========================
# Configuration
# =========================

DATA_DIR = "dataset/processed/full_39_class"

BATCH_SIZE = 16
IMAGE_SIZE = 224
NUM_WORKERS = 0

# =========================
# Device
# =========================

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

# =========================
# Transforms
# =========================

train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
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
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# =========================
# Datasets
# =========================

train_dataset = datasets.ImageFolder(
    f"{DATA_DIR}/train",
    transform=train_transform
)

val_dataset = datasets.ImageFolder(
    f"{DATA_DIR}/val",
    transform=eval_transform
)

test_dataset = datasets.ImageFolder(
    f"{DATA_DIR}/test",
    transform=eval_transform
)

# =========================
# DataLoaders
# =========================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS
)

# =========================
# Verification
# =========================

print("=" * 60)
print("FULL 39-CLASS DATA LOADER")
print("=" * 60)

print(f"Device: {DEVICE}")
print(f"Number of classes: {len(train_dataset.classes)}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Image size: {IMAGE_SIZE}x{IMAGE_SIZE}")

print()
print(f"Train images: {len(train_dataset)}")
print(f"Validation images: {len(val_dataset)}")
print(f"Test images: {len(test_dataset)}")

print()
print("Classes:")

for index, class_name in enumerate(train_dataset.classes):
    print(f"{index:2d} -> {class_name}")

print()
print("=" * 60)