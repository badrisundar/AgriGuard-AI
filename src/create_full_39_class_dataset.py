import os
import random
import shutil
from pathlib import Path

# =========================
# Configuration
# =========================

SOURCE_DIR = Path(
    "dataset/Plant_leave_diseases_dataset_without_augmentation"
)

OUTPUT_DIR = Path(
    "dataset/processed/full_39_class"
)

SEED = 42

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

random.seed(SEED)

# =========================
# Validation
# =========================

if not SOURCE_DIR.exists():
    raise FileNotFoundError(
        f"Source dataset not found: {SOURCE_DIR}"
    )

if abs(TRAIN_RATIO + VAL_RATIO + TEST_RATIO - 1.0) > 1e-6:
    raise ValueError("Split ratios must add up to 1.0")

# =========================
# Create output directories
# =========================

for split in ["train", "val", "test"]:
    (OUTPUT_DIR / split).mkdir(
        parents=True,
        exist_ok=True
    )

# =========================
# Get classes
# =========================

classes = sorted(
    [
        folder.name
        for folder in SOURCE_DIR.iterdir()
        if folder.is_dir()
    ]
)

print("=" * 70)
print("FULL 39-CLASS PLANTVILLAGE DATASET PREPARATION")
print("=" * 70)

print(f"Source: {SOURCE_DIR}")
print(f"Output: {OUTPUT_DIR}")
print(f"Number of classes: {len(classes)}")
print(f"Random seed: {SEED}")
print()

if len(classes) != 39:
    raise ValueError(
        f"Expected 39 classes, but found {len(classes)}"
    )

# =========================
# Process each class
# =========================

total_images = 0
total_train = 0
total_val = 0
total_test = 0

print(
    f"{'Class':<60} {'Total':>7} "
    f"{'Train':>7} {'Val':>7} {'Test':>7}"
)

print("-" * 95)

for class_name in classes:

    source_class_dir = SOURCE_DIR / class_name

    images = [
        file
        for file in source_class_dir.iterdir()
        if file.is_file()
        and file.suffix.lower() in {
            ".jpg",
            ".jpeg",
            ".png",
            ".bmp",
            ".webp"
        }
    ]

    random.shuffle(images)

    total = len(images)

    train_count = int(total * TRAIN_RATIO)
    val_count = int(total * VAL_RATIO)

    train_images = images[:train_count]
    val_images = images[
        train_count:train_count + val_count
    ]
    test_images = images[
        train_count + val_count:
    ]

    splits = {
        "train": train_images,
        "val": val_images,
        "test": test_images
    }

    for split, split_images in splits.items():

        destination_class_dir = (
            OUTPUT_DIR / split / class_name
        )

        destination_class_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        for image_path in split_images:

            destination_path = (
                destination_class_dir / image_path.name
            )

            shutil.copy2(
                image_path,
                destination_path
            )

    print(
        f"{class_name:<60} "
        f"{total:>7} "
        f"{len(train_images):>7} "
        f"{len(val_images):>7} "
        f"{len(test_images):>7}"
    )

    total_images += total
    total_train += len(train_images)
    total_val += len(val_images)
    total_test += len(test_images)

# =========================
# Final summary
# =========================

print("-" * 95)

print(
    f"{'TOTAL':<60} "
    f"{total_images:>7} "
    f"{total_train:>7} "
    f"{total_val:>7} "
    f"{total_test:>7}"
)

print()
print("=" * 70)
print("DATASET PREPARATION COMPLETE")
print("=" * 70)

print(f"Classes : {len(classes)}")
print(f"Total   : {total_images}")
print(f"Train   : {total_train}")
print(f"Val     : {total_val}")
print(f"Test    : {total_test}")

print()
print(f"Dataset saved to:")
print(OUTPUT_DIR)

print()
print("Raw dataset was NOT modified.")