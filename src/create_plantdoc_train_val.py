from pathlib import Path
import shutil
import random


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

SOURCE_DIR = (
    PROJECT_ROOT
    / "dataset"
    / "processed"
    / "plantdoc_adaptation"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "dataset"
    / "processed"
    / "plantdoc_finetune"
)


# ============================================================
# Configuration
# ============================================================

SEED = 42
TRAIN_RATIO = 0.80


# ============================================================
# Safety checks
# ============================================================

if not SOURCE_DIR.exists():
    raise FileNotFoundError(
        f"Source dataset not found:\n{SOURCE_DIR}"
    )

if OUTPUT_DIR.exists():
    raise FileExistsError(
        f"\nOutput directory already exists:\n{OUTPUT_DIR}\n\n"
        "Delete it manually if you want to recreate it."
    )


# ============================================================
# Random seed
# ============================================================

random.seed(SEED)


# ============================================================
# Create directories
# ============================================================

finetune_train_dir = OUTPUT_DIR / "train"
finetune_val_dir = OUTPUT_DIR / "validation"

finetune_train_dir.mkdir(
    parents=True,
    exist_ok=True
)

finetune_val_dir.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# Find classes
# ============================================================

classes = sorted([
    directory.name
    for directory in (SOURCE_DIR / "train").iterdir()
    if directory.is_dir()
])


print("\nCreating PlantDoc fine-tuning train/validation split")
print("=" * 70)

print(f"Random seed: {SEED}")
print(f"Training ratio: {TRAIN_RATIO * 100:.0f}%")
print(f"Validation ratio: {(1 - TRAIN_RATIO) * 100:.0f}%")


total_train = 0
total_val = 0


# ============================================================
# Split each class separately
# ============================================================

for class_name in classes:

    source_class_dir = (
        SOURCE_DIR
        / "train"
        / class_name
    )

    train_class_dir = (
        finetune_train_dir
        / class_name
    )

    val_class_dir = (
        finetune_val_dir
        / class_name
    )

    train_class_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    val_class_dir.mkdir(
        parents=True,
        exist_ok=True
    )


    # --------------------------------------------------------
    # Get images
    # --------------------------------------------------------

    images = [
        image
        for image in source_class_dir.iterdir()
        if image.is_file()
        and image.suffix.lower()
        in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    ]

    # --------------------------------------------------------
    # Shuffle
    # --------------------------------------------------------

    random.shuffle(images)


    # --------------------------------------------------------
    # Split
    # --------------------------------------------------------

    split_index = int(
        len(images) * TRAIN_RATIO
    )

    train_images = images[:split_index]
    val_images = images[split_index:]


    # --------------------------------------------------------
    # Copy training images
    # --------------------------------------------------------

    for image in train_images:

        shutil.copy2(
            image,
            train_class_dir / image.name
        )


    # --------------------------------------------------------
    # Copy validation images
    # --------------------------------------------------------

    for image in val_images:

        shutil.copy2(
            image,
            val_class_dir / image.name
        )


    print(
        f"{class_name:<55}"
        f"Train: {len(train_images):>4}   "
        f"Validation: {len(val_images):>3}"
    )


    total_train += len(train_images)
    total_val += len(val_images)


# ============================================================
# Summary
# ============================================================

print("\n" + "=" * 70)

print(
    f"Fine-tuning train images: "
    f"{total_train}"
)

print(
    f"Fine-tuning validation images: "
    f"{total_val}"
)

print(
    f"Total adaptation images: "
    f"{total_train + total_val}"
)

print(
    "\nOriginal PlantDoc TEST images remain separate:"
)

print(
    "56 images → "
    "dataset/processed/plantdoc_adaptation/test/"
)

print(
    f"\nOutput directory:\n{OUTPUT_DIR}"
)

print("=" * 70)