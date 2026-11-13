from pathlib import Path
import random
import shutil

# --------------------------------------------------
# 1. Project paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DATASET = (
    PROJECT_ROOT
    / "dataset"
    / "Plant_leave_diseases_dataset_without_augmentation"
)

OUTPUT_DATASET = (
    PROJECT_ROOT
    / "dataset"
    / "processed"
    / "experiment_1"
)


# --------------------------------------------------
# 2. Classes selected for Experiment 1
# --------------------------------------------------

SELECTED_CLASSES = [
    # Corn
    "Corn___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn___Common_rust",
    "Corn___Northern_Leaf_Blight",
    "Corn___healthy",

    # Grape
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",

    # Tomato
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___healthy",
]


# --------------------------------------------------
# 3. Split configuration
# --------------------------------------------------

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

RANDOM_SEED = 42

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".JPG",
    ".JPEG",
    ".PNG"
}


# --------------------------------------------------
# 4. Check paths
# --------------------------------------------------

if not RAW_DATASET.exists():
    raise FileNotFoundError(
        f"Raw dataset not found:\n{RAW_DATASET}"
    )

if OUTPUT_DATASET.exists():
    raise FileExistsError(
        f"Output dataset already exists:\n{OUTPUT_DATASET}\n\n"
        "Delete it manually if you want to recreate the split."
    )


# --------------------------------------------------
# 5. Create output directories
# --------------------------------------------------

for split in ["train", "validation", "test"]:
    (OUTPUT_DATASET / split).mkdir(
        parents=True,
        exist_ok=True
    )


# --------------------------------------------------
# 6. Random generator
# --------------------------------------------------

random_generator = random.Random(RANDOM_SEED)


# --------------------------------------------------
# 7. Process each class
# --------------------------------------------------

total_images = 0

print("\nCreating Experiment 1 dataset...")
print("=" * 70)

for class_name in SELECTED_CLASSES:

    source_folder = RAW_DATASET / class_name

    if not source_folder.exists():
        raise FileNotFoundError(
            f"Class folder not found:\n{source_folder}"
        )

    # Get all images
    images = [
        file
        for file in source_folder.iterdir()
        if file.is_file()
        and file.suffix in IMAGE_EXTENSIONS
    ]

    # Shuffle images using fixed seed
    random_generator.shuffle(images)

    total = len(images)

    # Calculate split sizes
    train_count = int(total * TRAIN_RATIO)
    val_count = int(total * VAL_RATIO)

    train_images = images[:train_count]
    val_images = images[train_count:train_count + val_count]
    test_images = images[train_count + val_count:]

    # Create class directories
    train_folder = OUTPUT_DATASET / "train" / class_name
    val_folder = OUTPUT_DATASET / "validation" / class_name
    test_folder = OUTPUT_DATASET / "test" / class_name

    train_folder.mkdir(parents=True, exist_ok=True)
    val_folder.mkdir(parents=True, exist_ok=True)
    test_folder.mkdir(parents=True, exist_ok=True)

    # Copy images
    for image in train_images:
        shutil.copy2(image, train_folder / image.name)

    for image in val_images:
        shutil.copy2(image, val_folder / image.name)

    for image in test_images:
        shutil.copy2(image, test_folder / image.name)

    total_images += total

    print(
        f"{class_name:<55} "
        f"Total: {total:>4} | "
        f"Train: {len(train_images):>4} | "
        f"Val: {len(val_images):>4} | "
        f"Test: {len(test_images):>4}"
    )


# --------------------------------------------------
# 8. Final summary
# --------------------------------------------------

print("=" * 70)
print("Dataset creation completed successfully!")

print(f"\nTotal images: {total_images}")

print("\nOutput location:")
print(OUTPUT_DATASET)

print("\nDataset structure:")
print("experiment_1/")
print("├── train/")
print("├── validation/")
print("└── test/")