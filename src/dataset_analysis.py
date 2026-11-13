from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Raw dataset directory
DATASET_DIR = (
    PROJECT_ROOT
    / "dataset"
    / "Plant_leave_diseases_dataset_without_augmentation"
)

# Supported image formats
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}


print("Dataset path:")
print(DATASET_DIR)

print("\nDataset exists:", DATASET_DIR.exists())

if not DATASET_DIR.exists():
    print("\nERROR: Dataset folder not found.")
    exit()


# Get all class folders
classes = sorted(
    folder
    for folder in DATASET_DIR.iterdir()
    if folder.is_dir()
)


print("\nTotal classes:", len(classes))

print("\nImage count per class:")
print("-" * 70)

total_images = 0

for class_folder in classes:

    image_count = sum(
        1
        for file in class_folder.iterdir()
        if file.is_file() and file.suffix in IMAGE_EXTENSIONS
    )

    total_images += image_count

    print(f"{class_folder.name:<55} {image_count:>6}")


print("-" * 70)
print(f"{'TOTAL':<55} {total_images:>6}")