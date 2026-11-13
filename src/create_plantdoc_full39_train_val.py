from pathlib import Path
import shutil
import random

# ============================================================
# Configuration
# ============================================================

SOURCE_ROOT = Path(
    "dataset/PlantDoc"
)

OUTPUT_ROOT = Path(
    "dataset/processed/plantdoc_full39_adaptation"
)

SEED = 42

TRAIN_RATIO = 0.80
VAL_RATIO = 0.20

# ============================================================
# PlantDoc -> PlantVillage mappings
# ============================================================

MAPPINGS = {
    "Apple Scab Leaf":
        "Apple___Apple_scab",

    "Apple rust leaf":
        "Apple___Cedar_apple_rust",

    "Bell_pepper leaf spot":
        "Pepper,_bell___Bacterial_spot",

    "Corn Gray leaf spot":
        "Corn___Cercospora_leaf_spot Gray_leaf_spot",

    "Corn rust leaf":
        "Corn___Common_rust",

    "Potato leaf early blight":
        "Potato___Early_blight",

    "Potato leaf late blight":
        "Potato___Late_blight",

    "Squash Powdery mildew leaf":
        "Squash___Powdery_mildew",

    "Tomato Early blight leaf":
        "Tomato___Early_blight",

    "Tomato Septoria leaf spot":
        "Tomato___Septoria_leaf_spot",

    "Tomato leaf bacterial spot":
        "Tomato___Bacterial_spot",

    "Tomato leaf late blight":
        "Tomato___Late_blight",

    "Tomato leaf mosaic virus":
        "Tomato___Tomato_mosaic_virus",

    "Tomato leaf yellow virus":
        "Tomato___Tomato_Yellow_Leaf_Curl_Virus",

    "Tomato mold leaf":
        "Tomato___Leaf_Mold",

    "Tomato two spotted spider mites leaf":
        "Tomato___Spider_mites Two-spotted_spider_mite",

    "grape leaf black rot":
        "Grape___Black_rot",
}

VALID_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
}

random.seed(SEED)

# ============================================================
# Prepare output directories
# ============================================================

if OUTPUT_ROOT.exists():
    shutil.rmtree(OUTPUT_ROOT)

TRAIN_ROOT = OUTPUT_ROOT / "train"
VAL_ROOT = OUTPUT_ROOT / "val"

TRAIN_ROOT.mkdir(
    parents=True,
    exist_ok=True
)

VAL_ROOT.mkdir(
    parents=True,
    exist_ok=True
)

# ============================================================
# Process PlantDoc TRAIN images only
# ============================================================

print("=" * 90)
print("CREATING PLANTDOC FULL-39 ADAPTATION DATASET")
print("=" * 90)

total_images = 0
total_train = 0
total_val = 0

for plantdoc_class, plantvillage_class in MAPPINGS.items():

    source_dir = (
        SOURCE_ROOT /
        "train" /
        plantdoc_class
    )

    if not source_dir.exists():
        print(
            f"SKIPPING: {plantdoc_class} "
            f"(directory not found)"
        )
        continue

    images = [
        file
        for file in source_dir.iterdir()
        if file.is_file()
        and file.suffix.lower() in VALID_EXTENSIONS
    ]

    if not images:
        print(
            f"SKIPPING: {plantdoc_class} "
            f"(0 images)"
        )
        continue

    random.shuffle(images)

    train_count = int(
        len(images) * TRAIN_RATIO
    )

    train_images = images[:train_count]
    val_images = images[train_count:]

    train_class_dir = (
        TRAIN_ROOT /
        plantvillage_class
    )

    val_class_dir = (
        VAL_ROOT /
        plantvillage_class
    )

    train_class_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    val_class_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    for image in train_images:

        shutil.copy2(
            image,
            train_class_dir / image.name
        )

    for image in val_images:

        shutil.copy2(
            image,
            val_class_dir / image.name
        )

    total_images += len(images)
    total_train += len(train_images)
    total_val += len(val_images)

    print(
        f"{plantdoc_class:<42} "
        f"Total={len(images):>4} | "
        f"Train={len(train_images):>4} | "
        f"Val={len(val_images):>4}"
    )

# ============================================================
# Summary
# ============================================================

print()
print("=" * 90)
print("ADAPTATION DATASET COMPLETE")
print("=" * 90)

print(f"Mapped classes: {len(MAPPINGS)}")
print(f"Total images: {total_images}")
print(f"Training images: {total_train}")
print(f"Validation images: {total_val}")

print()
print(f"Output:")
print(OUTPUT_ROOT)

print()
print("IMPORTANT:")
print("- Only PlantDoc TRAIN images were used.")
print("- PlantDoc TEST images were NOT copied.")
print("- The 134-image PlantDoc test set remains untouched.")
print("- Original PlantDoc dataset was not modified.")
print("- Random seed: 42")