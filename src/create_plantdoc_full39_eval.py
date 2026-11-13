from pathlib import Path
import shutil

PLANTDOC_ROOT = Path("dataset/PlantDoc")

OUTPUT_ROOT = Path(
    "dataset/processed/plantdoc_full39_eval/test"
)

MAPPINGS = {
    "Apple Scab Leaf": "Apple___Apple_scab",
    "Apple rust leaf": "Apple___Cedar_apple_rust",
    "Bell_pepper leaf spot": "Pepper,_bell___Bacterial_spot",
    "Corn Gray leaf spot": "Corn___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn rust leaf": "Corn___Common_rust",
    "Potato leaf early blight": "Potato___Early_blight",
    "Potato leaf late blight": "Potato___Late_blight",
    "Squash Powdery mildew leaf": "Squash___Powdery_mildew",
    "Tomato Early blight leaf": "Tomato___Early_blight",
    "Tomato Septoria leaf spot": "Tomato___Septoria_leaf_spot",
    "Tomato leaf bacterial spot": "Tomato___Bacterial_spot",
    "Tomato leaf late blight": "Tomato___Late_blight",
    "Tomato leaf mosaic virus": "Tomato___Tomato_mosaic_virus",
    "Tomato leaf yellow virus": "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato mold leaf": "Tomato___Leaf_Mold",
    "Tomato two spotted spider mites leaf":
        "Tomato___Spider_mites Two-spotted_spider_mite",
    "grape leaf black rot": "Grape___Black_rot",
}

VALID_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".webp"
}

# Remove previous evaluation copy so there are no empty/stale folders
if OUTPUT_ROOT.exists():
    shutil.rmtree(OUTPUT_ROOT)

OUTPUT_ROOT.mkdir(
    parents=True,
    exist_ok=True
)

print("=" * 80)
print("CREATING PLANTDOC ZERO-SHOT EVALUATION DATASET")
print("=" * 80)

total_copied = 0
classes_with_test_images = 0

for plantdoc_class, plantvillage_class in MAPPINGS.items():

    source_dir = (
        PLANTDOC_ROOT /
        "test" /
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
            f"(0 test images)"
        )
        continue

    destination_dir = (
        OUTPUT_ROOT /
        plantvillage_class
    )

    destination_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    for image_path in images:

        shutil.copy2(
            image_path,
            destination_dir / image_path.name
        )

    classes_with_test_images += 1
    total_copied += len(images)

    print(
        f"{plantdoc_class:<42} -> "
        f"{plantvillage_class:<55} "
        f"{len(images):>3} images"
    )

print()
print("=" * 80)
print("DATASET CREATION COMPLETE")
print("=" * 80)

print(f"Mapped classes: {len(MAPPINGS)}")
print(
    f"Classes with test images: "
    f"{classes_with_test_images}"
)
print(f"Total evaluation images: {total_copied}")

print()
print(f"Output: {OUTPUT_ROOT}")

print()
print("Only PlantDoc TEST images are included.")
print("PlantDoc TRAIN images were not used.")
print("Original PlantDoc dataset was not modified.")
print("This dataset is for ZERO-SHOT evaluation.")