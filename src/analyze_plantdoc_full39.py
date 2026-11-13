from pathlib import Path

# ============================================================
# PlantDoc -> PlantVillage 39-class safe mappings
# ============================================================

PLANTDOC_ROOT = Path("dataset/PlantDoc")

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


def count_images(directory):
    if not directory.exists():
        return 0

    return sum(
        1
        for file in directory.iterdir()
        if file.is_file()
        and file.suffix.lower() in VALID_EXTENSIONS
    )


# ============================================================
# Analysis
# ============================================================

print("=" * 100)
print("PLANTDOC -> PLANTVILLAGE FULL-39 MAPPING ANALYSIS")
print("=" * 100)

print()
print(
    f"{'PlantDoc Class':<42}"
    f"{'Train':>8}"
    f"{'Test':>8}"
    f"{'Total':>8}"
    f"  PlantVillage Class"
)

print("-" * 100)

total_train = 0
total_test = 0

for plantdoc_class, plantvillage_class in MAPPINGS.items():

    train_dir = (
        PLANTDOC_ROOT /
        "train" /
        plantdoc_class
    )

    test_dir = (
        PLANTDOC_ROOT /
        "test" /
        plantdoc_class
    )

    train_count = count_images(train_dir)
    test_count = count_images(test_dir)
    total_count = train_count + test_count

    total_train += train_count
    total_test += test_count

    print(
        f"{plantdoc_class:<42}"
        f"{train_count:>8}"
        f"{test_count:>8}"
        f"{total_count:>8}"
        f"  {plantvillage_class}"
    )

print("-" * 100)

print(
    f"{'TOTAL':<42}"
    f"{total_train:>8}"
    f"{total_test:>8}"
    f"{total_train + total_test:>8}"
)

print()
print("=" * 100)
print(f"Number of mapped classes: {len(MAPPINGS)}")
print(f"Mapped PlantDoc train images: {total_train}")
print(f"Mapped PlantDoc test images:  {total_test}")
print(f"Mapped PlantDoc total images: {total_train + total_test}")
print("=" * 100)