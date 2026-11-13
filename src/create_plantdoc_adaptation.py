from pathlib import Path
import shutil


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PLANTDOC_DIR = PROJECT_ROOT / "dataset" / "PlantDoc"

OUTPUT_DIR = (
    PROJECT_ROOT
    / "dataset"
    / "processed"
    / "plantdoc_adaptation"
)


# ============================================================
# PlantDoc -> AgriGuard class mapping
# ============================================================

CLASS_MAPPING = {
    "Corn Gray leaf spot":
        "Corn___Cercospora_leaf_spot Gray_leaf_spot",

    "Corn rust leaf":
        "Corn___Common_rust",

    "grape leaf black rot":
        "Grape___Black_rot",

    "Tomato leaf bacterial spot":
        "Tomato___Bacterial_spot",

    "Tomato Early blight leaf":
        "Tomato___Early_blight",

    "Tomato leaf late blight":
        "Tomato___Late_blight",

    "Tomato mold leaf":
        "Tomato___Leaf_Mold",
}


IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
}


# ============================================================
# Safety checks
# ============================================================

if not PLANTDOC_DIR.exists():
    raise FileNotFoundError(
        f"PlantDoc dataset not found:\n{PLANTDOC_DIR}"
    )


if OUTPUT_DIR.exists():
    raise FileExistsError(
        f"\nOutput directory already exists:\n{OUTPUT_DIR}\n\n"
        "Delete it manually if you want to recreate it."
    )


# ============================================================
# Create dataset
# ============================================================

print("\nCreating PlantDoc adaptation dataset")
print("=" * 70)


total_train = 0
total_test = 0


for plantdoc_class, target_class in CLASS_MAPPING.items():

    print(f"\n{plantdoc_class}")
    print(f"  -> {target_class}")

    train_output = (
        OUTPUT_DIR
        / "train"
        / target_class
    )

    test_output = (
        OUTPUT_DIR
        / "test"
        / target_class
    )

    train_output.mkdir(
        parents=True,
        exist_ok=True
    )

    test_output.mkdir(
        parents=True,
        exist_ok=True
    )


    # --------------------------------------------------------
    # PlantDoc TRAIN
    # --------------------------------------------------------

    source_train = (
        PLANTDOC_DIR
        / "train"
        / plantdoc_class
    )

    train_count = 0

    if source_train.exists():

        for image_path in source_train.iterdir():

            if (
                image_path.is_file()
                and image_path.suffix.lower()
                in IMAGE_EXTENSIONS
            ):

                destination = (
                    train_output
                    / image_path.name
                )

                shutil.copy2(
                    image_path,
                    destination
                )

                train_count += 1


    # --------------------------------------------------------
    # PlantDoc TEST
    # --------------------------------------------------------

    source_test = (
        PLANTDOC_DIR
        / "test"
        / plantdoc_class
    )

    test_count = 0

    if source_test.exists():

        for image_path in source_test.iterdir():

            if (
                image_path.is_file()
                and image_path.suffix.lower()
                in IMAGE_EXTENSIONS
            ):

                destination = (
                    test_output
                    / image_path.name
                )

                shutil.copy2(
                    image_path,
                    destination
                )

                test_count += 1


    print(
        f"  Train: {train_count}"
    )

    print(
        f"  Test:  {test_count}"
    )


    total_train += train_count
    total_test += test_count


# ============================================================
# Final summary
# ============================================================

print("\n" + "=" * 70)

print(
    f"Total PlantDoc train images: "
    f"{total_train}"
)

print(
    f"Total PlantDoc test images: "
    f"{total_test}"
)

print(
    f"Total images: "
    f"{total_train + total_test}"
)

print(
    f"\nOutput directory:\n{OUTPUT_DIR}"
)

print("=" * 70)