from pathlib import Path
import shutil

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PLANTDOC_DIR = PROJECT_ROOT / "dataset" / "PlantDoc"
OUTPUT_DIR = PROJECT_ROOT / "dataset" / "processed" / "plantdoc_eval"


# Only use mappings that are sufficiently clear.
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


if not PLANTDOC_DIR.exists():
    raise FileNotFoundError(
        f"PlantDoc dataset not found:\n{PLANTDOC_DIR}"
    )


if OUTPUT_DIR.exists():
    raise FileExistsError(
        f"\nOutput directory already exists:\n{OUTPUT_DIR}\n\n"
        "Delete it manually if you want to recreate it."
    )


print("\nCreating PlantDoc evaluation dataset")
print("=" * 70)

total_images = 0


for plantdoc_class, target_class in CLASS_MAPPING.items():

    print(f"\n{plantdoc_class}")
    print(f"  -> {target_class}")

    target_train_dir = OUTPUT_DIR / "test" / target_class
    target_train_dir.mkdir(parents=True, exist_ok=True)

    count = 0

    for split in ["train", "test"]:

        source_dir = PLANTDOC_DIR / split / plantdoc_class

        if not source_dir.exists():
            print(f"WARNING: Missing directory: {source_dir}")
            continue

        for image_path in source_dir.iterdir():

            if (
                image_path.is_file()
                and image_path.suffix.lower()
                in IMAGE_EXTENSIONS
            ):

                destination = target_train_dir / (
                    f"{split}_{image_path.name}"
                )

                shutil.copy2(
                    image_path,
                    destination
                )

                count += 1

    print(f"  Images copied: {count}")

    total_images += count


print("\n" + "=" * 70)
print(f"Total evaluation images: {total_images}")
print(f"Output directory: {OUTPUT_DIR}")
print("=" * 70)