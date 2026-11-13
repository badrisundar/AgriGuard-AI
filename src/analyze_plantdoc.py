from pathlib import Path

from PIL import Image


# --------------------------------------------------
# 1. Project path
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PLANTDOC_DIR = PROJECT_ROOT / "dataset" / "PlantDoc"


# --------------------------------------------------
# 2. Supported image extensions
# --------------------------------------------------

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
}


# --------------------------------------------------
# 3. Check dataset
# --------------------------------------------------

if not PLANTDOC_DIR.exists():

    raise FileNotFoundError(
        f"PlantDoc dataset not found:\n"
        f"{PLANTDOC_DIR}"
    )


# --------------------------------------------------
# 4. Analyze train/test
# --------------------------------------------------

print("\nPlantDoc Dataset Analysis")
print("=" * 70)

total_images = 0


for split in ["train", "test"]:

    split_dir = PLANTDOC_DIR / split

    print(f"\n{split.upper()}")
    print("-" * 70)

    split_total = 0

    class_directories = sorted(
        [
            directory
            for directory in split_dir.iterdir()
            if directory.is_dir()
        ]
    )

    for class_dir in class_directories:

        image_count = 0

        for file_path in class_dir.iterdir():

            if (
                file_path.is_file()
                and file_path.suffix.lower()
                in IMAGE_EXTENSIONS
            ):

                # Verify it is actually a readable image
                try:

                    with Image.open(file_path) as image:
                        image.verify()

                    image_count += 1

                except Exception:

                    print(
                        f"WARNING: Invalid image: "
                        f"{file_path}"
                    )

        print(
            f"{class_dir.name:<55}"
            f"{image_count:>5}"
        )

        split_total += image_count

    print("-" * 70)

    print(
        f"Total {split} images: "
        f"{split_total}"
    )

    total_images += split_total


# --------------------------------------------------
# 5. Overall total
# --------------------------------------------------

print("\n" + "=" * 70)

print(
    f"Total PlantDoc images: "
    f"{total_images}"
)

print("=" * 70)