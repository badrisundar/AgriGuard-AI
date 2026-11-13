from pathlib import Path
import random
import shutil

from torchvision.datasets import CIFAR10


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PLANT_DATASET = (
    PROJECT_ROOT
    / "dataset"
    / "processed"
    / "experiment_1"
    / "train"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "dataset"
    / "processed"
    / "leaf_validator"
)

SEED = 42

TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

MAX_POSITIVE_IMAGES = 5000
MAX_NEGATIVE_IMAGES = 5000


# --------------------------------------------------
# SET SEED
# --------------------------------------------------

random.seed(SEED)


# --------------------------------------------------
# CREATE DIRECTORIES
# --------------------------------------------------

for split in ["train", "validation", "test"]:
    (OUTPUT_DIR / split / "leaf").mkdir(
        parents=True,
        exist_ok=True
    )

    (OUTPUT_DIR / split / "non_leaf").mkdir(
        parents=True,
        exist_ok=True
    )


# --------------------------------------------------
# COLLECT PLANT LEAF IMAGES
# --------------------------------------------------

print("Collecting plant leaf images...")

positive_images = []

for class_dir in PLANT_DATASET.iterdir():

    if not class_dir.is_dir():
        continue

    for image_path in class_dir.iterdir():

        if image_path.suffix.lower() in [
            ".jpg",
            ".jpeg",
            ".png",
            ".webp"
        ]:
            positive_images.append(image_path)


random.shuffle(positive_images)

positive_images = positive_images[
    :MAX_POSITIVE_IMAGES
]

print(
    f"Leaf images selected: "
    f"{len(positive_images)}"
)


# --------------------------------------------------
# DOWNLOAD CIFAR-10
# --------------------------------------------------

print("Downloading/loading CIFAR-10...")

cifar_dataset = CIFAR10(
    root=str(
        PROJECT_ROOT / "dataset" / "cifar10"
    ),
    train=True,
    download=True
)


# CIFAR-10 labels
#
# 0 airplane
# 1 automobile
# 2 bird
# 3 cat
# 4 deer
# 5 dog
# 6 frog
# 7 horse
# 8 ship
# 9 truck
#
# None of these are plant-leaf images.

negative_indices = list(
    range(len(cifar_dataset))
)

random.shuffle(negative_indices)

negative_indices = negative_indices[
    :MAX_NEGATIVE_IMAGES
]


# --------------------------------------------------
# SPLIT FUNCTION
# --------------------------------------------------

def split_data(items):

    n = len(items)

    train_end = int(
        n * TRAIN_RATIO
    )

    val_end = train_end + int(
        n * VAL_RATIO
    )

    train_items = items[:train_end]

    val_items = items[
        train_end:val_end
    ]

    test_items = items[
        val_end:
    ]

    return (
        train_items,
        val_items,
        test_items
    )


positive_train, positive_val, positive_test = split_data(
    positive_images
)

negative_train, negative_val, negative_test = split_data(
    negative_indices
)


# --------------------------------------------------
# COPY POSITIVE IMAGES
# --------------------------------------------------

def copy_positive(
    image_paths,
    split
):

    destination = (
        OUTPUT_DIR
        / split
        / "leaf"
    )

    for index, image_path in enumerate(
        image_paths
    ):

        target = (
            destination
            / f"leaf_{index:05d}"
            f"{image_path.suffix.lower()}"
        )

        shutil.copy2(
            image_path,
            target
        )


# --------------------------------------------------
# SAVE CIFAR NEGATIVE IMAGES
# --------------------------------------------------

def save_negative(
    indices,
    split
):

    destination = (
        OUTPUT_DIR
        / split
        / "non_leaf"
    )

    for index, cifar_index in enumerate(
        indices
    ):

        image, label = cifar_dataset[
            cifar_index
        ]

        target = (
            destination
            / f"non_leaf_{index:05d}.png"
        )

        image.save(target)


# --------------------------------------------------
# CREATE DATASET
# --------------------------------------------------

print("Creating training split...")

copy_positive(
    positive_train,
    "train"
)

save_negative(
    negative_train,
    "train"
)


print("Creating validation split...")

copy_positive(
    positive_val,
    "validation"
)

save_negative(
    negative_val,
    "validation"
)


print("Creating test split...")

copy_positive(
    positive_test,
    "test"
)

save_negative(
    negative_test,
    "test"
)


# --------------------------------------------------
# SUMMARY
# --------------------------------------------------

print("\nValidator dataset created successfully.")

print("\nTraining:")
print(
    f"Leaf:     {len(positive_train)}"
)
print(
    f"Non-leaf: {len(negative_train)}"
)

print("\nValidation:")
print(
    f"Leaf:     {len(positive_val)}"
)
print(
    f"Non-leaf: {len(negative_val)}"
)

print("\nTest:")
print(
    f"Leaf:     {len(positive_test)}"
)
print(
    f"Non-leaf: {len(negative_test)}"
)

print(
    f"\nDataset location:\n"
    f"{OUTPUT_DIR}"
)