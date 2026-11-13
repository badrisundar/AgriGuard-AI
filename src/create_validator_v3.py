import os
import shutil
import random

SOURCE = "dataset/processed/leaf_validator"
REAL_NEGATIVES = "dataset/real_hard_negatives/images"
OUTPUT = "dataset/processed/leaf_validator_v3"

SEED = 42
random.seed(SEED)

# Images that contain vegetation/grass and should NOT
# be treated as strict non-leaf negatives.
EXCLUDE = {
    ("buildings_rooms", "11.jpg"),
    ("people_animals", "07.jpg"),
    ("people_animals", "05.jpg"),
    ("people_animals", "04.jpg"),
    ("roads_outdoor", "05.jpg"),
}


# --------------------------------------------------
# Create folders
# --------------------------------------------------

for split in ["train", "validation", "test"]:

    for label in ["leaf", "non_leaf"]:

        os.makedirs(
            os.path.join(
                OUTPUT,
                split,
                label
            ),
            exist_ok=True
        )


# --------------------------------------------------
# Copy original V1 dataset
# --------------------------------------------------

print("Copying original validator dataset...")

for split in ["train", "validation", "test"]:

    for label in ["leaf", "non_leaf"]:

        source_folder = os.path.join(
            SOURCE,
            split,
            label
        )

        output_folder = os.path.join(
            OUTPUT,
            split,
            label
        )

        for filename in os.listdir(source_folder):

            shutil.copy(
                os.path.join(
                    source_folder,
                    filename
                ),
                os.path.join(
                    output_folder,
                    filename
                )
            )


# --------------------------------------------------
# Collect clean real-world negatives
# --------------------------------------------------

print("\nCollecting clean real-world negatives...")

all_real = []

for category in sorted(os.listdir(REAL_NEGATIVES)):

    category_path = os.path.join(
        REAL_NEGATIVES,
        category
    )

    if not os.path.isdir(category_path):
        continue

    images = []

    for filename in os.listdir(category_path):

        if not filename.lower().endswith(
            (".jpg", ".jpeg", ".png", ".webp")
        ):
            continue

        if (category, filename) in EXCLUDE:
            continue

        images.append(filename)

    random.shuffle(images)

    print(
        f"{category:20s}: "
        f"{len(images)} clean images"
    )

    all_real.append(
        (category, images)
    )


# --------------------------------------------------
# Split each category
# --------------------------------------------------

train_count = 0
validation_count = 0
test_count = 0

print("\nSplitting real hard negatives...")
print("=" * 60)

for category, images in all_real:

    total = len(images)

    train_end = int(total * 0.60)
    validation_end = int(total * 0.80)

    train_images = images[:train_end]
    validation_images = images[
        train_end:validation_end
    ]
    test_images = images[
        validation_end:
    ]

    print(
        f"{category:20s} | "
        f"Train={len(train_images):2d} | "
        f"Val={len(validation_images):2d} | "
        f"Test={len(test_images):2d}"
    )

    # -----------------------------
    # Training
    # -----------------------------

    for i, filename in enumerate(train_images):

        source = os.path.join(
            REAL_NEGATIVES,
            category,
            filename
        )

        destination = os.path.join(
            OUTPUT,
            "train",
            "non_leaf",
            f"real_{category}_{i:03d}.jpg"
        )

        shutil.copy(source, destination)

        train_count += 1


    # -----------------------------
    # Validation
    # -----------------------------

    for i, filename in enumerate(validation_images):

        source = os.path.join(
            REAL_NEGATIVES,
            category,
            filename
        )

        destination = os.path.join(
            OUTPUT,
            "validation",
            "non_leaf",
            f"real_{category}_{i:03d}.jpg"
        )

        shutil.copy(source, destination)

        validation_count += 1


    # -----------------------------
    # Final test
    # -----------------------------

    for i, filename in enumerate(test_images):

        source = os.path.join(
            REAL_NEGATIVES,
            category,
            filename
        )

        destination = os.path.join(
            OUTPUT,
            "test",
            "non_leaf",
            f"real_{category}_{i:03d}.jpg"
        )

        shutil.copy(source, destination)

        test_count += 1


# --------------------------------------------------
# Final summary
# --------------------------------------------------

print("\n" + "=" * 60)
print("V3 DATASET SUMMARY")
print("=" * 60)

print(
    f"Real hard negatives → "
    f"Train: {train_count}, "
    f"Validation: {validation_count}, "
    f"Test: {test_count}"
)

for split in ["train", "validation", "test"]:

    leaf_count = len(
        os.listdir(
            os.path.join(
                OUTPUT,
                split,
                "leaf"
            )
        )
    )

    non_leaf_count = len(
        os.listdir(
            os.path.join(
                OUTPUT,
                split,
                "non_leaf"
            )
        )
    )

    print(
        f"{split:10s} | "
        f"Leaf={leaf_count} | "
        f"Non-leaf={non_leaf_count} | "
        f"Total={leaf_count + non_leaf_count}"
    )

print("\nV3 dataset created successfully.")