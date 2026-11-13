import os
import shutil
import random

SOURCE = "dataset/processed/leaf_validator"
HARD_NEGATIVES = "dataset/hard_negative_test/non_leaf"
OUTPUT = "dataset/processed/leaf_validator_v2"

SEED = 42
random.seed(SEED)

# Create output folders
folders = [
    "train/leaf",
    "train/non_leaf",
    "validation/leaf",
    "validation/non_leaf",
    "test/leaf",
    "test/non_leaf",
]

for folder in folders:
    os.makedirs(os.path.join(OUTPUT, folder), exist_ok=True)


# -----------------------------------------
# Copy existing LEAF images
# -----------------------------------------

for split in ["train", "validation", "test"]:

    source_folder = os.path.join(SOURCE, split, "leaf")
    output_folder = os.path.join(OUTPUT, split, "leaf")

    for filename in os.listdir(source_folder):
        shutil.copy(
            os.path.join(source_folder, filename),
            os.path.join(output_folder, filename)
        )


# -----------------------------------------
# Copy existing NON-LEAF images
# -----------------------------------------

for split in ["train", "validation", "test"]:

    source_folder = os.path.join(SOURCE, split, "non_leaf")
    output_folder = os.path.join(OUTPUT, split, "non_leaf")

    for filename in os.listdir(source_folder):
        shutil.copy(
            os.path.join(source_folder, filename),
            os.path.join(output_folder, filename)
        )


# -----------------------------------------
# Add hard negatives
# -----------------------------------------

images = [
    f for f in os.listdir(HARD_NEGATIVES)
    if f.lower().endswith(
        (".jpg", ".jpeg", ".png", ".webp")
    )
]

random.shuffle(images)

# 20 images → training
# 11 images → final unseen test
train_hard = images[:20]
test_hard = images[20:]

print("Hard negatives for training:", len(train_hard))
print("Hard negatives for final test:", len(test_hard))


# Copy 20 hard negatives to training
for filename in train_hard:

    shutil.copy(
        os.path.join(HARD_NEGATIVES, filename),
        os.path.join(
            OUTPUT,
            "train",
            "non_leaf",
            "hard_" + filename
        )
    )


# Copy 11 hard negatives to final test
for filename in test_hard:

    shutil.copy(
        os.path.join(HARD_NEGATIVES, filename),
        os.path.join(
            OUTPUT,
            "test",
            "non_leaf",
            "hard_" + filename
        )
    )


# -----------------------------------------
# Dataset summary
# -----------------------------------------

print("\nV2 Dataset Summary")
print("=" * 50)

for split in ["train", "validation", "test"]:

    leaf_count = len(
        os.listdir(
            os.path.join(OUTPUT, split, "leaf")
        )
    )

    non_leaf_count = len(
        os.listdir(
            os.path.join(OUTPUT, split, "non_leaf")
        )
    )

    print(
        f"{split:10s} | "
        f"leaf: {leaf_count:5d} | "
        f"non_leaf: {non_leaf_count:5d} | "
        f"total: {leaf_count + non_leaf_count:5d}"
    )


print("\nUnseen hard-negative test images:")
print("=" * 50)

for filename in sorted(test_hard):
    print(filename)