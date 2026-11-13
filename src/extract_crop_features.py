
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

TRAIN_DIR = PROJECT_ROOT / "dataset/processed/full_39_class/train"

MODEL_PATH = (
    PROJECT_ROOT
    / "models/mobilenetv3_full39_plantdoc_adapted_best.pth"
)

OUTPUT_DIR = PROJECT_ROOT / "models/crop_support"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH = OUTPUT_DIR / "train_crop_features.npz"

BATCH_SIZE = 32

NUM_WORKERS = 0


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)

print(f"Device: {device}")


# ============================================================
# IMAGE TRANSFORM
# ============================================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# LOAD DATASET
# ============================================================

print("\nLoading PlantVillage training dataset...")

dataset = datasets.ImageFolder(
    root=str(TRAIN_DIR),
    transform=transform
)

print(f"Total classes: {len(dataset.classes)}")
print(f"Total images: {len(dataset)}")

# Verify checkpoint class ordering
checkpoint = torch.load(
    MODEL_PATH,
    map_location="cpu",
    weights_only=False
)

checkpoint_classes = checkpoint["class_names"]

if dataset.classes != checkpoint_classes:
    raise ValueError(
        "Dataset class order does not match model checkpoint!"
    )

# Exclude background images from crop references
selected_indices = []

for index, (_, class_index) in enumerate(dataset.samples):

    class_name = dataset.classes[class_index]

    if class_name == "Background_without_leaves":
        continue

    selected_indices.append(index)

print(
    f"Crop reference images: {len(selected_indices)}"
)

train_dataset = Subset(
    dataset,
    selected_indices
)

loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS
)


# ============================================================
# LOAD MOBILENETV3 LARGE
# ============================================================

print("\nLoading MobileNetV3 Large...")

model = models.mobilenet_v3_large(weights=None)

model.classifier[3] = torch.nn.Linear(
    model.classifier[3].in_features,
    len(checkpoint_classes)
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.to(device)
model.eval()

print("Model loaded successfully.")


# ============================================================
# FEATURE EXTRACTION
# ============================================================

all_features = []
all_crops = []
all_classes = []

print("\nExtracting 1280-dimensional features...\n")

with torch.inference_mode():

    for batch_index, (images, labels) in enumerate(loader):

        images = images.to(device)

        # MobileNetV3 feature extraction
        x = model.features(images)

        x = model.avgpool(x)

        x = torch.flatten(x, 1)

        # 960 -> 1280 dimensional representation
        x = model.classifier[0](x)
        x = model.classifier[1](x)

        # L2 normalization for cosine-based comparison
        x = F.normalize(x, p=2, dim=1)

        features = x.cpu().numpy()

        all_features.append(features)

        labels_np = labels.numpy()

        for label in labels_np:

            class_name = dataset.classes[label]

            crop_name = class_name.split("___")[0]

            all_crops.append(crop_name)
            all_classes.append(class_name)

        processed = min(
            (batch_index + 1) * BATCH_SIZE,
            len(train_dataset)
        )

        if (batch_index + 1) % 50 == 0:
            print(
                f"Processed {processed}/"
                f"{len(train_dataset)} images"
            )


# ============================================================
# SAVE FEATURES
# ============================================================

features_array = np.concatenate(
    all_features,
    axis=0
)

crops_array = np.array(all_crops)
classes_array = np.array(all_classes)

np.savez_compressed(
    OUTPUT_PATH,
    features=features_array,
    crops=crops_array,
    classes=classes_array
)


# ============================================================
# SUMMARY
# ============================================================

print("\n========== FEATURE EXTRACTION RESULTS ==========")

print(f"Total feature vectors: {len(features_array)}")

print(f"Feature dimensions: {features_array.shape[1]}")

print(f"Supported crop types: {len(set(crops_array))}")

for crop in sorted(set(crops_array)):

    count = int(np.sum(crops_array == crop))

    print(f"{crop}: {count}")

print(f"\nSaved features to:\n{OUTPUT_PATH}")

print("\nFeature extraction completed successfully.")
