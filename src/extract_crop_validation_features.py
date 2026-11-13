
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

PROJECT_ROOT = Path(__file__).resolve().parents[1]

VAL_DIR = PROJECT_ROOT / "dataset/processed/full_39_class/val"

MODEL_PATH = (
    PROJECT_ROOT
    / "models/mobilenetv3_full39_plantdoc_adapted_best.pth"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "models/crop_support/validation_crop_features.npz"
)

device = torch.device(
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

print(f"Device: {device}")
print("\nLoading validation dataset...")

dataset = datasets.ImageFolder(
    root=str(VAL_DIR),
    transform=transform
)

checkpoint = torch.load(
    MODEL_PATH,
    map_location="cpu",
    weights_only=False
)

if dataset.classes != checkpoint["class_names"]:
    raise ValueError("Checkpoint and dataset classes do not match.")

model = models.mobilenet_v3_large(weights=None)
model.classifier[3] = torch.nn.Linear(
    model.classifier[3].in_features,
    len(checkpoint["class_names"])
)

model.load_state_dict(checkpoint["model_state_dict"])
model.to(device)
model.eval()

selected_indices = [
    i for i, (_, label) in enumerate(dataset.samples)
    if dataset.classes[label] != "Background_without_leaves"
]

print(f"Validation images: {len(selected_indices)}")

loader = DataLoader(
    torch.utils.data.Subset(dataset, selected_indices),
    batch_size=32,
    shuffle=False,
    num_workers=0
)

features_list = []
crops_list = []
classes_list = []

print("\nExtracting validation features...")

with torch.inference_mode():
    processed = 0

    for images, labels in loader:
        images = images.to(device)

        x = model.features(images)
        x = model.avgpool(x)
        x = torch.flatten(x, 1)
        x = model.classifier[0](x)
        x = model.classifier[1](x)

        x = F.normalize(x, p=2, dim=1)

        features_list.append(x.cpu().numpy())

        for label in labels.numpy():
            class_name = dataset.classes[label]
            crops_list.append(class_name.split("___")[0])
            classes_list.append(class_name)

        processed += len(images)

        if processed % 500 < 32:
            print(f"Processed {processed}/{len(selected_indices)}")

features = np.concatenate(features_list, axis=0)

np.savez_compressed(
    OUTPUT_PATH,
    features=features,
    crops=np.array(crops_list),
    classes=np.array(classes_list)
)

print("\n========== VALIDATION FEATURE RESULTS ==========")
print("Feature vectors:", len(features))
print("Feature dimensions:", features.shape[1])
print("Crop types:", len(set(crops_list)))
print("Saved:", OUTPUT_PATH)
