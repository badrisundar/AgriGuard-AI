
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm

import torch
import torch.nn as nn
from torchvision import models, transforms

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

MANIFEST = (
    Path.home()
    / "AgriGuard_Unknown_Crops"
    / "unknown_crop_final_test.csv"
)

CHECKPOINT = (
    ROOT
    / "models/mobilenetv3_full39_plantdoc_adapted_best.pth"
)

OUTPUT = (
    Path.home()
    / "AgriGuard_Unknown_Crops"
    / "unknown_crop_final_test_features.npz"
)

DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)

BATCH_SIZE = 32

print("Device:", DEVICE)

df = pd.read_csv(MANIFEST)

checkpoint = torch.load(
    CHECKPOINT,
    map_location=DEVICE,
    weights_only=False
)

class_names = checkpoint["class_names"]

model = models.mobilenet_v3_large(weights=None)
model.classifier[3] = nn.Linear(
    model.classifier[3].in_features,
    len(class_names)
)

model.load_state_dict(checkpoint["model_state_dict"])
model.to(DEVICE)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

all_features = []
all_crops = []
all_paths = []
errors = []

def extract_feature(image):
    x = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        x = model.features(x)
        x = model.avgpool(x)
        x = torch.flatten(x, 1)
        x = model.classifier[0](x)
        x = model.classifier[1](x)

        x = torch.nn.functional.normalize(x, p=2, dim=1)

    return x.squeeze(0).cpu().numpy()

print("\nExtracting calibration features...")

for _, row in tqdm(df.iterrows(), total=len(df)):
    image_path = Path(row["image_path"])

    try:
        with Image.open(image_path) as img:
            image = img.convert("RGB")

        feature = extract_feature(image)

        all_features.append(feature)
        all_crops.append(row["crop"])
        all_paths.append(str(image_path))

    except (OSError, UnidentifiedImageError, ValueError) as e:
        errors.append((str(image_path), str(e)))

if not all_features:
    raise RuntimeError("No calibration features were extracted.")

features = np.asarray(all_features, dtype=np.float32)
crops = np.asarray(all_crops)
paths = np.asarray(all_paths)

np.savez_compressed(
    OUTPUT,
    features=features,
    crops=crops,
    paths=paths
)

print("\n========== CALIBRATION FEATURE RESULTS ==========")
print("Features extracted:", len(features))
print("Feature dimensions:", features.shape[1])
print("Errors:", len(errors))
print("Crop distribution:")

for crop in sorted(np.unique(crops)):
    print(f"{crop}: {np.sum(crops == crop)}")

print("\nSaved:", OUTPUT)

if errors:
    print("\nFirst errors:")
    for path, error in errors[:10]:
        print(path, error)
