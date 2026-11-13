import os
import torch
from PIL import Image
from torchvision import models, transforms

MODEL_PATH = "models/leaf_validator_v2_best.pth"
IMAGE_DIR = "dataset/real_hard_negatives/images"

device = torch.device(
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)

print("Device:", device)

# Load V2 model
model = models.mobilenet_v3_small(weights=None)

model.classifier[3] = torch.nn.Linear(
    model.classifier[3].in_features,
    2
)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device,
    weights_only=True
)

model.load_state_dict(checkpoint)
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

extensions = (".jpg", ".jpeg", ".png", ".webp")

results = []

for category in sorted(os.listdir(IMAGE_DIR)):

    category_path = os.path.join(
        IMAGE_DIR,
        category
    )

    if not os.path.isdir(category_path):
        continue

    for filename in sorted(os.listdir(category_path)):

        if not filename.lower().endswith(extensions):
            continue

        path = os.path.join(
            category_path,
            filename
        )

        try:
            image = Image.open(path).convert("RGB")
        except Exception as e:
            print("Skipping:", path, e)
            continue

        tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(tensor)
            probabilities = torch.softmax(output, dim=1)

        leaf_probability = probabilities[0][0].item()
        non_leaf_probability = probabilities[0][1].item()

        prediction = (
            "LEAF"
            if leaf_probability >= 0.5
            else "NON_LEAF"
        )

        results.append({
            "category": category,
            "filename": filename,
            "leaf_probability": leaf_probability,
            "non_leaf_probability": non_leaf_probability,
            "prediction": prediction
        })


# Sort by leaf probability
results.sort(
    key=lambda x: x["leaf_probability"],
    reverse=True
)

print("\n" + "=" * 85)
print("REAL HARD-NEGATIVE SCREENING")
print("=" * 85)

print(
    f"{'Category':20s}"
    f"{'Image':15s}"
    f"{'Leaf Prob':12s}"
    f"{'Prediction'}"
)

print("-" * 85)

for r in results:

    print(
        f"{r['category']:20s}"
        f"{r['filename']:15s}"
        f"{r['leaf_probability']:.4f}       "
        f"{r['prediction']}"
    )


# Summary
leaf_count = sum(
    1 for r in results
    if r["prediction"] == "LEAF"
)

non_leaf_count = len(results) - leaf_count

print("\n" + "=" * 85)
print("SUMMARY")
print("=" * 85)

print("Total images :", len(results))
print("Predicted LEAF :", leaf_count)
print("Predicted NON_LEAF :", non_leaf_count)

print("\nTop suspicious images:")
print("(These are non-leaf candidates that the model thinks look like leaves.)")
print("-" * 85)

for r in results[:20]:

    print(
        f"{r['category']}/{r['filename']} "
        f"-> Leaf probability = "
        f"{r['leaf_probability']:.4f}"
    )