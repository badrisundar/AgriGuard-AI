import os
import torch
from PIL import Image
from torchvision import models, transforms

MODEL_PATH = "models/leaf_validator_v2_best.pth"
TEST_DIR = "dataset/processed/leaf_validator_v2/test/non_leaf"

device = torch.device(
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)

print("Device:", device)

# Load model
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

# Preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Only evaluate hard_* images
files = [
    f for f in os.listdir(TEST_DIR)
    if f.startswith("hard_")
    and f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))
]

if not files:
    print("No hard-negative images found.")
    exit()

correct = 0
total = 0

print("\nV2 HARD-NEGATIVE TEST")
print("=" * 75)

with torch.no_grad():

    for filename in sorted(files):

        path = os.path.join(TEST_DIR, filename)

        image = Image.open(path).convert("RGB")

        image_tensor = transform(image).unsqueeze(0).to(device)

        outputs = model(image_tensor)

        probabilities = torch.softmax(outputs, dim=1)

        leaf_probability = probabilities[0][0].item()
        non_leaf_probability = probabilities[0][1].item()

        prediction = torch.argmax(
            probabilities,
            dim=1
        ).item()

        if prediction == 1:
            label = "NON_LEAF"
            correct += 1
        else:
            label = "LEAF"

        total += 1

        print(
            f"{filename:35s} "
            f"Leaf={leaf_probability:.4f}  "
            f"NonLeaf={non_leaf_probability:.4f}  "
            f"Prediction={label}"
        )


accuracy = (correct / total) * 100

print("\n" + "=" * 75)
print("RESULT")
print("=" * 75)

print(f"Total unseen hard negatives : {total}")
print(f"Correctly rejected          : {correct}")
print(f"Incorrectly accepted        : {total - correct}")
print(f"Rejection accuracy          : {accuracy:.2f}%")