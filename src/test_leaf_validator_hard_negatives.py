import os
import torch
from PIL import Image
from torchvision import models, transforms

MODEL_PATH = "models/leaf_validator_mobilenetv3_small_best.pth"
TEST_DIR = "dataset/hard_negative_test/non_leaf"

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

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Get images
image_extensions = (".jpg", ".jpeg", ".png", ".webp")

files = [
    f for f in os.listdir(TEST_DIR)
    if f.lower().endswith(image_extensions)
]

if not files:
    print("No images found in:", TEST_DIR)
    exit()

correct_rejections = 0
total = 0

print("\nHard Negative Test")
print("=" * 70)

with torch.no_grad():

    for filename in sorted(files):

        path = os.path.join(TEST_DIR, filename)

        try:
            image = Image.open(path).convert("RGB")
        except Exception as e:
            print(f"Skipping {filename}: {e}")
            continue

        image_tensor = transform(image).unsqueeze(0).to(device)

        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)

        leaf_probability = probabilities[0][0].item()
        non_leaf_probability = probabilities[0][1].item()

        predicted_class = torch.argmax(
            probabilities, dim=1
        ).item()

        if predicted_class == 1:
            prediction = "NON_LEAF"
            correct_rejections += 1
        else:
            prediction = "LEAF"

        total += 1

        print(
            f"{filename:35s} "
            f"Leaf={leaf_probability:.4f}  "
            f"NonLeaf={non_leaf_probability:.4f}  "
            f"Prediction={prediction}"
        )

rejection_rate = (
    correct_rejections / total * 100
    if total > 0 else 0
)

print("\n" + "=" * 70)
print("HARD NEGATIVE TEST RESULTS")
print("=" * 70)

print(f"Total images                  : {total}")
print(f"Correctly rejected            : {correct_rejections}")
print(f"Incorrectly accepted as leaf : {total - correct_rejections}")
print(f"Rejection rate                : {rejection_rate:.2f}%")