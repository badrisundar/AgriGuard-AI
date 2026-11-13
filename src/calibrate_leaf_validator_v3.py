import torch
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

MODEL_PATH = "models/leaf_validator_v3_best.pth"
VAL_DIR = "dataset/processed/leaf_validator_v3/validation"

device = torch.device(
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)

print(f"Device: {device}")

# --------------------------------------------------
# Transform
# --------------------------------------------------

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

dataset = ImageFolder(VAL_DIR, transform=transform)

loader = DataLoader(
    dataset,
    batch_size=16,
    shuffle=False,
    num_workers=0
)

print(f"Validation images: {len(dataset)}")
print(f"Classes: {dataset.classes}")

# --------------------------------------------------
# Model
# --------------------------------------------------

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

# --------------------------------------------------
# Get predictions
# --------------------------------------------------

all_probs = []
all_labels = []

with torch.no_grad():

    for images, labels in loader:

        images = images.to(device)

        outputs = model(images)

        probabilities = torch.softmax(outputs, dim=1)

        # Class 0 = leaf
        leaf_prob = probabilities[:, 0]

        all_probs.extend(leaf_prob.cpu().tolist())
        all_labels.extend(labels.tolist())

# --------------------------------------------------
# Threshold analysis
# --------------------------------------------------

print("\n" + "=" * 70)
print("V3 LEAF VALIDATOR THRESHOLD CALIBRATION")
print("=" * 70)

thresholds = [
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
    0.75,
    0.80,
    0.85,
    0.90,
    0.95
]

best_threshold = None
best_coverage = 0
best_accuracy = 0

for threshold in thresholds:

    accepted = 0
    correct = 0
    wrong = 0

    for prob, label in zip(all_probs, all_labels):

        # If leaf probability >= threshold,
        # accept image as leaf.
        if prob >= threshold:

            accepted += 1

            if label == 0:
                correct += 1
            else:
                wrong += 1

    coverage = accepted / len(all_labels)

    if accepted > 0:
        accepted_accuracy = correct / accepted
    else:
        accepted_accuracy = 0

    print(
        f"{threshold:.2f} → "
        f"Accepted={accepted:4d} | "
        f"Coverage={coverage * 100:6.2f}% | "
        f"Accepted Accuracy={accepted_accuracy * 100:6.2f}% | "
        f"Wrong={wrong}"
    )

    # Select threshold giving >=99% accepted accuracy
    # while maximizing coverage.
    if accepted_accuracy >= 0.99 and coverage > best_coverage:

        best_threshold = threshold
        best_coverage = coverage
        best_accuracy = accepted_accuracy

# --------------------------------------------------
# Final selection
# --------------------------------------------------

print("\n" + "=" * 70)
print("SELECTED OPERATING THRESHOLD")
print("=" * 70)

if best_threshold is not None:

    print(f"Threshold        : {best_threshold:.2f}")
    print(f"Accepted Accuracy: {best_accuracy * 100:.2f}%")
    print(f"Coverage         : {best_coverage * 100:.2f}%")

else:

    print("No threshold achieved 99% accepted accuracy.")