import os
import torch
from PIL import Image
from torchvision import models, transforms
from sklearn.metrics import confusion_matrix


MODEL_PATH = "models/leaf_validator_v2_best.pth"
VAL_DIR = "dataset/processed/leaf_validator_v2/validation"


device = torch.device(
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)

print("Device:", device)


# -----------------------------
# Load model
# -----------------------------
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


# -----------------------------
# Transform
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# -----------------------------
# Collect validation images
# -----------------------------
samples = []

for class_name, true_label in [
    ("leaf", 0),
    ("non_leaf", 1)
]:

    folder = os.path.join(VAL_DIR, class_name)

    for filename in os.listdir(folder):

        if not filename.lower().endswith(
            (".jpg", ".jpeg", ".png", ".webp")
        ):
            continue

        samples.append(
            (
                os.path.join(folder, filename),
                true_label
            )
        )


print("Validation images:", len(samples))


# -----------------------------
# Get probabilities
# -----------------------------
leaf_probs = []
true_labels = []

with torch.no_grad():

    for path, label in samples:

        image = Image.open(path).convert("RGB")

        tensor = transform(image).unsqueeze(0).to(device)

        output = model(tensor)

        probabilities = torch.softmax(
            output,
            dim=1
        )

        leaf_probability = probabilities[0][0].item()

        leaf_probs.append(leaf_probability)
        true_labels.append(label)


# -----------------------------
# Threshold analysis
# -----------------------------
print("\nThreshold Analysis")
print("=" * 90)

print(
    f"{'Threshold':<12}"
    f"{'Leaf Accepted':<16}"
    f"{'Non-Leaf Rejected':<20}"
    f"{'False Leaf':<15}"
    f"{'False Reject':<15}"
    f"{'Accuracy':<12}"
)

print("-" * 90)


for threshold in [
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
]:

    predictions = []

    for probability in leaf_probs:

        if probability >= threshold:
            predictions.append(0)   # LEAF
        else:
            predictions.append(1)   # NON_LEAF

    tn, fp, fn, tp = confusion_matrix(
        true_labels,
        predictions,
        labels=[0, 1]
    ).ravel()

    # Here:
    # true leaf = 0
    # true non-leaf = 1

    leaf_accepted = tp
    non_leaf_rejected = tn

    # Actually calculate correctly using labels:
    true_leaf = sum(
        1 for x in true_labels if x == 0
    )

    true_non_leaf = sum(
        1 for x in true_labels if x == 1
    )

    correctly_accepted_leaf = sum(
        1
        for y, p in zip(true_labels, predictions)
        if y == 0 and p == 0
    )

    correctly_rejected_non_leaf = sum(
        1
        for y, p in zip(true_labels, predictions)
        if y == 1 and p == 1
    )

    false_leaf = sum(
        1
        for y, p in zip(true_labels, predictions)
        if y == 1 and p == 0
    )

    false_reject = sum(
        1
        for y, p in zip(true_labels, predictions)
        if y == 0 and p == 1
    )

    accuracy = (
        (correctly_accepted_leaf + correctly_rejected_non_leaf)
        / len(true_labels)
    ) * 100

    print(
        f"{threshold:<12.2f}"
        f"{correctly_accepted_leaf:<16}"
        f"{correctly_rejected_non_leaf:<20}"
        f"{false_leaf:<15}"
        f"{false_reject:<15}"
        f"{accuracy:<12.2f}"
    )


# -----------------------------
# Hard-negative check
# -----------------------------
print("\nHard-negative graph check")
print("=" * 50)

print(
    "graph1.jpeg was previously classified as "
    "LEAF with probability 0.7016."
)

print(
    "This image must be rejected by the final "
    "validator operating threshold."
)