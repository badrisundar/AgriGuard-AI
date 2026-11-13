import os
import torch
from PIL import Image
from torchvision import models, transforms
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


MODEL_PATH = "models/leaf_validator_v3_best.pth"
TEST_DIR = "dataset/processed/leaf_validator_v3/test"


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
# Load test images
# -----------------------------
samples = []

for class_name, label in [
    ("leaf", 0),
    ("non_leaf", 1)
]:

    folder = os.path.join(
        TEST_DIR,
        class_name
    )

    for filename in os.listdir(folder):

        if not filename.lower().endswith(
            (".jpg", ".jpeg", ".png", ".webp")
        ):
            continue

        samples.append(
            (
                os.path.join(folder, filename),
                label,
                filename
            )
        )


print("Test images:", len(samples))


# -----------------------------
# Evaluation
# -----------------------------
true_labels = []
predicted_labels = []
leaf_probabilities = []

with torch.no_grad():

    for path, true_label, filename in samples:

        image = Image.open(path).convert("RGB")

        tensor = transform(image).unsqueeze(0).to(device)

        output = model(tensor)

        probabilities = torch.softmax(
            output,
            dim=1
        )

        leaf_probability = probabilities[0][0].item()

        prediction = torch.argmax(
            probabilities,
            dim=1
        ).item()

        true_labels.append(true_label)
        predicted_labels.append(prediction)
        leaf_probabilities.append(leaf_probability)


# -----------------------------
# Metrics
# -----------------------------
accuracy = accuracy_score(
    true_labels,
    predicted_labels
)

precision = precision_score(
    true_labels,
    predicted_labels,
    pos_label=0
)

recall = recall_score(
    true_labels,
    predicted_labels,
    pos_label=0
)

f1 = f1_score(
    true_labels,
    predicted_labels,
    pos_label=0
)

cm = confusion_matrix(
    true_labels,
    predicted_labels,
    labels=[0, 1]
)


print("\n" + "=" * 70)
print("V3 FINAL TEST RESULTS")
print("=" * 70)

print(f"Accuracy  : {accuracy * 100:.2f}%")
print(f"Precision : {precision * 100:.2f}%")
print(f"Recall    : {recall * 100:.2f}%")
print(f"F1 Score  : {f1 * 100:.2f}%")

print("\nConfusion Matrix")
print("(Rows = Actual, Columns = Predicted)")
print("                 LEAF  NON_LEAF")
print(f"Actual LEAF      {cm[0][0]:4d}     {cm[0][1]:4d}")
print(f"Actual NON_LEAF  {cm[1][0]:4d}     {cm[1][1]:4d}")


# -----------------------------
# Real hard-negative test
# -----------------------------
real_hard = []

non_leaf_folder = os.path.join(
    TEST_DIR,
    "non_leaf"
)

for filename in os.listdir(non_leaf_folder):

    if filename.startswith("real_"):

        real_hard.append(filename)


correct_real_rejections = 0

print("\n" + "=" * 70)
print("UNSEEN REAL-WORLD HARD-NEGATIVE TEST")
print("=" * 70)

for filename in sorted(real_hard):

    path = os.path.join(
        non_leaf_folder,
        filename
    )

    image = Image.open(path).convert("RGB")

    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():

        output = model(tensor)

        probabilities = torch.softmax(
            output,
            dim=1
        )

    leaf_probability = probabilities[0][0].item()

    prediction = torch.argmax(
        probabilities,
        dim=1
    ).item()

    if prediction == 1:
        correct_real_rejections += 1
        result = "REJECTED"
    else:
        result = "ACCEPTED AS LEAF"

    print(
        f"{filename:45s} "
        f"Leaf={leaf_probability:.4f} "
        f"{result}"
    )


real_total = len(real_hard)

real_rejection_rate = (
    correct_real_rejections / real_total * 100
    if real_total > 0 else 0
)

print("\nReal hard-negative results:")
print(
    f"Total              : {real_total}"
)

print(
    f"Correctly rejected : "
    f"{correct_real_rejections}"
)

print(
    f"Incorrectly accepted: "
    f"{real_total - correct_real_rejections}"
)

print(
    f"Rejection rate     : "
    f"{real_rejection_rate:.2f}%"
)