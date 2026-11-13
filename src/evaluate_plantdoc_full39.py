import torch
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

from data_loader_full39 import DEVICE, train_dataset


# ============================================================
# Configuration
# ============================================================

MODEL_PATH = "models/mobilenetv3_full39_best.pth"

DATA_DIR = "dataset/processed/plantdoc_full39_eval/test"

BATCH_SIZE = 16
NUM_CLASSES = 39


# ============================================================
# PlantVillage class mapping
# ============================================================

plantvillage_classes = train_dataset.classes

plantvillage_to_index = {
    class_name: index
    for index, class_name in enumerate(plantvillage_classes)
}


# ============================================================
# PlantDoc folder -> PlantVillage class
# ============================================================

plantdoc_to_plantvillage = {
    "Apple___Apple_scab":
        "Apple___Apple_scab",

    "Apple___Cedar_apple_rust":
        "Apple___Cedar_apple_rust",

    "Pepper,_bell___Bacterial_spot":
        "Pepper,_bell___Bacterial_spot",

    "Corn___Cercospora_leaf_spot Gray_leaf_spot":
        "Corn___Cercospora_leaf_spot Gray_leaf_spot",

    "Corn___Common_rust":
        "Corn___Common_rust",

    "Potato___Early_blight":
        "Potato___Early_blight",

    "Potato___Late_blight":
        "Potato___Late_blight",

    "Squash___Powdery_mildew":
        "Squash___Powdery_mildew",

    "Tomato___Early_blight":
        "Tomato___Early_blight",

    "Tomato___Septoria_leaf_spot":
        "Tomato___Septoria_leaf_spot",

    "Tomato___Bacterial_spot":
        "Tomato___Bacterial_spot",

    "Tomato___Late_blight":
        "Tomato___Late_blight",

    "Tomato___Tomato_mosaic_virus":
        "Tomato___Tomato_mosaic_virus",

    "Tomato___Tomato_Yellow_Leaf_Curl_Virus":
        "Tomato___Tomato_Yellow_Leaf_Curl_Virus",

    "Tomato___Leaf_Mold":
        "Tomato___Leaf_Mold",

    "Grape___Black_rot":
        "Grape___Black_rot"
}


# ============================================================
# Transform
# ============================================================

eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# Load PlantDoc images
# ============================================================

print("=" * 70)
print("39-CLASS MOBILENETV3 ZERO-SHOT PLANTDOC EVALUATION")
print("=" * 70)

dataset = datasets.ImageFolder(
    DATA_DIR,
    transform=eval_transform
)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# ============================================================
# Create correct PlantVillage ground-truth labels
# ============================================================

correct_labels = []

for _, local_class_index in dataset.samples:

    local_class_name = dataset.classes[local_class_index]

    plantvillage_class = plantdoc_to_plantvillage[
        local_class_name
    ]

    plantvillage_index = plantvillage_to_index[
        plantvillage_class
    ]

    correct_labels.append(
        plantvillage_index
    )


# ============================================================
# Load model
# ============================================================

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
    weights_only=True
)

model = models.mobilenet_v3_large(
    weights=None
)

model.classifier[3] = torch.nn.Linear(
    model.classifier[3].in_features,
    NUM_CLASSES
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model = model.to(DEVICE)
model.eval()


print(f"Device: {DEVICE}")
print(f"Evaluation images: {len(dataset)}")
print(f"PlantDoc classes evaluated: {len(dataset.classes)}")
print(
    f"PlantVillage classes in model: "
    f"{NUM_CLASSES}"
)


# ============================================================
# Run inference
# ============================================================

predictions = []

with torch.no_grad():

    for images, _ in loader:

        images = images.to(DEVICE)

        outputs = model(images)

        predicted = torch.argmax(
            outputs,
            dim=1
        )

        predictions.extend(
            predicted.cpu().numpy()
        )


# ============================================================
# Overall metrics
# ============================================================

accuracy = accuracy_score(
    correct_labels,
    predictions
)

precision = precision_score(
    correct_labels,
    predictions,
    labels=[
        plantvillage_to_index[name]
        for name in dataset.classes
    ],
    average="weighted",
    zero_division=0
)

recall = recall_score(
    correct_labels,
    predictions,
    labels=[
        plantvillage_to_index[name]
        for name in dataset.classes
    ],
    average="weighted",
    zero_division=0
)

f1 = f1_score(
    correct_labels,
    predictions,
    labels=[
        plantvillage_to_index[name]
        for name in dataset.classes
    ],
    average="weighted",
    zero_division=0
)


# ============================================================
# Results
# ============================================================

print()
print("=" * 70)
print("CORRECTED PLANTDOC ZERO-SHOT RESULTS")
print("=" * 70)

print(f"Accuracy : {accuracy * 100:.2f}%")
print(f"Precision: {precision * 100:.2f}%")
print(f"Recall   : {recall * 100:.2f}%")
print(f"F1 Score : {f1 * 100:.2f}%")


# ============================================================
# Per-class report
# ============================================================

evaluated_indices = [
    plantvillage_to_index[name]
    for name in dataset.classes
]

evaluated_names = [
    name
    for name in dataset.classes
]

print()
print("=" * 70)
print("PER-CLASS RESULTS")
print("=" * 70)

report = classification_report(
    correct_labels,
    predictions,
    labels=evaluated_indices,
    target_names=evaluated_names,
    digits=4,
    zero_division=0
)

print(report)


# ============================================================
# Confusion matrix
# ============================================================

cm = confusion_matrix(
    correct_labels,
    predictions,
    labels=evaluated_indices
)

print()
print("=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)

print(cm)


# ============================================================
# Save results
# ============================================================

results_path = (
    "results/mobilenetv3_full39_plantdoc_zero_shot.txt"
)

with open(results_path, "w") as f:

    f.write(
        "39-CLASS MOBILENETV3 LARGE\n"
        "PLANTDOC ZERO-SHOT EVALUATION\n"
    )

    f.write("=" * 70 + "\n")

    f.write(
        f"Evaluation images: {len(dataset)}\n"
    )

    f.write(
        f"PlantDoc classes evaluated: "
        f"{len(dataset.classes)}\n"
    )

    f.write(
        f"Accuracy: {accuracy * 100:.4f}%\n"
    )

    f.write(
        f"Precision: {precision * 100:.4f}%\n"
    )

    f.write(
        f"Recall: {recall * 100:.4f}%\n"
    )

    f.write(
        f"F1 Score: {f1 * 100:.4f}%\n"
    )

    f.write("\n\nPER-CLASS RESULTS\n")
    f.write("=" * 70 + "\n")
    f.write(report)

    f.write("\n\nCONFUSION MATRIX\n")
    f.write("=" * 70 + "\n")
    f.write(str(cm))


print()
print("=" * 70)
print("RESULTS SAVED")
print("=" * 70)

print(results_path)