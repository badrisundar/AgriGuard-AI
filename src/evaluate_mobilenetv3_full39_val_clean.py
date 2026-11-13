
import os
import csv
import torch
import numpy as np

from torchvision import models, datasets
from torch.utils.data import DataLoader

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from data_loader_full39 import (
    DEVICE,
    BATCH_SIZE,
    eval_transform,
    train_dataset
)


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "models/mobilenetv3_full39_best.pth"

VAL_DIR = "dataset/processed/full_39_class/val_clean"

RESULT_DIR = "results"

NUM_CLASSES = 39

os.makedirs(RESULT_DIR, exist_ok=True)


# ============================================================
# LOAD CLEAN VALIDATION DATASET
# ============================================================

val_dataset = datasets.ImageFolder(
    VAL_DIR,
    transform=eval_transform
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

class_names = val_dataset.classes


# ============================================================
# DATASET VERIFICATION
# ============================================================

assert len(class_names) == NUM_CLASSES, (
    f"Expected {NUM_CLASSES} classes, found {len(class_names)}"
)

assert class_names == train_dataset.classes, (
    "Class ordering mismatch between training and validation datasets."
)

print("=" * 60)
print("CLEAN VALIDATION EVALUATION - MOBILENETV3 LARGE")
print("=" * 60)

print(f"Device: {DEVICE}")
print(f"Validation images: {len(val_dataset)}")
print(f"Number of classes: {len(class_names)}")


# ============================================================
# LOAD ORIGINAL MODEL CHECKPOINT
# ============================================================

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
    weights_only=True
)

model = models.mobilenet_v3_large(weights=None)

model.classifier[3] = torch.nn.Linear(
    model.classifier[3].in_features,
    NUM_CLASSES
)

model.load_state_dict(checkpoint["model_state_dict"])

model = model.to(DEVICE)
model.eval()


# ============================================================
# MODEL EVALUATION
# ============================================================

all_labels = []
all_predictions = []
all_confidences = []

with torch.no_grad():

    for images, labels in val_loader:

        images = images.to(DEVICE)

        outputs = model(images)

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        confidences, predictions = torch.max(
            probabilities,
            dim=1
        )

        all_labels.extend(labels.numpy())

        all_predictions.extend(
            predictions.cpu().numpy()
        )

        all_confidences.extend(
            confidences.cpu().numpy()
        )


# ============================================================
# CALCULATE PERFORMANCE METRICS
# ============================================================

accuracy = accuracy_score(
    all_labels,
    all_predictions
)

weighted_precision = precision_score(
    all_labels,
    all_predictions,
    average="weighted",
    zero_division=0
)

weighted_recall = recall_score(
    all_labels,
    all_predictions,
    average="weighted",
    zero_division=0
)

weighted_f1 = f1_score(
    all_labels,
    all_predictions,
    average="weighted",
    zero_division=0
)

macro_precision = precision_score(
    all_labels,
    all_predictions,
    average="macro",
    zero_division=0
)

macro_recall = recall_score(
    all_labels,
    all_predictions,
    average="macro",
    zero_division=0
)

macro_f1 = f1_score(
    all_labels,
    all_predictions,
    average="macro",
    zero_division=0
)


# ============================================================
# CLASSIFICATION REPORT AND CONFUSION MATRIX
# ============================================================

report = classification_report(
    all_labels,
    all_predictions,
    labels=list(range(NUM_CLASSES)),
    target_names=class_names,
    zero_division=0
)

matrix = confusion_matrix(
    all_labels,
    all_predictions,
    labels=list(range(NUM_CLASSES))
)


# ============================================================
# PRINT RESULTS
# ============================================================

correct = sum(
    np.array(all_labels) == np.array(all_predictions)
)

incorrect = len(all_labels) - correct

print("\nOVERALL RESULTS")
print("-" * 60)

print(f"Correct predictions: {correct}")
print(f"Incorrect predictions: {incorrect}")

print(
    f"Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)"
)

print(f"Weighted Precision: {weighted_precision:.4f}")
print(f"Weighted Recall: {weighted_recall:.4f}")
print(f"Weighted F1-score: {weighted_f1:.4f}")


print("\nMACRO METRICS")
print("-" * 60)

print(f"Macro Precision: {macro_precision:.4f}")
print(f"Macro Recall: {macro_recall:.4f}")
print(f"Macro F1-score: {macro_f1:.4f}")


print("\nCLASSIFICATION REPORT")
print("-" * 60)

print(report)


# ============================================================
# SAVE TEXT RESULTS
# ============================================================

text_path = os.path.join(
    RESULT_DIR,
    "mobilenetv3_full39_val_clean_results.txt"
)

with open(text_path, "w") as file:

    file.write(
        "MOBILENETV3 LARGE - CLEAN VALIDATION RESULTS\n"
    )

    file.write("=" * 60 + "\n")

    file.write(f"Validation images: {len(val_dataset)}\n")
    file.write(f"Number of classes: {NUM_CLASSES}\n")

    file.write(f"Correct predictions: {correct}\n")
    file.write(f"Incorrect predictions: {incorrect}\n")

    file.write(
        f"Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)\n"
    )

    file.write(
        f"Weighted Precision: {weighted_precision:.4f}\n"
    )

    file.write(
        f"Weighted Recall: {weighted_recall:.4f}\n"
    )

    file.write(
        f"Weighted F1-score: {weighted_f1:.4f}\n"
    )

    file.write(
        f"Macro Precision: {macro_precision:.4f}\n"
    )

    file.write(
        f"Macro Recall: {macro_recall:.4f}\n"
    )

    file.write(
        f"Macro F1-score: {macro_f1:.4f}\n\n"
    )

    file.write("CLASSIFICATION REPORT\n")
    file.write(report)

    file.write("\nCONFUSION MATRIX\n")
    file.write(np.array2string(matrix))


# ============================================================
# SAVE INDIVIDUAL PREDICTIONS
# ============================================================

predictions_path = os.path.join(
    RESULT_DIR,
    "mobilenetv3_full39_val_clean_predictions.csv"
)

with open(
    predictions_path,
    "w",
    newline=""
) as file:

    writer = csv.writer(file)

    writer.writerow([
        "image_path",
        "actual_class",
        "predicted_class",
        "confidence"
    ])

    for index, (actual, predicted, confidence) in enumerate(
        zip(
            all_labels,
            all_predictions,
            all_confidences
        )
    ):

        image_path = val_dataset.samples[index][0]

        writer.writerow([
            image_path,
            class_names[actual],
            class_names[predicted],
            f"{confidence:.6f}"
        ])


# ============================================================
# SAVE CONFUSION MATRIX
# ============================================================

matrix_path = os.path.join(
    RESULT_DIR,
    "mobilenetv3_full39_val_clean_confusion_matrix.csv"
)

with open(
    matrix_path,
    "w",
    newline=""
) as file:

    writer = csv.writer(file)

    writer.writerow(
        ["Actual / Predicted"] + class_names
    )

    for index, row in enumerate(matrix):

        writer.writerow(
            [class_names[index]] + row.tolist()
        )


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\nSAVED RESULTS")
print("-" * 60)

print(text_path)
print(predictions_path)
print(matrix_path)

print("\nCLEAN VALIDATION EVALUATION COMPLETED")
print("=" * 60)
