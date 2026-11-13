from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# --------------------------------------------------
# 1. Project paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RESULTS_DIR = PROJECT_ROOT / "results"

RESULTS_FILE = RESULTS_DIR / "resnet50_test_results.txt"

OUTPUT_FILE = RESULTS_DIR / "resnet50_confusion_matrix.png"


# --------------------------------------------------
# 2. Class names
# --------------------------------------------------

class_names = [
    "Corn - Cercospora",
    "Corn - Common Rust",
    "Corn - Northern Leaf Blight",
    "Corn - Healthy",
    "Grape - Black Rot",
    "Grape - Esca",
    "Grape - Leaf Blight",
    "Grape - Healthy",
    "Tomato - Bacterial Spot",
    "Tomato - Early Blight",
    "Tomato - Late Blight",
    "Tomato - Leaf Mold",
    "Tomato - Healthy",
]


# --------------------------------------------------
# 3. Confusion matrix
# --------------------------------------------------

confusion_matrix = np.array([
    [78, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 179, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [10, 0, 139, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 175, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 177, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 207, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 162, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 64, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 320, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 149, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 284, 0, 2],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 144, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 240],
])


# --------------------------------------------------
# 4. Plot
# --------------------------------------------------

plt.figure(figsize=(14, 11))

sns.heatmap(
    confusion_matrix,
    annot=True,
    fmt="d",
    xticklabels=class_names,
    yticklabels=class_names,
    cmap="Blues"
)

plt.title("ResNet50 Confusion Matrix - Experiment 1")

plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")

plt.xticks(
    rotation=45,
    ha="right"
)

plt.yticks(
    rotation=0
)

plt.tight_layout()

plt.savefig(
    OUTPUT_FILE,
    dpi=300
)

plt.close()

print("Confusion matrix saved to:")
print(OUTPUT_FILE)