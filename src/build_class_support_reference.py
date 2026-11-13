
from pathlib import Path
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

TRAIN_PATH = ROOT / "models/crop_support/train_crop_features.npz"
VAL_PATH = ROOT / "models/crop_support/validation_crop_features.npz"

OUTPUT_DIR = ROOT / "models/crop_support"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

REFERENCE_PATH = OUTPUT_DIR / "class_reference_profiles.npz"
THRESHOLD_PATH = OUTPUT_DIR / "class_support_threshold.json"

TARGET_REJECTION_PERCENTILE = 5


def normalize_rows(features):
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    return features / np.maximum(norms, 1e-12)


print("Loading training features...")
train_data = np.load(TRAIN_PATH, allow_pickle=True)

train_features = train_data["features"]
train_classes = train_data["classes"]

print("Training feature shape:", train_features.shape)

print("\nLoading validation features...")
val_data = np.load(VAL_PATH, allow_pickle=True)

val_features = val_data["features"]
val_classes = val_data["classes"]

print("Validation feature shape:", val_features.shape)

# Ensure feature dimensions match
if train_features.shape[1] != val_features.shape[1]:
    raise ValueError("Training and validation feature dimensions do not match.")

# Build one centroid for each of the 38 classes
class_names = np.unique(train_classes)

centroids = []

print("\nBuilding class reference profiles...")

for class_name in class_names:
    class_vectors = train_features[train_classes == class_name]

    centroid = class_vectors.mean(axis=0)
    centroid = centroid / max(np.linalg.norm(centroid), 1e-12)

    centroids.append(centroid)

    print(f"{class_name}: {len(class_vectors)} training vectors")

centroids = np.asarray(centroids, dtype=np.float32)

print("\nReference centroid shape:", centroids.shape)

# Normalize validation features
val_features = normalize_rows(
    val_features.astype(np.float32)
)

# Calculate maximum cosine similarity to any class centroid
print("\nCalculating validation similarities...")

similarities = val_features @ centroids.T

best_indices = similarities.argmax(axis=1)
best_scores = similarities.max(axis=1)

predicted_classes = class_names[best_indices]

# Select threshold using validation data only
threshold = float(
    np.percentile(
        best_scores,
        TARGET_REJECTION_PERCENTILE
    )
)

accepted = best_scores >= threshold

overall_coverage = float(accepted.mean() * 100)

print("\n========== CLASS SUPPORT CALIBRATION ==========")
print("Supported classes:", len(class_names))
print("Validation images:", len(val_features))
print("Feature dimensions:", val_features.shape[1])

print(f"\nCalibrated threshold: {threshold:.6f}")
print(f"Validation coverage: {overall_coverage:.2f}%")
print(f"Validation rejection: {100 - overall_coverage:.2f}%")

# Crop-wise coverage
crop_names = val_data["crops"]

print("\n========== CROP-WISE COVERAGE ==========")

for crop in sorted(np.unique(crop_names)):
    mask = crop_names == crop

    crop_accepted = accepted[mask]

    print(
        f"{crop}: "
        f"{crop_accepted.mean() * 100:.2f}% "
        f"({crop_accepted.sum()}/{mask.sum()})"
    )

# Class-wise coverage
print("\n========== CLASS-WISE COVERAGE ==========")

for class_name in sorted(np.unique(val_classes)):
    mask = val_classes == class_name

    class_accepted = accepted[mask]

    print(
        f"{class_name}: "
        f"{class_accepted.mean() * 100:.2f}% "
        f"({class_accepted.sum()}/{mask.sum()})"
    )

print("\n========== SIMILARITY DISTRIBUTION ==========")
print(f"Minimum: {best_scores.min():.6f}")
print(f"5th percentile: {np.percentile(best_scores, 5):.6f}")
print(f"Median: {np.median(best_scores):.6f}")
print(f"95th percentile: {np.percentile(best_scores, 95):.6f}")
print(f"Maximum: {best_scores.max():.6f}")

# Save reference profiles
np.savez_compressed(
    REFERENCE_PATH,
    class_names=class_names,
    centroids=centroids
)

metadata = {
    "method": "class_level_centroid_cosine_similarity",
    "threshold": threshold,
    "target_rejection_percentile": TARGET_REJECTION_PERCENTILE,
    "validation_images": int(len(val_features)),
    "feature_dimensions": int(val_features.shape[1]),
    "supported_classes": class_names.tolist(),
    "observed_validation_coverage": overall_coverage
}

with open(THRESHOLD_PATH, "w") as f:
    json.dump(metadata, f, indent=4)

print("\n========== SAVED FILES ==========")
print("Reference profiles:", REFERENCE_PATH)
print("Threshold metadata:", THRESHOLD_PATH)

print("\nCalibration completed successfully.")
