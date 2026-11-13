
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "models/crop_support"

TRAIN_PATH = OUTPUT_DIR / "train_crop_features.npz"
VAL_PATH = OUTPUT_DIR / "validation_crop_features.npz"

REFERENCE_PATH = OUTPUT_DIR / "multi_centroid_reference.npz"
THRESHOLD_PATH = OUTPUT_DIR / "multi_centroid_threshold.json"

N_CLUSTERS = 2
RANDOM_STATE = 42
TARGET_REJECTION_PERCENTILE = 5

def normalize_rows(x):
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norms, 1e-12)

print("Loading training features...")
train_data = np.load(TRAIN_PATH, allow_pickle=True)

train_features = train_data["features"].astype(np.float32)
train_classes = train_data["classes"]

print("Training shape:", train_features.shape)

print("\nLoading validation features...")
val_data = np.load(VAL_PATH, allow_pickle=True)

val_features = normalize_rows(
    val_data["features"].astype(np.float32)
)
val_classes = val_data["classes"]
val_crops = val_data["crops"]

class_names = np.unique(train_classes)

all_centroids = []
centroid_labels = []

print("\nBuilding multiple reference centroids...")

for class_name in class_names:
    class_features = train_features[
        train_classes == class_name
    ]

    if len(class_features) < N_CLUSTERS:
        raise ValueError(
            f"Not enough samples for {class_name}"
        )

    # Normalize features before clustering
    class_features = normalize_rows(class_features)

    clustering = KMeans(
        n_clusters=N_CLUSTERS,
        random_state=RANDOM_STATE,
        n_init=10
    )

    clustering.fit(class_features)

    for center in clustering.cluster_centers_:
        center = center / max(np.linalg.norm(center), 1e-12)
        all_centroids.append(center)
        centroid_labels.append(class_name)

    print(
        f"{class_name}: "
        f"{len(class_features)} images -> "
        f"{N_CLUSTERS} centroids"
    )

centroids = np.asarray(all_centroids, dtype=np.float32)
centroid_labels = np.asarray(centroid_labels)

print("\nReference shape:", centroids.shape)

# Calculate validation similarity
print("\nCalculating validation similarities...")

similarities = val_features @ centroids.T
best_indices = similarities.argmax(axis=1)
best_scores = similarities.max(axis=1)

matched_classes = centroid_labels[best_indices]

threshold = float(
    np.percentile(
        best_scores,
        TARGET_REJECTION_PERCENTILE
    )
)

accepted = best_scores >= threshold

print("\n========== MULTI-CENTROID VALIDATION ==========")
print("Classes:", len(class_names))
print("Centroids per class:", N_CLUSTERS)
print("Total reference centroids:", len(centroids))
print("Validation images:", len(val_features))
print(f"Threshold: {threshold:.6f}")
print(f"Coverage: {accepted.mean() * 100:.2f}%")
print(f"Rejection: {(~accepted).mean() * 100:.2f}%")

# Crop-wise coverage
print("\n========== CROP-WISE COVERAGE ==========")

crop_rows = []

for crop in sorted(np.unique(val_crops)):
    mask = val_crops == crop
    coverage = accepted[mask].mean() * 100

    crop_rows.append({
        "crop": crop,
        "total": int(mask.sum()),
        "accepted": int(accepted[mask].sum()),
        "coverage_percent": coverage
    })

    print(
        f"{crop}: {coverage:.2f}% "
        f"({accepted[mask].sum()}/{mask.sum()})"
    )

# Class-wise coverage
print("\n========== CLASS-WISE COVERAGE ==========")

class_rows = []

for class_name in sorted(np.unique(val_classes)):
    mask = val_classes == class_name
    coverage = accepted[mask].mean() * 100

    class_rows.append({
        "class": class_name,
        "total": int(mask.sum()),
        "accepted": int(accepted[mask].sum()),
        "coverage_percent": coverage
    })

    print(
        f"{class_name}: {coverage:.2f}% "
        f"({accepted[mask].sum()}/{mask.sum()})"
    )

# Save reference profiles
np.savez_compressed(
    REFERENCE_PATH,
    centroids=centroids,
    centroid_labels=centroid_labels,
    class_names=class_names
)

metadata = {
    "method": "two_centroid_per_class_kmeans",
    "n_clusters_per_class": N_CLUSTERS,
    "random_state": RANDOM_STATE,
    "threshold": threshold,
    "target_rejection_percentile": TARGET_REJECTION_PERCENTILE,
    "validation_images": int(len(val_features)),
    "observed_validation_coverage": float(
        accepted.mean() * 100
    ),
    "supported_classes": class_names.tolist()
}

with open(THRESHOLD_PATH, "w") as f:
    json.dump(metadata, f, indent=4)

pd.DataFrame(crop_rows).to_csv(
    OUTPUT_DIR / "multi_centroid_crop_validation.csv",
    index=False
)

pd.DataFrame(class_rows).to_csv(
    OUTPUT_DIR / "multi_centroid_class_validation.csv",
    index=False
)

print("\n========== SAVED FILES ==========")
print(REFERENCE_PATH)
print(THRESHOLD_PATH)
print("\nMulti-centroid calibration completed.")
