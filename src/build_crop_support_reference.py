
from pathlib import Path
import numpy as np
import json

# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

FEATURE_DIR = PROJECT_ROOT / "models/crop_support"

TRAIN_FILE = FEATURE_DIR / "train_crop_features.npz"
VAL_FILE = FEATURE_DIR / "validation_crop_features.npz"

REFERENCE_FILE = FEATURE_DIR / "crop_reference_profiles.npz"
THRESHOLD_FILE = FEATURE_DIR / "crop_support_threshold.json"

# Target: retain approximately 95% of known validation images
TARGET_REJECTION_PERCENTILE = 5


# ============================================================
# LOAD FEATURES
# ============================================================

print("Loading training features...")

train_data = np.load(TRAIN_FILE)

train_features = train_data["features"].astype(np.float32)
train_crops = train_data["crops"]

print("Training feature shape:", train_features.shape)

print("\nLoading validation features...")

val_data = np.load(VAL_FILE)

val_features = val_data["features"].astype(np.float32)
val_crops = val_data["crops"]

print("Validation feature shape:", val_features.shape)


# ============================================================
# BUILD CROP CENTROIDS
# ============================================================

print("\nBuilding supported crop reference profiles...")

crop_names = sorted(np.unique(train_crops))

centroids = []

for crop in crop_names:

    crop_features = train_features[train_crops == crop]

    centroid = np.mean(crop_features, axis=0)

    # Normalize centroid
    centroid = centroid / (
        np.linalg.norm(centroid) + 1e-12
    )

    centroids.append(centroid)

    print(
        f"{crop}: {len(crop_features)} training vectors"
    )

centroids = np.stack(centroids).astype(np.float32)

print("\nReference centroid shape:", centroids.shape)


# ============================================================
# CALCULATE VALIDATION SIMILARITIES
# ============================================================

print("\nCalculating validation similarities...")

# Features and centroids are L2 normalized.
# Their dot product gives cosine similarity.

similarities = val_features @ centroids.T

# Highest similarity to any supported crop
max_similarities = np.max(similarities, axis=1)

# Nearest supported crop
nearest_indices = np.argmax(similarities, axis=1)

nearest_crops = np.array(crop_names)[nearest_indices]


# ============================================================
# CALIBRATE GLOBAL THRESHOLD
# ============================================================

threshold = float(
    np.percentile(
        max_similarities,
        TARGET_REJECTION_PERCENTILE
    )
)

accepted = max_similarities >= threshold

overall_coverage = np.mean(accepted) * 100


# ============================================================
# SAVE REFERENCE PROFILES
# ============================================================

np.savez_compressed(
    REFERENCE_FILE,
    crop_names=np.array(crop_names),
    centroids=centroids
)


# ============================================================
# SAVE THRESHOLD
# ============================================================

metadata = {
    "method": "maximum_cosine_similarity_to_crop_centroids",
    "threshold": threshold,
    "target_rejection_percentile": TARGET_REJECTION_PERCENTILE,
    "validation_images": int(len(val_features)),
    "feature_dimensions": int(val_features.shape[1]),
    "supported_crops": crop_names,
    "target_validation_coverage_percent": 100 - TARGET_REJECTION_PERCENTILE,
    "observed_validation_coverage_percent": float(overall_coverage)
}

with open(THRESHOLD_FILE, "w") as f:
    json.dump(metadata, f, indent=4)


# ============================================================
# RESULTS
# ============================================================

print("\n========== CROP SUPPORT CALIBRATION ==========")

print(f"Supported crops: {len(crop_names)}")

print(f"Validation images: {len(val_features)}")

print(f"Feature dimensions: {val_features.shape[1]}")

print(f"\nCalibrated similarity threshold: {threshold:.6f}")

print(
    f"Validation coverage: "
    f"{overall_coverage:.2f}%"
)

print(
    f"Validation rejection: "
    f"{100 - overall_coverage:.2f}%"
)

print("\n========== CROP-WISE COVERAGE ==========")

for crop in crop_names:

    mask = val_crops == crop

    crop_coverage = np.mean(accepted[mask]) * 100

    print(
        f"{crop}: "
        f"{crop_coverage:.2f}% "
        f"({np.sum(accepted[mask])}/{np.sum(mask)})"
    )


# ============================================================
# SIMILARITY DISTRIBUTION
# ============================================================

print("\n========== SIMILARITY DISTRIBUTION ==========")

print(f"Minimum: {np.min(max_similarities):.6f}")
print(f"5th percentile: {np.percentile(max_similarities, 5):.6f}")
print(f"Median: {np.median(max_similarities):.6f}")
print(f"95th percentile: {np.percentile(max_similarities, 95):.6f}")
print(f"Maximum: {np.max(max_similarities):.6f}")


# ============================================================
# SAVED FILES
# ============================================================

print("\n========== SAVED FILES ==========")

print("Reference profiles:", REFERENCE_FILE)
print("Threshold metadata:", THRESHOLD_FILE)

print("\nCalibration completed successfully.")
