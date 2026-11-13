
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

UNKNOWN_DIR = (
    Path.home() / "AgriGuard_Unknown_Crops"
)

UNKNOWN_FEATURES = (
    UNKNOWN_DIR / "unknown_crop_calibration_features.npz"
)

VAL_FEATURES = (
    ROOT / "models/crop_support/validation_crop_features.npz"
)

REFERENCE = (
    ROOT / "models/crop_support/class_reference_profiles.npz"
)

OUTPUT = (
    ROOT / "models/crop_support/class_support_threshold.json"
)

unknown_data = np.load(UNKNOWN_FEATURES, allow_pickle=True)
val_data = np.load(VAL_FEATURES, allow_pickle=True)
reference_data = np.load(REFERENCE, allow_pickle=True)

unknown_features = unknown_data["features"]
unknown_crops = unknown_data["crops"]

val_features = val_data["features"]
val_crops = val_data["crops"]

centroids = reference_data["centroids"]

# Maximum cosine similarity to any supported class centroid
unknown_scores = (unknown_features @ centroids.T).max(axis=1)
val_scores = (val_features @ centroids.T).max(axis=1)

print("\n========== CURRENT THRESHOLD PERFORMANCE ==========")

with open(OUTPUT, "r") as f:
    metadata = json.load(f)

old_threshold = metadata["threshold"]

print(f"Current threshold: {old_threshold:.6f}")
print(
    "Supported validation coverage:",
    f"{(val_scores >= old_threshold).mean() * 100:.2f}%"
)
print(
    "Unsupported calibration acceptance:",
    f"{(unknown_scores >= old_threshold).mean() * 100:.2f}%"
)

print("\n========== UNSUPPORTED CROP SCORES ==========")

for crop in sorted(np.unique(unknown_crops)):
    scores = unknown_scores[unknown_crops == crop]
    accepted = scores >= old_threshold

    print(f"\n{crop}")
    print(f"Images: {len(scores)}")
    print(f"Accepted: {accepted.sum()}")
    print(f"Rejected: {(~accepted).sum()}")
    print(f"Mean similarity: {scores.mean():.4f}")
    print(f"Median similarity: {np.median(scores):.4f}")
    print(f"95th percentile: {np.percentile(scores, 95):.4f}")

# Explore candidate thresholds without changing the saved threshold
print("\n========== THRESHOLD TRADE-OFF ==========")

candidates = np.unique(
    np.concatenate([
        np.percentile(val_scores, [1, 2, 5, 10, 15, 20]),
        np.percentile(unknown_scores, [50, 60, 70, 80, 90, 95, 99])
    ])
)

print(
    f"{'Threshold':>12} "
    f"{'Supported coverage':>20} "
    f"{'Unknown rejection':>20}"
)

for threshold in sorted(candidates):
    supported_coverage = (val_scores >= threshold).mean() * 100
    unknown_rejection = (unknown_scores < threshold).mean() * 100

    print(
        f"{threshold:12.4f} "
        f"{supported_coverage:19.2f}% "
        f"{unknown_rejection:19.2f}%"
    )

print("\nNo threshold or model files were modified.")
