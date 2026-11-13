
from pathlib import Path
import numpy as np
import json

ROOT = Path(__file__).resolve().parents[1]
UNKNOWN_DIR = Path.home() / "AgriGuard_Unknown_Crops"

FEATURE_PATH = (
    UNKNOWN_DIR / "unknown_crop_final_test_features.npz"
)

REFERENCE_PATH = (
    ROOT / "models/crop_support/class_reference_profiles.npz"
)

THRESHOLD_PATH = (
    ROOT / "models/crop_support/class_support_threshold.json"
)

unknown_data = np.load(FEATURE_PATH, allow_pickle=True)
reference_data = np.load(REFERENCE_PATH, allow_pickle=True)

features = unknown_data["features"]
crops = unknown_data["crops"]
centroids = reference_data["centroids"]

with open(THRESHOLD_PATH, "r") as f:
    metadata = json.load(f)

threshold = float(metadata["threshold"])

# Calculate maximum cosine similarity
scores = (features @ centroids.T).max(axis=1)

accepted = scores >= threshold
rejected = ~accepted

print("\n========== FINAL UNKNOWN CROP EVALUATION ==========")
print("Total images:", len(features))
print("Frozen threshold:", f"{threshold:.6f}")

print("\nOverall results:")
print(f"Accepted as potentially supported: {accepted.sum()}")
print(f"Rejected as potentially unsupported: {rejected.sum()}")
print(f"False acceptance rate: {accepted.mean() * 100:.2f}%")
print(f"Unknown crop rejection rate: {rejected.mean() * 100:.2f}%")

print("\n========== CROP-WISE RESULTS ==========")

for crop in sorted(np.unique(crops)):
    mask = crops == crop
    crop_scores = scores[mask]
    crop_accepted = accepted[mask]

    print(f"\n{crop}")
    print("Total:", len(crop_scores))
    print("Accepted:", crop_accepted.sum())
    print("Rejected:", (~crop_accepted).sum())
    print(
        "False acceptance rate:",
        f"{crop_accepted.mean() * 100:.2f}%"
    )
    print(
        "Rejection rate:",
        f"{(~crop_accepted).mean() * 100:.2f}%"
    )
    print(
        "Mean similarity:",
        f"{crop_scores.mean():.4f}"
    )

print("\nEvaluation completed.")
print("No threshold was modified.")
