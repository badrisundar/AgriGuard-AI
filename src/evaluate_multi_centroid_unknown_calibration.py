
from pathlib import Path
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
UNKNOWN_DIR = Path.home() / "AgriGuard_Unknown_Crops"

FEATURE_PATH = (
    UNKNOWN_DIR / "unknown_crop_calibration_features.npz"
)

REFERENCE_PATH = (
    ROOT / "models/crop_support/multi_centroid_reference.npz"
)

THRESHOLD_PATH = (
    ROOT / "models/crop_support/multi_centroid_threshold.json"
)

data = np.load(FEATURE_PATH, allow_pickle=True)
reference = np.load(REFERENCE_PATH, allow_pickle=True)

features = data["features"].astype(np.float32)
crops = data["crops"]
centroids = reference["centroids"].astype(np.float32)

with open(THRESHOLD_PATH, "r") as f:
    metadata = json.load(f)

threshold = float(metadata["threshold"])

# Maximum cosine similarity across all 76 centroids
scores = (features @ centroids.T).max(axis=1)

accepted = scores >= threshold
rejected = ~accepted

print("\n========== 76-CENTROID UNKNOWN CALIBRATION ==========")
print("Images:", len(features))
print("Reference centroids:", len(centroids))
print(f"Threshold: {threshold:.6f}")

print("\nOverall:")
print("Accepted:", int(accepted.sum()))
print("Rejected:", int(rejected.sum()))
print(f"False acceptance: {accepted.mean() * 100:.2f}%")
print(f"Unknown rejection: {rejected.mean() * 100:.2f}%")

print("\n========== CROP-WISE RESULTS ==========")

for crop in sorted(np.unique(crops)):
    mask = crops == crop
    crop_scores = scores[mask]
    crop_accepted = accepted[mask]

    print(f"\n{crop}")
    print("Accepted:", int(crop_accepted.sum()))
    print("Rejected:", int((~crop_accepted).sum()))
    print(f"False acceptance: {crop_accepted.mean() * 100:.2f}%")
    print(f"Rejection rate: {(~crop_accepted).mean() * 100:.2f}%")
    print(f"Mean similarity: {crop_scores.mean():.4f}")

print("\nNo model or threshold files modified.")
