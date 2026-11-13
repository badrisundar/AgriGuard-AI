
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

FEATURE_PATH = (
    ROOT / "models/crop_support/validation_crop_features.npz"
)

REFERENCE_PATH = (
    ROOT / "models/crop_support/class_reference_profiles.npz"
)

THRESHOLD_PATH = (
    ROOT / "models/crop_support/class_support_threshold.json"
)

OUTPUT_DIR = ROOT / "models/crop_support"

# Load validation data
data = np.load(FEATURE_PATH, allow_pickle=True)
reference = np.load(REFERENCE_PATH, allow_pickle=True)

features = data["features"].astype(np.float32)
crops = data["crops"]
classes = data["classes"]

centroids = reference["centroids"].astype(np.float32)
reference_classes = reference["class_names"]

with open(THRESHOLD_PATH, "r") as f:
    metadata = json.load(f)

threshold = float(metadata["threshold"])

# Calculate maximum cosine similarity
similarities = features @ centroids.T
best_indices = similarities.argmax(axis=1)
best_scores = similarities.max(axis=1)

matched_classes = reference_classes[best_indices]
accepted = best_scores >= threshold

# Save image-level results
results = pd.DataFrame({
    "actual_crop": crops,
    "actual_class": classes,
    "matched_reference_class": matched_classes,
    "similarity": best_scores,
    "status": np.where(
        accepted,
        "accepted",
        "rejected"
    )
})

image_output = OUTPUT_DIR / "supported_validation_results.csv"
results.to_csv(image_output, index=False)

print("\n========== OVERALL SUPPORTED VALIDATION ==========")
print("Total images:", len(results))
print("Accepted:", int(accepted.sum()))
print("Rejected:", int((~accepted).sum()))
print(f"Coverage: {accepted.mean() * 100:.2f}%")
print(f"Rejection rate: {(~accepted).mean() * 100:.2f}%")
print(f"Threshold: {threshold:.6f}")

# Crop-level report
print("\n========== CROP-WISE RESULTS ==========")

crop_report = (
    results.groupby("actual_crop")
    .agg(
        total=("status", "size"),
        accepted=("status", lambda x: (x == "accepted").sum()),
        rejected=("status", lambda x: (x == "rejected").sum()),
        mean_similarity=("similarity", "mean")
    )
)

crop_report["coverage_percent"] = (
    crop_report["accepted"] / crop_report["total"] * 100
)

crop_report["rejection_percent"] = (
    crop_report["rejected"] / crop_report["total"] * 100
)

print(crop_report.round(2).to_string())

crop_output = OUTPUT_DIR / "supported_crop_report.csv"
crop_report.to_csv(crop_output)

# Class-level report
print("\n========== CLASS-WISE RESULTS ==========")

class_report = (
    results.groupby("actual_class")
    .agg(
        total=("status", "size"),
        accepted=("status", lambda x: (x == "accepted").sum()),
        rejected=("status", lambda x: (x == "rejected").sum()),
        mean_similarity=("similarity", "mean")
    )
)

class_report["coverage_percent"] = (
    class_report["accepted"] / class_report["total"] * 100
)

class_report["rejection_percent"] = (
    class_report["rejected"] / class_report["total"] * 100
)

class_report = class_report.sort_values("coverage_percent")

print(class_report.round(2).to_string())

class_output = OUTPUT_DIR / "supported_class_report.csv"
class_report.to_csv(class_output)

print("\n========== SAVED REPORTS ==========")
print(image_output)
print(crop_output)
print(class_output)

print("\nEvaluation completed. No model or threshold was modified.")
