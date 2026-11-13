
from pathlib import Path
import numpy as np
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[1]
FEATURE_PATH = ROOT / "models/crop_support/validation_crop_features.npz"
REFERENCE_PATH = ROOT / "models/crop_support/crop_reference_profiles.npz"
THRESHOLD_PATH = ROOT / "models/crop_support/crop_support_threshold.json"

import json

features_data = np.load(FEATURE_PATH, allow_pickle=True)
reference_data = np.load(REFERENCE_PATH, allow_pickle=True)

features = features_data["features"]
labels = features_data["crops"]
centroids = reference_data["centroids"]
crop_names = reference_data["crop_names"]

with open(THRESHOLD_PATH, "r") as f:
    threshold = json.load(f)["threshold"]

# Cosine similarity: vectors are already normalized
similarities = features @ centroids.T

best_indices = similarities.argmax(axis=1)
best_scores = similarities.max(axis=1)
predicted_crops = crop_names[best_indices]

results = defaultdict(list)

for actual, predicted, score in zip(
    labels, predicted_crops, best_scores
):
    results[str(actual)].append((str(predicted), float(score)))

print("\n========== CROP-WISE VALIDATION ANALYSIS ==========")

for crop in sorted(results):
    rows = results[crop]
    scores = np.array([row[1] for row in rows])
    accepted = scores >= threshold

    print(f"\n{crop}")
    print(f"  Images: {len(scores)}")
    print(f"  Accepted: {accepted.sum()}")
    print(f"  Rejected: {(~accepted).sum()}")
    print(f"  Minimum similarity: {scores.min():.4f}")
    print(f"  5th percentile: {np.percentile(scores, 5):.4f}")
    print(f"  Median similarity: {np.median(scores):.4f}")

    if crop == "Corn":
        print("  Corn images matched to:")
        matches = defaultdict(int)

        for predicted, score in rows:
            matches[predicted] += 1

        for name, count in sorted(
            matches.items(), key=lambda x: x[1], reverse=True
        ):
            print(f"    {name}: {count}")
