
from pathlib import Path
from collections import Counter, defaultdict
from PIL import Image
import csv
import sys

# ============================================================
# PROJECT IMPORT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.ml_model import predict_image

# ============================================================
# PATHS
# ============================================================

ROOT = Path.home() / "AgriGuard_Unknown_Crops"

MANIFEST = ROOT / "unknown_crop_evaluation_manifest.csv"

OUTPUT = ROOT / "baseline_evaluation_results.csv"

# ============================================================
# STORAGE
# ============================================================

results = []

status_counts = defaultdict(Counter)
predicted_crop_counts = defaultdict(Counter)

# ============================================================
# LOAD MANIFEST
# ============================================================

with open(MANIFEST, newline="") as f:
    rows = list(csv.DictReader(f))

print(f"\nEvaluating {len(rows)} images...\n")

# ============================================================
# BASELINE EVALUATION
# ============================================================

for index, row in enumerate(rows, start=1):

    actual_crop = row["crop"]
    image_path = Path(row["image_path"])

    try:

        with Image.open(image_path) as img:
            image = img.convert("RGB")
            result = predict_image(image)

        status = result.get("status", "error")

        # Capture both disease and healthy predictions
        predicted_class = (
            result.get("predicted_class")
            or result.get("disease")
            or ""
        )

        if "___" in predicted_class:
            predicted_crop = predicted_class.split("___")[0]
        else:
            predicted_crop = ""

        confidence = result.get("confidence")

        leaf_confidence = result.get("leaf_confidence")

        # Save complete prediction details
        results.append({
            "actual_crop": actual_crop,
            "source_class": row["source_class"],
            "image_path": str(image_path),
            "status": status,
            "predicted_class": predicted_class,
            "predicted_disease": result.get("disease"),
            "predicted_crop": predicted_crop,
            "confidence": confidence,
            "leaf_confidence": leaf_confidence,
            "message": result.get("message", "")
        })

        status_counts[actual_crop][status] += 1

        if predicted_crop:
            predicted_crop_counts[actual_crop][predicted_crop] += 1

    except Exception as e:

        print(f"ERROR: {image_path}: {e}")

        status_counts[actual_crop]["error"] += 1

    if index % 25 == 0 or index == len(rows):
        print(f"Processed {index}/{len(rows)}")

# ============================================================
# SAVE CSV RESULTS
# ============================================================

fieldnames = [
    "actual_crop",
    "source_class",
    "image_path",
    "status",
    "predicted_class",
    "predicted_disease",
    "predicted_crop",
    "confidence",
    "leaf_confidence",
    "message"
]

with open(OUTPUT, "w", newline="") as f:

    writer = csv.DictWriter(
        f,
        fieldnames=fieldnames
    )

    writer.writeheader()
    writer.writerows(results)

# ============================================================
# OVERALL RESULTS
# ============================================================

print("\n========== BASELINE RESULTS ==========")

total = len(results)

accepted = sum(
    1 for r in results
    if r["status"] in ("healthy", "prediction")
)

rejected = sum(
    1 for r in results
    if r["status"] in ("uncertain", "invalid_input")
)

errors = total - accepted - rejected

print(f"Total processed: {total}")
print(f"Accepted predictions: {accepted}")
print(f"Rejected predictions: {rejected}")
print(f"Errors: {errors}")

if total:

    print(
        f"False acceptance rate: "
        f"{accepted / total * 100:.2f}%"
    )

    print(
        f"Rejection rate: "
        f"{rejected / total * 100:.2f}%"
    )

# ============================================================
# CROP-WISE RESULTS
# ============================================================

for crop in ("Rice", "Banana", "Cotton", "Mango"):

    crop_rows = [
        r for r in results
        if r["actual_crop"] == crop
    ]

    crop_total = len(crop_rows)

    crop_accepted = sum(
        1 for r in crop_rows
        if r["status"] in ("healthy", "prediction")
    )

    crop_rejected = sum(
        1 for r in crop_rows
        if r["status"] in ("uncertain", "invalid_input")
    )

    print(f"\n========== {crop.upper()} ==========")

    print(f"Total images: {crop_total}")
    print(f"Accepted: {crop_accepted}")
    print(f"Rejected: {crop_rejected}")

    print("Statuses:")
    print(dict(status_counts[crop]))

    print("\nPredicted supported classes:")
    print(dict(predicted_crop_counts[crop]))

    if crop_total:
        print(
            f"False acceptance rate: "
            f"{crop_accepted / crop_total * 100:.2f}%"
        )

# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n======================================")
print("BASELINE EVALUATION COMPLETED")
print("======================================")

print(f"Detailed results saved to:\n{OUTPUT}")
