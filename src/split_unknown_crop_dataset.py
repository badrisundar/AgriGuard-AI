
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

MANIFEST = (
    Path.home()
    / "AgriGuard_Unknown_Crops"
    / "unknown_crop_evaluation_manifest.csv"
)

OUTPUT_DIR = MANIFEST.parent

df = pd.read_csv(MANIFEST)

print("Manifest columns:", df.columns.tolist())
print("Total images:", len(df))

# Find the crop column
crop_candidates = [
    col for col in df.columns
    if col.lower() in ["crop", "crop_name"]
]

if not crop_candidates:
    raise ValueError(
        "Crop column not found. Check manifest columns."
    )

crop_col = crop_candidates[0]

calibration_parts = []
test_parts = []

for crop, group in df.groupby(crop_col):
    group = group.sample(frac=1, random_state=42).reset_index(drop=True)

    if len(group) != 200:
        raise ValueError(
            f"{crop} has {len(group)} images instead of 200."
        )

    calibration_parts.append(group.iloc[:100])
    test_parts.append(group.iloc[100:])

calibration = pd.concat(
    calibration_parts,
    ignore_index=True
)

test = pd.concat(
    test_parts,
    ignore_index=True
)

calibration_path = OUTPUT_DIR / "unknown_crop_calibration.csv"
test_path = OUTPUT_DIR / "unknown_crop_final_test.csv"

calibration.to_csv(calibration_path, index=False)
test.to_csv(test_path, index=False)

print("\n========== SPLIT RESULTS ==========")
print("Calibration images:", len(calibration))
print("Final test images:", len(test))

print("\nCalibration distribution:")
print(calibration[crop_col].value_counts())

print("\nFinal test distribution:")
print(test[crop_col].value_counts())

print("\nSaved:")
print(calibration_path)
print(test_path)
