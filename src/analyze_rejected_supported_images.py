
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

INPUT = (
    ROOT
    / "models/crop_support/supported_validation_results.csv"
)

OUTPUT_DIR = ROOT / "models/crop_support"

df = pd.read_csv(INPUT)

rejected = df[df["status"] == "rejected"].copy()

rejected = rejected.sort_values(
    "similarity",
    ascending=True
)

output = OUTPUT_DIR / "rejected_supported_images_analysis.csv"
rejected.to_csv(output, index=False)

print("\n========== REJECTED IMAGE ANALYSIS ==========")
print("Total rejected images:", len(rejected))

print("\nLowest 20 similarity samples:")
print(
    rejected[
        [
            "actual_crop",
            "actual_class",
            "matched_reference_class",
            "similarity"
        ]
    ].head(20).to_string(index=False)
)

print("\n========== REJECTIONS BY CLASS ==========")

summary = (
    rejected.groupby("actual_class")
    .agg(
        rejected_images=("similarity", "size"),
        mean_similarity=("similarity", "mean"),
        minimum_similarity=("similarity", "min"),
        maximum_similarity=("similarity", "max")
    )
    .sort_values("rejected_images", ascending=False)
)

print(summary.round(4).to_string())

print("\n========== MATCHED REFERENCE CLASSES ==========")

matches = (
    rejected.groupby(
        ["actual_class", "matched_reference_class"]
    )
    .size()
    .reset_index(name="count")
    .sort_values("count", ascending=False)
)

print(matches.head(30).to_string(index=False))

print("\nSaved report:", output)
