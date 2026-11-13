import pandas as pd
import numpy as np
from PIL import Image
from pathlib import Path
from skimage.metrics import structural_similarity
import hashlib

CSV = Path("results/split_duplicate_report.csv")
OUT = Path("results/duplicate_inspection")
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(CSV)

pairs = df[
    (df["type"] == "NEAR") &
    (df["distance"] <= 4)
].copy()

def load_gray(path):
    with Image.open(path) as img:
        img = img.convert("L").resize((256, 256))
        return np.asarray(img, dtype=np.uint8)

def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

results = []

for index, row in pairs.iterrows():
    path1 = row["image_1"]
    path2 = row["image_2"]

    try:
        img1 = load_gray(path1)
        img2 = load_gray(path2)

        ssim = structural_similarity(img1, img2, data_range=255)

        results.append({
            "split_1": row["split_1"],
            "split_2": row["split_2"],
            "class_1": row["class_1"],
            "class_2": row["class_2"],
            "dhash_distance": row["distance"],
            "ssim": round(float(ssim), 5),
            "same_sha256": sha256(path1) == sha256(path2),
            "image_1": path1,
            "image_2": path2
        })

    except Exception as e:
        print("Error:", path1, path2, e)

result = pd.DataFrame(results)

result = result.sort_values(
    ["same_sha256", "ssim"],
    ascending=[False, False]
)

output = OUT / "strong_duplicate_audit.csv"
result.to_csv(output, index=False)

print("\nPairs analyzed:", len(result))
print("Exact SHA-256 matches:", result["same_sha256"].sum())

print("\nSSIM distribution:")
print(result["ssim"].describe())

print("\nHighest SSIM candidates:")
print(result.head(20)[[
    "class_1", "class_2",
    "dhash_distance", "ssim",
    "same_sha256"
]].to_string(index=False))

print("\nSaved:", output)
