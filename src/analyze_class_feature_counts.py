
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

PATH = ROOT / "models/crop_support/train_crop_features.npz"

data = np.load(PATH, allow_pickle=True)

classes = data["classes"]

counts = (
    pd.Series(classes)
    .value_counts()
    .sort_values()
)

report = pd.DataFrame({
    "class_name": counts.index,
    "training_images": counts.values
})

print("\n========== TRAINING SAMPLES PER CLASS ==========")
print(report.to_string(index=False))

print("\nTotal classes:", len(report))
print("Total training images:", len(classes))
print("Minimum samples:", counts.min())
print("Maximum samples:", counts.max())
print("Median samples:", counts.median())
