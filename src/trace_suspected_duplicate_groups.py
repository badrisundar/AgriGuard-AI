import pandas as pd
from pathlib import Path

source = Path(
    "results/duplicate_inspection/strong_duplicate_audit.csv"
)
output = Path(
    "results/duplicate_inspection/suspected_duplicate_groups.csv"
)

df = pd.read_csv(source)

# Strong same-class candidates from visual review
selected = [
    (417, 501),    # Cherry Powdery Mildew
    (592, 4476),   # Citrus Greening
    (790, 789),    # Background
    (4470, 3220),  # Tomato Yellow Leaf Curl Virus
    (3639, 1063),  # Tomato Yellow Leaf Curl Virus
    (520, 1063),   # Tomato Yellow Leaf Curl Virus
    (1030, 402),   # Tomato Yellow Leaf Curl Virus
]

def image_number(path):
    name = Path(path).stem
    digits = "".join(c for c in name if c.isdigit())
    return int(digits) if digits else -1

rows = []

for _, row in df.iterrows():
    n1 = image_number(row["image_1"])
    n2 = image_number(row["image_2"])

    if (n1, n2) in selected or (n2, n1) in selected:
        rows.append(row)

result = pd.DataFrame(rows)
result.to_csv(output, index=False)

print("Selected pairs found:", len(result))
print("\nSelected pairs:")
if not result.empty:
    print(result[[
        "split_1", "split_2",
        "class_1", "class_2",
        "dhash_distance", "ssim",
        "image_1", "image_2"
    ]].to_string(index=False))

print("\nSaved:", output)
