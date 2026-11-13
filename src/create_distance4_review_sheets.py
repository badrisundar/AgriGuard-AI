import pandas as pd
from PIL import Image, ImageDraw
from pathlib import Path
import math

csv_path = Path("results/split_duplicate_report.csv")
output_dir = Path("results/duplicate_inspection/distance4_review")

df = pd.read_csv(csv_path)

pairs = df[
    (df["type"] == "NEAR") &
    (df["class_1"] != df["class_2"]) &
    (df["distance"] == 4)
].copy()

# Prioritize leaf-to-leaf comparisons.
leaf_pairs = pairs[
    (~pairs["class_1"].str.contains("Background", case=False, na=False)) &
    (~pairs["class_2"].str.contains("Background", case=False, na=False))
].reset_index(drop=True)

print("All cross-class distance-4 pairs:", len(pairs))
print("Leaf-to-leaf pairs:", len(leaf_pairs))
print("Background-involved pairs:", len(pairs) - len(leaf_pairs))

output_dir.mkdir(parents=True, exist_ok=True)

pairs_per_sheet = 3
thumb_w, thumb_h = 350, 280
row_h = 350

for start in range(0, len(leaf_pairs), pairs_per_sheet):
    batch = leaf_pairs.iloc[start:start + pairs_per_sheet]
    sheet = Image.new(
        "RGB",
        (thumb_w * 2, row_h * len(batch)),
        "white"
    )
    draw = ImageDraw.Draw(sheet)

    for i, row in enumerate(batch.itertuples()):
        y = i * row_h

        for j, (path, cls, split) in enumerate([
            (row.image_1, row.class_1, row.split_1),
            (row.image_2, row.class_2, row.split_2)
        ]):
            with Image.open(path) as img:
                img = img.convert("RGB")
                img.thumbnail((thumb_w - 20, thumb_h - 20))
                x = j * thumb_w
                sheet.paste(
                    img,
                    (x + (thumb_w - img.width) // 2, y + 5)
                )

            draw.text(
                (j * thumb_w + 5, y + thumb_h),
                f"{split}: {cls[:42]}",
                fill="black"
            )

        draw.text(
            (5, y + thumb_h + 25),
            f"Pair {start + i + 1} | dHash: {row.distance}",
            fill="red"
        )

    sheet_no = start // pairs_per_sheet + 1
    output = output_dir / f"distance4_leaf_review_{sheet_no}.jpg"
    sheet.save(output, quality=95)
    print("Saved:", output)

print("\nReview sheets generated.")
