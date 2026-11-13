import pandas as pd
from PIL import Image, ImageDraw
from pathlib import Path

csv_path = Path("results/split_duplicate_report.csv")
output = Path("results/duplicate_inspection/cross_class_distance_le3.jpg")

df = pd.read_csv(csv_path)

near = df[df["type"] == "NEAR"].copy()
cross = near[near["class_1"] != near["class_2"]].copy()

cross = cross[cross["distance"] <= 3].sort_values("distance")

thumb_w = 240
thumb_h = 240
label_h = 85
pair_h = thumb_h + label_h
rows = len(cross)

sheet = Image.new("RGB", (thumb_w * 2, pair_h * rows), "white")
draw = ImageDraw.Draw(sheet)

for i, (_, row) in enumerate(cross.iterrows()):
    y = i * pair_h

    paths = [row["image_1"], row["image_2"]]
    classes = [row["class_1"], row["class_2"]]
    splits = [row["split_1"], row["split_2"]]

    for j in range(2):
        path = Path(paths[j])

        with Image.open(path) as img:
            img = img.convert("RGB")
            img.thumbnail((thumb_w - 10, thumb_h - 10))

            x = j * thumb_w
            sheet.paste(
                img,
                (
                    x + (thumb_w - img.width) // 2,
                    y + (thumb_h - img.height) // 2
                )
            )

        x = j * thumb_w + 5
        draw.text(
            (x, y + thumb_h + 3),
            f"{splits[j]} | {classes[j][:28]}",
            fill="black"
        )

    draw.text(
        (5, y + thumb_h + 23),
        f"Pair {i+1} | dHash distance: {row['distance']}",
        fill="red"
    )

output.parent.mkdir(parents=True, exist_ok=True)
sheet.save(output, quality=95)

print("Cross-class pairs inspected:", len(cross))
print("Saved:", output)
