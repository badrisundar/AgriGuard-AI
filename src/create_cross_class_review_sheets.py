import pandas as pd
from PIL import Image, ImageDraw
from pathlib import Path

df = pd.read_csv("results/split_duplicate_report.csv")
cross = df[
    (df["type"] == "NEAR") &
    (df["class_1"] != df["class_2"]) &
    (df["distance"] <= 3)
].sort_values("distance").reset_index(drop=True)

out_dir = Path("results/duplicate_inspection/review_sheets")
out_dir.mkdir(parents=True, exist_ok=True)

for sheet_no in range(4):
    batch = cross.iloc[sheet_no * 3:(sheet_no + 1) * 3]

    width, height = 1100, 390
    sheet = Image.new("RGB", (width, height * len(batch)), "white")
    draw = ImageDraw.Draw(sheet)

    for i, row in enumerate(batch.itertuples()):
        y = i * height

        for j, (path, cls, split) in enumerate([
            (row.image_1, row.class_1, row.split_1),
            (row.image_2, row.class_2, row.split_2)
        ]):
            with Image.open(path) as img:
                img = img.convert("RGB")
                img.thumbnail((350, 310))
                x = j * 550 + 90
                sheet.paste(img, (x, y + 5))

            draw.text(
                (j * 550 + 10, y + 325),
                f"{split}: {cls}",
                fill="black"
            )

        draw.text(
            (10, y + 355),
            f"Pair {sheet_no * 3 + i + 1} | Distance: {row.distance}",
            fill="red"
        )

    path = out_dir / f"cross_class_review_{sheet_no + 1}.jpg"
    sheet.save(path, quality=95)
    print("Saved:", path)
