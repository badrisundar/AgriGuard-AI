import pandas as pd
from PIL import Image, ImageDraw
from pathlib import Path

df = pd.read_csv("results/split_duplicate_report.csv")

pairs = df[
    (df["type"] == "NEAR") &
    (df["class_1"] != df["class_2"]) &
    (df["distance"] == 4) &
    (~df["class_1"].str.contains("Background", case=False, na=False)) &
    (~df["class_2"].str.contains("Background", case=False, na=False))
].sort_values("distance").reset_index(drop=True)

output_dir = Path("results/duplicate_inspection/priority_pairs")
output_dir.mkdir(parents=True, exist_ok=True)

for pair_number in [26, 37]:
    row = pairs.iloc[pair_number - 1]

    print(f"\nPAIR {pair_number}")
    print("Distance:", row["distance"])
    print("Image 1:", row["image_1"])
    print("Class 1:", row["class_1"])
    print("Image 2:", row["image_2"])
    print("Class 2:", row["class_2"])

    canvas = Image.new("RGB", (1100, 650), "white")
    draw = ImageDraw.Draw(canvas)

    for j, (path, label) in enumerate([
        (row["image_1"], f"{row['split_1']}: {row['class_1']}"),
        (row["image_2"], f"{row['split_2']}: {row['class_2']}")
    ]):
        with Image.open(path) as img:
            img = img.convert("RGB")
            img.thumbnail((500, 540))
            x = 50 + j * 550
            canvas.paste(img, (x, 60))
            draw.text((x, 20), label, fill="black")

    output = output_dir / f"pair_{pair_number}_comparison.jpg"
    canvas.save(output, quality=98)
    print("Comparison saved:", output)

print("\nBoth comparisons generated.")
