import pandas as pd
from PIL import Image, ImageDraw
from pathlib import Path

csv_path = Path(
    "results/duplicate_inspection/ssim_review/"
    "top20_cross_split_candidates.csv"
)
output_dir = Path(
    "results/duplicate_inspection/ssim_full_resolution"
)

df = pd.read_csv(csv_path)
output_dir.mkdir(parents=True, exist_ok=True)

# Candidate numbers from the top-20 report
selected = [1, 2, 3, 7, 13, 14, 19]

for number in selected:
    row = df.iloc[number - 1]

    panel_w = 750
    panel_h = 850
    image_h = 740

    canvas = Image.new(
        "RGB", (panel_w * 2, panel_h), "white"
    )
    draw = ImageDraw.Draw(canvas)

    for j, (path, split, cls) in enumerate([
        (row["image_1"], row["split_1"], row["class_1"]),
        (row["image_2"], row["split_2"], row["class_2"])
    ]):
        with Image.open(path) as img:
            img = img.convert("RGB")
            img.thumbnail((panel_w - 30, image_h - 20))

            x = j * panel_w
            canvas.paste(
                img,
                (x + (panel_w - img.width) // 2, 10)
            )

        draw.text(
            (j * panel_w + 10, image_h + 10),
            f"{split} | {cls}",
            fill="black"
        )
        draw.text(
            (j * panel_w + 10, image_h + 35),
            str(path),
            fill="black"
        )

    draw.text(
        (10, panel_h - 25),
        f"Candidate {number} | "
        f"SSIM={row['ssim']:.5f} | "
        f"dHash={row['dhash_distance']}",
        fill="red"
    )

    output = output_dir / f"candidate_{number}_full.jpg"
    canvas.save(output, quality=98)
    print("Saved:", output)

print("\nFull-resolution comparisons generated.")
