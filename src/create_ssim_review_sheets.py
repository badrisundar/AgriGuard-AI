import pandas as pd
from PIL import Image, ImageDraw
from pathlib import Path

csv_path = Path("results/duplicate_inspection/strong_duplicate_audit.csv")
out_dir = Path("results/duplicate_inspection/ssim_review")

df = pd.read_csv(csv_path)

# Only compare images belonging to different dataset splits.
df = df[df["split_1"] != df["split_2"]].copy()

df = df.sort_values("ssim", ascending=False).head(20).reset_index(drop=True)

print("Cross-split candidates:", len(df))

out_dir.mkdir(parents=True, exist_ok=True)

thumb_w = 400
thumb_h = 300
row_h = 390
pairs_per_sheet = 4

for start in range(0, len(df), pairs_per_sheet):
    batch = df.iloc[start:start + pairs_per_sheet]

    sheet = Image.new(
        "RGB",
        (thumb_w * 2, row_h * len(batch)),
        "white"
    )
    draw = ImageDraw.Draw(sheet)

    for i, row in enumerate(batch.itertuples()):
        y = i * row_h

        for j, (path, split, cls) in enumerate([
            (row.image_1, row.split_1, row.class_1),
            (row.image_2, row.split_2, row.class_2)
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
                (j * thumb_w + 8, y + thumb_h),
                f"{split}: {cls[:45]}",
                fill="black"
            )

        draw.text(
            (8, y + thumb_h + 28),
            f"Candidate {start + i + 1} | "
            f"SSIM: {row.ssim:.5f} | "
            f"dHash: {row.dhash_distance}",
            fill="red"
        )

    sheet_no = start // pairs_per_sheet + 1
    output = out_dir / f"ssim_review_{sheet_no}.jpg"
    sheet.save(output, quality=95)
    print("Saved:", output)

df.to_csv(out_dir / "top20_cross_split_candidates.csv", index=False)
print("\nTop 20 candidate report saved.")
