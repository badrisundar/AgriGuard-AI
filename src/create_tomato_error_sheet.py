import pandas as pd
from PIL import Image, ImageDraw
from pathlib import Path

df = pd.read_csv("results/mobilenetv3_full39_val_clean_errors.csv")
df = df[df["actual_class"].str.startswith("Tomato___")]

cols = 3
thumb_w, thumb_h = 300, 300
label_h = 100
rows = (len(df) + cols - 1) // cols

sheet = Image.new(
    "RGB",
    (cols * thumb_w, rows * (thumb_h + label_h)),
    "white"
)
draw = ImageDraw.Draw(sheet)

for i, (_, row) in enumerate(df.iterrows()):
    path = Path(row["image_path"])

    with Image.open(path) as img:
        img = img.convert("RGB")
        img.thumbnail((thumb_w, thumb_h))

        x = (i % cols) * thumb_w
        y = (i // cols) * (thumb_h + label_h)

        sheet.paste(
            img,
            (x + (thumb_w - img.width) // 2,
             y + (thumb_h - img.height) // 2)
        )

    actual = row["actual_class"].replace("Tomato___", "")
    predicted = row["predicted_class"].replace("Tomato___", "")
    confidence = float(row["confidence"]) * 100

    draw.text((x + 5, y + thumb_h + 5),
              f"Actual: {actual[:40]}", fill="black")
    draw.text((x + 5, y + thumb_h + 30),
              f"Predicted: {predicted[:40]}", fill="black")
    draw.text((x + 5, y + thumb_h + 55),
              f"Confidence: {confidence:.2f}%", fill="black")

output = Path("results/error_inspection/tomato_errors_detailed.jpg")
output.parent.mkdir(parents=True, exist_ok=True)
sheet.save(output, quality=95)

print(f"Images included: {len(df)}")
print(f"Saved to: {output}")
