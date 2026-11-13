import pandas as pd
from PIL import Image, ImageDraw
from pathlib import Path

csv_path = Path("results/mobilenetv3_full39_val_clean_errors.csv")
output_path = Path("results/error_inspection/corn_errors_detailed.jpg")

df = pd.read_csv(csv_path)

classes = [
    "Corn___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn___Northern_Leaf_Blight"
]

df = df[df["actual_class"].isin(classes)].copy()

thumb_w, thumb_h = 300, 300
label_h = 85
cols = 3
rows = (len(df) + cols - 1) // cols

sheet = Image.new(
    "RGB",
    (cols * thumb_w, rows * (thumb_h + label_h)),
    "white"
)
draw = ImageDraw.Draw(sheet)

for i, (_, row) in enumerate(df.iterrows()):
    image_path = Path(row["image_path"])

    with Image.open(image_path) as img:
        img = img.convert("RGB")
        img.thumbnail((thumb_w, thumb_h))
        x = (i % cols) * thumb_w
        y = (i // cols) * (thumb_h + label_h)

        sheet.paste(
            img,
            (x + (thumb_w - img.width) // 2,
             y + (thumb_h - img.height) // 2)
        )

    actual = row["actual_class"].replace("Corn___", "")
    predicted = row["predicted_class"].replace("Corn___", "")
    confidence = float(row["confidence"]) * 100

    draw.text((x + 5, y + thumb_h + 5),
              f"Actual: {actual[:38]}", fill="black")
    draw.text((x + 5, y + thumb_h + 27),
              f"Predicted: {predicted[:38]}", fill="black")
    draw.text((x + 5, y + thumb_h + 49),
              f"Confidence: {confidence:.2f}%", fill="black")

output_path.parent.mkdir(parents=True, exist_ok=True)
sheet.save(output_path, quality=95)

print(f"Images included: {len(df)}")
print(f"Saved to: {output_path}")
