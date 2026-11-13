import os
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

CSV = "results/mobilenetv3_full39_val_clean_errors.csv"
OUT = "results/error_inspection"

os.makedirs(OUT, exist_ok=True)

df = pd.read_csv(CSV)

groups = {
    "corn": df[df["actual_class"].str.startswith("Corn___")],
    "tomato": df[df["actual_class"].str.startswith("Tomato___")],
    "peach": df[df["actual_class"].str.startswith("Peach___")]
}

font = ImageFont.load_default()

for group_name, group in groups.items():

    cell_w, cell_h = 600, 480
    cols = 2
    rows = (len(group) + cols - 1) // cols

    sheet = Image.new(
        "RGB",
        (cols * cell_w, rows * cell_h),
        "white"
    )

    draw = ImageDraw.Draw(sheet)

    for i, (_, item) in enumerate(group.iterrows()):

        image = Image.open(item["image_path"]).convert("RGB")
        image.thumbnail((560, 390))

        x = (i % cols) * cell_w
        y = (i // cols) * cell_h

        image_x = x + (cell_w - image.width) // 2
        sheet.paste(image, (image_x, y + 5))

        actual = item["actual_class"].replace("___", " / ")
        predicted = item["predicted_class"].replace("___", " / ")

        draw.text(
            (x + 10, y + 405),
            f"Actual: {actual[:65]}",
            fill="black",
            font=font
        )

        draw.text(
            (x + 10, y + 425),
            f"Predicted: {predicted[:65]}",
            fill="red",
            font=font
        )

        draw.text(
            (x + 10, y + 445),
            f"Confidence: {item['confidence'] * 100:.2f}%",
            fill="blue",
            font=font
        )

    path = os.path.join(
        OUT,
        f"{group_name}_error_inspection.jpg"
    )

    sheet.save(path, quality=95)
    print("Saved:", path)

print("\nAll inspection sheets created.")
