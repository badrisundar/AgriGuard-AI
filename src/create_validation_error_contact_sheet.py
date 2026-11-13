import os
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

CSV_PATH = "results/mobilenetv3_full39_val_clean_errors.csv"
OUTPUT_PATH = "results/validation_error_contact_sheet.jpg"

df = pd.read_csv(CSV_PATH)

COLS = 4
CELL_W = 380
CELL_H = 330
THUMB_W = 360
THUMB_H = 245

rows = (len(df) + COLS - 1) // COLS

sheet = Image.new(
    "RGB",
    (COLS * CELL_W, rows * CELL_H),
    "white"
)

draw = ImageDraw.Draw(sheet)

try:
    font = ImageFont.truetype(
        "/System/Library/Fonts/Helvetica.ttc", 15
    )
except:
    font = ImageFont.load_default()

for index, row in df.iterrows():

    image_path = row["image_path"]

    image = Image.open(image_path).convert("RGB")
    image.thumbnail((THUMB_W, THUMB_H))

    col = index % COLS
    row_num = index // COLS

    x = col * CELL_W
    y = row_num * CELL_H

    image_x = x + (CELL_W - image.width) // 2

    sheet.paste(image, (image_x, y + 5))

    actual = str(row["actual_class"])
    predicted = str(row["predicted_class"])
    confidence = float(row["confidence"]) * 100

    actual = actual.replace("___", " / ")
    predicted = predicted.replace("___", " / ")

    draw.text(
        (x + 8, y + 258),
        f"Actual: {actual[:43]}",
        fill="black",
        font=font
    )

    draw.text(
        (x + 8, y + 280),
        f"Predicted: {predicted[:43]}",
        fill="red",
        font=font
    )

    draw.text(
        (x + 8, y + 302),
        f"Confidence: {confidence:.2f}%",
        fill="blue",
        font=font
    )

sheet.save(OUTPUT_PATH, quality=95)

print("Contact sheet created:", OUTPUT_PATH)
print("Total images:", len(df))
