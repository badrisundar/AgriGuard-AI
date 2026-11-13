import os
from PIL import Image, ImageDraw

IMAGE_DIR = "dataset/real_hard_negatives/images"

SUSPICIOUS = [
    ("soil_rocks", "13.jpg"),
    ("soil_rocks", "10.jpg"),
    ("buildings_rooms", "11.jpg"),
    ("people_animals", "07.jpg"),
    ("food_objects", "18.jpg"),
    ("soil_rocks", "02.jpg"),
    ("food_objects", "04.jpg"),
    ("buildings_rooms", "20.jpg"),
    ("people_animals", "05.jpg"),
    ("electronics", "10.jpg"),
    ("vehicles", "10.jpg"),
    ("vehicles", "09.jpg"),
    ("electronics", "03.jpg"),
    ("roads_outdoor", "05.jpg"),
    ("people_animals", "04.jpg"),
    ("people_animals", "09.jpg"),
    ("buildings_rooms", "14.jpg"),
    ("people_animals", "01.jpg"),
]

THUMB_W = 250
THUMB_H = 220
COLS = 3

ROWS = (len(SUSPICIOUS) + COLS - 1) // COLS

sheet = Image.new(
    "RGB",
    (COLS * THUMB_W, ROWS * THUMB_H),
    "white"
)

draw = ImageDraw.Draw(sheet)

for i, (category, filename) in enumerate(SUSPICIOUS):

    path = os.path.join(
        IMAGE_DIR,
        category,
        filename
    )

    image = Image.open(path).convert("RGB")
    image.thumbnail((230, 180))

    x = (i % COLS) * THUMB_W
    y = (i // COLS) * THUMB_H

    image_x = x + (THUMB_W - image.width) // 2
    image_y = y + 5

    sheet.paste(image, (image_x, image_y))

    label = f"{category}/{filename}"

    draw.text(
        (x + 5, y + 190),
        label,
        fill="black"
    )

output = "results/suspicious_hard_negatives.jpg"

sheet.save(output)

print("Created:", output)