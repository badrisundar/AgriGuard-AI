from pathlib import Path
from PIL import Image, ImageDraw
from skimage.metrics import structural_similarity
import numpy as np

files = {
    "TRAIN - image 3639": Path(
        "dataset/processed/full_39_class/train/Tomato___Tomato_Yellow_Leaf_Curl_Virus/image (3639).JPG"
    ),
    "VAL - image 520": Path(
        "dataset/processed/full_39_class/val/Tomato___Tomato_Yellow_Leaf_Curl_Virus/image (520).JPG"
    ),
    "TEST - image 1063": Path(
        "dataset/processed/full_39_class/test/Tomato___Tomato_Yellow_Leaf_Curl_Virus/image (1063).JPG"
    ),
}

images = {}

for name, path in files.items():
    if not path.exists():
        raise FileNotFoundError(path)

    with Image.open(path) as img:
        images[name] = img.convert("RGB").resize((256, 256))

arrays = {
    name: np.array(img.convert("L"))
    for name, img in images.items()
}

names = list(images)

print("\nPAIRWISE SSIM")
for i in range(len(names)):
    for j in range(i + 1, len(names)):
        score = structural_similarity(
            arrays[names[i]],
            arrays[names[j]],
            data_range=255
        )
        print(f"{names[i]} vs {names[j]}: {score:.5f}")

canvas = Image.new("RGB", (900, 340), "white")
draw = ImageDraw.Draw(canvas)

for i, name in enumerate(names):
    x = 20 + i * 295
    canvas.paste(images[name], (x, 45))
    draw.text((x, 15), name, fill="black")

output = Path(
    "results/duplicate_inspection/tylcv_three_way_comparison.jpg"
)
output.parent.mkdir(parents=True, exist_ok=True)
canvas.save(output, quality=98)

print("\nSaved:", output)
