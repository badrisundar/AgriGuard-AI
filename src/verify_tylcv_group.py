from pathlib import Path
from PIL import Image
import hashlib

files = [
    Path("dataset/processed/full_39_class/train/Tomato___Tomato_Yellow_Leaf_Curl_Virus/image (3639).JPG"),
    Path("dataset/processed/full_39_class/val/Tomato___Tomato_Yellow_Leaf_Curl_Virus/image (520).JPG"),
    Path("dataset/processed/full_39_class/test/Tomato___Tomato_Yellow_Leaf_Curl_Virus/image (1063).JPG"),
]

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

for path in files:
    if not path.exists():
        print("MISSING:", path)
        continue

    with Image.open(path) as img:
        print("\nFile:", path)
        print("Dimensions:", img.size)
        print("Format:", img.format)
        print("SHA-256:", sha256(path))
