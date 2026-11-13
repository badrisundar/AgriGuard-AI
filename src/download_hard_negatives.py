import os
import csv
import time
import requests

OUTPUT_DIR = "dataset/real_hard_negatives"
IMAGE_DIR = os.path.join(OUTPUT_DIR, "images")
METADATA_FILE = os.path.join(OUTPUT_DIR, "metadata.csv")

os.makedirs(IMAGE_DIR, exist_ok=True)

# Target: around 150 images
SEARCHES = {
    "screenshots_graphs": [
        "computer screenshot",
        "graph chart",
        "data visualization",
    ],
    "electronics": [
        "laptop",
        "smartphone",
        "computer",
        "electronic device",
    ],
    "vehicles": [
        "car",
        "truck",
        "tractor",
        "construction vehicle",
    ],
    "buildings_rooms": [
        "building",
        "house",
        "room interior",
        "office",
    ],
    "roads_outdoor": [
        "road",
        "street",
        "highway",
        "outdoor landscape",
    ],
    "soil_rocks": [
        "soil",
        "rocks",
        "stones",
        "dry ground",
    ],
    "food_objects": [
        "food",
        "kitchen object",
        "household object",
        "furniture",
    ],
    "people_animals": [
        "person",
        "people",
        "dog",
        "cat",
    ],
}

PER_SEARCH = 5

API_URL = "https://commons.wikimedia.org/w/api.php"

headers = {
    "User-Agent": "AgriGuard-AI-Research/1.0"
}


def search_images(query, limit=10):

    params = {
        "action": "query",
        "generator": "search",
        "gsrsearch": query,
        "gsrnamespace": 6,
        "gsrlimit": limit,
        "prop": "imageinfo",
        "iiprop": "url|mime|extmetadata",
        "iiurlwidth": 600,
        "format": "json"
    }

    response = requests.get(
        API_URL,
        params=params,
        headers=headers,
        timeout=30
    )

    response.raise_for_status()

    data = response.json()

    pages = data.get("query", {}).get("pages", {})

    results = []

    for page in pages.values():

        info = page.get("imageinfo", [{}])[0]

        mime = info.get("mime", "")

        if not mime.startswith("image/"):
            continue

        thumb_url = info.get("thumburl")

        if not thumb_url:
            thumb_url = info.get("url")

        metadata = info.get("extmetadata", {})

        artist = metadata.get(
            "Artist", {}
        ).get("value", "")

        license_name = metadata.get(
            "LicenseShortName", {}
        ).get("value", "")

        results.append({
            "title": page.get("title", ""),
            "url": info.get("url", ""),
            "download_url": thumb_url,
            "artist": artist,
            "license": license_name
        })

    return results


metadata_rows = []
downloaded = 0
used_urls = set()


for category, queries in SEARCHES.items():

    category_dir = os.path.join(
        IMAGE_DIR,
        category
    )

    os.makedirs(category_dir, exist_ok=True)

    category_count = 0

    print(f"\nCATEGORY: {category}")
    print("-" * 60)

    for query in queries:

        if category_count >= 20:
            break

        print("Searching:", query)

        try:
            results = search_images(
                query,
                limit=10
            )
        except Exception as e:
            print("Search failed:", e)
            continue

        for item in results:

            if category_count >= 20:
                break

            url = item["download_url"]

            if not url or url in used_urls:
                continue

            try:

                image_response = requests.get(
                    url,
                    headers=headers,
                    timeout=30
                )

                if image_response.status_code != 200:
                    continue

                content = image_response.content

                if len(content) < 5000:
                    continue

                filename = (
                    f"{category_count + 1:02d}.jpg"
                )

                filepath = os.path.join(
                    category_dir,
                    filename
                )

                with open(filepath, "wb") as f:
                    f.write(content)

                used_urls.add(url)

                metadata_rows.append({
                    "category": category,
                    "filename": filename,
                    "search_query": query,
                    "source_page": item["url"],
                    "download_url": url,
                    "artist": item["artist"],
                    "license": item["license"]
                })

                category_count += 1
                downloaded += 1

                print(
                    f"Downloaded {filename} "
                    f"({category_count}/20)"
                )

                time.sleep(0.5)

            except Exception as e:
                print(
                    "Download failed:",
                    e
                )

    print(
        f"Category total: {category_count}"
    )


# Save metadata
with open(
    METADATA_FILE,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    fieldnames = [
        "category",
        "filename",
        "search_query",
        "source_page",
        "download_url",
        "artist",
        "license"
    ]

    writer = csv.DictWriter(
        f,
        fieldnames=fieldnames
    )

    writer.writeheader()
    writer.writerows(metadata_rows)


print("\n" + "=" * 60)
print("DOWNLOAD COMPLETE")
print("=" * 60)

print("Total downloaded:", downloaded)

print(
    "Images:",
    IMAGE_DIR
)

print(
    "Metadata:",
    METADATA_FILE
)