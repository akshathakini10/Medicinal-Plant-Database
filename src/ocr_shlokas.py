import os
import json
import easyocr
import warnings
warnings.filterwarnings("ignore")

reader = easyocr.Reader(['en', 'hi'])

image_folder = "data/shloka_images"

results = []

files = sorted(os.listdir(image_folder))

for i, file in enumerate(files, start=1):

    if not file.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    path = os.path.join(image_folder, file)

    try:
        text = reader.readtext(path, detail=0)

        results.append({
            "image_file": file,
            "ocr_text": "\n".join(text)
        })

        print(f"{i}/{len(files)} : {file}")

    except Exception as e:
        print("Failed:", file)

with open(
    "data/shlokas_ocr.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        results,
        f,
        ensure_ascii=False,
        indent=2
    )

print("\nDONE")
print("OCR Records:", len(results))