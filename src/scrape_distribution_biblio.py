import os
import time
import requests
import pandas as pd
from bs4 import BeautifulSoup

# ==========================
# FILE PATHS
# ==========================

INPUT_CSV = "data/raw/final_medicinal_plants.csv"
OUTPUT_CSV = "data/raw/distribution_bibliography.csv"
CHECKPOINT = "data/raw/distribution_checkpoint.txt"

SAVE_INTERVAL = 200

# ==========================
# REQUEST SESSION
# ==========================

session = requests.Session()

headers = {
    "User-Agent": "Mozilla/5.0",
    "Referer": "https://www.medicinalplants.in/"
}

# ==========================
# LOAD CSV
# ==========================

plants = pd.read_csv(INPUT_CSV)

# ==========================
# RESUME
# ==========================

start = 0

if os.path.exists(CHECKPOINT):
    with open(CHECKPOINT, "r") as f:
        start = int(f.read().strip())
    print(f"Resuming from plant {start+1}")
else:
    print("Starting from first plant")

# ==========================
# CREATE OUTPUT FILE
# ==========================

if not os.path.exists(OUTPUT_CSV):
    pd.DataFrame(columns=[
        "Botanical Name",
        "Title",
        "Author",
        "Detail URL"
    ]).to_csv(
        OUTPUT_CSV,
        index=False,
        encoding="utf-8-sig"
    )

buffer = []

# ==========================
# MAIN LOOP
# ==========================

for i in range(start, len(plants)):

    row = plants.iloc[i]

    plant = row["Botanical Name"]
    url = row["Detail URL"]

    print(f"[{i+1}/{len(plants)}] {plant}")

    success = False

    for attempt in range(3):

        try:

            response = session.get(
                url,
                headers=headers,
                timeout=20
            )

            if response.status_code != 200:
                time.sleep(2)
                continue

            soup = BeautifulSoup(response.text, "html.parser")

            # Find ALL longdistribution blocks
            books = soup.find_all(
                "div",
                id=lambda x: x and x.startswith("longdistribution_")
            )

            print("Books Found:", len(books))

            for book in books:

                # -------------------
                # TITLE
                # -------------------

                title = ""

                title_div = book.find("div", class_="booktitle")

                if title_div:
                    title = title_div.get_text(" ", strip=True)
                    title = (
                        title.replace("Title", "")
                             .replace("-", "")
                             .strip()
                    )

                # -------------------
                # AUTHOR
                # -------------------

                author = ""

                author_div = book.find("div", class_="bookauthor")

                if author_div:
                    author = author_div.get_text(" ", strip=True)
                    author = (
                        author.replace("Author", "")
                              .replace("-", "")
                              .strip()
                    )

                buffer.append({

                    "Botanical Name": plant,
                    "Title": title,
                    "Author": author,
                    "Detail URL": url

                })

            success = True
            break

        except Exception as e:
            print("Retry:", attempt + 1, e)
            time.sleep(2)

    if not success:
        print("Failed:", plant)

    # ==========================
    # SAVE EVERY 200 PLANTS
    # ==========================

    if (i + 1) % SAVE_INTERVAL == 0:

        if buffer:

            pd.DataFrame(buffer).to_csv(
                OUTPUT_CSV,
                mode="a",
                header=False,
                index=False,
                encoding="utf-8-sig"
            )

            buffer = []

        with open(CHECKPOINT, "w") as f:
            f.write(str(i + 1))

        print("\n==========================")
        print("Progress Saved")
        print("Completed:", i + 1)
        print("==========================\n")

# ==========================
# FINAL SAVE
# ==========================

if buffer:

    pd.DataFrame(buffer).to_csv(
        OUTPUT_CSV,
        mode="a",
        header=False,
        index=False,
        encoding="utf-8-sig"
    )

if os.path.exists(CHECKPOINT):
    os.remove(CHECKPOINT)

print("\n====================================")
print("SCRAPING COMPLETED")
print("Saved to:", OUTPUT_CSV)
print("====================================")