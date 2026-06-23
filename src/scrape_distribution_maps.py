import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import re

session = requests.Session()

headers = {
    "User-Agent": "Mozilla/5.0",
    "Referer": "https://www.medicinalplants.in/distributionmaps"
}

session.get("https://www.medicinalplants.in/distributionmaps", headers=headers)

all_maps = []

for start in range(0, 1101, 20):
    url = f"https://www.medicinalplants.in/mapsoperationdetails/{start}/20"
    print("Scraping:", url)

    response = session.get(url, headers=headers)
    soup = BeautifulSoup(response.text, "html.parser")

    # Find plant names from text using pattern between + signs
    text = soup.get_text(" ", strip=True)

    names = re.findall(r"\+\s*([^+]+?)(?=\s*\+| book|\s*\d+\s+Next| Showing)", text)

    # Clean unwanted values
    clean_names = []
    for name in names:
        name = name.strip()
        if name and len(name) > 3 and not name.startswith("Distribution Maps"):
            clean_names.append(name)

    images = []
    for img in soup.find_all("img"):
        src = img.get("src")
        if src and "geomaps" in src:
            images.append(src)

    print("Names:", len(clean_names), "Images:", len(images))

    for name, img_url in zip(clean_names, images):
        all_maps.append({
            "Plant Name": name,
            "Distribution Map URL": img_url
        })

    time.sleep(0.3)

df = pd.DataFrame(all_maps)
df.drop_duplicates(inplace=True)
df.to_csv("../data/distribution_maps.csv", index=False)

print("Saved ../data/distribution_maps.csv")
print("Records:", len(df))