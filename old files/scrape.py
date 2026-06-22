import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import time

base_url = "https://www.medicinalplants.in/"
search_url = "https://www.medicinalplants.in/ayurvedasearchpage/getayurvedabotanical/pageno/0"

letters = "abcdefghijklmnopqrstuvwxyz"

plant_links = []

for letter in letters:
    print("Searching:", letter)

    response = requests.post(search_url, data={"fname": letter})
    soup = BeautifulSoup(response.text, "html.parser")

    for a in soup.find_all("a"):
        name = a.get_text(" ", strip=True)
        link = a.get("href")

        if name and link and "showdetails" in link:
            full_link = urljoin(base_url, link)

            plant_links.append({
                "name": name,
                "url": full_link
            })

    time.sleep(2)

print("Total links found:", len(plant_links))

for plant in plant_links[:10]:
    print(plant)

import pandas as pd

df = pd.DataFrame(plant_links)
df.drop_duplicates(subset=["url"], inplace=True)
df.to_csv("plant_links.csv", index=False)

print("Saved plant_links.csv")