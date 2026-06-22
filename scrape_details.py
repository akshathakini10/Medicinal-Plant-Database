import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
import time

def extract_between(text, start, end_options):
    start_match = re.search(start + r"\s*-\s*", text)
    if not start_match:
        return ""

    start_index = start_match.end()
    remaining = text[start_index:]

    end_positions = []
    for end in end_options:
        end_match = re.search(end, remaining)
        if end_match:
            end_positions.append(end_match.start())

    if not end_positions:
        return remaining[:300].strip()

    end_index = min(end_positions)
    return remaining[:end_index].replace("X", "").strip()

def scrape_plant(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    text = soup.get_text(" ", strip=True)

    botanical_name = extract_between(
        text,
        "Botanical Name",
        ["X Full Botanical citation", "Family"]
    )

    family = extract_between(
        text,
        "Family",
        ["System\\(s\\) of Indian Medicine", "Botanical Synonyms", "Vernacular names"]
    )

    systems = extract_between(
        text,
        "System\\(s\\) of Indian Medicine",
        ["Botanical Synonyms", "Vernacular names"]
    )

    synonym = extract_between(
        text,
        r"Botanical Synonyms\s*\(\d+\)",
        ["X Full Botanical citation", "Vernacular names"]
    )

    return {
        "Botanical Name": botanical_name,
        "Family": family,
        "Systems": systems,
        "Synonym": synonym,
        "Detail URL": url
    }

links_df = pd.read_csv("plant_links.csv")

all_data = []

for index, row in links_df.iterrows():
    print("Scraping:", index + 1, row["name"])

    try:
        data = scrape_plant(row["url"])
        all_data.append(data)
    except Exception as e:
        print("Failed:", row["url"], e)

    time.sleep(1)

df = pd.DataFrame(all_data)
df.to_csv("all_medicinal_plants_clean.csv", index=False)

print("Saved all_medicinal_plants_clean.csv")