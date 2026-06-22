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
        return remaining[:500].replace("X", "").strip()

    end_index = min(end_positions)
    return remaining[:end_index].replace("X", "").strip()

def scrape_plant(url):
    html = requests.get(url, timeout=20).text
    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text(" ", strip=True)

    botanical_name = extract_between(text, "Botanical Name", ["X Full Botanical citation", "Family"])
    family = extract_between(text, "Family", ["System\\(s\\) of Indian Medicine", "Botanical Synonyms", "Vernacular names"])
    systems = extract_between(text, "System\\(s\\) of Indian Medicine", ["Botanical Synonyms", "Vernacular names"])
    synonym = extract_between(text, r"Botanical Synonyms\s*\(\d+\)", ["X Full Botanical citation", "Vernacular names"])
    vernacular = extract_between(text, "Vernacular names", ["About Us", "Contact Us"])

    return {
        "Botanical Name": botanical_name,
        "Family": family,
        "Systems": systems,
        "Synonym": synonym,
        "Vernacular Names": vernacular,
        "Detail URL": url
    }

links_df = pd.read_csv("all_systems_links.csv")
unique_urls = links_df["Detail URL"].drop_duplicates().tolist()

print("Unique pages to scrape:", len(unique_urls))

all_data = []

# test first 20 only
for i, url in enumerate(unique_urls):
    print("Scraping:", i + 1, url)

    try:
        data = scrape_plant(url)
        all_data.append(data)
    except Exception as e:
        print("Failed:", url, e)

    time.sleep(0.3)

df = pd.DataFrame(all_data)
df.to_csv("final_medicinal_plants.csv", index=False)

print("Saved final_medicinal_plants.csv")