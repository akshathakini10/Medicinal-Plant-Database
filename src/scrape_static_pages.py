import requests
from bs4 import BeautifulSoup
import os

os.makedirs("../data/static_pages", exist_ok=True)

pages = {
    "faq": "https://www.medicinalplants.in/faq",
    "about_database": "https://www.medicinalplants.in/aboutfrlhtdb",
    "about_nmpb": "https://www.medicinalplants.in/aboutnmpb",
    "about_frlht": "https://www.medicinalplants.in/aboutfrlht",
    "contact": "https://www.medicinalplants.in/contact",
    "team": "https://www.medicinalplants.in/team"
}

for name, url in pages.items():
    print("Scraping:", name)

    html = requests.get(url).text
    soup = BeautifulSoup(html, "html.parser")

    text = soup.get_text("\n", strip=True)

    with open(f"../data/static_pages/{name}.txt", "w", encoding="utf-8") as f:
        f.write(text)

    print("Saved:", name)