import pandas as pd
import requests
from bs4 import BeautifulSoup

df = pd.read_csv("all_medicinal_plants.csv")

missing = df[df["Botanical Name"].isnull()].head(1)
url = missing.iloc[0]["Detail URL"]

print("URL:", url)

html = requests.get(url).text
soup = BeautifulSoup(html, "html.parser")

print(soup.get_text(" ", strip=True)[:5000])

print("\nIMAGES:")
for img in soup.find_all("img"):
    print(img.get("src"))