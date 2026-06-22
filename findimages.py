import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

url = "https://www.medicinalplants.in/searchpage/showdetails/xplant_id/824a2c2219788b48e2a3bb1633d43370"

html = requests.get(url).text
soup = BeautifulSoup(html, "html.parser")

for a in soup.find_all("a"):
    text = a.get_text(" ", strip=True)
    href = a.get("href")

    if href:
        if ("image" in href.lower() or
            "photo" in href.lower() or
            "gallery" in href.lower()):
            print(text, "->", urljoin(url, href))

    