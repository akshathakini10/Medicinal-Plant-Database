import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

url = "https://www.medicinalplants.in/statewiselist/karnataka"

html = requests.get(url).text
soup = BeautifulSoup(html, "html.parser")

print("Status: OK")
print("HTML length:", len(html))

print("\nText preview:")
print(soup.get_text(" ", strip=True)[:3000])

print("\nLinks:")
for a in soup.find_all("a"):
    text = a.get_text(" ", strip=True)
    href = a.get("href")
    if href:
        print(text, "->", urljoin(url, href))