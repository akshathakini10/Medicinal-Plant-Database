import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

session = requests.Session()

headers = {
    "User-Agent": "Mozilla/5.0",
    "Referer": "https://www.medicinalplants.in/distributionmaps"
}

session.get("https://www.medicinalplants.in/distributionmaps", headers=headers)

url = "https://www.medicinalplants.in/mapsoperationdetails/0/20"
response = session.get(url, headers=headers)

print("Status:", response.status_code)
print("Length:", len(response.text))

soup = BeautifulSoup(response.text, "html.parser")

print("\nText preview:")
print(soup.get_text(" ", strip=True)[:3000])

print("\nLinks:")
for a in soup.find_all("a"):
    text = a.get_text(" ", strip=True)
    href = a.get("href")
    if href:
        print(text, "->", urljoin(url, href))

print("\nImages:")
for img in soup.find_all("img"):
    src = img.get("src")
    if src:
        print(urljoin(url, src))