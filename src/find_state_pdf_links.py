import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

states = [
    "karnataka",
    "kerala",
    "tamilnadu",
    "andhrapradesh",
    "chattisgarh",
    "odisha",
    "sikkim",
    "rajasthan",
    "westbengal",
    "maharashtra"
]

for state in states:
    url = f"https://www.medicinalplants.in/statewiselist/{state}"
    html = requests.get(url).text
    soup = BeautifulSoup(html, "html.parser")

    print("\nSTATE:", state)

    for iframe in soup.find_all("iframe"):
        src = iframe.get("src")
        if src:
            print("PDF:", urljoin(url, src))