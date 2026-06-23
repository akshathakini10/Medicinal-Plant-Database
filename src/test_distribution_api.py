import requests
from bs4 import BeautifulSoup

session = requests.Session()

headers = {
    "User-Agent": "Mozilla/5.0",
    "Referer": "https://www.medicinalplants.in/distributionmaps"
}

# first open the main page to get cookies/session
main_url = "https://www.medicinalplants.in/distributionmaps"
session.get(main_url, headers=headers)

# now call the hidden API using same session
api_url = "https://www.medicinalplants.in/mapsoperationinitialize"
response = session.get(api_url, headers=headers)

print("Status:", response.status_code)
print("Length:", len(response.text))
print(response.text[:3000])

soup = BeautifulSoup(response.text, "html.parser")

print("\nLinks:")
for a in soup.find_all("a"):
    print(a.get_text(" ", strip=True), "->", a.get("href"))

print("\nForms:")
for form in soup.find_all("form"):
    print(form.prettify())
# import requests

# html = requests.get(
#     "https://www.medicinalplants.in/distributionmaps"
# ).text

# keyword = "mapsoperationinitialize"

# pos = html.find(keyword)

# print("Position:", pos)

# print("\n" + html[pos-1000:pos+3000])