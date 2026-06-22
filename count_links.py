import requests
from bs4 import BeautifulSoup

url = "https://www.medicinalplants.in/ayurvedasearchpage/getayurvedabotanical/pageno/0"

response = requests.post(url, data={"fname": "a"})
soup = BeautifulSoup(response.text, "html.parser")

links = soup.find_all("a")

print("Actual links returned:", len(links))
print("HTML length:", len(response.text))