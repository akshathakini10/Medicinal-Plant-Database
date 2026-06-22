import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import pandas as pd
import time

base_url = "https://www.medicinalplants.in/"

systems = {

    "Ayurveda": "https://www.medicinalplants.in/ayurvedasearchpage/getayurvedabotanical/pageno/0",
    "Siddha": "https://www.medicinalplants.in/siddhasearchpage/getsiddhabotanical/pageno/0",
    "Unani": "https://www.medicinalplants.in/unanisearchpage/getunanibotanical/pageno/0",
    "Homeopathy": "https://www.medicinalplants.in/homeopathysearchpage/gethomeopathybotanical/pageno/0",
    "Folk": "https://www.medicinalplants.in/folksearchpage/getfolkbotanical/pageno/0",
    "Sowa-Rigpa": "https://www.medicinalplants.in/tibetansearchpage/gettibetanbotanical/pageno/0",

}

letters = "abcdefghijklmnopqrstuvwxyz"

all_links = []

for system_name, search_url in systems.items():
    print("System:", system_name)

    for letter in letters:
        print("Searching:", system_name, letter)

        response = requests.post(search_url, data={"fname": letter})
        soup = BeautifulSoup(response.text, "html.parser")

        for a in soup.find_all("a"):
            name = a.get_text(" ", strip=True)
            link = a.get("href")

            if name and link and "showdetails" in link:
                full_link = urljoin(base_url, link)

                all_links.append({
                    "System Source": system_name,
                    "Search Name": name,
                    "Detail URL": full_link
                })

        time.sleep(0.3)

df = pd.DataFrame(all_links)

print("Total links:", len(df))
print("Unique detail URLs:", df["Detail URL"].nunique())

df.to_csv("all_systems_links.csv", index=False)

print("Saved all_systems_links.csv")