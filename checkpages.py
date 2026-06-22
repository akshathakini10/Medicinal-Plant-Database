import requests

for page in range(5):
    url = f"https://www.medicinalplants.in/ayurvedasearchpage/getayurvedabotanical/pageno/{page}"

    response = requests.post(
        url,
        data={"fname": "a"}
    )

    print("Page", page)
    print(response.text[:300])
    print("-" * 50)