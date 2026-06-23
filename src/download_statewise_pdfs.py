import requests
import os

os.makedirs("../data/statewise_pdfs", exist_ok=True)

states = {
    "karnataka": "https://envis.frlht.org/checklist/karna.pdf",
    "kerala": "https://envis.frlht.org/checklist/KERALA.pdf",
    "tamilnadu": "https://envis.frlht.org/checklist/TN.pdf",
    "andhrapradesh": "https://envis.frlht.org/checklist/ANDHRA.pdf",
    "chattisgarh": "https://envis.frlht.org/checklist/Chattisgadh.pdf",
    "odisha": "https://envis.frlht.org/checklist/Orissa.pdf",
    "sikkim": "https://envis.frlht.org/checklist/Sikkim.pdf",
    "rajasthan": "https://envis.frlht.org/checklist/Rajasthan.pdf",
    "westbengal": "https://envis.frlht.org/checklist/WestBengal.pdf",
    "maharashtra": "https://envis.frlht.org/checklist/maharas.pdf",
}

for state, url in states.items():
    print("Downloading:", state)

    response = requests.get(url)

    filename = f"../data/statewise_pdfs/{state}.pdf"

    with open(filename, "wb") as f:
        f.write(response.content)

    print("Saved:", filename, "Size:", len(response.content), "bytes")