import requests
import os

os.makedirs("../data/trade_pdfs", exist_ok=True)

pdfs = {
    "normal_trade.pdf": "https://www.medicinalplants.in/tradelist/normal_trade.pdf",
    "high_trade.pdf": "https://www.medicinalplants.in/tradelist/high_trade.pdf"
}

for filename, url in pdfs.items():
    print("Downloading:", filename)

    response = requests.get(url)

    with open(f"../data/trade_pdfs/{filename}", "wb") as f:
        f.write(response.content)

    print("Saved:", filename, "Size:", len(response.content), "bytes")

#check_trade
# with open("../data/trade_pdfs/high_trade.pdf", "rb") as f:
#     data = f.read()

# print(data)
