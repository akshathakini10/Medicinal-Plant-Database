import requests

url = "https://www.medicinalplants.in/trade"

html = requests.get(url).text

print("HTML length:", len(html))

print("\nPossible API / JavaScript lines:\n")

for line in html.split("\n"):
    line_lower = line.lower()

    if (
        "$.post" in line_lower or
        "$.get" in line_lower or
        "ajax" in line_lower or
        "trade" in line_lower or
        "function" in line_lower
    ):
        print(line.strip())