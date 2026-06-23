import requests

url = "https://www.medicinalplants.in/statewiselist/karnataka"

html = requests.get(url).text

print("HTML length:", len(html))

print("\nPossible API / JavaScript lines:\n")

for line in html.split("\n"):
    line_lower = line.lower()

    if (
        "$.post" in line_lower or
        "$.get" in line_lower or
        "ajax" in line_lower or
        "state" in line_lower or
        "karnataka" in line_lower or
        "iframe" in line_lower or
        "pdf" in line_lower
    ):
        print(line.strip())