# import requests
# from bs4 import BeautifulSoup
# import re

# url = "https://www.medicinalplants.in/searchpage/showdetails/xplant_id/824a2c2219788b48e2a3bb1633d43370"

# response = requests.get(url)
# soup = BeautifulSoup(response.text, "html.parser")

# text = soup.get_text(" ", strip=True)

# def extract_between(text, start, end):
#     pattern = start + r"\s*-\s*(.*?)\s*" + end
#     match = re.search(pattern, text)
#     return match.group(1).strip() if match else "Not found"

# botanical_name = extract_between(text, "Botanical Name", "X Full Botanical citation")
# family = extract_between(text, "Family", "System")
# systems = extract_between(text, "System\\(s\\) of Indian Medicine", "Botanical Synonyms")
# synonym = extract_between(text, "Botanical Synonyms \\(1\\)", "X Full Botanical citation")

# print("Botanical Name:", botanical_name)
# print("Family:", family)
# print("Systems:", systems)
# print("Synonym:", synonym)

# import pandas as pd

# data = [{
#     "Botanical Name": botanical_name,
#     "Family": family,
#     "Systems": systems,
#     "Synonym": synonym
# }]

# df = pd.DataFrame(data)
# df.to_csv("plants.csv", index=False)

# print("Saved to plants.csv")

import pandas as pd

df = pd.read_csv("all_medicinal_plants_clean.csv")

print("Plants:", len(df))
print("Families:", df["Family"].nunique())
print("Systems:", df["Systems"].nunique())