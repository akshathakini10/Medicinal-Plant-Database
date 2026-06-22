import pandas as pd

links = pd.read_csv("plant_links.csv")

print("Total names collected:", len(links))
print("Unique URLs:", links["url"].nunique())