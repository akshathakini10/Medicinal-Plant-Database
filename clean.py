import pandas as pd

df = pd.read_csv("all_medicinal_plants_clean.csv")

print(df)
print(df.isnull().sum())