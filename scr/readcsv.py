import pandas as pd

df = pd.read_csv("final_medicinal_plants.csv")

print(df.head())
print(df.shape)
print(df.isnull().sum())
print(df.duplicated().sum())