import pandas as pd

df = pd.read_csv("../data/distribution_maps.csv")
print(df.head())
print(df.shape)
print(df.isnull().sum())