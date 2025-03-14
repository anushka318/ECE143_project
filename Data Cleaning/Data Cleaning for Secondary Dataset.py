import pandas as pd

file_path = "../Data/StressLevelDataset.csv"
df = pd.read_csv(file_path, encoding = "utf-8-sig", skipinitialspace = True)
df

print(df.info())

print(df.head())

print(df.isnull().sum())

df.describe()

df = df.dropna()
df
