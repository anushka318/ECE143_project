import pandas as pd

file_path = "C:/Program Files/R Files/Social_Anxiety_Survey_Master.csv"
df = pd.read_csv(file_path, encoding="utf-8-sig", skipinitialspace=True)
df

print(df.info())

print(df.head())

print(df.isnull().sum())

print("Gender:", df["Gender"].unique())
print("Student:", df["Student"].unique())
print("Age:", df["Age"].unique())
print("Marital:", df["Marital"].unique())
