import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df_encoded = pd.read_csv("C:/Program Files/R Files/cleaned_data.csv")
df_encoded

correlation_data = df_encoded[["SPIN_T", "GAD_T", "SWL_T"]]
correlation_data

correlation_matrix = correlation_data.corr()
print("Correlation Matrix:")
print(correlation_matrix)

plt.figure(figsize=(6, 4))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Correlation Matrix: SPIN_T, GAD_T, SWL_T")
plt.show()

sns.pairplot(correlation_data)
plt.show()
