import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df_encoded = pd.read_csv("./Data/anxiety.csv")
df_encoded

X = df_encoded.drop(columns=["GAD_T", "SWL_T", "SPIN_T"])
y = df_encoded["SPIN_T"]
X = X.apply(lambda col: LabelEncoder().fit_transform(col) if col.dtype == 'object' else col)
if y.dtype == 'object':
    y = LabelEncoder().fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test[:5])
y_pred = [round(value, 2) for value in y_pred]
print("SPIN_T prediction:", y_pred)

feature_importances = model.feature_importances_
feature_names = X.columns
indices = np.argsort(feature_importances)[::-1]
sorted_features = [feature_names[i] for i in indices]
sorted_importances = feature_importances[indices]
top_features = list(zip(sorted_features[:10], sorted_importances[:10]))
print("Top 10 most important features affecting SPIN_T：")
for feature, importance in top_features:
    print(f"{feature}: {importance:.4f}")

plt.figure(figsize=(10, 6))
plt.barh(sorted_features[:10], sorted_importances[:10], color='skyblue')
plt.xlabel("Feature Importance")
plt.ylabel("Feature")
plt.title("Key Features Influencing SPIN_T (Social Anxiety Score)")
plt.gca().invert_yaxis()
plt.show()
