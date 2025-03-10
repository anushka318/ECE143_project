import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("./Data/anxiety.csv.csv")
df

df_encoded = df.copy()
for col in df_encoded.select_dtypes(include=["object"]).columns:
    df_encoded[col] = LabelEncoder().fit_transform(df_encoded[col])

X = df_encoded.drop(columns=["GAD_T", "SWL_T", "SPIN_T"])
y_gad = df_encoded["GAD_T"]
y_swl = df_encoded["SWL_T"]
y_spin = df_encoded["SPIN_T"]

X_train, X_test, y_train_gad, y_test_gad = train_test_split(X, y_gad, test_size=0.2, random_state=42)
X_train, X_test, y_train_swl, y_test_swl = train_test_split(X, y_swl, test_size=0.2, random_state=42)
X_train, X_test, y_train_spin, y_test_spin = train_test_split(X, y_spin, test_size=0.2, random_state=42)

gad_model = RandomForestRegressor(n_estimators=100, random_state=42)
gad_model.fit(X_train, y_train_gad)
swl_model = RandomForestRegressor(n_estimators=100, random_state=42)
swl_model.fit(X_train, y_train_swl)
spin_model = RandomForestRegressor(n_estimators=100, random_state=42)
spin_model.fit(X_train, y_train_spin)

feature_columns = X.columns
random_sample = {col: np.random.uniform(df_encoded[col].min(), df_encoded[col].max()) for col in feature_columns}
random_sample_df = pd.DataFrame([random_sample])
random_sample_df

gad_prediction = gad_model.predict(random_sample_df)
swl_prediction = swl_model.predict(random_sample_df)
spin_prediction = spin_model.predict(random_sample_df)

print("Random Test Sample:", random_sample_df.to_dict(orient="records"))

print(f"GAD_T Prediction: {gad_prediction[0]:.2f}")
print(f"SWL_T Prediction: {swl_prediction[0]:.2f}")
print(f"SPIN_T Prediction: {spin_prediction[0]:.2f}")

mental_health_scores = ["GAD_T", "SWL_T", "SPIN_T"]
gaming_behaviors = ["Hours", "Playstyle", "League", "streams"]
demographics = ["Age", "Gender", "earnings"]

correlation_matrix = df[mental_health_scores + gaming_behaviors + demographics].corr()
plt.figure(figsize=(10, 6))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Correlation Matrix of Mental Health Scores, Gaming, and Demographics")
plt.show()

for target in mental_health_scores:
    X = df_encoded[gaming_behaviors + demographics]
    y = df_encoded[target]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_scaled = sm.add_constant(X_scaled)
    model = sm.OLS(y, X_scaled).fit()
    print(f"Regression Results for {target}:")
    print(model.summary())
    print("\n" + "="*80 + "\n")

for target in mental_health_scores:
    plt.figure(figsize=(6, 4))
    sns.boxplot(x=df_encoded["Gender"], y=df_encoded[target])
    plt.xlabel("Gender")
    plt.ylabel(target)
    plt.title(f"{target} Distribution by Gender")
    plt.show()

for feature in ["Age", "earnings"]:
    for target in mental_health_scores:
        plt.figure(figsize=(6, 4))
        sns.scatterplot(x=df_encoded[feature], y=df_encoded[target], alpha=0.5)
        plt.xlabel(feature)
        plt.ylabel(target)
        plt.title(f"{target} vs {feature}")
        plt.show()
