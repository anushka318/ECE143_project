#!/usr/bin/env python
# coding: utf-8

# In[32]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
import pandas as pd


# In[33]:


df = pd.read_csv("C:/Program Files/R Files/cleaned_data.csv")
df


# In[34]:


df_encoded = df.copy()
for col in df_encoded.select_dtypes(include=['object']).columns:
    df_encoded[col] = LabelEncoder().fit_transform(df_encoded[col])


# In[35]:


X = df_encoded.drop(columns=["GAD_T", "SWL_T", "SPIN_T"])
y = df_encoded["GAD_T"]


# In[36]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)


# In[39]:


y_pred = model.predict(X_test[:5])
y_pred = [round(value, 2) for value in y_pred]
print("GAD_T prediction:", y_pred)


# In[43]:


model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
feature_importances = model.feature_importances_
feature_names = X.columns
indices = np.argsort(feature_importances)[::-1]
sorted_features = [feature_names[i] for i in indices]
sorted_importances = feature_importances[indices]
top_features = list(zip(sorted_features[:10], sorted_importances[:10]))
print("Top 10 most important features:")
for feature, importance in top_features:
    print(f"{feature}: {importance:.4f}")


# In[44]:


plt.figure(figsize=(10, 6))
plt.barh(sorted_features[:10], sorted_importances[:10], color='skyblue')
plt.xlabel("Feature Importance")
plt.ylabel("Features")
plt.title("Top Features Influencing GAD_T (Anxiety Score)")
plt.gca().invert_yaxis()
plt.show()


# In[ ]:




