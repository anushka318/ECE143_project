#!/usr/bin/env python
# coding: utf-8

# In[13]:


import pandas as pd


# In[14]:


file_path = "C:/Program Files/R Files/Social_Anxiety_Survey_Master.csv"
df = pd.read_csv(file_path, encoding="utf-8-sig", skipinitialspace=True)
df


# In[15]:


print(df.info())


# In[16]:


print(df.head())


# In[17]:


print(df.isnull().sum())


# In[18]:


print("Gender:", df["Gender"].unique())
print("Student:", df["Student"].unique())
print("Age:", df["Age"].unique())
print("Marital:", df["Marital"].unique())


# In[ ]:




