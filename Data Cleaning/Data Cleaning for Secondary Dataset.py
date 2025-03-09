#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


file_path = "C:/Program Files/R Files/StressLevelDataset.csv"
df = pd.read_csv(file_path, encoding = "utf-8-sig", skipinitialspace = True)
df


# In[3]:


print(df.info())


# In[4]:


print(df.head())


# In[5]:


print(df.isnull().sum())


# In[6]:


df.describe()


# In[8]:


df = df.dropna()
df


# In[ ]:




