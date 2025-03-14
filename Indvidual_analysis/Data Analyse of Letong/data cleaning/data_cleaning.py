# -*- coding: utf-8 -*-
"""data cleaning

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/182ZozmMacc_pLQXA4GrgYN4z3osa-1-W
"""

# Commented out IPython magic to ensure Python compatibility.
# This mounts your Google Drive to the Colab VM.
from google.colab import drive
drive.mount('/content/drive')


FOLDERNAME = 'ece143/'
assert FOLDERNAME is not None, "[!] Enter the foldername."

# Now that we've mounted your Drive, this ensures that
# the Python interpreter of the Colab VM can load
# python files from within it.
import sys
sys.path.append('/content/drive/My Drive/{}'.format(FOLDERNAME))

# This is later used to use the IMDB reviews
# %cd /content/drive/My\ Drive/$FOLDERNAME/

"""# import"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import plot_tree
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

data = pd.read_csv('anxiety.csv', encoding='ISO-8859-1')
data.head()

"""# Exploratory Data Analysis

* GAD means **Generalized Anxiety Disorde**, and 1~7 means the question number, GAD-T means the total score.
* SWL means **Satisf action with Life Scale**, and 1~5 means the question number, SWL-T means the total score.
* SPIN means **Social Phobia Inventory**, and 1~17 means the question number, SPIN-T means the total score.
"""

data.info()

data.isnull().sum()

for col in data:
    unique_vars = np.unique(data[col].astype(str))
    num_vars = len(unique_vars)
    if num_vars <= 1000:
        print('The number of value for feature {}: {} -- {}'.format(col, num_vars, unique_vars))
    else:
        top_15 = unique_vars[:15]
        print('The number of values for feature {}: {} -- {} ...'.format(col, num_vars, top_15))

"""# Data Preprocessing(each column seperately)

remove column 'S. No.', 'Timestamp', 'highestleague'
"""

data = data.drop(columns=['S. No.', 'Timestamp', 'highestleague'])

"""**Gender**: only remain male and female"""

data['Gender'].value_counts()

data = data[data['Gender'] != 'Other']

"""**Narcissism**: fills missing values (NaN) in the "Narcissism" column with the median value of that column"""

data['Narcissism'] = data['Narcissism'].fillna(data['Narcissism'].median())

"""**streams**: groups the "streams" column into bins and fills missing values (NaN) in the "streams" column with its median value"""

group_counts = data.groupby(pd.cut(data['streams'], bins=[-1, 40, 80, 120, 160, 200]))['streams'].count()

group_counts.plot(kind='bar')

plt.xlabel("streams")
plt.ylabel("Counts")
plt.title("Bar of streams")
plt.show()

data['streams'] = data['streams'].fillna(data['streams'].median())

"""**Game**: visualize the frequency of different "Game" categories in the dataset"""

plt.bar(data['Game'].value_counts().index, data['Game'].value_counts().values)
plt.xlabel("Game")
plt.ylabel("Counts")
plt.title("Bar of Game")
plt.tight_layout()
plt.xticks(rotation=45)
plt.show()

"""**Platform**:visualize the frequency of different "Platform" categories in the dataset"""

plt.bar(data['Platform'].value_counts().index, data['Platform'].value_counts().values)
plt.xlabel("Platform")
plt.ylabel("Counts")
plt.title("Bar of Platform")
plt.tight_layout()
plt.xticks(rotation=45)
plt.show()

"""Replace certain values in the "Platform" column with new labels. Change 'Console (PS, Xbox, ...)' to 'Console', 'Smartphone / Tablet' to 'MobileDevice'"""

data['Platform'] = data['Platform'].replace({'Console (PS, Xbox, ...)': 'Console',
                                     'Smartphone / Tablet': 'MobileDevice'})

"""**Accept**: count the number of occurrences of each unique value in the accept column of the data DataFrame"""

data['accept'].value_counts()

"""Since we only have one unique value in the accept column, so we remove the 'accept' column from the data DataFrame"""

data = data.drop(columns=['accept'])

"""**Work**

Visualize the distribution of values in the 'Work' column of the data DataFrame
"""

plt.bar(data['Work'].value_counts().index, data['Work'].value_counts().values)
plt.xlabel("Work")
plt.ylabel("Counts")
plt.title("Bar of Work")
plt.tight_layout()
plt.xticks(rotation=45)
plt.show()

"""1. Filling missing values (NaN) in the Work column of the data DataFrame with the value 'Unemployed / between jobs'
2. Replacing specific values with more concise labels for standardization.
"""

data['Work'] = data['Work'].fillna('Unemployed / between jobs')
data['Work'] = data['Work'].replace({'Student at college / university': 'CollegeStu',
                                     'Student at school': 'SchoolStu', 'Unemployed / between jobs':'Unemployed'})

"""**Degree**: Visualize the distribution of values in the 'Degree' column of the data DataFrame"""

plt.bar(data['Degree'].value_counts().index, data['Degree'].value_counts().values)
plt.xlabel("Degree")
plt.ylabel("Counts")
plt.title("Bar of Degree")
plt.tight_layout()
plt.xticks(rotation=45)
plt.show()

"""1. replace missing values (NaN) in the 'Degree' column of the data DataFrame with the string 'Other'
2. Replacing specific values with more concise labels for standardization.
"""

data['Degree'] = data['Degree'].fillna('Other')
data['Degree'] = data['Degree'].replace({'Bachelor\xa0(or equivalent)': 'BC',
                                         'High school diploma (or equivalent)': 'HS',
                                         'Master\xa0(or equivalent)': 'MA',
                                         'Ph.D., Psy. D., MD (or equivalent)': 'DC'})

"""**Reference**:visualize the frequency distribution of the values in the 'Reference' column of the data DataFrame"""

plt.bar(data['Reference'].value_counts().index, data['Reference'].value_counts().values)
plt.xlabel("Reference")
plt.ylabel("Counts")
plt.title("Bar of Reference")
plt.show()

""" replace missing values (NaN) in the 'Reference' column of the data DataFrame with the string 'Other'"""

data['Reference'] = data['Reference'].fillna('Other')

"""**GADE**

visualize the frequency distribution of the values in the 'GADE' column of the data DataFrame
"""

plt.bar(data['GADE'].value_counts().index, data['GADE'].value_counts().values)
plt.tight_layout()
plt.show()

""" fill missing values (NaN) in the 'GADE' column of the data DataFrame with the most frequent value (mode) of that column"""

data['GADE'] = data['GADE'].fillna(data['GADE'].mode()[0])

"""**Age**:visualizes the frequency distribution of the 'Age' column in the data DataFrame, divided into age groups or bins"""

group_counts = data.groupby(pd.cut(data['Age'], bins=[10, 20, 30, 40, 50, 60, 70]))['Age'].count()

group_counts.plot(kind='bar')

plt.xlabel("Age")
plt.ylabel("Counts")
plt.title("Bar of Age")
plt.show()

"""fill missing values (NaN) in the 'Age' column of the data DataFrame with the median value of the 'Age' column"""

data['Age'] = data['Age'].fillna(data['Age'].median())

"""**HOURS**:visualizes the frequency distribution of the 'Hours' column in the data DataFrame, divided into specified bins"""

group_counts = data.groupby(pd.cut(data['Hours'], bins=[-1, 20, 40, 60, 80, 100, 120, 140, 160, 180, 190]))['Hours'].count()

group_counts.plot(kind='bar')

plt.xlabel("Hours")
plt.ylabel("Counts")
plt.title("Bar of Hours")
plt.show()

"""fill missing values (NaN) in the 'Hours' column of the data DataFrame with the median value of the 'Hours' column"""

data['Hours'] = data['Hours'].fillna(data['Hours'].median())

"""**Residence**:visualize the distribution of values in the 'Residence' column of the data DataFrame"""

plt.bar(data['Residence'].value_counts().index, data['Residence'].value_counts().values)
plt.xlabel("Residence")
plt.ylabel("Counts")
plt.title("Bar of Residence")
plt.tight_layout()
plt.xticks(rotation=45)
plt.show()

"""1. Removes columns 'Residence_ISO3', 'Birthplace_ISO3', and 'Birthplace' which are not needed for the analysis because these columns provide redundant information.
2. Fills any missing values in the 'Residence' column with the most common residence value.
"""

data = data.drop(columns=['Residence_ISO3', 'Birthplace_ISO3', 'Birthplace'])
data['Residence'] = data['Residence'].fillna(data['Residence'].mode()[0])

"""**League**

print the number of different league
"""

for league, count in data['League'].value_counts().items():
    print(f"{league}: {count}")

"""Categorize the values in the 'League' column into broader categories based on certain keywords using regular expressions"""

data.loc[data['League'].str.contains(r'\bgold|\bgd|\bg1|\bg2|\bg3|\bg4|\bg5', case=False, na=False, regex=True), 'League'] = 'Gold'
data.loc[data['League'].str.contains(r'\bsilver|\bs1|\bs2|\bs3|\bs4|\bs5|\bsliver', case=False, na=False, regex=True), 'League'] = 'Silver'
data.loc[data['League'].str.contains(r'\bdiamond|\bd1|\bd2|\bd3|\bd4|\bd5', case=False, na=False, regex=True), 'League'] = 'Diamond'
data.loc[data['League'].str.contains(r'\bplatinum|\bplat|\bp1|\bp2|\bp3|\bp4|\bp5', case=False, na=False, regex=True), 'League'] = 'Platinum'
data.loc[data['League'].str.contains(r'\bbronze|\bb1|\bb2|\bb3|\bb4|\bb5', case=False, na=False, regex=True), 'League'] = 'Bronze'
data.loc[data['League'].str.contains(r'\bun|\bnone|\bna', case=False, na=False, regex=True), 'League'] = 'Unranked'
data.loc[data['League'].str.contains(r'\bmaster|\bm1|\bm2|\bm3|\bm4|\bm5', case=False, na=False, regex=True), 'League'] = 'Master'
data.loc[data['League'].str.contains(r'\bchallenger', case=False, na=False, regex=True), 'League'] = 'Challenger'

"""Filter the 'League' column by the frequency of occurrences of values. We remove any rows where the 'League' value appears fewer than 15 times"""

league_counts = data['League'].value_counts()
want_league = league_counts[league_counts>=15].index
data = data[data['League'].isin(want_league)]

"""**Playstyle**

print the number of different playstyle
"""

for Playstyle, count in data['Playstyle'].value_counts().items():
    print(f"{Playstyle}: {count}")

"""Select rows from the 'Playstyle' column that contain the word "all" (case-insensitive) and prints the filtered results"""

filtered_data = data['Playstyle'][data['Playstyle'].str.contains(r'\ball\b', case=False, na=False, regex=True)]

print(filtered_data)

"""Replacing values that contain the words "all", "above", or "them" with the string 'All'"""

data.loc[data['Playstyle'].str.contains(r'\ball|\babove|\bthem', case=False, na=False, regex=True), 'Playstyle'] = 'All'

"""Filter the 'Playstyle' column by the frequency of occurrences of values. We remove any rows where the 'Playstyle' value appears fewer than 25 times"""

Playstyle_counts = data['Playstyle'].value_counts()
want_Playstyle = Playstyle_counts[Playstyle_counts>=25].index
data = data[data['Playstyle'].isin(want_Playstyle)]

"""Replaces specific values in the 'Playstyle' column of the data DataFrame with new labels."""

data['Playstyle'] = data['Playstyle'].replace({'Multiplayer - offline (people in the same room)': 'offline',
                                     'Multiplayer - online - with online acquaintances or teammates': 'online acquaintances',
                                     'Multiplayer - online - with real life friends': 'online real life friends',
                                     'Multiplayer - online - with strangers': 'online strangers'})

"""**earnings**

print the number of different earnings
"""

for earnings, count in data['earnings'].value_counts().items():
    print(f"{earnings}: {count}")

"""Replaces specific values in the 'earnings' column of the data DataFrame with new labels."""

data['earnings'] = data['earnings'].replace({'I play for fun': 'ForFun',
                                     'I play mostly for fun but earn a little on the side (tournament winnings, streaming, etc)': 'Fun and else',
                                     'I earn a living by playing this game': 'Earn',
                                     'I play to win': 'Win'})

"""Filter the 'earnings' column by the frequency of occurrences of values. We remove any rows where the 'earnings' value appears fewer than 25 times"""

earnings_counts = data['earnings'].value_counts()
want_earnings = earnings_counts[earnings_counts>=25].index
data = data[data['earnings'].isin(want_earnings)]

"""**whyplay**

print the number of different reason for playing
"""

for whyplay, count in data['whyplay'].value_counts().items():
    print(f"{whyplay}: {count}")

"""1. Replaces values containing "all" or "above" with 'All'.
2. Replaces values containing "Improving" or "improve" with 'Improving'.
"""

data.loc[data['whyplay'].str.contains(r'\ball|\babove', case=False, na=False, regex=True), 'whyplay'] = 'All'
data.loc[data['whyplay'].str.contains(r'\bImproving|\bimprove', case=False, na=False, regex=True), 'whyplay'] = 'Improving'

"""Filter the 'whyplay' column by the frequency of occurrences of values. We remove any rows where the 'whyplay' value appears fewer than 25 times"""

whyplay_counts = data['whyplay'].value_counts()
want_whyplay = whyplay_counts[whyplay_counts>=25].index
data = data[data['whyplay'].isin(want_whyplay)]

"""Transforms the values in columns 'SWL1' to 'SWL5' by subtracting each value from 7. This is used to invert the scale."""

for i in range(1, 6):
    data['SWL'+str(i)] = 7-data['SWL'+str(i)]

"""Drops rows from the data DataFrame where there are missing values (NaN) in the columns SPIN1, SPIN2, ..., SPIN17"""

for i in range(1, 18):
    data = data.dropna(subset=[f'SPIN{i}'])

"""Reset the index after performing operations"""

data = data.reset_index(drop=True)

data.shape

"""Correlation matrix with heatmap"""

personal_corr_matrix = data.drop(columns = ['GADE', 'Game', 'Platform', 'earnings', 'whyplay', 'League',
                                            'Gender', 'Work', 'Degree', 'Residence', 'Playstyle', 'Reference']).corr()

mask = np.triu(np.ones_like(personal_corr_matrix, dtype=bool))

plt.figure(figsize=(20,18))
sns.heatmap(personal_corr_matrix, cmap='coolwarm', center=0, fmt=".2f", annot=True, linewidths=1, mask=mask)
plt.show()

data.info()

"""After data cleaning, there is no empty data in the dataset"""

data.isnull().sum()

for col in data:
    unique_vars = np.unique(data[col].astype(str))
    num_vars = len(unique_vars)
    if num_vars <= 1000:
        print('The number of value for feature {}: {} -- {}'.format(col, num_vars, unique_vars))
    else:
        top_15 = unique_vars[:15]
        print('The number of values for feature {}: {} -- {} ...'.format(col, num_vars, top_15))

"""save the cleaned dataset"""

from google.colab import files
data.to_csv('cleaned_data.csv', index=False)  # save document
files.download('cleaned_data.csv')  # download