import pandas as pd
import plotly as pl

df = pd.read_csv("./anxiety.csv", encoding = 'ISO-8859-1')
df

df.drop(columns=['Timestamp'])

from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
df[df.select_dtypes(include=['object']).columns] = df.select_dtypes(include=['object']).apply(label_encoder.fit_transform)
df.info()

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("./anxiety.csv", encoding='ISO-8859-1')

corr = df.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
plt.figure(figsize=(14, 12))
sns.heatmap(
    corr, 
    mask=mask, 
    cmap="viridis", 
    annot=True, 
    fmt=".1f",
    annot_kws={"size": 8},
    linewidths=0.5, 
    cbar_kws={"shrink": 0.8, "label": "Correlation"}
)
plt.title("Correlation Matrix Heatmap", fontsize=16)
plt.show()

print(df.Age.describe())

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

age_counts = df['Age'].value_counts().reset_index()
age_counts.columns = ['Age', 'Count']
top_n = 10
if len(age_counts) > top_n:
    other_count = age_counts.iloc[top_n:]['Count'].sum()
    age_counts = age_counts.iloc[:top_n]
    age_counts.loc[len(age_counts)] = ['Other', other_count] 
colors = plt.cm.Set2(np.linspace(0, 1, len(age_counts)))
plt.figure(figsize=(10, 8)) 
wedges, texts, autotexts = plt.pie(
    age_counts['Count'], 
    labels=age_counts['Age'], 
    autopct='%1.1f%%',
    colors=colors, 
    startangle=140,  
    wedgeprops={'edgecolor': 'black', 'linewidth': 1},
    pctdistance=0.9,
    labeldistance=1.1 
)
for autotext in autotexts:
    autotext.set_fontsize(8)
    autotext.set_color('black')
plt.legend(wedges, age_counts['Age'], title="Age Groups", loc="center left", bbox_to_anchor=(1, 0.5))
plt.title('Age Distribution')
plt.show()

print(df.Gender.describe())

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

gender_counts = df['Gender'].value_counts()
total_count = gender_counts.sum()
percentages = (gender_counts / total_count * 100).round(2)
labels = [f"{count} ({percent}%)" for count, percent in zip(gender_counts.values, percentages)]
colors = ['cornflowerblue', 'lightsalmon'] if len(gender_counts) == 2 else sns.color_palette("pastel")
plt.figure(figsize=(8, 5))
bars = plt.bar(gender_counts.index, gender_counts.values, color=colors, edgecolor='black')
for bar, label in zip(bars, labels):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), label, ha='center', va='bottom', fontsize=10)
plt.title("Count of Genders", fontsize=14)
plt.xlabel("Gender", fontsize=12)
plt.ylabel("Count", fontsize=12)
plt.xticks(rotation=0)
plt.show()

games_counts = df['Game'].value_counts().rename_axis('Game').reset_index(name='count')
games_counts

import matplotlib.pyplot as plt
import pandas as pd

games_counts_top = games_counts.head(top_n).copy()
games_counts_top['Label'] = games_counts_top.apply(lambda row: f"{row['count']} ({row['Percentage']}%)", axis=1)
max_count = games_counts_top['count'].max()
games_counts_top['TextPosition'] = games_counts_top['count'].apply(lambda x: 'inside' if x > 0.5 * max_count else 'outside')
fig, ax = plt.subplots(figsize=(12, 9))
bar_colors = ['royalblue'] * len(games_counts_top)
bars = ax.barh(games_counts_top['Game'], games_counts_top['count'], color=bar_colors)
for bar, label, count, pos in zip(bars, games_counts_top['Label'], games_counts_top['count'], games_counts_top['TextPosition']):
    if pos == 'inside':
        ax.text(count - max_count * 0.05, bar.get_y() + bar.get_height()/2, label, va='center', ha='right', color='white', fontsize=12, fontweight='bold')
    else:
        ax.text(count + max_count * 0.02, bar.get_y() + bar.get_height()/2, label, va='center', ha='left', color='black', fontsize=12)
ax.set_xlabel("Count", fontsize=14)
ax.set_ylabel("Game", fontsize=14)
ax.set_title("Top Games by Count", fontsize=16, fontweight='bold')
plt.gca().invert_yaxis()
plt.grid(axis='x', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

platform_counts = df['Platform'].value_counts().rename_axis('platform').reset_index(name='count')
platform_counts

print(df['Hours'].describe())

import seaborn as sns
import matplotlib.pyplot as plt

max_hours_week = 16 * 7
legit_hour_df = df.query('Hours <= @max_hours_week')
residence_hours = legit_hour_df.groupby('Residence').Hours.agg(['mean', 'size']).query('Residence != "Unknown" & size >= 100')
residence_hours.sort_values(by=['mean'], ascending=[False], inplace=True)
plt.figure(figsize=(15, 6))
sns.boxplot(data=legit_hour_df[legit_hour_df['Residence'].isin(residence_hours.index)], 
            x='Hours', y='Residence', showfliers=False, palette="coolwarm")
plt.title("Distribution of Hours Played by Residence")
plt.xlabel("Hours Played")
plt.ylabel("Residence")
plt.show()

import seaborn as sns

plt.figure(figsize=(10, 6))
sns.boxplot(
    x=legit_hour_df["Age"].astype(int), 
    y=legit_hour_df["Hours"], 
    palette="coolwarm"
)
plt.title("Age vs. Hours Played (Box Plot)", fontsize=14, fontweight="bold")
plt.xlabel("Age", fontsize=12)
plt.ylabel("Hours Played", fontsize=12)
plt.xticks(rotation=45)
plt.show()

import matplotlib.pyplot as plt

playstyle_counts_sorted = playstyle_counts_df.sort_values(by="Count", ascending=True)
colors = plt.cm.Set2(range(len(playstyle_counts_sorted)))
fig, ax = plt.subplots(figsize=(12, 6))
bars = ax.barh(playstyle_counts_sorted["Playstyle"], playstyle_counts_sorted["Count"], color=colors)
for bar, label in zip(bars, playstyle_counts_sorted["Label"]):
    ax.text(bar.get_width() + max(playstyle_counts_sorted["Count"]) * 0.02,
            bar.get_y() + bar.get_height()/2, 
            label, 
            va='center', ha='left', fontsize=12)
ax.set_xlabel("Count", fontsize=14)
ax.set_ylabel("Playstyle", fontsize=14)
ax.set_title("Distribution of Playstyle Choices", fontsize=16, fontweight='bold')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
plt.grid(axis='x', linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()
