# ECE143 Final Project Group 17

# Student Anxiety Analysis and Prediction System

# Team Members:
Anushka Chaudhary; Letong Wang; Jinbo Ma; Zhengyang Zhou

# Problem:
This project analyzes student anxiety data to reveal underlying patterns and key triggers.

# Dataset:
Kaggle open dataset (https://www.kaggle.com/datasets/petalme/student-anxiety-dataset)

The entire dataset consists of one csv file containing various information related to student anxiety, including factors contributing to anxiety, its manifestations, and potential mitigating influences. Through this dataset, we can gain deep insights into students' mental health, stress levels, academic pressures, coping strategies, and other related attributes.

# Proposed Solution and Real-World Application:
Our proposed solution focuses on utilizing data preprocessing and exploratory data analysis to study student anxiety, structuring the project into two main phases: data cleaning and data analysis with visualization. 

The primary objective is to develop interactive visualizations and dashboards that highlight patterns, trends, and correlations, such as common anxiety triggers and the effectiveness of various coping strategies. 

The real-world applications of this project include early intervention, where educators and mental health professionals can leverage these insights to identify high-risk students and implement timely support measures; policy-making, which involves analyzing key contributing factors to student anxiety to inform and refine school policies and mental health programs; and resource allocation, ensuring that mental health resources and support systems are distributed optimally based on data-driven insights.

# File Structure
**1. Data:** It contains the original dataset (anxiety.csv) and the output dataset (cleaned_data) after the cleaning is completed.

**2. Data Cleaning:** It is mainly used for data cleaning, which contains a PDF file with a detailed explanation of the data cleaning process and a dataset (cleaned_data) output after the cleaning is completed.

**3. Indvidual Analysis:** It mainly includes the extended data analysis independently conducted by Zhengyang Zhou and Letong Wang, relating to getting familiar with the data and finding correlations bewteen factors. It includes prediction and influencing factor analysis of GAD_T, SPIN_T and SWL_T (PDF with explanations), analysis of the interaction of GAD_T, SPIN_T and SWL_T (PDF with explanations), and analysis of the impact of GAD_T, SPIN_T and SWL_T against gaming behaviors, age, gender and earnings (PDF with explanations).

**4.module** This has the python modules used for visualization, and categorizing data.

**5.Analysis notebook**: The notebook with final analysis.

**6. Visualization:** This has the images of all the visualizations from the data analysis.

# Main Analysis Notebooks
**Attention: All python files are stored in both .py and .ipynb forms with the same file name.**

**1. Data cleaning.py /.ipynb:** In this data processing task, we systematically cleaned the raw data by identifying and removing inconsistencies, filling in missing values using appropriate imputation methods and standardizing the data to maintain uniformity across all variables. 

**2. Analysis /.ipynb:** This  Notebook explores the relationships between various factors in a survey dataset. The analysis examines how gaming habits and personal characteristics correlate with mental health indicators such as Generalized Anxiety Disorder (GAD) scores, Satisfaction with Life (SWL), and Social Phobia Inventory (SPIN) scores. Specifically, the notebook investigates the impact of games played, degree level, area of residence, and gaming playstyle on these psychological factors.

# Setup
Install the required dependencies:

```python
# Standard Libraries
import csv
import os
import numpy as np
import pandas as pd
import plotly as pl
import matplotlib.pyplot as plt
import seaborn as sns


# Scikit-Learn Components
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from wordcloud import WordCloud

# Machine Learning Models
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression

# Evaluation Metrics
import statsmodels.api as sm
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error  # Regression
from sklearn.metrics import confusion_matrix, accuracy_score  # Classification
```

# Third-party modules
The third-party modules we used include:

**numpy**, **pandas**, **plotly**, **matplotlib**, **seaborn**, **wordcloud**, and **scikit-learn**.

# How To Run
**Note 1:** The output files (e.g. the cleaned data files and the plots) already exist in the repository, so the following steps are simply to describe the process as if they had not already been generated.

**Note 2:** In the meantime, there are two ways to view the full Python code. You can choose to view the different sections of the files and run them step by step. Or choose to run the **Main Analysis.py** to see all the data analysis at one time.

**Step by Step:**

  1. First, navigate to the ECE143-Project directory.

  2. Then Open the file **Data Cleaning**, run **Data cleaning.py**  to create cleaned and combined CSV files. You will get **cleaned_data.csv**.

  3. Run the **Analysis.ipynb** file in Analysis Notebook directory.

     You can find all the plots in the **Visualization** file.

**All at Once:**



# Result, Conlusion and Suggestions
Analysis has been conducted to answer the following questions:

**What are some of the most popular games played by std?**

**What is the relationship between GAD and Satisfaction with Life (SWL)?**
A negative correlation is observed, indicating that higher anxiety levels are associated with lower life satisfaction.

**Does degree level impact Satisfaction with Life (SWL)?**
SWL score does seem to be higher for people witha PhD/Master's degree.

**How does area of residence affect the hours of gaming?**

**How does gaming playstyle relate to Social Phobia Inventory (SPIN) scores?**
Solo gamers tend to have higher SPIN scores, suggesting a possible link between social anxiety and a preference for single-player games.

**What is the distribution of gaming hours across different GAD levels?**
Individuals with higher GAD scores show mixed gaming habits, with no clear trend indicating whether gaming increases or reduces anxiety.

**Do people with higher SPIN scores have different satisfaction with life (SWL)?**
A negative relationship is observed; those with higher social anxiety (SPIN) tend to have lower satisfaction with life.

**How do SPIN, SWL, and GAD scores vary with employmen status**

**How does anxiety and satisfaction with life as we consider indviduals from different age groups**

**How do all these factors correlate with each other? Basically,  where do we focus on for reduces GAD,SPIN and higher SWL scores**
