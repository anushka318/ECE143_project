# ECE143 Final Project Group 17 (Winter 2025)

# Student Anxiety Analysis and Prediction System

# Team Members:
Anushka Chaudhary: anchaudhary@ucsd.edu

Letong Wang: lew030@ucsd.edu

Jinbo Ma: jim031@ucsd.edu

Zhengyang Zhou: zhz179@ucsd.edu

# Problem:
This project analyzes student anxiety data to reveal underlying patterns and key triggers.

# Dataset:
Main dataset (https://www.kaggle.com/datasets/petalme/student-anxiety-dataset)

The entire dataset consists of one csv file containing various information related to student anxiety, including factors contributing to anxiety, its manifestations, and potential mitigating influences. Through this dataset, we can gain deep insights into students' mental health, stress levels, academic pressures, coping strategies, and other related attributes.

Secondary dataset (https://www.kaggle.com/datasets/rxnach/student-stress-factors-a-comprehensive-analysis)

This dataset contains around 20 features that create the most impact on the Stress of a Student. The features are selected scientifically considering 5 major factors, they are Psychological, Physiological, Social, Environmental, and Academic Factors.

# Proposed Solution and Real-World Application:
Our proposed solution focuses on utilizing data preprocessing and exploratory data analysis to study student anxiety, structuring the project into two main phases: data cleaning and data analysis with visualization. 

The primary objective is to develop interactive visualizations and dashboards that highlight patterns, trends, and correlations, such as common anxiety triggers and the effectiveness of various coping strategies. 

The real-world applications of this project include early intervention, where educators and mental health professionals can leverage these insights to identify high-risk students and implement timely support measures; policy-making, which involves analyzing key contributing factors to student anxiety to inform and refine school policies and mental health programs; and resource allocation, ensuring that mental health resources and support systems are distributed optimally based on data-driven insights.

# File Structure:
```
📂 Analysis_notebook
  └── 📄 analysis.ipynb
📂 Data Cleaning
  ├── 📄 Data Cleaning for Secondary Dataset.py
          ⋮
  └── 📄 Data Cleaning.py
📂 Data
  ├── 📄 StressLevelDataset.csv
  ├── 📄 anxiety.csv
  └── 📄 cleaned_data.csv
📂 Indvidual_analysis
  ├── 📂 Data Analyse of Letong
    ├── 📂 data analysis
      ├── 📄 data analysis.ipynb
      └── 📄 data analysis.py
    ├── 📂 data cleaning
      ├── 📄 data cleaning.ipynb
      ├── 📄 data cleaning.py
      └── 📄 cleaned_data.csv
    ├── 📂 prediction
      ├── 📄 prediction_GAD.ipynb
              ⋮
      └── 📄 prediction_swl.py
  ├── 📂 Data Analyse of Zhengyang
    ├── 📂 Visualization
      ├── 📄 Age Distribution.png
              ⋮
      └── 📄 Top Games by Count.png
    ├── 📄 Data Analyse by Zhengyang.ipynb
    └── 📄 Data Analyse by Zhengyang.py
  ├── 📂 Pridiction
    ├── 📄 Correlation between SPIN_T, GAD_T and SWL_T.py
            ⋮
    └──📄 SWL_T.py
  ├── 📂 images
    ├── 📄anxiety_age_employment.png
    └── 📄anxiety_satisfaction.png
├── 📂 Visualization
    ├── 📄 Age.png
            ⋮
    └── 📄 WordCloud.png
├── 📂 module
    ├── 📄 categorize_data.py
    └── 📄 visualization.py
├── 📄 .gitignore
└── 📄 README.md
```

**1. Data:** It contains the original dataset (**anxiety.csv**), the output dataset (**cleaned_data.csv**) after the cleaning is completed and the secondary dataset (**StressLevelDataset.csv**).

**2. Data Cleaning:** It is mainly used for data cleaning, which contains a PDF file with a detailed explanation of the data cleaning process and a dataset (cleaned_data) output after the cleaning is completed.

**3. Indvidual Analysis:** This mainly includes the extended data analysis independently conducted by Zhengyang Zhou and Letong Wang, relating to getting familiar with the data and finding correlations bewteen factors. It includes prediction and influencing factor analysis of GAD_T, SPIN_T and SWL_T (PDF with explanations), analysis of the interaction of GAD_T, SPIN_T and SWL_T (PDF with explanations), and analysis of the impact of GAD_T, SPIN_T and SWL_T against gaming behaviors, age, gender and earnings (PDF with explanations).

**4. module:** This has the python modules used for visualization, and categorizing data.

**5. Analysis notebook:** The notebook with final analysis. The same analysis file is present in .ipynb and .py formats.

**6. Visualization:** This has the images of all the visualizations from the data analysis.

# Main Analysis Notebooks:
**Attention: All python files are stored in both .py and .ipynb forms with the same file name.**

**1. Data cleaning.py /.ipynb:** In this data processing task, we systematically cleaned the raw data by identifying and removing inconsistencies, filling in missing values using appropriate imputation methods and standardizing the data to maintain uniformity across all variables. 

**2. Analysis.ipynb:** This notebook explores the relationships between various factors in a survey dataset. The analysis examines how gaming habits and personal characteristics correlate with mental health indicators such as Generalized Anxiety Disorder (GAD) scores, Satisfaction with Life (SWL), and Social Phobia Inventory (SPIN) scores. 

# Setup:
Install the required dependencies:

```python
# Standard Libraries
import os
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly as pl

# Data Preprocessing & Transformation
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

# Machine Learning Models
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

# Model Evaluation & Statistical Analysis
import statsmodels.api as sm
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error  # Regression
from sklearn.metrics import confusion_matrix, accuracy_score  # Classification

# Visualization Tools
from wordcloud import WordCloud
```

# Third-party modules:
The third-party modules we used include:

**numpy**, **pandas**, **plotly**, **matplotlib**, **seaborn**, **wordcloud**, and **scikit-learn**.

# How To Run:
**Note 1:** The output files (e.g. the cleaned data files and the plots) already exist in the repository, so the following steps are simply to describe the process as if they had not already been generated.

**Note 2:** You can choose to view the different sections of the Analysis.ipynb file and run them step by step. 
  1. First, navigate to the **ECE143-Project** directory.

  2. Then Open the file **Data Cleaning**, run **Data cleaning.py**  to create cleaned and combined CSV files. You will get **cleaned_data.csv**.

  3. Open the file **Analysis_notebook**, run the **Analysis.ipynb** file in Analysis Notebook directory.

     You can find all the plots in the **Visualization** file.

# Questions addressed:
Analysis has been conducted to answer the following questions:

**1. What are some of the most popular games played by stdudents with higher anxiety disorder?**

**2. What is the relationship between GAD and Satisfaction with Life (SWL)?**

**3. Does degree level impact Satisfaction with Life (SWL)?**

**4. How does area of residence affect the hours of gaming?**

**5. How does gaming playstyle relate to Social Phobia Inventory (SPIN) scores?**

**7. Do people with higher SPIN scores have different satisfaction with life (SWL)?**

**8. How do SPIN, SWL, and GAD scores vary with employmen status?**

**9. How does anxiety and satisfaction with life as we consider indviduals from different age groups?**

**10. How do all these factors correlate with each other?**

**11. How do anxiety scores relate with academic performance and social support?**

**12. How do health and academic factors correlate with mental health issues among studnets?**

**13. What is the relative importance of features to see how they contribute to anxiety levels?**

# Conclusion and Suggestions: 

### **Conclusion**
The analysis highlights that student anxiety is influenced by multiple factors, including academic performance, social support, environmental conditions, and personal habits. Key findings show that **higher anxiety levels correlate with lower life satisfaction**, and factors such as **sleep quality, teacher-student relationships, and future career concerns** play a critical role in determining stress levels. Additionally, **noisy environments, lack of safety, and unmet basic needs** significantly contribute to increased anxiety among students. 

### **Suggestions to Reduce Stress and Anxiety Among Students**
 1. Improve Academic Support Systems
2. Enhance Mental Health Resources
3. Address Environmental and Lifestyle Factors
4. Promote Healthy Social Habits
5. Manage Academic and Career Pressures
