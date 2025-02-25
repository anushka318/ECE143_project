# ECE143_project

# Problem:
This project analyzes student anxiety data to reveal underlying patterns and key triggers.

# Dataset:
Kaggle open dataset (https://www.kaggle.com/datasets/petalme/student-anxiety-dataset)
The entire dataset consists of 1csv file containing various information related to student anxiety, including factors contributing to anxiety, its manifestations, and potential mitigating influences. Through this dataset, we can gain deep insights into students' mental health, stress levels, academic pressures, coping strategies, and other related attributes.

# Proposed Solution and Real-World Application:
Our proposed solution focuses on utilizing data preprocessing and exploratory data analysis to study student anxiety, structuring the project into two main phases: data cleaning and data analysis with visualization. The primary objective is to develop interactive visualizations and dashboards that highlight patterns, trends, and correlations, such as common anxiety triggers and the effectiveness of various coping strategies. The real-world applications of this project include early intervention, where educators and mental health professionals can leverage these insights to identify high-risk students and implement timely support measures; policy-making, which involves analyzing key contributing factors to student anxiety to inform and refine school policies and mental health programs; and resource allocation, ensuring that mental health resources and support systems are distributed optimally based on data-driven insights.

# File Structure / Running each file with main method to get data

# Main Analysis Notebooks
Attention: All python files are stored in both .py and .ipynb forms with the same file name.

Data cleaning.py /.ipynb: In this data processing task, we systematically cleaned the raw data by identifying and removing inconsistencies, filling in missing values using appropriate imputation methods, and standardizing the data to maintain uniformity across all variables. 

GAD_T.py /.ipynb: This analysis uses RandomForestRegressor to predict GAD_T scores and identify key influencing factors. It preprocesses data, trains the model, makes predictions, and analyzes feature importance.

SPIN_T.py /.ipynb: This analysis uses RandomForestRegressor to predict social anxiety scores (SPIN_T) and identify key influencing factors. It preprocesses the data, trains the model, makes predictions, and analyzes feature importance, highlighting the top factors affecting SPIN_T.

SWL_T.py /.ipynb: This analysis uses RandomForestRegressor to predict life satisfaction scores (SWL_T) and analyze key influencing factors. It preprocesses the data, trains the model, makes predictions, and evaluates feature importance, highlighting the top variables affecting SWL_T.

Correlation between SPIN_T, GAD_T and SWL_T.py /.ipynb: This code analyzes the correlation between social anxiety (SPIN_T), generalized anxiety (GAD_T), and life satisfaction (SWL_T). It calculates and prints the correlation matrix, visualizes it using a heatmap, and generates pair plots to explore relationships between these variables.

Correlations between mental health scores and gaming behaviors, age, gender and earnings.py /.ipynb: This code analyzes the relationships between mental health scores (GAD_T, SWL_T, SPIN_T), gaming behaviors, and demographic factors. It trains RandomForestRegressor models to predict these scores, generates a random sample for prediction, and evaluates feature correlations. Additionally, it performs multiple linear regression to assess the impact of gaming and demographics on mental health, visualizes gender-based distributions, and explores trends using scatter plots.

# Setup
Install the required dependencies:

import os
import numpy as np
import plotly as pl
import string as st
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier
from sklearn.linear_model import LinearRegression,Ridge,LogisticRegression
from sklearn.metrics import r2_score, mean_squared_error,confusion_matrix,accuracy_score,mean_absolute_error

import warnings
warnings.filterwarnings("ignore")
