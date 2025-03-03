# -*- coding: utf-8 -*-
"""prediction-SPIN2

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1j17t8_SloiO_3bwztnmhrojA9GYLBPnF
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

"""# Import"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

"""# Loading Dataset"""

dataset = pd.read_csv('cleaned_data.csv', encoding='latin-1')

df = pd.DataFrame(dataset)
df

"""# Feature Engineering

### Reference: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3545926/
"""

# Feature Variables
df_filtered = df
df_filtered['Fear_Component'] = df_filtered[['SPIN1', 'SPIN2', 'SPIN3', 'SPIN4', 'SPIN5', 'SPIN6']].median(axis=1).round().astype('int64')
df_filtered['Avoidance_Component'] = df_filtered[['SPIN7', 'SPIN8', 'SPIN9', 'SPIN10', 'SPIN11', 'SPIN12', 'SPIN13']].median(axis=1).round().astype('int64')
df_filtered['Physiological_Discomfort_Component'] = df_filtered[['SPIN14', 'SPIN15', 'SPIN16', 'SPIN17']].median(axis=1).round().astype('int64')

dataa = [1,2,3,4,1,2,3,4,1,2,3,4,1,2,3,4,0]
dd=[]

dfs1=dataa[:6]
dfs2=dataa[6:13]
dfs3=dataa[13:17]
d1 = np.array(dfs1)
d2 = np.array(dfs2)
d3 = np.array(dfs3)

print(d1)
print(d2)
print(d3)

dd.append(np.median(d1).round().astype('int64'))
dd.append(np.median(d2).round().astype('int64'))
dd.append(np.median(d3).round().astype('int64'))
# np.median(dfs)

print(dd)

# dfs['Fears']

# Target Variables
df_filtered['Social_Phobia_Level'] = df_filtered[['Fear_Component', 'Avoidance_Component', 'Physiological_Discomfort_Component']].mean(axis=1).round().astype('int64')
# print(df_filtered['Social_Phobia_Level'])

"""# Feature Selection"""

# Features and target variable
features = df_filtered[['Fear_Component', 'Avoidance_Component', 'Physiological_Discomfort_Component']]
print(features)
target = df_filtered['Social_Phobia_Level']

# Listing the number of rows and columns
features.shape, target.shape

print(target)

"""# Data Splitting"""

from sklearn.model_selection import train_test_split, cross_val_score

# Splitting data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Check the shape of x_train, x_test, y_train, y_test
x_train.shape,x_test.shape,y_train.shape,y_test.shape

"""# Model Implementation

## 1. Random Forest classifier
"""

# Import Library
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,precision_score,recall_score, f1_score, confusion_matrix


# Initialize the Random Forest classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the Random forest classifier
rf_classifier.fit(x_train, y_train)

# Accuracy Score on the training data
x_train_prediction = rf_classifier.predict(x_train)
training_data_accuracy = accuracy_score(y_train, x_train_prediction)
print('Train data Accuracy: ',training_data_accuracy)
# print('Test data Accuracy ', rf_classifier.score(X_test,y_test))

# Accuracy score on the test data
x_test_prediction = rf_classifier.predict(x_test)
test_data_accuracy = accuracy_score(y_test,x_test_prediction)
print('Test data Accuracy:',test_data_accuracy)

score=cross_val_score(rf_classifier,features,target,cv=10)
print('Cross_val_score :',np.mean(score))

#classifcation report
from sklearn.metrics import classification_report
print(classification_report(x_test_prediction,y_test))

"""## 2. Logistic Regression"""

# Accuracy
accuracy = accuracy_score(y_test, x_test_prediction)
print(f"Accuracy: {accuracy}")

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, x_test_prediction)
print("Confusion Matrix:")
print(conf_matrix)

# Precision, Recall, F1-score
precision = precision_score(y_test, x_test_prediction, average='weighted')
recall = recall_score(y_test, x_test_prediction, average='weighted')
f1 = f1_score(y_test, x_test_prediction, average='weighted')

print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1-score: {f1}")

from sklearn.linear_model import LogisticRegression
logis_reg = LogisticRegression(random_state=42)

# Training the Logistic Regression
logis_reg.fit(x_train, y_train)

# Accuracy Score on the training data
x_train_prediction = logis_reg.predict(x_train)
training_data_accuracy = accuracy_score(y_train, x_train_prediction)
print('Train data Accuracy: ',training_data_accuracy)


# Accuracy score on the test data
x_test_prediction = logis_reg.predict(x_test)
test_data_accuracy = accuracy_score(y_test,x_test_prediction)
print('Test data Accuracy:',test_data_accuracy)

# Cross Validation Score
score=cross_val_score(logis_reg,features,target,cv=10)
print('Cross_val_score :',np.mean(score))

#classification report Awo the test_data or test_target_value
print('\n Classification Report \n', classification_report(y_test,x_test_prediction))
print(confusion_matrix(y_test, x_test_prediction))
sns.heatmap(confusion_matrix(y_test, x_test_prediction), annot=True, cmap='Greys', fmt='g')

# print("\n AUC Score", roc_auc_score(y_test,x_test_prediction))

"""## K-NN Classifier"""

from sklearn.neighbors import KNeighborsClassifier
KNN_clls = KNeighborsClassifier(n_neighbors=4)

#Train the knn for the data
KNN_clls.fit(x_train, y_train)

# Accuracy Score on the training data
x_train_prediction = KNN_clls.predict(x_train)
training_data_accuracy = accuracy_score(y_train, x_train_prediction)
print('Train data Accuracy: ',training_data_accuracy)


# Accuracy score on the test data
x_test_prediction = KNN_clls.predict(x_test)
test_data_accuracy = accuracy_score(y_test,x_test_prediction)
print('Test data Accuracy:',test_data_accuracy)

# Cross Validation Score
score=cross_val_score(KNN_clls,features,target,cv=10)
print('Cross_val_score :',np.mean(score))

#classification report Awo the test_data or test_target_value
print('\n Classification Report \n', classification_report(x_test_prediction, y_test))
cm=confusion_matrix(y_test,x_test_prediction)
print(cm)

"""# Prediction"""

data = [
    [2, 3, 2],  # Example values for Fear_Component, Avoidance_Component, Physiological_Discomfort_Component
]
df_val = pd.DataFrame(data, columns=['Fear_Component', 'Avoidance_Component', 'Physiological_Discomfort_Component'])



# Making predictions on data
df_predictions = rf_classifier.predict(df_val)

# Create a DataFrame to display the data and predictions
df_with_predictions = df_val.copy()
df_with_predictions['Predicted_Social_Phobia_Level'] = df_predictions

# Display sample data with predictions
print("Data with Predictions:")
print(df_with_predictions)

def take_fixed_options_mcq_quiz(questions):
    # Fixed set of options for all questions
    options = ["Not At All", "A Little Bit", "Somewhat", "Very Much", "Extremely"]

    # Initialize a list to store user responses
    user_responses = []

    # Display questions and get user responses
    for i, question in enumerate(questions, start=1):
        print(f"\nQuestion {i}: {question}")
        for j, option in enumerate(options, start=1):
            print(f"{j}. {option}")


        # Get user input for the selected option with exception handling
        while True:
            try:
                user_input = int(input("Your answer (enter the option number): "))

                # Validate user input
                if 1 <= user_input <= len(options):
                    break
                else:
                    print("Invalid input. Please enter a valid option number.")
            except ValueError:
                print("Invalid input. Please enter a valid integer.")

        # Store the user response
        user_responses.append(user_input)

    # Display user responses
    print("\nYour selected answers:")
    for i, response in enumerate(user_responses, start=1):
        print(f"Question {i}: {options[response - 1]}")

    print(user_responses)
    dd=[]

    dfs1=user_responses[:6]
    dfs2=user_responses[6:13]
    dfs3=user_responses[13:17]
    d1 = np.array(dfs1)
    d2 = np.array(dfs2)
    d3 = np.array(dfs3)

    print(d1)
    print(d2)
    print(d3)
    print("Median of Fear-Avoidance-Physiological columns")
    dd.append(np.median(d1).round().astype('int64'))
    dd.append(np.median(d2).round().astype('int64'))
    dd.append(np.median(d3).round().astype('int64'))
    # np.median(dfs)

    print(dd)
    ddd=[dd]
    # Create the numpy array
    symptom = np.array(["none","Mild","Moderate", "High","Extreme"])

    df_val = pd.DataFrame(ddd, columns=['Fear_Component', 'Avoidance_Component', 'Physiological_Discomfort_Component'])
    # Making predictions on data
    df_predictions = rf_classifier.predict(df_val)
    print(symptom[df_predictions][0])


# Example questions (You can customize these questions)
quiz_questions = [
    "I am afraid of people in authority.",
    "I am bothered by blushing in front of people.",
    "Parties and social events scare me.",
    "I avoid talking to people I don’t know.",
    "Being criticized scares me a lot.",
    "I avoid doing things or speaking to people for fear of embarrassment.",
    "Sweating in front of people causes me distress.",
    "I avoid going to parties.",
    "I avoid activities in which I am the center of attention.",
    "Talking to strangers scares me.",
    "I avoid having to give speeches.",
    " I would do anything to avoid being criticized.",
    "Heart palpitations bother me when I am around people.",
    "I am afraid of doing things when people might be watching.",
    "Being embarrassed or looking stupid are among my worst fears.",
    "I avoid speaking to anyone in authority.",
    "Trembling or shaking in front of others is distressing to me.",
    # Add more questions as needed
]

# Call the function with the example questions
take_fixed_options_mcq_quiz(quiz_questions)

def bharat(data):
    # Create the numpy array
    symptom = np.array(["none","Mild","Moderate", "High","Extreme"])

    df_val = pd.DataFrame(data, columns=['Fear_Component', 'Avoidance_Component', 'Physiological_Discomfort_Component'])
    # Making predictions on data
    df_predictions = rf_classifier.predict(df_val)
#     print(df_predictions)
#     print(type(df_predictions))
    print(symptom[df_predictions][0])
    # Create a DataFrame to display the data and predictions
#     df_with_predictions = df_val.copy()
#     df_with_predictions['Predicted_Social_Phobia_Level'] = df_predictions
#     print(df_with_predictions)
#     print(symptom[df_with_predictions])
#     print(type(df_with_predictions))


d=[[2,2,2]]

bharat(d)

"""# Outcome     Symptom Levels
    0       None
    1       Mild
    2       Moderate
    3       High
    4       extreme
"""

# df_filtered[4]
# dataframe[['column1','column2']]

df_filtered=df_filtered[["Fear_Component","Avoidance_Component","Physiological_Discomfort_Component","Social_Phobia_Level"]]
# df_filtered.to_dict()

df_filtered.head(10)

#Creating a pickle file
import pickle

pickle.dump(rf_classifier, open("social_phobia_rf.pkl","wb"))

# To read
# model_loaded=pickle.load(open("social_phobia_rf.pkl","rb"))