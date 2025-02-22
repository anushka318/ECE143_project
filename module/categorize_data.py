import csv
import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt 
import os

def anxiety_level(gad_score):
    '''
    Interpreting the GAD-7 score and categorizing as follows:
    0-4: Minimal anxiety
    5-9: Mild anxiety
    10-14: Moderate anxiety
    15-21: Severe anxiety
    '''
    if gad_score <= 4:
        return 'Minimal anxiety'   
    elif gad_score <= 9:    
        return 'Mild anxiety'       
    elif gad_score <= 14:
        return 'Moderate anxiety'
    else:
        return 'Severe anxiety'

def social_phobia_inventory(spin_score):
    '''
    Interpreting the SPIN score and categorizing as follows:
    0–20 : No or mild social anxiety
    21–30: Moderate social anxiety
    31–40: Severe social anxiety
    41–68: Very severe social anxiety
    '''
    if spin_score <= 20:
        return 'No or mild social anxiety'
    elif spin_score <= 30:
        return 'Moderate social anxiety'
    elif spin_score <= 40:
        return 'Severe social anxiety'
    else:
        return 'Very severe social anxiety'
    
def swl(swl_score):
    '''
    Interpreting the SWL score and categorizing as follows:
    5–9: Extremely dissatisfied
    10–14: Dissatisfied
    15–19: Slightly dissatisfied
    20–24: Neutral
    25–29: Slightly satisfied
    30–34: Satisfied
    '''
    if swl_score <= 9:
        return 'Extremely dissatisfied'
    elif swl_score <= 14:
        return 'Dissatisfied'
    elif swl_score <= 19:
        return 'Slightly dissatisfied'
    elif swl_score <= 24:
        return 'Neutral'
    else:
        return 'Satisfied'
    
def categorize_employment(status):
    '''
    Categorizing employment status as follows:
    Part Time/ Full Time : Employed
    Else : Unemployed
    '''
    if status == 'Part Time' or status == 'Full Time':
        return 'Employed'
    else:
        return 'Unemployed'

def categorize_age(age):
    '''
    Categorizing age as follows:
    <25 : Youth
    25 < age < 40 : Middle aged
    '''
    if age < 25:
        return 'Youth'
    elif age < 40:
        return 'Middle aged'
    else:
        return 'Old'
    
def categorize_data(df):
    '''
    Categorizing the data based on the following columns:
    '''
    df['Anxiety_level'] = df['GAD_T'].apply(anxiety_level)
    df['Social_phobia'] = df['SPIN_T'].apply(social_phobia_inventory) 
    df['SWL'] = df['SWL_T'].apply(swl)
    df['Employment'] = df['Work'].apply(categorize_employment)
    df['Age'] = df['Age'].apply(categorize_age)
    return df
