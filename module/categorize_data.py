import csv
import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt 
import os

## --------------------------------------------##
## The following functions help in categorizing the data for ease of analysis.##
##----------------------------------------------##
def anxiety_level(gad_score):
    '''
    Interpreting the GAD-7 score and categorizing as follows:
    0-4: Minimal anxiety
    5-9: Mild anxiety
    10-14: Moderate anxiety
    15-21: Severe anxiety
    Parameters:
    GAD score : Integer
    '''
    assert isinstance(gad_score, int)
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
    Parameter
    spin_score : Integer
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
    Parameter
    swl_score : Integer
    '''
    assert isinstance(swl_score, int)
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
    Parameter:
    status : String
    '''
    assert isinstance(status, str)
    if status == 'Part Time' or status == 'Full Time':
        return 'Employed'
    else:
        return 'Unemployed'

def categorize_age(age):
    '''
    Categorizing age as follows:
    <25 : Youth
    25 < age < 40 : Middle aged
    Parameter:
    age : Integer
    '''
    assert isinstance(age, int)
    if age < 25:
        return 'Youth'
    elif age < 40:
        return 'Middle aged'
    else:
        return 'Old'
    
def categorize_playstyles(playstyle):
    '''
    Categorizing playstyles as follows:
    singleplayer: S
    multiplayer  online  with stranger: M_0
    multiplayer  online  with online acquaintances or teammates : M_1
    multiplayer  online  with real life friends :M_2
    Parameter:
    playstyle : String
    '''
    assert isinstance(playstyle, str)
    if playstyle == 'singleplayer':
        return 'S'
    elif playstyle == 'multiplayer  online  with strangers':
        return 'M_0'
    elif playstyle == 'multiplayer  online  with online acquaintances or teammates':
        return 'M_1'
    elif playstyle == 'multiplayer  online  with real life friends':
        return 'M_2'
    
def categorize_data(df):
    '''
    Categorizing the data based on the following columns:
    df : Input dataframe
    '''
    df['Anxiety_level'] = df['GAD_T'].apply(anxiety_level)
    df['Social_phobia'] = df['SPIN_T'].apply(social_phobia_inventory) 
    df['SWL'] = df['SWL_T'].apply(swl)
    df['Employment'] = df['Work'].apply(categorize_employment)
    df['Age_groups'] = df['Age'].apply(categorize_age)
    df['Play'] = df['Playstyle'].apply(categorize_playstyles)
    return df
