import matplotlib.pyplot as plt
import os
import seaborn as sns
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
## -----------------------------------------------------------##
## The following file has functions to plot the visualizations ##
## ------------------------------------------------------------##
def plot_boxplot(data, x_col, y_col, filter_values=None, filter_col=None, showfliers=False, palette="coolwarm", title=None):
    '''
    Function to plot boxplots
    Parameters:
    data: to plot the data
    x_col : X axis column
    y_col ; Y axis column
    filter_col : filtering the column data
    showfliers : outliers displayed or not
    palette : colours in plot
    title : to title the plot
    '''
    if filter_values is not None and filter_col is not None:
        data = data[data[filter_col].isin(filter_values)]
    
    plt.figure(figsize=(12,10))
    sns.boxplot(data=data, x=x_col, y=y_col, showfliers=showfliers, palette=palette)
    
    plt.title(title if title else f"Plot of {x_col} by {y_col}")
    plt.xlabel(x_col.replace('_', ' ').title())
    plt.ylabel(y_col.replace('_', ' ').title())
    plt.show()


def plot_barplot(x, y, data, title, xlabel, ylabel):
    '''
    Plotting a barplot for the given data
    Parameters:
    x : Data to be plotted on X axis
    y : Data to be plotted on Y axis
    title : Title of the plot
    xlabel: Label on X axis
    ylabel : Label on Y axis
    '''
    plt.figure(figsize=(10, 6))
    sns.barplot(x=x, y=y, data=data)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()

def plot_scatter(x, y, data, title, xlabel, ylabel):
    '''
    Plotting a scatterplot for the given data
    x : Data to be plotted on X axis
    y : Data to be plotted on Y axis
    title : Title of the plot
    xlabel: Label on X axis
    ylabel : Label on Y axis    
    '''
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=data[x], y=data[y])
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)    
    plt.show()

def plot_heatmap(data, title):
    '''
    Plotting a heatmap for the given data
    data : Data to be plotted as a heatmap
    title : Title of the plot
    '''
    plt.figure(figsize=(10, 6))
    sns.heatmap(data, annot=True)
    plt.title(title)
    plt.show()

def plot_countplot(x, data, title, xlabel):
    '''
    Plotting a countplot for the given data
    x : Column for which count is to be plotted
    data : Data to be plotted 
    title : Title of the plot
    xlabel: Label on X axis
    ''' 
    plt.figure(figsize=(10, 6))
    sns.countplot(x=x, data=data)
    plt.title(title)
    plt.xlabel(xlabel) 
    plt.show()

def plot_piechart(labels, values, title):
    '''
    Plots a pie chart with the given labels and values.
    Parameters:
    labels :  Labels on the plot
    values : Values for the piechart
    title :
    '''

    colors = plt.cm.Set2(np.linspace(0, 1, len(values)))  # Generate colors dynamically
    plt.figure(figsize=(10, 8)) 
    wedges, texts, autotexts = plt.pie(
        values, 
        labels=labels, 
        autopct='%1.1f%%',
        colors=colors, 
        startangle=140,  
        wedgeprops={'edgecolor': 'black', 'linewidth': 1},
        pctdistance=0.9,
        labeldistance=1.1 
    )
    # Customize text inside pie chart
    for autotext in autotexts:
        autotext.set_fontsize(8)
        autotext.set_color('black')
    # Add legend
    plt.legend(wedges, labels, title="Categories", loc="center left", bbox_to_anchor=(1, 0.5))
    plt.title(title)
    plt.show()
