import matplotlib.pyplot as plt
import os
import seaborn as sns

def plot_boxplot(x, y, data, title, xlabel, ylabel, save_as):
    '''
    Plotting a boxplot for the given data
    '''
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=x, y=y, data=data)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(os.path.join('images', save_as))
    plt.show()

def plot_barplot(x, y, data, title, xlabel, ylabel, save_as):
    '''
    Plotting a barplot for the given data
    '''
    plt.figure(figsize=(10, 6))
    sns.barplot(x=x, y=y, data=data)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(os.path.join('images', save_as))
    plt.show()

def plot_scatter(x, y, data, title, xlabel, ylabel, save_as):
    '''
    Plotting a scatterplot for the given data
    '''
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=data[x], y=data[y])
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)    
    if not os.path.exists('images'):
        os.makedirs('images')
    plt.savefig(os.path.join('images', save_as))
    plt.show()

def plot_heatmap(data, title, save_as):
    '''
    Plotting a heatmap for the given data
    '''
    plt.figure(figsize=(10, 6))
    sns.heatmap(data, annot=True)
    plt.title(title)
    plt.savefig(os.path.join('images', save_as))
    plt.show()

def plot_countplot(x, data, title, xlabel, save_as):
    '''
    Plotting a countplot for the given data
    ''' 
    plt.figure(figsize=(10, 6))
    sns.countplot(x=x, data=data)
    plt.title(title)
    plt.xlabel(xlabel) 
    plt.savefig(os.path.join('images', save_as))
    plt.show()
