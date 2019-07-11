""" 
Author      : Thomas Cintra and Yun Zhang
Class       : CS 181R
Date        : 2019 June 20
Description : Credit Score analysis
Name        :
Homework 3
"""

# seaborn module
import seaborn as sns

# python modules
import os

# numpy module
import numpy as np

# pandas module
import pandas as pd

# matplotlib module
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors

# import scikit learn module
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import LabelEncoder

path = os.path.join("..", "Data")

def main():
    print()

    # reads the dataset onto the DataFrame
    df = pd.read_csv(os.path.join(path, 'credit_train.csv'))
    df.drop(df.columns[df.columns.str.contains('unnamed',case = False)],axis = 1, inplace = True)

    # prints some basic information on the DataFrame
    basic_information(df)

    # prints the categorical columns
    data_types(df)

    # prints missing values
    table = missing_values_table(df)
    print(table.head())
    print(table.tail())
    
def basic_information(df):
    print(df.dtypes.value_counts())
    print(df.shape)
    print(df.head())

def missing_values_table(df):
    # Total missing values
    mis_val = df.isnull().sum()
    
    # Percentage of missing values
    mis_val_percent = 100 * df.isnull().sum() / len(df)
    
    # Make a table with the results
    mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
    
    # Rename the columns
    mis_val_table_ren_columns = mis_val_table.rename(
    columns = {0 : 'Missing Values', 1 : '% of Total Values'})
    
    # Sort the table by percentage of missing descending
    mis_val_table_ren_columns = mis_val_table_ren_columns[
        mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(
    '% of Total Values', ascending=False).round(1)
    
    # Print some summary information
    print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"      
        "There are " + str(mis_val_table_ren_columns.shape[0]) +
            " columns that have missing values.")
    
    # Return the dataframe with missing information
    return mis_val_table_ren_columns

def data_types(df):
    # prints how many times each type of data appears on df
    print(df.dtypes.value_counts())

    # prints the amount of unique entries of each categorical column as a pandas series
    categorical_columns = df.select_dtypes('object').apply(pd.Series.nunique, axis = 0)
    print(categorical_columns)

if __name__ == "__main__":
    main()