""" 
Author      : Thomas Cintra and Yun Zhang
Class       : CS 181R
Date        : 2019 June 20
Description : Credit Score analysis
Name        :
Homework 3
"""

# python modules
import os

# numpy module
import numpy as np

# pandas module
import pandas as pd

# matplotlib module
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors

# sklearn module
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

# SMOTE module
from imblearn.over_sampling import SMOTENC

# read csv file located in your desktop
path = os.path.join(os.path.join(os.path.expanduser('~')), 'Desktop') 
df = pd.read_csv(os.path.join(path, 'application_train.csv'))

# sample the DataFrame
sample = df.sample(n = 200000)

# drop columns with too many missingvalues
mis_val_percent = 100 * sample.isnull().sum() / len(sample)
empty_cols = []
for i in range(len(mis_val_percent)):
    if mis_val_percent[i] > 40:
        empty_cols.append(mis_val_percent.keys()[i])

sample.drop(empty_cols, axis = 1, inplace = True) 

# Create a label encoder object
le = LabelEncoder()
le_count = 0

encoded_sample = sample
integer_encoded_cols = []
count = 0

# Iterate through the columns
for col in encoded_sample:
    if encoded_sample[col].dtype == 'object':
        # If 2 or fewer unique categories
        if len(list(encoded_sample[col].unique())) <= 2:
            # Train on the training data
            le.fit(encoded_sample[col])
            # Transform both training and testing data
            encoded_sample[col] = le.transform(encoded_sample[col])
            
            # Keep track of how many columns were label encoded
            le_count += 1
            integer_encoded_cols.append(encoded_sample.columns[count])
    count += 1
            
print('%d columns were label encoded.' % le_count)

# Get all categorical cols
one_hot_cols = encoded_sample.select_dtypes('object').apply(pd.Series.nunique, axis = 0)
for i in range(len(one_hot_cols)):
    integer_encoded_cols.append(one_hot_cols.keys()[i])

categorical_cols = integer_encoded_cols

# Get all continuous cols
all_cols = list(encoded_sample.keys())
for x in categorical_cols:
    all_cols.remove(x)
continuous_cols = all_cols

# one-hot encoding of categorical variables
encoded_sample = pd.get_dummies(sample)

# replace anomalies
encoded_sample['DAYS_EMPLOYED'].replace({365243: np.nan}, inplace = True)

# imputing
imputer = SimpleImputer(strategy='most_frequent')
imputed_sample = pd.DataFrame(imputer.fit_transform(encoded_sample))
imputed_sample.columns = encoded_sample.columns
imputed_sample.index = encoded_sample.index

# find all categorical columns, ie. the columns that were either label or one hot encoded
all_encoded_cols = list(imputed_sample.keys())
for x in continuous_cols:
    all_encoded_cols.remove(x)
encoded_categorical_cols =  all_encoded_cols
categorical_index = []
for x in encoded_categorical_cols:
    categorical_index.append(list(imputed_sample.keys()).index(x))

# index array for SMOTENC oversampling
index_array = np.asarray(categorical_index) - 1

# train test split
x_train, x_test, y_train, y_test = train_test_split(imputed_sample.drop(['TARGET'], axis = 1), imputed_sample['TARGET'], test_size = 0.2, random_state = 42)
y_train, y_test = y_train.astype('int'), y_test.astype('int') 

# oversampling
sm = SMOTENC(categorical_features = index_array, random_state = 42, sampling_strategy = 1)
x_train_res, y_train_res = sm.fit_resample(x_train, y_train.ravel())

# resampling
# homework flow
# Accuracy
# not much missing values anymore
# automatic vs manual feature engineering
# how much machine learning should they do