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
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler, Imputer
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score

# Read path
path = os.path.join("..", "Data")

# Training data
app_train = pd.read_csv(os.path.join(path, 'credit_train.csv'))

# Testing data features
app_test = pd.read_csv(os.path.join(path, 'credit_test.csv'))

# -------------------------- ENCODING -------------------------- #

# Create a label encoder object
le = LabelEncoder()
le_count = 0

# Iterate through the columns
for col in app_train:
    if app_train[col].dtype == 'object':
        # If 2 or fewer unique categories
        if len(list(app_train[col].unique())) <= 2:
            # Train on the training data
            le.fit(app_train[col])
            # Transform both training and testing data
            app_train[col] = le.transform(app_train[col])
            app_test[col] = le.transform(app_test[col])
            
            # Keep track of how many columns were label encoded
            le_count += 1
            
print('%d columns were label encoded.' % le_count)

# one-hot encoding of categorical variables
app_train = pd.get_dummies(app_train)
app_test = pd.get_dummies(app_test)

print('Training Features shape: ', app_train.shape)
print('Testing Features shape: ', app_test.shape)

# -------------------------- ENCODING -------------------------- #


# -------------------------- ANOMALIES -------------------------- #

# replace anomalies
app_train['DAYS_EMPLOYED'].replace({365243: np.nan}, inplace = True)
app_test["DAYS_EMPLOYED"].replace({365243: np.nan}, inplace = True)

# -------------------------- ANOMALIES -------------------------- #


# -------------------------- ALIGNMENT -------------------------- #

train_labels = app_train['TARGET']

# Align the training and testing data, keep only columns present in both dataframes
app_train, app_test = app_train.align(app_test, join = 'inner', axis = 1)

# Add the target back in
app_train['TARGET'] = train_labels

print('Training Features shape after alignment: ', app_train.shape)
print('Testing Features shape after alignment: ', app_test.shape)

# -------------------------- ALIGNMENT -------------------------- #


# -------------------------- RANDOM FOREST -------------------------- #

# Drop the target from the training data and testing data
if 'TARGET' in app_train:
    train = app_train.drop(columns = ['TARGET'])
else:
    train = app_train.copy()

if 'TARGET' in app_test:
    y_true = app_test['TARGET']
    test = app_test.drop(columns = ['TARGET'])
else:
    test = app_test.copy()

# Feature names
features = list(train.columns)

# Median imputation of missing values
imputer = Imputer(strategy = 'median')

# Scale each feature to 0-1
scaler = MinMaxScaler(feature_range = (0, 1))

# Fit on the training data
imputer.fit(train)

# Transform both training and testing data
train = imputer.transform(train)
test = imputer.transform(test)

# Repeat with the scaler
scaler.fit(train)
train = scaler.transform(train)
test = scaler.transform(test)

print('Training data shape: ', train.shape)
print('Testing data shape: ', test.shape)

# Make the random forest classifier
random_forest = RandomForestClassifier(n_estimators = 100, random_state = 50, verbose = 1, n_jobs = -1, class_weight = "balanced")

# Train on the training data
random_forest.fit(train, train_labels)

# Extract feature importances
feature_importance_values = random_forest.feature_importances_
feature_importances = pd.DataFrame({'feature': features, 'importance': feature_importance_values})

# Make predictions on the test data
predictions = random_forest.predict(test)

# Make a submission dataframe
submit = app_test[['SK_ID_CURR']]
submit['TARGET'] = predictions

print(len(submit.loc[submit['TARGET'] == 1]))

y_pred = predictions
print(classification_report(y_true, y_pred))

print(roc_auc_score(y_true, y_pred))

# -------------------------- RANDOM FOREST -------------------------- #


def sqf(multiclass = False):
    """Load sqf dataset"""

    target_column = "TARGET"

    target_names = ["NO DEFAULT", "DEFAULT"]
    labels = [0, 1]

    # Read path
    path = os.path.join("..", "Data")

    # Training data
    app_train = pd.read_csv(os.path.join(path, 'credit_train.csv'))

    # Testing data features
    app_test = pd.read_csv(os.path.join(path, 'credit_test.csv'))

    # -------------------------- ENCODING -------------------------- #

    # Create a label encoder object
    le = LabelEncoder()
    le_count = 0

    # Iterate through the columns
    for col in app_train:
        if app_train[col].dtype == 'object':
            # If 2 or fewer unique categories
            if len(list(app_train[col].unique())) <= 2:
                # Train on the training data
                le.fit(app_train[col])
                # Transform both training and testing data
                app_train[col] = le.transform(app_train[col])
                app_test[col] = le.transform(app_test[col])
                
                # Keep track of how many columns were label encoded
                le_count += 1
                
    print('%d columns were label encoded.' % le_count)

    # one-hot encoding of categorical variables
    app_train = pd.get_dummies(app_train)
    app_test = pd.get_dummies(app_test)

    print('Training Features shape: ', app_train.shape)
    print('Testing Features shape: ', app_test.shape)

    # -------------------------- ENCODING -------------------------- #


    # -------------------------- ANOMALIES -------------------------- #

    # replace anomalies
    app_train['DAYS_EMPLOYED'].replace({365243: np.nan}, inplace = True)
    app_test["DAYS_EMPLOYED"].replace({365243: np.nan}, inplace = True)

    # -------------------------- ANOMALIES -------------------------- #


    # -------------------------- ALIGNMENT -------------------------- #

    train_labels = app_train['TARGET']

    # Align the training and testing data, keep only columns present in both dataframes
    app_train, app_test = app_train.align(app_test, join = 'inner', axis = 1)

    # Add the target back in
    app_train['TARGET'] = train_labels

    print('Training Features shape after alignment: ', app_train.shape)
    print('Testing Features shape after alignment: ', app_test.shape)

    # -------------------------- ALIGNMENT -------------------------- #

    #df = df.dropna(subset=[target_column])
    X = app_train.drop(target_column, axis=1)
    y = app_train[target_column]

    # get features, labels, and feature_names
    feature_names = X.columns

    for column in X:
        X[column] = X[column].astype(float)

    return X, y, labels, target_names, feature_names