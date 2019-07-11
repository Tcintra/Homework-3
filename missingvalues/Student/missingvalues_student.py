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