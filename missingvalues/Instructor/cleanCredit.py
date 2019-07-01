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

# read csv file
path = os.path.join("..", "Data")
df = pd.read_csv(os.path.join(path, 'application_train.csv'))


# cut sample down to 10000 entries instead of 33000
new_df = df.sample(n = 10000)

# export to new csv
new_df.to_csv(os.path.join(path, 'credit_train.csv'))