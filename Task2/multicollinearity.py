"""
Multicollinearity in the Linear Regression Model
"""

Importing packages
import os
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error, r2_score

#opening the dataset to work with "auto-mpg"
path = '.'
dataset = os.path.join("auto-mpg.csv")
df = pd.read_csv(dataset,na_values=['NA','?'])
#df.head(10)


"""
Preparation of the dataset
"""

#checking for missing values
df.isnull().values.any()

#Checking number of NANs for each column, in order to understand how many missing values there are in a dataframe.
print("# of NaN in each columns:", df.isnull().sum(), sep='\n')

#let's fill missing values of "Horsepower" with its median of the column
med = df['horsepower'].median()
df['horsepower'] = df['horsepower'].fillna(med)

#Checking for missing values the column "Horsepower" after filling it with medians value of the column
df["horsepower"].isnull().values.any()


