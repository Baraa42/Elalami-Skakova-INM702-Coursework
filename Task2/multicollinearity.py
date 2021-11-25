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


"""
Exploring correlation between columns of our dataset using different methods and possible multicollinearity.
"""
%matplotlib inline
# %matplotlib widget
X1 = df["weight"].values.reshape(-1,1)
X2 = df["mpg"].values.reshape(-1,1)
# X3 = df["displacement"].values.reshape(-1,1)
y = df["horsepower"].values.reshape(-1,1)
fig = plt.figure()
ax = plt.axes(projection = "3d")
ax.scatter3D(X1, y, X2, "gray")
plt.show()

# ploting a plane

X1 = df["weight"].values.reshape(-1, 1)
X2 = df["mpg"].values.reshape(-1, 1)
# X3 = df["displacement"].values.reshape(-1,1)
y = df["horsepower"].values.reshape(-1, 1)


def plot_plane_with_points(x, y, z):
    X = np.hstack((x, y))
    X = np.hstack((np.ones((x.shape[0], 1)), X))
    theta = np.dot(np.dot(np.linalg.pinv(np.dot(X.transpose(), X)), X.transpose()), z)
    k = int(max(np.max(x), np.max(y), np.max(z)))  # size of the plane

    p1, p2 = np.mgrid[:k, :k]
    P = np.hstack((np.reshape(p1, (k * k, 1)), np.reshape(p2, (k * k, 1))))
    P = np.hstack((np.ones((k * k, 1)), P))

    plane = np.reshape(np.dot(P, theta), (k, k));

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    ax.plot(x[:, 0], y[:, 0], z[:, 0], 'ro')  # scatter plot
    ax.plot_surface(p1, p2, plane)  # plane plot

    ax.set_xlabel('x1 label')
    ax.set_ylabel('x2 label')
    ax.set_zlabel('y label')

    return plt.show()
plot_plane_with_points(X1, y, X2)
"""
We can see that there is a high correlation between these three columns.
Method 2. Let's have a look at 2 specific columns of the dataset and their correlation.
"""
import matplotlib.pyplot as plt
X = df[["weight"]]
y = df["horsepower"]
plt.scatter(X, y)
plt.plot(X, slr.predict(X), color='red', linewidth=2);


