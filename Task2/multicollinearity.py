"""
Multicollinearity in the Linear Regression Model
"""

#Importing packages
import os
import math
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from numpy.random import default_rng
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error, r2_score

#Working with multicollinearity in Linear Regression
# Class to generate 2D data points
## Generate data points
rng = default_rng()


class Data2D:

    def __init__(self, slope=(1, 1), noise_variance=1):
        # intialize member variables here
        self.slope = slope
        self.noise_variance = noise_variance

    def generate_points(self, N=50):
        slope = self.slope
        noise_variance = self.noise_variance

        # initialize noise epsilon
        epsilon = rng.normal(0, noise_variance ** (1 / 2), N)

        ## Generate N points
        X_1 = 30 * (np.random.random(N) - 0.5)
        # creating multicollinearity
        X_2 = X_1 ** 2

        # We create a design matrix, where rows are datapoints and columns are features, or input dimensions
        X = np.vstack([X_1, X_2]).transpose()

        # initialize **y** with the equation of line here. Equation of line is written in the desciption above,
        # line parameters (a,b, c) are stored in a local variable

        y = X.dot(np.array(slope)) + epsilon

        return X, y

    def generate_outliers(self, n):
        pass

    ## function to generate 2D data points


def generate_data(slope=(1, 1), noise_variance=1, N=50):
    # initialize noise epsilon
    epsilon = rng.normal(0, noise_variance ** (1 / 2), N)

    ## Generate N points
    # for X_1 generate random points between -15,15
    X_1 = 30 * (np.random.random(N) - 0.5)
    # creating multicollinearity
    X_2 = X_1 ** 2

    # We create a design matrix, where rows are datapoints and columns are features, or input dimensions
    X = np.vstack([X_1, X_2]).transpose()

    # initialize **y** with the equation of line here. Equation of line is written in the desciption above,
    # line parameters (a,b, c) are stored in a local variable

    y = X.dot(np.array(slope)) + epsilon

    return X, y


## Regression function returns slope, intercept, reg score
def regression(X, y):
    reg = LinearRegression().fit(X.reshape(-1, 1), y)

    return reg.coef_[0], reg.intercept_, reg.score(X, y)

## Generating data plotting the regression line
s = (-1,2)
X, y = generate_data(s, 4, 100)
a, b, c = regression(X[:,1].reshape(-1,1), y)
print('Regression slope : %.3f' % a, 'Regression intercept : %.3f' % b, 'Regression score : %.3f' % c)
plt.plot(X[:,1],a*X[:,1]+b, c='red')
plt.scatter(X[:,1],y)
plt.title('Linear Regression fit on generated data')
plt.show();

#Using statistical analysis to find whether there is a high correlation that can impact our output
X_with_constant = sm.add_constant(X)
model = sm.OLS(y, X_with_constant)
results = model.fit()
print(results.summary())
#OLS Regression statistical analysis table, showed high condition number (8.59e+04),
# indicating the strong multicollinearity problem.

#We will use the PCA model and see how it will impact multicollinearity in our linear regression.


#let's apply this knowledge on the "auto-mpg" dataset.
#opening the dataset to work with "auto-mpg"
path = '.'
dataset = os.path.join("auto-mpg.csv")
df = pd.read_csv(dataset,na_values=['NA','?'])
#df.head(10)


"""
Before starting any data manipulation, it is important to check on possible missing values of any value of the variable in the dataset.
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
y = df["horsepower"].values.reshape(-1,1)
fig = plt.figure()
ax = plt.axes(projection = "3d")
ax.scatter3D(X1, y, X2, "gray")
plt.show()

# ploting a plane
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

As x varibale increases, y outcome variable increases in a almost perfectly correlated manner.

We can also have a look at correlation between all comulmns at our datset
"""
#scatterplot
sns.set()
cols = df.columns.drop(["name"])
sns.pairplot(df[cols], size = 2.5)
plt.show();

# heatmap
columns = df.columns
hm = sns.heatmap(df[columns].corr(), cbar=True, annot=True)

#Using statistical analysis to find whether there is a high correlation that can impact our output
X = df.drop(columns=["name"])
y = df["horsepower"]
X_with_constant = sm.add_constant(X)
model = sm.OLS(y, X_with_constant)
results = model.fit()
print(results.summary())

'''
In our analysis we have used 2D, 3D models, pairplots and heatmaps using matplolib and seaborn libraires. They have showed the presence of high correlation between “displacement”, “horsepower”, “weight” and “cylinders” values. 
We also called the OLS Regression statistical analysis table, which showed high condition number (8.59e+04), indicating the strong multicollinearity problem.
'''

'''
Variance Inflation Factor (VIF) is another tool of the data analysis that demonstrates how one input variable impacts the other. 
VIF higher than 10 is usually undesirable, since it highlights unreliability of computations in regression analysis and considered to be problematic. 
It is computed using the R-squared from the regression: VIF = 1/(1-R**2 in k). 
Our results showed extremely high VIF, confirming again high multicollinearity.
'''

X_VIF = df.drop(columns=["name"])
data = pd.DataFrame()
data["feature"] = X_VIF.columns
data["VIF"] = [variance_inflation_factor(X_VIF.values, i) for i in range(len(X_VIF.columns))]
print(data)

'''
In order to solve this issue, several options are considered: leaving the dataset as it still predicts the output; removing, 
but this can create omitted variable bias, since there will be variables outside of the model; 
combining multicollinear variables to create new ones; and using PCA (Principal Component Analysis). 
We have demonstrated the impact on multicollinearity of PCA and the method of subtracting the mean.
'''

#Substracting the mean on a copy of our dataset
new_df = df.copy()
m_mean = new_df.mean() # mean value
print (m_mean)
result =  new_df -new_df.mean()#after substraction
print (result)

'''
Subtract the mean method, which is also known as centering the variables.
This method removes the multicollinearity produced by interaction and higher-order terms as effectively as the other standardization methods,
but it has the added benefit of not changing the interpretation of the coefficients.
If you subtract the mean, each coefficient continues to estimate the change in the mean response per unit increase in X when all other predictors are held constant.
'''

#Checking VIF again
X_VIF = result.drop(columns=["name"])
data = pd.DataFrame()
data["feature"] = X_VIF.columns
data["VIF"] = [variance_inflation_factor(X_VIF.values, i) for i in range(len(X_VIF.columns))]
print(data)


'''
In our case subtracting the mean method still left some of the variables with high VIF, 
so we also implemented the PCA model with 3 n-components.
PCA model works with multicollinearity. Let's see how it will work on our dataset.
'''
#Choosing the number of components
X = df.drop(columns=["name"])
y = df["horsepower"]
X_std = StandardScaler().fit_transform(X)
pca = PCA().fit(X_std)
#Plotting the Cumulative Summation of the Explained Variance
plt.figure()
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of Components')
plt.ylabel('Variance (%)') #for each component
plt.title('Dataset Explained Variance')
plt.show()

np.cumsum(pca.explained_variance_ratio_)

'''
we can see that after the second variable, variance is increasing.
That is what we need.So, n_components=3 will used for our PCA model.
'''
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X_std)
X_pca_with_constant = sm.add_constant(X_pca)
print(X_pca_with_constant)

'''
Let's check again our statistical table
'''
model = sm.OLS(y, X_pca_with_constant)
results = model.fit()
print(results.summary())

'''
Now we have checked again our statistic analysis on the database,
we can see that there is no warning about the multiculliniarity
'''
'''
Comparing the MSE train and R**2 for initial dataset with the dataset using the PCA.
'''
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
slr.fit(X_train, y_train)
y_train_pred = slr.predict(X_train)
y_test_pred = slr.predict(X_test)
print('MSE train: {:.3f}, test: {:.3f}'.format(
        mean_squared_error(y_train, y_train_pred),
        mean_squared_error(y_test, y_test_pred)))
print('R^2 train: {:.3f}, test: {:.3f}'.format(
        r2_score(y_train, y_train_pred),
        r2_score(y_test, y_test_pred)))


X_train, X_test, y_train, y_test = train_test_split(X_pca_with_constant, y, test_size=0.3, random_state=0)
slr.fit(X_train, y_train)
y_train_pred = slr.predict(X_train)
y_test_pred = slr.predict(X_test)
print('MSE train: {:.3f}, test: {:.3f}'.format(
        mean_squared_error(y_train, y_train_pred),
        mean_squared_error(y_test, y_test_pred)))
print('R^2 train: {:.3f}, test: {:.3f}'.format(
        r2_score(y_train, y_train_pred),
        r2_score(y_test, y_test_pred)))

'''
Conlcusion. As a result of our manipulations, MSE train and R**2 are smaller.
'''