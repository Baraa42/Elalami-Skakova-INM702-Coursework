import numpy as np
import math
from abc import ABC, abstractmethod
from numpy.random import default_rng
import matplotlib.pyplot as plt
rng = default_rng()
from sklearn.utils import shuffle
from sklearn.linear_model import LinearRegression


# Class to generate 2D data points
## Generate data points
class Data2D:
    
    def __init__(self, slope = (1, 1), noise_variance = 1 ):
        
        # intialize member variables here 
        self.slope = slope
        self.noise_variance = noise_variance
        
        
        
    def generate_points(self, N = 50):
    
        slope = self.slope
        noise_variance = self.noise_variance

        # initialize noise epsilon 
        epsilon = rng.normal(0, noise_variance**(1/2), N)

        ## Generate N points

        # Adding the X_0 coordinates full of ones for vector multiplications
        X_0 = np.ones(N)
        # for X_1 generate random points between -15,15
        X_1 = 30*(np.random.random(N)-0.5)

        # We create a design matrix, where rows are datapoints and columns are features, or input dimensions
        X = np.vstack( [X_0, X_1]).transpose()

        # initialize **y** with the equation of line here. Equation of line is written in the desciption above, 
        # line parameters (a,b, c) are stored in a local variable

        y = X.dot(np.array(slope)) + epsilon

        return X, y

    def generate_outliers(self, n):
        pass 



## function to generate 2D data points 
def generate_data(slope = (1,1), noise_variance=1, N = 50):
    
    # initialize noise epsilon 
    epsilon = rng.normal(0, noise_variance**(1/2), N)
    
    ## Generate N points
    
    # Adding the X_0 coordinates full of ones for vector multiplications
    X_0 = np.ones(N)
    # for X_1 generate random points between -15,15
    X_1 = 30*(np.random.random(N)-0.5)
    
    # We create a design matrix, where rows are datapoints and columns are features, or input dimensions
    X = np.vstack( [X_0, X_1]).transpose()
    
    # initialize **y** with the equation of line here. Equation of line is written in the desciption above, 
    # line parameters (a,b, c) are stored in a local variable

    y = X.dot(np.array(slope)) + epsilon
    
    return X, y

  
## Regression function returns slope, intercept, reg score
def regression(X, y) :
    reg = LinearRegression().fit(X.reshape(-1,1), y)
    
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
        
     
## Adding one outlier
outlier = (20,0)
Xo = np.append(X[:,1],outlier[0])
yo = np.append(y,outlier[1])


## Seeing effect of adding the outlier
a, b, c = regression(Xo.reshape(-1,1), yo)
print('Regression slope : %.3f' % a, 'Regression intercept : %.3f' % b, 'Regression score : %.3f' % c)
plt.plot(Xo,a*Xo+b, c='red')
plt.scatter(Xo,yo)
plt.title('Linear Regression fit on generated data with added outlier')

plt.show();

## Studying the impact of adding one outlier on randomly generated data with one outlier
## Each time we change the outlier and see what happens
slopes = []
intercepts = []
reg_score = []
distances = []
for i in range(100):
    outlier = (-50+i, -30)
    Xo = np.append(X[:,1],outlier[0])
    yo = np.append(y,outlier[1])
    slope, intercept, score = regression(Xo.reshape(-1,1), yo)
    slopes.append(slope)
    intercepts.append(intercept)
    reg_score.append(score)
    a = s[1]
    b = s[0]
    distance = np.absolute((a*outlier[0]+b-outlier[1])/(np.sqrt(a**2+1)))
    distance_reg = np.absolute((slope*outlier[0]+intercept-outlier[1])/(np.sqrt(slope**2+1)))
    distances.append(distance)
    if i%5 == 0 :
        print('Regression slope : %.3f' % slope, 'Regression intercept : %.3f' % intercept, 'Regression score : %.3f' % score)
        print('The outlier is :', outlier)
        print('The outlier distance from the green line is : %.2f' % distance)
        print('The outlier distance from the red line is : %.2f' % distance_reg)
        z = np.ones(len(y)+1)
        z[len(y)] = 0
        plt.plot(Xo,slope*Xo+intercept, c='red', label = f'Regression line with outilier :  {outlier} ', alpha = 0.7)
        plt.plot(Xo,a*Xo+b, c='green', label = 'Original Line', alpha = 0.7)
        plt.scatter(Xo[:len(y)],yo[:len(y)], marker = 'o', cmap='Spectral', alpha = 0.6 )
        plt.scatter(Xo[len(y)],yo[len(y)], marker = 'x', cmap='summer')
        plt.legend(loc = 'upper left')
        plt.show();
    
    
''' 
 Conclusion :

The outlier pulls the line towards it. The further the outlier is from the line the stronger the pull, and the strong it's effect on the regression score.
     
'''





''' 
How to detect outliers ? 

Outliers are the points most distant to the line, let's look at an example
'''

## Lets generate some points using the class this time

data = Data2D(s, 3)
X, y = data.generate_points(100)
plt.scatter(X[:,1],y)
plt.title('Data generated without outliers')
plt.show();


## Adding 6 outliers
outliers = [(20,-10), (35,-30), (17,-17), (10,-30), (-5, -50), (15,-20)]
x_outliers = np.array([outliers[i][0] for i in range(len(outliers))])
y_outliers = np.array([outliers[i][1] for i in range(len(outliers))])

Xo = np.concatenate((X[:,1],x_outliers), axis=0)
yo = np.concatenate((y,y_outliers), axis=0)

plt.scatter(X[:,1],y)
plt.scatter(x_outliers,y_outliers, c='red', marker='x')
plt.title('Data generated with outliers')
plt.show();


## Regression and plots
a, b, c = regression(Xo.reshape(-1,1), yo)
a1, b1, c1 = regression(X[:,1].reshape(-1,1), y)
print('Regression slope : %.3f' % a1, 'Regression intercept : %.3f' % b1, 'Regression score : %.3f' % c1)
print('Outliers Regression slope : %.3f' % a, 'Outliers Regression intercept : %.3f' % b, 'Outliers Regression score : %.3f' % c)
plt.plot(Xo,a*Xo+b, c='red', label='Regression Line with outliers')
plt.plot(Xo,a1*Xo+b1, c='green', label='Regression Line without outliers')
plt.legend(loc='upper left')
plt.scatter(X[:,1],y)
plt.scatter(x_outliers,y_outliers, c='red', marker='x')
plt.title('Linear Regression fit on generated data with added outliers')
plt.show();


## Checking distance of the points from the fitted line
### We use the line fitted with outliers supposing that we havent removed them yet

distances = []
for i in range(len(yo)):
    distance = np.absolute((a*Xo[i]+b-yo[i])/(np.sqrt(a**2+1)))
    distances.append(distance)

points = np.arange(len(yo))
cm = np.ones(len(yo))
cm[:-6] = 0
plt.scatter(points[-6:], distances[-6:], c='red', label=' Outliers ')
plt.scatter(points[:-6], distances[:-6], label = 'Regular points ')
plt.legend()
plt.title('Scatter plot of distance of the points from the regression line')
plt.show();

"""
Multicollinearity in the Linear Regression Model
"""

#Importing packages
import os
# import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
# import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
# from sklearn.linear_model import LinearRegression
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error, r2_score

#opening the dataset to work with "auto-mpg"
path = '.'
dataset = os.path.join("auto-mpg.csv")
df = pd.read_csv(dataset,na_values=['NA','?'])
# df.head(10)

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

"""
as x varibale increases, y outcome variable increases in a almost perfectly correlated manner.

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
[2] The condition number is large, 8.69e+04. This might indicate that there are strong multicollinearity or other numerical problems.
'''

'''
Also, correlation can be checked using VIF. Presence of High Variance Inflation Factor in the dataset is usually undesirable, since it highlights unreliability of your computations in regression analysis.
'''
X_variables = df.drop(columns=["name"])
# [["horsepower","weight","displacement", "mpg"]]
vif_data = pd.DataFrame()
vif_data["feature"] = X_variables.columns
vif_data["VIF"] = [variance_inflation_factor(X_variables.values, i) for i in range(len(X_variables.columns))]
print(vif_data)

'''
Dealing with multicollinearity. First method - substracting the mean. For that we will use a copy of our dataset to manipulate the data.
'''

#Substracting the mean on a copy of our dataset
new_df = df.copy()
print ("The mean value")
print (new_df.mean())
print( "The value after subraction of mean")
adj_df = new_df -new_df.mean()
print (adj_df)

'''
Subtract the mean method, which is also known as centering the variables. 
This method removes the multicollinearity produced by interaction and higher-order terms as effectively as the other standardization methods, 
but it has the added benefit of not changing the interpretation of the coefficients. 
If you subtract the mean, each coefficient continues to estimate the change in the mean response per unit increase in X when all other predictors are held constant.
'''

#Checking VIF again
X_variables = adj_df.drop(columns=["name"])
# [["horsepower","weight","displacement", "mpg"]]
vif_data = pd.DataFrame()
vif_data["feature"] = X_variables.columns
vif_data["VIF"] = [variance_inflation_factor(X_variables.values, i) for i in range(len(X_variables.columns))]
print(vif_data)

'''
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