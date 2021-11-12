import numpy as np
import math
from abc import ABC, abstractmethod
from numpy.random import default_rng
import matplotlib.pyplot as plt
rng = default_rng()
from sklearn.utils import shuffle
from sklearn.linear_model import LinearRegression


# Generating Data
## Class for generating Data

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

    def generate_outliers(self, N):
        pass 

    
## Function to generate data points 

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

  
## Function that returns regression parameters
def regression(X, y) :
    reg = LinearRegression().fit(X.reshape(-1,1), y)
    
    return reg.coef_[0], reg.intercept_, reg.score(X, y)

## Testing Functions 
### Generating a Data set and fitting regression line 
s = (-1,2)
X, y = generate_data(s, 4, 100)
a, b, c = regression(X[:,1].reshape(-1,1), y)
print('Regression slope : %.3f' % a, 'Regression intercept : %.3f' % b, 'Regression score : %.3f' % c)
plt.plot(X[:,1],a*X[:,1]+b, c='red')
plt.scatter(X[:,1],y)
plt.title('Linear Regression fit on generated data')
plt.show();

### Adding one outlier and then solving
outlier = (20,0)
Xo = np.append(X[:,1],outlier[0])
yo = np.append(y,outlier[1])

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
        
     
# How to detect outliers ? 

## Outliers are the points most distant to the line, let's look at an example

## Generating Data
     
## Lets generate some points

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

## Solving regression and plots
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

norms = [np.sqrt(x) for x in distances]
points = np.arange(len(yo))
plt.scatter(points[-6:], norms[-6:], c='red', label=' Outliers ')
plt.scatter(points[:-6], norms[:-6], label = 'Regular points ')
plt.legend()
plt.title('Scatter plot of distance of the points from the regression line')
plt.show();

'''
Outliers have the biggest distance from the line
'''

## Detecting outliers 
### Generating Data
s = (-1,2)
X, y = generate_data(s, 2, 100)

## Adding one outlier
outliers = [(5,-5),(7,4)]
x_outliers = np.array([outliers[i][0] for i in range(len(outliers))])
y_outliers = np.array([outliers[i][1] for i in range(len(outliers))])
Xo = np.concatenate((X[:,1],x_outliers), axis=0)
yo = np.concatenate((y,y_outliers), axis=0)
plt.scatter(X[:,1],y)
plt.scatter(x_outliers,y_outliers, c='red', marker='x')
plt.title('Data generated with outliers')
plt.show();

## Regression Line
a, b, c = regression(Xo.reshape(-1,1), yo)
print('Regression slope : %.3f' % a, 'Regression intercept : %.3f' % b, 'Regression score : %.3f' % c)
plt.plot(Xo,a*Xo+b, c='red')
plt.scatter(Xo[:-2],yo[:-2])
plt.scatter(Xo[-2:],yo[-2:], marker='x')
plt.title('Linear Regression fit on generated data with added outlier')
plt.show();

## Computing s = sqrt(SSE/(n-2))

squared_errors = [(a*Xo[i]+b-yo[i])**2 for i in range(len(yo))]

sse = sum(squared_errors)

s = np.sqrt(sse/(len(yo)-2))

print('s : %.2f' % s)

##  plot the decision boundary for outliers

x = np.linspace(-20,20,50)
plt.plot(x,a*x+b, c='red')
plt.plot(x,a*x+b+2*s, c='green',  marker="_")
plt.plot(x,a*x+b-2*s, c='green',  marker="_" )
plt.scatter(Xo[:-2],yo[:-2])
plt.scatter(Xo[-2:],yo[-2:], marker='x')
plt.title('Linear Regression fit and outlier boundaries')
plt.show();


## Outliers detecting by direct computations

indices = []

for i in range(len(yo)) :
    if np.absolute(a*Xo[i]+b-yo[i]) > 2*s :
        indices.append(i)
        
print(indices)
   


    