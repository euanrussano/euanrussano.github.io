# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Pandas function `read_csv()` is used to read the csv file 'housingprices.csv' and place it as a dataframe.

df= pd.read_csv('housingprices.csv')
df.head()

print(df.columns)

# Check for NaN values
print(pd.isna(df).any())

"""
Split the dataset into inputs (x) and output(y). Use the method `values` to transform from a DataFrame object to an array object, which can efficiently managed by Numpy library.
"""

x = df[['Size','Nr Bedrooms','Nr Bathrooms']].values
y = df['Price'].values.reshape(-1,1)
m = len(y)
print(m)

"""Let's generate a simple visualization the price in relation to each input variable."""

fig,axs = plt.subplots(2,2)
axs[0, 0].plot(x[:,0],y,'o')
axs[0, 1].plot(x[:,1],y,'o')
axs[1, 0].plot(x[:,2],y,'o')
plt.show()

"""Linear correlation can be evaluated through Pearson's coefficient, which returns a value between 0 and 1. 0 means there is no correlation, while 1 means perfect correlation. Everything in betweeen indicates that the data is somehow correlated, though usually a correlation of more than 0.8 is expected for a variable to be considered a predictor, i.e an input to a Machine Learning model."""

# Check correlation of each input with the input
from scipy.stats import pearsonr
print(f'Correlation between x1 and y = {pearsonr(x[:,0],y[:,0])[0]:.2f}')
print(f'Correlation between x2 and y = {pearsonr(x[:,1],y[:,0])[0]:.2f}')
print(f'Correlation between x3 and y = {pearsonr(x[:,2],y[:,0])[0]:.2f}')

"""Add a bias column to the input vector. This is a column of ones so when we calibrate the parameters it will also multiply such bias.
"""

# Add a bias to the input vector
X = np.concatenate((np.ones((len(x),1)),x),axis=1)

"""Another important pre-processing is data normalization. In multivariate regression, the difference in the scale of each variable may cause difficulties for the optimization algorithm to converge, i.e to find the best optimum according the model structure. This procedure is also known as **Feature Scaling**."""

Xnorm = X.copy()
minx = np.min(X[:,1:])
maxx = np.max(X[:,1:])
Xnorm[:,1:] = (X[:,1:]-minx)/(maxx-minx)

ynorm = y.copy()
maxy = np.max(y)
miny = np.min(y)
ynorm = (y-miny)/(maxy - miny) 

# Initial estimate of parameters
theta0 = np.zeros((X.shape[1],1))+0.4
#theta0 = np.array([[0],[0.5],[2],[0.5]])

ypred = Xnorm.dot(theta0)

sortidx = np.argsort(ynorm[:,0]) # sort the values for better visualization
plt.figure()
plt.plot(ynorm[sortidx,0],'o')
plt.plot(ypred[sortidx,0],'--')
plt.show()

# Create a function `grad()` to compute the necessary gradients of the cost function.

def grad(theta):
    dJ = 1/m*np.sum((Xnorm.dot(theta)-ynorm)*Xnorm,axis=0).reshape(-1,1)
    return dJ

grad(theta0)

# Function to calculate the cost J

def cost(theta):
    J = np.sum((Xnorm.dot(theta)-ynorm)**2,axis=0)[0]
    return J

cost(theta0)

"""We are ready to implement the Gradient Descent algorithm! The steps of this algorithm consists of:
- Obtain the gradients of the cost function according the actual value of the parameters;
- Calculate the cost to keep track of it;
- Update the parameters according the following schedule:
"""

def GD(theta0,learning_rate = 0.5,epochs=1000,TOL=1e-7):
    
    theta_history = [theta0]
    J_history = [cost(theta0)]
    
    thetanew = theta0*10000
    print(f'epoch \t Cost(J) \t')
    for epoch in range(epochs):
        if epoch%100 == 0:
            print(f'{epoch:5d}\t{J_history[-1]:7.4f}\t')
        dJ = grad(theta0)
        J = cost(theta0)
        
        thetanew = theta0 - learning_rate*dJ
        theta_history.append(thetanew)
        J_history.append(J)
        
        if np.sum((thetanew - theta0)**2) < TOL:
            print('Convergence achieved.')
            break
        theta0 = thetanew

    return thetanew,theta_history,J_history


# Next, evaluate the Gradient Descent to determine the optimum set of parameters for the linear regression.


theta,theta_history,J_history = GD(theta0)

plt.figure()
plt.plot(J_history)
plt.show()

# We can perform predictions on the training set using the following code.

yprednorm = Xnorm.dot(theta)

ypred = yprednorm*(maxy-miny) + miny
plt.figure()
plt.plot(y[sortidx,0],'o')
plt.plot(ypred[sortidx,0],'--')
plt.show()

# The following function is to used to get an input, normalize it and perform predictions.

def predict(x,theta):
    xnorm = (x-minx)/(maxx-minx)
    yprednorm = xnorm.dot(theta)
    ypred = yprednorm*(maxy - miny) + miny
    return ypred

# Use our model to predict the price of a house with 73 square meters, 1 bedroom and 1 bathroom.

x = np.array([1,73,1,1])

print(predict(x,theta))

# Check if model is  significant using Pearson correlation.

print(pearsonr(ypred.reshape(-1),y.reshape(-1)))

