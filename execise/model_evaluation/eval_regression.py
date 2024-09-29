import pandas as pd
import numpy as np
from sklearn.datasets import load_boston
BostonData = load_boston()

# Define X (independent variables) and y (dependent variables)
X = BostonData.data
y = BostonData.target

# Print size of the dataset
print("Boston dataset dimensions: {}".format(X.shape)) 

# Import linear regression from Sklearn package
from sklearn.linear_model import LinearRegression
# Build a linear regression object
LinReg = LinearRegression()
# Fit a linear regression model
LinReg.fit(X, y)

# Predict the data
y_predict = LinReg.predict(X)

# Calculate mean absolute error
from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(y, y_predict)

# Calculate root mean squared error (RMSE)
from sklearn.metrics import mean_squared_error
rmse = (np.sqrt(mean_squared_error(y, y_predict)))

# Calculate r--squared
from sklearn.metrics import r2_score
r2 = r2_score(y, y_predict)

# calculate adjusted r--squared
k = len(LinReg.coef_)+1
n = len(y)
Adjusted_r2 = 1- ((1-r2)*(n-1) / (n-k-1) )

# print intercept and coefficients
print("The intercept of the fitted model is: %.3f" % LinReg.intercept_)
print("The coefficients of the model are:\n" , [float('{:.3f}'.format(x)) for x in LinReg.coef_ ])
print("The performance of the model")
print("--------------------------------------")
print('MAE is %.3f' % (mae))
print('RMSE is %.3f' % (rmse))
print('R-Squared is %.3f' % (r2))
print('Adjusted R-Squared is %.3f' % (Adjusted_r2))