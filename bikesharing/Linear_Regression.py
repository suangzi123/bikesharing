from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectKBest

import numpy as np
import pandas as pd
from sklearn import metrics
import matplotlib.pyplot as plt
import scipy
import MeasurePreprocess



X_train, X_test, y_train, y_test=MeasurePreprocess.GetData()
linreg = LinearRegression()
scores = cross_val_score(linreg, X_test,y_test, cv=10)
print "Cross validation scores:",scores
#Cross validation scores: [ 0.58255357  0.57289505  0.64241995  0.59190076  0.66598944  0.59132919  0.64471031  0.6490007   0.64190622  0.60148136]
linreg.fit(X_train, y_train)
y_pred = linreg.predict(X_test)
print "MSE:",metrics.mean_squared_error(y_test, y_pred)
print "Error rate:",MeasurePreprocess.ErrorRate(y_test,y_pred)
#MSE: 10325.9644869
#Error rate: 2.36005605451


# Visualising the predicted results
plt.scatter(y_pred,y_test,color='b')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)
plt.ylabel('Measured')
plt.xlabel('Predicted')
plt.title("Linear Regression predicted results")
plt.show()


#Linear regression for hour feature.
x0_train, x0_test, y0_train, y0_test=MeasurePreprocess.GetHourData()

linreg=LinearRegression()
x0_train=x0_train.reshape(-1, 1)
x0_test=x0_test.reshape(-1, 1)
scores = cross_val_score(linreg, x0_test,y0_test, cv=10)
print "Cross validation scores:",scores
#Cross validation scores: [ 0.14100185  0.14447664  0.11616706  0.22160484  0.11910182  0.14844926  0.16447769  0.18203852  0.12086551  0.18053057]
linreg.fit(x0_train,y0_train)
y0_pre=linreg.predict(x0_test)
print "gradient:",linreg.coef_
#gradient: [ 10.4745815]
print "intercept:",linreg.intercept_
#intercept: 69.8361082142
print "MSE:",metrics.mean_squared_error(y0_test, y0_pre)
#MSE: 28013.0003656
print "Error rate:",MeasurePreprocess.ErrorRate(y0_test,y0_pre)
#Error rate: 4.81197893906


#Windspeed Polynomial Regression
x_train, x_test, y_train, y_test=MeasurePreprocess.HourPolynomialRegression()
linreg = LinearRegression()

scores = cross_val_score(linreg, x_test,y_test, cv=10)
print "Cross validation scores:",scores
#Cross validation scores: [ 0.40211768  0.37545973  0.2889311   0.48806122  0.36187756  0.38482698  0.34199924  0.3731043   0.35835942  0.3766743 ]
linreg.fit(x_train, y_train)
y_pred = linreg.predict(x_test)
print "Hour Polynomial Regression MSE:",metrics.mean_squared_error(y_test, y_pred)
#Hour Polynomial Regression MSE:  20698.3456185
print "Error rate:",MeasurePreprocess.ErrorRate(y_test,y_pred)
#Error rate: 2.49101714254

