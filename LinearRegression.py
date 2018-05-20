from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectKBest

import numpy as np
import pandas as pd
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
import MeasurePreprocess
sns.set()


X_train, X_test, y_train, y_test=MeasurePreprocess.GetData()
linreg = LinearRegression()
scores = cross_val_score(linreg, X_test,y_test, cv=10)
print "Cross validation scores:",scores
#Cross validation scores: [ 0.39861152  0.29279047 -0.23444094  0.33524775 -0.66794863  0.22375366  0.37918367 -0.22291575 -0.26410568 -0.07763031]
linreg.fit(X_train, y_train)
y_pred = linreg.predict(X_test)
print "MSE:",metrics.mean_squared_error(y_test, y_pred)
print "Error rate:",MeasurePreprocess.ErrorRate(y_test,y_pred)
#MSE: 24062.8589904
#Error rate: 6.4601698913


# Visualising the predicted results
plt.scatter(y_pred,y_test,color='b')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)
plt.ylabel('Measured')
plt.xlabel('Predicted')
plt.title("Linear Regression predicted results")
plt.show()


#Linear regression for windspeed feature.
x0_train, x0_test, y0_train, y0_test=MeasurePreprocess.GetWindspeedData()

linreg=LinearRegression()
x0_train=x0_train.reshape(-1, 1)
x0_test=x0_test.reshape(-1, 1)
scores = cross_val_score(linreg, x0_test,y0_test, cv=10)
print "Cross validation scores:",scores
#Cross validation scores: [ 0.00659087 -0.23267376 -0.08699674  0.0291493  -0.57398698  0.00798402  -0.0081969  -0.00079042 -0.12915215 -0.02080022]
linreg.fit(x0_train,y0_train)
y0_pre=linreg.predict(x0_test)
print "gradient:",linreg.coef_
#gradient: [ 2.28489923]
print "intercept:",linreg.intercept_
#intercept: 162.172338362
print "MSE:",metrics.mean_squared_error(y0_test, y0_pre)
#MSE: 31058.1710239
print "Error rate:",MeasurePreprocess.ErrorRate(y0_test,y0_pre)
#Error rate: 10.0986768469


#Windspeed Polynomial Regression
x_train, x_test, y_train, y_test=MeasurePreprocess.WindspeedPolynomialRegression()
linreg = LinearRegression()

scores = cross_val_score(linreg, x_test,y_test, cv=10)
print "Cross validation scores:",scores
#Cross validation scores: [ -1.73749282e-01  -2.62752233e-01  -6.48312719e-02   1.63664982e-02   -3.25006570e+00   2.68568790e-03  -5.61825438e-03  -2.84347108e-02   -1.77784773e-01  -8.76974810e-03]
linreg.fit(x_train, y_train)
y_pred = linreg.predict(x_test)
print "Windspeed Polynomial Regression MSE:",metrics.mean_squared_error(y_test, y_pred)
#Windspeed Polynomial Regression MSE: 31032.8561746
print "Error rate:",MeasurePreprocess.ErrorRate(y_test,y_pred)
#Error rate: 10.1412484243

