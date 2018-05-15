from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict
import numpy as np
import pandas as pd
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
import MeasurePreprocess
sns.set()


X_train, X_test, y_train, y_test=MeasurePreprocess.GetData()
linreg = LinearRegression()
linreg.fit(X_train, y_train)
y_pred = linreg.predict(X_test)
print "MSE:",metrics.mean_squared_error(y_test, y_pred)
#MSE: 24062.8589904


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
linreg.fit(x0_train,y0_train)
y0_pre=linreg.predict(x0_test)
print "gradient:",linreg.coef_
#gradient: [ 2.28489923]
print "intercept:",linreg.intercept_
#intercept: 162.172338362
print "MSE:",metrics.mean_squared_error(y0_test, y0_pre)
#MSE: 31058.1710239
plt.scatter(x0_test,y0_test,color='k')
plt.plot(x0_test,y0_pre,color='b',linewidth=3)
plt.title('windspeed linear regression')
plt.ylabel('Count')
plt.xlabel('Windspeed')
plt.show()

#Windspeed Polynomial Regression
x_train, x_test, y_train, y_test=MeasurePreprocess.WindspeedPolynomialRegression()
linreg = LinearRegression()
linreg.fit(x_train, y_train)
y_pred = linreg.predict(x_test)
print "Windspeed Polynomial Regression MSE:",metrics.mean_squared_error(y_test, y_pred)
#Windspeed Polynomial Regression MSE: 31032.8561746



