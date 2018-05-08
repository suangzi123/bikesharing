from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict
import numpy as np
import pandas as pd
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()



dataframe=pd.read_csv("train.csv")
dataframe["hour"]=pd.DataFrame(dataframe.datetime.apply(lambda x:x.split()[1].split(":")[0]))
dataframe['hour'] = dataframe['hour'].astype('int64')
dataframe["month"]=pd.DataFrame(dataframe.datetime.apply(lambda x:x.split()[0].split("-")[1]))
dataframe['month'] = dataframe['month'].astype('int64')
for i in range(1,5):
    dataframe.ix[:,i]=dataframe.ix[:,i].astype('category')
dataset=dataframe[['hour','month','season','holiday','workingday','weather','temp','atemp','humidity','windspeed','count']]
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.005, random_state=0)
linreg = LinearRegression()
linreg.fit(X_train, y_train)
print "intercept:",linreg.intercept_
print "gradient:",linreg.coef_
y_pred = linreg.predict(X_test)
print "MSE:",metrics.mean_squared_error(y_test, y_pred)
print "RMSE:",np.sqrt(metrics.mean_squared_error(y_test, y_pred))

# Visualising the predicted results
plt.scatter(y_pred,y_test,color='b')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)
plt.ylabel('Measured')
plt.xlabel('Predicted')
plt.title("Linear Regression predicted results")
plt.show()


#Linear regression for each feature.
dataframe=pd.read_csv("train.csv")
dataframe["hour"]=pd.DataFrame(dataframe.datetime.apply(lambda x:x.split()[1].split(":")[0]))
dataframe['hour'] = dataframe['hour'].astype('int64')
dataframe["month"]=pd.DataFrame(dataframe.datetime.apply(lambda x:x.split()[0].split("-")[1]))
dataframe['month'] = dataframe['month'].astype('int64')
dataset0=dataframe[['hour','month','season','holiday','workingday','weather','temp','atemp','humidity','windspeed','casual','registered','count']]
for column in range(dataset0.shape[1]-1):
    x0, y0=dataset0.ix[:,column].values, dataset0.ix[:,-1].values
    x0_train, x0_test, y0_train, y0_test = train_test_split(x0, y0, test_size=0.005,random_state=0)
    linreg=LinearRegression()
    x0_train=x0_train.reshape(-1, 1)
    x0_test=x0_test.reshape(-1, 1)
    linreg.fit(x0_train,y0_train)
    y0_pre=linreg.predict(x0_test)
    print "gradient:",linreg.coef_
    print "intercept:",linreg.intercept_
    print "MSE:",metrics.mean_squared_error(y0_test, y0_pre)
    print "RMSE:",np.sqrt(metrics.mean_squared_error(y0_test, y0_pre))
    plt.scatter(x0_test,y0_test,color='k')
    plt.plot(x0_test,y0_pre,color='b',linewidth=3)
    plt.show()



