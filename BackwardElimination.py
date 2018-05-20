import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
import statsmodels.formula.api as sm
dataset=pd.read_csv("train.csv")
dataset=dataset[['season','holiday','workingday','weather','temp','humidity','windspeed','count']]
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
labelencoder = LabelEncoder()
onehotencoder = OneHotEncoder(categorical_features = range(4))
X = onehotencoder.fit_transform(X).toarray()
scaler = StandardScaler()
scaler.fit(X)
X=scaler.transform(X)
X=np.append(arr = np.ones((10886,1)).astype(float), values = X, axis = 1)

xelimination = X[:,[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]]
regressorOLS = sm.OLS(y, xelimination).fit()
regressorOLS.summary()

xelimination = X[:,[0,1,2,3,4,5,6,7,8,10,11,12,13,14,15]]
regressorOLS = sm.OLS(y, xelimination).fit()
regressorOLS.summary()

xelimination = X[:,[0,1,2,3,4,5,6,8,10,11,12,13,14,15]]
regressorOLS = sm.OLS(y, xelimination).fit()
regressorOLS.summary()

xelimination = X[:,[0,2,3,4,5,6,8,10,11,12,13,14,15]]
regressorOLS = sm.OLS(y, xelimination).fit()
regressorOLS.summary()

xelimination = X[:,[0,2,3,4,6,8,10,11,12,13,14,15]]
regressorOLS = sm.OLS(y, xelimination).fit()
regressorOLS.summary()

xelimination = X[:,[0,3,4,6,8,10,11,12,13,14,15]]
regressorOLS = sm.OLS(y, xelimination).fit()
regressorOLS.summary()

xelimination = X[:,[0,3,4,6,10,11,12,13,14,15]]
regressorOLS = sm.OLS(y, xelimination).fit()
regressorOLS.summary()

xelimination = X[:,[0,3,4,10,11,12,13,14,15]]
regressorOLS = sm.OLS(y, xelimination).fit()
regressorOLS.summary()

xelimination = X[:,[0,3,4,10,11,13,14,15]]
regressorOLS = sm.OLS(y, xelimination).fit()
regressorOLS.summary()

xelimination = X[:,[0,3,4,10,13,14,15]]
regressorOLS = sm.OLS(y, xelimination).fit()
regressorOLS.summary()