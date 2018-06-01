import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn import metrics
import MeasurePreprocess
from sklearn.model_selection import cross_val_score

X_train, X_test, y_train, y_test=MeasurePreprocess.GetData()
xlf = xgb.XGBRegressor()

scores = cross_val_score(xlf, X_test,y_test, cv=10)
print("Cross validation scores:",scores)
#Cross validation scores: [0.32496831, 0.20006842, 0.24337803, 0.3062608 , 0.28698802, 0.25849486, 0.32075113, 0.30951091, 0.18862874, 0.24116313]
xlf = xlf.fit(X_train, y_train)
y_pred=xlf.predict(X_test)
metrics.mean_squared_error(y_test, y_pred)
print("MSE:",metrics.mean_squared_error(y_test, y_pred))
#MSE:  19164.524348645537
print("Error rate:",MeasurePreprocess.ErrorRate(y_test,y_pred))
#Error rate: 4.512980009261412