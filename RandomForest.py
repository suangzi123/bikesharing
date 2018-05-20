import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
import MeasurePreprocess
from sklearn.model_selection import cross_val_score

X_train, X_test, y_train, y_test=MeasurePreprocess.GetData()
clf = RandomForestRegressor(n_estimators=10)
scores = cross_val_score(clf, X_test,y_test, cv=20)
print "Cross validation scores:",scores
#Cross validation scores: [-0.36347056 -0.11921261 -0.13807835  0.21465008 -0.51127025 -0.06214871  0.29271577 -0.11469477 -0.14443182  0.01398871]
clf = clf.fit(X_train, y_train)
y_pred=clf.predict(X_test)
metrics.mean_squared_error(y_test, y_pred)
print "MSE:",metrics.mean_squared_error(y_test, y_pred)
#MSE:  26316.4125374
print "Error rate:",MeasurePreprocess.ErrorRate(y_test,y_pred)
#Error rate:  5.66669955297