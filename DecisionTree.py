import numpy as np
import pandas as pd
from sklearn import tree
from sklearn import metrics
import MeasurePreprocess
from sklearn.model_selection import cross_val_score

X_train, X_test, y_train, y_test=MeasurePreprocess.GetData()
clf = tree.DecisionTreeRegressor()
scores = cross_val_score(clf, X_test,y_test, cv=10)
print "Cross validation scores:",scores
#Cross validation scores: [-0.64013985  0.43453123 -0.43475407 -1.52426647 -1.3498649  -0.9332302  -0.16428061 -0.57068506 -0.83505766 -1.03774652]
clf = clf.fit(X_train, y_train)
y_pred=clf.predict(X_test)
metrics.mean_squared_error(y_test, y_pred)
print "MSE:",metrics.mean_squared_error(y_test, y_pred)
#MSE:  38321.4145317
print "Error rate:",MeasurePreprocess.ErrorRate(y_test,y_pred)
#Error rate:  6.34400165952