import numpy as np
import pandas as pd
from sklearn import tree
from sklearn import metrics
import MeasurePreprocess
from sklearn.model_selection import cross_val_score

X_train, X_test, y_train, y_test=MeasurePreprocess.GetData()
clf = tree.DecisionTreeRegressor()
scores = cross_val_score(clf, X_test,y_test, cv=10,scoring='r2')
print "Cross validation scores:",scores
#Cross validation scores: [ 0.5193123   0.52608234  0.59057074  0.56564376  0.46504067  0.64762206  0.52083428  0.55179048  0.49140886  0.48461495]
clf = clf.fit(X_train, y_train)
y_pred=clf.predict(X_test)
metrics.mean_squared_error(y_test, y_pred)
print "MSE:",metrics.mean_squared_error(y_test, y_pred)
#MSE: 9487.82064711
print "Error rate:",MeasurePreprocess.ErrorRate(y_test,y_pred)
#Error rate: 0.692933479909