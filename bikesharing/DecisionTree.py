import numpy as np
import pandas as pd
from sklearn import tree
from sklearn import metrics
import MeasurePreprocess
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

X_train, X_test, y_train, y_test=MeasurePreprocess.GetData()
clf = tree.DecisionTreeRegressor()

forest_reg01=tree.DecisionTreeRegressor()
param_test01=[{'min_samples_split':[0.02,0.04,0.06,0.08,0.1]}]
gsearch1 = GridSearchCV(forest_reg01, param_test01, cv=5,return_train_score=True)
gsearch1.fit(X_train,y_train)
print gsearch1.best_params_, gsearch1.best_score_
#{'min_samples_split': 0.02} 0.662118834797

clf = tree.DecisionTreeRegressor(min_samples_split=0.02)
scores = cross_val_score(clf, X_test,y_test, cv=5,scoring='r2')
print "Cross validation scores:",scores
#Cross validation scores: [ 0.56081613  0.60183797  0.67513348  0.64539087  0.59581384]
clf = clf.fit(X_train, y_train)
y_pred=clf.predict(X_test)
metrics.mean_squared_error(y_test, y_pred)
print "MSE:",metrics.mean_squared_error(y_test, y_pred)
#MSE: 9095.94262104
print "Error rate:",MeasurePreprocess.ErrorRate(y_test,y_pred)
#Error rate: 1.25909731218