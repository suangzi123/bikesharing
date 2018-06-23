import numpy as np
import pandas as pd
from sklearn import svm
from sklearn import metrics
import MeasurePreprocess
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

X_train, X_test, y_train, y_test=MeasurePreprocess.GetData()


forest_reg01=svm.SVR()
param_test01=[{'C':[1000,2000,3000],'epsilon':[0.01,0.05,0.1]}]
gsearch1 = GridSearchCV(forest_reg01, param_test01, cv=5,return_train_score=True)
gsearch1.fit(X_train,y_train)
print gsearch1.best_params_, gsearch1.best_score_
#{'epsilon': 0.1, 'C': 2000} 0.631856914904

svr=svm.SVR(kernel='rbf',C=2000,epsilon=0.1)
scores = cross_val_score(svr, X_train,y_train, cv=5)
print "Cross validation scores:",scores
#Cross validation scores: [ 0.63428234  0.65285819  0.64489787  0.60023339  0.62701138]
svr.fit(X_train,y_train)
y_pred=svr.predict(X_test)
y_pred[y_pred<0]=0
metrics.mean_squared_error(y_test, y_pred)
print "MSE:",metrics.mean_squared_error(y_test, y_pred)
#MSE: 9382.43674104
print "Error rate:",MeasurePreprocess.ErrorRate(y_test,y_pred)
#Error rate:1.33612348935






