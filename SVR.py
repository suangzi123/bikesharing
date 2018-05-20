import numpy as np
import pandas as pd
from sklearn import svm
from sklearn import metrics
import MeasurePreprocess
from sklearn.model_selection import cross_val_score

X_train, X_test, y_train, y_test=MeasurePreprocess.GetData()
svr=svm.SVR(kernel='poly',C=100,degree=2)
scores = cross_val_score(svr, X_test,y_test, cv=10)
print "Cross validation scores:",scores
#Cross validation scores: [-0.46914043  0.06748065 -0.30325645 -0.21487793  0.02455051 -0.07880042  -0.04268087 -0.04708424 -0.37349436 -0.02067927]
svr.fit(X_train,y_train)
y_pred=svr.predict(X_test)
metrics.mean_squared_error(y_test, y_pred)
print "MSE:",metrics.mean_squared_error(y_test, y_pred)
#MSE:  31028.8659434
print "Error rate:",MeasurePreprocess.ErrorRate(y_test,y_pred)
#Error rate: 8.86526875404




pca = PCA(n_components=2) 
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.fit_transform(X_test)
svr_pca=svm.SVR(kernel='poly',C=100,degree=2)
scores = cross_val_score(svr, X_test_pca,y_test, cv=10)
#Cross validation scores: [-0.15773931 -0.16207743 -0.26691765  0.00665453  0.02097789 -0.11742149  -0.16410581 -0.03807304 -0.412426   -0.42998502]
print "Cross validation scores:",scores
svr_pca.fit(X_train_pca,y_train)
y_pred_pca=svr_pca.predict(X_test_pca)
print "MSE:",metrics.mean_squared_error(y_test, y_pred_pca)
#MSE: 34202.5554205
print "Error rate:",MeasurePreprocess.ErrorRate(y_test,y_pred_pca)
#Error rate: 7.5607588582