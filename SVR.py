import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn import metrics
from sklearn.decomposition import PCA
import MeasurePreprocess


X_train, X_test, y_train, y_test=MeasurePreprocess.GetData()
svr=svm.SVR(kernel='poly',C=100,degree=6)
svr.fit(X_train,y_train)
y_pred=svr.predict(X_test)
metrics.mean_squared_error(y_test, y_pred)
print "MSE:",metrics.mean_squared_error(y_test, y_pred)
#MSE: 24690.2488272

pca = PCA(n_components=2) 
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.fit_transform(X_test)
svr_pca=svm.SVR(kernel='poly',C=100,degree=2)
svr_pca.fit(X_train_pca,y_train)
y_pred_pca=svr_pca.predict(X_test_pca)
print "MSE:",metrics.mean_squared_error(y_test, y_pred_pca)
#MSE: 34202.5733315