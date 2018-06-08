import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
import MeasurePreprocess
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

X_train, X_test, y_train, y_test=MeasurePreprocess.GetData()
forest_reg01 = RandomForestRegressor(random_state=42)

param_test01 = {'n_estimators':range(50,110,10)}
gsearch1 = GridSearchCV(forest_reg01, param_test01, cv=5,return_train_score=True)
gsearch1.fit(X_train,y_train)
print gsearch1.best_params_, gsearch1.best_score_
#({'n_estimators': 100}, 0.81962400946853908)

forest_reg02 = RandomForestRegressor(random_state=42,n_estimators=100)
param_test02 = {'max_depth':[18,20,22,24,26],'max_features':[0.6,0.7,0.8,0.9,1]}
gsearch02 = GridSearchCV(forest_reg02,param_test02 ,cv=5,return_train_score=True)
gsearch02.fit(X_train,y_train)
print gsearch02.best_params_, gsearch02.best_score_
#{'max_features': 0.6, 'max_depth': 26} 0.822663541485



clf = RandomForestRegressor(n_estimators=100, max_features=0.6,max_depth=25,random_state=42)
scores = cross_val_score(clf, X_train,y_train, cv=20)
print "Cross validation scores:",scores
#Cross validation scores: [ 0.83043103  0.82686555  0.80727542  0.85014593  0.84159721  0.80716581  0.82672161  0.8116308   0.82029225  0.80796569  0.82020875  0.83660168  0.8190992   0.80456573  0.84575043  0.80377163  0.80916503  0.82285363  0.81563257  0.85259085]
clf = clf.fit(X_train, y_train)
y_pred=clf.predict(X_test)
metrics.mean_squared_error(y_test, y_pred)
print "MSE:",metrics.mean_squared_error(y_test, y_pred)
#MSE: 4818.77715928
print "Error rate:",MeasurePreprocess.ErrorRate(y_test,y_pred)
#Error rate: 0.658582842915

