import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn import metrics
import MeasurePreprocess
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt

X_train, X_test, y_train, y_test=MeasurePreprocess.GetData()
forest_reg01 = xgb.XGBRegressor()
param_test01=[{'max_depth':[6,7,8,9,10],'min_child_weight':range(5,10)}]
gsearch1 = GridSearchCV(forest_reg01, param_test01, cv=5,return_train_score=True)
gsearch1.fit(X_train,y_train)
print gsearch1.best_params_, gsearch1.best_score_
#{'max_depth': 10, 'min_child_weight': 8} 0.83598220439

forest_reg02 = xgb.XGBRegressor(max_depth=10,min_child_weight=8)
param_test02=[{'gamma':[0,1,2,3,4,5],'colsample_bytree':[0.5,0.6,0.7,0.8,0.9,1]}]
gsearch2 = GridSearchCV(forest_reg02, param_test02, cv=5,return_train_score=True)
gsearch2.fit(X_train,y_train)
print gsearch2.best_params_, gsearch2.best_score_
#{'colsample_bytree': 0.8, 'gamma': 4} 0.838352724451





xlf = xgb.XGBRegressor(max_depth=10,min_child_weight=8,gamma=4,colsample_bytree=0.8)
scores = cross_val_score(xlf, X_train,y_train, cv=5)
print"Cross validation scores:",scores
#Cross validation scores: [ 0.83592285  0.83862756  0.83806972  0.83190081  0.8472441 ]
xlf = xlf.fit(X_train, y_train)
y_pred=xlf.predict(X_test)
metrics.mean_squared_error(y_test, y_pred)
print"MSE:",metrics.mean_squared_error(y_test, y_pred)
#MSE: 4414.09082941
print"Error rate:",MeasurePreprocess.ErrorRate(y_test,y_pred)
#Error rate: 0.638691679657

plt.scatter(y_pred,y_test,color='b')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)
plt.ylabel('Measured')
plt.xlabel('Predicted')
plt.title("XGBoost Regression predicted results")
plt.show()
#Visualization of prediction results 