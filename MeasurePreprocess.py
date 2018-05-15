import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder,StandardScaler
from sklearn.model_selection import train_test_split
def GetData():
    dataset=pd.read_csv("train.csv")
    dataset=dataset[['season','holiday','workingday','weather','temp','humidity','windspeed','count']]
    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1].values


    labelencoder = LabelEncoder()
    onehotencoder = OneHotEncoder(categorical_features = range(4))
    X = onehotencoder.fit_transform(X).toarray()
    scaler = StandardScaler()
    scaler.fit(X)
    X=scaler.transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.01, random_state=0)
    return X_train, X_test, y_train, y_test

def GetWindspeedData():
    dataset=pd.read_csv("D:\\bikesharing\\data\\train.csv")
    dataset=dataset[['windspeed','count']]
    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.01, random_state=0)
    return X_train, X_test, y_train, y_test

def WindspeedPolynomialRegression():
    dataset=pd.read_csv("D:\\bikesharing\\data\\train.csv")
    dataset=dataset[['count','windspeed']]
    dataset['square of windspeed']=dataset.windspeed.values**2
    dataset['cube of windspeed']=dataset.windspeed.values**3
    X = dataset.iloc[:, 1:].values
    y = dataset.iloc[:, 0].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.01, random_state=0)
    return X_train, X_test, y_train, y_test

    