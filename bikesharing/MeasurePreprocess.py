import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder,StandardScaler
from sklearn.model_selection import train_test_split
def GetData():
    dataset=pd.read_csv("D:\\bikesharing\\data\\train.csv")
    dataset["hour"]=pd.DataFrame(dataset.datetime.apply(lambda x:x.split()[1].split(":")[0]))
    dataset['hour'] = dataset['hour'].astype('int64')
    dataset["month"]=pd.DataFrame(dataset.datetime.apply(lambda x:x.split()[0].split("-")[1]))
    dataset['month'] = dataset['month'].astype('int64')
    dataset=dataset[['season','holiday','workingday','weather','hour','month','temp','humidity','windspeed','count']]

    upper_bound=dataset['count'].mean()+3*dataset['count'].std()
    lower_bound=dataset['count'].mean()-3*dataset['count'].std()
    dataset=dataset[dataset["count"]<upper_bound]
    dataset=dataset[dataset["count"]>lower_bound]
    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1].values

    
    labelencoder = LabelEncoder()
    onehotencoder = OneHotEncoder(categorical_features = range(6))
    X = onehotencoder.fit_transform(X).toarray()
    #X=np.delete(X,[3,10,11,23,26,37,38,39,41],axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    return X_train, X_test, y_train, y_test

def GetHourData():
    dataset=pd.read_csv("D:\\bikesharing\\data\\train.csv")
    dataset["hour"]=pd.DataFrame(dataset.datetime.apply(lambda x:x.split()[1].split(":")[0]))
    dataset['hour'] = dataset['hour'].astype('int64')
    dataset=dataset[['hour','count']]
    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    return X_train, X_test, y_train, y_test

def HourPolynomialRegression():
    dataset=pd.read_csv("D:\\bikesharing\\data\\train.csv")
    dataset["hour"]=pd.DataFrame(dataset.datetime.apply(lambda x:x.split()[1].split(":")[0]))
    dataset['hour'] = dataset['hour'].astype('int64')
    dataset['square of hour']=dataset.hour.values**2
    dataset['cube of hour']=dataset.hour.values**3
    dataset=dataset[['hour','square of hour','cube of hour','count']]
    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    return X_train, X_test, y_train, y_test

   


def ErrorRate(test,pred):
    avg=[10]*len(test)
    for i in range(len(test)):
        try:
            avg[i]=abs(float(test[i]) - float(pred[i]))/float(test[i])
        except ZeroDivisionError,e:
            print e.message
    avg=np.array(avg)
    return avg.mean()