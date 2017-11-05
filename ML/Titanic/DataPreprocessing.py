#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 17:44:10 2017

@author: pinaki
"""
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

def importAndCleanData(fileName):
    dataset=pd.read_csv(fileName)
    dataset['Initial']=dataset.Name.str.extract('([A-Za-z]+)\.')
    dataset['Initial'].replace(['Master','Miss','Mlle','Mme','Ms','Mr','Countess','Mrs','Jonkheer','Don','Dr','Rev','Lady','Major','Sir','Col','Capt'],
['Age1','Age2','Age2','Age2','Age2','Age3','Age4','Age4','Age5','Age5','Age5','Age5','Age7','Age5','Age5','Age6','Age6'],inplace=True)
    dataset.loc[(dataset.Age.isnull())&(dataset.Initial=='Age1'),'Age']=5
    dataset.loc[(dataset.Age.isnull())&(dataset.Initial=='Age2'),'Age']=22
    dataset.loc[(dataset.Age.isnull())&(dataset.Initial=='Age3'),'Age']=32
    dataset.loc[(dataset.Age.isnull())&(dataset.Initial=='Age4'),'Age']=35
    dataset.loc[(dataset.Age.isnull())&(dataset.Initial=='Age5'),'Age']=43
    dataset.loc[(dataset.Age.isnull())&(dataset.Initial=='Age6'),'Age']=62
    dataset.loc[(dataset.Age.isnull())&(dataset.Initial=='Age7'),'Age']=48
    del dataset['Name']
    del dataset['Ticket']
    del dataset['Fare']
    del dataset['Cabin']
    del dataset['Initial']
    dataset.fillna("0",inplace=True)
    return dataset

trainingSet=importAndCleanData("train.csv")
testSet=importAndCleanData("test.csv")

X_train = trainingSet.iloc[:, 2:8].values
Y_train = trainingSet.iloc[:, 1].values

X_test = testSet.iloc[:, 1:7].values

# Encoding categorical data
labelencoder_X_1 = LabelEncoder()
X_train[:, 1] = labelencoder_X_1.fit_transform(X_train[:, 1])
X_test[:, 1] = labelencoder_X_1.transform(X_test[:, 1])
labelencoder_X_2 = LabelEncoder()
X_train[:, 5] = labelencoder_X_2.fit_transform(X_train[:, 5])
X_test[:, 5] = labelencoder_X_2.transform(X_test[:, 5])
onehotencoder = OneHotEncoder(categorical_features = [1])
X_train = onehotencoder.fit_transform(X_train).toarray()
X_test = onehotencoder.transform(X_test).toarray()
X_train = X_train[:, 1:]
X_test = X_test[:, 1:]

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

