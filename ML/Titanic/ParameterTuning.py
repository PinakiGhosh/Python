#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 11 19:36:11 2017

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


import time
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from keras.models import Sequential
from keras.layers import Dense

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""

def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 6))
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier

classifier = KerasClassifier(build_fn = build_classifier, batch_size = 10, epochs = 100)
start_time = time.time()
accuracies = cross_val_score(estimator = classifier, X = X_train, y = Y_train, cv = 10, n_jobs = -1)
print("--- %s seconds ---" % (time.time() - start_time))
mean = accuracies.mean()
variance = accuracies.std()
print(mean)
print(variance)

def build_classifier(optimizer):
    classifier = Sequential()
    #relu
    classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'tanh', input_dim = 6))
    #classifier.add(Dropout(p = 0.1))
    classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'tanh'))
    classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier

classifier = KerasClassifier(build_fn = build_classifier)
parameters = {'batch_size': [10,25,32],
              'epochs': [100, 500],
              'optimizer': ['adam', 'rmsprop'],
              }
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10)
start_time = time.time()
grid_search = grid_search.fit(X_train, Y_train)
print("--- %s seconds ---" % (time.time() - start_time))
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_
print(best_parameters)
print(best_accuracy)