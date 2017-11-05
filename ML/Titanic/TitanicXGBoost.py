#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 11:18:58 2017

@author: pinaki
"""

import pandas as pd
from xgboost import XGBClassifier
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
# Fitting XGBoost to the Training set
"""
classifier = XGBClassifier()
classifier.fit(X_train, y_train)

#Fine tuning the model
from sklearn.model_selection import GridSearchCV
parameters = [{'max_depth': [1, 10, 100, 1000],
               'learning_rate':[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
               'n_estimators': [10, 100, 1000],
               'gamma': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
               }
             ]
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10,
                           n_jobs = -1)
grid_search = grid_search.fit(X_train, y_train)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_
"""
#Creating new model based on prev results
classifier = XGBClassifier(gamma=0.6,learning_rate=0.2,max_depth=100,n_estimators=10)
classifier.fit(X_train, Y_train)

y_pred = classifier.predict(X_test)
"""
cm = confusion_matrix(y_test, y_pred)

accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)

print("Accuracy "+str(accuracies.mean())+" StdDev "+str(accuracies.std()))
"""
with open("Output_XG.csv","w") as f:
    f.write("PassengerId,Survived\n")
    for i in range(len(y_pred)):
        f.write(str(testSet['PassengerId'][i])+","+str(y_pred[i])+"\n")