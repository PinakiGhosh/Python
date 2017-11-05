#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  2 20:00:51 2017

@author: pinaki
"""

import pandas as pd
import os
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from collections import OrderedDict

os.chdir("/media/pinaki/MyStuff/Work/Workspace/Python/ML/Titanic")

#Classifier model analyser
def analyseModel(classifier,X_Test,Y_Test,X_Train,Y_Train,runCrossVal=False,cv=10):
    analyserObject=OrderedDict()
    analyserObject['ClassifierType']=str(type(classifier))
    classifier.fit(X_Train, Y_Train)
    y_pred = classifier.predict(X_Test)
    analyserObject['ConfusionMatrix'] = confusion_matrix(Y_Test, y_pred)
    analyserObject['Accuracy'] = accuracy_score(Y_Test, y_pred)
    if runCrossVal:
        accuracies = cross_val_score(estimator = classifier, X = X_Train, y = Y_Train, cv = cv)
        analyserObject['AccuracyList'] = accuracies
        analyserObject['MeanAccuracy'] = accuracies.mean()
        analyserObject['AccuracySD'] = accuracies.std()
    return analyserObject

# Importing the dataset
dataset = pd.read_csv('train.csv')

#Fill in NA for Age

"""
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
"""
del dataset['Name']
del dataset['Ticket']
del dataset['Fare']
del dataset['Cabin']
del dataset['Initial']
dataset.fillna("0",inplace=True)

X = dataset.iloc[:, 2:8].values
y = dataset.iloc[:, 1].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 5] = labelencoder_X_2.fit_transform(X[:, 5])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#List of performance monitors
performanceMonitors=[]

# Fitting XGBoost to the Training set
from xgboost import XGBClassifier
classifier = XGBClassifier()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

performanceMonitors.append(analyseModel(classifier,X_test,y_test,X_train,y_train,runCrossVal=True,cv=10))

# Fitting LogisticRegression to the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
performanceMonitors.append(analyseModel(classifier,X_test,y_test,X_train,y_train,runCrossVal=True,cv=10))

# Fitting KNeighborsClassifier to the Training set
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
performanceMonitors.append(analyseModel(classifier,X_test,y_test,X_train,y_train,runCrossVal=True,cv=10))

# Fitting SVM to the Training set
from sklearn.svm import SVC
classifier = SVC(kernel = 'linear', random_state = 0)
performanceMonitors.append(analyseModel(classifier,X_test,y_test,X_train,y_train,runCrossVal=True,cv=10))

# Fitting Kernel SVN to the Training set
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0)
performanceMonitors.append(analyseModel(classifier,X_test,y_test,X_train,y_train,runCrossVal=True,cv=10))

# Fitting Naiyesve Ba to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
performanceMonitors.append(analyseModel(classifier,X_test,y_test,X_train,y_train,runCrossVal=True,cv=10))

# Fitting DecisionTree to the Training set
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
performanceMonitors.append(analyseModel(classifier,X_test,y_test,X_train,y_train,runCrossVal=True,cv=10))

# Fitting RandomForest to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
performanceMonitors.append(analyseModel(classifier,X_test,y_test,X_train,y_train,runCrossVal=True,cv=10))

performanceMonitors=sorted(performanceMonitors, key=lambda k: k['MeanAccuracy'],reverse=True)

for i in performanceMonitors:
    print(i['ClassifierType'],i['Accuracy'])



