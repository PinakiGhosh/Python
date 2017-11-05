#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  2 21:11:54 2017

@author: pinaki
"""
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score

#Classifier model analyser
def analyseModel(classifier,X_Test,Y_Test,X_Train,Y_Train,runCrossVal=False,cv=10):
    analyserObject=dict()
    analyserObject['ClassifierType']=type(classifier)
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
