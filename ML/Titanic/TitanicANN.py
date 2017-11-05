#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 12:27:47 2017

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


from keras.models import Sequential
from keras.layers import Dense

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(output_dim = 4, init = 'uniform', activation = 'tanh', input_dim = 6))

# Adding the second hidden layer
classifier.add(Dense(output_dim = 4, init = 'uniform', activation = 'tanh'))

# Adding the output layer
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(X_train, Y_train, batch_size = 10, nb_epoch = 100)

# Predicting the Test set results
y_pred = classifier.predict(X_test)
predicted_list=[]
for i in range(len(y_pred)):
    #print(y_pred[i][0])
    if y_pred[i][0]>=0.5:
        y_pred[i]="1"
        predicted_list.append(1)
    else:
        y_pred[i]="0"
        predicted_list.append(0)

with open("Output_Ann.csv","w") as f:
    f.write("PassengerId,Survived\n")
    for i in range(len(y_pred)):
        f.write(str(testSet['PassengerId'][i])+","+str(predicted_list[i])+"\n")