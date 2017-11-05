#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 11:44:31 2017

@author: pinaki
"""

import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score

# Importing the dataset
dataset = pd.read_csv('train.csv')
