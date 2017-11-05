#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 30 12:54:19 2017

@author: pinaki
"""
import pandas as pd
import os
import numpy as np

def convertCurrencyToNumbers(currency):
    temp=currency.replace("€","")
    value=0
    if "M" in temp:
        temp=temp.replace("M","")
        value=float(temp)*1000000
    elif "K" in temp:
        temp=temp.replace("K","")
        value=float(temp)*1000
    else:
        value=float(temp)
    return value
    
os.chdir("/media/pinaki/MyStuff/Work/Workspace/Python/ML/Fifa18Analysis/")
dataFileName="fifa-18-demo-player-dataset/CompleteDataset.csv"
data=pd.read_csv(dataFileName)

clubwisePivot=pd.pivot_table(data,index="Club",values="Overall",aggfunc=np.average)
clubwisePivot.sort_values(by=('Overall'), ascending=False,inplace=True)

clubwisePivot.to_csv("Pivot.csv")

reqdData=dict()
reqdData['ID']=data['ID']
reqdData['name']=data['Name']
reqdData['Nationality']=data['Nationality']
reqdData['Overall']=data['Overall']
reqdData['Potential']=data['Potential']
reqdData['Club']=data['Club']
reqdData['Value']=data['Value']
reqdData['Wage']=data['Wage']
reqdData['Special']=data['Special']
dataSet=pd.DataFrame(reqdData)
dataSet["Value"]=dataSet["Value"].apply(lambda x: convertCurrencyToNumbers(x))
dataSet["Wage"]=dataSet["Wage"].apply(lambda x: convertCurrencyToNumbers(x))

wagePivot=pd.pivot_table(dataSet,index=["Club"],values="Wage",aggfunc=np.average)
wagePivot.reset_index( inplace=True)
ratingsPivot=pd.pivot_table(dataSet,index=["Club"],values="Overall",aggfunc=np.average)
ratingsPivot.reset_index( inplace=True)
ratingsVsWage=pd.merge(wagePivot,ratingsPivot,how='outer',on=["Club"])
ratingsVsWage["Wage/Performance"]=ratingsVsWage["Wage"]/ratingsVsWage["Overall"]
ratingsVsWage.sort_values(by=["Wage/Performance"],ascending=False)

from pylab import *
scatter(ratingsVsWage["Overall"],ratingsVsWage["Wage"])
xlabel('Item (s)')
ylabel('Value')
title('Python Line Chart: Plotting numbers')
grid(True)
show()
