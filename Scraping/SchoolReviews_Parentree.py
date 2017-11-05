#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 29 01:37:35 2017

@author: pinaki
"""
from urllib.request import urlopen
from bs4 import BeautifulSoup
import re
import pandas as pd

baseUrl="https://www.parentree.in/India/Chennai/19710-srimathi-sundaravalli-memorial-school-perungalathur/pg"
noOfPages=32
reviews=[]
for i in range(1,noOfPages+1):
    urlToOpen=baseUrl+str(i)
    print("Opening page",urlToOpen)
    html = urlopen(urlToOpen).read()
    soup = BeautifulSoup(html, 'html.parser')
    #DiscPost106126
    reviewList=soup.find_all('div',{'id':re.compile('DiscPost*')})
    for review in reviewList:
        d=dict()
        textLines=review.find_all('p')
        linesOfText=[]
        for text in textLines:
            linesOfText.append(text.text)
        #d['lines']=linesOfText
        d['text'] = "\n".join(linesOfText)
        reviews.append(d)
    reviewList=soup.find_all('div',{'class':'well'})
    for review in reviewList:
        d=dict()
        textLines=review.find_all('p')
        linesOfText=[]
        for text in textLines:
            linesOfText.append(text.text)
        #d['lines']=linesOfText
        d['text'] = "\n".join(linesOfText)
        reviews.append(d)

df=pd.DataFrame(reviews)
df.to_csv("Reviews_Parentree.csv")
'''
len(reviewList)
review=reviewList[0]
str(review)
textLines=review.find_all('p')
linesOfText=[]
for text in textLines:
    linesOfText.append(text.text)
"\n".join(linesOfText)
        #d['lines']=linesOfText
        d['text'] = "\n".join(linesOfText)
'''