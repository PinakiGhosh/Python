#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 30 00:00:34 2017

@author: pinaki
"""

from collections import OrderedDict
from bs4 import BeautifulSoup
import pandas as pd
from splinter import Browser
import sys

url="http://www.mouthshut.com/product-reviews/Vibgyor-High-School-Whitefield-Bangalore-reviews-925732507"
pageCount=5
listOfData=[]
browser = Browser()
for i in range(pageCount):
    if i==0:
        urlToNavigate=url
    else:
        urlToNavigate=url+"-page-"+str(i+1)
    print("Opening page",urlToNavigate)
    browser.visit(urlToNavigate)
    count=0
    for j in browser.find_by_text("Read More"):
        j.click()
        print("Count of link",count)
        count+=1
    html = browser.html
    soup = BeautifulSoup(html, 'html.parser')
    reviewList=soup.find_all('div',{'class':'col-10 review'})
    count=0
    for review in reviewList:
        d=OrderedDict()
        try:
            d['title']=review.find('strong').text
            d['stars']=len(review.find('p',{'class':'rating'}).find_all('i',{'class':'icon-rating rated-star'}))
            d['reviewText']=review.find('div',{'class':'more reviewdata'}).text.strip()
            d['noOfViews']=review.find('span',{'class':'views'}).text
            d['dateOfReview']=review.find('small').text
            d['noOfComments']=review.find(attrs={'class':'comment-clk'}).text.replace("Comments","").replace("(","").replace(")","")
            listOfData.append(d)
        except:
            print("Error",count)
            e = sys.exc_info()[0]
            print(e)
        count+=1

df=pd.DataFrame(listOfData)
df.to_csv("Reviews_3.csv",index=False)
browser.quit()