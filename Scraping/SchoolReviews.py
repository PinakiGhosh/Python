# -*- coding: utf-8 -*-
from urllib.request import urlopen
from bs4 import BeautifulSoup
import pandas as pd

url="http://www.mouthshut.com/product-reviews/Vibgyor-High-School-Whitefield-Bangalore-reviews-925732507"
pageCount=5
listOfData=[]
for i in range(5):
    if i==0:
        urlToNavigate=url
    else:
        urlToNavigate=url+"-page-"+str(i+1)
    html = urlopen(url).read()
    soup = BeautifulSoup(html, 'html.parser')
    reviewList=soup.find_all('div',{'class':'col-10 review'})
    for review in reviewList:
        d=dict()
        d['reviewText']=review.find('div',{'class':'more reviewdata'}).text.strip()
        d['stars']=len(review.find('p',{'class':'rating'}).find('i',{'class':'icon-rating rated-star'}))
        #d['dateOfReview']=review.find_all('small',{'id':'ctl00_ctl00_ContentPlaceHolderFooter_ContentPlaceHolderBody_rptreviews_ctl00_smdatetime'})[0].text
        d['noOfViews']=review.find_all('span',{'class':'views'})[0].text
        d['dateOfReview']=review.find('small').find('span').text

        listOfData.append(d)

df=pd.DataFrame(listOfData)
df.to_csv("Reviews.csv")

review=reviewList[12]
print(review)

d=dict()
#d['title']=review.find('strong').text
#d['reviewText']=review.find('div',{'class':'more reviewdata'}).text.strip()
#d['stars']=len(review.find('p',{'class':'rating'}).find('i',{'class':'icon-rating rated-star'}))
d['dateOfReview']=review.find('small').find('span').text
d['noOfViews']=review.find('span',{'class':'views'}).text
d['likes']=review.find_all('div',{'class':'tooltip like-count'})[0].text

review.find(attrs={'class':'tooltip-content'})